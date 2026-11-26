#!/usr/bin/env python3
"""Independent check of an instrumented engine against Hugging Face ``transformers``.

Usage (needs a Python with `transformers`; e.g. a venv with torch + transformers):
    <venv>/bin/python backend/verify_engine.py --model qwen3-1.7b
    <venv>/bin/python backend/verify_engine.py --model deepseek-r1-distill-qwen-1.5b

Steps, one model copy in memory at a time:
  1. verify the pinned snapshot's hashes (registry3.verify);
  2. our engine: render a fixed neutral prompt, record per-layer residual norms at every
     prompt position, the fp32 last-position logits, and a 20-token greedy decode;
  3. release it, then load the canonical transformers model (fp32, local files only) and
     feed it the *identical token ids*;
  4. compare: hidden-state norms per layer (relative), last-position logits (absolute,
     and top-1/top-10 agreement), greedy token ids (exact).

Prints a JSON summary and exits non-zero on any failure. Nothing is downloaded,
written to the repo, or sent off the machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import catalog  # noqa: E402
import registry3  # noqa: E402
from compare3 import engine_class  # noqa: E402

PROMPT = "Name three primary colours and say one fact about each."
TOL = {"h_norm_rel_max": 2e-3, "logits_abs_max": 5e-2, "greedy_tokens_exact": 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    entry = catalog.by_id(args.model)
    prov = registry3.verify(entry)
    snap = registry3.snapshot_dir(entry)

    # ---- our engine
    eng = engine_class(entry)(snap, repo_id=entry["repo"], revision=entry["pin"]["revision"])
    ids = eng.tokenizer.encode(eng.render_prompt(PROMPT, None), add_special_tokens=False).ids
    rec = {"per_pos": {}, "logit_lens": {}, "hidden": [dict() for _ in range(eng.L + 1)],
           "attn": [dict() for _ in range(eng.L)]}
    ours_logits = eng._forward(ids, 0, [None] * eng.L, rec, False).float().clone()
    L = eng.L
    ours_hnorm = [[rec["per_pos"][P]["h_norm"][li] for li in range(L)] for P in range(len(ids))]
    run = eng.run(PROMPT, max_new_tokens=TOL["greedy_tokens_exact"], temperature=0.0)
    ours_gen = [s["chosen_id"] for s in run["steps"]]
    eos = sorted(eng.eos_ids)
    eng.release_weights(); del eng, rec, run; gc.collect()

    # ---- reference: transformers
    from transformers import AutoModelForCausalLM
    try:
        ref = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32, local_files_only=True)
    except TypeError:
        ref = AutoModelForCausalLM.from_pretrained(str(snap), torch_dtype=torch.float32, local_files_only=True)
    ref.eval()
    x = torch.tensor([ids])
    with torch.no_grad():
        out = ref(x, output_hidden_states=True)
        ref_logits = out.logits[0, -1].float()
        # hidden_states[i] = input to layer i for i < L (the last entry is post-final-norm)
        ref_hnorm = [[float(out.hidden_states[li][0, P].norm()) for li in range(L)] for P in range(len(ids))]
        # plain greedy, matching the engine: override the checkpoint's generation_config
        # (e.g. Qwen ships repetition_penalty 1.05, top_k/top_p) so nothing else is applied
        gen = ref.generate(x, max_new_tokens=TOL["greedy_tokens_exact"], do_sample=False,
                           repetition_penalty=1.0, top_k=None, top_p=None, temperature=None,
                           eos_token_id=eos, pad_token_id=eos[0])
    ref_gen = gen[0, len(ids):].tolist()
    del ref, out; gc.collect()

    rel = max(abs(a - b) / max(abs(b), 1e-6) for ra, rb in zip(ours_hnorm, ref_hnorm) for a, b in zip(ra, rb))
    labs = float((ours_logits - ref_logits).abs().max())
    top1 = int(ours_logits.argmax()) == int(ref_logits.argmax())
    o10, r10 = set(ours_logits.topk(10).indices.tolist()), set(ref_logits.topk(10).indices.tolist())
    n = min(len(ours_gen), len(ref_gen))
    greedy_equal = ours_gen[:n] == ref_gen[:n] and len(ours_gen) == len(ref_gen)
    checks = {
        "hidden_norm_rel_max": {"value": rel, "tolerance": TOL["h_norm_rel_max"], "pass": rel <= TOL["h_norm_rel_max"]},
        "last_logits_abs_max": {"value": labs, "tolerance": TOL["logits_abs_max"], "pass": labs <= TOL["logits_abs_max"]},
        "top1_agrees": {"value": top1, "pass": top1},
        "top10_overlap": {"value": len(o10 & r10) / 10, "pass": len(o10 & r10) >= 9},
        "greedy_tokens_identical": {"value": f"{n} compared; ours {len(ours_gen)} / ref {len(ref_gen)} tokens"
                                    + ("" if greedy_equal else f"; first divergence at step "
                                       f"{next((i for i in range(n) if ours_gen[i] != ref_gen[i]), n)}"),
                                    "pass": greedy_equal},
    }
    ok = all(c["pass"] for c in checks.values())
    import transformers
    print(json.dumps({"model": entry["id"], "engine": engine_class(entry).__name__,
                      "reference": f"transformers {transformers.__version__} (fp32, local files)",
                      "revision": prov["revision"], "weights_verified": True,
                      "prompt_tokens": len(ids), "layers_compared": L,
                      "checks": checks, "all_pass": ok}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
