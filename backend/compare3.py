"""Same-prompt comparison of up to three runnable models.

One identical prompt and identical decoding policy are sent to each selected model.
Models are loaded one at a time, verified against their pinned hashes, run, and then
released, so peak memory stays near the largest single model rather than their sum.
Only the latest comparison is kept, in memory. Prompts and traces are never written
to disk or logged; stage messages never contain the prompt.

For the comparison *table* each model contributes a small, honest set of values:
  * its answer and why it stopped                              (measured)
  * first-token confidence, average decoding uncertainty       (measured / derived)
  * residual-stream growth across layers at the last prompt token (measured)
  * first-layer attention focus (entropy)                      (measured)
  * prefill/decode time                                        (measured, host-dependent)
  * parameters / shape / licence                               (metadata)
Cross-model token alignment is shown only when the tokenizers produce identical
prompt token ids; otherwise each trace is read on its own and that is stated.
"""

from __future__ import annotations

import gc
import hashlib
import os
import resource
import subprocess
import sys
import threading
import time
import uuid

import registry3 as reg
from catalog import head_dim, is_osi
from qwen2_explorer import LIMITS, QwenExplorer


def engine_class(entry):
    """Pick the instrumented forward pass for a runnable entry's architecture."""
    if entry["model_type"] == "qwen2":
        return QwenExplorer
    if entry["model_type"] == "qwen3":
        from qwen3_explorer import Qwen3Explorer
        return Qwen3Explorer
    raise ValueError(f"no verified engine for model_type {entry['model_type']!r}")


def rss_bytes():
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) * 1024
    except Exception:
        return None


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)  # macOS bytes, Linux KiB


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "running"
        self.stages = []
        self.error = None
        self.result = None
        self.t0 = time.monotonic()

    def stage(self, msg, **extra):
        self.stages.append({"t_s": round(time.monotonic() - self.t0, 2), "msg": msg,
                            "rss_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes(), **extra})

    def status(self):
        out = {"job_id": self.id, "kind": self.kind, "state": self.state,
               "stages": self.stages, "elapsed_s": round(time.monotonic() - self.t0, 2)}
        if self.error:
            out["error"] = self.error
        if self.state == "done":
            out["result"] = self.result
        return out


class Compare3Store:
    """At most one running job and one finished comparison (the latest), in memory only."""

    def __init__(self):
        self.job = None
        self.cmp = None
        self.lock = threading.Lock()
        self.jlock = threading.Lock()

    # ------------------------------------------------------------------ jobs
    def start(self, kind, fn):
        with self.jlock:
            if self.job is not None and self.job.state == "running":
                raise RuntimeError("a comparison job is already running")
            job = Job(kind)
            self.job = job

        def work():
            try:
                job.result = fn(job)
                job.state = "done"
                job.stage("done")
            except reg.ProvenanceError as e:
                job.error = "provenance check failed: " + str(e)
                job.state = "error"
            except MemoryError:
                job.error = "out of memory while loading a model"
                job.state = "error"
            except Exception as e:  # message never includes the prompt
                job.error = str(e) if isinstance(e, (ValueError, KeyError)) else f"{type(e).__name__}: {e}"
                job.state = "error"
            finally:
                gc.collect()

        threading.Thread(target=work, name=f"compare3-{job.id}", daemon=True).start()
        return job

    def status(self, job_id):
        j = self.job
        if j is None or j.id != job_id:
            raise KeyError("job not found (only the latest job is kept)")
        return j.status()

    # ------------------------------------------------------------------ request
    def run_job(self, req):
        prompt = req.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        slots = req.get("slots", ["a", "b", "c"])
        if not isinstance(slots, list) or not 1 <= len(slots) <= 3:
            raise ValueError("slots must be a list of 1..3 runnable slot ids")
        for s in slots:
            reg.spec_for_slot(s)  # raises KeyError on an unknown/non-runnable slot
        settings = {"system": req.get("system"),
                    "max_new_tokens": int(req.get("max_new_tokens", 40)),
                    "temperature": float(req.get("temperature", 0.0)),
                    "seed": int(req.get("seed", 0)),
                    "use_chat_template": bool(req.get("chat_template", True))}
        return self.start("compare3", lambda job: self._compare(job, prompt, slots, settings))

    # ------------------------------------------------------------------ one model
    def _run_one(self, job, entry, prompt, settings):
        job.stage(f"verifying {entry['label']} against official hashes")
        prov = reg.verify(entry)
        job.stage(f"verified {entry['label']} (sha256 {prov['weights_sha256'][:12]}...,"
                  f" {'cached' if prov['hash_cached_in_process'] else str(prov['hash_seconds']) + ' s'})")
        job.stage(f"loading {entry['label']} weights (fp32, CPU)")
        eng = engine_class(entry)(reg.snapshot_dir(entry), repo_id=entry["repo"],
                                  revision=entry["pin"]["revision"])
        job.stage(f"loaded {entry['label']} in {eng.load_seconds:.1f} s")
        try:
            job.stage(f"running {entry['label']}")
            out = eng.run(prompt, system=settings["system"],
                          max_new_tokens=settings["max_new_tokens"],
                          temperature=settings["temperature"], seed=settings["seed"],
                          use_chat_template=settings["use_chat_template"])
            rendered = (eng.render_prompt(prompt, settings["system"])
                        if settings["use_chat_template"] else prompt)
            meta = {"eos_ids": sorted(eng.eos_ids),
                    "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "chat_template_sha256": hashlib.sha256(eng.chat_template_source.encode()).hexdigest(),
                    "provenance_check": prov}
        finally:
            eng.release_weights()
            del eng
            gc.collect()
            job.stage(f"released {entry['label']} weights (summary kept)")
        return out, meta

    # ------------------------------------------------------------------ summary
    def _summary(self, entry, out, meta):
        pl = out["prompt_len"]
        per_pos = out["per_pos"]
        last = per_pos.get(str(pl - 1), {})
        h_norm = last.get("h_norm") or []
        L = out["dims"]["layers"]
        first_n = h_norm[0] if len(h_norm) > 0 else None
        last_n = h_norm[L] if len(h_norm) > L else None
        # layer-0 attention focus: mean head entropy over measured positions
        layer0_ent = _mean([e for row in per_pos.values()
                            if (row.get("head_entropy") and row["head_entropy"][0])
                            for e in row["head_entropy"][0]])
        steps = out["steps"]
        cfg = entry["config"]
        # per-layer series at the last prompt token (for the layer-dynamics charts)
        he = last.get("head_entropy") or []
        per_layer_attn_ent = []
        for l in range(L):
            row = he[l] if l < len(he) and he[l] else None
            per_layer_attn_ent.append(round(sum(row) / len(row), 4) if row else None)
        layer_series = {
            "h_norm": h_norm,                                 # length L+1
            "attn_norm": last.get("attn_norm") or [],         # length L
            "mlp_norm": last.get("mlp_norm") or [],           # length L
            "cos_prev": last.get("cos_prev") or [],           # length L
            "attn_entropy": per_layer_attn_ent,               # length L
        }
        # per-generated-token series (for the decoding-confidence chart + heatstrip)
        decode_series = [{"text": s["chosen_text"], "p": s["chosen_p"], "entropy": s["entropy"]}
                         for s in steps]
        # layer x token grid for the heatmap: every measured position, one value per layer
        positions = sorted(int(k) for k in per_pos)
        toks = out["tokens"]

        def head_mean(row, l):
            he = row.get("head_entropy") or []
            v = he[l] if l < len(he) else None
            return round(sum(v) / len(v), 4) if v else None

        token_layer = {
            "tokens": [{"pos": p, "text": toks[p]["text"], "segment": toks[p]["segment"]} for p in positions],
            "layers": L,
            # each metric: [token][layer]
            "h_norm": [(per_pos[str(p)].get("h_norm") or [None] * (L + 1))[1:] for p in positions],
            "attn_norm": [per_pos[str(p)].get("attn_norm") or [None] * L for p in positions],
            "mlp_norm": [per_pos[str(p)].get("mlp_norm") or [None] * L for p in positions],
            "attn_entropy": [[head_mean(per_pos[str(p)], l) for l in range(L)] for p in positions],
        }
        return {
            "id": entry["id"], "label": entry["label"], "repo": entry["repo"],
            "revision": entry["pin"]["revision"], "tuning": entry["tuning"],
            "runnable": True,
            "license": {"id": entry["license"], "osi_open_source": is_osi(entry["license"])},
            "provenance": {"passed": True, "weights_sha256": meta["provenance_check"]["weights_sha256"],
                           "weights_bytes": meta["provenance_check"]["weights_bytes"]},
            "metadata": {"parameters": entry["params"], "layers": cfg["num_hidden_layers"],
                         "hidden_size": cfg["hidden_size"], "attention_heads": cfg["num_attention_heads"],
                         "kv_heads": cfg["num_key_value_heads"],
                         "head_dim": head_dim(cfg)},
            "answer": out["generated_text"], "stop_reason": out["stop_reason"],
            "prompt_tokens": pl, "generated_tokens": len(steps),
            "settings": out["settings"],
            "rendered_prompt_sha256": meta["rendered_prompt_sha256"],
            "chat_template_sha256": meta["chat_template_sha256"],
            "eos_ids": meta["eos_ids"],
            "prompt_token_ids": [t["id"] for t in out["tokens"][:pl]],
            "generated_token_ids": [t["id"] for t in out["tokens"][pl:]],
            "measured": {
                "first_token_confidence": (steps[0]["chosen_p"] if steps else None),
                "mean_entropy": (round(_mean([s["entropy"] for s in steps]), 4)
                                 if steps else None),
                "residual_first_layer_norm": first_n,
                "residual_last_layer_norm": last_n,
                "residual_growth_ratio": (round(last_n / first_n, 3)
                                          if (first_n and last_n) else None),
                "attention_entropy_layer0": (round(layer0_ent, 4) if layer0_ent is not None else None),
                "prefill_s": out["timing"]["prefill_s"], "decode_s": out["timing"]["decode_s"],
                "tokens_per_s": out["timing"]["tokens_per_s"],
            },
            # compact residual-norm-by-layer profile (last prompt token) for a sparkline
            "layer_norm_profile": h_norm,
            "layer_series": layer_series,
            "decode_series": decode_series,
            "token_layer": token_layer,
        }

    # ------------------------------------------------------------------ compare
    def _compare(self, job, prompt, slots, settings):
        entries = [reg.spec_for_slot(s) for s in slots]
        # de-dup identical selections: run each unique checkpoint once
        runs, metas = {}, {}
        for entry in entries:
            if entry["id"] not in runs:
                runs[entry["id"]], metas[entry["id"]] = self._run_one(job, entry, prompt, settings)
        summaries = {}
        for s, entry in zip(slots, entries):
            summaries[s] = self._summary(entry, runs[entry["id"]], metas[entry["id"]])

        # parity + alignment across the selected slots
        rp = {s: summaries[s]["rendered_prompt_sha256"] for s in slots}
        ids = {s: summaries[s]["prompt_token_ids"] for s in slots}
        tokenizers_match = all(ids[s] == ids[slots[0]] for s in slots)
        parity = {
            "settings_equal": all(summaries[s]["settings"] == summaries[slots[0]]["settings"] for s in slots),
            "rendered_prompt_equal": all(rp[s] == rp[slots[0]] for s in slots),
            "tokenizers_match": tokenizers_match,
            "settings": summaries[slots[0]]["settings"],
            "instrumentation": "same engine (qwen2_explorer.QwenExplorer), fp32 CPU, identical capture limits",
            "limits": LIMITS,
        }
        align = self._alignment(slots, summaries) if tokenizers_match else {
            "comparable": False,
            "note": "Tokenizers produced different prompt token ids; per-token alignment is not shown. "
                    "Each model's trace is read on its own."}

        cmp_id = uuid.uuid4().hex[:12]
        self.cmp = {"id": cmp_id, "slots": slots, "summaries": summaries}
        return {
            "cmp_id": cmp_id, "slots": slots,
            "provenance": "measured: each model's trace was computed locally from its own pinned, "
                          "hash-verified checkpoint, one model at a time. No model was substituted.",
            "models": summaries, "parity": parity, "alignment": align,
            "memory": {"peak_rss_bytes": peak_rss_bytes(), "rss_bytes_now": rss_bytes()},
        }

    def _alignment(self, slots, summaries):
        gens = {s: summaries[s]["generated_token_ids"] for s in slots}
        shared = 0
        n = min(len(gens[s]) for s in slots)
        while shared < n and all(gens[s][shared] == gens[slots[0]][shared] for s in slots):
            shared += 1
        diverge = shared if any(shared < len(gens[s]) for s in slots) else None
        return {"comparable": True, "shared_generated": shared,
                "first_divergence_step": diverge,
                "generated_len": {s: len(gens[s]) for s in slots},
                "note": "Models received identical prompt token ids. They agreed on the first "
                        f"{shared} generated token(s)" + (f", then diverged at step {diverge}."
                                                          if diverge is not None else ".")}
