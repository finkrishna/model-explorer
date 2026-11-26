"""Instrumented, inference-only Qwen2 forward pass for the local internals explorer.

This is a direct re-implementation of the published Qwen2 decoder architecture
(RMSNorm, rotary position embeddings, grouped-query attention with QKV bias, SwiGLU
MLP, tied input/output embeddings) that reads the pinned checkpoint read-only. It is
written out explicitly so every intermediate value can be measured without hooks or
the `transformers` package (not installed on this host).

Weights are never modified. Captures are bounded (see LIMITS) and kept in memory
only for the latest run; nothing about the prompt is written to disk or logged.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import jinja2
import jinja2.sandbox
import torch
from tokenizers import Tokenizer

from safetensors_io import SafetensorsFile

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
                    / "snapshots" / PINNED_REVISION)

LIMITS = {
    "max_prompt_tokens": 256,      # after chat templating
    "max_new_tokens": 96,
    "mlp_top_k": 8,                # top |activation| intermediate neurons stored per token/layer
    "logit_lens_top_k": 5,
    "next_token_top_k": 10,
}


def _r(x: float, nd: int = 4) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return None  # JSON-safe
    return float(f"{x:.{nd}g}")


def _rl(t: torch.Tensor, nd: int = 4):
    return [_r(v, nd) for v in t.tolist()]


class QwenExplorer:
    def __init__(self, snapshot: Path = DEFAULT_SNAPSHOT, device: str = "cpu",
                 repo_id: str = REPO_ID, revision: str = PINNED_REVISION):
        self.snapshot = Path(snapshot)
        self.repo_id, self.revision = repo_id, revision
        self.config = json.loads((self.snapshot / "config.json").read_text())
        self.generation_config = json.loads((self.snapshot / "generation_config.json").read_text())
        tok_cfg = json.loads((self.snapshot / "tokenizer_config.json").read_text())
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.chat_template = env.from_string(tok_cfg["chat_template"])
        self.chat_template_source = tok_cfg["chat_template"]
        self.tokenizer = Tokenizer.from_file(str(self.snapshot / "tokenizer.json"))
        self.device = torch.device(device)
        self.dtype = torch.float32  # stored bf16 weights are upcast to fp32 for CPU inference

        c = self.config
        self.L = c["num_hidden_layers"]
        self.D = c["hidden_size"]
        self.H = c["num_attention_heads"]
        self.KVH = c["num_key_value_heads"]
        self.hd = self.D // self.H
        self.I = c["intermediate_size"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        eos = self.generation_config.get("eos_token_id", c["eos_token_id"])
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

        self.st = SafetensorsFile(self.snapshot / "model.safetensors")
        load_start = time.monotonic()
        self.w = {k: self.st.tensor(k).to(self.device, self.dtype) for k in self.st.keys()}
        self.load_seconds = time.monotonic() - load_start
        if c.get("tie_word_embeddings") and "lm_head.weight" not in self.w:
            self.lm_head = self.w["model.embed_tokens.weight"]  # tied, shared storage
        else:
            self.lm_head = self.w["lm_head.weight"]
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        self.inv_freq = inv.to(self.device, torch.float32)
        self.last_run = None  # in-memory only

    # ------------------------------------------------------------------ helpers
    def render_prompt(self, user: str, system: str | None) -> str:
        messages = []
        if system is not None and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.chat_template.render(messages=messages, tools=None, add_generation_prompt=True)

    def _rms(self, x, w):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * w

    def _rope(self, x, positions):
        # x: [heads, T, hd]; Qwen2 uses rotate_half layout
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)  # [T, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def _decode(self, ids):
        return [self.tokenizer.decode([i], skip_special_tokens=False) for i in ids]

    def _topk_tokens(self, logits, k):
        probs = torch.softmax(logits.float(), -1)
        p, idx = probs.topk(k)
        texts = self._decode(idx.tolist())
        return [{"id": int(i), "text": t, "p": _r(float(pp))} for i, t, pp in zip(idx.tolist(), texts, p.tolist())]

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def _forward(self, ids, start_pos, cache, rec, lens_rows):
        """Run tokens `ids` at absolute positions start_pos.. using KV `cache`.

        Records measurements into `rec` (the run store). Returns final-norm logits at
        the last position. `lens_rows` -> collect logit-lens for the last position.
        """
        T = len(ids)
        pos = torch.arange(start_pos, start_pos + T, device=self.device)
        x = self.w["model.embed_tokens.weight"][torch.tensor(ids, device=self.device)]  # [T, D]
        hs_last = [x[-1]]
        self._record_hidden(rec, 0, pos, x, None)
        group = self.H // self.KVH
        scale = 1.0 / math.sqrt(self.hd)
        for l in range(self.L):
            p = f"model.layers.{l}."
            h_in = x
            n1 = self._rms(x, self.w[p + "input_layernorm.weight"])
            q = n1 @ self.w[p + "self_attn.q_proj.weight"].T + self.w[p + "self_attn.q_proj.bias"]
            k = n1 @ self.w[p + "self_attn.k_proj.weight"].T + self.w[p + "self_attn.k_proj.bias"]
            v = n1 @ self.w[p + "self_attn.v_proj.weight"].T + self.w[p + "self_attn.v_proj.bias"]
            q = self._rope(q.view(T, self.H, self.hd).transpose(0, 1), pos)
            k = self._rope(k.view(T, self.KVH, self.hd).transpose(0, 1), pos)
            v = v.view(T, self.KVH, self.hd).transpose(0, 1)
            if cache[l] is None:
                cache[l] = (k, v)
            else:
                cache[l] = (torch.cat([cache[l][0], k], 1), torch.cat([cache[l][1], v], 1))
            K, V = cache[l]
            S = K.shape[1]
            Kx = K.repeat_interleave(group, 0)  # [H, S, hd]
            Vx = V.repeat_interleave(group, 0)
            scores = (q @ Kx.transpose(1, 2)) * scale  # [H, T, S]
            qpos = pos.view(T, 1)
            kpos = torch.arange(S, device=self.device).view(1, S)
            scores = scores.masked_fill((kpos > qpos).unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, -1)
            head_out = probs @ Vx  # [H, T, hd]
            Wo = self.w[p + "self_attn.o_proj.weight"]  # [D, H*hd]
            # per-head contribution to the residual stream (o_proj split by input columns)
            Wo_h = Wo.view(self.D, self.H, self.hd).permute(1, 2, 0)  # [H, hd, D]
            contrib = torch.bmm(head_out, Wo_h)  # [H, T, D]
            attn_out = contrib.sum(0)
            x = x + attn_out
            n2 = self._rms(x, self.w[p + "post_attention_layernorm.weight"])
            gate = n2 @ self.w[p + "mlp.gate_proj.weight"].T
            up = n2 @ self.w[p + "mlp.up_proj.weight"].T
            act = torch.nn.functional.silu(gate) * up  # [T, I]
            mlp_out = act @ self.w[p + "mlp.down_proj.weight"].T
            x = x + mlp_out
            self._record_layer(rec, l, pos, probs, contrib, attn_out, act, mlp_out)
            self._record_hidden(rec, l + 1, pos, x, h_in)
            hs_last.append(x[-1])
        final = self._rms(x, self.w["model.norm.weight"])
        logits = final[-1] @ self.lm_head.T
        if lens_rows:
            # intermediate layers: final RMSNorm + tied unembedding applied early (derived);
            # last row equals the model's real output logits.
            H = torch.stack(hs_last[:-1])
            lg_all = self._rms(H, self.w["model.norm.weight"]) @ self.lm_head.T
            lens = [self._topk_tokens(lg_all[i], LIMITS["logit_lens_top_k"]) for i in range(self.L)]
            lens.append(self._topk_tokens(logits, LIMITS["logit_lens_top_k"]))
            rec["logit_lens"][int(pos[-1])] = lens
        return logits

    def _record_hidden(self, rec, li, pos, x, prev):
        norms = x.norm(dim=-1)
        absx = x.abs()
        mx, arg = absx.max(-1)
        cos = None
        if prev is not None:
            cos = torch.nn.functional.cosine_similarity(x, prev, dim=-1)
        for t in range(x.shape[0]):
            P = int(pos[t])
            row = rec["per_pos"].setdefault(P, self._empty_pos())
            row["h_norm"][li] = _r(float(norms[t]))
            row["h_mean"][li] = _r(float(x[t].mean()))
            row["h_std"][li] = _r(float(x[t].std()))
            row["h_maxabs"][li] = _r(float(mx[t]))
            row["h_maxabs_dim"][li] = int(arg[t])
            if cos is not None:
                row["cos_prev"][li - 1] = _r(float(cos[t]))
            rec["hidden"][li][P] = x[t].to(torch.float16).clone()

    def _record_layer(self, rec, l, pos, probs, contrib, attn_out, act, mlp_out):
        T = probs.shape[1]
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)  # [H, T]
        head_norm = contrib.norm(dim=-1)  # [H, T]
        a_norm = attn_out.norm(dim=-1)
        m_norm = mlp_out.norm(dim=-1)
        act_norm = act.norm(dim=-1)
        k = LIMITS["mlp_top_k"]
        tv, ti = act.abs().topk(k, dim=-1)
        signed = torch.gather(act, -1, ti)
        for t in range(T):
            P = int(pos[t])
            row = rec["per_pos"][P]
            row["attn_norm"][l] = _r(float(a_norm[t]))
            row["mlp_norm"][l] = _r(float(m_norm[t]))
            row["act_norm"][l] = _r(float(act_norm[t]))
            row["head_entropy"][l] = _rl(ent[:, t])
            row["head_norm"][l] = _rl(head_norm[:, t])
            row["mlp_top"][l] = [[int(i), _r(float(v))] for i, v in zip(ti[t].tolist(), signed[t].tolist())]
            # full attention rows kept in memory (float16) for on-demand head views
            rec["attn"][l][P] = probs[:, t, : P + 1].to(torch.float16).clone()

    def _empty_pos(self):
        L = self.L
        return {
            "h_norm": [None] * (L + 1), "h_mean": [None] * (L + 1), "h_std": [None] * (L + 1),
            "h_maxabs": [None] * (L + 1), "h_maxabs_dim": [None] * (L + 1),
            "cos_prev": [None] * L, "attn_norm": [None] * L, "mlp_norm": [None] * L,
            "act_norm": [None] * L, "head_entropy": [None] * L, "head_norm": [None] * L,
            "mlp_top": [None] * L,
        }

    # ------------------------------------------------------------------ public run
    @torch.no_grad()
    def run(self, prompt: str, system: str | None = None, max_new_tokens: int = 48,
            temperature: float = 0.0, seed: int = 0, use_chat_template: bool = True):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        max_new_tokens = max(1, min(int(max_new_tokens), LIMITS["max_new_tokens"]))
        temperature = max(0.0, min(float(temperature), 2.0))
        text = self.render_prompt(prompt, system) if use_chat_template else prompt
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) > LIMITS["max_prompt_tokens"]:
            raise ValueError(f"prompt is {len(ids)} tokens after templating; limit is "
                             f"{LIMITS['max_prompt_tokens']} to keep captures bounded")
        rec = {"per_pos": {}, "logit_lens": {}, "hidden": [dict() for _ in range(self.L + 1)],
               "attn": [dict() for _ in range(self.L)]}
        cache = [None] * self.L
        gen = torch.Generator().manual_seed(int(seed))
        t0 = time.monotonic()
        logits = self._forward(ids, 0, cache, rec, lens_rows=True)
        t_prefill = time.monotonic() - t0
        generated, steps = [], []
        stop_reason = "max_new_tokens"
        cur = len(ids)
        t1 = time.monotonic()
        for step in range(max_new_tokens):
            dist = self._topk_tokens(logits, LIMITS["next_token_top_k"])
            probs_full = torch.softmax(logits.float(), -1)
            entropy = float(-(probs_full * torch.log(probs_full.clamp_min(1e-12))).sum())
            if temperature == 0.0:
                nxt = int(torch.argmax(logits))
            else:
                pr = torch.softmax(logits.float() / temperature, -1)
                nxt = int(torch.multinomial(pr, 1, generator=gen))
            steps.append({"predicted_at_pos": cur - 1, "chosen_id": nxt,
                          "chosen_text": self._decode([nxt])[0],
                          "chosen_p": _r(float(probs_full[nxt])), "entropy": _r(entropy),
                          "top": dist})
            generated.append(nxt)
            if nxt in self.eos_ids:
                stop_reason = "eos"
                break
            if step == max_new_tokens - 1:
                break  # the final sampled token is not fed back; no activations exist for it
            logits = self._forward([nxt], cur, cache, rec, lens_rows=True)
            cur += 1
        t_decode = time.monotonic() - t1

        all_ids = ids + generated
        texts = self._decode(all_ids)
        tokens = []
        for i, (tid, tt) in enumerate(zip(all_ids, texts)):
            tokens.append({"pos": i, "id": tid, "text": tt,
                           "segment": "prompt" if i < len(ids) else "generated",
                           "measured": i in rec["per_pos"]})
        run_id = uuid.uuid4().hex[:12]
        attn_bytes = sum(t.numel() * 2 for layer in rec["attn"] for t in layer.values())
        hid_bytes = sum(t.numel() * 2 for layer in rec["hidden"] for t in layer.values())
        # per_pos/steps/temperature are kept so an optional reference cross-check
        # (reference_check.py) can compare the served trace; still latest run only.
        self.last_run = {"run_id": run_id, "attn": rec["attn"], "hidden": rec["hidden"],
                         "tokens": tokens, "per_pos": rec["per_pos"], "steps": steps,
                         "temperature": temperature}
        return {
            "run_id": run_id,
            "provenance": "measured: computed by this local process from the pinned checkpoint",
            "model": {"repo": self.repo_id, "revision": self.revision,
                      "compute_dtype": str(self.dtype).replace("torch.", ""),
                      "stored_dtype": "bfloat16", "device": str(self.device)},
            "settings": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                         "decoding": "greedy" if temperature == 0 else "sampled (seeded)",
                         "seed": seed, "chat_template": use_chat_template,
                         "system_prompt": "custom" if (system and system.strip()) else "template default",
                         "repetition_penalty": "not applied", "limits": LIMITS},
            "timing": {"prefill_s": _r(t_prefill), "decode_s": _r(t_decode),
                       "tokens_per_s": _r(len(generated) / t_decode) if t_decode > 0 else None},
            "dims": {"layers": self.L, "heads": self.H, "kv_heads": self.KVH,
                     "hidden": self.D, "intermediate": self.I, "head_dim": self.hd},
            "prompt_len": len(ids),
            "tokens": tokens,
            "generated_text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "stop_reason": stop_reason,
            "per_pos": {str(k): v for k, v in sorted(rec["per_pos"].items())},
            "logit_lens": {str(k): v for k, v in sorted(rec["logit_lens"].items())},
            "steps": steps,
            "memory_bytes_held": {"attention_fp16": attn_bytes, "hidden_fp16": hid_bytes},
        }

    def release_weights(self):
        """Drop the fp32 weight copies and the mmap (sequential loading).

        The bounded trace of the latest run (self.last_run) is kept, so its attention
        rows and residual vectors can still be served and reference-checked."""
        self.w = None
        self.lm_head = None
        if getattr(self, "st", None) is not None:
            self.st.close()
            self.st = None

    def attention_matrix(self, run_id, layer, head):
        lr = self.last_run
        if not lr or lr["run_id"] != run_id:
            raise KeyError("run not in memory (only the latest run is kept)")
        rows = lr["attn"][layer]
        out = {}
        for P, t in sorted(rows.items()):
            out[str(P)] = [round(float(v), 4) for v in t[head].tolist()]
        return {"run_id": run_id, "layer": layer, "head": head,
                "kv_head": head // (self.H // self.KVH), "rows": out,
                "provenance": "measured softmax attention weights (stored fp16, rounded to 4 dp)"}

    def hidden_vector(self, run_id, layer, pos):
        lr = self.last_run
        if not lr or lr["run_id"] != run_id:
            raise KeyError("run not in memory (only the latest run is kept)")
        t = lr["hidden"][layer].get(pos)
        if t is None:
            raise KeyError("no activation captured at that position")
        return {"run_id": run_id, "layer": layer, "pos": pos,
                "values": [round(float(v), 4) for v in t.tolist()],
                "provenance": "measured residual-stream vector after layer (stored fp16)"}

    @torch.no_grad()
    def consistency_check(self, ids):
        """Full-sequence prefill vs. incremental KV-cache decode: last-position logits diff."""
        dummy = lambda: {"per_pos": {}, "logit_lens": {}, "hidden": [dict() for _ in range(self.L + 1)],
                         "attn": [dict() for _ in range(self.L)]}
        full = self._forward(ids, 0, [None] * self.L, dummy(), False)
        cache, rec = [None] * self.L, dummy()
        inc = self._forward(ids[:-1], 0, cache, rec, False)
        inc = self._forward(ids[-1:], len(ids) - 1, cache, rec, False)
        return float((full - inc).abs().max())

    @torch.no_grad()
    def mean_nll(self, text):
        """Teacher-forced mean negative log-likelihood (nats/token) of plain text."""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        cache = [None] * self.L
        rec = {"per_pos": {}, "logit_lens": {}, "hidden": [dict() for _ in range(self.L + 1)],
               "attn": [dict() for _ in range(self.L)]}
        nll = []
        for i in range(len(ids) - 1):
            lg = self._forward([ids[i]], i, cache, rec, False)
            nll.append(-float(torch.log_softmax(lg, -1)[ids[i + 1]]))
        return sum(nll) / len(nll), len(nll)
