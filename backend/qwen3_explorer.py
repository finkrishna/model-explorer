"""Instrumented Qwen3 forward pass (Qwen3 adapter).

Qwen3 differs from the Qwen2 decoder implemented in ``qwen2_explorer`` in exactly
these ways (per the Hugging Face ``transformers`` Qwen3 implementation):

  * the Q/K/V projections have **no bias** (``attention_bias: false``);
  * each head's query and key are **RMS-normalised over head_dim** (``q_norm`` /
    ``k_norm``) *before* rotary position embedding;
  * ``head_dim`` is an explicit config value (not hidden_size / heads);
  * checkpoints may be **sharded** across several safetensors files;
  * the chat template has a thinking mode; this explorer renders it with
    ``enable_thinking=False`` so answers are comparable to the other models.

Everything else (RMSNorm, rotate-half RoPE, grouped-query attention, SwiGLU MLP,
tied embeddings, all measurement capture and bounds) is inherited unchanged from
the verified Qwen2 engine. Weights are read-only; nothing is written or logged.
The adapter's output is checked against ``transformers`` by ``verify_qwen3.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jinja2
import jinja2.sandbox
import torch
from tokenizers import Tokenizer

from qwen2_explorer import LIMITS, QwenExplorer
from safetensors_io import SafetensorsFile


class ShardedSafetensors:
    """Read-only view over a sharded checkpoint, same API as SafetensorsFile."""

    def __init__(self, snapshot: Path):
        idx = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self.files = {f: SafetensorsFile(snapshot / f) for f in sorted(set(self.weight_map.values()))}
        self.header = {k: self.files[f].header[k] for k, f in self.weight_map.items()}

    def keys(self):
        return sorted(self.weight_map)

    def info(self, name):
        return self.files[self.weight_map[name]].info(name)

    def tensor(self, name):
        return self.files[self.weight_map[name]].tensor(name)

    def close(self):
        for f in self.files.values():
            f.close()


def open_weights(snapshot: Path):
    single = snapshot / "model.safetensors"
    return SafetensorsFile(single) if single.is_file() else ShardedSafetensors(snapshot)


class Qwen3Explorer(QwenExplorer):
    def __init__(self, snapshot: Path, device: str = "cpu", repo_id: str = "", revision: str = ""):
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
        self.dtype = torch.float32

        c = self.config
        if c.get("model_type") != "qwen3":
            raise ValueError(f"Qwen3Explorer cannot run model_type {c.get('model_type')!r}")
        if c.get("attention_bias"):
            raise ValueError("this adapter assumes attention_bias=false (true for all Qwen3 dense models)")
        if c.get("rope_scaling"):
            raise ValueError("rope_scaling is not implemented by this adapter")
        self.L = c["num_hidden_layers"]
        self.D = c["hidden_size"]
        self.H = c["num_attention_heads"]
        self.KVH = c["num_key_value_heads"]
        self.hd = int(c.get("head_dim") or self.D // self.H)
        self.I = c["intermediate_size"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        eos = self.generation_config.get("eos_token_id", c["eos_token_id"])
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

        self.st = open_weights(self.snapshot)
        t0 = time.monotonic()
        self.w = {k: self.st.tensor(k).to(self.device, self.dtype) for k in self.st.keys()}
        self.load_seconds = time.monotonic() - t0
        if c.get("tie_word_embeddings") and "lm_head.weight" not in self.w:
            self.lm_head = self.w["model.embed_tokens.weight"]
        else:
            self.lm_head = self.w["lm_head.weight"]
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        self.inv_freq = inv.to(self.device, torch.float32)
        self.last_run = None

    def render_prompt(self, user: str, system: str | None) -> str:
        messages = []
        if system is not None and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self.chat_template.render(messages=messages, tools=None, add_generation_prompt=True,
                                         enable_thinking=False)

    @torch.no_grad()
    def _forward(self, ids, start_pos, cache, rec, lens_rows):
        T = len(ids)
        pos = torch.arange(start_pos, start_pos + T, device=self.device)
        x = self.w["model.embed_tokens.weight"][torch.tensor(ids, device=self.device)]
        hs_last = [x[-1]]
        self._record_hidden(rec, 0, pos, x, None)
        group = self.H // self.KVH
        scale = 1.0 / math.sqrt(self.hd)
        for l in range(self.L):
            p = f"model.layers.{l}."
            h_in = x
            n1 = self._rms(x, self.w[p + "input_layernorm.weight"])
            q = (n1 @ self.w[p + "self_attn.q_proj.weight"].T).view(T, self.H, self.hd)
            k = (n1 @ self.w[p + "self_attn.k_proj.weight"].T).view(T, self.KVH, self.hd)
            v = (n1 @ self.w[p + "self_attn.v_proj.weight"].T).view(T, self.KVH, self.hd)
            # Qwen3: per-head RMSNorm of q and k over head_dim, before RoPE
            q = self._rms(q, self.w[p + "self_attn.q_norm.weight"])
            k = self._rms(k, self.w[p + "self_attn.k_norm.weight"])
            q = self._rope(q.transpose(0, 1), pos)
            k = self._rope(k.transpose(0, 1), pos)
            v = v.transpose(0, 1)
            if cache[l] is None:
                cache[l] = (k, v)
            else:
                cache[l] = (torch.cat([cache[l][0], k], 1), torch.cat([cache[l][1], v], 1))
            K, V = cache[l]
            S = K.shape[1]
            Kx = K.repeat_interleave(group, 0)
            Vx = V.repeat_interleave(group, 0)
            scores = (q @ Kx.transpose(1, 2)) * scale
            qpos = pos.view(T, 1)
            kpos = torch.arange(S, device=self.device).view(1, S)
            scores = scores.masked_fill((kpos > qpos).unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, -1)
            head_out = probs @ Vx                                   # [H, T, hd]
            Wo = self.w[p + "self_attn.o_proj.weight"]              # [D, H*hd]
            Wo_h = Wo.view(self.D, self.H, self.hd).permute(1, 2, 0)  # [H, hd, D]
            contrib = torch.bmm(head_out, Wo_h)                     # [H, T, D]
            attn_out = contrib.sum(0)
            x = x + attn_out
            n2 = self._rms(x, self.w[p + "post_attention_layernorm.weight"])
            gate = n2 @ self.w[p + "mlp.gate_proj.weight"].T
            up = n2 @ self.w[p + "mlp.up_proj.weight"].T
            act = torch.nn.functional.silu(gate) * up
            mlp_out = act @ self.w[p + "mlp.down_proj.weight"].T
            x = x + mlp_out
            self._record_layer(rec, l, pos, probs, contrib, attn_out, act, mlp_out)
            self._record_hidden(rec, l + 1, pos, x, h_in)
            hs_last.append(x[-1])
        final = self._rms(x, self.w["model.norm.weight"])
        logits = final[-1] @ self.lm_head.T
        if lens_rows:
            Hs = torch.stack(hs_last[:-1])
            lg_all = self._rms(Hs, self.w["model.norm.weight"]) @ self.lm_head.T
            lens = [self._topk_tokens(lg_all[i], LIMITS["logit_lens_top_k"]) for i in range(self.L)]
            lens.append(self._topk_tokens(logits, LIMITS["logit_lens_top_k"]))
            rec["logit_lens"][int(pos[-1])] = lens
        return logits
