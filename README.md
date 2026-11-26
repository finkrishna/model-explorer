# Open-weight model explorer — compare models on one prompt, locally

A local-first tool to compare up to **three open-weight language models** on one prompt and
inspect their **measured internals** side by side, plus a static, publishable **catalog** of ten
current open-source models with a release timeline, publisher-reported benchmarks and an
architecture/weights page for each.

Two layers:

- **Static site** (catalog, timeline, per-model pages) — no backend, no network. Safe for a
  static host such as GitHub Pages.
- **Local comparison backend** (loopback only) — loads pinned checkpoints from your own Hugging
  Face cache and runs the prompt with an instrumented forward pass. Prompts, answers and
  activations never leave your machine; nothing is written to disk or logged.

## Requirements

Python 3.9+ with `torch`, `safetensors`, `tokenizers` and `jinja2` (for chat templates), plus the
Hugging Face CLI (`hf`) if you need to fetch a checkpoint. `pytest` for the tests; `transformers`
only for `backend/verify_engine.py`. No GPU is needed — everything runs fp32 on CPU.

## Runnable models

A model is runnable only if (1) an instrumented engine for its architecture exists here and has
been checked against Hugging Face `transformers`, and (2) its local snapshot passes an exact
provenance check (revision, SHA-256 of every weight file, small-file ids). The Compare page asks
the backend which ones verify **on your machine**; anything else is shown as reference specs only.

| Slot | Model | Engine | Licence | Revision | Weights |
|---|---|---|---|---|---|
| A | `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2 | Apache-2.0 | `7ae557604adf…` | 0.99 GB, sha256 `fdf756fa…` |
| B | `Qwen/Qwen2.5-1.5B-Instruct` | Qwen2 | Apache-2.0 | `989aa7980e4c…` | 3.09 GB, sha256 `dd924a11…` |
| C | `Qwen/Qwen2.5-0.5B` (**base**) | Qwen2 | Apache-2.0 | `060db6499f32…` | 0.99 GB, sha256 `88c14255…` |
| D | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2 | MIT | `ad9f0ae0864d…` | 3.55 GB, sha256 `58858233…` |
| E | `Qwen/Qwen3-1.7B` | Qwen3 adapter | Apache-2.0 | `70d244cc86cc…` | 4.06 GB in 2 shards |

Full pins are in `backend/catalog.py`. The server never downloads. To fetch a checkpoint
yourself (one-time, from Hugging Face):

```bash
HF_HUB_DISABLE_TELEMETRY=1 hf download Qwen/Qwen3-1.7B \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```
(Use the matching repo, files and revision from the table for the others.)

**Engines.** `backend/qwen2_explorer.py` is a dependency-light, instrumented Qwen2 forward pass. `backend/qwen3_explorer.py` adapts it to Qwen3: no QKV bias, per-head Q/K RMSNorm
before RoPE, explicit `head_dim`, sharded weights, chat template rendered with thinking off.
DeepSeek-R1-Distill is a Qwen2 architecture and runs on the base engine; note that it reasons in a
`<think>` block first, so short token budgets may stop mid-reasoning.

## Catalog

Ten entries: the five runnable models above plus five reference-only models, all Apache-2.0 / MIT
and ungated — **Inkling-2-Mini** (Thinking Machines' open-weight MoE), **SmolLM3-3B**,
**Phi-4-mini-instruct**, **OLMo-2-7B-Instruct** and **Granite-4.0-H-Micro**. Each reference card
says why it can't run here (a different architecture that would need its own verified adapter,
size, not cached). Gated or non-OSI models (e.g. Gemma, Llama) are excluded. Mixture-of-experts
models that don't publish standard dims show those fields as "—" rather than invented values.

The catalog page also shows:
- a **release timeline** — verified first-publication date on Hugging Face against parameter count;
- **publisher-reported benchmarks** — each publisher's own headline number, labelled with the exact
  benchmark and protocol. These are **not** a head-to-head comparison (different benchmarks,
  shot counts, base vs instruct); "see card" means no clean figure was found.

## Run the live comparison

```bash
python3 backend/server.py            # serves http://127.0.0.1:8777/
```

Pick any three of the ten models and one prompt, then press **Compare**. Runnable models get a
measured column; reference models get a verified-spec column with measured rows marked "not run".
Every row has an info button (hover, click or keyboard) explaining what it shows, whether it is
measured / derived / metadata, and what it **cannot** tell you. Models load one at a time and are
released, so peak memory stays near the largest single model.

**Visualizations** (all from the measured trace, inline SVG/canvas, no libraries): decoding
confidence per generated token, residual-stream growth by depth, attention-vs-MLP write per layer,
per-token confidence heatstrips, and a **layer × token heatmap** with a metric picker (attention
write, MLP write, residual norm, attention entropy).

## Privacy / publishing

- The backend binds to **127.0.0.1 only** and rejects non-loopback `Host` headers.
- Every response and every static page sets a Content-Security-Policy whose `connect-src` allows
  only this origin and the local loopback — no external origin, so a prompt cannot reach
  telemetry or a remote inference endpoint. Inline styles/scripts are also blocked; the UI styles
  elements through the CSSOM, which the policy allows.
- Served from a public host, the live comparison is **disabled** and no prompt is ever sent.
- Prompts, answers and activations exist only in memory for the latest run; the request log records
  method + path only.

## Verify

```bash
python3 backend/selfcheck3.py        # end-to-end: loopback, CSP, host rejection, real run,
                                     # provenance, prompt-not-logged, negative controls
python3 -m pytest tests/             # catalog, provenance controls, explanations, privacy
<venv-with-torch+transformers>/bin/python backend/verify_engine.py --model qwen3-1.7b
                                     # engine vs transformers on identical token ids:
                                     # per-layer norms, logits, 20-token greedy decode
```

### Verification results (2026-11-26)

`verify_engine.py`, our engine vs `transformers` 5.24 (fp32) on identical token ids:

| Model | Engine | max \|Δ logit\| | max rel Δ ‖h‖ | greedy 20 tokens |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Qwen2 | 7.4e-5 | 4.6e-4 | identical |
| Qwen2.5-1.5B-Instruct | Qwen2 | 5.1e-4 | 4.7e-4 | identical |
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2 | 6.7e-5 | 4.6e-4 | identical |
| Qwen3-1.7B | Qwen3 adapter | 9.7e-5 | 4.8e-4 | identical |
| Qwen3-1.7B, broken on purpose (negative control) | — | 36.6 | 1.01 | diverges at token 0 |

Norm differences come from the engine rounding captured values to 4 significant figures.
Qwen2.5-0.5B base shares the 0.5B-Instruct architecture and dimensions and was not run separately.
This is an end-to-end check (logits, norms, greedy decode), not a per-tensor equality proof; a
stricter per-value check of 1.5B internals found a localized tolerance exceedance.
Also: `pytest` 57 passed, `selfcheck3.py` 19/19.

## Limits

- Greedy decoding by default; bounded capture (256 prompt tokens, ≤96 new tokens).
- "Measured" = computed from the checkpoint this run; "derived" = summaries of measured values;
  "metadata" = copied from the model card, not measured. Confidence/entropy are not correctness.
- Tokens are aligned across models only when their tokenizers produce identical prompt ids.
- 7B+ dense and large MoE models don't fit in CPU memory on a 16 GB machine even with an adapter.

## Licence

No licence has been chosen yet, so all rights are reserved for now. Model weights are not part of
this repository; each model is under its own licence (shown on its catalog page).
