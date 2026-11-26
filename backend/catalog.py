"""Verified open-weight model catalog for the public three-model explorer.

Every fact here was read from the official Hugging Face model page / API and the
model's own ``config.json`` at the pinned revision (recorded in ``VERIFIED_FROM``).
Dimensions, parameter counts, weight format/size, context length and licence are
**metadata** copied from those sources -- they are not measured by this tool. The
distinction is carried through to the UI (every catalog field is labelled
"metadata"; only the live comparison table carries "measured" values).

*Runnable* entries carry a ``pin`` block: an architecture with an instrumented,
reference-checked engine here (Qwen2 via ``qwen2_explorer``, Qwen3 via the
``qwen3_explorer`` adapter) and exact provenance (revision, SHA-256 of every weight
file, git blob ids / LFS SHA-256 of the small files, byte sizes). The live comparison
loads a runnable model only after its local snapshot passes that check; the UI asks
the backend which ones verify on *this* machine. The rest are catalog / reference
entries: a curated set of the strongest current open-source models (Apache-2.0 / MIT,
ungated), each stating *why* it is not runnable here. No gated or non-OSI model is
offered as runnable, and no model is ever silently substituted.

This module is self-contained: no network access and no absolute/private paths, so
it can be copied into a standalone public release unchanged.
"""

from __future__ import annotations

VERIFIED_FROM = (
    "Official Hugging Face API (https://huggingface.co/api/models/<repo>?blobs=true) "
    "and each model's config.json at the pinned revision, read 2026-11-25."
)

# Licence ids and whether they are an OSI-approved open-source licence. Gated and
# non-OSI models (e.g. Llama community / Gemma terms) are excluded from this catalog.
_OSI = {"apache-2.0": True, "mit": True, "other": False, "gemma": False, "llama3.2": False}


def head_dim(cfg: dict):
    """head dim from config, or None when the model does not expose standard dims (MoE)."""
    if cfg.get("head_dim"):
        return int(cfg["head_dim"])
    h, a = cfg.get("hidden_size"), cfg.get("num_attention_heads")
    if h and a:
        return int(h) // int(a)
    return None


# ---------------------------------------------------------------------------
# The catalog. Order is display order: the three runnable models first, then the
# curated current open-source reference set (led by the new Inkling-2 series).
# `pin` (runnable only) is the exact provenance the registry verifies before load.
# ---------------------------------------------------------------------------
CATALOG = [
    {
        "id": "qwen2.5-0.5b-instruct", "slot": "a",
        "label": "Qwen2.5-0.5B-Instruct", "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen2.5", "arch": "Qwen2ForCausalLM", "model_type": "qwen2",
        "tuning": "instruction-tuned", "params": 494032768,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": False,
        "weight_bytes": 988097824, "gated": False,
        "config": {"num_hidden_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
                   "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
                   "max_position_embeddings": 32768, "tie_word_embeddings": True,
                   "rope_theta": 1000000.0, "torch_dtype": "bfloat16"},
        "runnable": True, "status_reason": "In local cache; Qwen2 architecture; full provenance verified.",
        "pin": {"revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "weights_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
                "weights_bytes": 988097824,
                "blobs": {"config.json": "0dbb161213629a23f0fc00ef286e6b1e366d180f",
                          "generation_config.json": "dfc11073787daf1b0f9c0f1499487ab5f4c93738",
                          "tokenizer.json": "443909a61d429dff23010e5bddd28ff530edda00",
                          "tokenizer_config.json": "07bfe0640cb5a0037f9322287fbfc682806cf672"}},
    },
    {
        "id": "qwen2.5-1.5b-instruct", "slot": "b",
        "label": "Qwen2.5-1.5B-Instruct", "repo": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen2.5", "arch": "Qwen2ForCausalLM", "model_type": "qwen2",
        "tuning": "instruction-tuned", "params": 1543714304,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": False,
        "weight_bytes": 3087467144, "gated": False,
        "config": {"num_hidden_layers": 28, "hidden_size": 1536, "num_attention_heads": 12,
                   "num_key_value_heads": 2, "intermediate_size": 8960, "vocab_size": 151936,
                   "max_position_embeddings": 32768, "tie_word_embeddings": True,
                   "rope_theta": 1000000.0, "torch_dtype": "bfloat16"},
        "runnable": True, "status_reason": "In local cache; Qwen2 architecture; full provenance verified.",
        "pin": {"revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                "weights_sha256": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
                "weights_bytes": 3087467144,
                "blobs": {"config.json": "f81ead14ab072d65a07817f83a3ee0e5a1890d10",
                          "generation_config.json": "dfc11073787daf1b0f9c0f1499487ab5f4c93738",
                          "tokenizer.json": "443909a61d429dff23010e5bddd28ff530edda00",
                          "tokenizer_config.json": "07bfe0640cb5a0037f9322287fbfc682806cf672"}},
    },
    {
        "id": "qwen2.5-0.5b-base", "slot": "c",
        "label": "Qwen2.5-0.5B (base)", "repo": "Qwen/Qwen2.5-0.5B",
        "family": "Qwen2.5", "arch": "Qwen2ForCausalLM", "model_type": "qwen2",
        "tuning": "base (pretrained, NOT instruction-tuned)", "params": 494032768,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": False,
        "weight_bytes": 988097824, "gated": False,
        "config": {"num_hidden_layers": 24, "hidden_size": 896, "num_attention_heads": 14,
                   "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
                   "max_position_embeddings": 32768, "tie_word_embeddings": True,
                   "rope_theta": 1000000.0, "torch_dtype": "bfloat16"},
        "runnable": True,
        "status_reason": ("In local cache; Qwen2 architecture; full provenance verified. "
                          "Base model: it was not instruction-tuned, so it continues text rather "
                          "than following chat instructions. Shares the 0.5B-Instruct tokenizer."),
        "pin": {"revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
                "weights_sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
                "weights_bytes": 988097824,
                "blobs": {"config.json": "141506237b9022c4ab3e01734cfafb9310b57d2d",
                          "generation_config.json": "cbbb3133034e192527e5321b4c679154e4819ab8",
                          "tokenizer.json": "443909a61d429dff23010e5bddd28ff530edda00",
                          "tokenizer_config.json": "ba7e4c5637b9732dadcd66286ce48334e8b31e9e"}},
    },
    # --------------------------------------------- current best open-source (reference)
    {
        "id": "inkling-2-mini", "slot": None,
        "label": "Inkling-2-Mini", "repo": "thinkingmachines/Inkling-2-Mini",
        "family": "Inkling-2 · Thinking Machines", "arch": "Inkling2LoopForConditionalGeneration",
        "model_type": "inkling2_loop", "tuning": "instruction-tuned (mixture-of-experts)",
        "params": 26388646440, "license": "apache-2.0", "weight_format": "safetensors",
        "sharded": True, "weight_bytes": 20161820584, "gated": False,
        "config": {"num_hidden_layers": None, "hidden_size": None, "num_attention_heads": None,
                   "num_key_value_heads": None, "intermediate_size": None, "vocab_size": None,
                   "max_position_embeddings": None, "tie_word_embeddings": None,
                   "rope_theta": None, "torch_dtype": None, "moe": True},
        "runnable": False, "revision": "c9d028302cd0f035fdf8e7e91f089a4ce94d2195",
        "status_reason": ("Thinking Machines' new open-weight Inkling-2 series (Apache-2.0, ungated). "
                          "A ~26B-parameter mixture-of-experts with a custom 'inkling2_loop' "
                          "architecture that does not expose standard transformer dims and is far "
                          "outside the Qwen2 engine. Not cached. Catalog/reference only."),
    },
    {
        "id": "qwen3-1.7b", "slot": "e",
        "label": "Qwen3-1.7B", "repo": "Qwen/Qwen3-1.7B",
        "family": "Qwen3", "arch": "Qwen3ForCausalLM", "model_type": "qwen3",
        "tuning": "post-trained (hybrid thinking; run with thinking off)", "params": 2031739904,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": True,
        "weight_bytes": 4063515592, "gated": False,
        "config": {"num_hidden_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
                   "num_key_value_heads": 8, "intermediate_size": 6144, "vocab_size": 151936,
                   "max_position_embeddings": 40960, "tie_word_embeddings": True,
                   "head_dim": 128, "rope_theta": 1000000, "torch_dtype": "bfloat16"},
        "runnable": True,
        "status_reason": ("Current Qwen3 generation, run by this tool's Qwen3 adapter (no QKV bias, "
                          "per-head Q/K RMSNorm before RoPE, explicit head_dim, sharded weights) and "
                          "checked against Hugging Face transformers. Rendered with thinking mode off."),
        "pin": {"revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "weights_sha256": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
                "weights_bytes": 4063515592,
                "shards": {"model-00001-of-00002.safetensors": {
                               "sha256": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
                               "bytes": 3441185608},
                           "model-00002-of-00002.safetensors": {
                               "sha256": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
                               "bytes": 622329984}},
                "blobs": {"config.json": "044a86ecf7cb32238f3fae4184e55d354787edec",
                          "generation_config.json": "20a8a9156fc8c3f25295ca067f61fdf120d517c5",
                          "tokenizer_config.json": "417d038a63fa3de29cfde265caedae14d1a58d92",
                          "model.safetensors.index.json": "986d7db875b47d21f68530f6baac038f1b297b39"},
                "lfs_small": {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}},
    },
    {
        "id": "smollm3-3b", "slot": None,
        "label": "SmolLM3-3B", "repo": "HuggingFaceTB/SmolLM3-3B",
        "family": "SmolLM3", "arch": "SmolLM3ForCausalLM", "model_type": "smollm3",
        "tuning": "instruction-tuned", "params": 3075098624,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": True,
        "weight_bytes": 6150235008, "gated": False,
        "config": {"num_hidden_layers": 36, "hidden_size": 2048, "num_attention_heads": 16,
                   "num_key_value_heads": 4, "intermediate_size": 11008, "vocab_size": 128256,
                   "max_position_embeddings": 65536, "tie_word_embeddings": True,
                   "rope_theta": 5000000.0, "torch_dtype": "bfloat16"},
        "runnable": False, "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "status_reason": ("Hugging Face's fully-open SmolLM3 (open weights, data and recipe). Its "
                          "SmolLM3 architecture and tokenizer differ from Qwen2; an adapter would be "
                          "required, and it is not cached. Catalog/reference only."),
    },
    {
        "id": "phi-4-mini-instruct", "slot": None,
        "label": "Phi-4-mini-instruct", "repo": "microsoft/Phi-4-mini-instruct",
        "family": "Phi-4", "arch": "Phi3ForCausalLM", "model_type": "phi3",
        "tuning": "instruction-tuned", "params": 3836021760,
        "license": "mit", "weight_format": "safetensors", "sharded": True,
        "weight_bytes": 7672066216, "gated": False,
        "config": {"num_hidden_layers": 32, "hidden_size": 3072, "num_attention_heads": 24,
                   "num_key_value_heads": 8, "intermediate_size": 8192, "vocab_size": 200064,
                   "max_position_embeddings": 131072, "tie_word_embeddings": True,
                   "rope_theta": 10000.0, "torch_dtype": "bfloat16"},
        "runnable": False, "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "status_reason": ("Microsoft Phi-4-mini (MIT). The Phi-3/4 architecture (fused QKV, partial "
                          "rotary, 200K vocab) differs from Qwen2; an adapter would be required, and "
                          "it is not cached. Catalog/reference only."),
    },
    {
        "id": "olmo-2-7b-instruct", "slot": None,
        "label": "OLMo-2-7B-Instruct", "repo": "allenai/OLMo-2-1124-7B-Instruct",
        "family": "OLMo-2 · Allen AI", "arch": "Olmo2ForCausalLM", "model_type": "olmo2",
        "tuning": "instruction-tuned", "params": 7298617344,
        "license": "apache-2.0", "weight_format": "safetensors", "sharded": True,
        "weight_bytes": 14597276128, "gated": False,
        "config": {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32,
                   "num_key_value_heads": 32, "intermediate_size": 11008, "vocab_size": 100352,
                   "max_position_embeddings": 4096, "tie_word_embeddings": False,
                   "rope_theta": 500000, "torch_dtype": "bfloat16"},
        "runnable": False, "revision": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
        "status_reason": ("Allen AI OLMo-2 — a fully-open model (open weights, training data and "
                          "code). The OLMo-2 architecture (QK-norm, reordered norms) differs from "
                          "Qwen2; an adapter would be required, and it is not cached. Reference only."),
    },
    {
        "id": "deepseek-r1-distill-qwen-1.5b", "slot": "d",
        "label": "DeepSeek-R1-Distill-Qwen-1.5B", "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "DeepSeek-R1-Distill", "arch": "Qwen2ForCausalLM", "model_type": "qwen2",
        "tuning": "reasoning distillation (Qwen2 base)", "params": 1777088000,
        "license": "mit", "weight_format": "safetensors", "sharded": False,
        "weight_bytes": 3554214621, "gated": False,
        "config": {"num_hidden_layers": 28, "hidden_size": 1536, "num_attention_heads": 12,
                   "num_key_value_heads": 2, "intermediate_size": 8960, "vocab_size": 151936,
                   "max_position_embeddings": 131072, "tie_word_embeddings": False,
                   "rope_theta": 10000, "torch_dtype": "bfloat16"},
        "runnable": True,
        "status_reason": ("DeepSeek's R1 reasoning distilled into a Qwen2 1.5B (MIT). Qwen2 architecture, "
                          "so the base engine runs it unchanged; checkpoint downloaded with owner approval "
                          "and verified against the official hashes. It reasons in a <think> block first, "
                          "so short token budgets may end mid-reasoning."),
        "pin": {"revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
                "weights_sha256": "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
                "weights_bytes": 3554214621,
                "blobs": {"config.json": "68c044b63c894bc0c3d334a44bed2856d6d6815d",
                          "generation_config.json": "052ab54633116a634da950ab483233c4ace0aa82",
                          "tokenizer.json": "a34650995da6939a945c330eadb0687147ac3ef8",
                          "tokenizer_config.json": "9967ff32d94b21c94dc7e2b3bcbea295a46cde50"}},
    },
    {
        "id": "granite-4.0-h-micro", "slot": None,
        "label": "Granite-4.0-H-Micro", "repo": "ibm-granite/granite-4.0-h-micro",
        "family": "Granite 4.0 · IBM", "arch": "GraniteMoeHybridForCausalLM",
        "model_type": "granitemoehybrid", "tuning": "instruction-tuned (hybrid MoE)",
        "params": 3191396096, "license": "apache-2.0", "weight_format": "safetensors",
        "sharded": True, "weight_bytes": 6382844048, "gated": False,
        "config": {"num_hidden_layers": 40, "hidden_size": 2048, "num_attention_heads": 32,
                   "num_key_value_heads": 8, "intermediate_size": 8192, "vocab_size": 100352,
                   "max_position_embeddings": 131072, "tie_word_embeddings": True,
                   "rope_theta": 10000, "torch_dtype": "bfloat16", "moe": True},
        "runnable": False, "revision": "d5f01a3ea75f088947be3aae039f4ad52837dfde",
        "status_reason": ("IBM Granite 4.0 (Apache-2.0) — a hybrid Mamba/transformer mixture-of-"
                          "experts. Its hybrid architecture is far outside the Qwen2 transformer "
                          "engine, and it is not cached. Catalog/reference only."),
    },
]

assert len(CATALOG) <= 10, "catalog must have at most ten entries"

# Verified first-publication date = the model repo's createdAt on the Hugging Face API
# (read 2026-11-25). Used for the catalog timeline.
RELEASED = {
    "qwen2.5-0.5b-base": "2024-09-15", "qwen2.5-0.5b-instruct": "2024-09-16",
    "qwen2.5-1.5b-instruct": "2024-09-17", "olmo-2-7b-instruct": "2024-12-18",
    "deepseek-r1-distill-qwen-1.5b": "2025-01-20", "phi-4-mini-instruct": "2025-02-19",
    "smollm3-3b": "2025-07-08", "qwen3-1.7b": "2025-04-27",
    "granite-4.0-h-micro": "2025-09-16", "inkling-2-mini": "2026-11-17",
}

# Publisher-reported headline benchmark, each labelled with the EXACT benchmark variant
# and protocol it was reported under. These are self-reported by each publisher under
# DIFFERENT protocols (shot count, base vs instruct, MMLU vs MMLU-Pro/Redux/Global vs
# MATH) and are NOT a controlled, head-to-head comparison -- they point to each model's
# own card. `value=None` means no clean comparable figure was found; see the card.
BENCHMARK = {
    "qwen2.5-0.5b-base": {"label": "MMLU", "value": 47.5, "protocol": "base · 5-shot (Qwen2.5 report)"},
    "qwen2.5-0.5b-instruct": {"label": "MMLU-Redux", "value": 24.1, "protocol": "instruct (Qwen2.5 report)"},
    "qwen2.5-1.5b-instruct": {"label": "MMLU-Redux", "value": 50.7, "protocol": "instruct (Qwen2.5 report)"},
    "qwen3-1.7b": {"label": None, "value": None, "protocol": "see model card / Qwen3 report"},
    "smollm3-3b": {"label": "Global-MMLU", "value": 53.5, "protocol": "0-shot · multilingual"},
    "phi-4-mini-instruct": {"label": "MMLU", "value": 67.3, "protocol": "5-shot (model card)"},
    "olmo-2-7b-instruct": {"label": None, "value": None, "protocol": "see model card"},
    "deepseek-r1-distill-qwen-1.5b": {"label": "MATH-500", "value": 83.9, "protocol": "pass@1 · reasoning"},
    "granite-4.0-h-micro": {"label": "MMLU", "value": 67.4, "protocol": "5-shot (model card)"},
    "inkling-2-mini": {"label": None, "value": None, "protocol": "see model card (new release)"},
}

_BY_ID = {e["id"]: e for e in CATALOG}
RUNNABLE_SLOTS = {e["slot"]: e["id"] for e in CATALOG if e["runnable"]}  # {"a":..,"b":..,"c":..}


def by_id(mid: str) -> dict:
    if mid not in _BY_ID:
        raise KeyError(f"unknown model id: {mid!r}")
    return _BY_ID[mid]


def is_osi(license_id: str) -> bool:
    return bool(_OSI.get(license_id, False))


def public_view(entry: dict) -> dict:
    """Catalog facts for the UI, with explicit metadata/derived labels. No secrets, no paths."""
    cfg = entry["config"]
    rev = entry.get("pin", {}).get("revision") or entry.get("revision")
    return {
        "id": entry["id"], "label": entry["label"], "repo": entry["repo"],
        "family": entry["family"], "tuning": entry["tuning"],
        "architecture": entry["arch"], "model_type": entry["model_type"],
        "runnable": entry["runnable"], "slot": entry.get("slot"),
        "status_reason": entry["status_reason"], "revision": rev,
        "released": RELEASED.get(entry["id"]),
        "benchmark": BENCHMARK.get(entry["id"]),
        "license": {"id": entry["license"], "osi_open_source": is_osi(entry["license"]),
                    "gated": entry["gated"],
                    "note": ("OSI-approved open-source licence." if is_osi(entry["license"])
                             else "NOT an OSI-approved open-source licence -- review before reuse."),
                    "source": f"https://huggingface.co/{entry['repo']}/blob/{rev}/LICENSE"},
        "source_links": {
            "model_card": f"https://huggingface.co/{entry['repo']}",
            "config": f"https://huggingface.co/{entry['repo']}/blob/{rev}/config.json",
        },
        # every field below is METADATA read from the model card/config, not measured here
        "metadata": {
            "parameters": entry["params"], "layers": cfg.get("num_hidden_layers"),
            "hidden_size": cfg.get("hidden_size"), "attention_heads": cfg.get("num_attention_heads"),
            "kv_heads": cfg.get("num_key_value_heads"), "head_dim": head_dim(cfg),
            "mlp_intermediate": cfg.get("intermediate_size"), "vocab_size": cfg.get("vocab_size"),
            "context_length": cfg.get("max_position_embeddings"),
            "tied_embeddings": cfg.get("tie_word_embeddings"), "rope_theta": cfg.get("rope_theta"),
            "stored_dtype": cfg.get("torch_dtype"), "weight_format": entry["weight_format"],
            "sharded": entry["sharded"], "weight_bytes": entry["weight_bytes"],
            "mixture_of_experts": bool(cfg.get("moe")),
        },
        "provenance": ({"verified": True, "revision": entry["pin"]["revision"],
                        "weights_sha256": entry["pin"]["weights_sha256"],
                        "weights_bytes": entry["pin"]["weights_bytes"]}
                       if entry["runnable"] else
                       {"verified": False,
                        "note": "Metadata pinned to the revision above; weights are not loaded here."}),
    }


def catalog_payload() -> dict:
    return {"verified_from": VERIFIED_FROM, "count": len(CATALOG),
            "runnable_count": sum(1 for e in CATALOG if e["runnable"]),
            "models": [public_view(e) for e in CATALOG]}
