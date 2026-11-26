"""Plain-language explanations for every comparison row and catalog field.

Single source of truth for the hover / click-through popovers. The local server
exposes this at /api/metrics; the static-site builder writes the same content to
``explanations.json`` so the public catalog pages work with no backend. Each entry:

    kind   -- "measured" (computed from the checkpoint this session),
              "derived" (computed from measured values),
              "metadata" (copied from the model card/config; not measured here).
    title  -- short human label.
    plain  -- what it shows, in lay terms.
    read   -- how to read / compare it across models.
    cannot -- what it does NOT establish (the honest limit).

No prompt text, secrets or private paths appear here.
"""

from __future__ import annotations

KIND_LABELS = {
    "measured": "Measured now — computed by this tool from the model's own weights during your run.",
    "derived": "Derived — calculated from measured values (a summary, not a direct reading).",
    "metadata": "Metadata — copied from the official model card/config. Not measured here.",
}

METRICS = {
    # ---- identity / metadata rows -----------------------------------------
    "model": {"kind": "metadata", "title": "Model",
              "plain": "Which checkpoint this column is, and whether it was instruction-tuned or is a raw base model.",
              "read": "A base model continues text; an instruction-tuned model follows chat-style requests. Expect different answers for that reason alone.",
              "cannot": "The name does not tell you quality; compare the actual answers and internals."},
    "revision": {"kind": "metadata", "title": "Revision (pin)",
                 "plain": "The exact Git commit of the model on Hugging Face that was loaded, so the run is reproducible.",
                 "read": "Same revision = same weights. The provenance row shows it was hash-checked before loading.",
                 "cannot": "It does not change between runs; it is an identity, not a measurement."},
    "provenance": {"kind": "measured", "title": "Provenance check",
                   "plain": "Before loading, the local file's SHA-256 and git blob ids are compared to the official Hugging Face values.",
                   "read": "PASS means the bytes on disk are exactly the published checkpoint. A mismatch aborts the run; nothing is substituted.",
                   "cannot": "It proves file identity, not that the model is good or safe for any use."},
    "license": {"kind": "metadata", "title": "Licence",
                "plain": "The licence the model is published under, and whether that licence is OSI-approved open source.",
                "read": "Apache-2.0 is OSI open source. 'Other' (e.g. Qwen Research) is use-restricted and is not treated as OSS here.",
                "cannot": "This tool shows the licence id; it is not legal advice."},
    "params": {"kind": "metadata", "title": "Parameters",
               "plain": "How many numbers (weights) the model has, from the official card.",
               "read": "More parameters usually means more capacity and more memory/time, not guaranteed better answers.",
               "cannot": "Parameter count alone does not predict the answer to your prompt."},
    "dims": {"kind": "metadata", "title": "Shape (layers x width x heads)",
             "plain": "The model's architecture: number of transformer layers, the width of the residual stream, and attention head counts.",
             "read": "Taller/wider models have more internal steps. Query vs KV heads shows grouped-query attention (fewer KV heads saves memory).",
             "cannot": "Shape is metadata from config.json; it is not measured from the weights here."},
    # ---- same-input parity -------------------------------------------------
    "prompt_parity": {"kind": "measured", "title": "Same input?",
                      "plain": "Confirms all models received identical user text and decoding settings, and whether their tokenizers produced the same token ids.",
                      "read": "If the rendered prompt hashes match, the comparison is apples-to-apples. If token ids differ, positions can't be aligned 1:1 and each trace is read on its own.",
                      "cannot": "Equal settings don't force equal answers; models still differ."},
    "tokenization": {"kind": "measured", "title": "Prompt tokens",
                     "plain": "How many tokens each model split your prompt into after its chat template.",
                     "read": "Same-family models often tokenize identically; different families usually don't. Different counts mean token positions are not comparable across those models.",
                     "cannot": "Token count is about the tokenizer, not about understanding."},
    # ---- the answer --------------------------------------------------------
    "answer": {"kind": "measured", "title": "Answer",
               "plain": "The text each model generated for the shared prompt, with greedy (deterministic) decoding by default.",
               "read": "This is the headline comparison. Base vs instruct and bigger vs smaller models typically diverge here.",
               "cannot": "A fluent answer is not necessarily a correct one; the tool does not fact-check."},
    "stop_reason": {"kind": "measured", "title": "Why it stopped",
                    "plain": "Whether the model emitted an end-of-text token or hit the token budget.",
                    "read": "'eos' means the model chose to stop; 'max_new_tokens' means it was cut off by the length cap.",
                    "cannot": "Hitting the cap doesn't mean the answer was wrong, only unfinished."},
    # ---- measured internals (per model) -----------------------------------
    "first_token_confidence": {"kind": "measured", "title": "First-token confidence",
                               "plain": "The probability the model assigned to the very first token it chose to generate.",
                               "read": "Higher = the model was more sure of how to start. Compare how decisively each model begins.",
                               "cannot": "Confidence is not correctness; a model can be confidently wrong."},
    "mean_entropy": {"kind": "derived", "title": "Avg. decoding uncertainty",
                     "plain": "Average Shannon entropy of the next-token distribution across the generated tokens (in nats).",
                     "read": "Lower means the model was consistently more certain token-to-token; higher means it was hedging among many options.",
                     "cannot": "It summarizes the probability spread, not whether the content is accurate."},
    "residual_growth": {"kind": "measured", "title": "Residual-stream growth",
                        "plain": "How much the internal vector at the last prompt token grows in length from the first layer to the last.",
                        "read": "Shows how much each layer adds to the running 'thought' vector. Useful to compare depth of processing between models.",
                        "cannot": "A bigger number is not 'more understanding'; it's the measured norm, nothing more."},
    "attention_entropy": {"kind": "measured", "title": "Attention focus (layer 0)",
                          "plain": "Average entropy of the attention weights in the first layer: low = heads focus on few tokens, high = they spread attention widely.",
                          "read": "Compare whether a model's early attention is sharp or diffuse on the same prompt.",
                          "cannot": "Attention weights show where the model looked, not why, and not what it concluded."},
    "timing": {"kind": "measured", "title": "Compute time",
               "plain": "Seconds this machine took to prefill the prompt and to generate the answer for each model.",
               "read": "Bigger models cost more time and memory. This is hardware-dependent, for rough comparison only.",
               "cannot": "Timing depends on your machine and load; it is not a model quality metric."},
    "memory": {"kind": "measured", "title": "Peak memory",
               "plain": "The highest resident memory the whole comparison process reached while loading models one at a time.",
               "read": "Indicates the RAM headroom you need. Models are loaded sequentially and released to keep this bounded.",
               "cannot": "It reflects this run's engine and host, not an inherent model property."},
    # ---- alignment ---------------------------------------------------------
    "benchmark": {"kind": "metadata", "title": "Publisher-reported benchmark",
                  "plain": "The headline benchmark score the model's publisher reports on its card or report.",
                  "read": "Use it to find a model's own claimed strength. Note which benchmark and protocol it is.",
                  "cannot": "Publishers use different benchmarks and settings, so these numbers can't rank models against each other."},
    "layer_token_heatmap": {"kind": "measured", "title": "Layer × token heatmap",
                            "plain": "One cell per token (across) and layer (down), coloured by the chosen measured quantity.",
                            "read": "Bright cells mark where in the sequence and at which depth that quantity is large. Compare the patterns across models.",
                            "cannot": "It shows magnitudes and spread, not what a layer 'means' or why the model chose a token."},
    # ---- visualizations ----------------------------------------------------
    "viz_confidence": {"kind": "measured", "title": "Decoding confidence over the answer",
                       "plain": "For each token the model generated, the probability it assigned to the token it chose.",
                       "read": "Peaks = the model was sure; dips = it hesitated among options. Compare how steady each model is, and where a base vs instruct model wobbles.",
                       "cannot": "High probability is the model's own certainty, not a guarantee the token is correct."},
    "viz_layer_dynamics": {"kind": "measured", "title": "Where each layer does its work",
                           "plain": "Layer by layer, the size of what attention writes and what the MLP writes into the running vector, at the last prompt token.",
                           "read": "Tall bars/points mark the layers doing the most rewriting. Models often concentrate work in early and late layers; compare the shapes.",
                           "cannot": "Write size is a magnitude, not meaning; it doesn't say what the layer computed."},
    "viz_residual_depth": {"kind": "measured", "title": "Residual-stream growth by depth",
                           "plain": "The length of the internal vector at each layer, plotted against relative depth so models of different heights line up.",
                           "read": "A steeper climb means later layers keep adding more to the 'thought' vector. Compare the trajectories across models.",
                           "cannot": "A larger norm is not 'more understanding'; it is only the measured vector length."},
    "token_heatstrip": {"kind": "measured", "title": "Per-token confidence strip",
                        "plain": "Each generated token, shaded by how confident the model was when it chose it (darker = more confident).",
                        "read": "Scan for pale tokens: those are the model's uncertain moments. Hover a token to see its exact probability.",
                        "cannot": "Colour reflects the model's probability, not whether the word is right."},
    "token_alignment": {"kind": "derived", "title": "Token alignment",
                        "plain": "How many leading generated tokens the models produced identically before they first diverged.",
                        "read": "Only shown when tokenizers match. More shared tokens = the models agreed for longer before branching.",
                        "cannot": "Divergence is expected and normal; it does not mean one model is wrong."},
}

# The ordered rows of the comparison table (id -> which METRICS key explains it).
COMPARISON_ROWS = [
    "model", "provenance", "license", "params", "dims",
    "answer", "stop_reason", "tokenization",
    "first_token_confidence", "mean_entropy", "residual_growth",
    "attention_entropy", "timing", "memory",
]


def metrics_payload() -> dict:
    return {"kind_labels": KIND_LABELS, "metrics": METRICS,
            "comparison_rows": COMPARISON_ROWS,
            "note": "Explanations describe what each row shows and its honest limits. "
                    "Labels: measured (computed now), derived (from measured), metadata (from the model card)."}
