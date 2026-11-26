"""Row explanations: coverage, required fields, honest-limit text."""
import explanations


def test_every_comparison_row_has_an_explanation():
    for row in explanations.COMPARISON_ROWS:
        assert row in explanations.METRICS, f"no explanation for row {row}"


def test_metric_fields_present_and_valid_kind():
    valid = set(explanations.KIND_LABELS)
    for mid, m in explanations.METRICS.items():
        for field in ("kind", "title", "plain", "read", "cannot"):
            assert m.get(field), f"{mid} missing {field}"
        assert m["kind"] in valid, f"{mid} has unknown kind {m['kind']}"


def test_every_used_kind_has_a_label():
    used = {m["kind"] for m in explanations.METRICS.values()}
    assert used <= set(explanations.KIND_LABELS)


def test_cannot_text_is_an_honest_limit():
    # Each explanation must state a limit (non-trivial "what it can't tell you").
    for mid, m in explanations.METRICS.items():
        assert len(m["cannot"]) > 15, f"{mid} has a too-short limit statement"


def test_payload_shape():
    p = explanations.metrics_payload()
    assert set(p) >= {"kind_labels", "metrics", "comparison_rows"}
    assert p["comparison_rows"] == explanations.COMPARISON_ROWS
