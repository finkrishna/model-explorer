"""Catalog integrity + licence labelling."""
import catalog


def test_at_most_ten_entries():
    assert len(catalog.CATALOG) <= 10


def test_runnable_entries_have_distinct_slots():
    runnable = [e for e in catalog.CATALOG if e["runnable"]]
    slots = [e["slot"] for e in runnable]
    assert len(slots) == len(set(slots)), "runnable slots must be unique"
    assert {"a", "b", "c"} <= set(slots)
    # every non-runnable entry has no slot
    assert all(e["slot"] is None for e in catalog.CATALOG if not e["runnable"])


def test_runnable_models_have_a_verified_engine():
    # only architectures with an instrumented, reference-checked forward pass may be runnable
    for e in catalog.CATALOG:
        if e["runnable"]:
            assert e["model_type"] in ("qwen2", "qwen3"), e["id"]


def test_release_dates_and_benchmarks_cover_all():
    for e in catalog.CATALOG:
        v = catalog.public_view(e)
        assert v["released"] and len(v["released"]) == 10, e["id"]
        b = v["benchmark"]
        assert b is not None and "protocol" in b, e["id"]
        if b["value"] is not None:
            assert b["label"], f"{e['id']}: a benchmark value must name its benchmark"


def test_ids_are_unique():
    ids = [e["id"] for e in catalog.CATALOG]
    assert len(ids) == len(set(ids))


def test_runnable_entries_have_full_pins():
    for e in catalog.CATALOG:
        if e["runnable"]:
            pin = e["pin"]
            assert len(pin["revision"]) == 40
            assert len(pin["weights_sha256"]) == 64
            assert pin["weights_bytes"] > 0
            assert {"config.json", "generation_config.json", "tokenizer_config.json"} <= set(pin["blobs"])
            # tokenizer.json is pinned either by git blob id or (if stored in LFS) by sha256
            assert "tokenizer.json" in pin["blobs"] or "tokenizer.json" in pin.get("lfs_small", {})
            if "shards" in pin:
                assert sum(s["bytes"] for s in pin["shards"].values()) == pin["weights_bytes"]
                assert all(len(s["sha256"]) == 64 for s in pin["shards"].values())


def test_non_osi_licences_are_not_runnable():
    # A use-restricted / non-OSI licence must never be offered as a runnable model.
    for e in catalog.CATALOG:
        if not catalog.is_osi(e["license"]):
            assert not e["runnable"], f"{e['id']} has non-OSI licence but is runnable"


def test_gated_models_are_not_runnable():
    for e in catalog.CATALOG:
        if e["gated"]:
            assert not e["runnable"]


def test_license_labeling_in_public_view():
    # non-OSI ids must be flagged by is_osi (even though none are currently runnable/listed)
    assert catalog.is_osi("other") is False
    assert catalog.is_osi("gemma") is False
    assert catalog.is_osi("llama3.2") is False
    assert catalog.is_osi("apache-2.0") is True
    assert catalog.is_osi("mit") is True
    v = catalog.public_view(catalog.by_id("qwen2.5-0.5b-instruct"))
    assert v["license"]["osi_open_source"] is True
    assert "OSI-approved" in v["license"]["note"]


def test_all_catalog_licences_are_osi():
    # the refreshed catalog is an all-OSS lineup (Apache-2.0 / MIT); no gated/non-OSI entries
    for e in catalog.CATALOG:
        assert catalog.is_osi(e["license"]), f"{e['id']} has non-OSI licence {e['license']}"
        assert e["gated"] is False


def test_public_view_core_fields_and_provenance():
    for e in catalog.CATALOG:
        v = catalog.public_view(e)
        md = v["metadata"]
        # every entry must carry params, weight size and a real source link
        assert md["parameters"] and md["weight_bytes"]
        assert v["source_links"]["model_card"].startswith("https://huggingface.co/")
        assert v["provenance"]["verified"] == e["runnable"]
        # standard-transformer entries must expose consistent dims; MoE/custom archs may be null
        if md["hidden_size"] and md["attention_heads"]:
            assert md["head_dim"] * md["attention_heads"] == md["hidden_size"] or e["config"].get("head_dim")


def test_runnable_models_expose_full_dims():
    for e in catalog.CATALOG:
        if e["runnable"]:
            md = catalog.public_view(e)["metadata"]
            for k in ("layers", "hidden_size", "attention_heads", "kv_heads", "head_dim",
                      "mlp_intermediate", "vocab_size", "context_length"):
                assert md[k] is not None, f"{e['id']} missing {k}"


def test_base_model_is_labeled():
    base = catalog.by_id("qwen2.5-0.5b-base")
    assert base["tuning"].startswith("base")
    assert base["runnable"] is True


def test_catalog_payload_counts():
    p = catalog.catalog_payload()
    assert p["count"] == len(catalog.CATALOG)
    assert p["runnable_count"] == sum(1 for e in catalog.CATALOG if e["runnable"])
