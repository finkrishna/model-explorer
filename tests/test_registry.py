"""Provenance gate: real verification + negative controls.

These need the three pinned checkpoints in the local HF cache. If a checkpoint is
absent the positive check is skipped, but the negative controls (which must reject
tampered metadata) always run.
"""
import copy

import catalog
import pytest
import registry3


def _entry(mid):
    return copy.deepcopy(catalog.by_id(mid))


RUNNABLE_IDS = [e["id"] for e in catalog.CATALOG if e["runnable"]]


def _present(entry):
    """True only if every pinned weight file is fully on disk (not mid-download)."""
    snap = registry3.snapshot_dir(entry)
    pin = entry["pin"]
    shards = pin.get("shards") or {"model.safetensors": {"bytes": pin["weights_bytes"]}}
    return snap.is_dir() and all((snap / f).is_file() and (snap / f).stat().st_size == s["bytes"]
                                 for f, s in shards.items())


@pytest.mark.parametrize("mid", RUNNABLE_IDS)
def test_runnable_verifies_when_cached(mid):
    entry = _entry(mid)
    if not _present(entry):
        pytest.skip(f"{mid} not in local cache")
    prov = registry3.verify(entry)
    assert prov["ok"]
    assert prov["weights_sha256"] == entry["pin"]["weights_sha256"]
    assert prov["weights_bytes"] == entry["pin"]["weights_bytes"]


@pytest.mark.parametrize("mid", RUNNABLE_IDS)
def test_tampered_sha256_is_rejected(mid):
    entry = _entry(mid)
    if not _present(entry):
        pytest.skip(f"{mid} not in local cache")
    entry["pin"]["weights_sha256"] = "0" * 64
    with pytest.raises(registry3.ProvenanceError):
        registry3.verify(entry)


@pytest.mark.parametrize("mid", RUNNABLE_IDS)
def test_tampered_blob_id_is_rejected(mid):
    entry = _entry(mid)
    if not _present(entry):
        pytest.skip(f"{mid} not in local cache")
    entry["pin"]["blobs"]["config.json"] = "deadbeef" * 5
    with pytest.raises(registry3.ProvenanceError):
        registry3.verify(entry)


@pytest.mark.parametrize("mid", RUNNABLE_IDS)
def test_wrong_byte_size_is_rejected(mid):
    entry = _entry(mid)
    if not _present(entry):
        pytest.skip(f"{mid} not in local cache")
    entry["pin"]["weights_bytes"] = entry["pin"]["weights_bytes"] + 1
    with pytest.raises(registry3.ProvenanceError):
        registry3.verify(entry)


def test_missing_revision_is_rejected():
    entry = _entry("qwen2.5-0.5b-instruct")
    entry["pin"]["revision"] = "f" * 40  # a revision that is not in the cache
    with pytest.raises(registry3.ProvenanceError):
        registry3.verify(entry)


def test_non_runnable_entry_cannot_be_verified():
    entry = _entry("inkling-2-mini")  # reference-only, has no "pin"
    with pytest.raises(registry3.ProvenanceError):
        registry3.verify(entry)


def test_unknown_slot_raises():
    with pytest.raises(KeyError):
        registry3.spec_for_slot("z")
