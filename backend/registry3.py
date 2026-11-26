"""Provenance gate for the runnable models.

A model is only ever loaded after its local snapshot is checked, byte for byte,
against the official Hugging Face metadata pinned in ``catalog.py``:

  * the snapshot directory for the pinned revision exists;
  * each small file's git blob id matches (LFS-stored small files: SHA-256);
  * model.safetensors -- or every shard of a sharded checkpoint -- has the exact
    official byte size and SHA-256, and the shard sizes sum to the pinned total.

A mismatch raises ProvenanceError and the model is not loaded -- nothing is
substituted and no other checkpoint is silently used in its place. Pins are sourced
from the shared catalog so there is one source of truth.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from catalog import RUNNABLE_SLOTS, by_id

HUB = Path.home() / ".cache/huggingface/hub"


class ProvenanceError(RuntimeError):
    pass


def spec_for_slot(slot: str) -> dict:
    if slot not in RUNNABLE_SLOTS:
        raise KeyError(f"no runnable model in slot {slot!r}")
    return by_id(RUNNABLE_SLOTS[slot])


def snapshot_dir(entry: dict) -> Path:
    rev = entry["pin"]["revision"]
    return HUB / ("models--" + entry["repo"].replace("/", "--")) / "snapshots" / rev


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


_VERIFIED: dict = {}  # (resolved path, size, mtime_ns, inode) -> sha info; per process, in memory


def verify(entry: dict) -> dict:
    """Check the local snapshot against the pinned official metadata. Raises on mismatch."""
    if not entry.get("runnable") or "pin" not in entry:
        raise ProvenanceError(f"{entry['repo']}: not a runnable, pinned model")
    pin = entry["pin"]
    snap = snapshot_dir(entry)
    if snap.name != pin["revision"] or not snap.is_dir():
        raise ProvenanceError(f"{entry['repo']}: snapshot for pinned revision {pin['revision']} is missing")
    blob_ids = {}
    for name, want in pin["blobs"].items():
        p = snap / name
        if not p.is_file():
            raise ProvenanceError(f"{entry['repo']}: {name} missing from the pinned snapshot")
        got = _git_blob_sha1(p)
        if got != want:
            raise ProvenanceError(f"{entry['repo']}: {name} blob id {got} != official {want}")
        blob_ids[name] = got
    # small files stored via Git LFS (their git blob id is the pointer, so check SHA-256)
    for name, want in pin.get("lfs_small", {}).items():
        p = snap / name
        if not p.is_file():
            raise ProvenanceError(f"{entry['repo']}: {name} missing from the pinned snapshot")
        got, _, _ = _hashed(p)
        if got != want:
            raise ProvenanceError(f"{entry['repo']}: {name} sha256 {got} != official {want}")
        blob_ids[name] = "sha256:" + got
    # weights: a single model.safetensors or every shard of a sharded checkpoint
    shards = pin.get("shards") or {"model.safetensors": {"sha256": pin["weights_sha256"],
                                                         "bytes": pin["weights_bytes"]}}
    first = next(iter(shards))
    if shards[first]["sha256"] != pin["weights_sha256"]:
        raise ProvenanceError(f"{entry['repo']}: pin is inconsistent (weights_sha256 is not shard {first})")
    total, digests, secs, cached = 0, {}, 0.0, True
    for fname, spec in shards.items():
        w = snap / fname
        if not w.is_file():
            raise ProvenanceError(f"{entry['repo']}: {fname} missing from the pinned snapshot")
        size = w.stat().st_size
        if size != spec["bytes"]:
            raise ProvenanceError(f"{entry['repo']}: {fname} is {size} B, official {spec['bytes']} B")
        digest, s, was_cached = _hashed(w)
        if digest != spec["sha256"]:
            raise ProvenanceError(f"{entry['repo']}: {fname} sha256 {digest} != official {spec['sha256']}")
        total += size; digests[fname] = digest; secs += s; cached = cached and was_cached
    if total != pin["weights_bytes"]:
        raise ProvenanceError(f"{entry['repo']}: weights total {total} B, official {pin['weights_bytes']} B")
    return {"ok": True, "repo": entry["repo"], "revision": pin["revision"],
            "label": entry["label"], "license": entry["license"],
            "weights_sha256": digests[first], "weights_bytes": total,
            "weight_files": digests,
            "hash_seconds": round(secs, 2), "hash_cached_in_process": cached,
            "small_file_blob_ids": blob_ids}


def _hashed(path: Path):
    """SHA-256 of a file, cached per process by (path, size, mtime, inode)."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, st.st_ino)
    if key in _VERIFIED:
        return _VERIFIED[key]["sha256"], 0.0, True
    t0 = time.monotonic()
    _VERIFIED[key] = {"sha256": _sha256(path), "hash_seconds": round(time.monotonic() - t0, 2)}
    return _VERIFIED[key]["sha256"], _VERIFIED[key]["hash_seconds"], False


def available_slots() -> dict:
    """Which runnable slots are actually present & verifiable on this machine, with reasons."""
    out = {}
    for slot in sorted(RUNNABLE_SLOTS):
        entry = spec_for_slot(slot)
        try:
            prov = verify(entry)
            out[slot] = {"id": entry["id"], "label": entry["label"], "ready": True,
                         "revision": prov["revision"], "weights_sha256": prov["weights_sha256"]}
        except ProvenanceError as e:
            out[slot] = {"id": entry["id"], "label": entry["label"], "ready": False, "reason": str(e)}
    return out
