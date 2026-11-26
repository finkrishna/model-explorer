"""Minimal read-only safetensors reader (no `safetensors` package required).

Format: 8-byte little-endian header length, a JSON header mapping tensor names to
{dtype, shape, data_offsets}, then the raw little-endian tensor bytes. The file is
memory-mapped read-only; nothing is ever written back.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import torch

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = self.path.open("rb")
        (header_len,) = struct.unpack("<Q", self._fh.read(8))
        header = json.loads(self._fh.read(header_len))
        self.metadata = header.pop("__metadata__", {}) or {}
        self.header = header
        self._data_start = 8 + header_len
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return sorted(self.header)

    def info(self, name: str) -> dict:
        entry = self.header[name]
        return {"dtype": entry["dtype"], "shape": list(entry["shape"]),
                "nbytes": entry["data_offsets"][1] - entry["data_offsets"][0]}

    def tensor(self, name: str) -> torch.Tensor:
        """Return a CPU tensor copy of the stored values (stored dtype)."""
        entry = self.header[name]
        start, end = entry["data_offsets"]
        dtype = _DTYPES[entry["dtype"]]
        raw = bytearray(self._mm[self._data_start + start:self._data_start + end])
        t = torch.frombuffer(raw, dtype=dtype) if raw else torch.empty(0, dtype=dtype)
        return t.reshape(entry["shape"])

    def close(self):
        self._mm.close()
        self._fh.close()
