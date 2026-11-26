#!/usr/bin/env python3
"""Generate the static data files the catalog/detail pages fall back to.

Writes ``ui/catalog.json`` and ``ui/explanations.json`` from the Python sources of
truth (catalog.py, explanations.py). With these present, the catalog and per-model
architecture pages work with no backend and no network -- exactly what a GitHub Pages
host serves. The live prompt comparison still needs the local backend; the static
pages never attempt it.

Run:  python3 backend/build_site.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
UI = HERE.parent / "ui"

import catalog        # noqa: E402
import explanations   # noqa: E402


def main():
    UI.mkdir(exist_ok=True)
    (UI / "catalog.json").write_text(
        json.dumps(catalog.catalog_payload(), ensure_ascii=False, indent=1))
    (UI / "explanations.json").write_text(
        json.dumps(explanations.metrics_payload(), ensure_ascii=False, indent=1))
    print("wrote", UI / "catalog.json")
    print("wrote", UI / "explanations.json")


if __name__ == "__main__":
    main()
