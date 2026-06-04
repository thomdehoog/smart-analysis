"""detect_objects -- Cellpose object detection for one image tile."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _segmentation import segment_tiff  # noqa: E402


METADATA = {
    "description": "Detect objects in a TIFF tile with Cellpose",
    "version": "1.0",
    "max_workers": 1,
    "environment": "SMART--object_analysis--vision",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    verbose = pipeline_data.get("metadata", {}).get("verbose", 0)
    inp = pipeline_data["input"]
    detection = segment_tiff(
        inp["image_path"],
        state,
        channels=inp.get("channels", params.get("channels", None)),
        gpu=inp.get("gpu", params.get("gpu", False)),
        verbose=verbose,
        log_prefix="detect_objects",
    )

    pipeline_data["detect_objects"] = detection
    # Bridge to the shared classical feature extractor's current input contract.
    pipeline_data["preprocess"] = {"image": detection["image"]}
    pipeline_data["segment"] = {
        "masks": detection["masks"],
        "n_cells": detection["n_objects"],
    }
    return pipeline_data
