"""
Segment — Run Cellpose v4 (CPSAM) segmentation.

Reads preprocessed image from pipeline_data, returns label masks.
"""

METADATA = {
    "description": "Cellpose v4 CPSAM segmentation",
    "version": "1.0",
    "max_workers": 1,
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    from cellpose import models

    verbose = pipeline_data["metadata"].get("verbose", 0)
    diameter = params.get("diameter", None)
    gpu = params.get("gpu", False)

    img_pre = pipeline_data["preprocess"]["image_preprocessed"]

    # Warm model: load once, reuse across calls
    if "model" not in state:
        state["model"] = models.CellposeModel(gpu=gpu)
    masks, flows, styles = state["model"].eval(img_pre, diameter=diameter)

    n_cells = int(masks.max())

    if verbose >= 2:
        print(f"  [segment] Cells found: {n_cells}")
        print(f"  [segment] diameter={diameter}, gpu={gpu}")

    pipeline_data["segment"] = {
        "masks": masks,
        "n_cells": n_cells,
        "diameter": diameter,
    }

    return pipeline_data
