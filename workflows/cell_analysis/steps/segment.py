"""Segment -- Cellpose instance segmentation.

Runs the Cellpose v4 (CPSAM) model on the preprocessed image. The model
is loaded once on the first call and kept warm in ``state["model"]``
for subsequent calls on the same worker, so per-image latency stays
low after a short warmup.

Inputs (from previous step)
    pipeline_data["preprocess"]["image_preprocessed"] : ndarray
        The preprocessed image to segment.

Parameters (via YAML / **params)
    diameter : float or None, default None
        Approximate cell diameter in pixels. None lets Cellpose
        auto-estimate.
    gpu : bool, default False
        Use the GPU if available.

Outputs (under pipeline_data["segment"])
    masks : int32 ndarray
        Label-image: 0 is background, each integer is one cell.
    n_cells : int
        Number of cells detected.
    diameter : the value used (echoed for downstream reference).
"""

METADATA = {
    "description": "Cellpose v4 (CPSAM) instance segmentation",
    "version": "1.0",
    "max_workers": 1,
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    from cellpose import models

    verbose = pipeline_data["metadata"].get("verbose", 0)
    diameter = params.get("diameter", None)
    gpu = params.get("gpu", False)

    img_pre = pipeline_data["preprocess"]["image_preprocessed"]

    # Warm-load the model on first call; reuse it on every subsequent call.
    if "model" not in state:
        if verbose >= 2:
            print(f"  [segment] cold start: loading CellposeModel(gpu={gpu})")
        state["model"] = models.CellposeModel(gpu=gpu)

    masks, flows, styles = state["model"].eval(img_pre, diameter=diameter)
    n_cells = int(masks.max())

    if verbose >= 2:
        print(f"  [segment] cells: {n_cells}")

    pipeline_data["segment"] = {
        "masks": masks,
        "n_cells": n_cells,
        "diameter": diameter,
    }
    return pipeline_data
