"""segment_tile -- cellpose segmentation for target acquisition overview tiles.

Loads the tile image from disk, runs cellpose, returns masks.

Inputs (from submission payload)
    pipeline_data["input"]["image_path"] : str
        Path to the acquired TIFF.

Parameters (via YAML / **params)
    diameter : float or None, default None
        Approximate cell diameter in pixels for cellpose.
    channel : int, default 0
        Channel to extract from multi-channel images.
    gpu : bool, default False
        Use GPU if available.

Outputs (under pipeline_data["segment_tile"])
    image_2d : ndarray
        The 2D image used for segmentation (single source of truth).
    masks : int32 ndarray
        Label image: 0 is background, each integer is one cell.
    n_cells : int
        Number of cells detected.
    image_size_px : tuple[int, int]
        (nx, ny) of the actual analysis image.
"""

import tifffile

METADATA = {
    "max_workers": 1,
    "environment": "lasxapi_extended",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    verbose = pipeline_data.get("metadata", {}).get("verbose", 0)
    diameter = params.get("diameter", None)
    channel = params.get("channel", 0)
    gpu = params.get("gpu", False)

    inp = pipeline_data["input"]
    image = tifffile.imread(inp["image_path"])
    image_2d = _ensure_2d(image, channel)
    ny, nx = image_2d.shape

    if "model" not in state:
        # Lazy import: cellpose pulls in torch (a heavy optional
        # dependency). Importing it only when we actually need to
        # construct a model lets tests stub state["model"] without
        # requiring a working torch install.
        from cellpose import models

        if verbose >= 2:
            print(f"  [segment_tile] cold start: loading CellposeModel(gpu={gpu})")
        state["model"] = models.CellposeModel(gpu=gpu)

    masks, flows, styles = state["model"].eval(image_2d, diameter=diameter)
    n_cells = int(masks.max())

    print(f"  [segment_tile] image={nx}x{ny}, cells={n_cells}")

    pipeline_data["segment_tile"] = {
        "image_2d": image_2d,
        "masks": masks,
        "n_cells": n_cells,
        "image_size_px": (nx, ny),
    }
    return pipeline_data


def _ensure_2d(image, channel: int):
    if channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[0] <= 4:
        n_ch = image.shape[0]
        if channel >= n_ch:
            raise ValueError(
                f"channel={channel} but image has {n_ch} channels "
                f"(shape {image.shape}, C-first).")
        return image[channel]
    if image.ndim == 3 and image.shape[2] <= 4:
        n_ch = image.shape[2]
        if channel >= n_ch:
            raise ValueError(
                f"channel={channel} but image has {n_ch} channels "
                f"(shape {image.shape}, C-last).")
        return image[:, :, channel]
    raise ValueError(
        f"Cannot extract 2D from image with shape {image.shape}. "
        f"Expected 2D (H, W) or 3D (C, H, W) / (H, W, C) with C <= 4.")
