"""Shared TIFF loading and Cellpose segmentation helpers."""

from __future__ import annotations

import tifffile


def segment_tiff(
    image_path,
    state: dict,
    *,
    channel: int = 0,
    diameter=None,
    gpu: bool = False,
    verbose: int = 0,
    log_prefix: str = "segment",
) -> dict:
    """Load a TIFF, extract a 2D plane, and run a warm Cellpose model."""
    image = tifffile.imread(image_path)
    image_2d = ensure_2d(image, channel)
    ny, nx = image_2d.shape

    if "model" not in state:
        from cellpose import models

        if verbose >= 2:
            print(f"  [{log_prefix}] cold start: loading CellposeModel(gpu={gpu})")
        state["model"] = models.CellposeModel(gpu=gpu)

    masks, flows, styles = state["model"].eval(image_2d, diameter=diameter)
    n_objects = int(masks.max())

    if verbose >= 1:
        print(f"  [{log_prefix}] image={nx}x{ny}, objects={n_objects}")

    return {
        "image_2d": image_2d,
        "masks": masks,
        "n_objects": n_objects,
        "image_size_px": [int(nx), int(ny)],
    }


def ensure_2d(image, channel: int):
    """Return one 2D plane from a 2D, channel-first, or channel-last image."""
    if channel < 0:
        raise ValueError(f"channel must be >= 0, got {channel}.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[0] <= 4:
        n_ch = image.shape[0]
        if channel >= n_ch:
            raise ValueError(
                f"channel={channel} but image has {n_ch} channels "
                f"(shape {image.shape}, C-first)."
            )
        return image[channel]
    if image.ndim == 3 and image.shape[2] <= 4:
        n_ch = image.shape[2]
        if channel >= n_ch:
            raise ValueError(
                f"channel={channel} but image has {n_ch} channels "
                f"(shape {image.shape}, C-last)."
            )
        return image[:, :, channel]
    raise ValueError(
        f"Cannot extract 2D from image with shape {image.shape}. "
        f"Expected 2D (H, W) or 3D (C, H, W) / (H, W, C) with C <= 4."
    )
