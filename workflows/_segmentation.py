"""Shared TIFF loading and Cellpose segmentation helpers."""

from __future__ import annotations

import numpy as np
import tifffile


def segment_tiff(
    image_path,
    state: dict,
    *,
    channels=None,
    gpu: bool = False,
    verbose: int = 0,
    log_prefix: str = "segment",
) -> dict:
    """Load a TIFF, select up to three channels, and run a warm Cellpose model.

    Cellpose runs on the selected channels (2D or up to 3-channel). ``image``
    preserves those selected channels for downstream feature extraction, while
    ``image_2d`` is the primary channel for code paths that require a plane.
    """
    image = tifffile.imread(image_path)
    seg_input = select_channels(image, channels)
    ny, nx = seg_input.shape[:2]

    if "model" not in state:
        from cellpose import models

        if verbose >= 2:
            print(f"  [{log_prefix}] cold start: loading CellposeModel(gpu={gpu})")
        state["model"] = models.CellposeModel(gpu=gpu)

    channel_axis = -1 if seg_input.ndim == 3 else None
    masks, flows, styles = state["model"].eval(seg_input, channel_axis=channel_axis)
    n_objects = int(masks.max())

    image_2d = seg_input if seg_input.ndim == 2 else seg_input[..., 0]

    if verbose >= 1:
        print(f"  [{log_prefix}] image={nx}x{ny}, objects={n_objects}")

    return {
        "image": seg_input,
        "image_2d": image_2d,
        "masks": masks,
        "n_objects": n_objects,
        "image_size_px": [int(nx), int(ny)],
    }


def select_channels(image, channels=None):
    """Return up to three channels for Cellpose as ``(H, W)`` or ``(H, W, k)``.

    ``channels`` chooses which channels to keep; ``None`` uses the first up to
    three. Channel-first ``(C, H, W)`` input is normalized to channel-last, and a
    single selected channel is returned as a 2D plane.
    """
    if image.ndim == 2:
        if channels is not None:
            indices = _channel_indices(channels)
            if indices not in ([], [0]):
                raise ValueError("channels for a 2D image must be None or [0].")
        return image
    if image.ndim != 3:
        raise ValueError(
            f"Cannot select channels from image with shape {image.shape}. "
            f"Expected 2D (H, W) or 2D plus channels: (C, H, W) / (H, W, C)."
        )

    # The channel axis is the smaller end (channels are fewer than spatial
    # dims); channel-first (C, H, W) is normalized to channel-last (H, W, C).
    if image.shape[0] <= image.shape[-1]:
        stack = np.moveaxis(image, 0, -1)
    else:
        stack = image

    n_channels = stack.shape[-1]
    if channels is None:
        indices = list(range(min(n_channels, 3)))
    else:
        indices = _channel_indices(channels)
        if not indices:
            raise ValueError("channels must contain at least one channel.")
        if len(indices) > 3:
            raise ValueError("Cellpose accepts at most 3 channels.")
        if any(c < 0 or c >= n_channels for c in indices):
            raise ValueError(
                f"channels {indices} out of range for {n_channels} channels."
            )

    selected = stack[..., indices]
    if selected.shape[-1] == 1:
        return selected[..., 0]
    return selected


def _channel_indices(channels) -> list[int]:
    if isinstance(channels, (int, np.integer)):
        return [int(channels)]
    return [int(c) for c in channels]
