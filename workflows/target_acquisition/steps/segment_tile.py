"""segment_tile -- cellpose segmentation for target acquisition overview tiles.

Loads the tile image (acquired or mock), runs cellpose, returns masks.
The image source is controlled by ``analysis_image_source`` in the
submission payload: ``"acquired"`` loads the TIFF, ``"skimage_human_mitosis"``
uses the scikit-image sample image for repeatable testing.

Inputs (from submission payload)
    pipeline_data["input"]["image_path"] : str
        Path to the acquired TIFF (used when source is "acquired").
    pipeline_data["input"]["analysis_image_source"] : str
        "acquired" or "skimage_human_mitosis".

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
    image_source : str
        Which source was used.
"""

METADATA = {
    "max_workers": 1,
    "environment": "dino3_test",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    verbose = pipeline_data.get("metadata", {}).get("verbose", 0)
    diameter = params.get("diameter", None)
    channel = params.get("channel", 0)
    gpu = params.get("gpu", False)

    inp = pipeline_data["input"]
    source = inp.get("analysis_image_source", "acquired")

    image = _load_image(source, inp)
    image_2d = _ensure_2d(image, channel)
    ny, nx = image_2d.shape

    from cellpose import models

    if "model" not in state:
        if verbose >= 2:
            print(f"  [segment_tile] cold start: loading CellposeModel(gpu={gpu})")
        state["model"] = models.CellposeModel(gpu=gpu)

    masks, flows, styles = state["model"].eval(image_2d, diameter=diameter)
    n_cells = int(masks.max())

    print(f"  [segment_tile] source={source}, image={nx}x{ny}, cells={n_cells}")

    pipeline_data["segment_tile"] = {
        "image_2d": image_2d,
        "masks": masks,
        "n_cells": n_cells,
        "image_size_px": (nx, ny),
        "image_source": source,
    }
    return pipeline_data


def _load_image(source: str, inp: dict):
    import tifffile

    if source == "acquired":
        image_path = inp["image_path"]
        return tifffile.imread(image_path)

    if source == "skimage_human_mitosis":
        from skimage.data import human_mitosis
        return human_mitosis()

    raise ValueError(
        f"Unknown analysis_image_source: {source!r}. "
        f"Expected 'acquired' or 'skimage_human_mitosis'.")


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
