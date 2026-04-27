"""Preprocess -- load image and apply standard preprocessing.

Loads from a path on disk or from a named skimage sample dataset.
Applies Gaussian smoothing and CLAHE contrast normalisation.

Inputs (via pipeline_data["input"])
    data_source : str
        Either a path to an image file (anything skimage.io.imread reads)
        or "skimage.<dataset>" e.g. "skimage.human_mitosis".

Parameters (via YAML / **params)
    sigma : float, default 1.0
        Gaussian blur sigma applied before CLAHE.
    clip_limit : float, default 0.03
        Contrast clipping limit for skimage.exposure.equalize_adapthist.

Outputs (under pipeline_data["preprocess"])
    image : ndarray
        The original (un-preprocessed) image, kept for downstream
        intensity measurements.
    image_preprocessed : uint8 ndarray
        The smoothed + CLAHE-normalised image, ready for segmentation.
    shape, sigma, clip_limit, data_source : metadata about this run.
"""

METADATA = {
    "description": "Load and preprocess image",
    "version": "1.0",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    import numpy as np
    from skimage.filters import gaussian
    from skimage.exposure import equalize_adapthist

    verbose = pipeline_data["metadata"].get("verbose", 0)
    sigma = params.get("sigma", 1.0)
    clip_limit = params.get("clip_limit", 0.03)
    data_source = pipeline_data["input"].get(
        "data_source", "skimage.human_mitosis"
    )

    if data_source.startswith("skimage."):
        from skimage import data as skdata
        dataset = data_source.split(".", 1)[1]
        loader = getattr(skdata, dataset, None)
        if loader is None:
            raise ValueError(
                f"Unknown skimage dataset: {dataset}"
            )
        img = loader()
    else:
        from skimage.io import imread
        img = imread(data_source)

    img_smooth = gaussian(img, sigma=sigma)
    img_pre = equalize_adapthist(img_smooth, clip_limit=clip_limit)
    img_pre = (img_pre * 255).astype(np.uint8)

    if verbose >= 2:
        print(f"  [preprocess] {img.shape} from {data_source}")

    pipeline_data["preprocess"] = {
        "image": img,
        "image_preprocessed": img_pre,
        "shape": img.shape,
        "sigma": sigma,
        "clip_limit": clip_limit,
        "data_source": data_source,
    }
    return pipeline_data
