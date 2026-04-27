"""Extract Features -- measure cell properties via skimage regionprops_table.

Wraps ``skimage.measure.regionprops_table`` with the original (non-
preprocessed) image as ``intensity_image`` so intensity-based properties
reflect the raw pixel values.

Inputs (from previous steps)
    pipeline_data["segment"]["masks"] : label image
    pipeline_data["preprocess"]["image"] : original image (intensity)

Parameters (via YAML / **params)
    properties : list of str
        Which regionprops to measure. Defaults to a sensible set
        covering shape and intensity. See the skimage docs for the
        full list of available properties.

Outputs (under pipeline_data["extract_features"])
    properties : dict[str, ndarray]
        The regionprops_table output. Keys are property names (with
        per-axis suffixes, e.g. centroid-0 / centroid-1), values are
        per-cell arrays aligned with the "label" key.
    n_cells : int
        Number of measured cells.
"""

METADATA = {
    "description": "Region properties on segmented masks",
    "version": "1.0",
}


DEFAULT_PROPERTIES = [
    "label",
    "area",
    "centroid",
    "eccentricity",
    "mean_intensity",
    "max_intensity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
]


def run(pipeline_data: dict, state: dict, **params) -> dict:
    from skimage.measure import regionprops_table

    verbose = pipeline_data["metadata"].get("verbose", 0)
    properties = params.get("properties", DEFAULT_PROPERTIES)

    masks = pipeline_data["segment"]["masks"]
    img = pipeline_data["preprocess"]["image"]

    props = regionprops_table(
        masks, intensity_image=img, properties=properties
    )

    if "label" in props:
        n_cells = len(props["label"])
    else:
        # Fall back to whichever key is present.
        n_cells = len(next(iter(props.values()))) if props else 0

    if verbose >= 2:
        print(f"  [extract_features] cells: {n_cells}, "
              f"properties: {sorted(props)}")

    pipeline_data["extract_features"] = {
        "properties": props,
        "n_cells": n_cells,
    }
    return pipeline_data
