"""package_objects -- publish detection output under the public contract."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import to_builtin, validate_tile_detection  # noqa: E402


METADATA = {
    "description": "Package object measurements into the public detection contract",
    "version": "1.0",
    "max_workers": 1,
}


FEATURE_COLUMN_MAP = {
    "label": "label",
    "centroid-0": "centroid_row_px",
    "centroid-1": "centroid_col_px",
    "bbox-0": "bbox_min_row_px",
    "bbox-1": "bbox_min_col_px",
    "bbox-2": "bbox_max_row_px",
    "bbox-3": "bbox_max_col_px",
    "area": "area",
    "intensity_mean": "intensity_mean",
    "eccentricity": "eccentricity",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    feature_output = pipeline_data["extract_features"]
    props = feature_output["properties"]
    n_objects = int(feature_output["n_cells"])

    public_props = {}
    for source, public in FEATURE_COLUMN_MAP.items():
        if source not in props:
            raise ValueError(
                f"extract_features output missing required column {source!r} "
                f"for public column {public!r}."
            )
        public_props[public] = to_builtin(props[source])

    mapped_sources = set(FEATURE_COLUMN_MAP)
    for name, values in props.items():
        if name not in mapped_sources and name not in public_props:
            public_props[name] = to_builtin(values)

    geometry = _geometry_from_input(
        pipeline_data["input"],
        pipeline_data.get("detect_objects", {}),
    )
    tile_detection = validate_tile_detection({
        "objects": {
            "properties": public_props,
            "n_objects": n_objects,
        },
        "geometry": geometry,
    })

    pipeline_data["object_detection"] = tile_detection
    return pipeline_data


def _geometry_from_input(inp: dict, detection: dict) -> dict:
    required = [
        "tile_id",
        "tile_stage_xy_um",
        "tile_zwide_um",
        "source_pixel_size_um",
        "image_to_stage",
    ]
    for name in required:
        if name not in inp:
            raise ValueError(f"input missing required geometry field {name!r}.")

    return {
        "tile_id": inp["tile_id"],
        "tile_stage_xy_um": inp["tile_stage_xy_um"],
        "tile_zwide_um": inp["tile_zwide_um"],
        "source_pixel_size_um": inp["source_pixel_size_um"],
        "source_image_size_px": inp.get(
            "source_image_size_px", detection.get("image_size_px")
        ),
        "image_to_stage": inp["image_to_stage"],
    }
