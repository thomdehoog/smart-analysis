"""discover_targets -- select stage targets from object detection tables."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_overview, validate_targets  # noqa: E402
from _geometry import image_point_to_stage_xy  # noqa: E402


METADATA = {
    "description": "Select revisit targets from object detection tables",
    "version": "1.0",
    "max_workers": 1,
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    """Engine entry point for target discovery.

    Input is an aggregated overview under ``pipeline_data["input"]["tiles"]``.
    The step never reads masks or recomputes region properties; it ranks
    directly from row-aligned object feature columns.
    """
    inp = pipeline_data["input"]
    overview = validate_overview({"tiles": inp["tiles"]})

    feature = _setting(inp, params, "feature", "area")
    direction = _setting(inp, params, "direction", "high")
    n_per_tile = _setting(inp, params, "n_per_tile", 5)
    border_margin_px = float(_setting(inp, params, "border_margin_px", 0.0) or 0.0)
    area_threshold = _setting(inp, params, "area_threshold", None)
    intensity_threshold = _setting(inp, params, "intensity_threshold", None)

    if direction not in {"high", "low"}:
        raise ValueError("direction must be 'high' or 'low'.")
    if n_per_tile is not None:
        n_per_tile = int(n_per_tile)
        if n_per_tile < 0:
            raise ValueError("n_per_tile must be >= 0 or None.")

    targets = []
    for tile in overview["tiles"]:
        tile_targets = _targets_for_tile(
            tile,
            feature=feature,
            direction=direction,
            n_per_tile=n_per_tile,
            border_margin_px=border_margin_px,
            area_threshold=area_threshold,
            intensity_threshold=intensity_threshold,
        )
        targets.extend(tile_targets)

    result = validate_targets({"targets": targets})
    pipeline_data["target_discovery"] = result
    pipeline_data["discover_targets"] = result
    return pipeline_data


def _setting(inp: dict, params: dict, key: str, default):
    return inp[key] if key in inp else params.get(key, default)


def _targets_for_tile(
    tile: dict,
    *,
    feature: str,
    direction: str,
    n_per_tile: int | None,
    border_margin_px: float,
    area_threshold,
    intensity_threshold,
) -> list[dict]:
    objects = tile["objects"]
    props = objects["properties"]
    geometry = tile["geometry"]

    if feature not in props:
        raise ValueError(f"Object table is missing feature column {feature!r}.")

    candidates = []
    n_objects = objects["n_objects"]
    for row in range(n_objects):
        if not _passes_thresholds(props, row, area_threshold, intensity_threshold):
            continue
        if not _inside_border(props, geometry, row, border_margin_px):
            continue

        score = _as_number(props[feature][row])
        if score is None:
            continue

        stage_x_um, stage_y_um = image_point_to_stage_xy(
            centroid_row_px=props["centroid_row_px"][row],
            centroid_col_px=props["centroid_col_px"][row],
            image_size_px=geometry["source_image_size_px"],
            pixel_size_um=geometry["source_pixel_size_um"],
            tile_stage_xy_um=geometry["tile_stage_xy_um"],
            image_to_stage=geometry["image_to_stage"],
        )

        label = int(props["label"][row])
        tile_id = list(geometry["tile_id"])
        candidates.append({
            "target_id": tile_id + [label],
            "tile_id": tile_id,
            "object_label": label,
            "score": float(score),
            "source_feature": feature,
            "centroid_row_px": float(props["centroid_row_px"][row]),
            "centroid_col_px": float(props["centroid_col_px"][row]),
            "bbox_min_row_px": int(props["bbox_min_row_px"][row]),
            "bbox_min_col_px": int(props["bbox_min_col_px"][row]),
            "bbox_max_row_px": int(props["bbox_max_row_px"][row]),
            "bbox_max_col_px": int(props["bbox_max_col_px"][row]),
            "stage_x_um": float(stage_x_um),
            "stage_y_um": float(stage_y_um),
        })

    reverse = direction == "high"
    candidates.sort(
        key=lambda target: (
            -target["score"] if reverse else target["score"],
            target["object_label"],
        )
    )
    if n_per_tile is None:
        return candidates
    return candidates[:n_per_tile]


def _passes_thresholds(
    props: dict, row: int, area_threshold, intensity_threshold
) -> bool:
    if area_threshold is not None:
        area = _as_number(props["area"][row])
        if area is None or area < float(area_threshold):
            return False
    if intensity_threshold is not None:
        intensity = _as_number(props["intensity_mean"][row])
        if intensity is None or intensity < float(intensity_threshold):
            return False
    return True


def _inside_border(
    props: dict, geometry: dict, row: int, border_margin_px: float
) -> bool:
    if border_margin_px <= 0:
        return True

    nx, ny = geometry["source_image_size_px"]
    row_px = float(props["centroid_row_px"][row])
    col_px = float(props["centroid_col_px"][row])
    return (
        border_margin_px <= row_px <= float(ny) - border_margin_px
        and border_margin_px <= col_px <= float(nx) - border_margin_px
    )


def _as_number(value):
    if value is None:
        return None
    return float(value)
