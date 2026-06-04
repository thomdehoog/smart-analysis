from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_targets, validate_tile_detection  # noqa: E402


_STEP_PATH = Path(__file__).resolve().parents[1] / "steps" / "select_targets.py"
_spec = importlib.util.spec_from_file_location("select_targets", _STEP_PATH)
select_targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_targets)


def make_tile(
    tile_id=("R0", 0, 0),
    *,
    tile_stage_xy_um=(1000.0, 2000.0),
    image_size=(100, 100),
    image_to_stage=None,
):
    if image_to_stage is None:
        image_to_stage = [[1.0, 0.0], [0.0, 1.0]]
    return validate_tile_detection({
        "objects": {
            "properties": {
                "label": np.array([1, 2, 3]),
                "centroid_row_px": np.array([50.0, 20.0, 90.0]),
                "centroid_col_px": np.array([50.0, 80.0, 5.0]),
                "bbox_min_row_px": np.array([45, 18, 88]),
                "bbox_min_col_px": np.array([45, 78, 3]),
                "bbox_max_row_px": np.array([55, 23, 95]),
                "bbox_max_col_px": np.array([55, 85, 8]),
                "area": np.array([60.0, 100.0, 20.0]),
                "intensity_mean": np.array([10.0, 40.0, 80.0]),
                "eccentricity": np.array([0.3, 0.1, 0.9]),
                "stage_x_um": np.array([1000.0, 1021.0, 990.0]),
                "stage_y_um": np.array([2000.0, 2015.0, 1980.0]),
            },
            "n_objects": np.int64(3),
        },
        "geometry": {
            "tile_id": tile_id,
            "tile_stage_xy_um": tile_stage_xy_um,
            "tile_zwide_um": 0.0,
            "source_pixel_size_um": (0.5, 0.7),
            "source_image_size_px": image_size,
            "image_to_stage": image_to_stage,
        },
    })


def run_discovery(tiles, **input_overrides):
    pipeline_data = {
        "input": {"tiles": tiles, **input_overrides},
        "metadata": {"verbose": 0},
    }
    return select_targets.run(pipeline_data, {})["target_discovery"]["targets"]


def test_ranks_by_feature_from_table_per_tile():
    tiles = [make_tile(("R0", 0, 0)), make_tile(("R0", 0, 1))]

    targets = run_discovery(tiles, feature="area", direction="high", n_per_tile=2)

    assert [t["object_label"] for t in targets] == [2, 1, 2, 1]
    assert [t["score"] for t in targets] == [100.0, 60.0, 100.0, 60.0]


def test_low_direction_uses_smallest_scores():
    targets = run_discovery([make_tile()], feature="area", direction="low", n_per_tile=2)

    assert [t["object_label"] for t in targets] == [3, 1]


def test_explicit_thresholds_filter_before_ranking():
    targets = run_discovery(
        [make_tile()],
        feature="intensity_mean",
        direction="high",
        n_per_tile=None,
        area_threshold=50,
        intensity_threshold=20,
    )

    assert [t["object_label"] for t in targets] == [2]


def test_border_margin_uses_each_tile_image_size():
    tile = make_tile()
    # Label 1 has a safe centroid but a wide bbox crossing the left
    # margin. Border filtering must use bbox bounds, not centroid alone.
    tile["objects"]["properties"]["bbox_min_col_px"][0] = 5

    targets = run_discovery(
        [tile],
        feature="area",
        direction="high",
        n_per_tile=None,
        border_margin_px=10,
    )

    assert [t["object_label"] for t in targets] == [2]


def test_stage_coordinates_come_from_object_table():
    tile = make_tile(
        tile_stage_xy_um=(1000.0, 2000.0),
        image_to_stage=[[0.0, -1.0], [1.0, 0.0]],
    )
    tile["objects"]["properties"]["stage_x_um"][1] = 1234.5
    tile["objects"]["properties"]["stage_y_um"][1] = 2345.5

    target = run_discovery([tile], feature="area", n_per_tile=1)[0]

    assert target["object_label"] == 2
    assert target["stage_x_um"] == pytest.approx(1234.5)
    assert target["stage_y_um"] == pytest.approx(2345.5)


def test_output_contains_no_numpy_scalars():
    targets = run_discovery([make_tile()], feature="area", n_per_tile=1)
    validated = validate_targets({"targets": targets})

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        else:
            yield value

    assert not any(type(value).__module__ == "numpy" for value in walk(validated))


def test_n_per_tile_none_returns_all_valid_targets():
    targets = run_discovery([make_tile()], feature="area", n_per_tile=None)

    assert [t["object_label"] for t in targets] == [2, 1, 3]


def test_missing_feature_column_fails_clearly():
    tile = make_tile()

    with pytest.raises(ValueError, match="missing feature column 'not_a_feature'"):
        run_discovery([tile], feature="not_a_feature", n_per_tile=1)


def test_invalid_direction_fails_clearly():
    with pytest.raises(ValueError, match="direction"):
        run_discovery([make_tile()], direction="sideways")
