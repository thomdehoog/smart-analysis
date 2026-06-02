from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _contracts import (  # noqa: E402
    REQUIRED_GEOMETRY_FIELDS,
    REQUIRED_OBJECT_COLUMNS,
    load_overview,
    save_overview,
    to_builtin,
    validate_overview,
    validate_targets,
    validate_tile_detection,
)


def make_tile(tile_id=("R0", 0, 0), labels=(1, 3, 5)):
    n = len(labels)
    return {
        "objects": {
            "properties": {
                "label": np.asarray(labels, dtype=np.int64),
                "centroid_row_px": np.asarray([10.0, 20.0, 30.0][:n]),
                "centroid_col_px": np.asarray([100.0, 200.0, 300.0][:n]),
                "bbox_min_row_px": np.asarray([8, 18, 28][:n]),
                "bbox_min_col_px": np.asarray([95, 195, 295][:n]),
                "bbox_max_row_px": np.asarray([13, 23, 33][:n]),
                "bbox_max_col_px": np.asarray([105, 205, 305][:n]),
                "area": np.asarray([50.0, 80.0, 30.0][:n]),
                "intensity_mean": np.asarray([5.0, 8.0, 3.0][:n]),
                "eccentricity": np.asarray([0.1, 0.2, 0.3][:n]),
                "optional_feature": np.asarray([np.nan, 1.5, 2.5][:n]),
            },
            "n_objects": np.int64(n),
        },
        "geometry": {
            "tile_id": tile_id,
            "tile_stage_xy_um": np.asarray([1000.0, 2000.0]),
            "tile_zwide_um": np.float64(250.0),
            "source_pixel_size_um": (0.5, 0.7),
            "source_image_size_px": (512, 512),
            "image_to_stage": [[1.0, 0.0], [0.0, 1.0]],
        },
    }


def test_validate_tile_detection_normalizes_numpy_and_tuples():
    tile = validate_tile_detection(make_tile())

    assert tile["objects"]["n_objects"] == 3
    assert tile["objects"]["properties"]["label"] == [1, 3, 5]
    assert tile["objects"]["properties"]["optional_feature"] == [None, 1.5, 2.5]
    assert tile["geometry"]["tile_id"] == ["R0", 0, 0]
    assert tile["geometry"]["source_pixel_size_um"] == [0.5, 0.7]


def test_empty_detection_is_valid_when_columns_are_empty():
    tile = make_tile(labels=())
    for values in tile["objects"]["properties"].values():
        values.resize(0, refcheck=False)
    tile["objects"]["n_objects"] = 0

    validated = validate_tile_detection(tile)

    assert validated["objects"]["n_objects"] == 0
    assert validated["objects"]["properties"]["label"] == []


def test_row_alignment_is_required_for_all_property_columns():
    tile = make_tile()
    tile["objects"]["properties"]["area"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="area.*length 2"):
        validate_tile_detection(tile)


@pytest.mark.parametrize("column", REQUIRED_OBJECT_COLUMNS)
def test_missing_required_object_column_fails_clearly(column):
    tile = make_tile()
    del tile["objects"]["properties"][column]

    with pytest.raises(ValueError, match=column):
        validate_tile_detection(tile)


@pytest.mark.parametrize("field", REQUIRED_GEOMETRY_FIELDS)
def test_missing_required_geometry_field_fails_clearly(field):
    tile = make_tile()
    del tile["geometry"][field]

    with pytest.raises(ValueError, match=field):
        validate_tile_detection(tile)


def test_validate_overview_round_trips_json(tmp_path):
    overview = validate_overview({"tiles": [make_tile(), make_tile(("R0", 0, 1))]})
    path = tmp_path / "overview.json"

    save_overview(path, overview)
    loaded = load_overview(path)

    assert loaded == overview
    assert loaded["tiles"][1]["geometry"]["tile_id"] == ["R0", 0, 1]


def test_validate_targets_requires_scalar_public_fields():
    result = validate_targets({
        "targets": [{
            "target_id": ("R0", 0, 0, 3),
            "tile_id": ("R0", 0, 0),
            "object_label": np.int64(3),
            "score": np.float64(80.0),
            "source_feature": "area",
            "centroid_row_px": np.float64(20.0),
            "centroid_col_px": np.float64(200.0),
            "bbox_min_row_px": np.int64(18),
            "bbox_min_col_px": np.int64(195),
            "bbox_max_row_px": np.int64(23),
            "bbox_max_col_px": np.int64(205),
            "stage_x_um": np.float64(900.0),
            "stage_y_um": np.float64(1900.0),
        }]
    })

    target = result["targets"][0]
    assert target["target_id"] == ["R0", 0, 0, 3]
    assert type(target["score"]) is float
    assert type(target["object_label"]) is int


def test_validate_targets_rejects_list_valued_public_scalars():
    with pytest.raises(ValueError, match="centroid_row_px.*scalar"):
        validate_targets({
            "targets": [{
                "target_id": ["R0", 0, 0, 3],
                "tile_id": ["R0", 0, 0],
                "object_label": 3,
                "score": 80.0,
                "source_feature": "area",
                "centroid_row_px": [20.0],
                "centroid_col_px": 200.0,
                "bbox_min_row_px": 18,
                "bbox_min_col_px": 195,
                "bbox_max_row_px": 23,
                "bbox_max_col_px": 205,
                "stage_x_um": 900.0,
                "stage_y_um": 1900.0,
            }]
        })


def test_to_builtin_rejects_unknown_objects():
    class Custom:
        pass

    with pytest.raises(TypeError, match="Custom"):
        to_builtin(Custom())
