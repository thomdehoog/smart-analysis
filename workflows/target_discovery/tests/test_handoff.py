from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_overview, validate_targets, validate_tile_detection  # noqa: E402


_STEP_PATH = Path(__file__).resolve().parents[1] / "steps" / "select_targets.py"
_spec = importlib.util.spec_from_file_location("select_targets_handoff", _STEP_PATH)
select_targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_targets)


def _tile(tile_id, stage_xy, label_offset):
    return validate_tile_detection({
        "objects": {
            "properties": {
                "label": np.array([label_offset + 1, label_offset + 2]),
                "centroid_row_px": np.array([50.0, 25.0]),
                "centroid_col_px": np.array([50.0, 75.0]),
                "bbox_min_row_px": np.array([45, 20]),
                "bbox_min_col_px": np.array([45, 70]),
                "bbox_max_row_px": np.array([55, 30]),
                "bbox_max_col_px": np.array([55, 80]),
                "area": np.array([20.0, 80.0]),
                "intensity_mean": np.array([5.0, 8.0]),
                "eccentricity": np.array([0.1, 0.2]),
                "stage_x_um": np.array([stage_xy[0], stage_xy[0] + 25.0]),
                "stage_y_um": np.array([stage_xy[1], stage_xy[1] - 25.0]),
            },
            "n_objects": 2,
        },
        "geometry": {
            "tile_id": tile_id,
            "tile_stage_xy_um": stage_xy,
            "tile_zwide_um": 0.0,
            "source_pixel_size_um": [1.0, 1.0],
            "source_image_size_px": [100, 100],
            "image_to_stage": [[1.0, 0.0], [0.0, 1.0]],
        },
    })


def test_validated_overview_hands_off_to_discovery_per_tile():
    overview = validate_overview({
        "tiles": [
            _tile(["R0", 0, 0], [1000.0, 2000.0], 0),
            _tile(["R0", 0, 1], [1100.0, 2000.0], 10),
        ]
    })

    result = select_targets.run({
        "input": {
            "tiles": overview["tiles"],
            "feature": "area",
            "direction": "high",
            "n_per_tile": 1,
        },
        "metadata": {"verbose": 0},
    }, {})["target_discovery"]
    validated = validate_targets(result)

    targets = validated["targets"]
    assert [target["object_label"] for target in targets] == [2, 12]
    assert targets[0]["stage_x_um"] == pytest.approx(1025.0)
    assert targets[0]["stage_y_um"] == pytest.approx(1975.0)
    assert targets[1]["stage_x_um"] == pytest.approx(1125.0)
    assert targets[1]["stage_y_um"] == pytest.approx(1975.0)
