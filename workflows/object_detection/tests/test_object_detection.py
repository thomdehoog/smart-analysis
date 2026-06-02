from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_tile_detection  # noqa: E402
from _segmentation import ensure_2d  # noqa: E402


WORKFLOW = Path(__file__).resolve().parents[1]
STEPS_DIR = WORKFLOW / "steps"
YAML_PATH = WORKFLOW / "pipelines" / "object_detection.yaml"


def _load_step(name: str):
    path = STEPS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_objects = _load_step("package_objects")
detect_objects = _load_step("detect_objects")


def _feature_pipeline_data(labels=(1, 3, 5)):
    labels = list(labels)
    n = len(labels)
    return {
        "extract_features": {
            "properties": {
                "label": np.asarray(labels, dtype=np.int64),
                "centroid-0": np.asarray([10.0, 30.0, 50.0][:n]),
                "centroid-1": np.asarray([100.0, 300.0, 500.0][:n]),
                "bbox-0": np.asarray([8, 28, 48][:n]),
                "bbox-1": np.asarray([95, 295, 495][:n]),
                "bbox-2": np.asarray([14, 34, 54][:n]),
                "bbox-3": np.asarray([106, 306, 506][:n]),
                "area": np.asarray([20.0, 60.0, 40.0][:n]),
                "intensity_mean": np.asarray([2.0, 6.0, 4.0][:n]),
                "eccentricity": np.asarray([0.1, 0.3, 0.5][:n]),
                "intensity_median": np.asarray([1.5, 5.5, 3.5][:n]),
            },
            "n_cells": n,
        },
        "detect_objects": {
            "image_size_px": [512, 256],
        },
        "input": {
            "tile_id": ("R0", 2, 4),
            "tile_stage_xy_um": (1000.0, 2000.0),
            "tile_zwide_um": 250.0,
            "source_pixel_size_um": (0.5, 0.7),
            "image_to_stage": [[1.0, 0.0], [0.0, 1.0]],
        },
        "metadata": {"verbose": 0},
    }


def test_package_objects_maps_regionprops_columns_to_public_scalars():
    result = package_objects.run(_feature_pipeline_data(), {})["object_detection"]
    tile = validate_tile_detection(result)
    props = tile["objects"]["properties"]

    assert props["label"] == [1, 3, 5]
    assert props["centroid_row_px"] == [10.0, 30.0, 50.0]
    assert props["centroid_col_px"] == [100.0, 300.0, 500.0]
    assert props["bbox_min_row_px"] == [8, 28, 48]
    assert props["bbox_min_col_px"] == [95, 295, 495]
    assert props["bbox_max_row_px"] == [14, 34, 54]
    assert props["bbox_max_col_px"] == [106, 306, 506]
    assert "centroid-0" not in props
    assert "bbox-0" not in props
    assert props["intensity_median"] == [1.5, 5.5, 3.5]


def test_package_objects_uses_detected_image_size_when_input_omits_it():
    result = package_objects.run(_feature_pipeline_data(), {})["object_detection"]

    assert result["geometry"]["source_image_size_px"] == [512, 256]


def test_package_objects_preserves_sparse_label_row_alignment():
    result = package_objects.run(_feature_pipeline_data(labels=(1, 3, 5)), {})
    props = result["object_detection"]["objects"]["properties"]

    assert props["label"] == [1, 3, 5]
    assert props["area"] == [20.0, 60.0, 40.0]
    assert props["centroid_row_px"] == [10.0, 30.0, 50.0]


def test_empty_detection_output_is_valid():
    result = package_objects.run(_feature_pipeline_data(labels=()), {})
    tile = validate_tile_detection(result["object_detection"])

    assert tile["objects"]["n_objects"] == 0
    assert tile["objects"]["properties"]["label"] == []


@pytest.mark.parametrize("field", [
    "tile_id",
    "tile_stage_xy_um",
    "tile_zwide_um",
    "source_pixel_size_um",
    "image_to_stage",
])
def test_missing_geometry_field_fails_clearly(field):
    pd = _feature_pipeline_data()
    del pd["input"][field]

    with pytest.raises(ValueError, match=field):
        package_objects.run(pd, {})


@pytest.mark.parametrize("column", package_objects.FEATURE_COLUMN_MAP)
def test_missing_required_feature_column_fails_clearly(column):
    pd = _feature_pipeline_data()
    del pd["extract_features"]["properties"][column]

    with pytest.raises(ValueError, match=column):
        package_objects.run(pd, {})


def test_no_numpy_scalars_in_public_output():
    result = package_objects.run(_feature_pipeline_data(), {})["object_detection"]

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        else:
            yield value

    assert not any(type(value).__module__ == "numpy" for value in walk(result))


def test_detect_objects_reads_tiff_and_wires_feature_inputs(tmp_path):
    import tifffile

    image_path = tmp_path / "tile.ome.tiff"
    image = np.zeros((16, 20), dtype=np.uint8)
    image[5:8, 6:9] = 255
    tifffile.imwrite(image_path, image, photometric="minisblack")

    state = {"model": _StubCellposeModel()}
    result = detect_objects.run({
        "input": {"image_path": str(image_path)},
        "metadata": {"verbose": 0},
    }, state)

    detection = result["detect_objects"]
    assert detection["image_size_px"] == [20, 16]
    assert detection["n_objects"] == 1
    assert result["preprocess"]["image"].shape == (16, 20)
    assert result["segment"]["n_cells"] == 1


def test_ensure_2d_supports_channel_first_and_channel_last():
    c_first = np.zeros((3, 10, 20), dtype=np.uint8)
    c_last = np.zeros((10, 20, 3), dtype=np.uint8)

    assert ensure_2d(c_first, 2).shape == (10, 20)
    assert ensure_2d(c_last, 1).shape == (10, 20)


def test_ensure_2d_rejects_ambiguous_3d_shape():
    with pytest.raises(ValueError, match="Cannot extract 2D"):
        ensure_2d(np.zeros((10, 20, 30), dtype=np.uint8), 0)


def test_yaml_registers():
    from engine import Engine

    with Engine() as engine:
        engine.register("object_detection", str(YAML_PATH))
        status = engine.status("object_detection")

    assert status["completed"] == 0
    assert status["failed"] == 0


@pytest.mark.cellpose
@pytest.mark.slow
def test_real_cellpose_detection_packages_and_discovers_targets(tmp_path):
    import tifffile
    from skimage.data import human_mitosis

    extract_object_features = _load_step("extract_object_features")
    discover_targets = _load_target_discovery_step()

    image = human_mitosis()
    image_path = tmp_path / "human_mitosis.ome.tiff"
    tifffile.imwrite(image_path, image, photometric="minisblack")

    pipeline_data = {
        "input": {
            "image_path": str(image_path),
            "tile_id": ["R0", 0, 0],
            "tile_stage_xy_um": [10000.0, 15000.0],
            "tile_zwide_um": 2500.0,
            "source_pixel_size_um": [0.65, 0.65],
            "source_image_size_px": [int(image.shape[1]), int(image.shape[0])],
            "image_to_stage": [[0.0, -1.0], [1.0, 0.0]],
            "channel": 0,
            "diameter": None,
            "gpu": False,
        },
        "metadata": {"verbose": 0},
    }

    detected = detect_objects.run(pipeline_data, {})
    assert detected["detect_objects"]["n_objects"] > 0

    featured = extract_object_features.run(detected, {})
    packaged = package_objects.run(featured, {})
    tile = validate_tile_detection(packaged["object_detection"])
    assert tile["objects"]["n_objects"] == detected["detect_objects"]["n_objects"]

    discovery_pd = {
        "input": {
            "tiles": [tile],
            "feature": "area",
            "direction": "high",
            "n_per_tile": 3,
        },
        "metadata": {"verbose": 0},
    }
    discovered = discover_targets.run(discovery_pd, {})["target_discovery"]

    assert 1 <= len(discovered["targets"]) <= 3
    assert all(isinstance(target["stage_x_um"], float)
               for target in discovered["targets"])
    assert all(isinstance(target["stage_y_um"], float)
               for target in discovered["targets"])


def _load_target_discovery_step():
    path = WORKFLOW.parent / "target_discovery" / "steps" / "discover_targets.py"
    spec = importlib.util.spec_from_file_location("discover_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubCellposeModel:
    def eval(self, image_2d, diameter=None):
        masks = np.zeros(image_2d.shape, dtype=np.int32)
        masks[5:8, 6:9] = 1
        return masks, None, None
