from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_targets, validate_tile_detection  # noqa: E402
from _object_crops import extract_object_crops  # noqa: E402


WORKFLOW = Path(__file__).resolve().parents[1]
STEPS_DIR = WORKFLOW / "steps"
CLASSICAL_YAML = WORKFLOW / "pipelines" / "object_analysis.yaml"
DEEP_YAML = WORKFLOW / "pipelines" / "object_analysis_deep.yaml"


def _load_step(name: str):
    path = STEPS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect_objects = _load_step("detect_objects")
extract_classical_features = _load_step("extract_classical_features")
extract_deep_features = _load_step("extract_deep_features")
build_object_table = _load_step("build_object_table")


def _load_target_discovery_step():
    path = WORKFLOW.parent / "target_discovery" / "steps" / "select_targets.py"
    spec = importlib.util.spec_from_file_location("select_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_tile(tmp_path):
    import tifffile

    image = np.zeros((40, 48), dtype=np.uint8)
    image[5:11, 6:12] = 80
    image[20:30, 25:35] = 160
    path = tmp_path / "tile.ome.tiff"
    tifffile.imwrite(path, image, photometric="minisblack")
    return path


def _write_human_mitosis_tile(tmp_path):
    import tifffile
    from skimage.data import human_mitosis

    image = human_mitosis()
    path = tmp_path / "human_mitosis.ome.tiff"
    tifffile.imwrite(path, image, photometric="minisblack")
    return path, image


def _write_immunohistochemistry_tile(tmp_path):
    import tifffile
    from skimage.data import immunohistochemistry

    image = immunohistochemistry()
    path = tmp_path / "immunohistochemistry.ome.tiff"
    tifffile.imwrite(path, image, photometric="rgb")
    return path, image


def _payload(image_path: Path, **extra):
    payload = {
        "image_path": str(image_path),
        "tile_id": ["R0", 3, 7],
        "tile_stage_xy_um": [1000.0, 2000.0],
        "tile_zwide_um": 250.0,
        "source_pixel_size_um": [2.0, 3.0],
        "source_image_size_px": [48, 40],
        "image_to_stage": [[1.0, 0.0], [0.0, 1.0]],
        "gpu": False,
    }
    payload.update(extra)
    return payload


def _run_classical(tmp_path, **payload_extra):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, **payload_extra),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    return build_object_table.run(pipeline_data, {})


def _run_engine_workflow(name: str, yaml_path: Path, payload: dict, timeout=180):
    import time
    from engine import Engine

    with Engine() as engine:
        engine.register(name, str(yaml_path))
        engine.submit(name, payload)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            results = engine.results(name)
            if results:
                return results[0]
            status = engine.status(name)
            if status["failed"]:
                failure = status["failures"][0]
                raise AssertionError(f"{failure['step']}: {failure['error']}")
            time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {name}")


def test_classical_object_analysis_end_to_end_with_stub(tmp_path):
    result = _run_classical(tmp_path)
    tile = validate_tile_detection(result["object_analysis"])
    props = tile["objects"]["properties"]

    assert tile["objects"]["n_objects"] == 2
    assert props["label"] == [1, 2]
    assert props["object_id"] == [
        "R0_r003_c007_obj00001",
        "R0_r003_c007_obj00002",
    ]
    assert props["tile_name"] == ["R0_r003_c007", "R0_r003_c007"]
    assert "crop_height_px" not in props
    assert "crop_width_px" not in props
    assert "crop_path" not in props
    assert "embeddings" not in tile["objects"]
    assert props["stage_x_um"] == pytest.approx([969.0, 1011.0])
    assert props["stage_y_um"] == pytest.approx([1962.5, 2013.5])
    assert "centroid-0" not in props
    assert "bbox-0" not in props
    assert "preprocess" not in result
    assert "segment" not in result
    assert "extract_features" not in result
    assert "masks" not in result["detect_objects"]


def test_deep_path_writes_position_based_crop_artifacts(tmp_path):
    output_dir = tmp_path / "analysis" / "object_analysis" / "run_001"
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(
            image_path,
            output_dir=str(output_dir),
            backend="mock",
            embedding_dim=6,
        ),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    pipeline_data = extract_deep_features.run(pipeline_data, {})
    result = build_object_table.run(pipeline_data, {})
    props = result["object_analysis"]["objects"]["properties"]

    masks_path = output_dir / "tiles" / "R0_r003_c007" / "masks.tif"
    object_dir = output_dir / "objects" / "R0_r003_c007_obj00001"
    assert masks_path.exists()
    assert (object_dir / "crop.tif").exists()
    assert (object_dir / "mask.tif").exists()
    assert props["crop_path"][0] == str(object_dir / "crop.tif")
    assert props["mask_path"][0] == str(object_dir / "mask.tif")


def test_single_cell_crop_masks_context_pixels(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, crop_mode="single_cell"),
        "metadata": {"verbose": 0},
    }
    pipeline_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    crops = extract_object_crops(
        image=pipeline_data["detect_objects"]["image_2d"],
        masks=pipeline_data["detect_objects"]["masks"],
        tile_id=pipeline_data["input"]["tile_id"],
        mode="single_cell",
    )
    obj = crops["objects"][0]

    crop = obj["crop_image"]
    mask = obj["crop_mask"].astype(bool)
    assert crop[mask].max() > 0
    assert np.all(crop[~mask] == 0)


def test_mock_deep_features_merge_into_object_table(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, backend="mock", embedding_dim=6),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    pipeline_data = extract_deep_features.run(pipeline_data, {})
    result = build_object_table.run(pipeline_data, {})["object_analysis"]

    embeddings = result["objects"]["embeddings"]
    assert embeddings["backend"] == "mock"
    assert embeddings["label"] == result["objects"]["properties"]["label"]
    assert len(embeddings["vectors"]) == 2
    assert len(embeddings["vectors"][0]) == 6


def test_embedding_row_alignment_is_validated(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    pipeline_data["extract_deep_features"] = {
        "embeddings": {
            "label": [1],
            "vectors": [[1.0, 0.0]],
            "model": "bad",
            "backend": "mock",
        }
    }

    with pytest.raises(ValueError, match="embeddings.label"):
        build_object_table.run(pipeline_data, {})


def test_dino_rgb_conversion_handles_fewer_than_three_channels():
    gray = np.full((6, 7), 11, dtype=np.uint8)
    one = np.full((6, 7, 1), 13, dtype=np.uint8)
    two = np.zeros((6, 7, 2), dtype=np.uint8)
    two[..., 0] = 17
    two[..., 1] = 19

    gray_rgb = extract_deep_features._as_rgb(gray)
    one_rgb = extract_deep_features._as_rgb(one)
    two_rgb = extract_deep_features._as_rgb(two)

    assert gray_rgb.shape == (6, 7, 3)
    assert one_rgb.shape == (6, 7, 3)
    assert two_rgb.shape == (6, 7, 3)
    assert np.all(gray_rgb[..., 0] == 11)
    assert np.all(one_rgb[..., 2] == 13)
    assert np.all(two_rgb[..., 0] == 17)
    assert np.all(two_rgb[..., 1] == 19)
    assert np.all(two_rgb[..., 2] == 19)


def test_dino_rgb_conversion_rejects_non_crop_shapes():
    channel_first = np.zeros((2, 6, 7), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D crop"):
        extract_deep_features._as_rgb(channel_first)


def test_yaml_registers_classical_and_deep():
    from engine import Engine

    with Engine() as engine:
        engine.register("object_analysis", str(CLASSICAL_YAML))
        engine.register("object_analysis_deep", str(DEEP_YAML))
        classical = engine.status("object_analysis")
        deep = engine.status("object_analysis_deep")

    assert classical["completed"] == 0
    assert classical["failed"] == 0
    assert deep["completed"] == 0
    assert deep["failed"] == 0


def test_object_analysis_hands_off_to_target_discovery(tmp_path):
    result = _run_classical(tmp_path)
    tile = result["object_analysis"]
    select_targets = _load_target_discovery_step()

    discovery_pd = {
        "input": {
            "tiles": [tile],
            "feature": "area",
            "direction": "high",
            "n_per_tile": 1,
        },
        "metadata": {"verbose": 0},
    }
    targets = select_targets.run(discovery_pd, {})["target_discovery"]
    validated = validate_targets(targets)

    assert len(validated["targets"]) == 1
    assert validated["targets"][0]["object_label"] == 2


@pytest.mark.cellpose
@pytest.mark.slow
def test_real_cellpose_object_analysis_end_to_end(tmp_path):
    image_path, image = _write_human_mitosis_tile(tmp_path)

    result = _run_engine_workflow(
        "object_analysis_real_cellpose",
        CLASSICAL_YAML,
        _payload(
            image_path,
            tile_id=["R0", 0, 0],
            tile_stage_xy_um=[10000.0, 15000.0],
            source_pixel_size_um=[0.65, 0.65],
            source_image_size_px=[int(image.shape[1]), int(image.shape[0])],
            image_to_stage=[[0.0, -1.0], [1.0, 0.0]],
        ),
    )
    tile = validate_tile_detection(result["object_analysis"])

    assert tile["objects"]["n_objects"] > 0
    assert tile["objects"]["properties"]["object_id"][0].startswith("R0_r000_c000_obj")
    assert all(
        isinstance(value, float)
        for value in tile["objects"]["properties"]["stage_x_um"]
    )


@pytest.mark.cellpose
@pytest.mark.pooch
@pytest.mark.slow
def test_real_cpsam_multichannel_immunohistochemistry_end_to_end(tmp_path):
    image_path, image = _write_immunohistochemistry_tile(tmp_path)

    result = _run_engine_workflow(
        "object_analysis_real_cpsam_multichannel",
        CLASSICAL_YAML,
        _payload(
            image_path,
            tile_id=["IHC", 0, 0],
            tile_stage_xy_um=[5000.0, 6000.0],
            source_pixel_size_um=[0.5, 0.5],
            source_image_size_px=[int(image.shape[1]), int(image.shape[0])],
            image_to_stage=[[1.0, 0.0], [0.0, 1.0]],
            channels=None,
            gpu=True,
        ),
        timeout=240,
    )
    tile = validate_tile_detection(result["object_analysis"])
    props = tile["objects"]["properties"]

    assert tile["objects"]["n_objects"] > 0
    for channel in range(3):
        key = f"intensity_mean_c{channel}"
        assert key in props
        assert len(props[key]) == tile["objects"]["n_objects"]
    assert props["intensity_mean"] == props["intensity_mean_c0"]


@pytest.mark.cellpose
@pytest.mark.deep
@pytest.mark.slow
def test_real_dinov2_embedding_end_to_end(tmp_path):
    image_path, image = _write_human_mitosis_tile(tmp_path)

    result = _run_engine_workflow(
        "object_analysis_real_deep",
        DEEP_YAML,
        _payload(
            image_path,
            backend="dinov2",
            model_name="dinov2_vitb14",
            input_size_px=224,
            batch_size=1,
            gpu=True,
            source_image_size_px=[int(image.shape[1]), int(image.shape[0])],
        ),
        timeout=240,
    )

    tile = validate_tile_detection(result["object_analysis"])
    embeddings = tile["objects"]["embeddings"]
    assert embeddings["backend"] == "dinov2"
    assert embeddings["label"] == tile["objects"]["properties"]["label"]
    assert len(embeddings["vectors"]) == tile["objects"]["n_objects"]
    assert len(embeddings["vectors"][0]) > 100


class _StubCellposeModel:
    def eval(self, x, channel_axis=None, **kwargs):
        masks = np.zeros(x.shape[:2], dtype=np.int32)
        masks[5:11, 6:12] = 1
        masks[20:30, 25:35] = 2
        return masks, None, None
