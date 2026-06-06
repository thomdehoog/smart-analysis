from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import validate_targets, validate_tile_detection  # noqa: E402
from _detection_checkpoint import segmentation_params_hash  # noqa: E402
from _object_crops import extract_object_crops  # noqa: E402


WORKFLOW = Path(__file__).resolve().parents[1]
STEPS_DIR = WORKFLOW / "steps"
CLASSICAL_YAML = WORKFLOW / "pipelines" / "object_analysis.yaml"
DEEP_YAML = WORKFLOW / "pipelines" / "object_analysis_deep.yaml"
DETECTION_YAML = WORKFLOW / "pipelines" / "object_detection.yaml"
FEATURES_DEEP_YAML = WORKFLOW / "pipelines" / "object_features_deep.yaml"


def _load_step(name: str):
    path = STEPS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detect_objects = _load_step("detect_objects")
load_detected_objects = _load_step("load_detected_objects")
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
    pipeline_data = extract_deep_features.run(pipeline_data, {}, crop_size_px=10)
    result = build_object_table.run(pipeline_data, {})
    props = result["object_analysis"]["objects"]["properties"]

    masks_path = output_dir / "tiles" / "R0_r003_c007" / "masks.tif"
    object_dir = output_dir / "objects" / "R0_r003_c007_obj00001"
    assert masks_path.exists()
    assert (object_dir / "crop.tif").exists()
    assert (object_dir / "mask.tif").exists()
    assert props["crop_path"][0] == str(object_dir / "crop.tif")
    assert props["mask_path"][0] == str(object_dir / "mask.tif")


def test_detection_checkpoint_can_reenter_feature_extraction(tmp_path):
    detection_dir = tmp_path / "object_detection"
    features_dir = tmp_path / "object_features"
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(
            image_path,
            output_dir=str(detection_dir),
        ),
        "metadata": {"verbose": 0},
    }

    detection_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    detection = detection_data["detect_objects"]
    checkpoint_path = detection["artifacts"]["detection_checkpoint_json"]
    masks_path = detection["artifacts"]["masks_tif"]

    assert Path(checkpoint_path).exists()
    assert Path(masks_path).exists()

    feature_data = {
        "input": {
            "detection_checkpoint_path": checkpoint_path,
            "segmentation_params_hash": detection["segmentation_params_hash"],
            "output_dir": str(features_dir),
            "backend": "mock",
            "embedding_dim": 6,
        },
        "metadata": {"verbose": 0},
    }
    feature_data = load_detected_objects.run(feature_data, {})
    feature_data = extract_classical_features.run(feature_data, {})
    feature_data = extract_deep_features.run(feature_data, {}, crop_size_px=12)
    result = build_object_table.run(feature_data, {})["object_analysis"]
    props = result["objects"]["properties"]
    embeddings = result["objects"]["embeddings"]

    assert result["objects"]["n_objects"] == 2
    assert props["label"] == [1, 2]
    assert embeddings["backend"] == "mock"
    assert embeddings["label"] == props["label"]
    assert len(embeddings["vectors"]) == 2
    assert len(embeddings["vectors"][0]) == 6
    assert (features_dir / "objects" / "R0_r003_c007_obj00001" / "crop.tif").exists()


def test_detection_checkpoint_hash_ignores_runtime_and_area_filter():
    base = {
        "channels": None,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.4,
        "niter": None,
        "diameter": None,
        "max_segmentation_size_px": 512,
    }

    assert segmentation_params_hash(base | {"gpu": False, "min_area_px": 10}) == (
        segmentation_params_hash(base | {"gpu": True, "min_area_px": 2000})
    )


def test_checkpoint_reentry_can_retune_area_filter_without_redetection(tmp_path):
    detection_dir = tmp_path / "object_detection"
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(
            image_path,
            output_dir=str(detection_dir),
        ),
        "metadata": {"verbose": 0},
    }
    detection_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    detection = detection_data["detect_objects"]

    feature_data = {
        "input": {
            "detection_checkpoint_path": detection["artifacts"]["detection_checkpoint_json"],
            "segmentation_params_hash": detection["segmentation_params_hash"],
            "min_area_px": 80,
        },
        "metadata": {"verbose": 0},
    }
    feature_data = load_detected_objects.run(feature_data, {})

    assert feature_data["detect_objects"]["n_raw_objects"] == 2
    assert feature_data["detect_objects"]["n_objects"] == 1
    assert feature_data["detect_objects"]["dropped_labels"] == [1]


def test_detection_checkpoint_hash_mismatch_rejects_stale_masks(tmp_path):
    detection_dir = tmp_path / "object_detection"
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, output_dir=str(detection_dir)),
        "metadata": {"verbose": 0},
    }
    detection_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    checkpoint_path = detection_data["detect_objects"]["artifacts"]["detection_checkpoint_json"]

    with pytest.raises(ValueError, match="does not match"):
        load_detected_objects.run(
            {
                "input": {
                    "detection_checkpoint_path": checkpoint_path,
                    "segmentation_params_hash": "definitely-not-the-current-hash",
                },
                "metadata": {"verbose": 0},
            },
            {},
        )


def test_detection_checkpoint_rejects_changed_image_file(tmp_path):
    detection_dir = tmp_path / "object_detection"
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, output_dir=str(detection_dir)),
        "metadata": {"verbose": 0},
    }
    detection_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    checkpoint_path = detection_data["detect_objects"]["artifacts"]["detection_checkpoint_json"]

    import tifffile

    tifffile.imwrite(image_path, np.zeros((40, 48), dtype=np.uint8))

    with pytest.raises(ValueError, match="image hash"):
        load_detected_objects.run(
            {
                "input": {
                    "detection_checkpoint_path": checkpoint_path,
                    "segmentation_params_hash": detection_data["detect_objects"]["segmentation_params_hash"],
                },
                "metadata": {"verbose": 0},
            },
            {},
        )


def test_split_detection_features_engine_path_with_mock_backend(tmp_path, monkeypatch):
    import textwrap
    import time
    import yaml
    from engine import Engine

    fake_pkg = tmp_path / "fakepkgs" / "cellpose"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "models.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class CellposeModel:
                def __init__(self, gpu=False):
                    self.gpu = gpu

                def eval(self, x, channel_axis=None, **kwargs):
                    masks = np.zeros(x.shape[:2], dtype=np.int32)
                    masks[5:11, 6:12] = 1
                    masks[20:30, 25:35] = 2
                    return masks, None, None
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path / "fakepkgs") + (";" + sys.path[0] if sys.path else ""),
    )

    wrapper_dir = tmp_path / "steps"
    wrapper_dir.mkdir()
    for name in (
        "detect_objects",
        "load_detected_objects",
        "extract_classical_features",
        "extract_deep_features",
        "build_object_table",
    ):
        real_path = (STEPS_DIR / f"{name}.py").as_posix()
        (wrapper_dir / f"{name}.py").write_text(
            textwrap.dedent(
                f"""
                import importlib.util
                spec = importlib.util.spec_from_file_location("_real_{name}", r"{real_path}")
                _real = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_real)
                METADATA = {{"environment": None, "max_workers": 1}}

                def run(pipeline_data, state, **params):
                    return _real.run(pipeline_data, state, **params)
                """
            ),
            encoding="utf-8",
        )

    detection_yaml = tmp_path / "object_detection.yaml"
    detection_yaml.write_text(
        yaml.safe_dump({
            "metadata": {"functions_dir": str(wrapper_dir), "verbose": 0},
            "object_detection": [
                {"detect_objects": {
                    "channels": None,
                    "cellprob_threshold": 0.0,
                    "flow_threshold": 0.4,
                    "niter": None,
                    "diameter": None,
                    "max_segmentation_size_px": None,
                    "min_area_px": 80,
                    "max_area_px": None,
                    "persist_only": True,
                }},
            ],
        }, sort_keys=False),
        encoding="utf-8",
    )
    features_yaml = tmp_path / "object_features.yaml"
    features_yaml.write_text(
        yaml.safe_dump({
            "metadata": {"functions_dir": str(wrapper_dir), "verbose": 0},
            "object_features": [
                {"load_detected_objects": {"min_area_px": 0, "max_area_px": None}},
                {"extract_classical_features": None},
                {"extract_deep_features": {
                    "backend": "mock",
                    "embedding_dim": 6,
                    "crop_size_px": 12,
                    "mask": True,
                }},
                {"build_object_table": None},
            ],
        }, sort_keys=False),
        encoding="utf-8",
    )

    image_path = _write_synthetic_tile(tmp_path)
    detection_output = tmp_path / "object_detection"
    features_output = tmp_path / "object_features"
    payload = _payload(image_path, output_dir=str(detection_output), gpu=True)

    with Engine(execution_timeout=30) as engine:
        engine.register("object_detection_mock", str(detection_yaml))
        engine.submit("object_detection_mock", payload)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            results = engine.results("object_detection_mock")
            if results:
                detection_result = results[0]
                break
            assert not engine.status("object_detection_mock")["failed"]
            time.sleep(0.1)
        else:
            raise AssertionError("Timed out waiting for object_detection_mock")

        detection = detection_result["detect_objects"]
        assert "masks" not in detection
        checkpoint_path = detection["artifacts"]["detection_checkpoint_json"]
        expected_hash = detection["segmentation_params_hash"]

        engine.register("object_features_mock", str(features_yaml))
        engine.submit(
            "object_features_mock",
            {
                "detection_checkpoint_path": checkpoint_path,
                "segmentation_params_hash": expected_hash,
                "output_dir": str(features_output),
            },
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            results = engine.results("object_features_mock")
            if results:
                features_result = results[0]
                break
            assert not engine.status("object_features_mock")["failed"]
            time.sleep(0.1)
        else:
            raise AssertionError("Timed out waiting for object_features_mock")

    tile = features_result["object_analysis"]
    assert tile["objects"]["n_objects"] == 2
    assert tile["objects"]["embeddings"]["backend"] == "mock"


def test_fixed_crop_size_is_used_for_every_object(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path),
        "metadata": {"verbose": 0},
    }
    pipeline_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    crops = extract_object_crops(
        image=pipeline_data["detect_objects"]["image_2d"],
        masks=pipeline_data["detect_objects"]["masks"],
        tile_id=pipeline_data["input"]["tile_id"],
        crop_size_px=24,
        mask=False,
    )

    assert crops["crop_policy"] == {
        "crop_size_px": 24,
        "mask": False,
        "drop_incomplete": True,
    }
    assert crops["skipped_incomplete_labels"] == [1]
    assert [obj["label"] for obj in crops["objects"]] == [2]
    assert [obj["crop_height_px"] for obj in crops["objects"]] == [24]
    assert [obj["crop_width_px"] for obj in crops["objects"]] == [24]


def test_masked_crop_keeps_padded_context_when_object_is_complete():
    image = np.ones((20, 20), dtype=np.uint8)
    masks = np.zeros((20, 20), dtype=np.int32)
    masks[2:6, 2:6] = 1

    crops = extract_object_crops(
        image=image,
        masks=masks,
        tile_id=["R0", 0, 0],
        crop_size_px=12,
        mask=True,
        drop_incomplete=True,
    )

    assert crops["skipped_incomplete_labels"] == []
    assert [obj["label"] for obj in crops["objects"]] == [1]
    obj = crops["objects"][0]
    assert obj["crop_complete"] is True
    assert obj["crop_in_bounds"] is False
    assert obj["object_complete"] is True
    assert obj["object_fits_crop"] is True
    assert obj["crop_origin_row_px"] < 0
    assert obj["crop_origin_col_px"] < 0
    assert np.all(obj["crop_image"][obj["crop_mask"] == 0] == 0)


def test_unmasked_crop_drops_padded_context():
    image = np.ones((20, 20), dtype=np.uint8)
    masks = np.zeros((20, 20), dtype=np.int32)
    masks[2:6, 2:6] = 1

    crops = extract_object_crops(
        image=image,
        masks=masks,
        tile_id=["R0", 0, 0],
        crop_size_px=12,
        mask=False,
        drop_incomplete=True,
    )

    assert crops["objects"] == []
    assert crops["skipped_incomplete_labels"] == [1]


def test_masked_crop_drops_objects_touching_image_boundary():
    image = np.ones((20, 20), dtype=np.uint8)
    masks = np.zeros((20, 20), dtype=np.int32)
    masks[0:5, 8:12] = 1

    crops = extract_object_crops(
        image=image,
        masks=masks,
        tile_id=["R0", 0, 0],
        crop_size_px=12,
        mask=True,
        drop_incomplete=True,
    )

    assert crops["objects"] == []
    assert crops["skipped_incomplete_labels"] == [1]


def test_crop_size_must_contain_the_whole_object():
    image = np.ones((30, 30), dtype=np.uint8)
    masks = np.zeros((30, 30), dtype=np.int32)
    masks[10:20, 10:24] = 1

    crops = extract_object_crops(
        image=image,
        masks=masks,
        tile_id=["R0", 0, 0],
        crop_size_px=8,
        mask=True,
        drop_incomplete=True,
    )

    assert crops["objects"] == []
    assert crops["skipped_incomplete_labels"] == [1]


def test_masked_crop_zeroes_context_pixels(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path),
        "metadata": {"verbose": 0},
    }
    pipeline_data = detect_objects.run(pipeline_data, {"model": _StubCellposeModel()})
    crops = extract_object_crops(
        image=pipeline_data["detect_objects"]["image_2d"],
        masks=pipeline_data["detect_objects"]["masks"],
        tile_id=pipeline_data["input"]["tile_id"],
        crop_size_px=10,
        mask=True,
    )
    obj = crops["objects"][0]

    crop = obj["crop_image"]
    mask = obj["crop_mask"].astype(bool)
    assert crop[mask].max() > 0
    assert np.all(crop[~mask] == 0)


def test_deep_crop_geometry_comes_from_step_params(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(
            image_path,
            backend="mock",
            embedding_dim=6,
            crop_size_px=9,
            mask=True,
        ),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    pipeline_data = extract_deep_features.run(
        pipeline_data,
        {},
        crop_size_px=28,
        mask=False,
    )
    result = build_object_table.run(pipeline_data, {})["object_analysis"]
    props = result["objects"]["properties"]

    assert props["label"] == [2]
    assert props["crop_height_px"] == [28]
    assert props["crop_width_px"] == [28]
    assert props["crop_complete"] == [True]
    assert result["objects"]["embeddings"]["crop_policy"] == {
        "crop_size_px": 28,
        "mask": False,
        "drop_incomplete": True,
    }


def test_mock_deep_features_merge_into_object_table(tmp_path):
    image_path = _write_synthetic_tile(tmp_path)
    pipeline_data = {
        "input": _payload(image_path, backend="mock", embedding_dim=6),
        "metadata": {"verbose": 0},
    }
    state = {"model": _StubCellposeModel()}
    pipeline_data = detect_objects.run(pipeline_data, state)
    pipeline_data = extract_classical_features.run(pipeline_data, {})
    pipeline_data = extract_deep_features.run(pipeline_data, {}, crop_size_px=12)
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
            "label": [2, 1],
            "vectors": [[1.0, 0.0], [0.0, 1.0]],
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
    assert np.all(two_rgb[..., 2] == 18)


def test_dino_default_device_prefers_cuda_then_mps_then_cpu():
    class _Cuda:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

    class _Mps:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

    class _Backends:
        def __init__(self, mps_available):
            self.mps = _Mps(mps_available)

    class _Torch:
        def __init__(self, cuda_available, mps_available):
            self.cuda = _Cuda(cuda_available)
            self.backends = _Backends(mps_available)

    assert extract_deep_features._default_torch_device(_Torch(True, True)) == "cuda"
    assert extract_deep_features._default_torch_device(_Torch(False, True)) == "mps"
    assert extract_deep_features._default_torch_device(_Torch(False, False)) == "cpu"


def test_dino_two_channel_uint16_rgb_chain_scales_by_dtype():
    image = np.zeros((6, 7, 2), dtype=np.uint16)
    image[..., 0] = 12000
    image[..., 1] = 24000

    rgb = extract_deep_features._as_rgb(image)
    scaled = extract_deep_features._raw_image_to_float01(rgb)

    assert rgb.dtype == np.uint16
    assert scaled.shape == (6, 7, 3)
    assert scaled[..., 0].mean() == pytest.approx(12000 / 65535)
    assert scaled[..., 1].mean() == pytest.approx(24000 / 65535)
    assert scaled[..., 2].mean() == pytest.approx(18000 / 65535)


def test_dino_raw_integer_conversion_preserves_relative_brightness():
    image = np.asarray(
        [
            [[0, 25, 50], [50, 75, 100]],
            [[100, 125, 150], [150, 175, 200]],
        ],
        dtype=np.uint8,
    )

    scaled = extract_deep_features._raw_image_to_float01(image)

    assert scaled[0, 0, 0] == pytest.approx(0.0)
    assert scaled[0, 1, 0] == pytest.approx(50 / 255)
    assert scaled[1, 1, 2] == pytest.approx(200 / 255)


def test_dino_float_crops_must_be_preprocessed_to_unit_range():
    image = np.full((4, 5, 3), 2.0, dtype=np.float32)

    with pytest.raises(ValueError, match="preprocessing"):
        extract_deep_features._raw_image_to_float01(image)


def test_dino_rgb_conversion_rejects_non_crop_shapes():
    channel_first = np.zeros((2, 6, 7), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D crop"):
        extract_deep_features._as_rgb(channel_first)


def test_yaml_registers_classical_and_deep():
    from engine import Engine

    with Engine() as engine:
        engine.register("object_analysis", str(CLASSICAL_YAML))
        engine.register("object_analysis_deep", str(DEEP_YAML))
        engine.register("object_detection", str(DETECTION_YAML))
        engine.register("object_features_deep", str(FEATURES_DEEP_YAML))
        classical = engine.status("object_analysis")
        deep = engine.status("object_analysis_deep")
        detection = engine.status("object_detection")
        features = engine.status("object_features_deep")

    assert classical["completed"] == 0
    assert classical["failed"] == 0
    assert deep["completed"] == 0
    assert deep["failed"] == 0
    assert detection["completed"] == 0
    assert detection["failed"] == 0
    assert features["completed"] == 0
    assert features["failed"] == 0


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
@pytest.mark.pooch
@pytest.mark.slow
def test_real_cellpose_object_analysis_end_to_end(tmp_path):
    image_path, image = _write_immunohistochemistry_tile(tmp_path)

    result = _run_engine_workflow(
        "object_analysis_real_cellpose",
        CLASSICAL_YAML,
        _payload(
            image_path,
            tile_id=["IHC", 0, 0],
            tile_stage_xy_um=[10000.0, 15000.0],
            source_pixel_size_um=[0.5, 0.5],
            source_image_size_px=[int(image.shape[1]), int(image.shape[0])],
            image_to_stage=[[0.0, -1.0], [1.0, 0.0]],
            channels=None,
            gpu=True,
        ),
    )
    tile = validate_tile_detection(result["object_analysis"])

    assert tile["objects"]["n_objects"] > 0
    assert tile["objects"]["properties"]["object_id"][0].startswith("IHC_r000_c000_obj")
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
@pytest.mark.pooch
@pytest.mark.slow
def test_real_dinov2_embedding_end_to_end(tmp_path):
    image_path, image = _write_immunohistochemistry_tile(tmp_path)

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
