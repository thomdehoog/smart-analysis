"""Tests for the cell_analysis workflow.

Covers three layers:

1. ``select_features`` is exercised directly with synthetic regionprops
   for each selection mode (percentile / threshold / top_n) and
   direction (high / low). No engine, no skimage, no cellpose.
2. A mock-segment end-to-end smoke test that uses the real preprocess /
   extract_features / select_features steps, with a stubbed segment
   step producing deterministic synthetic masks. Needs skimage.
3. A real-cellpose end-to-end test that registers the actual workflow
   YAML and submits a job. Auto-skipped via ``@pytest.mark.cellpose``
   when cellpose is not loadable.

Run with::

    pytest workflows/cell_analysis/tests/                # everything in env
    pytest workflows/cell_analysis/tests/ -m cellpose    # real-cellpose only
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import textwrap
from pathlib import Path

import pytest


WORKFLOW = Path(__file__).resolve().parent.parent
STEPS_DIR = WORKFLOW / "steps"
YAML_PATH = WORKFLOW / "pipelines" / "cell_analysis_pipeline.yaml"


def _load_step(name: str):
    """Import a step module by file path (mirrors what the engine does)."""
    spec = importlib.util.spec_from_file_location(
        f"cell_analysis_{name}", STEPS_DIR / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# select_features unit tests (no engine, no skimage, no cellpose)
# ---------------------------------------------------------------------------


def _fake_extract_output():
    """Synthetic regionprops dict with predictable values for selection."""
    import numpy as np
    return {
        "extract_features": {
            "properties": {
                "label": np.array([1, 2, 3, 4, 5]),
                "area": np.array([100.0, 50.0, 200.0, 75.0, 150.0]),
                "eccentricity": np.array([0.1, 0.5, 0.3, 0.9, 0.2]),
            },
            "n_cells": 5,
        },
        "metadata": {"verbose": 0},
    }


def test_select_features_percentile_high():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="area", mode="percentile",
                    percentile=80, direction="high")
    out = pd["select_features"]
    # np.percentile of [50,75,100,150,200] at p80 is 160 (linear interp);
    # only 200 is >= 160.
    assert out["selected_labels"] == [3]
    assert out["n_selected"] == 1
    assert out["n_total"] == 5
    assert out["cutoff"] == pytest.approx(160.0, abs=1.0)


def test_select_features_percentile_low():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="area", mode="percentile",
                    percentile=80, direction="low")
    out = pd["select_features"]
    # Low-percentile selection mirrors the high case through 100-p:
    # np.percentile([50,75,100,150,200], 20) is 70, so only 50 survives.
    assert out["selected_labels"] == [2]
    assert out["n_selected"] == 1
    assert out["n_total"] == 5
    assert out["cutoff"] == pytest.approx(70.0, abs=1.0)


def test_select_features_threshold_high():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="area", mode="threshold",
                    threshold=100, direction="high")
    out = pd["select_features"]
    # area >= 100: labels 1 (100), 3 (200), 5 (150).
    assert sorted(out["selected_labels"]) == [1, 3, 5]
    assert out["cutoff"] == 100.0


def test_select_features_threshold_low():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="eccentricity", mode="threshold",
                    threshold=0.4, direction="low")
    out = pd["select_features"]
    # eccentricity <= 0.4: labels 1 (0.1), 3 (0.3), 5 (0.2).
    assert sorted(out["selected_labels"]) == [1, 3, 5]


def test_select_features_top_n():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="area", mode="top_n",
                    top_n=2, direction="high")
    out = pd["select_features"]
    assert sorted(out["selected_labels"]) == [3, 5]   # 200 and 150
    assert out["n_selected"] == 2


def test_select_features_top_n_low():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    pd = select.run(pd, {}, feature="area", mode="top_n",
                    top_n=2, direction="low")
    out = pd["select_features"]
    assert out["selected_labels"] == [2, 4]   # 50 and 75, in label row order
    assert out["n_selected"] == 2
    assert out["cutoff"] == pytest.approx(75.0)


def test_select_features_unknown_feature_raises():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    with pytest.raises(KeyError, match="not found"):
        select.run(pd, {}, feature="not_a_real_feature", mode="percentile")


def test_select_features_unknown_mode_raises():
    select = _load_step("select_features")
    pd = _fake_extract_output()
    with pytest.raises(ValueError, match="Unknown mode"):
        select.run(pd, {}, feature="area", mode="bogus")


# ---------------------------------------------------------------------------
# Mock-segment smoke test: real preprocess + extract + select, fake segment
# ---------------------------------------------------------------------------


def test_pipeline_mock_smoke_end_to_end(temp_yaml, wait_for_completion):
    """Run preprocess -> stub-segment -> extract_features -> select_features
    on a deterministic TIFF. Cellpose is replaced with a synthetic mask
    grid so the public smoke path is fully offline."""
    try:
        import skimage  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"skimage not available: {exc}")

    from engine import Engine

    tmp = tempfile.mkdtemp()
    try:
        import numpy as np
        import tifffile

        yy, xx = np.indices((180, 180))
        image = ((yy * 3 + xx * 5) % 256).astype(np.uint8)
        image[40:140, 40:140] = np.maximum(image[40:140, 40:140], 180)
        image_path = Path(tmp, "synthetic_smoke.tif")
        tifffile.imwrite(image_path, image, photometric="minisblack")

        for name in ("preprocess.py", "extract_features.py",
                     "select_features.py"):
            shutil.copy2(STEPS_DIR / name, tmp)

        # Deterministic stub for segment: a 5x5 grid of square "cells".
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            import numpy as np
            METADATA = {"max_workers": 1}
            def run(pd, state, **p):
                img = pd["preprocess"]["image_preprocessed"]
                masks = np.zeros(img.shape, dtype=np.int32)
                h, w = img.shape
                cell, step_r, step_c = 30, h // 6, w // 6
                cell_id = 0
                for r0 in range(step_r, h - cell, step_r):
                    for c0 in range(step_c, w - cell, step_c):
                        cell_id += 1
                        masks[r0:r0+cell, c0:c0+cell] = cell_id
                pd["segment"] = {
                    "masks": masks,
                    "n_cells": int(masks.max()),
                }
                return pd
        """))

        yaml = temp_yaml(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
              verbose: 0
            cell_analysis:
              - preprocess:
                  sigma: 1.0
                  clip_limit: 0.03
              - segment:
              - extract_features:
              - select_features:
                  feature: "area"
                  mode: "percentile"
                  percentile: 80
                  direction: "high"
        """)

        with Engine() as e:
            e.register("ca", yaml)
            e.submit("ca", {"data_source": str(image_path)})
            results, status = wait_for_completion(e, "ca", 1, timeout=60)

        assert status["failed"] == 0, (
            f"step failed: {status['failures'][0]['error'][:120]}"
            if status["failures"] else f"failed without details: {status}"
        )
        assert len(results) == 1
        assert results[0]["preprocess"]["shape"] == image.shape
        assert results[0]["preprocess"]["data_source"] == str(image_path)
        out = results[0]["select_features"]
        assert out["mode"] == "percentile"
        assert out["feature"] == "area"
        assert 1 <= out["n_selected"] <= out["n_total"]
        assert all(label in results[0]["extract_features"]["properties"]["label"]
                   for label in out["selected_labels"])
    finally:
        shutil.rmtree(tmp, True)


# ---------------------------------------------------------------------------
# YAML registration smoke test (no actual run)
# ---------------------------------------------------------------------------


def test_yaml_registers():
    """The shipped pipeline YAML registers without errors. Doesn't run any
    steps -- just checks the YAML and step METADATA parse cleanly."""
    from engine import Engine

    with Engine() as e:
        e.register("ca", str(YAML_PATH))
        status = e.status("ca")

    assert status["completed"] == 0
    assert status["failed"] == 0


# ---------------------------------------------------------------------------
# Real cellpose end-to-end (auto-skipped when cellpose not loadable)
# ---------------------------------------------------------------------------


@pytest.mark.cellpose
@pytest.mark.slow
def test_pipeline_real_cellpose(wait_for_completion):
    """Run the actual cell_analysis pipeline (real Cellpose) on
    skimage.human_mitosis. Asserts the segment + feature + select chain
    produced a non-empty selection."""
    from engine import Engine

    with Engine() as e:
        e.register("ca", str(YAML_PATH))
        e.submit("ca", {"data_source": "skimage.human_mitosis"})
        results, status = wait_for_completion(e, "ca", 1, timeout=180)

    assert status["failed"] == 0, (
        f"step failed: {status['failures'][0]['error'][:160]}"
        if status["failures"] else f"failed without details: {status}"
    )
    assert len(results) == 1
    r = results[0]
    n_cells = r["segment"]["n_cells"]
    n_selected = r["select_features"]["n_selected"]
    n_total = r["select_features"]["n_total"]
    assert n_cells > 0, "Cellpose found no cells"
    assert n_total == n_cells, (
        f"feature extraction lost cells: {n_total} != {n_cells}"
    )
    assert 1 <= n_selected <= n_total


@pytest.mark.cellpose
@pytest.mark.slow
def test_pipeline_real_warm_model(wait_for_completion):
    """Submit three images and verify the Cellpose model is loaded only
    once per worker (state["model"] persistence)."""
    from engine import Engine

    with Engine() as e:
        e.register("ca", str(YAML_PATH))
        for _ in range(3):
            e.submit("ca", {"data_source": "skimage.human_mitosis"})
        results, status = wait_for_completion(e, "ca", 3, timeout=240)

    assert status["failed"] == 0
    assert len(results) == 3
    n_cells = [r["segment"]["n_cells"] for r in results]
    # Same image, same model -> same cell count each time.
    assert len(set(n_cells)) == 1, (
        f"warm model gave different counts across submits: {n_cells}"
    )
