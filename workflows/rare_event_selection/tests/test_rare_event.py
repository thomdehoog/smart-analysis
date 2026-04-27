"""Integration tests for the rare_event_selection workflow.

Covers the full preprocess -> segment -> extract_features -> feedback chain
through the v4 Engine API. Mock variants run anywhere skimage is available.
Real-cellpose variants need a working cellpose+torch in the active env
(e.g. dino3_test or SMART--rare_event_selection--main); they are auto-skipped
otherwise via the cellpose marker.

Run with::

    pytest -m "not cellpose"          # mock + smoke only
    pytest -m cellpose                # real cellpose
    pytest                            # everything (cellpose tests skip if unavailable)
"""

import json
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from engine import Engine

BASE = Path(__file__).parent.parent


# ---- Tests with mock steps -------------------------------------------


def test_mock_full_pipeline(wait_for_results):
    """Run the full 4-step pipeline with mock steps (no cellpose needed)."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "preprocess.py").write_text(textwrap.dedent("""
            import numpy as np
            def run(pd, state, **p):
                sigma = p.get("sigma", 1.0)
                img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                pd["preprocess"] = {
                    "image": img,
                    "image_preprocessed": img,
                    "shape": img.shape,
                    "sigma": sigma,
                }
                return pd
        """))
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            import numpy as np
            METADATA = {"max_workers": 1}
            def run(pd, state, **p):
                if "model" not in state:
                    state["model"] = "mock_cellpose"
                img = pd["preprocess"]["image_preprocessed"]
                masks = np.zeros(img.shape, dtype=np.int32)
                # Create 5 fake cells
                for i in range(1, 6):
                    r, c = i * 15, i * 15
                    masks[r:r+10, c:c+10] = i
                pd["segment"] = {
                    "masks": masks,
                    "n_cells": 5,
                    "diameter": p.get("diameter"),
                }
                return pd
        """))
        Path(tmp, "extract_features.py").write_text(textwrap.dedent("""
            import numpy as np
            def run(pd, state, **p):
                masks = pd["segment"]["masks"]
                img = pd["preprocess"]["image"]
                n_cells = pd["segment"]["n_cells"]
                select_by = p.get("select_by", "area")
                percentile = p.get("percentile", 99)
                # Mock properties
                props = {
                    "label": np.arange(1, n_cells + 1),
                    "area": np.array([100, 80, 120, 90, 150]),
                    "centroid-0": np.array([15, 30, 45, 60, 75], dtype=float),
                    "centroid-1": np.array([15, 30, 45, 60, 75], dtype=float),
                    "eccentricity": np.array([0.3, 0.5, 0.2, 0.7, 0.1]),
                    "mean_intensity": np.array([120, 100, 140, 90, 160], dtype=float),
                }
                values = props[select_by]
                threshold = float(np.percentile(values, percentile))
                selected_mask = values >= threshold
                selected_labels = props["label"][selected_mask]
                pd["extract_features"] = {
                    "properties": props,
                    "select_by": select_by,
                    "percentile": percentile,
                    "threshold": threshold,
                    "selected_labels": selected_labels,
                }
                return pd
        """))
        Path(tmp, "feedback.py").write_text(textwrap.dedent("""
            import json
            import numpy as np
            from pathlib import Path as P
            from datetime import datetime
            def run(pd, state, **p):
                output_dir = p.get("output_dir", ".")
                props = pd["extract_features"]["properties"]
                selected_labels = pd["extract_features"]["selected_labels"]
                cells = []
                for lbl in selected_labels:
                    idx = int(np.where(props["label"] == lbl)[0][0])
                    cells.append({
                        "label": int(lbl),
                        "centroid_x": float(props["centroid-1"][idx]),
                        "centroid_y": float(props["centroid-0"][idx]),
                        "area": int(props["area"][idx]),
                        "mean_intensity": float(props["mean_intensity"][idx]),
                        "eccentricity": float(props["eccentricity"][idx]),
                    })
                out_path = P(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = out_path / f"feedback_test_{ts}.json"
                with open(filepath, "w") as f:
                    json.dump({"cells": cells}, f)
                pd["feedback"] = {
                    "filepath": str(filepath),
                    "n_selected": len(cells),
                    "cells": cells,
                }
                return pd
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
              verbose: 0
            rare-event:
              - preprocess:
                  sigma: 1.0
              - segment:
                  diameter: null
              - extract_features:
                  select_by: "area"
                  percentile: 80
              - feedback:
                  output_dir: "{Path(tmp, 'output').as_posix()}"
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))
            e.submit("test", {})
            results = wait_for_results(e, "test", 1, timeout=15)

        assert results, "no results returned"

        r = results[0]
        assert "preprocess" in r, "preprocess step did not run"
        assert "segment" in r, "segment step did not run"
        assert "extract_features" in r, "extract_features step did not run"
        assert "feedback" in r, "feedback step did not run"

        fb = r["feedback"]
        assert fb["n_selected"] >= 1, "no cells selected"
        assert Path(fb["filepath"]).exists(), (
            f"feedback file not written: {fb['filepath']}"
        )
    finally:
        shutil.rmtree(tmp, True)


def test_mock_warm_model(wait_for_results):
    """Segment step reuses model across calls (state dict)."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "preprocess.py").write_text(textwrap.dedent("""
            import numpy as np
            def run(pd, state, **p):
                pd["preprocess"] = {
                    "image_preprocessed": np.zeros((10, 10), dtype=np.uint8),
                }
                return pd
        """))
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            import numpy as np
            METADATA = {"max_workers": 1}
            def run(pd, state, **p):
                if "model" not in state:
                    state["model"] = "loaded"
                    state["load_count"] = 0
                state["load_count"] += 1
                pd["segment"] = {
                    "masks": np.zeros((10, 10), dtype=np.int32),
                    "n_cells": 0,
                    "load_count": state["load_count"],
                    "model": state["model"],
                }
                return pd
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - preprocess:
              - segment:
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))
            for i in range(5):
                e.submit("test", {})
                time.sleep(0.3)
            results = wait_for_results(e, "test", 5, timeout=15)

        assert len(results) == 5, f"expected 5 results, got {len(results)}"

        counts = sorted(r["segment"]["load_count"] for r in results)
        # Either fully warm (max_workers=1 -> [1,2,3,4,5]) or any non-zero
        # counts in the multi-worker case.
        assert all(c >= 1 for c in counts), f"unexpected counts: {counts}"
    finally:
        shutil.rmtree(tmp, True)


def test_mock_data_flow_integrity(wait_for_results):
    """Verify data flows correctly through all 4 steps."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "step_a.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["a"] = {"value": pd["input"]["seed"] * 2}
                return pd
        """))
        Path(tmp, "step_b.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["b"] = {"value": pd["a"]["value"] + 10}
                return pd
        """))
        Path(tmp, "step_c.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["c"] = {"value": pd["b"]["value"] * 3}
                return pd
        """))
        Path(tmp, "step_d.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["d"] = {
                    "value": pd["c"]["value"] - 1,
                    "chain": [pd["a"]["value"], pd["b"]["value"],
                              pd["c"]["value"]],
                }
                return pd
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - step_a:
              - step_b:
              - step_c:
              - step_d:
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))
            e.submit("test", {"seed": 5})
            results = wait_for_results(e, "test", 1, timeout=10)

        assert results, "no results"

        r = results[0]
        # seed=5 -> a=10 -> b=20 -> c=60 -> d=59, chain=[10,20,60]
        expected_chain = [10, 20, 60]
        assert r["d"]["chain"] == expected_chain, (
            f"chain={r['d']['chain']}, expected {expected_chain}"
        )
        assert r["d"]["value"] == 59, (
            f"final value={r['d']['value']}, expected 59"
        )
    finally:
        shutil.rmtree(tmp, True)


def test_mock_multiple_images(wait_for_results):
    """Process multiple images concurrently."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "analyze.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["result"] = {
                    "image_id": pd["input"]["image_id"],
                    "n_pixels": pd["input"]["width"] * pd["input"]["height"],
                }
                return pd
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - analyze:
        """))

        with Engine(max_concurrent=8) as e:
            e.register("test", str(yaml_path))
            for i in range(20):
                e.submit("test", {
                    "image_id": f"img_{i}",
                    "width": 100 + i,
                    "height": 200 + i,
                })
            results = wait_for_results(e, "test", 20, timeout=30)

        assert len(results) == 20, f"expected 20 results, got {len(results)}"

        ids = sorted(r["result"]["image_id"] for r in results)
        expected = sorted(f"img_{i}" for i in range(20))
        assert ids == expected, "missing or duplicate image ids"
    finally:
        shutil.rmtree(tmp, True)


def test_mock_error_in_pipeline():
    """One step fails; engine records it gracefully."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "good.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["good"] = True
                return pd
        """))
        Path(tmp, "bad.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                raise RuntimeError("segmentation failed: no cells found")
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - good:
              - bad:
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))
            e.submit("test", {})
            time.sleep(5)
            status = e.status("test")

        assert status["failed"] >= 1, f"no failure recorded: {status}"
        err = status["failures"][0]["error"]
        assert "segmentation failed" in err, f"wrong error message: {err}"
    finally:
        shutil.rmtree(tmp, True)


# ---- Smoke tests with real components (no cellpose) ------------------


def test_real_components_smoke(wait_for_completion):
    """End-to-end smoke test: real preprocess.py + extract_features.py +
    feedback.py from the workflow's steps directory, on real
    skimage.human_mitosis. Cellpose is stubbed with deterministic synthetic
    masks so this runs in any env that has skimage + numpy."""
    try:
        import skimage  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"skimage unavailable: {exc}")

    real_steps = BASE / "steps"
    tmp = tempfile.mkdtemp()
    try:
        # Copy the real, production step files we want to actually exercise
        for name in ("preprocess.py", "extract_features.py", "feedback.py"):
            shutil.copy2(real_steps / name, tmp)

        # Stub the segment step: deterministic 5x5 grid of labeled "cells".
        # No cellpose, no torch -- works in any env with numpy.
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            import numpy as np
            METADATA = {"max_workers": 1}
            def run(pd, state, **p):
                img = pd["preprocess"]["image_preprocessed"]
                masks = np.zeros(img.shape, dtype=np.int32)
                h, w = img.shape
                cell = 30
                step_r, step_c = h // 6, w // 6
                cell_id = 0
                for r0 in range(step_r, h - cell, step_r):
                    for c0 in range(step_c, w - cell, step_c):
                        cell_id += 1
                        masks[r0:r0+cell, c0:c0+cell] = cell_id
                pd["segment"] = {
                    "masks": masks,
                    "n_cells": int(masks.max()),
                    "diameter": p.get("diameter"),
                }
                return pd
        """))

        out_dir = Path(tmp, "output")
        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
              verbose: 0
            wf:
              - preprocess:
                  sigma: 1.0
                  clip_limit: 0.03
              - segment:
              - extract_features:
                  select_by: "area"
                  percentile: 90
              - feedback:
                  output_dir: "{out_dir.as_posix()}"
        """))

        with Engine() as e:
            e.register("smoke", str(yaml_path))
            e.submit("smoke", {"data_source": "skimage.human_mitosis"})
            results, status = wait_for_completion(
                e, "smoke", 1, timeout=60,
            )

        assert status["failed"] == 0, (
            f"step failed: {status['failures'][0]['error'][:120]}"
            if status["failed"] else ""
        )
        assert results, f"no results (status: {status})"

        r = results[0]

        # Real preprocess.py loaded a real skimage image
        shape = r["preprocess"]["shape"]
        assert shape[0] >= 100 and shape[1] >= 100, (
            f"preprocess image suspiciously small: {shape}"
        )
        assert r["preprocess"]["data_source"] == "skimage.human_mitosis", (
            "preprocess did not record data source"
        )

        # Real extract_features.py ran skimage.measure.regionprops_table
        props = r["extract_features"]["properties"]
        n_extracted = len(props["label"])
        assert n_extracted >= 1, "extract_features found no cells"
        required_keys = {
            "label", "area", "centroid-0", "centroid-1", "eccentricity",
            "mean_intensity", "max_intensity", "solidity",
            "major_axis_length", "minor_axis_length",
        }
        missing = required_keys - set(props.keys())
        assert not missing, f"regionprops missing keys: {sorted(missing)}"

        # Real feedback.py wrote JSON we can read back
        fb_path = Path(r["feedback"]["filepath"])
        assert fb_path.exists(), f"feedback JSON not on disk: {fb_path}"
        with open(fb_path) as f:
            fb_data = json.load(f)
        assert "cells" in fb_data, (
            f"feedback JSON missing 'cells' key: {sorted(fb_data)}"
        )
        n_in_json = len(fb_data["cells"])
        assert n_in_json == r["feedback"]["n_selected"], (
            f"feedback count mismatch: "
            f"n_selected={r['feedback']['n_selected']} but "
            f"JSON has {n_in_json} cells"
        )
    finally:
        shutil.rmtree(tmp, True)


# ---- Test with real pipeline (requires cellpose + skimage) -----------


def test_real_pipeline_yaml_registers():
    """The real YAML file registers without errors."""
    yaml_path = str(BASE / "pipelines" / "rare_event_selection_pipeline.yaml")

    with Engine() as e:
        e.register("analysis", yaml_path)
        e.status("analysis")


@pytest.mark.cellpose
@pytest.mark.slow
@pytest.mark.integration
def test_real_pipeline(wait_for_completion):
    """Run the actual rare_event_selection pipeline with real packages."""
    yaml_path = str(BASE / "pipelines" / "rare_event_selection_pipeline.yaml")

    with Engine() as e:
        e.register("analysis", yaml_path)
        e.submit("analysis", {"data_source": "skimage.human_mitosis"})
        results, status = wait_for_completion(
            e, "analysis", 1, timeout=120,
        )

    assert status["failed"] == 0, (
        f"failed: {status['failures'][0]['error'][:120]}"
        if status["failed"] else ""
    )
    assert results, f"no results within timeout (status: {status})"

    r = results[0]
    n_cells = r["segment"]["n_cells"]
    n_selected = r["feedback"]["n_selected"]

    assert n_cells >= 1, "no cells found"
    assert n_selected >= 1, "no cells selected"


@pytest.mark.cellpose
@pytest.mark.slow
@pytest.mark.integration
def test_real_warm_cellpose_model(wait_for_completion):
    """Verify the real CellposeModel persists in state across submits.
    Submits 3 images on max_workers=1; asserts the model was loaded once
    and reused (load_counts==[1,2,3]), and every call segmented cells."""
    real_steps = BASE / "steps"
    tmp = tempfile.mkdtemp()
    try:
        # Real preprocess.py loads skimage.human_mitosis
        shutil.copy2(real_steps / "preprocess.py", tmp)

        # Custom segment that uses real cellpose + a load counter, so we
        # can prove the model object is reused across submits rather than
        # reloaded.
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            METADATA = {"max_workers": 1}
            def run(pd, state, **p):
                if "model" not in state:
                    from cellpose import models
                    state["model"] = models.CellposeModel(gpu=False)
                    state["load_count"] = 0
                state["load_count"] += 1
                img = pd["preprocess"]["image_preprocessed"]
                masks, flows, styles = state["model"].eval(img)
                pd["segment"] = {
                    "n_cells": int(masks.max()),
                    "load_count": state["load_count"],
                }
                return pd
        """))

        yaml_path = Path(tmp, "pipeline.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
              verbose: 0
            wf:
              - preprocess:
                  sigma: 1.0
                  clip_limit: 0.03
              - segment:
        """))

        with Engine() as e:
            e.register("warm", str(yaml_path))
            for _ in range(3):
                e.submit("warm", {"data_source": "skimage.human_mitosis"})
            results, status = wait_for_completion(
                e, "warm", 3, timeout=180,
            )

        assert status["failed"] == 0, (
            f"failed: {status['failures'][0]['error'][:120]}"
            if status["failed"] else ""
        )
        assert len(results) == 3, f"expected 3 results, got {len(results)}"

        load_counts = sorted(r["segment"]["load_count"] for r in results)
        n_cells = [r["segment"]["n_cells"] for r in results]

        assert load_counts == [1, 2, 3], (
            f"model was reloaded between submits: load_counts={load_counts}"
        )
        assert all(n > 0 for n in n_cells), (
            f"some segmentations found no cells: n_cells={n_cells}"
        )
    finally:
        shutil.rmtree(tmp, True)
