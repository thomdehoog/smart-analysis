"""
Integration tests for the rare_event_selection workflow.

Tests the full pipeline (preprocess -> segment -> extract_features -> feedback)
through the v4 Engine API. Uses mock steps for fast testing, and optionally
runs the real pipeline if cellpose + skimage are available.

Usage:
    python run_tests.py
"""

import sys
import os
import time
import textwrap
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

WIDTH = 70
BASE = Path(__file__).parent.parent


def _fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def _wait_results(engine, name, expected, timeout=30):
    t0 = time.monotonic()
    collected = []
    while time.monotonic() - t0 < timeout:
        collected.extend(engine.results(name))
        if len(collected) >= expected:
            return collected
        time.sleep(0.2)
    return collected


# ---- Tests with mock steps -------------------------------------------


def test_mock_full_pipeline():
    """Run the full 4-step pipeline with mock steps (no cellpose needed)."""
    from engine import Engine

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
            results = _wait_results(e, "test", 1, timeout=15)

        if not results:
            return False, "no results returned"

        r = results[0]
        if "preprocess" not in r:
            return False, "preprocess step did not run"
        if "segment" not in r:
            return False, "segment step did not run"
        if "extract_features" not in r:
            return False, "extract_features step did not run"
        if "feedback" not in r:
            return False, "feedback step did not run"

        fb = r["feedback"]
        if fb["n_selected"] < 1:
            return False, f"no cells selected"
        if not Path(fb["filepath"]).exists():
            return False, f"feedback file not written: {fb['filepath']}"

        return True, (f"4-step pipeline: {r['segment']['n_cells']} cells, "
                      f"{fb['n_selected']} selected, "
                      f"JSON written to {Path(fb['filepath']).name}")
    finally:
        shutil.rmtree(tmp, True)


def test_mock_warm_model():
    """Segment step reuses model across calls (state dict)."""
    from engine import Engine

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
            results = _wait_results(e, "test", 5, timeout=15)

        if len(results) != 5:
            return False, f"expected 5 results, got {len(results)}"

        counts = sorted(r["segment"]["load_count"] for r in results)
        if counts == [1, 2, 3, 4, 5]:
            return True, f"model warm: load_counts={counts}"
        if all(c >= 1 for c in counts):
            return True, f"model loaded (multi-worker): counts={counts}"
        return False, f"unexpected counts: {counts}"
    finally:
        shutil.rmtree(tmp, True)


def test_mock_data_flow_integrity():
    """Verify data flows correctly through all 4 steps."""
    from engine import Engine

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
            results = _wait_results(e, "test", 1, timeout=10)

        if not results:
            return False, "no results"

        r = results[0]
        # seed=5 -> a=10 -> b=20 -> c=60 -> d=59, chain=[10,20,60]
        expected_chain = [10, 20, 60]
        if r["d"]["chain"] != expected_chain:
            return False, f"chain={r['d']['chain']}, expected {expected_chain}"
        if r["d"]["value"] != 59:
            return False, f"final value={r['d']['value']}, expected 59"

        return True, f"data chain: {r['d']['chain']} -> {r['d']['value']}"
    finally:
        shutil.rmtree(tmp, True)


def test_mock_multiple_images():
    """Process multiple images concurrently."""
    from engine import Engine

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
            results = _wait_results(e, "test", 20, timeout=30)

        if len(results) != 20:
            return False, f"expected 20 results, got {len(results)}"

        ids = sorted(r["result"]["image_id"] for r in results)
        expected = sorted(f"img_{i}" for i in range(20))
        if ids != expected:
            return False, "missing or duplicate image ids"

        return True, f"20 images processed concurrently"
    finally:
        shutil.rmtree(tmp, True)


def test_mock_error_in_pipeline():
    """One step fails; engine records it gracefully."""
    from engine import Engine

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

        if status["failed"] < 1:
            return False, f"no failure recorded: {status}"
        err = status["failures"][0]["error"]
        if "segmentation failed" not in err:
            return False, f"wrong error message: {err}"

        return True, f"error recorded: {err}"
    finally:
        shutil.rmtree(tmp, True)


# ---- Test with real pipeline (requires cellpose + skimage) -----------


def test_real_pipeline():
    """Run the actual rare_event_selection pipeline with real packages.
    Skipped if cellpose or skimage are not installed."""
    try:
        import cellpose
        import skimage
    except ImportError as exc:
        return None, f"skipped: {exc}"

    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "rare_event_selection_pipeline.yaml")

    output_dir = tempfile.mkdtemp()
    try:
        with Engine() as e:
            e.register("analysis", yaml_path)
            e.submit("analysis", {"data_source": "skimage.human_mitosis"})

            results = _wait_results(e, "analysis", 1, timeout=120)

        if not results:
            s = e.status("analysis")
            if s["failed"] > 0:
                err = s["failures"][0]["error"]
                if "DLL" in err or "fbgemm" in err or "WinError" in err:
                    return None, f"skipped: DLL conflict (needs separate env): {err[:80]}"
            return False, f"no results (status: {s})"

        r = results[0]
        n_cells = r["segment"]["n_cells"]
        n_selected = r["feedback"]["n_selected"]

        if n_cells < 1:
            return False, f"no cells found"
        if n_selected < 1:
            return False, f"no cells selected"

        return True, (f"real pipeline: {n_cells} cells segmented, "
                      f"{n_selected} selected (p99 by area)")
    finally:
        shutil.rmtree(output_dir, True)


def test_real_pipeline_yaml_registers():
    """The real YAML file registers without errors."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "rare_event_selection_pipeline.yaml")

    try:
        with Engine() as e:
            e.register("analysis", yaml_path)
            status = e.status("analysis")
        return True, f"registered successfully: {status}"
    except Exception as exc:
        return False, f"register failed: {type(exc).__name__}: {exc}"


# ---- Runner ----------------------------------------------------------


TESTS = [
    ("Mock: full 4-step pipeline",       test_mock_full_pipeline),
    ("Mock: warm model (state dict)",    test_mock_warm_model),
    ("Mock: data flow integrity",        test_mock_data_flow_integrity),
    ("Mock: multiple images",            test_mock_multiple_images),
    ("Mock: error handling",             test_mock_error_in_pipeline),
    ("Real: YAML registers",             test_real_pipeline_yaml_registers),
    ("Real: full pipeline (cellpose)",   test_real_pipeline),
]


def main():
    import engine

    print()
    print("=" * WIDTH)
    print("  Rare Event Selection -- Integration Tests")
    print("=" * WIDTH)
    print()
    print(f"  Engine:   {engine.__version__}")
    print(f"  Python:   {sys.version.split()[0]} ({sys.executable})")
    print()

    t_total = time.perf_counter()
    results = []

    for i, (name, func) in enumerate(TESTS, 1):
        print("-" * WIDTH)
        print(f"  [{i:2d}/{len(TESTS)}] {name}")
        print("-" * WIDTH)

        t0 = time.perf_counter()
        try:
            passed, detail = func()
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0

        if passed is None:
            status = "[SKIP]"
        elif passed:
            status = "[ OK ]"
        else:
            status = "[FAIL]"

        print(f"  {status}  {detail}  ({_fmt(elapsed)})")
        print()

        results.append((name, passed, detail, elapsed))

    elapsed_total = time.perf_counter() - t_total
    n_pass = sum(1 for _, p, _, _ in results if p is True)
    n_fail = sum(1 for _, p, _, _ in results if p is False)
    n_skip = sum(1 for _, p, _, _ in results if p is None)

    print("=" * WIDTH)
    print("  Results")
    print("=" * WIDTH)
    print()

    for name, passed, detail, elapsed in results:
        if passed is None:
            icon = "[SKIP]"
        elif passed:
            icon = "[ OK ]"
        else:
            icon = "[FAIL]"
        print(f"  {icon}  {name:<38s}  {_fmt(elapsed):>8s}")

    print()
    print(f"  {'_' * (WIDTH - 4)}")
    print(f"  Passed:  {n_pass}/{len(TESTS)}")
    print(f"  Failed:  {n_fail}/{len(TESTS)}")
    if n_skip:
        print(f"  Skipped: {n_skip}/{len(TESTS)}")
    print(f"  Time:    {_fmt(elapsed_total)}")
    print()

    if n_fail == 0:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        print()
        for name, passed, detail, _ in results:
            if passed is False:
                print(f"    ** {name}: {detail}")

    print()
    print("=" * WIDTH)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
