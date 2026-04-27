"""Integration tests for the v4 pipeline engine.

End-to-end tests using real YAML pipelines through the v4 Engine API,
including adaptive microscopy patterns, scoped aggregation, and the
rare-event-selection workflow shape.

Run with::

    pytest -m integration

Marker: integration (registered in pyproject.toml).
"""

import shutil
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from engine import Engine


BASE = Path(__file__).parent.parent


@pytest.mark.integration
def test_local_single_step(wait_for_results):
    """Single local step executes and returns data."""
    yaml_path = str(BASE / "pipelines" / "test_local_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "hello"})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    step_data = r.get("step_local", {})
    assert step_data.get("executed"), "step_local did not execute"
    assert step_data.get("params_used", {}).get("test_param") == "hello", \
        f"params wrong: {step_data.get('params_used')}"


@pytest.mark.integration
def test_mixed_data_flow(wait_for_results):
    """Multiple local steps pass data between them."""
    yaml_path = str(BASE / "pipelines" / "test_mixed_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"from_user": True})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    local1 = r.get("step_local", {})
    local2 = r.get("step_local_2", {})

    assert local1.get("executed"), "step_local did not execute"
    assert local2.get("executed"), "step_local_2 did not execute"
    assert "step_local" in local2.get("previous_steps_found", []), \
        "step_local_2 did not see step_local data"


@pytest.mark.integration
@pytest.mark.slow
def test_error_handling():
    """Pipeline with error step records failure gracefully."""
    yaml_path = str(BASE / "pipelines" / "test_error_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {})
        time.sleep(5)
        status = e.status("test")

    assert status["failed"] >= 1, f"no failure recorded: {status}"
    assert "Deliberate test error" in status["failures"][0].get("error", ""), \
        f"wrong error: {status['failures']}"


@pytest.mark.integration
def test_scoped_spatial_pipeline(wait_for_results):
    """Real scoped YAML: tile processing followed by stitching."""
    yaml_path = str(BASE / "pipelines" / "test_scoped_spatial_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)

        # Submit a 2x2 tile grid
        for row in range(2):
            for col in range(2):
                is_last = (row == 1 and col == 1)
                e.submit(
                    "test",
                    {"row": row, "col": col},
                    scope={"region": "grid_A"},
                    complete="region" if is_last else None,
                )

        # Expect 4 Phase 0 results + 1 scoped result
        results = wait_for_results(e, "test", 5, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result (stitch did not trigger)"

    stitched = scoped[0].get("stitched", {})
    assert stitched.get("n_tiles") == 4, \
        f"expected 4 tiles, got {stitched.get('n_tiles')}"
    assert stitched.get("n_rows") == 2 and stitched.get("n_cols") == 2, \
        f"grid wrong: {stitched.get('n_rows')}x{stitched.get('n_cols')}"


@pytest.mark.integration
def test_scoped_multi_step_pipeline(wait_for_results):
    """Multi-step Phase 0 (local + tile) followed by scoped stitch."""
    yaml_path = str(BASE / "pipelines" / "test_scoped_multi_step_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)

        for row in range(3):
            is_last = (row == 2)
            e.submit(
                "test",
                {"row": row, "col": 0},
                scope={"region": "strip"},
                complete="region" if is_last else None,
            )

        results = wait_for_results(e, "test", 4, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result"

    stitched = scoped[0].get("stitched", {})
    assert stitched.get("n_tiles") == 3, \
        f"expected 3 tiles, got {stitched.get('n_tiles')}"


@pytest.mark.integration
def test_identity_passthrough(wait_for_results):
    """Identity step returns data unmodified; pipeline continues."""
    yaml_path = str(BASE / "pipelines" / "test_identity_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "test_value"})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    assert r.get("step_local", {}).get("executed"), \
        "step_local did not execute"
    assert r.get("step_local_2", {}).get("executed"), \
        "step_local_2 did not execute after passthrough"


@pytest.mark.integration
def test_metadata_tamper(wait_for_results):
    """Step modifies metadata; next step still works."""
    yaml_path = str(BASE / "pipelines" / "test_metadata_tamper_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"original": True})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    assert r.get("step_metadata_tamper", {}).get("executed"), \
        "tamper step did not execute"
    assert r.get("step_local_2", {}).get("executed"), \
        "step after tamper did not execute"


@pytest.mark.integration
def test_adaptive_microscopy_pattern(wait_for_results):
    """Simulates the full adaptive microscopy loop:
    submit tiles -> poll results -> get feedback -> submit more."""
    tmp = tempfile.mkdtemp()
    try:
        # Create simple step files
        Path(tmp, "preprocess.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["processed"] = pd["input"]["tile_id"]
                return pd
        """))
        Path(tmp, "analyze.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                tiles = [r["processed"] for r in pd["results"]]
                pd["feedback"] = {"interesting": tiles[0] if tiles else None,
                                  "n_tiles": len(tiles)}
                return pd
        """))
        yaml_content = textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            overview:
              - preprocess:
              - analyze:
                  scope: group
        """)
        yaml_path = Path(tmp, "overview.yaml")
        yaml_path.write_text(yaml_content)

        with Engine() as e:
            e.register("overview", str(yaml_path))

            # Round 1: submit tiles for region R1
            for i in range(4):
                is_last = (i == 3)
                e.submit("overview", {"tile_id": f"R1_t{i}"},
                         scope={"group": "R1"},
                         complete="group" if is_last else None)

            # Poll for feedback
            results = wait_for_results(e, "overview", 5, timeout=15)
            scoped = [r for r in results if r.get("_phase") == 1]

            assert scoped, "no feedback from round 1"

            feedback = scoped[0].get("feedback", {})
            assert feedback.get("n_tiles") == 4, \
                f"round 1: expected 4 tiles, got {feedback.get('n_tiles')}"

            # Round 2: submit more tiles based on feedback (adaptive)
            for i in range(2):
                is_last = (i == 1)
                e.submit("overview", {"tile_id": f"R2_t{i}"},
                         scope={"group": "R2"},
                         complete="group" if is_last else None)

            results2 = wait_for_results(e, "overview", 3, timeout=15)
            scoped2 = [r for r in results2 if r.get("_phase") == 1]

            assert scoped2, "no feedback from round 2"

            feedback2 = scoped2[0].get("feedback", {})
            assert feedback2.get("n_tiles") == 2, \
                f"round 2: expected 2 tiles, got {feedback2.get('n_tiles')}"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_multi_pipeline_concurrent(wait_for_results):
    """Two pipelines running concurrently on same engine."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "mark.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["pipeline"] = p.get("pipeline", "unknown")
                pd["idx"] = pd["input"]["idx"]
                return pd
        """))
        yaml_a = Path(tmp, "a.yaml")
        yaml_a.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf_a:
              - mark:
                  pipeline: a
        """))
        yaml_b = Path(tmp, "b.yaml")
        yaml_b.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf_b:
              - mark:
                  pipeline: b
        """))

        with Engine(max_concurrent=8) as e:
            e.register("a", str(yaml_a))
            e.register("b", str(yaml_b))

            for i in range(10):
                e.submit("a", {"idx": i})
                e.submit("b", {"idx": i})

            ra = wait_for_results(e, "a", 10, timeout=30)
            rb = wait_for_results(e, "b", 10, timeout=30)

        assert len(ra) == 10, f"pipeline a: {len(ra)}/10"
        assert len(rb) == 10, f"pipeline b: {len(rb)}/10"

        a_pipelines = set(r["pipeline"] for r in ra)
        b_pipelines = set(r["pipeline"] for r in rb)
        assert a_pipelines == {"a"} and b_pipelines == {"b"}, \
            "cross-contamination between pipelines"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_state_dict_warm_model(wait_for_results):
    """State dict persists across jobs (simulates warm ML model)."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "model_step.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                if "model" not in state:
                    state["model"] = "loaded"
                    state["load_count"] = 0
                state["load_count"] += 1
                pd["load_count"] = state["load_count"]
                pd["model_status"] = state["model"]
                return pd
        """))
        yaml_path = Path(tmp, "model.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - model_step:
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))
            for i in range(5):
                e.submit("test", {})
                time.sleep(0.3)  # sequential to hit same worker

            results = wait_for_results(e, "test", 5, timeout=15)

        assert len(results) == 5, f"only {len(results)}/5 results"

        counts = sorted(r["load_count"] for r in results)
        # Either single-worker (1..5) or multi-worker (any >=1) is acceptable
        assert all(c >= 1 for c in counts), f"unexpected counts: {counts}"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_chained_scopes(wait_for_results):
    """Pipeline with two scope levels: group then all."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "process.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["v"] = pd["input"]["v"]
                return pd
        """))
        Path(tmp, "group_agg.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["group_sum"] = sum(r["v"] for r in pd["results"])
                return pd
        """))
        Path(tmp, "final_agg.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["total"] = sum(r["group_sum"] for r in pd["results"])
                pd["n_groups"] = len(pd["results"])
                return pd
        """))
        yaml_path = Path(tmp, "chained.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            wf:
              - process:
              - group_agg:
                  scope: group
              - final_agg:
                  scope: all
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))

            # Group A: 1 + 2 = 3
            e.submit("test", {"v": 1}, scope={"group": "A"})
            e.submit("test", {"v": 2}, scope={"group": "A"},
                     complete="group")

            # Group B: 10 + 20 + 30 = 60
            e.submit("test", {"v": 10}, scope={"group": "B"})
            e.submit("test", {"v": 20}, scope={"group": "B"})
            e.submit("test", {"v": 30}, scope={"group": "B"},
                     complete=["group", "all"])

            # Expect: 5 phase0 + 2 phase1 + 1 phase2 = 8
            results = wait_for_results(e, "test", 8, timeout=30)

        phase2 = [r for r in results if r.get("_phase") == 2]
        assert phase2, "final aggregation did not trigger"

        r = phase2[0]
        assert r.get("n_groups") == 2, \
            f"expected 2 groups, got {r.get('n_groups')}"
        assert r.get("total") == 63, \
            f"expected total=63 (3+60), got {r.get('total')}"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_rare_event_selection_pattern(wait_for_results):
    """Multi-step pipeline inspired by rare_event_selection:
    preprocess -> analyze -> select -> feedback.
    Tests realistic data flow through 4 sequential steps."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "preprocess.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                sigma = p.get("sigma", 1.0)
                image = pd["input"]["image"]
                pd["preprocessed"] = {
                    "smoothed": [v * sigma for v in image],
                    "shape": len(image),
                }
                return pd
        """))
        Path(tmp, "segment.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                if "model" not in state:
                    state["model"] = "mock_cellpose"
                    state["call_count"] = 0
                state["call_count"] += 1
                data = pd["preprocessed"]["smoothed"]
                # Mock segmentation: values above threshold are "cells"
                threshold = p.get("threshold", 5)
                masks = [1 if v > threshold else 0 for v in data]
                pd["segmented"] = {
                    "masks": masks,
                    "n_cells": sum(masks),
                    "model": state["model"],
                    "call_count": state["call_count"],
                }
                return pd
        """))
        Path(tmp, "extract_features.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                masks = pd["segmented"]["masks"]
                percentile = p.get("percentile", 90)
                # Mock feature extraction
                features = [{"label": i, "area": i * 10, "selected": False}
                            for i, m in enumerate(masks) if m == 1]
                if features:
                    areas = sorted(f["area"] for f in features)
                    cutoff = areas[int(len(areas) * percentile / 100)]
                    for f in features:
                        f["selected"] = f["area"] >= cutoff
                pd["features"] = {
                    "cells": features,
                    "n_total": len(features),
                    "n_selected": sum(1 for f in features if f["selected"]),
                }
                return pd
        """))
        Path(tmp, "feedback.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                features = pd["features"]
                pd["feedback"] = {
                    "n_cells": features["n_total"],
                    "n_selected": features["n_selected"],
                    "selected_labels": [f["label"] for f in features["cells"]
                                        if f["selected"]],
                }
                return pd
        """))

        yaml_path = Path(tmp, "rare_event.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            rare-event:
              - preprocess:
                  sigma: 1.0
              - segment:
                  threshold: 3
              - extract_features:
                  percentile: 50
              - feedback:
        """))

        with Engine() as e:
            e.register("analysis", str(yaml_path))

            # Submit 3 "images"
            for i in range(3):
                image = list(range(1, 11))  # [1, 2, ..., 10]
                e.submit("analysis", {"image": image})

            results = wait_for_results(e, "analysis", 3, timeout=15)

        assert len(results) == 3, f"expected 3 results, got {len(results)}"

        # Verify the full pipeline ran
        for r in results:
            assert "preprocessed" in r, "preprocess did not run"
            assert "segmented" in r, "segment did not run"
            assert "features" in r, "extract_features did not run"
            assert "feedback" in r, "feedback did not run"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_scoped_rare_event_pattern(wait_for_results):
    """Rare event selection with scoped aggregation:
    per-tile analysis -> scoped aggregate across all tiles."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "analyze_tile.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                tile = pd["input"]
                n_cells = tile.get("n_cells", 0)
                pd["tile_result"] = {
                    "tile_id": tile["tile_id"],
                    "n_cells": n_cells,
                    "rare_cells": [i for i in range(n_cells) if i > 5],
                }
                return pd
        """))
        Path(tmp, "aggregate.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                results = pd["results"]
                all_rare = []
                total_cells = 0
                for r in results:
                    tr = r["tile_result"]
                    total_cells += tr["n_cells"]
                    all_rare.extend(tr["rare_cells"])
                pd["aggregate"] = {
                    "n_tiles": len(results),
                    "total_cells": total_cells,
                    "total_rare": len(all_rare),
                }
                return pd
        """))

        yaml_path = Path(tmp, "scoped_rare.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            scoped-rare:
              - analyze_tile:
              - aggregate:
                  scope: group
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))

            # Submit tiles for region R1
            tiles = [
                {"tile_id": "R1_t0", "n_cells": 10},
                {"tile_id": "R1_t1", "n_cells": 8},
                {"tile_id": "R1_t2", "n_cells": 12},
            ]
            for i, tile in enumerate(tiles):
                is_last = (i == len(tiles) - 1)
                e.submit("test", tile, scope={"group": "R1"},
                         complete="group" if is_last else None)

            results = wait_for_results(e, "test", 4, timeout=15)

        scoped = [r for r in results if r.get("_phase") == 1]
        assert scoped, "aggregate step did not trigger"

        agg = scoped[0].get("aggregate", {})
        assert agg.get("n_tiles") == 3, \
            f"expected 3 tiles, got {agg.get('n_tiles')}"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_overview_target_interleave(wait_for_results):
    """THE primary smart-microscopy use case:
    - Overview pipeline processes tiles, produces feedback at scope boundary
    - Target pipeline acts on feedback positions
    - Both run concurrently with different priorities
    - Overview is high priority (feedback needed fast)
    """
    tmp = tempfile.mkdtemp()
    try:
        # Overview steps
        Path(tmp, "ov_preprocess.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["tile_id"] = pd["input"]["tile_id"]
                pd["region"] = pd["input"]["region"]
                return pd
        """))
        Path(tmp, "ov_segment.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                if "model" not in state:
                    state["model"] = "cellpose_loaded"
                pd["cells"] = [{"x": i * 10, "y": i * 5}
                               for i in range(pd["input"].get("n_cells", 3))]
                pd["model_status"] = state["model"]
                return pd
        """))
        Path(tmp, "ov_feedback.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                results = pd["results"]
                all_cells = []
                for r in results:
                    for c in r.get("cells", []):
                        all_cells.append(c)
                interesting = [c for c in all_cells if c["x"] > 15]
                pd["feedback"] = {
                    "n_tiles": len(results),
                    "n_cells_total": len(all_cells),
                    "n_interesting": len(interesting),
                    "positions": interesting[:5],
                }
                return pd
        """))

        # Target steps
        Path(tmp, "tgt_acquire.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["target"] = {
                    "position": pd["input"]["position"],
                    "acquired": True,
                }
                return pd
        """))
        Path(tmp, "tgt_analyze.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                pd["analysis"] = {
                    "position": pd["target"]["position"],
                    "quality": "high",
                }
                return pd
        """))

        # YAMLs
        ov_yaml = Path(tmp, "overview.yaml")
        ov_yaml.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            overview:
              - ov_preprocess:
              - ov_segment:
              - ov_feedback:
                  scope: group
        """))

        tgt_yaml = Path(tmp, "target.yaml")
        tgt_yaml.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            target:
              - tgt_acquire:
              - tgt_analyze:
        """))

        with Engine(max_concurrent=8) as e:
            e.register("overview", str(ov_yaml))
            e.register("target", str(tgt_yaml))

            # Microscope acquires overview tiles for region R1
            for i in range(4):
                is_last = (i == 3)
                e.submit("overview",
                         {"tile_id": f"R1_t{i}", "region": "R1",
                          "n_cells": 5},
                         scope={"group": "R1"},
                         priority=10,
                         complete="group" if is_last else None)

            # Poll for overview feedback
            ov_results = wait_for_results(e, "overview", 5, timeout=15)
            ov_scoped = [r for r in ov_results if r.get("_phase") == 1]

            assert ov_scoped, "no overview feedback received"

            feedback = ov_scoped[0]["feedback"]

            # Act on feedback: submit targets at interesting positions
            for pos in feedback["positions"][:3]:
                e.submit("target", {"position": pos})

            tgt_results = wait_for_results(e, "target", 3, timeout=15)

            # Meanwhile, start overview for region R2
            for i in range(3):
                is_last = (i == 2)
                e.submit("overview",
                         {"tile_id": f"R2_t{i}", "region": "R2",
                          "n_cells": 4},
                         scope={"group": "R2"},
                         priority=10,
                         complete="group" if is_last else None)

            ov_results2 = wait_for_results(e, "overview", 4, timeout=15)
            ov_scoped2 = [r for r in ov_results2 if r.get("_phase") == 1]

        # Verify overview feedback
        assert feedback["n_tiles"] == 4, \
            f"R1 feedback: expected 4 tiles, got {feedback['n_tiles']}"

        # Verify targets were acquired
        assert len(tgt_results) >= 3, \
            f"expected 3 target results, got {len(tgt_results)}"
        for r in tgt_results:
            assert r.get("analysis", {}).get("quality"), \
                "target analysis missing"

        # Verify R2 feedback
        assert ov_scoped2, "no R2 feedback"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_max_workers_parallelism(wait_for_results):
    """5 workers process 20 tiles in parallel, then scoped step aggregates.
    Verifies that max_workers actually speeds up processing."""
    tmp = tempfile.mkdtemp()
    try:
        # Step with max_workers=5 and a small delay to measure parallelism
        Path(tmp, "process_tile.py").write_text(textwrap.dedent("""
            import time
            import os

            METADATA = {"max_workers": 5}

            def run(pd, state, **p):
                time.sleep(0.2)
                pd["tile_id"] = pd["input"]["tile_id"]
                pd["worker_pid"] = os.getpid()
                return pd
        """))
        Path(tmp, "stitch.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                results = pd["results"]
                pids = set(r["worker_pid"] for r in results)
                pd["stitched"] = {
                    "n_tiles": len(results),
                    "n_workers_used": len(pids),
                    "tile_ids": sorted(r["tile_id"] for r in results),
                }
                return pd
        """))

        yaml_path = Path(tmp, "parallel.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            parallel:
              - process_tile:
              - stitch:
                  scope: group
        """))

        with Engine(max_concurrent=8) as e:
            e.register("test", str(yaml_path))

            t0 = time.perf_counter()
            for i in range(20):
                is_last = (i == 19)
                e.submit("test", {"tile_id": f"t{i}"},
                         scope={"group": "batch"},
                         complete="group" if is_last else None)

            results = wait_for_results(e, "test", 21, timeout=30)
            elapsed = time.perf_counter() - t0

        scoped = [r for r in results if r.get("_phase") == 1]
        assert scoped, "stitch did not trigger"

        s = scoped[0]["stitched"]
        assert s["n_tiles"] == 20, f"expected 20 tiles, got {s['n_tiles']}"

        n_workers = s["n_workers_used"]
        sequential_time = 20 * 0.2  # 4.0s

        # The whole point of max_workers=5 is parallel execution. Verify it
        # actually happened: must be faster than sequential AND must have
        # used more than one worker process.
        assert elapsed < sequential_time, \
            (f"no parallelism: {elapsed:.1f}s "
             f">= sequential {sequential_time:.1f}s")
        assert n_workers >= 2, \
            (f"only {n_workers} worker PID(s) used; "
             f"max_workers=5 should yield at least 2")
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_partial_failure_at_scope(wait_for_results):
    """1 of 20 tiles fails; scoped step still runs with 19 results."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "tile_step.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                if pd["input"].get("corrupt"):
                    raise ValueError("corrupt tile data")
                pd["tile_id"] = pd["input"]["tile_id"]
                pd["value"] = pd["input"]["value"]
                return pd
        """))
        Path(tmp, "stitch_step.py").write_text(textwrap.dedent("""
            def run(pd, state, **p):
                results = pd["results"]
                failures = pd.get("failures", [])
                pd["stitched"] = {
                    "n_tiles": len(results),
                    "n_failures": len(failures),
                    "total_value": sum(r.get("value", 0) for r in results),
                    "tile_ids": sorted(r.get("tile_id", "?") for r in results),
                }
                return pd
        """))

        yaml_path = Path(tmp, "partial.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            partial:
              - tile_step:
              - stitch_step:
                  scope: group
        """))

        with Engine() as e:
            e.register("test", str(yaml_path))

            # Submit 10 tiles, 1 is corrupt
            for i in range(10):
                is_last = (i == 9)
                data = {"tile_id": f"t{i}", "value": i * 10}
                if i == 5:
                    data["corrupt"] = True
                e.submit("test", data,
                         scope={"group": "R1"},
                         complete="group" if is_last else None)

            results = wait_for_results(e, "test", 10, timeout=30)

        scoped = [r for r in results if r.get("_phase") == 1]
        assert scoped, "stitch did not trigger (failure may have blocked it)"

        s = scoped[0]["stitched"]
        assert s["n_tiles"] == 9, \
            f"expected 9 tiles (1 failed), got {s['n_tiles']}"
        assert s["n_failures"] >= 1, "no failures reported to scoped step"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_status_polling_loop():
    """Monitor engine via status() while jobs process (dashboard pattern)."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "slow_step.py").write_text(textwrap.dedent("""
            import time
            def run(pd, state, **p):
                time.sleep(0.1)
                pd["done"] = True
                return pd
        """))

        yaml_path = Path(tmp, "poll.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            poll:
              - slow_step:
        """))

        with Engine(max_concurrent=4) as e:
            e.register("test", str(yaml_path))

            for i in range(20):
                e.submit("test", {"idx": i})

            # Poll status until all complete
            statuses = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 30:
                s = e.status("test")
                statuses.append(s.copy())
                if s["pending"] == 0 and s["completed"] + s["failed"] >= 20:
                    break
                time.sleep(0.3)

            final = e.status("test")

        assert final["completed"] >= 20, \
            f"only {final['completed']}/20 completed"
    finally:
        shutil.rmtree(tmp, True)


@pytest.mark.integration
def test_large_batch_post_acquisition(wait_for_results):
    """100 images, no scopes, pure throughput (post-acquisition pattern)."""
    tmp = tempfile.mkdtemp()
    try:
        Path(tmp, "analyze.py").write_text(textwrap.dedent("""
            METADATA = {"max_workers": 4}
            def run(pd, state, **p):
                state.setdefault("count", 0)
                state["count"] += 1
                pd["idx"] = pd["input"]["idx"]
                pd["call_count"] = state["count"]
                return pd
        """))

        yaml_path = Path(tmp, "batch.yaml")
        yaml_path.write_text(textwrap.dedent(f"""
            metadata:
              functions_dir: "{Path(tmp).as_posix()}"
            batch:
              - analyze:
        """))

        with Engine(max_concurrent=8) as e:
            e.register("batch", str(yaml_path))

            for i in range(100):
                e.submit("batch", {"idx": i})

            results = wait_for_results(e, "batch", 100, timeout=60)

        assert len(results) == 100, f"only {len(results)}/100 completed"

        indices = sorted(r["idx"] for r in results)
        assert indices == list(range(100)), "missing or duplicate indices"
    finally:
        shutil.rmtree(tmp, True)
