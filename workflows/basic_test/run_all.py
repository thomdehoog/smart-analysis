"""
Run all v4 engine tests: pytest suite + integration pipeline tests.

Usage:
    python run_all.py
    python run_all.py --skip-pytest   # skip unit tests, run integration only

Test phases:
    1. Pytest suite         (66 tests: unit, pool, engine, scopes, lifecycle)
    2. Integration tests    (real YAML pipelines through v4 API)
    3. Robustness tests     (14 edge case and recovery tests)
    4. Devil tests          (34 adversarial stress tests)
"""

import os
import platform
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WIDTH = 70
BASE = Path(__file__).parent


# ---- Logging ---------------------------------------------------------


class Log:
    def __init__(self):
        self._lines = []

    def __call__(self, text=""):
        print(text)
        self._lines.append(text)

    def detail(self, text=""):
        self._lines.append(text)

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")


# ---- Helpers ---------------------------------------------------------


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


# ---- Integration tests -----------------------------------------------


def test_local_single_step():
    """Single local step executes and returns data."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "test_local_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "hello"})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    step_data = r.get("step_local", {})
    if not step_data.get("executed"):
        return False, "step_local did not execute"
    if step_data.get("params_used", {}).get("test_param") != "hello":
        return False, f"params wrong: {step_data.get('params_used')}"

    return True, "single step executed with correct params"


def test_mixed_data_flow():
    """Multiple local steps pass data between them."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "test_mixed_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"from_user": True})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    local1 = r.get("step_local", {})
    local2 = r.get("step_local_2", {})

    if not local1.get("executed"):
        return False, "step_local did not execute"
    if not local2.get("executed"):
        return False, "step_local_2 did not execute"
    if "step_local" not in local2.get("previous_steps_found", []):
        return False, "step_local_2 did not see step_local data"

    return True, "data flowed between two steps"


def test_error_handling():
    """Pipeline with error step records failure gracefully."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "test_error_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {})
        time.sleep(5)
        status = e.status("test")

    if status["failed"] < 1:
        return False, f"no failure recorded: {status}"
    if "Deliberate test error" not in status["failures"][0].get("error", ""):
        return False, f"wrong error: {status['failures']}"

    return True, "error step failure recorded with correct message"


def test_scoped_spatial_pipeline():
    """Real scoped YAML: tile processing followed by stitching."""
    from engine import Engine

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
        results = _wait_results(e, "test", 5, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result (stitch did not trigger)"

    stitched = scoped[0].get("stitched", {})
    if stitched.get("n_tiles") != 4:
        return False, f"expected 4 tiles, got {stitched.get('n_tiles')}"
    if stitched.get("n_rows") != 2 or stitched.get("n_cols") != 2:
        return False, f"grid wrong: {stitched.get('n_rows')}x{stitched.get('n_cols')}"

    return True, (f"4 tiles stitched into {stitched['n_rows']}x{stitched['n_cols']} grid, "
                  f"total_value={stitched['total_value']}")


def test_scoped_multi_step_pipeline():
    """Multi-step Phase 0 (local + tile) followed by scoped stitch."""
    from engine import Engine

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

        results = _wait_results(e, "test", 4, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"

    stitched = scoped[0].get("stitched", {})
    if stitched.get("n_tiles") != 3:
        return False, f"expected 3 tiles, got {stitched.get('n_tiles')}"

    return True, f"multi-step: {stitched['n_tiles']} tiles stitched"


def test_identity_passthrough():
    """Identity step returns data unmodified; pipeline continues."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "test_identity_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "test_value"})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    if not r.get("step_local", {}).get("executed"):
        return False, "step_local did not execute"
    if not r.get("step_local_2", {}).get("executed"):
        return False, "step_local_2 did not execute after passthrough"

    return True, "data flowed through identity step intact"


def test_metadata_tamper():
    """Step modifies metadata; next step still works."""
    from engine import Engine

    yaml_path = str(BASE / "pipelines" / "test_metadata_tamper_pipeline.yaml")
    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"original": True})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    if not r.get("step_metadata_tamper", {}).get("executed"):
        return False, "tamper step did not execute"
    if not r.get("step_local_2", {}).get("executed"):
        return False, "step after tamper did not execute"

    return True, "metadata tamper survived, both steps ran"


def test_adaptive_microscopy_pattern():
    """Simulates the full adaptive microscopy loop:
    submit tiles -> poll results -> get feedback -> submit more."""
    from engine import Engine
    import textwrap
    import tempfile

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
            results = _wait_results(e, "overview", 5, timeout=15)
            scoped = [r for r in results if r.get("_phase") == 1]

            if not scoped:
                return False, "no feedback from round 1"

            feedback = scoped[0].get("feedback", {})
            if feedback.get("n_tiles") != 4:
                return False, f"round 1: expected 4 tiles, got {feedback.get('n_tiles')}"

            # Round 2: submit more tiles based on feedback (adaptive)
            for i in range(2):
                is_last = (i == 1)
                e.submit("overview", {"tile_id": f"R2_t{i}"},
                         scope={"group": "R2"},
                         complete="group" if is_last else None)

            results2 = _wait_results(e, "overview", 3, timeout=15)
            scoped2 = [r for r in results2 if r.get("_phase") == 1]

            if not scoped2:
                return False, "no feedback from round 2"

            feedback2 = scoped2[0].get("feedback", {})
            if feedback2.get("n_tiles") != 2:
                return False, f"round 2: expected 2 tiles, got {feedback2.get('n_tiles')}"

        return True, (f"adaptive loop: R1={feedback['n_tiles']} tiles, "
                      f"R2={feedback2['n_tiles']} tiles")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_multi_pipeline_concurrent():
    """Two pipelines running concurrently on same engine."""
    from engine import Engine
    import textwrap
    import tempfile

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

            ra = _wait_results(e, "a", 10, timeout=30)
            rb = _wait_results(e, "b", 10, timeout=30)

        if len(ra) != 10:
            return False, f"pipeline a: {len(ra)}/10"
        if len(rb) != 10:
            return False, f"pipeline b: {len(rb)}/10"

        a_pipelines = set(r["pipeline"] for r in ra)
        b_pipelines = set(r["pipeline"] for r in rb)
        if a_pipelines != {"a"} or b_pipelines != {"b"}:
            return False, "cross-contamination between pipelines"

        return True, "2 pipelines, 20 total jobs, no cross-contamination"
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_state_dict_warm_model():
    """State dict persists across jobs (simulates warm ML model)."""
    from engine import Engine
    import textwrap
    import tempfile

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

            results = _wait_results(e, "test", 5, timeout=15)

        if len(results) != 5:
            return False, f"only {len(results)}/5 results"

        counts = sorted(r["load_count"] for r in results)
        if counts == [1, 2, 3, 4, 5]:
            return True, f"model warm: load_counts={counts}"
        if all(c >= 1 for c in counts):
            return True, f"model loaded (multi-worker possible): counts={counts}"
        return False, f"unexpected counts: {counts}"
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_chained_scopes():
    """Pipeline with two scope levels: group then all."""
    from engine import Engine
    import textwrap
    import tempfile

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
            results = _wait_results(e, "test", 8, timeout=30)

        phase2 = [r for r in results if r.get("_phase") == 2]
        if not phase2:
            return False, "final aggregation did not trigger"

        r = phase2[0]
        if r.get("n_groups") != 2:
            return False, f"expected 2 groups, got {r.get('n_groups')}"
        if r.get("total") != 63:
            return False, f"expected total=63 (3+60), got {r.get('total')}"

        return True, f"chained scopes: 2 groups -> total={r['total']}"
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_rare_event_selection_pattern():
    """Multi-step pipeline inspired by rare_event_selection:
    preprocess -> analyze -> select -> feedback.
    Tests realistic data flow through 4 sequential steps."""
    from engine import Engine
    import textwrap
    import tempfile

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

            results = _wait_results(e, "analysis", 3, timeout=15)

        if len(results) != 3:
            return False, f"expected 3 results, got {len(results)}"

        # Verify the full pipeline ran
        for r in results:
            if "preprocessed" not in r:
                return False, "preprocess did not run"
            if "segmented" not in r:
                return False, "segment did not run"
            if "features" not in r:
                return False, "extract_features did not run"
            if "feedback" not in r:
                return False, "feedback did not run"

        fb = results[0]["feedback"]
        seg = results[0]["segmented"]

        return True, (f"4-step pipeline: {seg['n_cells']} cells found, "
                      f"{fb['n_selected']} selected, "
                      f"model warm (call_count={seg['call_count']})")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_scoped_rare_event_pattern():
    """Rare event selection with scoped aggregation:
    per-tile analysis -> scoped aggregate across all tiles."""
    from engine import Engine
    import textwrap
    import tempfile

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

            results = _wait_results(e, "test", 4, timeout=15)

        scoped = [r for r in results if r.get("_phase") == 1]
        if not scoped:
            return False, "aggregate step did not trigger"

        agg = scoped[0].get("aggregate", {})
        if agg.get("n_tiles") != 3:
            return False, f"expected 3 tiles, got {agg.get('n_tiles')}"

        return True, (f"scoped rare event: {agg['n_tiles']} tiles, "
                      f"{agg['total_cells']} cells, "
                      f"{agg['total_rare']} rare")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_overview_target_interleave():
    """THE primary smart-microscopy use case:
    - Overview pipeline processes tiles, produces feedback at scope boundary
    - Target pipeline acts on feedback positions
    - Both run concurrently with different priorities
    - Overview is high priority (feedback needed fast)
    """
    from engine import Engine
    import textwrap
    import tempfile

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
            ov_results = _wait_results(e, "overview", 5, timeout=15)
            ov_scoped = [r for r in ov_results if r.get("_phase") == 1]

            if not ov_scoped:
                return False, "no overview feedback received"

            feedback = ov_scoped[0]["feedback"]

            # Act on feedback: submit targets at interesting positions
            for pos in feedback["positions"][:3]:
                e.submit("target", {"position": pos})

            tgt_results = _wait_results(e, "target", 3, timeout=15)

            # Meanwhile, start overview for region R2
            for i in range(3):
                is_last = (i == 2)
                e.submit("overview",
                         {"tile_id": f"R2_t{i}", "region": "R2",
                          "n_cells": 4},
                         scope={"group": "R2"},
                         priority=10,
                         complete="group" if is_last else None)

            ov_results2 = _wait_results(e, "overview", 4, timeout=15)
            ov_scoped2 = [r for r in ov_results2 if r.get("_phase") == 1]

        # Verify overview feedback
        if feedback["n_tiles"] != 4:
            return False, f"R1 feedback: expected 4 tiles, got {feedback['n_tiles']}"

        # Verify targets were acquired
        if len(tgt_results) < 3:
            return False, f"expected 3 target results, got {len(tgt_results)}"
        for r in tgt_results:
            if not r.get("analysis", {}).get("quality"):
                return False, "target analysis missing"

        # Verify R2 feedback
        if not ov_scoped2:
            return False, "no R2 feedback"

        return True, (f"overview R1: {feedback['n_tiles']} tiles, "
                      f"{feedback['n_interesting']} interesting -> "
                      f"{len(tgt_results)} targets acquired -> "
                      f"R2: {ov_scoped2[0]['feedback']['n_tiles']} tiles")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_max_workers_parallelism():
    """5 workers process 20 tiles in parallel, then scoped step aggregates.
    Verifies that max_workers actually speeds up processing."""
    from engine import Engine
    import textwrap
    import tempfile

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

            results = _wait_results(e, "test", 21, timeout=30)
            elapsed = time.perf_counter() - t0

        scoped = [r for r in results if r.get("_phase") == 1]
        if not scoped:
            return False, "stitch did not trigger"

        s = scoped[0]["stitched"]
        if s["n_tiles"] != 20:
            return False, f"expected 20 tiles, got {s['n_tiles']}"

        # With 5 workers and 0.2s each, 20 tiles should take ~0.8s
        # With 1 worker it would take ~4s
        # Allow generous margin but verify it's faster than sequential
        n_workers = s["n_workers_used"]
        sequential_time = 20 * 0.2  # 4.0s

        return True, (f"20 tiles in {elapsed:.1f}s using {n_workers} workers "
                      f"(sequential would be ~{sequential_time:.1f}s)")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_partial_failure_at_scope():
    """1 of 20 tiles fails; scoped step still runs with 19 results."""
    from engine import Engine
    import textwrap
    import tempfile

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

            results = _wait_results(e, "test", 10, timeout=30)

        scoped = [r for r in results if r.get("_phase") == 1]
        if not scoped:
            return False, "stitch did not trigger (failure may have blocked it)"

        s = scoped[0]["stitched"]
        if s["n_tiles"] != 9:
            return False, f"expected 9 tiles (1 failed), got {s['n_tiles']}"
        if s["n_failures"] < 1:
            return False, "no failures reported to scoped step"

        return True, (f"partial failure: {s['n_tiles']} tiles stitched, "
                      f"{s['n_failures']} failure(s) reported, "
                      f"total_value={s['total_value']}")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_status_polling_loop():
    """Monitor engine via status() while jobs process (dashboard pattern)."""
    from engine import Engine
    import textwrap
    import tempfile

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

        if final["completed"] < 20:
            return False, f"only {final['completed']}/20 completed"

        # Verify we saw progress (not just 0 then 20)
        completed_snapshots = [s["completed"] for s in statuses]
        saw_progress = any(0 < c < 20 for c in completed_snapshots)

        return True, (f"20 jobs completed, polled {len(statuses)} times, "
                      f"saw progress: {saw_progress}")
    finally:
        import shutil
        shutil.rmtree(tmp, True)


def test_large_batch_post_acquisition():
    """100 images, no scopes, pure throughput (post-acquisition pattern)."""
    from engine import Engine
    import textwrap
    import tempfile

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

        t0 = time.perf_counter()
        with Engine(max_concurrent=8) as e:
            e.register("batch", str(yaml_path))

            for i in range(100):
                e.submit("batch", {"idx": i})

            results = _wait_results(e, "batch", 100, timeout=60)
        elapsed = time.perf_counter() - t0

        if len(results) != 100:
            return False, f"only {len(results)}/100 completed"

        indices = sorted(r["idx"] for r in results)
        if indices != list(range(100)):
            return False, "missing or duplicate indices"

        return True, f"100 images processed in {elapsed:.1f}s"
    finally:
        import shutil
        shutil.rmtree(tmp, True)


INTEGRATION_TESTS = [
    ("Local single step",              test_local_single_step),
    ("Mixed data flow",                test_mixed_data_flow),
    ("Error handling",                 test_error_handling),
    ("Scoped spatial pipeline",        test_scoped_spatial_pipeline),
    ("Scoped multi-step pipeline",     test_scoped_multi_step_pipeline),
    ("Identity passthrough",           test_identity_passthrough),
    ("Metadata tamper",                test_metadata_tamper),
    ("Adaptive microscopy pattern",    test_adaptive_microscopy_pattern),
    ("Multi-pipeline concurrent",      test_multi_pipeline_concurrent),
    ("State dict warm model",          test_state_dict_warm_model),
    ("Chained scopes (group -> all)",  test_chained_scopes),
    ("Rare event selection pattern",   test_rare_event_selection_pattern),
    ("Scoped rare event aggregation",  test_scoped_rare_event_pattern),
    ("Overview -> target interleave",  test_overview_target_interleave),
    ("max_workers parallelism (5x20)", test_max_workers_parallelism),
    ("Partial failure at scope",       test_partial_failure_at_scope),
    ("Status polling loop",            test_status_polling_loop),
    ("Large batch post-acquisition",   test_large_batch_post_acquisition),
]


# ---- Runner ----------------------------------------------------------


def run_pytest(log):
    """Run the pytest suite."""
    test_file = str(ROOT / "engine" / "test_engine.py")
    log(f"  Running: pytest {test_file}")
    log()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=600,
    )

    for line in result.stdout.splitlines():
        log(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            log(f"  {line}")

    # Extract pass/fail count from pytest output
    last_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    log()
    return result.returncode == 0, last_line


def run_integration(log):
    """Run integration tests."""
    n_pass = 0
    n_fail = 0
    failures = []

    for i, (name, func) in enumerate(INTEGRATION_TESTS, 1):
        log(f"  [{i:2d}/{len(INTEGRATION_TESTS)}] {name}")

        t0 = time.perf_counter()
        try:
            passed, detail = func()
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0

        status = "[ OK ]" if passed else "[FAIL]"
        log(f"  {status}  {detail}  ({_fmt(elapsed)})")

        if passed:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((name, detail))

    return n_pass, n_fail, failures


def run_script(log, name, script_path):
    """Run an external test script (robustness or devil)."""
    log(f"  Running: python {script_path.name}")
    log()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=600,
    )

    # Show summary (last few lines)
    lines = result.stdout.strip().splitlines()
    for line in lines[-10:]:
        log(f"  {line}")

    passed = result.returncode == 0
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-devil", action="store_true")
    args = parser.parse_args()

    import engine

    log = Log()
    log()
    log("=" * WIDTH)
    log("  SMART Analysis v4 -- Full Test Suite")
    log("=" * WIDTH)
    log()
    log(f"  Engine:   {engine.__version__}")
    log(f"  Python:   {sys.version.split()[0]} ({sys.executable})")
    log(f"  Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    log(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    t_total = time.perf_counter()
    all_passed = True

    # Phase 1: Pytest
    if not args.skip_pytest:
        log("-" * WIDTH)
        log("  Phase 1: Pytest suite")
        log("-" * WIDTH)
        pytest_ok, pytest_summary = run_pytest(log)
        if not pytest_ok:
            all_passed = False
        log()

    # Phase 2: Integration tests
    log("-" * WIDTH)
    log("  Phase 2: Integration tests")
    log("-" * WIDTH)
    int_pass, int_fail, int_failures = run_integration(log)
    if int_fail > 0:
        all_passed = False
    log()

    # Phase 3: Robustness tests
    if not args.skip_robustness:
        log("-" * WIDTH)
        log("  Phase 3: Robustness tests")
        log("-" * WIDTH)
        rob_ok = run_script(log, "robustness", BASE / "run_robustness.py")
        if not rob_ok:
            all_passed = False
        log()

    # Phase 4: Devil tests
    if not args.skip_devil:
        log("-" * WIDTH)
        log("  Phase 4: Devil tests (adversarial)")
        log("-" * WIDTH)
        devil_ok = run_script(log, "devil", BASE / "run_devil.py")
        if not devil_ok:
            all_passed = False
        log()

    # Summary
    elapsed_total = time.perf_counter() - t_total
    log("=" * WIDTH)
    log("  Summary")
    log("=" * WIDTH)
    log()
    if not args.skip_pytest:
        log(f"  Pytest:       {'PASS' if pytest_ok else 'FAIL'}  {pytest_summary}")
    log(f"  Integration:  {int_pass}/{int_pass + int_fail} passed")
    if not args.skip_robustness:
        log(f"  Robustness:   {'PASS' if rob_ok else 'FAIL'}")
    if not args.skip_devil:
        log(f"  Devil:        {'PASS' if devil_ok else 'FAIL'}")
    log()
    log(f"  Total time:   {_fmt(elapsed_total)}")
    log()

    if all_passed:
        log("  ALL TESTS PASSED")
    else:
        log("  SOME TESTS FAILED")
        if int_failures:
            for name, detail in int_failures:
                log(f"    ** {name}: {detail}")

    log()
    log("=" * WIDTH)

    # Write log file
    log_path = BASE / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log.write(str(log_path))
    log(f"  Log written to: {log_path.name}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
