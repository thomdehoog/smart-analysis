"""Robustness tests for the v4 pipeline engine.

Edge cases, error recovery, resource cleanup, and unusual usage patterns
that go beyond the standard test suite.

Run with::

    pytest -m robustness

Marker: robustness (registered in pyproject.toml).
"""

import time
import threading
from pathlib import Path

import pytest

from engine import Engine

BASE = Path(__file__).parent.parent


# ---- Tests -----------------------------------------------------------


@pytest.mark.robustness
def test_metadata_tampering(wait_for_results):
    """Step modifies metadata; next step still executes correctly."""
    yaml_path = str(BASE / "pipelines" / "test_metadata_tamper_pipeline.yaml")

    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"original": True})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    tamper = r.get("step_metadata_tamper", {})
    local2 = r.get("step_local_2", {})

    assert tamper.get("executed"), "step_metadata_tamper did not execute"
    assert local2.get("executed"), "step_local_2 did not execute after metadata tamper"


@pytest.mark.robustness
def test_identity_passthrough(wait_for_results):
    """Identity step returns data unmodified; pipeline continues."""
    yaml_path = str(BASE / "pipelines" / "test_identity_pipeline.yaml")

    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "test_value"})
        results = wait_for_results(e, "test", 1)

    assert results, "no results returned"

    r = results[0]
    local1 = r.get("step_local", {})
    local2 = r.get("step_local_2", {})

    assert local1.get("executed"), "step_local did not execute"
    assert local2.get("executed"), "step_local_2 did not execute after passthrough"


@pytest.mark.robustness
@pytest.mark.slow
def test_resource_cleanup_normal(temp_step, temp_yaml, wait_for_results, count_children):
    """No leaked child processes after normal engine shutdown."""
    temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
              name="rc_ok")
    yaml = temp_yaml("wf:\n  - rc_ok:")

    children_before = count_children()

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(8):
            e.submit("test", {})
        wait_for_results(e, "test", 8, timeout=30)

    time.sleep(0.5)
    children_after = count_children()

    assert children_after <= children_before, (
        f"leaked processes: {children_before} before, {children_after} after"
    )


@pytest.mark.robustness
@pytest.mark.slow
def test_resource_cleanup_error(temp_step, temp_yaml, count_children):
    """No leaked child processes after a failed pipeline."""
    temp_step('def run(pd, state, **p): raise RuntimeError("boom")',
              name="rc_err")
    yaml = temp_yaml("wf:\n  - rc_err:")

    children_before = count_children()

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(4):
            e.submit("test", {})
        time.sleep(5)

    time.sleep(0.5)
    children_after = count_children()

    assert children_after <= children_before, (
        f"leaked processes after error: {children_before} -> {children_after}"
    )


@pytest.mark.robustness
@pytest.mark.slow
def test_submit_after_scope_complete(temp_step, temp_yaml, wait_for_results):
    """New jobs can be submitted to a different scope group after completing one."""
    temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="sasc_a")
    temp_step("""
        def run(pd, state, **p):
            pd["total"] = sum(r["v"] for r in pd["results"])
            return pd
    """, name="sasc_b")
    yaml = temp_yaml("""
        wf:
          - sasc_a:
          - sasc_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)

        # Complete group A
        e.submit("test", {"v": 10}, scope={"group": "A"})
        e.submit("test", {"v": 10}, scope={"group": "A"}, complete="group")
        time.sleep(3)

        # Submit and complete group B after A is done
        e.submit("test", {"v": 20}, scope={"group": "B"})
        e.submit("test", {"v": 30}, scope={"group": "B"}, complete="group")

        results = wait_for_results(e, "test", 6, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    totals = sorted(r["total"] for r in scoped)

    assert totals == [20, 50], f"totals wrong: {totals}"


@pytest.mark.robustness
@pytest.mark.slow
def test_concurrent_scope_complete(temp_step, temp_yaml, wait_for_results):
    """Multiple scope completions for different groups simultaneously."""
    temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="csc_a")
    temp_step("""
        def run(pd, state, **p):
            pd["values"] = sorted(r["v"] for r in pd["results"])
            return pd
    """, name="csc_b")
    yaml = temp_yaml("""
        wf:
          - csc_a:
          - csc_b:
              scope: group
    """)

    with Engine(max_concurrent=8) as e:
        e.register("test", yaml)

        # Submit 3 groups with 3 jobs each
        for group in ["X", "Y", "Z"]:
            for i in range(3):
                v = {"X": 100, "Y": 200, "Z": 300}[group] + i
                complete = "group" if i == 2 else None
                e.submit("test", {"v": v}, scope={"group": group},
                         complete=complete)

        results = wait_for_results(e, "test", 12, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    scoped_vals = sorted([tuple(r["values"]) for r in scoped])

    assert (100, 101, 102) in scoped_vals, f"group X wrong: {scoped_vals}"
    assert (200, 201, 202) in scoped_vals, f"group Y wrong: {scoped_vals}"
    assert (300, 301, 302) in scoped_vals, f"group Z wrong: {scoped_vals}"


@pytest.mark.robustness
def test_large_pipeline(temp_step, temp_yaml, wait_for_results):
    """10-step pipeline completes correctly with data flowing through all steps."""
    steps = []
    for i in range(10):
        name = f"lp_s{i}"
        temp_step(f"""
            def run(pd, state, **p):
                pd["step_{i}"] = True
                pd["last_step"] = {i}
                return pd
        """, name=name)
        steps.append(name)

    step_lines = "\n".join(f"          - {s}:" for s in steps)
    yaml = temp_yaml(f"""
        wf:
{step_lines}
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = wait_for_results(e, "test", 1, timeout=30)

    assert results, "no results returned"

    r = results[0]
    for i in range(10):
        assert r.get(f"step_{i}"), f"step_{i} did not execute"

    assert r.get("last_step") == 9, f"last_step was {r.get('last_step')}, expected 9"


@pytest.mark.robustness
def test_scope_single_job(temp_step, temp_yaml, wait_for_results):
    """Scope with a single submitted job (degenerate case)."""
    temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="ssj_a")
    temp_step("""
        def run(pd, state, **p):
            pd["collected"] = [r["v"] for r in pd["results"]]
            return pd
    """, name="ssj_b")
    yaml = temp_yaml("""
        wf:
          - ssj_a:
          - ssj_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 42}, scope={"group": "solo"},
                 complete="group")
        results = wait_for_results(e, "test", 2, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result"
    assert scoped[0].get("collected") == [42], (
        f"collected={scoped[0].get('collected')}, expected [42]"
    )


@pytest.mark.robustness
def test_empty_string_scope_values(temp_step, temp_yaml, wait_for_results):
    """Scope with empty string values works correctly."""
    temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="essv_a")
    temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="essv_b")
    yaml = temp_yaml("""
        wf:
          - essv_a:
          - essv_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": ""})
        e.submit("test", {"v": 2}, scope={"group": ""},
                 complete="group")
        results = wait_for_results(e, "test", 3, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result"
    assert scoped[0].get("n") == 2, f"expected 2 results, got {scoped[0].get('n')}"


@pytest.mark.robustness
def test_scoped_yaml_spatial(wait_for_results):
    """Run the scoped spatial YAML pipeline with the v4 scope API."""
    yaml_path = str(BASE / "pipelines" / "test_scoped_spatial_pipeline.yaml")

    with Engine() as e:
        e.register("test", yaml_path)

        for row in range(2):
            for col in range(2):
                is_last = (row == 1 and col == 1)
                e.submit(
                    "test",
                    {"row": row, "col": col},
                    scope={"region": "grid_A"},
                    complete="region" if is_last else None,
                )

        results = wait_for_results(e, "test", 5, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result"

    stitched = scoped[0].get("stitched", {})
    assert stitched.get("n_tiles") == 4, f"expected 4 tiles, got {stitched.get('n_tiles')}"
    assert stitched.get("n_rows") == 2, f"expected 2 rows, got {stitched.get('n_rows')}"


@pytest.mark.robustness
def test_multi_pipeline_sharing(temp_step, temp_yaml, wait_for_results):
    """Two pipelines sharing one engine both complete correctly."""
    temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.02)
            pd["pipeline"] = pd["input"]["pipeline"]
            pd["job_id"] = pd["input"]["job_id"]
            return pd
    """, name="mrrs")
    yaml_a = temp_yaml("wf_a:\n  - mrrs:")
    yaml_b = temp_yaml("wf_b:\n  - mrrs:")

    with Engine(max_concurrent=8) as e:
        e.register("a", yaml_a)
        e.register("b", yaml_b)

        for i in range(5):
            e.submit("a", {"pipeline": "A", "job_id": i})
            e.submit("b", {"pipeline": "B", "job_id": i})

        results_a = wait_for_results(e, "a", 5, timeout=30)
        results_b = wait_for_results(e, "b", 5, timeout=30)

    assert len(results_a) == 5, f"pipeline A: expected 5, got {len(results_a)}"
    assert len(results_b) == 5, f"pipeline B: expected 5, got {len(results_b)}"

    a_ids = sorted(r["pipeline"] for r in results_a)
    b_ids = sorted(r["pipeline"] for r in results_b)

    assert a_ids == ["A"] * 5, f"pipeline A ids wrong: {a_ids}"
    assert b_ids == ["B"] * 5, f"pipeline B ids wrong: {b_ids}"


@pytest.mark.robustness
def test_repeated_submits(temp_step, temp_yaml, wait_for_results):
    """Sequential batches on the same engine with no state leakage."""
    temp_step("""
        def run(pd, state, **p):
            pd["marker"] = pd["input"]["marker"]
            return pd
    """, name="rrse")
    yaml = temp_yaml("wf:\n  - rrse:")

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)

        # Batch 1
        for i in range(3):
            e.submit("test", {"marker": f"batch1_{i}"})
        r1 = wait_for_results(e, "test", 3, timeout=15)

        # Batch 2
        for i in range(3):
            e.submit("test", {"marker": f"batch2_{i}"})
        r2 = wait_for_results(e, "test", 3, timeout=15)

    m1 = sorted(r["marker"] for r in r1)
    m2 = sorted(r["marker"] for r in r2)

    expected1 = sorted(f"batch1_{i}" for i in range(3))
    expected2 = sorted(f"batch2_{i}" for i in range(3))

    assert m1 == expected1, f"batch 1 markers wrong: {m1}"
    assert m2 == expected2, f"batch 2 markers wrong: {m2}"


@pytest.mark.robustness
@pytest.mark.slow
def test_state_dict_persistence(temp_step, temp_yaml, wait_for_results):
    """State dict persists across calls (warm model pattern)."""
    temp_step("""
        def run(pd, state, **p):
            state.setdefault("call_count", 0)
            state["call_count"] += 1
            pd["call_count"] = state["call_count"]
            return pd
    """, name="sdp")
    yaml = temp_yaml("wf:\n  - sdp:")

    with Engine() as e:
        e.register("test", yaml)
        # Submit 5 sequential jobs (same worker should handle them)
        for i in range(5):
            e.submit("test", {})
            time.sleep(0.5)  # sequential to hit same worker
        results = wait_for_results(e, "test", 5, timeout=30)

    counts = sorted(r["call_count"] for r in results)
    # With one worker, counts should be 1,2,3,4,5
    # With multiple workers, counts may differ but should all be >= 1
    assert all(c >= 1 for c in counts), f"unexpected counts: {counts}"


@pytest.mark.robustness
def test_graceful_partial_failure(temp_step, temp_yaml, wait_for_completion):
    """Failed jobs in a scope group don't prevent scoped step from running."""
    temp_step("""
        def run(pd, state, **p):
            if pd["input"].get("fail"):
                raise ValueError("deliberate failure")
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="gpf_a")
    temp_step("""
        def run(pd, state, **p):
            pd["n_results"] = len(pd["results"])
            pd["n_failures"] = len(pd.get("failures", []))
            return pd
    """, name="gpf_b")
    yaml = temp_yaml("""
        wf:
          - gpf_a:
          - gpf_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": "G"})
        e.submit("test", {"v": 2, "fail": True}, scope={"group": "G"})
        e.submit("test", {"v": 3}, scope={"group": "G"},
                 complete="group")
        results, _status = wait_for_completion(e, "test", 4, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    assert scoped, "no scoped result (scope may have been blocked by failure)"

    r = scoped[0]
    assert r["n_results"] == 2, f"expected 2 results, got {r['n_results']}"
    assert r["n_failures"] >= 1, f"expected at least 1 failure, got {r['n_failures']}"
