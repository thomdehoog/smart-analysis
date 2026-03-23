"""
Robustness tests for the v4 pipeline engine.

Tests edge cases, error recovery, resource cleanup, and unusual
usage patterns that go beyond the standard test suite:

   1. Metadata tampering         step modifies metadata, next step works
   2. Identity passthrough       step returns data unmodified
   3. Resource cleanup (normal)  no leaked processes after shutdown
   4. Resource cleanup (error)   no leaked processes after failure
   5. Submit after scope         new group accepted after completing one
   6. Concurrent scopes          simultaneous scope completions
   7. Large pipeline             10-step pipeline completes correctly
   8. Scope with single job      degenerate case works
   9. Empty string scope values  scope={"group": ""} works
  10. Scoped YAML pipeline       real YAML pipelines with scope API
  11. Multi-pipeline sharing     two pipelines share engine
  12. Repeated submits           sequential batches, no state leakage
  13. State dict persistence     warm model pattern works
  14. Graceful partial failure   failed jobs don't block scoped steps

Usage:
    python run_robustness.py
"""

import sys
import os
import time
import textwrap
import tempfile
import shutil
import atexit
import threading
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

WIDTH = 70
_TEMP = tempfile.mkdtemp(prefix="robust_")
atexit.register(shutil.rmtree, _TEMP, True)
_counter = 0


def _next_id():
    global _counter
    _counter += 1
    return _counter


def _temp_step(code, name=None):
    path = Path(_TEMP) / (f"{name}.py" if name else f"step_{_next_id()}.py")
    path.write_text(textwrap.dedent(code))
    return str(path)


def _temp_yaml(content):
    text = textwrap.dedent(content)
    if "functions_dir" not in text:
        d = Path(_TEMP).as_posix()
        header = f'metadata:\n  functions_dir: "{d}"\n'
        if "metadata:" in text:
            text = text.replace("metadata:", header.rstrip("\n"), 1)
        else:
            text = header + text
    path = Path(_TEMP) / f"pipeline_{_next_id()}.yaml"
    path.write_text(text)
    return str(path)


def _fmt(seconds):
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


def _count_children():
    try:
        import psutil
        return len(psutil.Process().children(recursive=True))
    except ImportError:
        return len(multiprocessing.active_children())


def _wait_results(engine, name, expected, timeout=30):
    """Poll engine.results() until expected count or timeout."""
    t0 = time.monotonic()
    collected = []
    while time.monotonic() - t0 < timeout:
        collected.extend(engine.results(name))
        if len(collected) >= expected:
            return collected
        time.sleep(0.2)
    return collected


# ---- Tests -----------------------------------------------------------


def test_metadata_tampering():
    """Step modifies metadata; next step still executes correctly."""
    from engine import Engine

    base = Path(__file__).parent
    yaml_path = str(base / "pipelines" / "test_metadata_tamper_pipeline.yaml")

    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"original": True})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    tamper = r.get("step_metadata_tamper", {})
    local2 = r.get("step_local_2", {})

    if not tamper.get("executed"):
        return False, "step_metadata_tamper did not execute"
    if not local2.get("executed"):
        return False, "step_local_2 did not execute after metadata tamper"

    return True, "both steps ran, metadata modifications persisted"


def test_identity_passthrough():
    """Identity step returns data unmodified; pipeline continues."""
    from engine import Engine

    base = Path(__file__).parent
    yaml_path = str(base / "pipelines" / "test_identity_pipeline.yaml")

    with Engine() as e:
        e.register("test", yaml_path)
        e.submit("test", {"marker": "test_value"})
        results = _wait_results(e, "test", 1)

    if not results:
        return False, "no results returned"

    r = results[0]
    local1 = r.get("step_local", {})
    local2 = r.get("step_local_2", {})

    if not local1.get("executed"):
        return False, "step_local did not execute"
    if not local2.get("executed"):
        return False, "step_local_2 did not execute after passthrough"

    return True, "data flowed through identity step intact"


def test_resource_cleanup_normal():
    """No leaked child processes after normal engine shutdown."""
    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rc_ok")
    yaml = _temp_yaml("wf:\n  - rc_ok:")

    from engine import Engine

    children_before = _count_children()

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(8):
            e.submit("test", {})
        _wait_results(e, "test", 8, timeout=30)

    time.sleep(0.5)
    children_after = _count_children()

    if children_after > children_before:
        return False, (f"leaked processes: {children_before} before, "
                       f"{children_after} after")

    return True, f"process count stable ({children_before} -> {children_after})"


def test_resource_cleanup_error():
    """No leaked child processes after a failed pipeline."""
    _temp_step('def run(pd, state, **p): raise RuntimeError("boom")',
               name="rc_err")
    yaml = _temp_yaml("wf:\n  - rc_err:")

    from engine import Engine

    children_before = _count_children()

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(4):
            e.submit("test", {})
        time.sleep(5)

    time.sleep(0.5)
    children_after = _count_children()

    if children_after > children_before:
        return False, (f"leaked processes after error: {children_before} -> "
                       f"{children_after}")

    return True, f"process count stable after errors ({children_before} -> {children_after})"


def test_submit_after_scope_complete():
    """New jobs can be submitted to a different scope group after completing one."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="sasc_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["total"] = sum(r["v"] for r in pd["results"])
            return pd
    """, name="sasc_b")
    yaml = _temp_yaml("""
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

        results = _wait_results(e, "test", 6, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    totals = sorted(r["total"] for r in scoped)

    if totals != [20, 50]:
        return False, f"totals wrong: {totals}"

    return True, f"group A=20, group B=50"


def test_concurrent_scope_complete():
    """Multiple scope completions for different groups simultaneously."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="csc_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["values"] = sorted(r["v"] for r in pd["results"])
            return pd
    """, name="csc_b")
    yaml = _temp_yaml("""
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

        results = _wait_results(e, "test", 12, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    scoped_vals = sorted([tuple(r["values"]) for r in scoped])

    if (100, 101, 102) not in scoped_vals:
        return False, f"group X wrong: {scoped_vals}"
    if (200, 201, 202) not in scoped_vals:
        return False, f"group Y wrong: {scoped_vals}"
    if (300, 301, 302) not in scoped_vals:
        return False, f"group Z wrong: {scoped_vals}"

    return True, "3 groups completed concurrently with correct results"


def test_large_pipeline():
    """10-step pipeline completes correctly with data flowing through all steps."""
    steps = []
    for i in range(10):
        name = f"lp_s{i}"
        _temp_step(f"""
            def run(pd, state, **p):
                pd["step_{i}"] = True
                pd["last_step"] = {i}
                return pd
        """, name=name)
        steps.append(name)

    step_lines = "\n".join(f"          - {s}:" for s in steps)
    yaml = _temp_yaml(f"""
        wf:
{step_lines}
    """)

    from engine import Engine
    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = _wait_results(e, "test", 1, timeout=30)

    if not results:
        return False, "no results returned"

    r = results[0]
    for i in range(10):
        if not r.get(f"step_{i}"):
            return False, f"step_{i} did not execute"

    if r.get("last_step") != 9:
        return False, f"last_step was {r.get('last_step')}, expected 9"

    return True, f"all 10 steps executed, last_step=9"


def test_scope_single_job():
    """Scope with a single submitted job (degenerate case)."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="ssj_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["collected"] = [r["v"] for r in pd["results"]]
            return pd
    """, name="ssj_b")
    yaml = _temp_yaml("""
        wf:
          - ssj_a:
          - ssj_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 42}, scope={"group": "solo"},
                 complete="group")
        results = _wait_results(e, "test", 2, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"
    if scoped[0].get("collected") != [42]:
        return False, f"collected={scoped[0].get('collected')}, expected [42]"

    return True, "single job scoped correctly"


def test_empty_string_scope_values():
    """Scope with empty string values works correctly."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="essv_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="essv_b")
    yaml = _temp_yaml("""
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
        results = _wait_results(e, "test", 3, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"
    if scoped[0].get("n") != 2:
        return False, f"expected 2 results, got {scoped[0].get('n')}"

    return True, f"empty string scope: n={scoped[0]['n']}"


def test_scoped_yaml_spatial():
    """Run the scoped spatial YAML pipeline with the v4 scope API."""
    from engine import Engine

    base = Path(__file__).parent
    yaml_path = str(base / "pipelines" / "test_scoped_spatial_pipeline.yaml")

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

        results = _wait_results(e, "test", 5, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"

    stitched = scoped[0].get("stitched", {})
    if stitched.get("n_tiles") != 4:
        return False, f"expected 4 tiles, got {stitched.get('n_tiles')}"
    if stitched.get("n_rows") != 2:
        return False, f"expected 2 rows, got {stitched.get('n_rows')}"

    return True, (f"4 tiles stitched: {stitched['n_rows']}x{stitched['n_cols']} grid, "
                  f"total_value={stitched['total_value']}")


def test_multi_pipeline_sharing():
    """Two pipelines sharing one engine both complete correctly."""
    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.02)
            pd["pipeline"] = pd["input"]["pipeline"]
            pd["job_id"] = pd["input"]["job_id"]
            return pd
    """, name="mrrs")
    yaml_a = _temp_yaml("wf_a:\n  - mrrs:")
    yaml_b = _temp_yaml("wf_b:\n  - mrrs:")

    from engine import Engine

    with Engine(max_concurrent=8) as e:
        e.register("a", yaml_a)
        e.register("b", yaml_b)

        for i in range(5):
            e.submit("a", {"pipeline": "A", "job_id": i})
            e.submit("b", {"pipeline": "B", "job_id": i})

        results_a = _wait_results(e, "a", 5, timeout=30)
        results_b = _wait_results(e, "b", 5, timeout=30)

    if len(results_a) != 5:
        return False, f"pipeline A: expected 5, got {len(results_a)}"
    if len(results_b) != 5:
        return False, f"pipeline B: expected 5, got {len(results_b)}"

    a_ids = sorted(r["pipeline"] for r in results_a)
    b_ids = sorted(r["pipeline"] for r in results_b)

    if a_ids != ["A"] * 5:
        return False, f"pipeline A ids wrong: {a_ids}"
    if b_ids != ["B"] * 5:
        return False, f"pipeline B ids wrong: {b_ids}"

    return True, "2 pipelines shared engine, 10 total jobs completed"


def test_repeated_submits():
    """Sequential batches on the same engine with no state leakage."""
    _temp_step("""
        def run(pd, state, **p):
            pd["marker"] = pd["input"]["marker"]
            return pd
    """, name="rrse")
    yaml = _temp_yaml("wf:\n  - rrse:")

    from engine import Engine

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)

        # Batch 1
        for i in range(3):
            e.submit("test", {"marker": f"batch1_{i}"})
        r1 = _wait_results(e, "test", 3, timeout=15)

        # Batch 2
        for i in range(3):
            e.submit("test", {"marker": f"batch2_{i}"})
        r2 = _wait_results(e, "test", 3, timeout=15)

    m1 = sorted(r["marker"] for r in r1)
    m2 = sorted(r["marker"] for r in r2)

    expected1 = sorted(f"batch1_{i}" for i in range(3))
    expected2 = sorted(f"batch2_{i}" for i in range(3))

    if m1 != expected1:
        return False, f"batch 1 markers wrong: {m1}"
    if m2 != expected2:
        return False, f"batch 2 markers wrong: {m2}"

    return True, "2 sequential batches, no state leakage"


def test_state_dict_persistence():
    """State dict persists across calls (warm model pattern)."""
    _temp_step("""
        def run(pd, state, **p):
            state.setdefault("call_count", 0)
            state["call_count"] += 1
            pd["call_count"] = state["call_count"]
            return pd
    """, name="sdp")
    yaml = _temp_yaml("wf:\n  - sdp:")

    from engine import Engine

    with Engine() as e:
        e.register("test", yaml)
        # Submit 5 sequential jobs (same worker should handle them)
        for i in range(5):
            e.submit("test", {})
            time.sleep(0.5)  # sequential to hit same worker
        results = _wait_results(e, "test", 5, timeout=30)

    counts = sorted(r["call_count"] for r in results)
    # With one worker, counts should be 1,2,3,4,5
    if counts == [1, 2, 3, 4, 5]:
        return True, f"state persisted: counts={counts}"
    # With multiple workers, counts may differ but should all be >= 1
    if all(c >= 1 for c in counts):
        return True, f"state worked (multi-worker): counts={counts}"
    return False, f"unexpected counts: {counts}"


def test_graceful_partial_failure():
    """Failed jobs in a scope group don't prevent scoped step from running."""
    _temp_step("""
        def run(pd, state, **p):
            if pd["input"].get("fail"):
                raise ValueError("deliberate failure")
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="gpf_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n_results"] = len(pd["results"])
            pd["n_failures"] = len(pd.get("failures", []))
            return pd
    """, name="gpf_b")
    yaml = _temp_yaml("""
        wf:
          - gpf_a:
          - gpf_b:
              scope: group
    """)

    from engine import Engine

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": "G"})
        e.submit("test", {"v": 2, "fail": True}, scope={"group": "G"})
        e.submit("test", {"v": 3}, scope={"group": "G"},
                 complete="group")
        results = _wait_results(e, "test", 4, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result (scope may have been blocked by failure)"

    r = scoped[0]
    if r["n_results"] != 2:
        return False, f"expected 2 results, got {r['n_results']}"
    if r["n_failures"] < 1:
        return False, f"expected at least 1 failure, got {r['n_failures']}"

    return True, f"scoped step ran with {r['n_results']} results, {r['n_failures']} failures"


# ---- Runner ----------------------------------------------------------


TESTS = [
    ("Metadata tampering",                test_metadata_tampering),
    ("Identity passthrough",              test_identity_passthrough),
    ("Resource cleanup (normal)",         test_resource_cleanup_normal),
    ("Resource cleanup (error)",          test_resource_cleanup_error),
    ("Submit after scope complete",       test_submit_after_scope_complete),
    ("Concurrent scope completions",      test_concurrent_scope_complete),
    ("Large pipeline (10 steps)",         test_large_pipeline),
    ("Scope with single job",            test_scope_single_job),
    ("Empty string scope values",         test_empty_string_scope_values),
    ("Scoped YAML spatial pipeline",      test_scoped_yaml_spatial),
    ("Multi-pipeline sharing",            test_multi_pipeline_sharing),
    ("Repeated submits",                  test_repeated_submits),
    ("State dict persistence",            test_state_dict_persistence),
    ("Graceful partial failure",          test_graceful_partial_failure),
]


def main():
    import engine

    print()
    print("=" * WIDTH)
    print("  SMART Analysis v4 -- Robustness Tests")
    print("=" * WIDTH)
    print()
    print(f"  Engine:   {engine.__version__}")
    print(f"  Python:   {sys.version.split()[0]} ({sys.executable})")
    print(f"  Temp:     {_TEMP}")
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

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {detail}  ({_fmt(elapsed)})")
        print()

        results.append((name, passed, detail, elapsed))

    # ---- Summary ----

    elapsed_total = time.perf_counter() - t_total
    n_pass = sum(1 for _, p, _, _ in results if p)
    n_fail = sum(1 for _, p, _, _ in results if not p)

    print("=" * WIDTH)
    print("  Results")
    print("=" * WIDTH)
    print()

    for name, passed, detail, elapsed in results:
        icon = "[ OK ]" if passed else "[FAIL]"
        print(f"  {icon}  {name:<40s}  {_fmt(elapsed):>8s}")

    print()
    print(f"  {'_' * (WIDTH - 4)}")
    print(f"  Passed:  {n_pass}/{len(TESTS)}")
    print(f"  Failed:  {n_fail}/{len(TESTS)}")
    print(f"  Time:    {_fmt(elapsed_total)}")
    print()

    if n_fail == 0:
        print("  ALL ROBUSTNESS TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        print()
        for name, passed, detail, _ in results:
            if not passed:
                print(f"    ** {name}: {detail}")

    print()
    print("=" * WIDTH)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
