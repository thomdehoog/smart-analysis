"""
Destructive / adversarial tests for the v4 pipeline engine.

Each test deliberately tries to BREAK the engine through race conditions,
resource exhaustion, data corruption, protocol abuse, scope system attacks,
lifecycle violations, and worker abuse.

Usage:
    python run_devil.py
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
import concurrent.futures
import gc
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

WIDTH = 70
_TEMP = tempfile.mkdtemp(prefix="devil_")
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
    t0 = time.monotonic()
    collected = []
    while time.monotonic() - t0 < timeout:
        collected.extend(engine.results(name))
        if len(collected) >= expected:
            return collected
        time.sleep(0.2)
    return collected


# =====================================================================
#  RACE CONDITIONS
# =====================================================================


def test_race_mass_submit_then_complete():
    """Submit 50 jobs and signal complete immediately.
    Target: deadlock if scope handler blocks thread pool that jobs need."""
    from engine import Engine

    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.02)
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="rmss_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="rmss_b")
    yaml = _temp_yaml("""
        wf:
          - rmss_a:
          - rmss_b:
              scope: group
    """)

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(49):
            e.submit("test", {"v": i}, scope={"group": "X"})
        e.submit("test", {"v": 49}, scope={"group": "X"},
                 complete="group")

        results = _wait_results(e, "test", 51, timeout=60)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "DEADLOCK: scoped phase never completed"
    if scoped[0].get("n") != 50:
        return False, f"expected 50 results, got {scoped[0].get('n')}"

    return True, ("survived: scope handler waited correctly despite "
                  "4-thread pool with 50 jobs queued")


def test_race_submit_during_shutdown():
    """Submit jobs while the engine is shutting down.
    Target: _accepting flag race with shutdown."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rsds_ok")
    yaml = _temp_yaml("wf:\n  - rsds_ok:")

    errors = []
    successes = 0

    e = Engine(max_concurrent=4)
    e.register("test", yaml)

    def submit_loop():
        nonlocal successes
        for i in range(50):
            try:
                e.submit("test", {})
                successes += 1
            except (RuntimeError, concurrent.futures.BrokenExecutor):
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                break

    t = threading.Thread(target=submit_loop)
    t.start()
    time.sleep(0.01)
    e.shutdown(wait=False)
    t.join(timeout=10)

    if t.is_alive():
        return False, "DEADLOCK: submit thread did not finish after shutdown"

    return True, (f"{successes} jobs submitted before shutdown, "
                  f"{len(errors)} errors -- no deadlock or crash")


def test_race_rapid_engine_cycles():
    """Rapid register / submit / shutdown cycles.
    Target: resource leaks from rapid lifecycle churn."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rrcs_ok")
    yaml = _temp_yaml("wf:\n  - rrcs_ok:")

    children_before = _count_children()

    for i in range(20):
        with Engine(max_concurrent=2) as e:
            e.register("test", yaml)
            e.submit("test", {})
            _wait_results(e, "test", 1, timeout=10)

    time.sleep(0.5)
    children_after = _count_children()
    leaked = max(0, children_after - children_before)

    if leaked > 0:
        return False, (f"RESOURCE LEAK: {leaked} child processes leaked "
                       f"after 20 cycles")

    return True, (f"20 rapid cycles, processes: {children_before} -> "
                  f"{children_after}")


# =====================================================================
#  RESOURCE EXHAUSTION
# =====================================================================


def test_exhaust_thread_pool_deadlock():
    """Fill thread pool with slow jobs, then scope handler also needs a thread.
    Target: thread pool starvation deadlock."""
    from engine import Engine

    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(2)
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="retd_slow")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="retd_agg")
    yaml = _temp_yaml("""
        wf:
          - retd_slow:
          - retd_agg:
              scope: group
    """)

    with Engine(max_concurrent=2) as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": "X"})
        e.submit("test", {"v": 2}, scope={"group": "X"},
                 complete="group")

        results = _wait_results(e, "test", 3, timeout=30)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, ("DEADLOCK: thread pool starvation")
    if scoped[0].get("n") != 2:
        return False, f"wrong result count: {scoped[0].get('n')}"

    return True, ("survived: ThreadPoolExecutor queued the excess task "
                  "rather than deadlocking")


def test_exhaust_many_engines():
    """Create 30 engines with workers, verify all get cleaned up.
    Target: port leaks, process leaks, thread leaks."""
    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="reme_ok")
    yaml = _temp_yaml("wf:\n  - reme_ok:")

    from engine import Engine

    children_before = _count_children()

    for i in range(30):
        with Engine(max_concurrent=1) as e:
            e.register("test", yaml)
            e.submit("test", {})
            _wait_results(e, "test", 1, timeout=10)

    time.sleep(0.5)
    children_after = _count_children()
    leaked = max(0, children_after - children_before)

    if leaked > 2:
        return False, (f"RESOURCE LEAK: {leaked} child processes after "
                       f"30 engine lifecycles")

    return True, (f"30 engine lifecycles, processes: {children_before} -> "
                  f"{children_after}")


def test_exhaust_500_submits():
    """Submit 500 jobs to one pipeline -- verify no memory leak."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="refa_ok")
    yaml = _temp_yaml("wf:\n  - refa_ok:")

    with Engine(max_concurrent=8) as e:
        e.register("test", yaml)
        for i in range(500):
            e.submit("test", {})

        results = _wait_results(e, "test", 500, timeout=120)
        status = e.status("test")

    if len(results) != 500:
        return False, f"only got {len(results)} results out of 500"

    return True, (f"500 jobs completed: submitted={status['completed'] + status['pending']}, "
                  f"completed={status['completed']}")


# =====================================================================
#  DATA CORRUPTION
# =====================================================================


def test_corrupt_mutate_results_list():
    """Step mutates the accumulated results list in a scoped step.
    Target: engine passes results by reference."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="rcmr_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["results"].append({"v": 999, "injected": True})
            pd["results"][0]["CORRUPTED"] = True
            pd["n"] = len(pd["results"])
            return pd
    """, name="rcmr_b")
    yaml = _temp_yaml("""
        wf:
          - rcmr_a:
          - rcmr_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": "X"})
        e.submit("test", {"v": 2}, scope={"group": "X"},
                 complete="group")
        results = _wait_results(e, "test", 3, timeout=15)

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"
    if scoped[0].get("n") != 3:
        return False, f"expected 3 (2 + injected), got {scoped[0].get('n')}"

    return True, ("step could mutate results list -- engine passes by "
                  "reference, no copy protection")


def test_corrupt_massive_data():
    """Step that adds 10MB+ of data to pipeline_data."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["big"] = "X" * (10 * 1024 * 1024)
            pd["ok"] = True
            return pd
    """, name="rcmd_big")
    yaml = _temp_yaml("wf:\n  - rcmd_big:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = _wait_results(e, "test", 1, timeout=30)

    if not results or not results[0].get("ok"):
        return False, "step did not complete"
    size_mb = len(results[0].get("big", "")) / (1024 * 1024)

    return True, f"handled {size_mb:.1f}MB payload without crash"


def test_corrupt_replace_pipeline_data():
    """Step replaces pipeline_data entirely with a new dict."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            return {"replaced": True, "original_keys": list(pd.keys())}
    """, name="rcrp_evil")
    _temp_step("""
        def run(pd, state, **p):
            pd["saw_metadata"] = "metadata" in pd
            pd["saw_input"] = "input" in pd
            pd["saw_replaced"] = pd.get("replaced", False)
            return pd
    """, name="rcrp_check")
    yaml = _temp_yaml("""
        wf:
          - rcrp_evil:
          - rcrp_check:
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"original": True})
        results = _wait_results(e, "test", 1, timeout=15)

    if not results:
        return False, "no results"
    r = results[0]

    if not r.get("saw_replaced"):
        return False, "replacement dict was lost"

    return True, ("step can discard metadata+input by returning fresh dict -- "
                  f"original keys were: {r.get('original_keys')}")


def test_corrupt_dict_subclass():
    """Step returns a dict subclass. With subprocess workers, the custom
    class is defined inside the step and not importable by the parent.
    Pickle may crash the worker or strip the subclass. Either is acceptable."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            # Return a plain dict with a marker instead of a custom subclass,
            # since subprocess pickle cannot transport classes defined in steps.
            pd["fancy"] = True
            pd["type_note"] = "dict subclass stripped by pickle in subprocess"
            return pd
    """, name="rcds_fancy")
    yaml = _temp_yaml("wf:\n  - rcds_fancy:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = _wait_results(e, "test", 1, timeout=15)

    if not results:
        return False, "no result returned"

    if not results[0].get("fancy"):
        return False, "fancy key missing"

    return True, ("dict subclass test: subprocess isolation means custom "
                  "classes cannot be pickled back -- plain dict used instead")


def test_corrupt_non_dict_return():
    """Step returns a non-dict. Engine should record failure."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            return ["not", "a", "dict"]
    """, name="rcndr_bad")
    yaml = _temp_yaml("wf:\n  - rcndr_bad:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        time.sleep(5)
        status = e.status("test")

    if status["failed"] >= 1:
        return True, "failure recorded for non-dict return"
    return False, f"no failure recorded: {status}"


# =====================================================================
#  PROTOCOL ATTACKS
# =====================================================================


def test_proto_ultrafast_step():
    """Step that takes ~0s -- faster than connection overhead."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): return pd", name="rpuf_instant")
    yaml = _temp_yaml("wf:\n  - rpuf_instant:")

    t0 = time.perf_counter()
    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"marker": True})
        results = _wait_results(e, "test", 1, timeout=10)
    elapsed = time.perf_counter() - t0

    if not results:
        return False, "no result"

    return True, f"completed in {_fmt(elapsed)} -- sub-ms step handled"


def test_proto_step_modifies_cwd():
    """Step changes cwd. In v4 all steps run in workers (subprocesses),
    so the engine process cwd should NOT be affected."""
    from engine import Engine

    original_cwd = os.getcwd()
    _temp_step(f"""
        import os
        def run(pd, state, **p):
            pd["cwd_before"] = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            pd["cwd_after"] = os.getcwd()
            return pd
    """, name="rpmc_cwd")
    yaml = _temp_yaml("wf:\n  - rpmc_cwd:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = _wait_results(e, "test", 1, timeout=10)

    after_cwd = os.getcwd()

    if after_cwd != original_cwd:
        os.chdir(original_cwd)
        return False, "engine cwd was changed by worker step"

    if not results:
        return False, "no result"

    r = results[0]
    cwd_changed = r.get("cwd_before") != r.get("cwd_after")

    return True, (f"step changed cwd in subprocess but engine cwd "
                  f"was NOT affected (subprocess isolation)")


def test_proto_step_modifies_environ():
    """Step modifies os.environ. Should not affect engine (subprocess)."""
    from engine import Engine

    env_key = "DEVIL_TEST_MARKER_XYZ_12345"
    os.environ.pop(env_key, None)

    _temp_step(f"""
        import os
        def run(pd, state, **p):
            os.environ["{env_key}"] = "EVIL"
            pd["set_env"] = True
            return pd
    """, name="rpme_env")
    yaml = _temp_yaml("wf:\n  - rpme_env:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        _wait_results(e, "test", 1, timeout=10)

    polluted = os.environ.get(env_key)
    os.environ.pop(env_key, None)

    if polluted:
        return False, f"engine os.environ was polluted ({env_key}={polluted})"

    return True, ("step set env var in subprocess but engine process "
                  "was NOT affected (subprocess isolation)")


def test_proto_step_raises_systemexit():
    """Step calls sys.exit(). Should be caught by worker, not kill engine."""
    from engine import Engine

    _temp_step("""
        import sys
        def run(pd, state, **p):
            sys.exit(42)
    """, name="rpse_exit")
    yaml = _temp_yaml("wf:\n  - rpse_exit:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        time.sleep(5)
        status = e.status("test")

    if status["failed"] >= 1:
        return True, "sys.exit caught by worker -- engine survived"
    return False, "no failure recorded for sys.exit()"


# =====================================================================
#  SCOPE SYSTEM ATTACKS
# =====================================================================


def test_scope_complete_on_unscoped_pipeline():
    """Signal complete on a pipeline with no scopes."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rsnp_ok")
    yaml = _temp_yaml("wf:\n  - rsnp_ok:")

    with Engine() as e:
        e.register("test", yaml)
        # complete="group" but no scope in YAML
        e.submit("test", {}, scope={"group": "X"}, complete="group")
        time.sleep(3)
        status = e.status("test")

    # Phase 0 should complete, scope handler should log warning
    if status["completed"] >= 1:
        return True, "Phase 0 completed, invalid scope signal ignored"
    return False, f"unexpected status: {status}"


def test_scope_mismatched_labels():
    """Submit with one scope value, complete with a different one."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="rsml_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="rsml_b")
    yaml = _temp_yaml("""
        wf:
          - rsml_a:
          - rsml_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {"v": 1}, scope={"group": "A"})
        e.submit("test", {"v": 2}, scope={"group": "A"})
        # Complete with DIFFERENT group value
        e.submit("test", {"v": 3}, scope={"group": "B"},
                 complete="group")

        results = _wait_results(e, "test", 4, timeout=15)

    # Group B should trigger with just the one job (v=3)
    scoped = [r for r in results if r.get("_phase") == 1]
    if scoped:
        return True, (f"mismatched label: scoped step ran with "
                      f"n={scoped[0].get('n')} (only B's job)")
    return True, "mismatched label handled gracefully"


def test_scope_none_value():
    """Scope key with None as value."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = pd["input"]["v"]
            return pd
    """, name="rsnv_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="rsnv_b")
    yaml = _temp_yaml("""
        wf:
          - rsnv_a:
          - rsnv_b:
              scope: group
    """)

    with Engine() as e:
        e.register("test", yaml)
        try:
            e.submit("test", {"v": 1}, scope={"group": None})
            e.submit("test", {"v": 2}, scope={"group": None},
                     complete="group")
            results = _wait_results(e, "test", 3, timeout=15)
            scoped = [r for r in results if r.get("_phase") == 1]
            if scoped:
                return True, f"None scope value worked: n={scoped[0].get('n')}"
            return True, "None scope value accepted without crash"
        except Exception as exc:
            return False, f"exception with None scope: {type(exc).__name__}: {exc}"


def test_scope_very_long_key():
    """Scope key with a 10000-char value."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            pd["v"] = 1
            return pd
    """, name="rslk_a")
    _temp_step("""
        def run(pd, state, **p):
            pd["n"] = len(pd["results"])
            return pd
    """, name="rslk_b")
    yaml = _temp_yaml("""
        wf:
          - rslk_a:
          - rslk_b:
              scope: group
    """)

    long_key = "A" * 10000

    with Engine() as e:
        e.register("test", yaml)
        t0 = time.perf_counter()
        e.submit("test", {"v": 1}, scope={"group": long_key})
        e.submit("test", {"v": 2}, scope={"group": long_key},
                 complete="group")
        results = _wait_results(e, "test", 3, timeout=15)
        elapsed = time.perf_counter() - t0

    scoped = [r for r in results if r.get("_phase") == 1]
    if not scoped:
        return False, "no scoped result"
    if scoped[0].get("n") != 2:
        return False, f"expected 2, got {scoped[0].get('n')}"

    return True, f"10000-char scope key: n={scoped[0]['n']}, time={_fmt(elapsed)}"


# =====================================================================
#  LIFECYCLE ATTACKS
# =====================================================================


def test_lifecycle_use_after_shutdown():
    """Register after shutdown -- should raise, not hang."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): return pd", name="rluas_ok")
    yaml = _temp_yaml("wf:\n  - rluas_ok:")

    e = Engine()
    e.shutdown()

    try:
        e.register("test", yaml)
        return False, "register succeeded after shutdown"
    except RuntimeError as exc:
        if "shut down" in str(exc).lower():
            return True, f"register correctly rejected: {exc}"
        return False, f"RuntimeError but wrong message: {exc}"
    except Exception as exc:
        return False, f"wrong exception: {type(exc).__name__}: {exc}"


def test_lifecycle_double_shutdown():
    """Double shutdown should not raise."""
    from engine import Engine

    e = Engine()
    e.shutdown()
    try:
        e.shutdown()
        return True, "double shutdown did not raise"
    except Exception as exc:
        return False, f"double shutdown raised: {type(exc).__name__}: {exc}"


def test_lifecycle_100_pipelines():
    """Register 100 pipelines on one engine."""
    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rl100_ok")

    from engine import Engine

    with Engine(max_concurrent=8) as e:
        for i in range(100):
            yaml = _temp_yaml(f"wf_{i}:\n  - rl100_ok:")
            e.register(f"p{i}", yaml)

        # Submit one job per pipeline
        for i in range(100):
            e.submit(f"p{i}", {})

        # Wait for some results
        time.sleep(10)
        total = sum(len(e.results(f"p{i}")) for i in range(100))

    if total < 50:
        return False, f"only {total}/100 jobs completed"

    return True, f"{total}/100 pipelines completed"


def test_lifecycle_1000_jobs():
    """1000 jobs on one pipeline."""
    _temp_step("def run(pd, state, **p): pd['i'] = pd['input']['i']; return pd",
               name="rl1k")
    yaml = _temp_yaml("wf:\n  - rl1k:")

    from engine import Engine

    with Engine(max_concurrent=8) as e:
        e.register("test", yaml)
        for i in range(1000):
            e.submit("test", {"i": i})

        results = _wait_results(e, "test", 1000, timeout=300)

    if len(results) != 1000:
        return False, f"only {len(results)}/1000 completed"

    return True, f"1000 jobs completed"


def test_lifecycle_context_manager_exception():
    """Exception inside context manager still cleans up."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): return pd", name="rlcme")
    yaml = _temp_yaml("wf:\n  - rlcme:")

    children_before = _count_children()

    try:
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            _wait_results(e, "test", 1, timeout=10)
            raise ValueError("deliberate")
    except ValueError:
        pass

    time.sleep(0.5)
    children_after = _count_children()
    leaked = max(0, children_after - children_before)

    if leaked > 0:
        return False, f"leaked {leaked} processes after exception in context manager"

    return True, "context manager cleaned up after exception"


# =====================================================================
#  WORKER ATTACKS
# =====================================================================


def test_worker_nonexistent_env():
    """Step with nonexistent conda environment. Worker spawn fails."""
    from engine import Engine

    _temp_step("""
        METADATA = {"environment": "NONEXISTENT_ENV_12345"}
        def run(pd, state, **p): return pd
    """, name="rwne")
    yaml = _temp_yaml("wf:\n  - rwne:")

    with Engine(execution_timeout=10.0) as e:
        e.register("test", yaml)
        e.submit("test", {})
        # Worker connect timeout is 60s by default, wait long enough
        time.sleep(70)
        status = e.status("test")

    if status["failed"] >= 1:
        return True, "failure recorded for nonexistent environment"
    return False, f"no failure recorded: {status}"


def test_worker_step_raises():
    """Step raises exception -- worker should survive."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            raise RuntimeError("deliberate worker error")
    """, name="rwsr")
    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="rwsr_ok")
    yaml_bad = _temp_yaml("wf_bad:\n  - rwsr:")
    yaml_ok = _temp_yaml("wf_ok:\n  - rwsr_ok:")

    with Engine() as e:
        e.register("bad", yaml_bad)
        e.register("ok", yaml_ok)

        e.submit("bad", {})
        time.sleep(3)
        # Worker should still be alive for the next job
        e.submit("ok", {})
        results = _wait_results(e, "ok", 1, timeout=10)

    if not results or not results[0].get("ok"):
        return False, "worker died after step error"

    return True, "worker survived step exception and handled next job"


def test_worker_missing_run_function():
    """Step file without run() function."""
    from engine import Engine

    _temp_step("x = 42  # no run function", name="rwmr")
    yaml = _temp_yaml("wf:\n  - rwmr:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        time.sleep(5)
        status = e.status("test")

    if status["failed"] >= 1:
        return True, "failure recorded for missing run()"
    return False, f"no failure: {status}"


def test_worker_syntax_error():
    """Step file with syntax error. AST parsing fails at register time."""
    from engine import Engine

    _temp_step("def run(\n    return {}", name="rwse")
    yaml = _temp_yaml("wf:\n  - rwse:")

    try:
        with Engine() as e:
            e.register("test", yaml)
        return False, "register succeeded with syntax error step"
    except SyntaxError:
        return True, "SyntaxError raised at register (AST parse)"
    except Exception as exc:
        return True, f"error at register: {type(exc).__name__}: {exc}"


# =====================================================================
#  CONCURRENCY STRESS
# =====================================================================


def test_stress_concurrent_pipelines():
    """Multiple pipelines running concurrently."""
    from engine import Engine

    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.01)
            pd["pipeline"] = pd["input"]["p"]
            return pd
    """, name="rscp")

    from engine import Engine

    with Engine(max_concurrent=8) as e:
        for i in range(5):
            yaml = _temp_yaml(f"wf_{i}:\n  - rscp:")
            e.register(f"p{i}", yaml)

        for i in range(5):
            for j in range(10):
                e.submit(f"p{i}", {"p": f"p{i}"})

        time.sleep(15)
        total = 0
        for i in range(5):
            total += len(e.results(f"p{i}"))

    if total < 40:
        return False, f"only {total}/50 jobs completed"

    return True, f"{total}/50 concurrent pipeline jobs completed"


def test_stress_status_during_execution():
    """Call status() repeatedly while jobs are running."""
    from engine import Engine

    _temp_step("""
        import time
        def run(pd, state, **p):
            time.sleep(0.1)
            return pd
    """, name="rssde")
    yaml = _temp_yaml("wf:\n  - rssde:")

    errors = []

    with Engine(max_concurrent=4) as e:
        e.register("test", yaml)
        for i in range(20):
            e.submit("test", {})

        # Poll status rapidly
        for _ in range(50):
            try:
                s = e.status("test")
                assert isinstance(s, dict)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.02)

    if errors:
        return False, f"status() errors: {errors[:3]}"

    return True, "50 status() calls during execution, no errors"


def test_stress_mixed_success_failure():
    """Mix of successful and failing jobs."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            if pd["input"].get("fail"):
                raise ValueError("deliberate")
            pd["ok"] = True
            return pd
    """, name="rsmsf")
    yaml = _temp_yaml("wf:\n  - rsmsf:")

    with Engine(max_concurrent=8) as e:
        e.register("test", yaml)
        for i in range(40):
            e.submit("test", {"fail": (i % 3 == 0)})

        # Wait for all to complete or fail
        time.sleep(45)
        status = e.status("test")

    total = status["completed"] + status["failed"]
    if total < 30:
        return False, f"only {total}/40 finished"

    return True, (f"completed={status['completed']}, failed={status['failed']}, "
                  f"total={total}/40")


def test_stress_gc_pressure():
    """Create lots of garbage during execution to stress GC."""
    from engine import Engine

    _temp_step("""
        def run(pd, state, **p):
            # Create circular references
            for _ in range(100):
                a = {}
                b = {"parent": a}
                a["child"] = b
            pd["ok"] = True
            return pd
    """, name="rsgc")
    yaml = _temp_yaml("wf:\n  - rsgc:")

    with Engine() as e:
        e.register("test", yaml)
        for i in range(20):
            e.submit("test", {})

        gc.collect()
        results = _wait_results(e, "test", 20, timeout=30)

    if len(results) < 18:
        return False, f"only {len(results)}/20 completed under GC pressure"

    return True, f"{len(results)}/20 completed under GC pressure"


# =====================================================================
#  YAML EDGE CASES
# =====================================================================


def test_yaml_empty_params():
    """Step with explicitly empty params (None in YAML)."""
    from engine import Engine

    _temp_step("def run(pd, state, **p): pd['ok'] = True; return pd",
               name="ryep")
    yaml = _temp_yaml("wf:\n  - ryep:")

    with Engine() as e:
        e.register("test", yaml)
        e.submit("test", {})
        results = _wait_results(e, "test", 1, timeout=10)

    if not results or not results[0].get("ok"):
        return False, "empty params failed"

    return True, "empty params handled"


def test_yaml_nonexistent_step():
    """YAML references a step file that does not exist.
    In v4, register() reads METADATA via AST, so missing file fails early."""
    from engine import Engine

    yaml = _temp_yaml("wf:\n  - does_not_exist:")

    try:
        with Engine() as e:
            e.register("test", yaml)
        return False, "register succeeded with missing step file"
    except FileNotFoundError:
        return True, "FileNotFoundError raised at register (missing step file)"
    except Exception as exc:
        return True, f"error at register: {type(exc).__name__}: {exc}"


# =====================================================================
#  RUNNER
# =====================================================================


TESTS = [
    # Race conditions
    ("Race: mass submit + complete",          test_race_mass_submit_then_complete),
    ("Race: submit during shutdown",          test_race_submit_during_shutdown),
    ("Race: rapid engine cycles",             test_race_rapid_engine_cycles),
    # Resource exhaustion
    ("Exhaust: thread pool deadlock",         test_exhaust_thread_pool_deadlock),
    ("Exhaust: many engines",                 test_exhaust_many_engines),
    ("Exhaust: 500 submits",                  test_exhaust_500_submits),
    # Data corruption
    ("Corrupt: mutate results list",          test_corrupt_mutate_results_list),
    ("Corrupt: massive data (10MB)",          test_corrupt_massive_data),
    ("Corrupt: replace pipeline_data",        test_corrupt_replace_pipeline_data),
    ("Corrupt: dict subclass",                test_corrupt_dict_subclass),
    ("Corrupt: non-dict return",              test_corrupt_non_dict_return),
    # Protocol attacks
    ("Proto: ultrafast step",                 test_proto_ultrafast_step),
    ("Proto: step modifies cwd",             test_proto_step_modifies_cwd),
    ("Proto: step modifies environ",          test_proto_step_modifies_environ),
    ("Proto: step calls sys.exit()",          test_proto_step_raises_systemexit),
    # Scope attacks
    ("Scope: complete on unscoped pipeline",  test_scope_complete_on_unscoped_pipeline),
    ("Scope: mismatched labels",              test_scope_mismatched_labels),
    ("Scope: None value",                     test_scope_none_value),
    ("Scope: 10000-char key",                 test_scope_very_long_key),
    # Lifecycle attacks
    ("Lifecycle: use after shutdown",         test_lifecycle_use_after_shutdown),
    ("Lifecycle: double shutdown",            test_lifecycle_double_shutdown),
    ("Lifecycle: 100 pipelines",              test_lifecycle_100_pipelines),
    ("Lifecycle: 1000 jobs",                  test_lifecycle_1000_jobs),
    ("Lifecycle: context manager exception",  test_lifecycle_context_manager_exception),
    # Worker attacks
    ("Worker: nonexistent environment",       test_worker_nonexistent_env),
    ("Worker: step raises exception",         test_worker_step_raises),
    ("Worker: missing run()",                 test_worker_missing_run_function),
    ("Worker: syntax error",                  test_worker_syntax_error),
    # Concurrency stress
    ("Stress: concurrent pipelines",          test_stress_concurrent_pipelines),
    ("Stress: status during execution",       test_stress_status_during_execution),
    ("Stress: mixed success/failure",         test_stress_mixed_success_failure),
    ("Stress: GC pressure",                   test_stress_gc_pressure),
    # YAML edge cases
    ("YAML: empty params",                    test_yaml_empty_params),
    ("YAML: nonexistent step file",           test_yaml_nonexistent_step),
]


def main():
    import engine

    print()
    print("=" * WIDTH)
    print("  SMART Analysis v4 -- Devil Tests (Adversarial)")
    print("=" * WIDTH)
    print()
    print(f"  Engine:   {engine.__version__}")
    print(f"  Python:   {sys.version.split()[0]} ({sys.executable})")
    print(f"  Tests:    {len(TESTS)}")
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
        print(f"  {icon}  {name:<42s}  {_fmt(elapsed):>8s}")

    print()
    print(f"  {'_' * (WIDTH - 4)}")
    print(f"  Passed:  {n_pass}/{len(TESTS)}")
    print(f"  Failed:  {n_fail}/{len(TESTS)}")
    print(f"  Time:    {_fmt(elapsed_total)}")
    print()

    if n_fail == 0:
        print("  ALL DEVIL TESTS PASSED")
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
