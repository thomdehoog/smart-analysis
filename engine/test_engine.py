"""
Test suite for the v4 pipeline engine.

Covers: exception hierarchy, AST-based METADATA extraction, phase splitting,
per-environment workers with state dicts, worker pool with per-step
concurrency (semaphores), Engine API (register/submit/status/results),
scope tracking with single-axis completion, concurrent execution,
graceful failure handling, and lifecycle management.

Structure
---------
- TestErrors                  Exception hierarchy
- TestLoader                  AST-based METADATA extraction
- TestPhases                  Phase splitting from YAML step lists
- TestWorkerProtocol          Per-env workers, state dict, caching
- TestWorkerErrorPaths        Crash, timeout, missing file
- TestPool                    Per-env pools, semaphores, reaper
- TestEngineRegister          Pipeline registration
- TestEngineSubmit            Job submission, immediate execution
- TestEngineScopes            Single-axis scope completion
- TestEngineResults           Results queue, phase tagging
- TestEngineConcurrency       Parallel jobs, max_workers
- TestEngineErrors            Graceful failure handling
- TestEngineLifecycle         Shutdown, context manager
- TestEngineStatus            Observability
- TestEngineMultiPipeline     Multiple registered pipelines
- TestPackageAPI              Public exports and versioning

Usage
-----
    python -m pytest engine/test_engine.py -v
    python -m pytest engine/test_engine.py -k Scopes -v
"""

import atexit
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

# Ensure the engine package is importable
ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR.parent))

from engine._loader import get_step_settings
from engine._run import split_phases, parse_yaml, StepConfig, Phase
from engine._errors import (
    WorkerError, WorkerSpawnError, WorkerCrashedError,
    StepExecutionError, ScopeError,
)

# Test fixtures
BASIC_TEST = ENGINE_DIR.parent / "workflows" / "basic_test"
STEPS_DIR = BASIC_TEST / "steps"
PIPELINES_DIR = BASIC_TEST / "pipelines"

# All temp files go here; cleaned up on exit
_TEMP_DIR = tempfile.mkdtemp(prefix="engine_test_")
atexit.register(shutil.rmtree, _TEMP_DIR, True)
_counter = 0


def _next_id():
    global _counter
    _counter += 1
    return _counter


def _temp_step(code, name=None):
    """Write a temporary step .py file to _TEMP_DIR."""
    path = Path(_TEMP_DIR) / (f"{name}.py" if name else
                               f"step_{_next_id()}.py")
    path.write_text(textwrap.dedent(code))
    return str(path)


def _temp_yaml(content):
    """Write a temporary YAML pipeline file to _TEMP_DIR."""
    text = textwrap.dedent(content)
    if "functions_dir" not in text:
        functions_dir = Path(_TEMP_DIR).as_posix()
        header = f'metadata:\n  functions_dir: "{functions_dir}"\n'
        if "metadata:" in text:
            text = text.replace("metadata:", header.rstrip("\n"), 1)
        else:
            text = header + text
    path = Path(_TEMP_DIR) / f"pipeline_{_next_id()}.yaml"
    path.write_text(text)
    return str(path)


# ---- Errors ----------------------------------------------------------


class TestErrors(unittest.TestCase):

    def test_worker_hierarchy(self):
        self.assertTrue(issubclass(WorkerSpawnError, WorkerError))
        self.assertTrue(issubclass(WorkerCrashedError, WorkerError))
        self.assertTrue(issubclass(StepExecutionError, WorkerError))

    def test_step_execution_error_stores_traceback(self):
        err = StepExecutionError("boom", remote_traceback="tb")
        self.assertEqual(str(err), "boom")
        self.assertEqual(err.remote_traceback, "tb")

    def test_step_execution_error_traceback_default_none(self):
        self.assertIsNone(StepExecutionError("x").remote_traceback)

    def test_scope_error_independent(self):
        self.assertFalse(issubclass(ScopeError, WorkerError))
        self.assertTrue(issubclass(ScopeError, Exception))


# ---- Loader ----------------------------------------------------------


class TestLoader(unittest.TestCase):

    def test_defaults_no_metadata(self):
        path = _temp_step("def run(pd, state, **p): return pd")
        s = get_step_settings(Path(path))
        self.assertIsNone(s["environment"])
        self.assertEqual(s["max_workers"], 1)

    def test_explicit_environment(self):
        path = _temp_step("""
            METADATA = {"environment": "gpu_env"}
            def run(pd, state, **p): return pd
        """)
        s = get_step_settings(Path(path))
        self.assertEqual(s["environment"], "gpu_env")

    def test_max_workers(self):
        path = _temp_step("""
            METADATA = {"max_workers": 5}
            def run(pd, state, **p): return pd
        """)
        s = get_step_settings(Path(path))
        self.assertEqual(s["max_workers"], 5)

    def test_max_workers_default_1(self):
        path = _temp_step('METADATA = {"environment": "some_env"}')
        s = get_step_settings(Path(path))
        self.assertEqual(s["max_workers"], 1)

    def test_no_device_in_output(self):
        """v4 does not have a device field."""
        path = _temp_step('METADATA = {"environment": "e", "device": "gpu"}')
        s = get_step_settings(Path(path))
        self.assertNotIn("device", s)

    def test_does_not_execute_module_code(self):
        path = _temp_step("""
            import nonexistent_package_xyz
            METADATA = {"environment": "safe", "max_workers": 3}
            def run(pd, state, **p): return pd
        """)
        s = get_step_settings(Path(path))
        self.assertEqual(s["environment"], "safe")
        self.assertEqual(s["max_workers"], 3)


# ---- Phases ----------------------------------------------------------


class TestPhases(unittest.TestCase):

    def test_no_scope_single_phase(self):
        steps = [{"a": None}, {"b": {"x": 1}}]
        phases = split_phases(steps)
        self.assertEqual(len(phases), 1)
        self.assertIsNone(phases[0].scope)
        self.assertEqual(phases[0].steps[0].name, "a")
        self.assertEqual(phases[0].steps[1].params, {"x": 1})

    def test_one_scope_two_phases(self):
        steps = [
            {"preprocess": None},
            {"segment": None},
            {"stitch": {"scope": "group"}},
            {"analyze": None},
        ]
        phases = split_phases(steps)
        self.assertEqual(len(phases), 2)
        self.assertIsNone(phases[0].scope)
        self.assertEqual([s.name for s in phases[0].steps],
                         ["preprocess", "segment"])
        self.assertEqual(phases[1].scope, "group")
        self.assertEqual([s.name for s in phases[1].steps],
                         ["stitch", "analyze"])

    def test_two_scopes_three_phases(self):
        steps = [
            {"a": None},
            {"b": {"scope": "group"}},
            {"c": None},
            {"d": {"scope": "all"}},
        ]
        phases = split_phases(steps)
        self.assertEqual(len(phases), 3)
        self.assertIsNone(phases[0].scope)
        self.assertEqual(phases[1].scope, "group")
        self.assertEqual(phases[2].scope, "all")

    def test_scope_params_separated(self):
        steps = [{"step": {"scope": "region", "sigma": 1.0}}]
        phases = split_phases(steps)
        self.assertEqual(phases[0].steps[0].params, {"sigma": 1.0})
        self.assertNotIn("scope", phases[0].steps[0].params)

    def test_scope_on_first_step(self):
        steps = [{"a": {"scope": "region"}}, {"b": None}]
        phases = split_phases(steps)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].scope, "region")


# ---- Worker (protocol) -----------------------------------------------


class TestWorkerProtocol(unittest.TestCase):

    def test_execute_returns_result(self):
        from engine._worker import Worker
        path = _temp_step("""
            def run(pd, state, **p):
                pd["ran"] = True
                pd["x"] = p.get("x")
                return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            result = w.execute(path, {"input": 1}, {"x": 42}, timeout=10)
        finally:
            w.shutdown()
        self.assertTrue(result["ran"])
        self.assertEqual(result["x"], 42)
        self.assertEqual(result["input"], 1)

    def test_different_steps_same_worker(self):
        from engine._worker import Worker
        path_a = _temp_step("""
            def run(pd, state, **p): pd["from"] = "a"; return pd
        """)
        path_b = _temp_step("""
            def run(pd, state, **p): pd["from"] = "b"; return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            ra = w.execute(path_a, {}, {}, timeout=10)
            rb = w.execute(path_b, {}, {}, timeout=10)
            self.assertEqual(ra["from"], "a")
            self.assertEqual(rb["from"], "b")
        finally:
            w.shutdown()

    def test_module_caching(self):
        from engine._worker import Worker
        path = _temp_step("""
            _n = 0
            def run(pd, state, **p):
                global _n
                _n += 1
                pd["n"] = _n
                return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            r1 = w.execute(path, {}, {}, timeout=10)
            r2 = w.execute(path, {}, {}, timeout=10)
            self.assertEqual(r1["n"], 1)
            self.assertEqual(r2["n"], 2)
        finally:
            w.shutdown()

    def test_state_dict_persists(self):
        """State dict persists across calls for the same step."""
        from engine._worker import Worker
        path = _temp_step("""
            def run(pd, state, **p):
                state.setdefault("count", 0)
                state["count"] += 1
                pd["count"] = state["count"]
                return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            r1 = w.execute(path, {}, {}, timeout=10)
            r2 = w.execute(path, {}, {}, timeout=10)
            r3 = w.execute(path, {}, {}, timeout=10)
            self.assertEqual(r1["count"], 1)
            self.assertEqual(r2["count"], 2)
            self.assertEqual(r3["count"], 3)
        finally:
            w.shutdown()

    def test_state_dict_isolated_per_step(self):
        """Different steps get separate state dicts."""
        from engine._worker import Worker
        path_a = _temp_step("""
            def run(pd, state, **p):
                state.setdefault("key", "a")
                pd["state_key"] = state["key"]
                return pd
        """)
        path_b = _temp_step("""
            def run(pd, state, **p):
                state.setdefault("key", "b")
                pd["state_key"] = state["key"]
                return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            ra = w.execute(path_a, {}, {}, timeout=10)
            rb = w.execute(path_b, {}, {}, timeout=10)
            self.assertEqual(ra["state_key"], "a")
            self.assertEqual(rb["state_key"], "b")
        finally:
            w.shutdown()

    def test_persistent_reuses_process(self):
        from engine._worker import Worker
        path = _temp_step("""
            import os
            def run(pd, state, **p): pd["pid"] = os.getpid(); return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            r1 = w.execute(path, {}, {}, timeout=10)
            r2 = w.execute(path, {}, {}, timeout=10)
            self.assertEqual(r1["pid"], r2["pid"])
        finally:
            w.shutdown()

    def test_shutdown_and_respawn(self):
        from engine._worker import Worker
        path = _temp_step("""
            import os
            def run(pd, state, **p): pd["pid"] = os.getpid(); return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        try:
            r1 = w.execute(path, {}, {}, timeout=10)
            w.shutdown()
            r2 = w.execute(path, {}, {}, timeout=10)
            self.assertNotEqual(r1["pid"], r2["pid"])
        finally:
            w.shutdown()

    def test_complex_types(self):
        from engine._worker import Worker
        path = _temp_step("def run(pd, state, **p): return pd")
        data = {
            "tuple": (1, 2), "set": {3, 4}, "bytes": b"\xff",
            "nested": {"a": [None, True, {"b": 2.5}]},
        }
        w = Worker(environment=None, connect_timeout=10)
        try:
            r = w.execute(path, data, {}, timeout=10)
        finally:
            w.shutdown()
        self.assertEqual(r["tuple"], (1, 2))
        self.assertEqual(r["set"], {3, 4})
        self.assertEqual(r["nested"]["a"][2]["b"], 2.5)

    def test_worker_status(self):
        from engine._worker import Worker
        w = Worker(environment=None, connect_timeout=10)
        s = w.status
        self.assertEqual(s["state"], "stopped")
        w.shutdown()


# ---- Worker Error Paths ----------------------------------------------


class TestWorkerErrorPaths(unittest.TestCase):

    def test_crash_raises_worker_crashed(self):
        from engine._worker import Worker
        path = _temp_step("""
            import os
            def run(pd, state, **p): os._exit(1)
        """)
        w = Worker(environment=None, connect_timeout=10)
        with self.assertRaises(WorkerCrashedError):
            w.execute(path, {}, {}, timeout=10)
        w.shutdown()

    def test_timeout_raises_step_execution_error(self):
        from engine._worker import Worker
        path = _temp_step("""
            import time
            def run(pd, state, **p): time.sleep(30); return pd
        """)
        w = Worker(environment=None, connect_timeout=10)
        with self.assertRaises(StepExecutionError) as ctx:
            w.execute(path, {}, {}, timeout=1)
        self.assertIn("timed out", str(ctx.exception))
        w.shutdown()

    def test_step_error_has_traceback(self):
        from engine._worker import Worker
        path = _temp_step("""
            def run(pd, state, **p): raise ValueError("test")
        """)
        w = Worker(environment=None, connect_timeout=10)
        with self.assertRaises(StepExecutionError) as ctx:
            w.execute(path, {}, {}, timeout=10)
        self.assertIn("test", str(ctx.exception))
        self.assertIn("ValueError", ctx.exception.remote_traceback)
        w.shutdown()


# ---- Pool ------------------------------------------------------------


class TestPool(unittest.TestCase):

    def test_per_env_worker_reuse(self):
        from engine._pool import WorkerPool
        path = _temp_step("""
            import os
            def run(pd, state, **p): pd["pid"] = os.getpid(); return pd
        """)
        pool = WorkerPool(idle_timeout=60)
        r1 = pool.execute(None, path, {}, {}, timeout=10)
        r2 = pool.execute(None, path, {}, {}, timeout=10)
        self.assertEqual(r1["pid"], r2["pid"])
        pool.shutdown_all()

    def test_shutdown_before_use(self):
        from engine._pool import WorkerPool
        pool = WorkerPool()
        pool.shutdown_all()

    def test_error_through_pool(self):
        from engine._pool import WorkerPool
        path = _temp_step("""
            def run(pd, state, **p): raise ValueError("pool err")
        """)
        pool = WorkerPool()
        with self.assertRaises(StepExecutionError) as ctx:
            pool.execute(None, path, {}, {}, timeout=10)
        self.assertIn("pool err", str(ctx.exception))
        pool.shutdown_all()

    def test_reaper_removes_idle(self):
        from engine._pool import WorkerPool
        path = _temp_step("def run(pd, state, **p): return pd")
        pool = WorkerPool(idle_timeout=0.2)
        pool.execute(None, path, {}, {}, timeout=10)

        env_pool = pool._env_pools[None]
        self.assertTrue(len(env_pool._idle) > 0
                        or len(env_pool._busy) > 0)

        time.sleep(0.4)
        env_pool.reap_idle()
        self.assertEqual(len(env_pool._idle), 0)
        pool.shutdown_all()

    def test_semaphore_limits_concurrency(self):
        """max_workers=1 serializes execution of the same step."""
        from engine._pool import WorkerPool
        path = _temp_step("""
            import time
            def run(pd, state, **p):
                time.sleep(0.3)
                pd["done"] = True
                return pd
        """)
        pool = WorkerPool()
        results = []

        def run_one():
            r = pool.execute(None, path, {}, {}, max_workers=1, timeout=15)
            results.append(r)

        t0 = time.monotonic()
        threads = [threading.Thread(target=run_one) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - t0
        pool.shutdown_all()

        self.assertEqual(len(results), 2)
        self.assertGreater(elapsed, 0.5,
                           "max_workers=1 should serialize execution")

    def test_semaphore_allows_parallelism(self):
        """max_workers=4 allows parallel execution."""
        from engine._pool import WorkerPool
        path = _temp_step("""
            import time
            def run(pd, state, **p):
                time.sleep(0.3)
                pd["done"] = True
                return pd
        """)
        pool = WorkerPool()
        results = []

        def run_one():
            r = pool.execute(None, path, {}, {}, max_workers=4, timeout=15)
            results.append(r)

        t0 = time.monotonic()
        threads = [threading.Thread(target=run_one) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - t0
        pool.shutdown_all()

        self.assertEqual(len(results), 4)
        self.assertLess(elapsed, 1.0,
                        "max_workers=4 should allow parallel execution")


# ---- Engine (register) -----------------------------------------------


class TestEngineRegister(unittest.TestCase):

    def test_register_simple(self):
        _temp_step("def run(pd, state, **p): return pd", name="reg_a")
        yaml = _temp_yaml("wf:\n  - reg_a:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)

    def test_register_duplicate_raises(self):
        _temp_step("def run(pd, state, **p): return pd", name="reg_b")
        yaml = _temp_yaml("wf:\n  - reg_b:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            with self.assertRaises(ValueError):
                e.register("test", yaml)

    def test_register_bad_yaml(self):
        path = Path(_TEMP_DIR) / f"bad_{_next_id()}.yaml"
        path.write_text("metadata:\n  functions_dir: .")
        from engine import Engine
        with Engine() as e:
            with self.assertRaises(ValueError):
                e.register("bad", str(path))


# ---- Engine (submit) -------------------------------------------------


class TestEngineSubmit(unittest.TestCase):

    def test_simple_submit(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["ok"] = True
                return pd
        """, name="sub_a")
        yaml = _temp_yaml("wf:\n  - sub_a:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            results = e.results("test")
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0]["ok"])

    def test_multi_step(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["s1"] = 1
                return pd
        """, name="ms_a")
        _temp_step("""
            def run(pd, state, **p):
                pd["s2"] = pd["s1"] + 1
                return pd
        """, name="ms_b")
        yaml = _temp_yaml("wf:\n  - ms_a:\n  - ms_b:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            results = e.results("test")
        self.assertEqual(results[0]["s1"], 1)
        self.assertEqual(results[0]["s2"], 2)

    def test_data_flows_between_steps(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["from_a"] = "hello"
                return pd
        """, name="df_a")
        _temp_step("""
            def run(pd, state, **p):
                pd["saw"] = pd.get("from_a")
                return pd
        """, name="df_b")
        yaml = _temp_yaml("wf:\n  - df_a:\n  - df_b:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            results = e.results("test")
        self.assertEqual(results[0]["saw"], "hello")

    def test_input_data(self):
        _temp_step("def run(pd, state, **p): return pd", name="inp")
        yaml = _temp_yaml("wf:\n  - inp:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {"key": "val"})
            time.sleep(2)
            results = e.results("test")
        self.assertEqual(results[0]["input"]["key"], "val")

    def test_params_from_yaml(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["x"] = p.get("x")
                return pd
        """, name="par")
        yaml = _temp_yaml("wf:\n  - par:\n      x: 42")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            results = e.results("test")
        self.assertEqual(results[0]["x"], 42)

    def test_concurrent_submits(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["job"] = pd["input"]["job"]
                return pd
        """, name="conc")
        yaml = _temp_yaml("wf:\n  - conc:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            for i in range(5):
                e.submit("test", {"job": i})
            time.sleep(5)
            results = e.results("test")
        self.assertEqual(len(results), 5)
        self.assertEqual(sorted(r["job"] for r in results), list(range(5)))


# ---- Engine (scopes) -------------------------------------------------


class TestEngineScopes(unittest.TestCase):

    def test_scope_collects_results(self):
        """Scoped step receives accumulated results from all jobs."""
        _temp_step("""
            def run(pd, state, **p):
                pd["tile"] = pd["input"]["tile"]
                return pd
        """, name="sc_seg")
        _temp_step("""
            def run(pd, state, **p):
                tiles = [r["tile"] for r in pd["results"]]
                pd["tiles"] = sorted(tiles)
                return pd
        """, name="sc_stitch")
        yaml = _temp_yaml("""
            wf:
              - sc_seg:
              - sc_stitch:
                  scope: group
        """)
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            for i in range(3):
                complete = "group" if i == 2 else None
                e.submit("test", {"tile": i},
                         scope={"group": "R1"},
                         complete=complete)
            time.sleep(5)
            results = e.results("test")

        # Should have 3 Phase 0 results + 1 scoped result
        phase0 = [r for r in results if r.get("_phase") == 0]
        scoped = [r for r in results if r.get("_phase") == 1]
        self.assertEqual(len(phase0), 3)
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["tiles"], [0, 1, 2])

    def test_scope_preserves_submission_order(self):
        _temp_step("""
            import time
            def run(pd, state, **p):
                time.sleep(0.05)
                pd["val"] = pd["input"]["val"]
                return pd
        """, name="ord_step")
        _temp_step("""
            def run(pd, state, **p):
                pd["order"] = [r["val"] for r in pd["results"]]
                return pd
        """, name="ord_collect")
        yaml = _temp_yaml("""
            wf:
              - ord_step:
              - ord_collect:
                  scope: group
        """)
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            for i in range(5):
                complete = "group" if i == 4 else None
                e.submit("test", {"val": i},
                         scope={"group": "G1"},
                         complete=complete)
            time.sleep(8)
            results = e.results("test")

        scoped = [r for r in results if r.get("_phase") == 1]
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["order"], [0, 1, 2, 3, 4])

    def test_multiple_scope_groups(self):
        """Different scope groups are collected independently."""
        _temp_step("""
            def run(pd, state, **p):
                pd["val"] = pd["input"]["val"]
                return pd
        """, name="mg_step")
        _temp_step("""
            def run(pd, state, **p):
                pd["vals"] = sorted([r["val"] for r in pd["results"]])
                return pd
        """, name="mg_collect")
        yaml = _temp_yaml("""
            wf:
              - mg_step:
              - mg_collect:
                  scope: group
        """)
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            # Group A: values 10, 20
            e.submit("test", {"val": 10}, scope={"group": "A"})
            e.submit("test", {"val": 20}, scope={"group": "A"},
                     complete="group")
            # Group B: values 30, 40, 50
            e.submit("test", {"val": 30}, scope={"group": "B"})
            e.submit("test", {"val": 40}, scope={"group": "B"})
            e.submit("test", {"val": 50}, scope={"group": "B"},
                     complete="group")
            time.sleep(8)
            results = e.results("test")

        scoped = [r for r in results if r.get("_phase") == 1]
        scoped_vals = sorted([tuple(r["vals"]) for r in scoped])
        self.assertIn((10, 20), scoped_vals)
        self.assertIn((30, 40, 50), scoped_vals)

    def test_complete_list(self):
        """complete parameter accepts a list of scope levels."""
        _temp_step("""
            def run(pd, state, **p):
                pd["v"] = pd["input"]["v"]
                return pd
        """, name="cl_step")
        _temp_step("""
            def run(pd, state, **p):
                pd["group_vals"] = [r["v"] for r in pd["results"]]
                return pd
        """, name="cl_group")
        _temp_step("""
            def run(pd, state, **p):
                pd["all_vals"] = [r.get("group_vals", [])
                                   for r in pd["results"]]
                return pd
        """, name="cl_all")
        yaml = _temp_yaml("""
            wf:
              - cl_step:
              - cl_group:
                  scope: group
              - cl_all:
                  scope: all
        """)
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {"v": 1}, scope={"group": "G1"})
            e.submit("test", {"v": 2}, scope={"group": "G1"},
                     complete=["group", "all"])
            time.sleep(8)
            results = e.results("test")

        phase2 = [r for r in results if r.get("_phase") == 2]
        self.assertEqual(len(phase2), 1)

    def test_all_scope_collects_everything(self):
        """Scope 'all' (not a key in any scope dict) collects everything."""
        _temp_step("""
            def run(pd, state, **p):
                pd["v"] = pd["input"]["v"]
                return pd
        """, name="al_step")
        _temp_step("""
            def run(pd, state, **p):
                pd["total"] = sum(r["v"] for r in pd["results"])
                return pd
        """, name="al_sum")
        yaml = _temp_yaml("""
            wf:
              - al_step:
              - al_sum:
                  scope: all
        """)
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {"v": 10}, scope={"group": "A"})
            e.submit("test", {"v": 20}, scope={"group": "B"})
            e.submit("test", {"v": 30}, scope={"group": "C"},
                     complete="all")
            time.sleep(5)
            results = e.results("test")

        scoped = [r for r in results if r.get("_phase") == 1]
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["total"], 60)


# ---- Engine (results) ------------------------------------------------


class TestEngineResults(unittest.TestCase):

    def test_results_consumed_on_retrieval(self):
        _temp_step("def run(pd, state, **p): return pd", name="drain")
        yaml = _temp_yaml("wf:\n  - drain:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            r1 = e.results("test")
            r2 = e.results("test")
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 0)

    def test_results_tagged_with_phase(self):
        _temp_step("def run(pd, state, **p): return pd", name="tag")
        yaml = _temp_yaml("wf:\n  - tag:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            results = e.results("test")
        self.assertEqual(results[0]["_phase"], 0)
        self.assertIsNone(results[0]["_scope_level"])

    def test_unregistered_pipeline_raises(self):
        from engine import Engine
        with Engine() as e:
            with self.assertRaises(KeyError):
                e.results("nonexistent")


# ---- Engine (concurrency) -------------------------------------------


class TestEngineConcurrency(unittest.TestCase):

    def test_many_concurrent_jobs(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["idx"] = pd["input"]["idx"]
                return pd
        """, name="many")
        yaml = _temp_yaml("wf:\n  - many:")
        from engine import Engine
        with Engine(max_concurrent=8) as e:
            e.register("test", yaml)
            for i in range(20):
                e.submit("test", {"idx": i})
            time.sleep(15)
            results = e.results("test")
        self.assertEqual(len(results), 20)
        self.assertEqual(sorted(r["idx"] for r in results), list(range(20)))


# ---- Engine (errors) -------------------------------------------------


class TestEngineErrors(unittest.TestCase):

    def test_failed_job_does_not_crash_pipeline(self):
        """Other jobs continue when one fails."""
        _temp_step("""
            def run(pd, state, **p):
                if pd["input"].get("fail"):
                    raise ValueError("deliberate failure")
                pd["ok"] = True
                return pd
        """, name="graceful")
        yaml = _temp_yaml("wf:\n  - graceful:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {"fail": True})
            e.submit("test", {"fail": False})
            e.submit("test", {"fail": False})
            time.sleep(5)
            status = e.status("test")
        self.assertGreaterEqual(status["completed"], 2)
        self.assertGreaterEqual(status["failed"], 1)

    def test_failures_in_status(self):
        _temp_step("""
            def run(pd, state, **p): raise RuntimeError("boom")
        """, name="fail_status")
        yaml = _temp_yaml("wf:\n  - fail_status:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(3)
            status = e.status("test")
        self.assertEqual(status["failed"], 1)
        self.assertTrue(len(status["failures"]) > 0)
        self.assertIn("boom", status["failures"][0]["error"])

    def test_return_non_dict_raises(self):
        _temp_step("""
            def run(pd, state, **p): return "not a dict"
        """, name="bad_ret")
        yaml = _temp_yaml("wf:\n  - bad_ret:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(3)
            status = e.status("test")
        self.assertEqual(status["failed"], 1)


# ---- Engine (lifecycle) ----------------------------------------------


class TestEngineLifecycle(unittest.TestCase):

    def test_context_manager(self):
        _temp_step("def run(pd, state, **p): return pd", name="ctx")
        yaml = _temp_yaml("wf:\n  - ctx:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)

    def test_shutdown_then_submit_raises(self):
        from engine import Engine
        e = Engine()
        e.shutdown()
        with self.assertRaises(RuntimeError):
            _temp_step("def run(pd, state, **p): return pd", name="shut")
            yaml = _temp_yaml("wf:\n  - shut:")
            e.register("test", yaml)

    def test_double_shutdown(self):
        from engine import Engine
        e = Engine()
        e.shutdown()
        e.shutdown()  # should not raise


# ---- Engine (status) -------------------------------------------------


class TestEngineStatus(unittest.TestCase):

    def test_status_single_pipeline(self):
        _temp_step("def run(pd, state, **p): return pd", name="st")
        yaml = _temp_yaml("wf:\n  - st:")
        from engine import Engine
        with Engine() as e:
            e.register("test", yaml)
            e.submit("test", {})
            time.sleep(2)
            status = e.status("test")
        self.assertIn("completed", status)
        self.assertIn("failed", status)
        self.assertIn("pending", status)

    def test_status_all_pipelines(self):
        _temp_step("def run(pd, state, **p): return pd", name="st2")
        yaml = _temp_yaml("wf:\n  - st2:")
        from engine import Engine
        with Engine() as e:
            e.register("a", yaml)
            e.register("b", yaml)
            status = e.status()
        self.assertIn("a", status)
        self.assertIn("b", status)

    def test_status_nonexistent_raises(self):
        from engine import Engine
        with Engine() as e:
            with self.assertRaises(KeyError):
                e.status("ghost")


# ---- Engine (multi-pipeline) ----------------------------------------


class TestEngineMultiPipeline(unittest.TestCase):

    def test_two_pipelines_shared_workers(self):
        _temp_step("""
            def run(pd, state, **p):
                pd["from"] = p.get("from", "unknown")
                return pd
        """, name="shared")
        yaml_a = _temp_yaml("wf_a:\n  - shared:\n      from: a")
        yaml_b = _temp_yaml("wf_b:\n  - shared:\n      from: b")
        from engine import Engine
        with Engine() as e:
            e.register("a", yaml_a)
            e.register("b", yaml_b)
            e.submit("a", {})
            e.submit("b", {})
            time.sleep(3)
            ra = e.results("a")
            rb = e.results("b")
        self.assertEqual(len(ra), 1)
        self.assertEqual(len(rb), 1)
        self.assertEqual(ra[0]["from"], "a")
        self.assertEqual(rb[0]["from"], "b")


# ---- Package API -----------------------------------------------------


class TestPackageAPI(unittest.TestCase):

    def test_public_imports(self):
        from engine import Engine
        from engine import WorkerError, WorkerSpawnError
        from engine import WorkerCrashedError, StepExecutionError
        from engine import ScopeError

    def test_version(self):
        import engine
        self.assertEqual(engine.__version__, "4.0.0")

    def test_engine_in_all(self):
        import engine
        self.assertIn("Engine", engine.__all__)

    def test_no_run_pipeline(self):
        """v4 does not have run_pipeline."""
        import engine
        self.assertFalse(hasattr(engine, "run_pipeline"))

    def test_no_pipeline_engine(self):
        """v4 does not have PipelineEngine (renamed to Engine)."""
        import engine
        self.assertFalse(hasattr(engine, "PipelineEngine"))


if __name__ == "__main__":
    unittest.main()
