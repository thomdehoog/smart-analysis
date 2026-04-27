# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `pyproject.toml` with project metadata and pytest configuration, including
  registered markers (`integration`, `robustness`, `adversarial`, `slow`,
  `cellpose`, `conda_env`).
- Root `conftest.py` with shared pytest fixtures (`temp_step`, `temp_yaml`,
  `wait_for_results`, `wait_for_completion`, `wait_for_status`,
  `count_children`, `engine_factory`).
- Auto-skip behaviour for `@pytest.mark.cellpose` tests when the cellpose
  runtime fails to load (broken Windows torch DLL, missing package, etc.).
- `AGENTS.md` — repository brief targeted at AI agents.
- `CHANGELOG.md` — this file.
- `CONTRIBUTING.md` — development setup and contribution guidelines.

### Changed
- All test files now follow the standard pytest idiom. The previous mix of
  pytest, custom function-style runners, and a master subprocess driver has
  been replaced with a single `pytest` entry point. Marker selectors
  (e.g. `pytest -m adversarial`) replace the old per-phase `--skip-*` flags.
- `engine/test_conda_utils.py` now imports from `engine.conda_utils` so it
  works when pytest is invoked from the repo root.
- `requirements.txt` slimmed to runtime dependencies only; test/dev
  dependencies live in `pyproject.toml` `[project.optional-dependencies]`.
- `README.md` rewritten for v4: new layered structure (TL;DR → install →
  hello-world → concepts → real-world example → testing), updated API
  examples (the v4 `Engine` class, not the v3 `run_pipeline` function),
  corrected file paths, accurate test counts.

### Removed
- `workflows/basic_test/tests/run_devil.py` (renamed to `test_adversarial.py`,
  the technical term).
- `workflows/basic_test/tests/run_robustness.py` (renamed to `test_robustness.py`).
- `workflows/basic_test/tests/run_all.py` (superseded by `pytest`).
- `workflows/rare_event_selection/tests/run_tests.py` (renamed to `test_rare_event.py`).
- `multiprocessing.active_children()` fallback in resource-leak tests; psutil
  is now a hard dev dependency. (The fallback was vacuously passing on hosts
  without psutil because v4 uses `subprocess.Popen`, which
  `multiprocessing.active_children()` does not see.)

## [4.0.0] — 2026-04

The v4 rewrite. Ground-up simplification of v3 around a single primary
use case: live adaptive microscopy.

### Added
- `Engine` class with the four-method public API: `register`, `submit`,
  `status`, `results`, plus `shutdown` and context-manager support.
- Scoped aggregation: submissions tagged with `scope={"group": "R3", ...}`
  accumulate, and signaling `complete="group"` triggers any subsequent
  scoped step that operates on the accumulated batch.
- Per-step warm worker pools: `state["model"]` survives across calls on
  the same worker so heavy resources (Cellpose models, CUDA contexts)
  load once and stay hot.
- Priority scheduling on `Engine.submit(priority=...)`. Higher priority
  jobs jump pending lower-priority work; FIFO within the same priority.
  Default `priority=0` preserves submission order.
- AST-based METADATA extraction: the engine never imports step code in
  the orchestrator process. Routing decisions come from
  `ast.literal_eval` of the `METADATA` dict literal.
- Dedicated stderr-drainer thread per worker (Windows pipe-buffer fix).
- `PYTHONIOENCODING=utf-8` set on worker subprocess env.
- `creationflags=CREATE_NEW_PROCESS_GROUP` on Windows worker spawn.
- `_parent_alive` heartbeat using `os.kill(pid, 0)` on Unix and ctypes
  `OpenProcess` on Windows.
- New end-to-end smoke test using real `preprocess.py` /
  `extract_features.py` / `feedback.py` from the rare-event workflow.
- New real-cellpose tests: full pipeline end-to-end on
  `skimage.human_mitosis`, plus warm-model verification across submits.
- Test for real conda-env isolation
  (`TestEngineEnvironmentIsolation::test_step_runs_in_declared_environment`)
  that submits a step with `METADATA={"environment": "..."}` and asserts
  `sys.executable` reflects the declared env.

### Changed
- Subprocess-only execution. The engine no longer runs step code in its
  own process. ("Local" mode from v3 is gone.)
- Single scope axis. v3's `spatial`/`temporal` two-axis scoping
  collapsed to one (`scope={"key": "value"}`).
- Step interface: `def run(pipeline_data, state, **params) -> dict`. The
  `state` parameter (per-step per-worker) is new in v4.
- Step METADATA: only `environment` and `max_workers` are honoured by
  the engine; description, version, etc. are documentation only.
- `pipeline_data` flows through scoped phases as
  `{"results": [...], "failures": [...], "metadata": {...}}` —
  failures are now visible to the aggregator.

### Removed
- `run_pipeline()` function (replaced by `Engine.register` + `submit` +
  `results`).
- `PipelineEngine` class name (renamed to `Engine`).
- GPU/CPU device categories and GPU slot management. Steps that need a
  GPU declare a conda env that has GPU libraries installed and set
  `max_workers: 1` if they cannot share VRAM.
- Isolation modes (minimal/maximal). Subprocess isolation is the only mode.
- Two-axis scope.

### Fixed
- `_PriorityThreadPool` replaces the FIFO `ThreadPoolExecutor`, finally
  honouring the documented `priority` parameter (which was previously
  silently dropped).
- Stale path resolution in `run_robustness.py` after the directory
  reorganisation in commit d81bd6c.
- Three test functions whose every code path returned `True`
  (`test_scope_mismatched_labels`, `test_scope_none_value`, and
  `test_corrupt_dict_subclass`) now assert real behaviour.
- `test_max_workers_parallelism` now actually asserts on elapsed time
  and worker-PID count instead of just measuring them.
- `test_shutdown_then_submit_raises` was misnamed (tested `register`
  after shutdown); split into a renamed `test_shutdown_then_register_raises`
  and a real `test_shutdown_then_submit_raises`.

### Documentation
- `docs/Engine_v4_Design.md` — full rewrite of the design doc for v4.
- `docs/Engine_v3_Design.md` retained for historical context until v4.0.0
  ships, then will be removed.

## [3.x] — historical

The v3 engine introduced YAML-defined pipelines, conda env switching,
and the four execution modes (local / pipeline-level / step-level /
mixed). Superseded by v4. See `docs/Engine_v3_Design.md` for details.
