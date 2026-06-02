# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Public workflow documentation for `workflows/target_acquisition/`.
- Conda setup and cleanup scripts for the
  `SMART--target_acquisition--main` workflow environment.
- Shared workflow environment setup helper used by the Cellpose workflow
  environment scripts.
- CLI runner for `workflows/target_acquisition/`.
- `pyproject.toml` with project metadata, optional test/dev
  dependencies, and registered pytest markers.
- Root `conftest.py` with shared pytest fixtures for engine and workflow
  tests.
- Auto-skip behavior for `@pytest.mark.cellpose` tests when Cellpose or
  Torch cannot be imported.
- `CONTRIBUTING.md` with development setup, test markers, workflow layout,
  and engine invariants.

### Changed

- `README.md` rewritten as a public project onramp for v4.
- `target_acquisition.segment_tile` now declares the reproducible
  `SMART--target_acquisition--main` environment instead of a local
  maintainer environment.
- `pyproject.toml` test extras now install the non-Cellpose workflow test
  dependencies (`numpy`, `scikit-image>=0.23`, `tifffile`, `imagecodecs`)
  so the advertised `pip install -e .[test] && pytest` path works on a
  clean clone.
- Cellpose test availability is now probed in a subprocess, so broken
  Torch native-library loads cannot pollute or crash the pytest process.
- `cell_analysis.extract_features` now computes `intensity_median`
  itself instead of depending on a non-portable scikit-image
  `regionprops_table` property.
- Scoped failure aggregation now prunes consumed failures from pipeline
  status while preserving unrelated scope failures.
- Test files now follow the standard pytest idiom. Marker selectors such
  as `pytest -m adversarial` replace the old per-phase runner scripts.
- `engine/test_conda_utils.py` imports from `engine.conda_utils`, so it
  works when pytest is invoked from the repository root.
- `requirements.txt` now lists runtime dependencies only; test and
  development dependencies live in `pyproject.toml`.

### Removed

- Internal feature-review notes from `workflows/cell_analysis/`.
- Legacy custom test runners superseded by pytest:
  `workflows/basic_test/tests/run_devil.py`,
  `workflows/basic_test/tests/run_robustness.py`,
  `workflows/basic_test/tests/run_all.py`, and
  `workflows/rare_event_selection/tests/run_tests.py`.
- `multiprocessing.active_children()` fallback in resource-leak tests;
  `psutil` is now a test dependency because v4 uses `subprocess.Popen`.

## [4.0.0] - 2026-04

The v4 rewrite. Ground-up simplification of v3 around the primary use
case: local live adaptive microscopy.

### Added

- `Engine` class with public API methods `register`, `submit`, `status`,
  `results`, and `shutdown`, plus context-manager support.
- Scoped aggregation: submissions tagged with `scope={"group": "R3"}`
  accumulate until the caller signals `complete="group"`.
- Per-step warm worker pools. `state` survives across calls on the same
  worker so heavy resources such as Cellpose models and CUDA contexts can
  stay loaded.
- Priority scheduling on `Engine.submit(priority=...)`. Higher-priority
  jobs jump pending lower-priority work; FIFO ordering is preserved within
  the same priority.
- AST-based `METADATA` extraction. The engine never imports step code in
  the orchestrator process.
- Dedicated stderr-drainer thread per worker to avoid Windows pipe-buffer
  stalls.
- Worker subprocess `PYTHONIOENCODING=utf-8`.
- Windows worker spawn with `CREATE_NEW_PROCESS_GROUP`.
- Parent-process heartbeat for worker shutdown when the orchestrator dies.
- End-to-end smoke test using real rare-event workflow steps.
- Real Cellpose tests for the full rare-event workflow and warm-model
  reuse.
- Real conda-environment isolation test for a step with
  `METADATA={"environment": "..."}`.

### Changed

- Subprocess-only execution. The engine no longer runs step code in its
  own process.
- Single scope axis. v3's separate spatial and temporal scopes collapsed
  to one `scope` dictionary.
- Step interface is now
  `def run(pipeline_data, state, **params) -> dict`.
- Engine-recognized step `METADATA` keys are `environment` and
  `max_workers`; other keys are documentation.
- Scoped phases pass aggregator input as
  `{"results": [...], "failures": [...], "metadata": {...}}`, so
  failures are visible to the aggregator.

### Removed

- `run_pipeline()` function, replaced by `Engine.register`,
  `Engine.submit`, and `Engine.results`.
- `PipelineEngine` class name, replaced by `Engine`.
- GPU/CPU device categories and GPU slot management. Steps now choose
  their environment and concurrency.
- Local/minimal/maximal isolation modes. Subprocess isolation is the only
  execution mode.
- Two-axis scope.

### Fixed

- `_PriorityThreadPool` now honors the documented `priority` parameter.
- Stale path resolution in the robustness tests after directory
  reorganization.
- Scope and corruption tests that previously returned truthy values on
  every code path now assert behavior directly.
- `test_max_workers_parallelism` now asserts elapsed time and worker PID
  count.
- Shutdown tests now separately cover register-after-shutdown and
  submit-after-shutdown.

### Documentation

- `docs/Engine_v4_Design.md` documents the v4 execution model and design
  rationale.
- `docs/Usage_Guide.md` provides a tutorial-style walkthrough.

## [3.x] - historical

The v3 engine introduced YAML-defined pipelines, conda environment
switching, and multiple execution modes. It is superseded by v4.
