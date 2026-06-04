# Contributing

This document covers local development, testing, and contribution
expectations for Smart Analysis.

## Development setup

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
conda create -n SMART--analysis-dev python=3.12 -y
conda activate SMART--analysis-dev
python -m pip install -e ".[test]"
pytest -m "not cellpose and not slow"
```

The runtime dependency is `pyyaml`. Test and development extras add
`pytest`, `psutil`, NumPy, scikit-image, pooch, tifffile, and
imagecodecs.

## Conda environments for environment-switching tests

Some tests exercise per-step conda environment switching. They need real
conda environments to switch into:

```bash
# Tiny envs for engine unit and integration tests
python workflows/basic_test/environments/setup_env.py
python workflows/basic_test/environments/clean_env.py

# Full env for rare-event-selection workflow tests
python workflows/rare_event_selection/environments/setup_env.py

# Full env for target-acquisition workflow tests and demos
python workflows/target_acquisition/environments/setup_env.py

# Full envs for object-analysis workflow tests and demos
python workflows/object_analysis/environments/setup_env.py
python workflows/object_analysis/environments/setup_env.py --step classical

# Target discovery envs
python workflows/target_discovery/environments/setup_env.py
python workflows/target_discovery/environments/setup_env.py --step cluster
```

Tests marked `@pytest.mark.conda_env` skip cleanly if the `basic_test`
environments do not exist. Tests marked `@pytest.mark.cellpose` skip when
Cellpose or Torch cannot be imported. Tests marked `@pytest.mark.deep`
need a Torch/DINO-capable environment such as
`SMART--object_analysis--vision`. Tests marked `@pytest.mark.cluster`
need `SMART--target_discovery--cluster`.

## Running tests

```bash
pytest -m "not cellpose and not deep and not cluster and not conda_env"  # CI-shaped
pytest -m "not cellpose and not slow"  # fast public smoke test
pytest                          # everything runnable in this env
pytest -m "not slow"            # fast subset including Cellpose if available
pytest -m cellpose              # only real Cellpose tests
pytest -m adversarial           # stress, race, corruption, protocol tests
pytest engine/                  # engine unit tests only
pytest -k Priority              # by name pattern
pytest -x                       # stop on first failure
```

### Markers

Defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `integration` | End-to-end tests using real YAML pipelines through the v4 API. |
| `robustness` | Edge cases, error recovery, and resource cleanup. |
| `adversarial` | Stress, race-condition, corruption, and protocol-attack tests. |
| `slow` | Tests that legitimately take more than about 5 seconds. |
| `cellpose` | Requires real Cellpose, scikit-image, and working Torch imports. |
| `conda_env` | Requires the `SMART--basic_test--env_a` conda environment. |

## Repository layout

See [`README.md`](README.md#project-structure) for the public project
map. Engine internals are described in
[`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md).

## Code style

- Follow PEP 8 for Python. No formatter is enforced; match the
  surrounding file.
- Prefer plain dictionaries and standard library types over heavy
  abstractions.
- Type hints are welcome but not required. Engine internals use light
  hints; workflow steps are mostly unannotated.
- Keep step files, YAML, and source comments ASCII-only. Engine file
  loading intentionally does not add custom encoding handling.

## Engine invariants

Read [`docs/Engine_v4_Design.md`](docs/Engine_v4_Design.md) before
changing engine internals. Preserve these invariants:

1. The engine never runs step code in its own process.
2. The caller signals scope completion; the engine does not infer it.
3. Workers are spawned lazily and reused while warm.
4. Step `METADATA` is read by AST parsing only; the orchestrator does not
   import step modules for routing.
5. v4 has one scope axis. Adding another changes the execution model and
   requires design work, not a local patch.

## Adding tests

Engine unit tests go in `engine/test_engine.py`. Integration,
robustness, and adversarial tests for the engine go in
`workflows/basic_test/tests/`. Workflow-specific tests go in
`workflows/<name>/tests/`.

Use the shared fixtures in `conftest.py` (`temp_step`, `temp_yaml`,
`wait_for_results`, `wait_for_completion`, `wait_for_status`,
`count_children`, and `engine_factory`). Async tests should poll with a
bounded wait rather than sleeping for a fixed number of seconds.

Apply markers honestly: use `slow` only for tests that legitimately take
more than about 5 seconds, and use `cellpose` for anything needing the
Cellpose runtime.

## Adding a workflow

Mirror the established workflow layout:

```text
workflows/<name>/
  pipelines/
    <name>_pipeline.yaml
  steps/
    preprocess.py
    ...
  tests/
    test_<name>.py
  run_pipeline.py
```

Add `environments/setup_env.py` and `environments/clean_env.py` when the
workflow owns a dedicated conda environment.

## Commits and pull requests

- Use a short imperative subject line, for example
  `engine: fix priority drop on shutdown`.
- Explain why the change is needed; the diff already shows what changed.
- Keep one logical change per commit.
- Add or update tests for behavior changes.
- Run `pytest` before submitting. Use a Cellpose-capable environment if
  your change touches worker, pool, lifecycle, or Cellpose workflow code.

## Reporting bugs

Open a GitHub issue with:

- A minimal pipeline YAML and step file that reproduce the bug.
- The full traceback. Set `verbose: 3` in pipeline metadata when engine
  logs are relevant.
- Environment details: `python --version`, `conda --version`, OS, and
  relevant package versions.

## License

By contributing, you agree your contributions are licensed under the
project's MIT license. See [`LICENSE`](LICENSE).
