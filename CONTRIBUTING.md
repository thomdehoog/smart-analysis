# Contributing

Thanks for your interest. This document covers what you need to develop,
test, and submit changes to Smart Analysis.

If you are an AI agent, also read [`AGENTS.md`](AGENTS.md) — it has the
repository brief in a more structured form.

## Setting up a development environment

```bash
git clone https://github.com/thomdehoog/smart-analysis.git
cd smart-analysis
python -m venv .venv && source .venv/bin/activate    # or use conda
pip install -e .[test]
pytest
```

The runtime dependency is just `pyyaml`. Test/dev adds `pytest` and
`psutil`.

### Conda envs for env-switching tests

Some tests exercise the engine's per-step conda environment switching.
They need actual conda envs to switch into:

```bash
# Tiny envs for the engine unit / integration tests
cd workflows/basic_test/environments
python setup_env.py        # creates SMART--basic_test--env_a / env_b / env_c
python clean_env.py        # tears them down

# Full env for the rare-event-selection workflow (cellpose + skimage + torch)
cd workflows/rare_event_selection/environments
python setup_env.py        # auto-detects GPU; pip-only to avoid DLL conflicts
```

Tests marked `@pytest.mark.conda_env` skip cleanly if the basic_test envs
do not exist. Tests marked `@pytest.mark.cellpose` skip if cellpose / torch
fail to load in the active env.

## Running tests

A single command, with marker selectors for subsets:

```bash
pytest                          # everything runnable in this env
pytest -m "not slow"            # fast subset (~30s)
pytest -m cellpose              # only real-cellpose tests
pytest -m adversarial           # only stress / race / corruption
pytest engine/                  # engine unit tests only
pytest -k Priority              # by name pattern
pytest -x                       # stop on first failure
```

The full suite is ~165 tests, ~3 minutes wall time on a fast machine
without slow markers. With `-m cellpose` from a cellpose-capable env,
add ~45 s for the real Cellpose runs.

### Markers

Defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `integration` | End-to-end tests using real YAML pipelines through the v4 API. |
| `robustness` | Edge cases, error recovery, resource cleanup. |
| `adversarial` | Stress / race-condition / corruption / protocol-attack tests. |
| `slow` | Tests that legitimately take more than ~5 s. |
| `cellpose` | Requires real cellpose + skimage + working torch in the active env. |
| `conda_env` | Requires the SMART--basic_test--env_a conda env. |

## Repository layout

See [`README.md`](README.md#project-structure) for the full tree, or
[`AGENTS.md`](AGENTS.md#file-map) for a one-line role for each file.

## Making changes

### Code style

- Follow PEP 8 for Python. No formatter is enforced; match the
  surrounding file's style.
- Prefer plain `dict` and standard library types over heavy abstractions.
- Type hints are welcome but not required. The engine itself uses light
  hints; steps are unannotated.
- ASCII-only content in step files, YAMLs, and source comments. The
  engine assumes this and `open()` calls do not specify an encoding.
  See [`AGENTS.md`](AGENTS.md#trust-model).

### Engine internals

Read the design philosophy section in [`AGENTS.md`](AGENTS.md#design-philosophy-what-to-preserve-when-refactoring)
before changing engine internals. Five things in particular:

1. **Subprocess-only execution.** The engine never runs step code in
   its own process. Non-negotiable.
2. **Caller signals scope completion.** The engine does not count
   submissions or guess.
3. **Warm workers, lazy spawn.** Don't reload models per call.
4. **AST-only METADATA.** No step module is ever imported in the
   orchestrator.
5. **One scope axis.** Don't add a second axis without rewriting the
   design doc.

### Adding a test

Engine unit tests go in `engine/test_engine.py`. Integration / robustness /
adversarial tests for the engine go in `workflows/basic_test/tests/`.
Workflow-specific tests go in `workflows/<name>/tests/`.

Use the conftest fixtures (`temp_step`, `temp_yaml`,
`wait_for_results`, etc.) — don't roll your own. Every async test
should poll, never `time.sleep(N)`. The fixtures handle the bounded
polling pattern.

Apply markers honestly: `slow` only for tests that legitimately take
>5 s, `cellpose` for anything needing the cellpose runtime.

### Adding a workflow

Mirror the layout of `workflows/rare_event_selection/`:

```
workflows/<name>/
├── environments/
│   ├── setup_env.py
│   └── clean_env.py
├── pipelines/
│   └── <name>_pipeline.yaml
├── steps/
│   ├── preprocess.py
│   ├── ...
├── tests/
│   └── test_<name>.py
└── run_pipeline.py     # optional CLI runner
```

## Commits and pull requests

- Subject line: short imperative, present tense (e.g. `engine: fix priority drop on shutdown`).
- Body: explain *why*, not just *what*. The diff already says what.
- One logical change per commit. If the work spans engine + tests +
  docs, make three commits.
- Tests are required for behaviour changes. The audit suite will catch
  most regressions; aim higher than that.
- Run `pytest` before submitting. From a cellpose-capable env if your
  change touches the engine's worker / pool / lifecycle code.

## Reporting bugs

Open a GitHub issue with:

- A minimal pipeline YAML + step file that reproduces the bug.
- The full traceback (set `verbose: 3` in pipeline metadata for engine
  logs).
- Your env: `python --version`, `conda --version`, OS, and the output
  of `pip freeze | grep -i 'pyyaml\|psutil\|pytest'`.

## License

By contributing, you agree your contributions are licensed under the
project's MIT license. See [`LICENSE`](LICENSE).
