# AGENTS.md — repo brief for AI agents

Hello, agent. This file is a fast onramp for autonomous coding agents
working on or with this repository. It is short on purpose. For depth,
follow the linked files at the bottom.

## What this repo is

**Smart Analysis** is a Python pipeline engine for scientific image
analysis workflows, with first-class support for *live adaptive
microscopy* — the engine accepts work tile-by-tile from an acquiring
microscope and feeds results back into acquisition decisions in real time.

It also runs as a plain post-acquisition batch tool. Both modes use the
**same** `Engine.submit()` API; the engine does not care which one you are
doing.

## What this repo is NOT

- It is **not** a general workflow orchestrator. Use Airflow, Prefect, or
  Dagster for ETL pipelines, DAGs across machines, or scheduled jobs.
- It is **not** a notebook framework or a results-storage system. Steps
  read from and write to a shared in-memory dict (`pipeline_data`); the
  engine does not persist anything for you.
- It is **not** a deep-learning framework. Steps may *use* PyTorch /
  cellpose / scikit-image, but the engine is unopinionated.
- It is **not** a multi-machine system. Everything is local subprocesses
  on one host. There is no scheduler, no broker, no queue daemon.

## Core concepts (one sentence each)

| Term | Meaning |
|---|---|
| **Pipeline** | A registered workflow, defined by a YAML file, with a name. |
| **Step** | A Python file at `<functions_dir>/<name>.py` with a `run(pipeline_data, state, **params) -> dict` function and an optional module-level `METADATA` dict. |
| **`pipeline_data`** | A `dict` carried through every step in a single submission; each step reads what it needs and adds its own outputs. Often abbreviated `pd` in step bodies (Python convention from pandas, *not* a pandas DataFrame). |
| **`state`** | A per-step per-worker `dict` that survives across calls on the same worker — used to keep heavy objects (e.g. a Cellpose model) hot between submits. Reset when the worker shuts down. |
| **Scope** | An aggregation boundary. Submissions tagged with `scope={"group": "R3"}` accumulate results for `R3`; signaling `complete="group"` triggers any scoped step that aggregates that batch. |
| **Phase** | A contiguous run of steps with the same scope level. The engine splits the YAML steps list into phases at scope changes. Phase 0 is the immediate (per-submission) phase. |
| **Worker** | A subprocess that loads one or more steps and executes them. The engine talks to workers over a TCP `Listener` with a per-worker authkey. |
| **METADATA** | A dict literal at module level in a step file: `{"environment": "...", "max_workers": ...}`. Read by the engine via `ast.literal_eval` — step code is **never** executed during routing. |

## Trust model

- The engine **never imports step code in the orchestrator process**. It
  reads `METADATA` via `ast.literal_eval` and spawns a subprocess (in the
  declared conda env) to actually run `run()`. This means a broken or
  malicious step cannot crash the engine; it can only fail its own job.
- Step files are **trusted Python**. The engine assumes the project
  controls them. There is no sandboxing.
- Step content is **ASCII-only** by project policy — see `README.md`.
  Don't add UTF-8 encoding handling to engine `open()` calls.

## File map

| Path | Role |
|---|---|
| `engine/_pipeline.py` | The `Engine` class. Public API: `register`, `submit`, `status`, `results`, `shutdown`. |
| `engine/_pool.py` | `WorkerPool` and per-environment worker reuse. |
| `engine/_worker.py` | A single worker subprocess: spawn, connect, execute, shutdown. |
| `engine/worker_script.py` | The script that runs *inside* the worker subprocess. Loads the step module and dispatches `run()`. |
| `engine/_loader.py` | AST-based METADATA extraction. |
| `engine/_run.py` | YAML parsing, `PipelineState`, phase splitting, scope tracking. |
| `engine/_errors.py` | Exception hierarchy: `WorkerError`, `WorkerSpawnError`, `WorkerCrashedError`, `StepExecutionError`, `ScopeError`. |
| `engine/conda_utils.py` | Conda discovery + GPU detection. Used by env-setup scripts and the engine's `conda run` invocation. |
| `engine/__init__.py` | Public re-exports. The only package surface. |
| `conftest.py` | Root pytest fixtures: `temp_step`, `temp_yaml`, `wait_for_results`, `wait_for_completion`, `wait_for_status`, `count_children`, `engine_factory`. |
| `pyproject.toml` | Project metadata + pytest markers (`integration`, `robustness`, `adversarial`, `slow`, `cellpose`, `conda_env`). |
| `workflows/basic_test/` | Synthetic workflow used for engine testing. |
| `workflows/rare_event_selection/` | Real production workflow: cellpose segmentation + skimage feature extraction + JSON feedback. |
| `docs/Engine_v4_Design.md` | Long-form design rationale. ~1000 lines. Authoritative when this file is ambiguous. |
| `docs/Usage_Guide.md` | Tutorial-style walkthrough. |

## Where to start when extending

| Task | Touch this |
|---|---|
| Add a new public Engine method | `engine/_pipeline.py` + `engine/__init__.py` (re-export) + a test in `engine/test_engine.py`. |
| Add a new metadata key | `engine/_loader.py` (parse) + design-doc update + `engine/test_engine.py::TestLoader`. |
| Change worker protocol | `engine/_worker.py` *and* `engine/worker_script.py` together. They share a private wire format. |
| Add a workflow | `workflows/<name>/{steps,pipelines,environments,tests}/`. Mirror `rare_event_selection`. |
| Add tests | `engine/test_engine.py` for unit, `workflows/basic_test/tests/test_*.py` for integration / robustness / adversarial, `workflows/<name>/tests/test_*.py` for workflow-specific. |

## Running tests

```bash
pytest                        # everything that can run in the active env
pytest -m "not slow"          # fast subset
pytest -m cellpose            # only real-cellpose tests (need cellpose+torch in env)
pytest -m adversarial         # only the adversarial / stress suite
pytest engine/                # only engine unit tests
```

Tests marked `cellpose` auto-skip when cellpose / torch fail to load (see
`conftest.py::pytest_collection_modifyitems`). Run from a cellpose-capable
env (e.g. `dino3_test` on the maintainer's machine, or the dedicated
`SMART--rare_event_selection--main` env created by
`workflows/rare_event_selection/environments/setup_env.py`) to actually
exercise them.

## Design philosophy (what to preserve when refactoring)

1. **Subprocess-only execution.** The engine never runs step code in its
   own process. This is non-negotiable — it's the load-bearing assumption
   for crash isolation, env switching, and protocol stability.
2. **Caller knows when scopes complete.** The engine does not count
   submissions or guess. The acquisition layer signals `complete="..."`.
3. **Warm workers, lazy spawn, idle reaping.** Workers are reused until
   idle; `state["model"]` survives across calls.
4. **AST-only METADATA.** No step module is ever imported in the
   orchestrator. Routing decisions come from `ast.literal_eval`.
5. **One scope axis.** v3 had spatial *and* temporal scope; v4 collapsed
   to one. Don't add a second axis without re-reading the design doc.
6. **No dependencies beyond `pyyaml`.** Workflows can import whatever
   they want, but the engine itself stays thin.

## Things that look wrong but are intentional

- The engine reaches into `multiprocessing.connection.Listener._listener._socket`
  to set a connect timeout. Undocumented CPython internals; works on all
  three platforms. If a Python release breaks it, fix it; do not switch
  transports.
- A `_StderrDrainer` thread reads each worker's stderr continuously. This
  is on purpose — Windows pipe buffers are ~4 KB and a chatty step would
  block the worker without it.
- The orchestrator process passes `PYTHONIOENCODING=utf-8` to workers.
  Required on Windows where the system default code page is locale-bound.
- Step files are loaded via `importlib` from disk inside the worker, not
  imported by package name. This lets `functions_dir` point anywhere.

## Anti-patterns to avoid

- ❌ Calling `Engine.submit` from inside a step. Steps run in subprocesses
  that don't have an Engine reference. Build pipelines, don't recurse.
- ❌ Sharing mutable globals between steps. Each step gets its own worker
  module load; `state` is the only persistence between calls.
- ❌ Long synchronous loops inside `run()`. The engine cannot interrupt a
  running step except by killing the worker. Break work into multiple
  smaller submissions.
- ❌ Returning anything other than a `dict` from `run()`. The engine
  records this as `StepExecutionError`.
- ❌ Heavy imports at the top of step files in the orchestrator's env.
  AST parsing skips them at routing time, but if the orchestrator ever
  imports the step (e.g. via `unittest`) the imports must succeed in
  *some* env. Keep imports relevant to the env declared in METADATA.

## Linked deeper reading

- `README.md` — newcomer onramp + 30-line hello world
- `docs/Engine_v4_Design.md` — full design rationale (1000 lines)
- `docs/Usage_Guide.md` — tutorial walkthrough
- `CONTRIBUTING.md` — how to set up, test, and submit changes
- `CHANGELOG.md` — version history
