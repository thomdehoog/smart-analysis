# Feature Refactor Review

Date: 2026-04-27

Reviewed scope:

- `workflows/cell_analysis/steps/extract_features.py`
- `workflows/cell_analysis/tests/test_extract_features.py`
- `workflows/cell_analysis/pipelines/cell_analysis_pipeline.yaml`
- local commit log for the most recent refactor commit message

Run environment:

- `C:\ProgramData\MinicondaZMB\envs\dino3_test\python.exe`

## Summary

The architectural refactor is behaviour-preserving. I compared the refactored
module with the previous commit's extractor on the 2048 x 2048 / 1000-object
synthetic image and found identical schemas, dtypes, and values for the three
requested configurations.

I made documentation-only fixes while reviewing:

- collapsed user-facing layer wording to "default features" and "opt-in feature
  groups";
- removed a static group table from the module docstring so `FEATURE_GROUPS`
  remains the only group table;
- updated the pipeline YAML comment to point users at feature groups instead of
  an incomplete individual-extra list;
- corrected `_to_uint8` docstring drift.

No implementation logic was changed.

## Test Results

| Command | Result |
| --- | --- |
| `python -m pytest -q workflows/cell_analysis/tests/` | 42 passed, 1 warning, 76.71 s |
| `python -m pytest -q workflows/cell_analysis/tests/test_extract_features.py` after final doc edits | 32 passed, 1 warning, 4.58 s |

The warning is from `skimage.measure.regionprops_table` on the empty-mask test.

## Per-Section Verdicts

| Section | Verdict | Notes |
| --- | --- | --- |
| Output equivalence | Correct | Previous and refactored outputs are exactly equal for defaults, all extras, and all extras with spacing. |
| Single source of truth | Correct after doc cleanup | Python dispatch, group expansion, validation, bbox setup, and `FEATURE_GROUPS["all"]` derive from `EXTRAS`. A static docstring group table was removed. |
| `_Context` dataclass | Suggestion | Shape is reasonable and frozen. Caches are not worth adding yet; if profiling shows repeated quantisation cost, add explicit cache fields rather than mutable lazy state. |
| Family taxonomy | Suggestion | Current taxonomy is usable. `rg_spread` is a shape-owned composite that also emits an intensity-weighted column; split only if users need finer selection. |
| Handler order independence | Correct | No handler reads another handler's columns. Some handlers read default/derived columns, which is safe because derived columns run before extras. |
| Dropped future annotations | Correct | With Python 3.10+ syntax, annotations are concrete at import time and dataclass loading works under the test loader. No forward references remain. |
| Handler docstrings | Suggestion fixed | `_local_bg_collar`, `_lbp_features`, `_fft_features`, and `_glrlm_features` match code. `_to_uint8` had stale wording and was fixed. |
| Performance | Correct | Direct old-vs-new module timings are within noise; no configuration regressed by more than 10%. |
| Neutral naming | Correct | No source-name leaks found in the extractor, test file, latest commit message, or this report. |
| Test gaps | Suggestion | Add registry invariant tests so future extras cannot bypass group expansion or bbox declaration checks. |

## Output Equivalence

Comparison target: previous extractor from commit `0eed787859fc49f99c91fb775aefd875020221b6`.

Synthetic input:

- 2048 x 2048 image
- 1000 non-overlapping disk labels
- mean area 291.264 px
- same image generation as the benchmark scripts

Comparison method:

- loaded the previous extractor directly from the local git object store;
- loaded the refactored extractor from the working tree;
- compared column sets, dtypes, shapes, and `np.array_equal(..., equal_nan=True)`
  for every shared column.

| Configuration | Old columns | New columns | Missing | Added | Dtype diffs | Value diffs |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| defaults only | 32 | 32 | none | none | 0 | 0 |
| `extras=["all"]` | 67 | 67 | none | none | 0 | 0 |
| `extras=["all"], pixel_size_um=[0.65, 0.65]` | 67 | 67 | none | none | 0 | 0 |

Verdict: exact output equivalence.

## Registry Review

`EXTRAS` is the single runtime registry:

- `_run_extras` uses `EXTRAS[name].handler`;
- bbox setup uses `EXTRAS[name].needs_bbox`;
- execution order uses `EXTRAS[n].family`;
- `_expand_extras` expands from `FEATURE_GROUPS` and individual `EXTRAS` names;
- validation messages print `sorted(FEATURE_GROUPS)` and `sorted(EXTRAS)`;
- `FEATURE_GROUPS` is a comprehension over `EXTRAS`;
- `FEATURE_GROUPS["all"] = set(EXTRAS)`.

Adding a new entry to `EXTRAS` automatically updates dispatcher coverage, group
expansion, the unknown-extra error message, and the `all` group.

The only drift risk found was prose, not runtime code: the module docstring had
a literal group table and the pipeline YAML comment had an incomplete extras
example. Both were corrected.

## Context Design

Current `_Context` fields are appropriate:

- `props`, `masks`, `img`, `labels`: core data every handler may need;
- `slices`: centralises optional `find_objects` setup;
- `params`: keeps handler-specific defaults local to handlers;
- `pixel_size_um`: prevents coordinate-owning handlers from re-parsing params.

I would not add a quantised image cache yet. The current texture handlers use
different quantisation rules:

- statistical texture uses `n_intensity_bins`;
- LBP uses uint8 scaling;
- GLRLM uses `glrlm_levels`;
- FFT uses the raw float crop.

If profiling later shows repeated scaling cost, add explicit fields computed by
the dispatcher, for example `img_u8` or `img_max`, rather than mutable lazy
caches on a frozen dataclass.

## Taxonomy And Dependencies

The current families are reasonable:

- `intensity`: global/local background features;
- `neighbourhood`: centroid-to-centroid spatial relations;
- `texture`: gradients, histogram texture, LBP, FFT, GLRLM;
- `morphology`: radius of gyration and the paired radial intensity statistic.

`rg_spread` is the one mixed case. It emits one shape column and one
intensity-weighted radial column. Keeping it in morphology is acceptable because
the second column depends on the radius-of-gyration geometry. If users need
independent selection later, split it into two extras.

`local_bg` reads `intensity_mean`, `num_pixels` or `area`, and `intensity_total`
when present. That is not a handler-order dependency, because those columns come
from defaults/derived columns, but it is a hidden source-column dependency. If
this registry grows, consider adding metadata such as:

```python
requires_columns=("intensity_mean",)
produces=("bg_local_mean", "mean_minus_local_bg", ...)
```

That would make missing-source behaviour more inspectable.

## Handler Order

The dispatcher sorts extras by `(family, name)`. I found no extra handler that
reads output from another extra handler.

Handlers that read `ctx.props`:

- `local_bg`: reads default/derived intensity columns only;
- `neighbours`: reads centroid columns only.

Everything else consumes masks/images/labels/slices directly. The order is
therefore reproducible but not semantically required.

## Annotation Import Check

Dropping postponed annotations is robust for Python 3.10, 3.11, and 3.12
because the file uses concrete built-in generics and union syntax available in
those versions. The dataclasses are declared after their referenced classes are
defined, so no forward references are needed:

- `_Context` is declared before handler annotations use it;
- `_Extra` is declared after `_Context`;
- `EXTRAS` is declared after `_Extra`.

Registering dynamically loaded modules in `sys.modules` inside the tests would
also be reasonable as a project-wide test-loader hardening step. For this file,
removing postponed annotations is simpler and avoids coupling the production
module to a test import detail.

## Docstring Spot Check

| Handler/helper | Verdict | Notes |
| --- | --- | --- |
| `_local_bg_collar` | Correct | Describes hole filling, dilation, neighbour exclusion, and added columns; code matches. |
| `_lbp_features` | Correct | Describes whole-image LBP context and added columns; code matches. |
| `_fft_features` | Correct | Describes bbox mask, `fft2`, magnitude stats, explicit entropy histogram, and no windowing; code matches. |
| `_glrlm_features` | Correct | Describes four directions, background sentinel, one-based gray weighting, and added columns; code matches. |
| `_to_uint8` | Fixed | Previous wording overstated its scope and return behaviour; updated to match code. |

## Performance

### Direct Old-vs-New Module Timing

These timings call `extract_features.run` directly for the old and refactored
modules on the same 2048 x 2048 / 1000-object synthetic input.

| Configuration | Previous ms | Refactor ms | Delta |
| --- | ---: | ---: | ---: |
| defaults only | 990.2 | 1017.3 | +2.7% |
| `extras=["all"]` | 5218.5 | 4894.1 | -6.2% |
| `extras=["all"], pixel_size_um=[0.65, 0.65]` | 4853.6 | 4857.9 | +0.1% |

Verdict: no performance regression over 10%.

### Standalone Benchmark Scripts

The scripts under `C:\Users\t.de` are standalone implementations and do not
import the refactored module, so they are useful as a baseline pattern check,
not as refactor-regression evidence.

| Script | Result |
| --- | --- |
| `bench_features.py` | total 19631.6 ms |
| `bench_features_v2.py` compared feature groups | naive 22673.9 ms, bbox 2592.4 ms, 8.7x speedup |

The v2 equivalence section still reports only local-background divergence, the
same expected baseline issue documented earlier.

## Neutral-Name Sweep

Checked:

- `workflows/cell_analysis/steps/extract_features.py`
- `workflows/cell_analysis/tests/test_extract_features.py`
- `workflows/cell_analysis/pipelines/cell_analysis_pipeline.yaml`
- latest commit message from `.git/logs/HEAD`
- this report

Result: clean for the requested source-name patterns.

Latest commit message checked:

```text
workflows: refactor cell_analysis extract_features architecture
```

## Proposed Tests

1. Registry invariants:
   - `set(FEATURE_GROUPS["all"]) == set(EXTRAS)`;
   - every registered name appears in exactly one non-`all` family;
   - every `EXTRAS` value has a callable handler and boolean `needs_bbox`.

2. Bbox declaration guard:
   - monkeypatch or wrap a `needs_bbox=True` handler and assert it receives a
     context with `slices is not None`;
   - assert `needs_bbox=False` only keeps `slices is None` when no other
     requested extra needs bboxes.

3. Group expansion and validation:
   - request one family plus one individual extra and assert de-duplication;
   - assert the unknown-extra error lists the dynamically derived groups and
     extras.

## Prioritised Fixes

Already applied:

1. Replace user-facing layer wording with default features plus opt-in feature
   groups.
2. Remove the static module-docstring group table.
3. Update pipeline YAML comments to use groups rather than a partial extra list.
4. Fix `_to_uint8` docstring drift.

Recommended next:

1. Add the registry invariant tests above.
2. Consider `requires_columns` / `produces` metadata if the registry grows.
3. Split `rg_spread` only if users ask for shape-only versus radial-intensity
   selection.
4. Keep context caches out until profiling shows a measurable repeated-scaling
   cost.
