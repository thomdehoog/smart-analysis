# Feature Texture Review

Date: 2026-04-27

Reviewed files:

- `workflows/cell_analysis/steps/extract_features.py`
- `workflows/cell_analysis/tests/test_extract_features.py`
- `FEATURE_LIBRARY_MAPPING.md`
- `FEATURE_BENCHMARK_REVIEW.md`

Run environment:

- `C:\ProgramData\MinicondaZMB\envs\dino3_test\python.exe`

Focused test run:

```text
23 passed, 1 warning in 4.71s
```

The one warning comes from `skimage.measure.regionprops_table` on an empty mask
test and is not related to the texture extras.

## Executive Summary

The LBP, FFT, and GLRLM extras are generally correct against the current
source-neutral mapping.

One small test comment was wrong and has been fixed: skimage default LBP returns
code `255` for equal neighbours with `P=8`, not code `0`. The assertion values
were still correct because all sampled object pixels had the same LBP code.

Main recommendations:

1. Add a non-square two-gray-level GLRLM test. The current diagonal extraction
   is correct, but this is the best regression test for it.
2. Add an object-border LBP test showing the current image-wide convention.
3. Add an FFT border-object test and document `fft2`, magnitude, no windowing,
   and histogram entropy conventions.
4. Consider vectorising LBP histogram statistics. LBP is now one of the larger
   optional costs.
5. Keep GLRLM opt-in. Four-direction GLRLM is the largest of the three new
   texture extras.

## Per-Extra Verdict

| Extra | Verdict | Notes |
| --- | --- | --- |
| LBP | Correct, with convention risk | `P=8`, `R=1`, `method="default"` is the right skimage call for a 3x3 neighbourhood. The image-wide LBP image means object-border pixels see nearby non-object pixels. This should be documented as the default convention. |
| FFT | Correct for pinned closed-form test, under-specified by convention | `fft2` plus magnitude statistics are internally consistent. `fftshift` does not change magnitudes. Entropy over magnitude histograms is sensitive to sparse spectra and the DC component. |
| GLRLM | Correct for current four-direction summed convention | The 3x3 hand test, non-square diagonal sanity check, one-row/one-column checks, and disconnected-pixel check all pass independent derivation. Direction aggregation and quantisation should stay explicit parameters if users need alternate conventions. |

## Hand-Computed Test References

| Test | Re-derived result | Verdict |
| --- | --- | --- |
| `test_glrlm_3x3_uniform_object_matches_hand_computation` | Horizontal: 3 runs of length 3. Vertical: 3 runs of length 3. Each diagonal family: lengths `[1, 2, 3, 2, 1]`. Summed matrix at zero-based row 8: `P[8,1]=4`, `P[8,2]=4`, `P[8,3]=8`, `TR=16`. With one-based `g=9`: `RLNU=6`, `GLNU=16`, `LGLRE=1/81`, `HGLRE=81`. | Correct |
| `test_fft_constant_rectangle_matches_closed_form` | The bbox crop is a 4x4 array filled with 5. `fft2` has one nonzero DC coefficient of `16*5=80`; `fftshift` only moves it. Magnitudes have mean `80/16=5`, population std `sqrt(375)`, energy `80^2=6400`. | Correct |
| `test_lbp_uniform_object_has_zero_std_and_unit_energy` | For a constant image around the object, skimage default LBP with `P=8`, `R=1` gives code `255` for all sampled object pixels. Therefore std is `0`, energy is `1`, entropy is `0`. | Assertions correct; comment fixed |

## GLRLM Checks

The implementation uses:

- quantised gray levels `0..n_levels-1` in the matrix rows;
- one-based gray weighting `g=1..n_levels` for LGLRE/HGLRE;
- background sentinel `-1`, filtered before run accumulation;
- four directions: horizontal, vertical, down-right diagonals, down-left
  diagonals;
- summed matrices before computing the four features.

Independent sanity checks:

| Case | Result |
| --- | --- |
| 3x5 non-square two-gray matrix | `np.diagonal` extraction matched an independent coordinate-walk implementation exactly. Nonzero entries: gray 0 had length counts `(1:11, 2:9, 3:1)`; gray 1 had `(1:11, 2:4, 3:3)`. |
| One-row object | Matched independent coordinate-walk implementation. Diagonal directions contribute singleton runs, which is expected under four-direction summation. |
| One-column object | Matched one-row behaviour after transposition. |
| Two disconnected same-gray pixels | No run crosses the background sentinel. All counted runs are length 1. |

The four 2D directions are the right default for direction-agnostic 2D texture.
The one-based `g` convention is also the right default for the formulas used by
the implementation because it avoids a zero gray-level denominator. The part
that is convention-dependent is aggregation: computing features from the summed
matrix is valid, but some libraries average per-direction feature values
instead. If comparability to another package matters, expose an aggregation
parameter.

Suggested future parameters:

- `glrlm_directions=("h", "v", "diag_down", "diag_up")`
- `glrlm_aggregation="sum_matrix"` or `"mean_features"`
- `glrlm_quantization="image_max"` or `"fixed_range"`

## FFT Checks

`np.fft.fft2` is the safer default than `np.fft.rfft2` for the current feature
definitions. The current mean/std/energy/skew/kurtosis are over the full 2D
magnitude spectrum. Switching to `rfft2` would cut compute but change the bin
population and require weighting mirrored frequencies to preserve comparable
statistics.

The implementation uses `abs(F)`, not `abs(F)**2`, for mean/std/skew/kurtosis.
Energy is then computed as `sum(abs(F)**2)`. This is internally consistent and
matches the current mapping, but it should be documented directly in the code
or parameter metadata.

Entropy currently uses:

```python
hist, _ = np.histogram(F, bins=n_bins)
```

Because no range is provided, NumPy uses each object's min/max magnitude range.
That avoids absolute-scale collapse, but spectra with a dominant DC component
still put many values in the lowest bin and one/few values in high bins. This
is a reasonable first definition, not a universal texture entropy definition.

Possible alternatives if FFT entropy becomes important:

- use `np.log1p(abs(F))` before histogramming;
- remove or separately report the DC component;
- normalise magnitudes to a probability distribution and compute entropy over
  spectral power;
- expose `fft_entropy_mode="linear_hist" | "log_hist" | "power_prob"`.

The bbox crop with object mask and no windowing is a defensible default because
it keeps the computation local and cheap. It is also a convention choice. A
windowed crop or a padded fixed-size crop would produce different values.

## LBP Checks

`local_binary_pattern(image, P=8, R=1, method="default")` is the correct
skimage call for the current 3x3-neighbourhood mapping.

The implementation computes the LBP image once over the full image, then samples
LBP values inside each label. This means object-border pixels encode nearby
background or neighbouring-object pixels. That is usually preferable for a
whole-image workflow because it avoids artificial crop-border padding, but it
is not the same as computing LBP inside each object crop after masking outside
the object.

Observed sanity contrast:

- full constant image around a disk: all object LBP codes are `255`, std `0`,
  energy `1`;
- constant object with zero background: object-border pixels get many different
  LBP codes, std about `81`, energy about `0.383` for the tested disk.

Alternative methods:

- `method="uniform"`: lower-dimensional histogram, often better for compact
  texture summaries;
- `method="ror"`: rotation-invariant codes;
- `method="nri_uniform"`: non-rotation-invariant uniform patterns;
- `method="var"`: local variance, not a code histogram replacement.

The current API already exposes `lbp_method`, so the default can remain
`"default"` while allowing explicit alternatives.

## Forbidden-Name Sweep

Checked:

- `workflows/cell_analysis/steps/extract_features.py`
- `workflows/cell_analysis/tests/test_extract_features.py`
- last three commit messages in `.git/logs/HEAD`

Result: no forbidden source-name hits in those targets.

Last three commit messages checked:

```text
workflows: rewrite cell_analysis extract_features (Tier A + extras)
tests: unit-test cell_analysis extract_features
workflows: add LBP, FFT, GLRLM texture extras to extract_features
```

## Performance

Measured on the 2048 x 2048, 1000-disk synthetic image used by the benchmark
scripts. Timings below call the real `extract_features.run`, so they include
the default regionprops/derived feature cost.

| Run | Time ms | Increment over default |
| --- | ---: | ---: |
| Default features only | 968.9 | 0.0 |
| LBP only | 2215.7 | 1246.8 |
| FFT only | 1540.6 | 571.7 |
| GLRLM only | 2741.3 | 1772.4 |
| LBP + FFT + GLRLM | 4223.0 | 3254.1 |

Comparison to the earlier benchmark:

- LBP remains expensive because stats are still computed per object with scipy
  skew/kurtosis. It can likely use the same per-label histogram strategy as
  statistical texture.
- FFT is moderate and close to the earlier standalone estimate.
- GLRLM is now four directions, so it is much larger than the older
  one-direction placeholder. It is the dominant new optional texture cost.

For a 1000-object 2K image, enabling all three texture extras adds about 3.25 s
on this machine. That is acceptable for opt-in analysis but too heavy for the
default feature set.

## Proposed Additional Tests

1. `test_glrlm_nonsquare_two_gray_matches_manual_matrix`
   - Use a 3x5 object with two gray levels.
   - Assert the exact nonzero run counts and final features.
   - This catches diagonal range mistakes on rectangular crops.

2. `test_glrlm_degenerate_shapes_do_not_make_long_runs`
   - Include one-row, one-column, and disconnected-pixel masks.
   - Assert no background-separated pixels are merged into longer runs.

3. `test_fft_object_touching_image_border_runs_and_matches_bbox_crop`
   - Place a constant rectangle at the image edge.
   - Assert the same closed-form FFT values as an interior rectangle of the
     same bbox shape.

4. `test_lbp_object_border_sees_image_context`
   - Compare a full constant image to a constant object on zero background.
   - Assert the first has one LBP code and the second has multiple codes.
   - This locks in the image-wide LBP convention.

5. `test_lbp_uniform_method_changes_code_range`
   - Run `lbp_method="default"` and `lbp_method="uniform"` on the same object.
   - Assert both emit finite columns and distinct distributions.

## Prioritised Fixes

1. Add the non-square two-gray-level GLRLM test.
2. Add the LBP image-context test and document the convention in the extractor
   docstring.
3. Add the FFT border-object test.
4. Vectorise LBP histogram statistics for the default `P=8` path.
5. Consider exposing GLRLM aggregation and FFT entropy modes if users need
   comparability outside this workflow.

## Code Change Made

Patched the LBP test docstring so it states the correct skimage default code
for equal neighbours. No implementation code change was needed.

