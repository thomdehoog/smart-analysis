# Feature Benchmark Review

Date: 2026-04-27

Reviewed scripts:

- `C:\Users\t.de\bench_features.py`
- `C:\Users\t.de\bench_features_v2.py`

Reference docs:

- `FEATURE_SIDE_BY_SIDE.md`
- `FEATURE_LIBRARY_MAPPING.md`
- `FEATURE_MEASURES_REPORT.md`

Runtime:

- `C:\ProgramData\MinicondaZMB\envs\dino3_test\python.exe`
- `skimage 0.26.0`, `scipy 1.17.1`, `numpy 2.4.4`

I added an assertion-style naive-vs-bbox equivalence check to
`C:\Users\t.de\bench_features_v2.py`. The check stores each tier's output,
compares naive and bbox results with `np.isclose(..., rtol=1e-10, atol=1e-10,
equal_nan=True)`, and reports max absolute/relative error and the number of
different values.

## Executive Summary

The bbox optimisation in v2 is correct for six of the seven compared tiers.
The only divergence is local background. That is not a bbox bug: the original
v1-style `grey_dilation(label_image)` shortcut is not an exact per-object
collar. It can assign shared collar pixels according to the maximum label id,
not according to the object being measured. The bbox version matched an exact
per-object full-image dilation check for the sampled divergent labels.

Main formula issues:

- `orientation_deg = (np.degrees(orientation) + 180) % 180` is likely wrong if
  the desired angle is 0-180 degrees from the x-axis. For skimage orientation,
  use `(90 - np.degrees(orientation)) % 180`.
- `intensity_total = intensity_mean * area` is correct only when `area` is a
  pixel count. If `spacing=` is used, total intensity must be an actual pixel
  sum or `intensity_mean * num_pixels`.
- Statistical texture energy currently returns public "uniformity". The IN
  Carta formula includes a factor of 1000. Keep the public-library name if we
  omit the factor.
- GLCM is not masked correctly. Multiplying the bbox crop by the object mask
  injects zero-valued background transitions into the texture matrix.
- FFT statistics are under-specified. Entropy over raw floating FFT magnitudes
  is especially weak unless we define binning/normalisation.
- GLRLM horizontal matrix construction is correct for the implemented one
  direction, including one-based gray-level weighting and run-length indexing.
  It is incomplete for a production GLRLM because it uses only horizontal
  runs.
- Chord features are approximations, not formula-equivalent. `skel.sum()` is
  not the longest skeleton path.

Most important optimisation:

- Replace per-object `scipy.stats.skew/kurtosis` statistical texture loops with
  a single per-label intensity histogram via `np.bincount`. On the same 2K
  synthetic image this reduced Tier 7 statistical texture from 570.1 ms to
  18.5 ms, a 30.8x speedup, with numerical equivalence to scipy at
  `max_abs = 6.2e-15`.

## Timings Reproduced

### Full v1 Script

`bench_features.py` on 1000 disk labels, 2048 x 2048, mean area 291 px:

| Feature group | Time ms |
| --- | ---: |
| Tier2 derived columns | 0.1 |
| Tier4 KDTree | 4.8 |
| Tier5 global background | 8.4 |
| Tier12 chord features | 10.2 |
| Tier9 GLCM | 136.5 |
| Tier11 GLRLM horizontal | 318.9 |
| Tier10 FFT | 638.5 |
| Tier1+3 regionprops_table full property set | 952.4 |
| Tier6 Prewitt | 1043.9 |
| Tier6 Roberts | 1046.6 |
| Tier8 radius of gyration + spreading | 1053.8 |
| Tier8 Pearson r | 1598.2 |
| Tier7 statistical texture | 1657.8 |
| Tier7 LBP | 1967.1 |
| Tier5 local background | 2187.8 |
| Total | 12624.9 |

### v2 Naive vs Bbox

`bench_features_v2.py` after adding equivalence checks:

| Feature | Naive ms | Bbox ms | Speedup |
| --- | ---: | ---: | ---: |
| Tier5 local-bg collar | 2160.3 | 148.2 | 14.6x |
| Tier6 Prewitt magnitude | 1005.4 | 52.5 | 19.1x |
| Tier6 Roberts magnitude | 1000.7 | 45.9 | 21.8x |
| Tier7 statistical texture | 1607.1 | 543.9 | 3.0x |
| Tier7 LBP + 6 stats | 1947.9 | 831.5 | 2.3x |
| Tier8 Pearson r | 1533.7 | 139.8 | 11.0x |
| Tier8 Rg + spreading | 6601.8 | 21.9 | 301.8x |
| Total | 15856.9 | 1783.7 | 8.9x |

`find_objects` setup was 6.4 ms.

### v2 Equivalence Check

| Feature | Status | Max abs | Max rel | Different values |
| --- | --- | ---: | ---: | ---: |
| Tier5 local-bg collar | FAIL | 0.364 | 0.00579 | 142 |
| Tier6 Prewitt magnitude | OK | 0 | 0 | 0 |
| Tier6 Roberts magnitude | OK | 0 | 0 | 0 |
| Tier7 LBP + 6 stats | OK | 0 | 0 | 0 |
| Tier7 statistical texture | OK | 0 | 0 | 0 |
| Tier8 Pearson r | OK | 0 | 0 | 0 |
| Tier8 Rg + spreading | OK | 0 | 0 | 0 |

For the local-background divergence, I compared the first ten divergent labels
against an exact full-image per-object dilation:

- bbox matched exact full-image dilation for all checked labels.
- `grey_dilation` differed by up to 0.364 intensity units in the checked set.

Verdict: the bbox implementation is the correct one; the v1 `grey_dilation`
shortcut is the divergent implementation.

## Per-Tier Verdict

| Tier | Verdict | Notes |
| --- | --- | --- |
| Tier1+3 regionprops_table | Correct with caveats | Good source for native morphology/intensity. Avoid `coords` for production Rg/spreading because it is memory-heavy. Use `num_pixels` if `spacing=` is enabled. |
| Tier2 derived columns | Bug + suggestions | Circularity, aspect ratio, CV are fine. `intensity_total = mean * area` breaks when area is physical. Orientation conversion is likely x/y swapped for an x-axis 0-180 convention. |
| Tier4 KDTree spatial | Correct with caveats | Correct in pixels. For production, pass centroids in physical units and define behaviour for fewer than 2 or 6 objects. |
| Tier5 global background | Correct | `image[label_image == 0].mean()` matches the mapped formula. |
| Tier5 local background | v1 bug, v2 correct | v1 `grey_dilation` shortcut is not exact. v2 bbox collar matches exact per-label dilation. For holes, fill the object before making the outside collar. |
| Tier6 Prewitt/Roberts | Correct against mapping | Image-wide filter plus per-object mean is equivalent between v1/v2. Exact vendor kernel normalization is still a documented convention risk. |
| Tier7 statistical texture | Correct for public naming, not exact vendor scaling | Uniformity omits the vendor's `1000 *` factor. Entropy uses skimage default base 2; vendor formula text uses `log`, so exact base is unresolved. Replace scipy loops with vectorised histogram formulas. |
| Tier7 LBP | Correct against mapping | `P=8`, `R=1`, `method='default'` matches the mapped 3x3 LBP choice. Same entropy-base caveat. Image-wide LBP means object-border pixels see surrounding background. |
| Tier8 Pearson r | Correct with edge cases | Formula is correct. Guard single-pixel and constant-intensity objects. A per-label sum formula will be faster than `scipy.stats.pearsonr` loops. |
| Tier8 Rg + spreading | Correct in pixel units | v2 bbox result matches naive. Reuse bbox crops and do not request `regionprops_table('coords')`. Apply pixel spacing to coordinates before `d2` if physical units are required. |
| Tier9 GLCM | Bug if treated as object texture | `crop * mask` creates artificial zero-gray background transitions. Use a masked pair counter or skip GLCM because it is not a vendor feature. |
| Tier10 FFT | Suggestion / unresolved | The mapped composition is implemented, but windowing, crop padding, magnitude vs power, normalization, and entropy binning are not defined by the docs. Do not ship as a default. |
| Tier11 GLRLM | Correct but incomplete | Horizontal run matrix construction is correct. Need 4 directions, documented quantization, and tests on tiny hand-computed masks before production. |
| Tier12 chord | Bug / approximation | Straight chord is endpoint max on skeleton; curved chord is skeleton pixel count. The mapped feature requires endpoint selection plus longest skeleton path/geodesic length. |

## Correctness Details

### Orientation

Skimage `orientation` is measured relative to the row axis, not the x-axis. If
we want a 0-180 degree image x-axis convention, use:

```python
orientation_deg = (90.0 - np.degrees(rp["orientation"])) % 180.0
```

The current benchmark uses:

```python
orientation_deg = (np.degrees(rp["orientation"]) + 180) % 180
```

That maps vertical major axes to 0 degrees and horizontal major axes to 90
degrees. This is probably not the intended public `orientation_deg` convention.

### Total Intensity

This is safe only without physical spacing:

```python
intensity_total = intensity_mean * area
```

Production code should use one of these:

```python
intensity_total = intensity_mean * num_pixels
# or, when per-label pixels are already available:
intensity_total = np.bincount(label_image[mask], weights=image[mask])[labels]
```

### Local Background

Do not use `grey_dilation(label_image)` for exact local collars. It assigns a
background pixel to one label, usually the highest label in the footprint, even
when several objects' collars include that pixel.

Use a bbox crop per object. For objects with holes, fill the object before
making the outside collar so internal holes are not counted as local
background:

```python
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, disk

def local_bg_mean_bbox(label_image, image, slices, labels, radius=5):
    out = np.full(len(labels), np.nan, dtype=np.float64)
    footprint = disk(radius)
    for i, lab in enumerate(labels):
        sl = slices[lab - 1]
        if sl is None:
            continue
        sy, sx = sl
        sp = (
            slice(max(0, sy.start - radius - 1), min(label_image.shape[0], sy.stop + radius + 1)),
            slice(max(0, sx.start - radius - 1), min(label_image.shape[1], sx.stop + radius + 1)),
        )
        crop_lab = label_image[sp]
        obj = crop_lab == lab
        filled = binary_fill_holes(obj)
        collar = dilation(filled, footprint=footprint) & ~filled & (crop_lab == 0)
        if collar.any():
            out[i] = image[sp][collar].mean()
    return out
```

### Statistical Texture

The scipy loop is correct but too slow. A per-label histogram gives the same
uniformity, entropy, skewness, and Fisher excess kurtosis:

```python
def statistical_texture_hist(label_image, image_u8, labels, n_bins=256):
    mask = label_image > 0
    lut = np.full(int(label_image.max()) + 1, -1, dtype=np.int64)
    lut[labels] = np.arange(len(labels))
    rows = lut[label_image[mask]]
    vals = image_u8[mask].astype(np.int64)

    counts = np.bincount(rows * n_bins + vals, minlength=len(labels) * n_bins)
    counts = counts.reshape(len(labels), n_bins).astype(np.float64)
    n = counts.sum(axis=1)
    p = np.divide(counts, n[:, None], out=np.zeros_like(counts), where=n[:, None] > 0)

    levels = np.arange(n_bins, dtype=np.float64)
    uniformity = (p * p).sum(axis=1)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    entropy = -(p * logp).sum(axis=1)

    mean = (p * levels).sum(axis=1)
    centered = levels[None, :] - mean[:, None]
    m2 = (p * centered**2).sum(axis=1)
    m3 = (p * centered**3).sum(axis=1)
    m4 = (p * centered**4).sum(axis=1)

    skewness = np.full(len(labels), np.nan)
    kurtosis = np.full(len(labels), np.nan)
    nz = m2 > 0
    skewness[nz] = m3[nz] / (m2[nz] ** 1.5)
    kurtosis[nz] = m4[nz] / (m2[nz] ** 2) - 3.0
    return np.column_stack([uniformity, entropy, skewness, kurtosis])
```

Benchmark on the disk synthetic image:

| Implementation | Time ms | Equivalence |
| --- | ---: | --- |
| bbox loop with scipy stats | 570.1 | reference |
| NumPy histogram | 18.5 | `allclose=True`, `max_abs=6.2e-15` |

### Masked GLCM

If GLCM is kept as a bonus feature, do not multiply the crop by the mask. Count
only pixel pairs where both pixels are inside the object:

```python
def graycomatrix_masked_0deg(crop_q, obj_mask, levels, symmetric=True, normed=True):
    valid = obj_mask[:, :-1] & obj_mask[:, 1:]
    left = crop_q[:, :-1][valid].astype(np.int64)
    right = crop_q[:, 1:][valid].astype(np.int64)
    P = np.bincount(left * levels + right, minlength=levels * levels)
    P = P.reshape(levels, levels).astype(np.float64)
    if symmetric:
        P = P + P.T
    if normed and P.sum() > 0:
        P /= P.sum()
    return P
```

### GLRLM

The current one-direction matrix construction has no off-by-one bug:

- background is encoded as `-1` and skipped;
- run lengths are stored in column `L - 1`;
- gray levels are weighted one-based as `g = 1..n_levels`, which avoids the
  zero-gray divide-by-zero problem in LGLRE.

It can be vectorised for horizontal runs by finding all row runs once and
aggregating by label. On the same disk benchmark:

| Implementation | Time ms | Equivalence |
| --- | ---: | --- |
| per-object horizontal GLRLM | 318.4 | reference |
| vector horizontal GLRLM | 54.4 | `allclose=True`, `max_abs=3.1e-17` |

The same idea can be extended to vertical and diagonal directions, but the
diagonal version needs careful line traversal or shifted-pair logic.

### Chord Features

The benchmark implementation is useful only as a timing placeholder:

```python
chd_crv[lab - 1] = skel.sum()
```

This is not the curved chord formula. For production, use skeleton graph
endpoints and compute the longest shortest path over the skeleton graph, with
edge weights of 1 for orthogonal steps and `sqrt(2)` for diagonal steps. The
straight chord should then be the Euclidean distance between the selected path
endpoints, not just the max endpoint pair unless that convention is explicitly
chosen.

## Synthetic Data Realism

The default benchmark places non-overlapping disks away from borders. It is
good for throughput smoke testing, but it misses real cell-analysis cases:

- elongated and irregular cells;
- branched or neurite-like objects;
- holes;
- objects touching the image border;
- single-pixel objects;
- very large objects;
- close objects with overlapping local-background collars;
- labels with id gaps;
- anisotropic pixel spacing.

I ran a one-off stress generator with 1000 labels containing perturbed ellipses,
branched objects, holes, border objects, single pixels, and one large object.
Mean area was 523 px; max area was 18540 px.

| Feature | Stress time ms |
| --- | ---: |
| `find_objects` | 7.0 |
| local-bg bbox | 156.8 |
| Prewitt bbox | 54.1 |
| Roberts bbox | 47.7 |
| statistical texture bbox with scipy | 566.6 |
| statistical texture NumPy histogram | 22.0 |
| LBP bbox | 852.0 |
| Pearson bbox | 149.8 |
| Rg + spreading bbox | 29.4 |
| chord approximation bbox | 59.3 |

The chord placeholder rose from 10.2 ms on disks to 59.3 ms on the stress
generator. An exact skeleton longest-path implementation will cost more than
this placeholder.

Recommended benchmark extension:

- Add a `scenario` argument: `disks`, `stress`, `edge_cases`.
- Keep `disks` for repeatable throughput comparisons.
- Use `stress` before accepting morphology/texture timings.
- Use `edge_cases` for correctness assertions, not speed.

## Multi-Core Assessment

Joblib is safe only for independent per-object bbox work. It is not useful for
the image-wide operations themselves.

| Tier | Parallelise? | Expected 8-core effect |
| --- | --- | --- |
| `regionprops_table` | No | Library call plus memory pressure; external parallelism unlikely to help. |
| Derived columns | No | Already sub-ms. |
| KDTree spatial | No | Already ~5 ms for 1000 labels. |
| Global background | No | Single vectorised reduction. |
| Local background bbox | Yes | Independent small crops. Threads may give 2-4x; processes need memmapping to avoid copying images. |
| Prewitt/Roberts | Mostly no | Filter image once. Per-label means should be vectorised with `np.bincount`, not joblib. |
| Statistical texture | No | Use histogram vectorisation; joblib is the wrong optimisation. |
| LBP stats | Maybe | LBP image is image-wide. Per-object stats can be parallelised, but vector histograms are preferable. |
| Pearson r | Maybe | Independent per-object calculations. A vectorised per-label sum formula is likely better than joblib. |
| Rg + spreading | No for current scale | Bbox version is ~22-29 ms; joblib overhead will dominate. |
| FFT per object | Yes if enabled | Independent variable-size crops. Use threads first; processes only with shared/memmapped arrays. |
| GLRLM | Yes, but vectorise first | Per-object version can parallelise; row-run aggregation is faster for horizontal/vertical directions. |
| Chord exact path | Yes | Likely the best candidate once exact skeleton graph paths are implemented. |

For the current v2 seven-tier benchmark, 8-core joblib will not produce an 8x
wall-clock speedup because image-wide filters, LBP image creation, memory
bandwidth, and Python scheduling remain serial costs. A realistic estimate is:

- current code, no vectorisation: 1.5-2.5x total improvement at best;
- after statistical texture vectorisation: perhaps 1.2-1.8x total improvement;
- exact chord/FFT/GLRLM-heavy optional texture runs: 3-5x is plausible if arrays
  are shared and tasks are chunky enough.

## Edge Cases Missing From The Benchmark

Add correctness tests for:

- single-pixel objects: zero `Rg`, undefined axes, invalid Pearson, undefined
  skew/kurt;
- constant-intensity objects: Pearson/skew/kurt division by zero;
- objects touching image border: truncated local collar and FFT crop;
- holes: local background must not include internal holes unless explicitly
  intended;
- close objects: overlapping collars, neighbour exclusion;
- labels with id gaps: do not assume rows are `label - 1`;
- very large objects: bbox crops can dominate FFT, GLCM, GLRLM, and chord time;
- anisotropic pixel spacing: coordinates must be scaled before distance,
  area, Rg, spreading, and neighbour features;
- fewer than 6 objects: KDTree `k=6` logic must degrade cleanly.

## Prioritised Changes For `extract_features.py`

1. Add a label-indexing layer:
   - `labels = np.array(sorted(np.unique(label_image)[1:]))`
   - never assume `row = label - 1` in production output.

2. Add full Tier A `regionprops_table` properties, but avoid `coords` by
   default:
   - use `bbox`, `centroid`, `weighted_centroid`, `num_pixels`, area/axis/
     perimeter/intensity properties;
   - use `find_objects` for optional per-object bbox work.

3. Add derived core with unit-safe formulas:
   - `circularity = 4*pi*area/perimeter_crofton**2`;
   - `aspect_ratio = axis_major_length/axis_minor_length`;
   - `orientation_deg = (90 - degrees(orientation)) % 180`;
   - `intensity_total = intensity_mean * num_pixels` or direct weighted
     bincount sum;
   - `intensity_cv = intensity_std/intensity_mean`.

4. Implement local background with bbox crops, not `grey_dilation`; fill holes
   before creating the outside collar.

5. Implement statistical texture with per-label histograms, not per-object
   scipy loops.

6. Implement Prewitt/Roberts as image-wide filters plus per-label
   `np.bincount` means.

7. Implement Rg/spreading from bbox-local coordinates and pixel spacing; do
   not request `regionprops_table('coords')`.

8. Defer or opt-in:
   - GLRLM: use vector run aggregation and four directions, with tiny
     hand-computed tests.
   - Chord: replace placeholder with skeleton longest-path code.
   - FFT: require explicit binning/windowing/normalisation choices.
   - GLCM: either skip or use masked pair counting.

9. Add tests before wiring to YAML:
   - tiny masks with hand-computed intensity/stat texture;
   - local background with a neighbour and with a hole;
   - GLRLM on a tiny quantized matrix;
   - orientation on horizontal and vertical ellipses;
   - label id gaps.
