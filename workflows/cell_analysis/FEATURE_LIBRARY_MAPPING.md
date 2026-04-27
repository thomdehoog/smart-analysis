# Feature Library Mapping — Public-API Implementation Plan

Goal: implement the feature set entirely in **scikit-image / NumPy / SciPy**, using
their standard names and citing their public documentation. No vendor reference
required in the resulting code or docs.

Companion notes:
- `FEATURE_MEASURES_REPORT.md` — vendor-source reference (formulas + status codes).
- `FEATURE_IMPLEMENTATION_SURVEY.md` — broader package survey.

Scope: 2D, single label image, single intensity image (with explicit pixel
spacing). Multichannel, parent–child linked masks, neurite/fiber, and 3D are
out of scope here.

Verified runtime: `dino3_test` (`skimage 0.26.0`, `scipy 1.17.1`, `numpy 2.4.4`).

---

## TL;DR

- **~95 % of the desired feature set is reachable from `skimage`/`numpy`/`scipy`** with no new dependency.
- **One real gap**: GLRLM (run-length texture). Either skip, write ~50 lines of NumPy, or add PyRadiomics.
- **Drop one feature outright**: vendor "Compactness" (= `2π·R_g²/A`) is non-standard and redundant with Gyration Radius + Area; the term "compactness" is overloaded in the literature, so don't ship it under that name.
- **Two features are vendor-specific composites and need named-but-custom code**: vendor "Intensity Spreading" (intensity-weighted radial variance) and the chord/curved-chord pair (skeleton + geodesic length). Implement only if requested; document the formula inline.

---

## Naming convention shift

When the vendor name has a public-domain equivalent, use the public name in column
headers, docstrings, and YAML keys. This removes the vendor reference at the API
boundary.

| Vendor name (avoid in code) | Public/library name (use) | Rationale |
|---|---|---|
| Form Factor v2 | **Circularity** (= `4πA/P²`) | Standard ImageJ/Fiji name; same formula. |
| Elongation v2 | **Aspect ratio** (= `axis_major / axis_minor`) | ImageJ "AR"; universal. |
| Chord Ratio v2 | Curved-to-straight path length ratio | Plain English. |
| Straight Chord | Geodesic endpoint distance | Skeleton-derived. |
| Curved Chord | Skeleton path length | Skeleton-derived. |
| First / Second Principal Axis Length | **Major / Minor axis length** | skimage names. |
| Gyration Radius | **Radius of gyration** | Physics standard term. |
| Compactness (vendor: `2π·R_g²/A`) | *(do not ship under this name)* | "Compactness" usually means `4πA/P²` (= Circularity). The vendor formula is unusual and adds nothing on top of `R_g` + Area. |
| Center of Gravity X / Y | **Weighted centroid** (skimage) / **center of mass** (scipy) | Standard. |
| Major Axis Angle v2 | **Orientation** (degrees, 0–180) | skimage gives radians ±π/2; convert. |
| Mean Intensity, Total Intensity, Intensity SD, Max Intensity, Median Intensity | same names | Already universal. |
| Intensity CV | **Coefficient of variation** | Universal stats term. |
| Intensity Spreading | **Intensity radial variance (normalised)** | Vendor-specific composite — keep the formula, rename the column. |
| Background Intensity (Local / Global) | **Background mean (local collar / global)** | Plain English. |
| MIBR | `mean_over_local_bg = I_avg / I_local` | Plain English. |
| TIBR | `total_over_local_bg = I_sum / I_local` | Plain English. |
| MILB | `mean_minus_local_bg = I_avg − I_local` | Plain English. |
| TILB v2 | `total_minus_local_bg = (I_avg − I_local)·A` | Plain English. |
| Pearson correlation | **Pearson r** (`scipy.stats.pearsonr`) | Universal. |
| Distance to Nearest Neighbour | same | Universal; SciPy KDTree. |
| Short / Mid / Long Range Neighbour Count | **Neighbour count within radius** (configurable) | Hard-coded radii (5/50/250 µm) become parameters. |

---

## Tier A — direct `skimage.measure.regionprops_table` properties

One call. Already verified on `dino3_test`. Documentation:
<https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops_table>.

| Feature | Property name | Notes |
|---|---|---|
| Pixel area / pixel count | `area`, `num_pixels` | Pass `spacing=(dy, dx)` for physical units. |
| Bounding box | `bbox` | (min_row, min_col, max_row, max_col). |
| Centroid | `centroid` | Geometric. |
| Weighted centroid | `weighted_centroid` | Intensity-weighted. |
| Equivalent disc diameter | `equivalent_diameter_area` | `2·√(A/π)`. |
| Max Feret diameter | `feret_diameter_max` | "Diameter" / longest object span. |
| Perimeter | `perimeter` | Pixel-edge sum (Crofton-like at boundary). |
| Perimeter (less grid-biased) | `perimeter_crofton` | Prefer for circularity. |
| Major axis length | `axis_major_length` (alias `major_axis_length`) | Equivalent ellipse fit. |
| Minor axis length | `axis_minor_length` (alias `minor_axis_length`) | Equivalent ellipse fit. |
| Orientation | `orientation` | Radians, range `[-π/2, π/2]`. |
| Eccentricity | `eccentricity` | 0 = disc, →1 = elongated. |
| Solidity | `solidity` | `area / area_convex`. |
| Extent | `extent` | `area / bbox_area`. |
| Convex area | `area_convex` | |
| Filled area | `area_filled` | Holes filled. |
| Hu invariant moments | `moments_hu` | 7-vector. |
| Mean intensity | `intensity_mean` (alias `mean_intensity`) | |
| Min intensity | `intensity_min` (alias `min_intensity`) | |
| Max intensity | `intensity_max` (alias `max_intensity`) | |
| Std intensity | `intensity_std` | |
| Median intensity | `intensity_median` | Confirmed available in 0.26.0. |

> **Pixel spacing:** pass `spacing=(dy_um, dx_um)` to `regionprops_table` so that
> `area`, `perimeter`, `axis_*_length`, `feret_diameter_max`,
> `equivalent_diameter_area`, `centroid`, and `weighted_centroid` come out in µm /
> µm². This is the cleanest way to avoid the vendor's pixel-vs-physical-unit
> ambiguity.

---

## Tier B — composed from public primitives (no new dependency)

A handful of stdlib-style calls each. All citable via the function URLs given.

### Derived shape

| Feature | One-line implementation | Reference |
|---|---|---|
| Circularity (= vendor "Form Factor") | `4 * np.pi * area / perimeter_crofton**2` | [skimage perimeter_crofton](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.perimeter_crofton) |
| Aspect ratio (= vendor "Elongation v2") | `axis_major_length / axis_minor_length` | [skimage regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) |
| Orientation in degrees, 0–180 | `(np.degrees(orientation) + 180) % 180` | skimage convention. |
| Radius of gyration (= vendor "Gyration Radius") | `np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))` | `coords` from `regionprops_table('coords')`; standard physics. |

### Derived intensity

| Feature | Implementation | Reference |
|---|---|---|
| Total / integrated intensity | `intensity_mean * area` *(if `area` is in pixels)* or `np.sum(intensity_image[mask])` | Same as ImageJ "RawIntDen". |
| Coefficient of variation | `intensity_std / intensity_mean` (guard zero) | NumPy. |

### Background

`I_global` and `I_local` are not in any library by name, but each step is.

| Feature | Implementation sketch | Refs |
|---|---|---|
| Global background mean | `intensity_image[label_image == 0].mean()` | NumPy mask. |
| Local background mean (5-px collar, neighbour-excluded) | `dilated = binary_dilation(mask, footprint=disk(5))`; `collar = dilated & ~mask & (label_image == 0)`; `I_local = intensity_image[collar].mean()` | [`skimage.morphology.binary_dilation`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation), [`disk`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.disk). The "5 px" should be a parameter. |
| `mean_over_local_bg`, `total_over_local_bg`, `mean_minus_local_bg`, `total_minus_local_bg` | Trivial arithmetic on `intensity_mean`, `area`, and the local background above. | NumPy. Guard divide-by-zero. |

### Spatial neighbours

| Feature | Implementation | Refs |
|---|---|---|
| Distance to nearest neighbour | `tree = cKDTree(centroids); d, _ = tree.query(centroids, k=2); d[:, 1]` | [`scipy.spatial.cKDTree.query`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html) |
| Mean distance to k nearest neighbours | `d, _ = tree.query(centroids, k=k+1); d[:, 1:].mean(axis=1)` | Same; default `k=5` matches vendor. |
| Neighbour count within radius `r` | `lens = [len(x)-1 for x in tree.query_ball_point(centroids, r=r)]` | [`query_ball_point`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html) |
| Distance to image border | `min(centroid_y, H-centroid_y, centroid_x, W-centroid_x)` | NumPy. |

> Pass centroids in physical units (multiply by spacing or use weighted_centroid
> after passing `spacing=`) so the radii (5 / 50 / 250 µm in the vendor docs)
> become real µm values.

### Colocalization

| Feature | Implementation | Refs |
|---|---|---|
| Pearson correlation between two channels per object | `pearsonr(image_a[mask], image_b[mask]).statistic` | [`scipy.stats.pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html) — algebraically identical to the vendor formula `S_XY / √(S_XX · S_YY)`. |

---

## Tier C — texture

Texture is the most divergent area. Recommend gating it behind an opt-in flag and
treating it as a separate module. Citations below remove the need to mention any
vendor.

### Statistical (intensity histogram of object pixels)

The vendor formulas reduce to standard moments and entropies of the per-object
pixel set. Verified algebraically below.

| Feature | Implementation | Equivalent standard quantity |
|---|---|---|
| **Energy / Uniformity** | `np.sum((np.bincount(pixels) / area)**2)` | "Uniformity" / "Angular second moment". Vendor formula `1000·(1/A²)·Σnₖ²` = `1000 × Uniformity`. |
| **Entropy** | `shannon_entropy(image_or_pixels, base=np.e)` | Shannon entropy. Vendor formula `−(Σnₖ·log nₖ − A·log A)/A` simplifies to `−Σpₖ·log pₖ`. |
| **Skewness** | `scipy.stats.skew(pixels, bias=True)` | Standard sample skewness. |
| **Kurtosis** | `scipy.stats.kurtosis(pixels, fisher=True, bias=True)` | Standard Fisher excess kurtosis (the `−3` is built in). |

References:
- [`skimage.measure.shannon_entropy`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.shannon_entropy)
- [`scipy.stats.skew`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)
- [`scipy.stats.kurtosis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)

### Gradient (Prewitt / Roberts magnitude)

| Feature | Implementation | Refs |
|---|---|---|
| Prewitt magnitude per object | `g = skimage.filters.prewitt(image); g[mask].mean()` | [`skimage.filters.prewitt`](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt) |
| Roberts magnitude per object | `g = skimage.filters.roberts(image); g[mask].mean()` | [`skimage.filters.roberts`](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts) |

`skimage.filters.prewitt` already returns the gradient magnitude `√(Gₓ² + G_y²)`
using the standard 3×3 Prewitt kernels; `roberts` uses the 2×2 cross.
Kernel-normalisation differences vs the vendor are documented in the skimage
source — call this out in the docstring rather than the README.

### LBP (local binary patterns)

| Feature | Implementation | Refs |
|---|---|---|
| LBP image | `lbp = local_binary_pattern(image, P=8, R=1, method='default')` | [`skimage.feature.local_binary_pattern`](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern) |
| LBP mean / std | `lbp[mask].mean()`, `.std()` | NumPy. |
| LBP energy / entropy | histogram of `lbp[mask]` → uniformity / Shannon entropy (same formulas as above) | NumPy + skimage. |
| LBP skewness / kurtosis | `scipy.stats.skew/kurtosis(lbp[mask])` | scipy.stats. |

> The vendor docs describe a 3×3 neighbourhood (P=8, R=1) with `method='default'`. Make these arguments explicit.

### FFT statistics

No named library function. Compose:

```python
crop = intensity_image[bbox] * mask[bbox]
F = np.abs(np.fft.fftshift(np.fft.fft2(crop)))
fft_mean   = F.mean()
fft_std    = F.std()
fft_energy = (F**2).sum()
fft_entr   = shannon_entropy(F)
fft_skew   = scipy.stats.skew(F.ravel())
fft_kurt   = scipy.stats.kurtosis(F.ravel())
```

References: [`numpy.fft.fft2`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html), `scipy.stats`. Document magnitude vs power, windowing (none by default), and the fact that the bounding-box crop is masked.

### GLRLM (run-length texture) — the only real gap

`skimage` ships GLCM (`graycomatrix` / `graycoprops`) but **not GLRLM**. Three options:

1. **Skip.** GLRLM features rarely add information beyond GLCM and statistical texture for cell images.
2. **Custom NumPy** (~50 lines): build the gray-level run-length matrix `P(g, r)` for the four standard directions, then compute `RLNU`, `LGLRE`, `HGLRE`, `GLNU` from the formulas already in `FEATURE_MEASURES_REPORT.md`. Cite Galloway, M.M. (1975), *Texture analysis using gray level run lengths.*
3. **PyRadiomics dependency** (heavy; brings SimpleITK transitively). Only if exact parity with a published radiomics workflow is required.

Recommendation: **option 1 or 2.** Skip first; if a user asks, ship a documented NumPy implementation behind an opt-in flag.

### GLCM (Haralick) — bonus, available

`skimage.feature.graycomatrix` + `graycoprops('contrast' | 'dissimilarity' | 'homogeneity' | 'energy' | 'correlation' | 'ASM')` covers Haralick-style features. Not in the vendor's named set, but worth offering since it's free.

---

## Tier D — explicit gaps and "do not ship"

| Feature | Decision | Reason |
|---|---|---|
| **Compactness (vendor: `2π·R_g²/A`)** | Do not ship | Non-standard definition. Use Circularity (`4πA/P²`) if user wants "compactness", or expose Radius of Gyration + Area separately. |
| **Curved Chord / Straight Chord / Chord Ratio v2** | Custom only on demand | Requires `skimage.morphology.medial_axis` or `skeletonize`, then a longest-path / geodesic-length walk. No single-call equivalent. Add only if a user asks. |
| **Intensity Spreading** | Custom; rename to `intensity_radial_variance_normalised` | `Σ(uᵢ·rᵢ²) / (⟨u⟩·N·R_g²)` over object pixels. ~5 lines of NumPy once you have `R_g`. |
| **Linked-target measures (Area Ratio, Cyto Translocation, etc.)** | Out of scope | Requires parent-child label pairing; not a single-mask workflow concern. |
| **Neurite / Fiber measures** | Out of scope | Vendor-specific target types. |
| **GLRLM** | Optional, custom NumPy | See Tier C. |

---

## Recommended naming for the output table

Suggested column-name template (snake_case, library-aligned):

```
label
area, area_convex, area_filled, num_pixels
bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
centroid_y, centroid_x
weighted_centroid_y, weighted_centroid_x
equivalent_diameter, feret_diameter_max
perimeter, perimeter_crofton
axis_major_length, axis_minor_length
orientation_deg                    # converted to 0–180
eccentricity, solidity, extent
circularity                         # = 4πA/P²
aspect_ratio                        # = axis_major / axis_minor
radius_of_gyration                  # custom
moments_hu_0 ... moments_hu_6
intensity_mean, intensity_std, intensity_min, intensity_max, intensity_median
intensity_total                     # mean × area or sum over mask
intensity_cv                        # std / mean
bg_global_mean
bg_local_mean
mean_over_local_bg, total_over_local_bg
mean_minus_local_bg, total_minus_local_bg
intensity_radial_variance_normalised
nn_distance, nn5_mean_distance
neighbours_within_5um, neighbours_within_50um, neighbours_within_250um
pearson_r_<chA>_<chB>
prewitt_magnitude_mean, roberts_magnitude_mean
lbp_mean, lbp_std, lbp_energy, lbp_entropy, lbp_skewness, lbp_kurtosis
intensity_uniformity, intensity_entropy, intensity_skewness, intensity_kurtosis
fft_mean, fft_std, fft_energy, fft_entropy, fft_skewness, fft_kurtosis
# (glrlm_* only if Tier-C option 2 is built)
```

This naming derives every column from a public-API or universal-statistics term.

---

## Implementation order

1. **Extend `regionprops_table` properties to all of Tier A.** Add `spacing=` so units are µm. (No new code beyond YAML/properties list.)
2. **Add Tier B derived columns** (circularity, aspect ratio, intensity_total, intensity_cv, orientation_deg). Trivial NumPy.
3. **Add Tier B background columns.** Local-collar code is the longest piece — make collar radius and "exclude other labels" both parameters.
4. **Add KDTree neighbours** (configurable radii in µm).
5. **Add Pearson r** (only meaningful once multi-channel input is supported).
6. **Behind opt-in flag: gradient + LBP + statistical texture + FFT.** All citable.
7. **Defer**: GLRLM (decide skip vs custom), chord features (decide skip vs custom), Intensity Radial Variance (decide skip vs custom).

Every function above has a public docs URL; the resulting code never has to name the vendor.
