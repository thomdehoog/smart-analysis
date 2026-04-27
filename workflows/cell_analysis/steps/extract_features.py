"""Extract Features -- per-object shape, intensity, spatial, and (opt-in)
texture/background features via skimage / numpy / scipy.

Always computed (Tier A native + Tier B derived):
    - regionprops_table properties listed in DEFAULT_PROPERTIES
    - circularity, aspect_ratio, orientation_deg, intensity_total, intensity_cv

Opt-in via the ``extras`` parameter (list of group names):
    "global_bg"     mean intensity outside all labels (broadcast as scalar
                    column); adds bg_global_mean.
    "local_bg"      5-px collar around each object, neighbour-excluded, with
                    holes filled before forming the collar; adds bg_local_mean,
                    mean_minus_local_bg, mean_over_local_bg,
                    total_minus_local_bg, total_over_local_bg.
    "gradients"     image-wide Prewitt and Roberts magnitudes summarised per
                    object via vectorised bincount means; adds
                    prewitt_magnitude_mean, roberts_magnitude_mean.
    "stat_texture"  per-object intensity histogram statistics computed in one
                    vectorised pass; adds intensity_uniformity,
                    intensity_entropy, intensity_skewness, intensity_kurtosis.
    "lbp"           local binary pattern image (P=8, R=1, default method) +
                    six per-object histogram statistics; adds lbp_mean,
                    lbp_std, lbp_energy, lbp_entropy, lbp_skewness,
                    lbp_kurtosis.
    "fft"           per-object 2D FFT on the bbox crop with the object mask
                    applied; six statistics over the magnitude spectrum;
                    adds fft_mean, fft_std, fft_energy, fft_entropy,
                    fft_skewness, fft_kurtosis. Entropy uses a 256-bin
                    histogram of magnitudes.
    "glrlm"         gray-level run-length matrix summed over four
                    directions (horizontal, vertical, +45, -45) with
                    one-based gray-level indexing to keep LGLRE finite;
                    adds glrlm_rlnu, glrlm_lglre, glrlm_hglre, glrlm_glnu.
    "rg_spread"     radius of gyration + normalised intensity radial variance,
                    computed from bbox-local coordinates (no regionprops
                    ``coords`` materialisation); adds radius_of_gyration,
                    intensity_radial_variance_normalised.

Inputs (from previous steps)
    pipeline_data["segment"]["masks"]    : int label image
    pipeline_data["preprocess"]["image"] : raw image used for intensity

Parameters (via YAML / **params)
    properties       : list of regionprops_table property names.
                       Default: DEFAULT_PROPERTIES (the Tier A set).
                       Must contain "label" for downstream selection to work.
    extras           : list of opt-in group names from the table above.
    pixel_size_um    : (dy, dx) tuple or None. When given, regionprops_table
                       returns area/perimeter/axis lengths/centroids in
                       physical units; intensity_total uses num_pixels so it
                       remains a true integrated intensity regardless.
    bg_radius        : int, default 5. Collar width in pixels for local_bg.
    n_intensity_bins : int, default 256. Histogram bins for stat_texture.
    lbp_P            : int, default 8.   Number of LBP sample points.
    lbp_R            : float, default 1. LBP sample radius in pixels.
    lbp_method       : str, default "default". skimage local_binary_pattern
                       method ("default" / "ror" / "uniform" / "var").
    glrlm_levels     : int, default 16. Gray levels for the run-length matrix.
    fft_entropy_bins : int, default 256. Magnitude histogram bins for fft.

Outputs (under pipeline_data["extract_features"])
    properties : dict[str, ndarray]    per-cell arrays aligned with "label"
    n_cells    : int

The output rows are indexed by their position in ``properties["label"]``;
do not assume row index equals label id.
"""

import numpy as np


METADATA = {
    "description": "Per-object features (shape / intensity / spatial / opt-in texture)",
    "version": "2.0",
}


DEFAULT_PROPERTIES = [
    "label",
    "bbox",
    "centroid", "weighted_centroid",
    "area", "num_pixels", "area_convex",
    "equivalent_diameter_area", "feret_diameter_max",
    "perimeter", "perimeter_crofton",
    "axis_major_length", "axis_minor_length", "orientation",
    "eccentricity", "solidity", "extent",
    "intensity_mean", "intensity_min", "intensity_max",
    "intensity_std", "intensity_median",
]


_VALID_EXTRAS = {
    "global_bg", "local_bg", "gradients", "stat_texture", "rg_spread",
    "lbp", "fft", "glrlm",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    from skimage.measure import regionprops_table

    verbose = pipeline_data["metadata"].get("verbose", 0)

    properties = list(params.get("properties", DEFAULT_PROPERTIES))
    extras = list(params.get("extras", []) or [])
    pixel_size_um = params.get("pixel_size_um")

    unknown = set(extras) - _VALID_EXTRAS
    if unknown:
        raise ValueError(
            f"Unknown extras {sorted(unknown)}. "
            f"Expected subset of {sorted(_VALID_EXTRAS)}."
        )

    masks = pipeline_data["segment"]["masks"]
    img = pipeline_data["preprocess"]["image"]

    spacing_kw = {"spacing": tuple(pixel_size_um)} if pixel_size_um else {}
    props = regionprops_table(
        masks, intensity_image=img, properties=properties, **spacing_kw
    )

    labels = np.asarray(props.get("label", []))
    n_cells = int(len(labels))

    if n_cells > 0:
        _add_derived(props)
        if extras:
            _add_extras(props, masks, img, labels, extras, params, pixel_size_um)

    if verbose >= 2:
        print(f"  [extract_features] cells: {n_cells}, "
              f"properties: {sorted(props)}")

    pipeline_data["extract_features"] = {
        "properties": props,
        "n_cells": n_cells,
    }
    return pipeline_data


# ---------------------------------------------------------------------------
# Tier B: always-on derived columns (skipped silently if sources are absent).
# ---------------------------------------------------------------------------

def _add_derived(props: dict) -> None:
    # np.where evaluates both branches before selecting, so divisions by zero
    # would emit warnings even though the result is correctly NaN. Suppress
    # those warnings within this scope; the np.where calls handle the math.
    with np.errstate(divide="ignore", invalid="ignore"):
        if "area" in props and "perimeter_crofton" in props:
            a = np.asarray(props["area"], dtype=float)
            p = np.asarray(props["perimeter_crofton"], dtype=float)
            props["circularity"] = np.where(p > 0, 4 * np.pi * a / (p * p), np.nan)

        if "axis_major_length" in props and "axis_minor_length" in props:
            maj = np.asarray(props["axis_major_length"], dtype=float)
            mn = np.asarray(props["axis_minor_length"], dtype=float)
            props["aspect_ratio"] = np.where(mn > 0, maj / mn, np.nan)

        if "orientation" in props:
            # skimage `orientation` is the angle from the row axis in [-pi/2, pi/2].
            # Converting via (90 - degrees(orientation)) maps a horizontal major
            # axis to 0 deg and a vertical major axis to 90 deg, in [0, 180).
            props["orientation_deg"] = (90.0 - np.degrees(props["orientation"])) % 180.0

        if "intensity_mean" in props:
            mean_i = np.asarray(props["intensity_mean"], dtype=float)
            # num_pixels is a true pixel count even when `spacing=` is set, so
            # intensity_total stays a real integrated intensity.
            if "num_pixels" in props:
                n_px = np.asarray(props["num_pixels"], dtype=float)
                props["intensity_total"] = mean_i * n_px
            elif "area" in props:
                props["intensity_total"] = mean_i * np.asarray(props["area"], dtype=float)
            if "intensity_std" in props:
                std_i = np.asarray(props["intensity_std"], dtype=float)
                props["intensity_cv"] = np.where(mean_i > 0, std_i / mean_i, np.nan)


# ---------------------------------------------------------------------------
# Opt-in extras dispatcher.
# ---------------------------------------------------------------------------

_BBOX_EXTRAS = {"local_bg", "rg_spread", "lbp", "fft", "glrlm"}


def _add_extras(props, masks, img, labels, extras, params, pixel_size_um) -> None:
    from scipy.ndimage import find_objects

    slices = find_objects(masks) if _BBOX_EXTRAS & set(extras) else None

    if "global_bg" in extras:
        bg_pixels = img[masks == 0]
        scalar = float(bg_pixels.mean()) if bg_pixels.size else float("nan")
        props["bg_global_mean"] = np.full(len(labels), scalar)

    if "local_bg" in extras:
        _add_local_bg(props, masks, img, slices, labels,
                      int(params.get("bg_radius", 5)))

    if "gradients" in extras:
        _add_gradients(props, masks, img, labels)

    if "stat_texture" in extras:
        _add_stat_texture(props, masks, img, labels,
                          int(params.get("n_intensity_bins", 256)))

    if "lbp" in extras:
        _add_lbp(props, masks, img, slices, labels,
                 P=int(params.get("lbp_P", 8)),
                 R=float(params.get("lbp_R", 1)),
                 method=str(params.get("lbp_method", "default")))

    if "fft" in extras:
        _add_fft(props, masks, img, slices, labels,
                 n_bins=int(params.get("fft_entropy_bins", 256)))

    if "glrlm" in extras:
        _add_glrlm(props, masks, img, slices, labels,
                   n_levels=int(params.get("glrlm_levels", 16)))

    if "rg_spread" in extras:
        _add_rg_spread(props, masks, img, slices, labels, pixel_size_um)


# ---------------------------------------------------------------------------
# Local background collar (per-object bbox, neighbour-excluded, holes filled).
# ---------------------------------------------------------------------------

def _add_local_bg(props, masks, img, slices, labels, radius) -> None:
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import dilation, disk

    n = len(labels)
    bg_local = np.full(n, np.nan, dtype=np.float64)
    footprint = disk(radius)
    H, W = masks.shape

    for i, lab in enumerate(labels):
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        sy, sx = sl
        sp = (
            slice(max(0, sy.start - radius - 1), min(H, sy.stop + radius + 1)),
            slice(max(0, sx.start - radius - 1), min(W, sx.stop + radius + 1)),
        )
        crop_lab = masks[sp]
        obj = crop_lab == lab
        filled = binary_fill_holes(obj)
        collar = dilation(filled, footprint=footprint) & ~filled & (crop_lab == 0)
        if collar.any():
            bg_local[i] = float(img[sp][collar].mean())

    props["bg_local_mean"] = bg_local

    with np.errstate(divide="ignore", invalid="ignore"):
        if "intensity_mean" in props:
            mean_i = np.asarray(props["intensity_mean"], dtype=float)
            props["mean_minus_local_bg"] = mean_i - bg_local
            props["mean_over_local_bg"] = np.where(
                bg_local > 0, mean_i / bg_local, np.nan
            )
            # TILB v2 = (I_avg - I_local) * A
            if "num_pixels" in props:
                n_px = np.asarray(props["num_pixels"], dtype=float)
            elif "area" in props:
                n_px = np.asarray(props["area"], dtype=float)
            else:
                n_px = None
            if n_px is not None:
                props["total_minus_local_bg"] = (mean_i - bg_local) * n_px
        if "intensity_total" in props:
            tot = np.asarray(props["intensity_total"], dtype=float)
            props["total_over_local_bg"] = np.where(
                bg_local > 0, tot / bg_local, np.nan
            )


# ---------------------------------------------------------------------------
# Gradient magnitude per object via vectorised bincount means.
# ---------------------------------------------------------------------------

def _add_gradients(props, masks, img, labels) -> None:
    from skimage.filters import prewitt, roberts

    props["prewitt_magnitude_mean"] = _per_label_mean(prewitt(img), masks, labels)
    props["roberts_magnitude_mean"] = _per_label_mean(roberts(img), masks, labels)


def _per_label_mean(values_image, label_image, labels):
    """Mean of `values_image` over each label in `labels`, vectorised."""
    n_lbl = int(label_image.max()) + 1
    flat_lbl = label_image.ravel()
    flat_v = values_image.ravel().astype(np.float64)
    sums = np.bincount(flat_lbl, weights=flat_v, minlength=n_lbl)
    counts = np.bincount(flat_lbl, minlength=n_lbl)
    out = np.full(len(labels), np.nan, dtype=np.float64)
    valid = counts[labels] > 0
    out[valid] = sums[labels][valid] / counts[labels][valid]
    return out


# ---------------------------------------------------------------------------
# Statistical texture: per-label intensity histogram in a single vector pass.
# Implements uniformity, Shannon entropy (base 2), Fisher excess kurtosis,
# and skewness over the per-object intensity distribution.
# ---------------------------------------------------------------------------

def _add_stat_texture(props, masks, img, labels, n_bins) -> None:
    img_arr = np.asarray(img)
    vmax = float(img_arr.max()) if img_arr.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    img_q = np.clip(img_arr / vmax * (n_bins - 1), 0, n_bins - 1).astype(np.int64)

    fg = masks > 0
    label_to_row = np.full(int(masks.max()) + 1, -1, dtype=np.int64)
    label_to_row[labels] = np.arange(len(labels))

    rows = label_to_row[masks[fg]]
    vals = img_q[fg]
    valid = rows >= 0
    rows = rows[valid]; vals = vals[valid]

    counts = np.bincount(
        rows * n_bins + vals, minlength=len(labels) * n_bins
    ).reshape(len(labels), n_bins).astype(np.float64)
    n_per = counts.sum(axis=1)
    p = np.divide(
        counts, n_per[:, None], out=np.zeros_like(counts), where=n_per[:, None] > 0
    )

    levels = np.arange(n_bins, dtype=np.float64)
    uniformity = (p * p).sum(axis=1)
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    entropy = -(p * logp).sum(axis=1)

    mean = (p * levels).sum(axis=1)
    centered = levels[None, :] - mean[:, None]
    m2 = (p * centered ** 2).sum(axis=1)
    m3 = (p * centered ** 3).sum(axis=1)
    m4 = (p * centered ** 4).sum(axis=1)

    skewness = np.full(len(labels), np.nan)
    kurtosis = np.full(len(labels), np.nan)
    nz = m2 > 0
    skewness[nz] = m3[nz] / (m2[nz] ** 1.5)
    kurtosis[nz] = m4[nz] / (m2[nz] ** 2) - 3.0

    props["intensity_uniformity"] = uniformity
    props["intensity_entropy"] = entropy
    props["intensity_skewness"] = skewness
    props["intensity_kurtosis"] = kurtosis


# ---------------------------------------------------------------------------
# Radius of gyration + intensity radial variance from bbox-local coordinates.
# Avoids regionprops_table('coords') so memory stays bounded.
# ---------------------------------------------------------------------------

def _add_rg_spread(props, masks, img, slices, labels, pixel_size_um) -> None:
    n = len(labels)
    rg = np.full(n, np.nan, dtype=np.float64)
    spread = np.full(n, np.nan, dtype=np.float64)
    dy, dx = pixel_size_um if pixel_size_um else (1.0, 1.0)

    for i, lab in enumerate(labels):
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        sy, sx = sl
        crop_lab = masks[sl]
        obj = crop_lab == lab
        if not obj.any():
            continue
        ys, xs = np.where(obj)
        ys = (ys + sy.start) * dy
        xs = (xs + sx.start) * dx
        cm_y = ys.mean()
        cm_x = xs.mean()
        d2 = (ys - cm_y) ** 2 + (xs - cm_x) ** 2
        rg_v = float(np.sqrt(d2.mean()))
        rg[i] = rg_v
        u = img[sl][obj].astype(np.float64)
        u_mean = float(u.mean())
        N = u.size
        if rg_v > 0 and u_mean > 0:
            spread[i] = float((u * d2).sum() / (u_mean * N * rg_v ** 2))

    props["radius_of_gyration"] = rg
    props["intensity_radial_variance_normalised"] = spread


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(img):
    """Map any non-negative image to uint8 [0, 255] for texture computations."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return np.clip(arr.astype(np.float64) / vmax * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Local Binary Patterns: image computed once, six per-object stats.
# ---------------------------------------------------------------------------

def _add_lbp(props, masks, img, slices, labels, P, R, method) -> None:
    from skimage.feature import local_binary_pattern
    from skimage.measure import shannon_entropy
    from scipy.stats import skew as sstat_skew, kurtosis as sstat_kurt

    lbp = local_binary_pattern(_to_uint8(img), P=P, R=R, method=method).astype(np.int32)

    n = len(labels)
    arr = np.full((n, 6), np.nan, dtype=np.float64)
    for i, lab in enumerate(labels):
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        v = lbp[sl][masks[sl] == lab]
        if v.size == 0:
            continue
        arr[i, 0] = v.mean()
        arr[i, 1] = v.std()
        h = np.bincount(v.astype(np.int64))
        if h.sum() > 0:
            p = h / h.sum()
            arr[i, 2] = float((p * p).sum())
        arr[i, 3] = float(shannon_entropy(v))
        if v.size > 1 and v.std() > 0:
            arr[i, 4] = float(sstat_skew(v))
            arr[i, 5] = float(sstat_kurt(v, fisher=True, bias=True))

    props["lbp_mean"] = arr[:, 0]
    props["lbp_std"] = arr[:, 1]
    props["lbp_energy"] = arr[:, 2]
    props["lbp_entropy"] = arr[:, 3]
    props["lbp_skewness"] = arr[:, 4]
    props["lbp_kurtosis"] = arr[:, 5]


# ---------------------------------------------------------------------------
# Per-object 2D FFT magnitude statistics.
# Entropy uses an explicit n-bin histogram to keep float magnitudes well-defined.
# ---------------------------------------------------------------------------

def _add_fft(props, masks, img, slices, labels, n_bins) -> None:
    from scipy.stats import skew as sstat_skew, kurtosis as sstat_kurt

    n = len(labels)
    arr = np.full((n, 6), np.nan, dtype=np.float64)
    for i, lab in enumerate(labels):
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        m = masks[sl] == lab
        if not m.any():
            continue
        crop = img[sl].astype(np.float64) * m
        F = np.abs(np.fft.fftshift(np.fft.fft2(crop)))
        flat = F.ravel()
        arr[i, 0] = float(flat.mean())
        arr[i, 1] = float(flat.std())
        arr[i, 2] = float((F * F).sum())
        hist, _ = np.histogram(F, bins=n_bins)
        s = hist.sum()
        if s > 0:
            p = hist / s
            nz = p > 0
            arr[i, 3] = float(-(p[nz] * np.log2(p[nz])).sum())
        if flat.size > 1 and flat.std() > 0:
            arr[i, 4] = float(sstat_skew(flat))
            arr[i, 5] = float(sstat_kurt(flat, fisher=True, bias=True))

    props["fft_mean"] = arr[:, 0]
    props["fft_std"] = arr[:, 1]
    props["fft_energy"] = arr[:, 2]
    props["fft_entropy"] = arr[:, 3]
    props["fft_skewness"] = arr[:, 4]
    props["fft_kurtosis"] = arr[:, 5]


# ---------------------------------------------------------------------------
# Gray-Level Run-Length Matrix (GLRLM) features.
# 4 directions summed (0, 45, 90, 135 deg). Background marked as -1 inside the
# bbox so runs never cross outside the object. Gray levels are 1-based in the
# matrix to keep LGLRE finite (no divide-by-zero on g=0).
# ---------------------------------------------------------------------------

def _runs_in_line(line):
    """Return (gray_levels, run_lengths) from a 1D int array. Pixels marked
    as negative (background sentinel) are filtered out."""
    if line.size == 0:
        return np.empty(0, dtype=line.dtype), np.empty(0, dtype=np.int64)
    diff = np.diff(line, prepend=line[0] - 1, append=line[-1] + 1)
    idx = np.flatnonzero(diff)
    starts = idx[:-1]
    lengths = np.diff(idx).astype(np.int64)
    vals = line[starts]
    valid = vals >= 0
    return vals[valid], lengths[valid]


def _glrlm_matrix_4dir(crop_q, n_levels):
    """Sum of run-length matrices over 4 directions: 0, 45, 90, 135 deg."""
    H, W = crop_q.shape
    if H == 0 or W == 0:
        return np.zeros((n_levels, 1), dtype=np.int64)
    P = np.zeros((n_levels, max(H, W)), dtype=np.int64)

    def _accumulate(lines):
        for line in lines:
            vals, lens = _runs_in_line(line)
            if vals.size:
                np.add.at(P, (vals, lens - 1), 1)

    _accumulate(crop_q)                                                          # 0
    _accumulate(crop_q.T)                                                        # 90
    _accumulate([np.diagonal(crop_q, k) for k in range(-H + 1, W)])              # 45
    _accumulate([np.diagonal(np.fliplr(crop_q), k) for k in range(-H + 1, W)])   # 135
    return P


def _add_glrlm(props, masks, img, slices, labels, n_levels) -> None:
    img_arr = np.asarray(img)
    vmax = float(img_arr.max()) if img_arr.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    img_q = np.clip(img_arr.astype(np.float64) / vmax * (n_levels - 1),
                    0, n_levels - 1).astype(np.int16)

    n = len(labels)
    arr = np.full((n, 4), np.nan, dtype=np.float64)
    g = np.arange(1, n_levels + 1, dtype=np.float64)[:, None]

    for i, lab in enumerate(labels):
        sl = slices[int(lab) - 1]
        if sl is None:
            continue
        m = masks[sl] == lab
        if not m.any():
            continue
        crop_q_obj = np.where(m, img_q[sl], -1).astype(np.int16)
        P = _glrlm_matrix_4dir(crop_q_obj, n_levels).astype(np.float64)
        TR = float(P.sum())
        if TR == 0:
            continue
        sum_g = P.sum(axis=0)   # over gray levels -> per run length
        sum_r = P.sum(axis=1)   # over run lengths -> per gray level
        rlnu = float((sum_g ** 2).sum() / TR)
        glnu = float((sum_r ** 2).sum() / TR)
        lglre = float((P / (g ** 2)).sum() / TR)
        hglre = float((P * (g ** 2)).sum() / TR)
        arr[i] = [rlnu, lglre, hglre, glnu]

    props["glrlm_rlnu"] = arr[:, 0]
    props["glrlm_lglre"] = arr[:, 1]
    props["glrlm_hglre"] = arr[:, 2]
    props["glrlm_glnu"] = arr[:, 3]
