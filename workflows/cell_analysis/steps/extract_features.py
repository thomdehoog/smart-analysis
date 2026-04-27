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


_VALID_EXTRAS = {"global_bg", "local_bg", "gradients", "stat_texture", "rg_spread"}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    from skimage.measure import regionprops_table

    verbose = pipeline_data["metadata"].get("verbose", 0)

    properties = list(params.get("properties", DEFAULT_PROPERTIES))
    extras = list(params.get("extras", []) or [])
    pixel_size_um = params.get("pixel_size_um")
    bg_radius = int(params.get("bg_radius", 5))
    n_bins = int(params.get("n_intensity_bins", 256))

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
            _add_extras(props, masks, img, labels, extras,
                        bg_radius, n_bins, pixel_size_um)

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

def _add_extras(props, masks, img, labels, extras,
                bg_radius, n_bins, pixel_size_um) -> None:
    from scipy.ndimage import find_objects

    slices = find_objects(masks) if {"local_bg", "rg_spread"} & set(extras) else None

    if "global_bg" in extras:
        bg_pixels = img[masks == 0]
        scalar = float(bg_pixels.mean()) if bg_pixels.size else float("nan")
        props["bg_global_mean"] = np.full(len(labels), scalar)

    if "local_bg" in extras:
        _add_local_bg(props, masks, img, slices, labels, bg_radius)

    if "gradients" in extras:
        _add_gradients(props, masks, img, labels)

    if "stat_texture" in extras:
        _add_stat_texture(props, masks, img, labels, n_bins)

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
