"""Unit tests for ``steps/extract_features.py``.

Covers the rewritten feature module without going through the engine. The
suite isolates one behaviour per test on tiny, hand-computed synthetic
images, so failures point directly at the relevant feature implementation.

Areas exercised:

- Default property set (Tier A regionprops).
- Always-on derived columns (circularity, aspect_ratio, orientation_deg,
  intensity_total, intensity_cv).
- Orientation conversion on horizontal vs vertical ellipses.
- Local background extras: neighbour exclusion and hole-filling.
- Gradient extras (Prewitt / Roberts) match a direct skimage computation.
- Statistical texture matches hand-computed uniformity/entropy on
  controlled intensity distributions.
- Radius of gyration matches the continuous R/sqrt(2) formula on a
  uniform disk; intensity radial variance equals 1 for uniform intensity.
- Sparse label ids do not break per-object indexing.
- Single-pixel object does not raise; degenerate features are NaN.
- Empty mask returns zero cells.
- Unknown ``extras`` group raises a clear error.
- ``pixel_size_um`` scales area but ``intensity_total`` stays unit-safe.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


WORKFLOW = Path(__file__).resolve().parent.parent
EXTRACT_PATH = WORKFLOW / "steps" / "extract_features.py"


def _load_extract():
    spec = importlib.util.spec_from_file_location("ef", EXTRACT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pd(masks, img):
    return {
        "metadata": {"verbose": 0},
        "segment": {"masks": masks},
        "preprocess": {"image": img},
    }


def _disk(masks, cy, cx, r, label):
    H, W = masks.shape
    yy, xx = np.ogrid[:H, :W]
    masks[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label


def _ellipse(masks, cy, cx, ry, rx, label):
    H, W = masks.shape
    yy, xx = np.ogrid[:H, :W]
    masks[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1] = label


# ---------------------------------------------------------------------------
# Tier A: defaults
# ---------------------------------------------------------------------------


def test_defaults_include_core_properties():
    ef = _load_extract()
    masks = np.zeros((50, 50), dtype=np.int32)
    _disk(masks, 25, 25, 10, 1)
    img = np.full((50, 50), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    p = pd["extract_features"]["properties"]
    for key in ("label", "area", "num_pixels", "perimeter_crofton",
                "axis_major_length", "axis_minor_length", "orientation",
                "intensity_mean", "intensity_median", "intensity_std"):
        assert key in p, f"missing default property: {key}"
    assert pd["extract_features"]["n_cells"] == 1


# ---------------------------------------------------------------------------
# Tier B: always-on derived columns
# ---------------------------------------------------------------------------


def test_circularity_close_to_one_for_disk():
    ef = _load_extract()
    masks = np.zeros((100, 100), dtype=np.int32)
    _disk(masks, 50, 50, 25, 1)
    img = np.full((100, 100), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    circ = pd["extract_features"]["properties"]["circularity"][0]
    assert 0.85 < circ <= 1.05


def test_aspect_ratio_one_for_circle():
    ef = _load_extract()
    masks = np.zeros((100, 100), dtype=np.int32)
    _disk(masks, 50, 50, 25, 1)
    img = np.full((100, 100), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    ar = pd["extract_features"]["properties"]["aspect_ratio"][0]
    assert ar == pytest.approx(1.0, abs=0.05)


def test_intensity_total_equals_mean_times_num_pixels():
    ef = _load_extract()
    masks = np.zeros((50, 50), dtype=np.int32)
    _disk(masks, 25, 25, 10, 1)
    img = np.full((50, 50), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    p = pd["extract_features"]["properties"]
    assert p["intensity_total"][0] == pytest.approx(
        float(p["intensity_mean"][0]) * float(p["num_pixels"][0])
    )


def test_intensity_cv_zero_for_uniform_object():
    ef = _load_extract()
    masks = np.zeros((50, 50), dtype=np.int32)
    _disk(masks, 25, 25, 10, 1)
    img = np.full((50, 50), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    cv = pd["extract_features"]["properties"]["intensity_cv"][0]
    assert cv == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Orientation: horizontal ellipse -> 0 deg, vertical ellipse -> 90 deg.
# Catches the (90 - degrees) % 180 conversion direction.
# ---------------------------------------------------------------------------


def test_orientation_horizontal_ellipse_is_zero_degrees():
    ef = _load_extract()
    masks = np.zeros((100, 100), dtype=np.int32)
    _ellipse(masks, 50, 50, ry=8, rx=25, label=1)  # wider than tall
    img = np.full((100, 100), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    deg = float(pd["extract_features"]["properties"]["orientation_deg"][0])
    # Either 0 or 180 (mod 180 wraps the boundary).
    assert min(abs(deg), abs(deg - 180)) < 2.0


def test_orientation_vertical_ellipse_is_ninety_degrees():
    ef = _load_extract()
    masks = np.zeros((100, 100), dtype=np.int32)
    _ellipse(masks, 50, 50, ry=25, rx=8, label=1)  # taller than wide
    img = np.full((100, 100), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    deg = float(pd["extract_features"]["properties"]["orientation_deg"][0])
    assert deg == pytest.approx(90.0, abs=2.0)


# ---------------------------------------------------------------------------
# Local background extras
# ---------------------------------------------------------------------------


def test_local_bg_basic_isolated_object():
    ef = _load_extract()
    masks = np.zeros((50, 50), dtype=np.int32)
    _disk(masks, 25, 25, 8, 1)
    img = np.full((50, 50), 5.0, dtype=np.float32)
    img[masks > 0] = 100.0
    pd = ef.run(_pd(masks, img), {}, extras=["global_bg", "local_bg"])
    p = pd["extract_features"]["properties"]
    assert p["bg_global_mean"][0] == pytest.approx(5.0, abs=0.01)
    assert p["bg_local_mean"][0] == pytest.approx(5.0, abs=0.01)
    assert p["mean_minus_local_bg"][0] == pytest.approx(95.0, abs=0.5)
    assert p["mean_over_local_bg"][0] == pytest.approx(20.0, abs=0.5)


def test_local_bg_excludes_neighbour():
    ef = _load_extract()
    masks = np.zeros((50, 60), dtype=np.int32)
    _disk(masks, 25, 15, 8, label=1)
    _disk(masks, 25, 35, 8, label=2)
    img = np.full(masks.shape, 5.0, dtype=np.float32)
    img[masks == 1] = 100.0
    img[masks == 2] = 200.0
    pd = ef.run(_pd(masks, img), {}, extras=["local_bg"])
    p = pd["extract_features"]["properties"]
    # Object 1's local bg must be ~5 (the dim outside background), not a
    # blend with the bright neighbour.
    assert p["bg_local_mean"][0] == pytest.approx(5.0, abs=1.0)
    assert p["bg_local_mean"][1] == pytest.approx(5.0, abs=1.0)


def test_local_bg_fills_holes_so_inside_is_not_counted():
    ef = _load_extract()
    H = W = 60
    yy, xx = np.ogrid[:H, :W]
    outer = (yy - 30) ** 2 + (xx - 30) ** 2 <= 15 ** 2
    inner = (yy - 30) ** 2 + (xx - 30) ** 2 <= 7 ** 2
    masks = np.zeros((H, W), dtype=np.int32)
    masks[outer & ~inner] = 1                       # annular object
    img = np.full((H, W), 5.0, dtype=np.float32)    # outside is dim
    img[masks == 1] = 100.0                         # ring is bright
    img[inner] = 999.0                              # poison the hole
    pd = ef.run(_pd(masks, img), {}, extras=["local_bg"])
    bg = float(pd["extract_features"]["properties"]["bg_local_mean"][0])
    # If the hole were treated as background, the mean would be far above 5.
    assert bg == pytest.approx(5.0, abs=2.0)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradients_match_direct_computation():
    from skimage.filters import prewitt, roberts
    ef = _load_extract()
    masks = np.zeros((40, 40), dtype=np.int32)
    _disk(masks, 20, 20, 8, 1)
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, size=(40, 40)).astype(np.float32)
    img[masks > 0] += 50
    pd = ef.run(_pd(masks, img), {}, extras=["gradients"])
    p = pd["extract_features"]["properties"]
    assert p["prewitt_magnitude_mean"][0] == pytest.approx(
        float(prewitt(img)[masks == 1].mean())
    )
    assert p["roberts_magnitude_mean"][0] == pytest.approx(
        float(roberts(img)[masks == 1].mean())
    )


# ---------------------------------------------------------------------------
# Statistical texture (hand-computed against known distributions)
# ---------------------------------------------------------------------------


def test_stat_texture_uniform_intensity_gives_unity_uniformity():
    """All pixels at the same intensity -> single bin -> uniformity 1, entropy 0."""
    ef = _load_extract()
    masks = np.zeros((40, 40), dtype=np.int32)
    _disk(masks, 20, 20, 8, 1)
    img = np.zeros((40, 40), dtype=np.uint8)
    img[masks > 0] = 100
    pd = ef.run(_pd(masks, img), {}, extras=["stat_texture"])
    p = pd["extract_features"]["properties"]
    assert p["intensity_uniformity"][0] == pytest.approx(1.0, abs=1e-6)
    assert p["intensity_entropy"][0] == pytest.approx(0.0, abs=1e-6)


def test_stat_texture_two_equal_bins_gives_half_uniformity_one_bit_entropy():
    """Half pixels at intensity A, half at B -> uniformity 0.5, entropy 1 bit."""
    ef = _load_extract()
    H = W = 20
    masks = np.zeros((H, W), dtype=np.int32)
    masks[5:15, 5:15] = 1                           # 100 object pixels
    img = np.zeros((H, W), dtype=np.uint8)
    img[5:10, 5:15] = 100                           # 50 pixels at 100
    img[10:15, 5:15] = 200                          # 50 pixels at 200
    pd = ef.run(_pd(masks, img), {}, extras=["stat_texture"])
    p = pd["extract_features"]["properties"]
    assert p["intensity_uniformity"][0] == pytest.approx(0.5, abs=1e-6)
    assert p["intensity_entropy"][0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Radius of gyration + intensity radial variance
# ---------------------------------------------------------------------------


def test_rg_uniform_disk_matches_continuous_formula():
    """A continuous disk of radius R has R_g = R/sqrt(2). Check the digital
    approximation lands within 5%; check that uniform intensity gives a
    spreading value of exactly 1 (closed-form result)."""
    ef = _load_extract()
    H = W = 100
    masks = np.zeros((H, W), dtype=np.int32)
    R = 20
    _disk(masks, H // 2, W // 2, R, 1)
    img = np.full((H, W), 100.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {}, extras=["rg_spread"])
    p = pd["extract_features"]["properties"]
    assert p["radius_of_gyration"][0] == pytest.approx(R / np.sqrt(2), rel=0.05)
    assert p["intensity_radial_variance_normalised"][0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_label_id_gaps_keep_per_label_alignment():
    """Sparse label ids [1, 3, 5] must still produce one row per label."""
    ef = _load_extract()
    masks = np.zeros((40, 60), dtype=np.int32)
    _disk(masks, 20, 10, 5, label=1)
    _disk(masks, 20, 30, 5, label=3)
    _disk(masks, 20, 50, 5, label=5)
    img = np.zeros(masks.shape, dtype=np.float32)
    img[masks == 1] = 10.0
    img[masks == 3] = 30.0
    img[masks == 5] = 50.0
    pd = ef.run(_pd(masks, img), {}, extras=["local_bg", "rg_spread", "stat_texture"])
    p = pd["extract_features"]["properties"]
    assert list(p["label"]) == [1, 3, 5]
    assert p["intensity_mean"].tolist() == pytest.approx([10.0, 30.0, 50.0])
    assert np.all(np.isfinite(p["radius_of_gyration"]))


def test_single_pixel_object_does_not_error():
    ef = _load_extract()
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[5, 5] = 1
    img = np.full((10, 10), 50.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {}, extras=["rg_spread", "stat_texture"])
    p = pd["extract_features"]["properties"]
    assert p["radius_of_gyration"][0] == 0.0
    assert np.isnan(p["intensity_radial_variance_normalised"][0])


def test_no_objects_returns_empty_output():
    ef = _load_extract()
    masks = np.zeros((20, 20), dtype=np.int32)
    img = np.full((20, 20), 50.0, dtype=np.float32)
    pd = ef.run(_pd(masks, img), {})
    assert pd["extract_features"]["n_cells"] == 0


def test_unknown_extras_raises():
    ef = _load_extract()
    masks = np.zeros((20, 20), dtype=np.int32)
    _disk(masks, 10, 10, 5, 1)
    img = np.full((20, 20), 50.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown extras"):
        ef.run(_pd(masks, img), {}, extras=["bogus_group"])


def test_pixel_size_um_scales_area_but_intensity_total_is_unit_safe():
    ef = _load_extract()
    masks = np.zeros((40, 40), dtype=np.int32)
    _disk(masks, 20, 20, 10, 1)
    img = np.full((40, 40), 100.0, dtype=np.float32)
    pd_pix = ef.run(_pd(masks, img.copy()), {})
    pd_um = ef.run(_pd(masks, img.copy()), {}, pixel_size_um=[2.0, 2.0])
    p_pix = pd_pix["extract_features"]["properties"]
    p_um = pd_um["extract_features"]["properties"]
    assert p_um["area"][0] == pytest.approx(p_pix["area"][0] * 4.0, rel=0.01)
    assert p_um["intensity_total"][0] == pytest.approx(p_pix["intensity_total"][0])
    assert p_um["num_pixels"][0] == p_pix["num_pixels"][0]
