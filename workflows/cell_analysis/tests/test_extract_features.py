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


# ---------------------------------------------------------------------------
# LBP
# ---------------------------------------------------------------------------


def test_lbp_runs_and_emits_six_columns():
    ef = _load_extract()
    masks = np.zeros((40, 40), dtype=np.int32)
    _disk(masks, 20, 20, 8, 1)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(40, 40)).astype(np.uint8)
    pd = ef.run(_pd(masks, img), {}, extras=["lbp"])
    p = pd["extract_features"]["properties"]
    for k in ("lbp_mean", "lbp_std", "lbp_energy", "lbp_entropy",
              "lbp_skewness", "lbp_kurtosis"):
        assert k in p
        assert p[k].shape == (1,)
        assert np.isfinite(p[k][0])


def test_lbp_uniform_object_has_zero_std_and_unit_energy():
    """Constant intensity inside a disk -> LBP code 0 (or P+1) for centre
    pixels (no neighbour exceeds), so the LBP distribution collapses to a
    single value -> std 0, energy 1, entropy 0."""
    ef = _load_extract()
    masks = np.zeros((30, 30), dtype=np.int32)
    _disk(masks, 15, 15, 6, 1)
    img = np.full((30, 30), 100, dtype=np.uint8)
    pd = ef.run(_pd(masks, img), {}, extras=["lbp"])
    p = pd["extract_features"]["properties"]
    assert p["lbp_std"][0] == pytest.approx(0.0, abs=1e-9)
    assert p["lbp_energy"][0] == pytest.approx(1.0, abs=1e-9)
    assert p["lbp_entropy"][0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FFT (hand-computed on a rectangular constant-intensity object)
# ---------------------------------------------------------------------------


def test_fft_constant_rectangle_matches_closed_form():
    """A 4x4 constant-intensity object that fills its bounding box gives
    FFT[0,0] = N*I and all other components 0 (constant input). Closed-form
    statistics of the magnitude spectrum follow."""
    ef = _load_extract()
    H = W = 8
    N_side = 4
    intensity = 5.0
    masks = np.zeros((H, W), dtype=np.int32)
    masks[2:2 + N_side, 2:2 + N_side] = 1
    img = np.zeros((H, W), dtype=np.float32)
    img[masks == 1] = intensity
    pd = ef.run(_pd(masks, img), {}, extras=["fft"])
    p = pd["extract_features"]["properties"]
    N = N_side * N_side
    bbox_n = N_side * N_side
    dc = N * intensity                                  # = 80
    expected_mean = dc / bbox_n                         # = 5
    expected_var = ((dc - expected_mean) ** 2 + (bbox_n - 1) * expected_mean ** 2) / bbox_n
    expected_std = float(np.sqrt(expected_var))
    expected_energy = dc ** 2                           # = 6400
    assert p["fft_mean"][0] == pytest.approx(expected_mean, rel=1e-6)
    assert p["fft_std"][0] == pytest.approx(expected_std, rel=1e-6)
    assert p["fft_energy"][0] == pytest.approx(expected_energy, rel=1e-6)


# ---------------------------------------------------------------------------
# GLRLM (hand-computed on a 3x3 constant-intensity object, 4 directions summed)
# ---------------------------------------------------------------------------


def test_glrlm_3x3_uniform_object_matches_hand_computation():
    """3x3 uniform-intensity object, 4 directions summed, n_levels=16.

    Object fills a single quantised gray level. Run breakdown:
      - horizontal rows : 3 runs of length 3 -> P[g, 2] += 3
      - vertical columns: 3 runs of length 3 -> P[g, 2] += 3
      - 45  diagonals   : lengths [1, 2, 3, 2, 1] -> P[g, 0..2] += [2, 2, 1]
      - 135 diagonals   : lengths [1, 2, 3, 2, 1] -> P[g, 0..2] += [2, 2, 1]
    Sums: P[g, 0] = 4, P[g, 1] = 4, P[g, 2] = 8; TR = 16.
      - RLNU  = (sum_g P)^2 summed over r / TR   = (4^2 + 4^2 + 8^2)/16 = 6
      - GLNU  = (sum_r P)^2 summed over g / TR   = (4+4+8)^2 / 16        = 16
      - With g_1-based = 9 (intensity quantises to matrix row 8 = (n_levels-1)/2):
        LGLRE = (4 + 4 + 8) / 81 / 16 = 1/81
        HGLRE = (4 + 4 + 8) * 81 / 16 = 81
    """
    ef = _load_extract()
    H = W = 5
    masks = np.zeros((H, W), dtype=np.int32)
    masks[1:4, 1:4] = 1                                 # 3x3 object
    img = np.zeros((H, W), dtype=np.uint8)
    img[masks == 1] = 8                                 # quantises to row 8
    img[0, 0] = 15                                      # force vmax = 15
    pd = ef.run(_pd(masks, img), {}, extras=["glrlm"], glrlm_levels=16)
    p = pd["extract_features"]["properties"]
    assert p["glrlm_rlnu"][0] == pytest.approx(6.0, rel=1e-9)
    assert p["glrlm_glnu"][0] == pytest.approx(16.0, rel=1e-9)
    assert p["glrlm_lglre"][0] == pytest.approx(1.0 / 81.0, rel=1e-9)
    assert p["glrlm_hglre"][0] == pytest.approx(81.0, rel=1e-9)


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
