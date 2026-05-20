"""Tests for target_acquisition steps: segment_tile and pick_targets.

Most tests exercise pick_targets directly with synthetic masks —
no cellpose dependency. The real cellpose test is marked slow
and skipped when cellpose is unavailable.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Step loading helper
# ---------------------------------------------------------------------------

_STEPS_DIR = Path(__file__).resolve().parents[1] / "steps"


def _load_step(name: str):
    """Import a step module by filename (without .py)."""
    path = _STEPS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mask(*regions):
    """Build a label mask from (row_start, col_start, height, width, label) tuples."""
    size = 512
    mask = np.zeros((size, size), dtype=np.int32)
    for r0, c0, h, w, lbl in regions:
        mask[r0:r0 + h, c0:c0 + w] = lbl
    return mask


def _make_image(mask):
    """Intensity image: each label gets intensity = label * 100."""
    return (mask.astype(np.float64) * 100.0)


def _make_segment_tile_output(mask, image=None):
    """Build a segment_tile output dict from a mask.

    The dict mirrors segment_tile.run's return contract: image_2d,
    masks, n_cells, image_size_px. No image_source key -- the engine's
    response schema dropped that field when the analysis_image_source
    branch was removed (D1). See TestSegmentTileResponseSchema below.
    """
    if image is None:
        image = _make_image(mask)
    ny, nx = image.shape[:2]
    n_cells = int(mask.max()) if mask is not None else 0
    return {
        "image_2d": image,
        "masks": mask,
        "n_cells": n_cells,
        "image_size_px": (nx, ny),
    }


def _make_pipeline_data(
    mask,
    image=None,
    tile_id=("0", 0, 0),
    tile_stage_xy_um=(1000.0, 2000.0),
    tile_zwide_um=500.0,
    pixel_size=(0.5, 0.5),
    image_size_px=None,
    image_to_stage=None,
    n_picks=5,
    feature="area",
):
    """Build a complete pipeline_data dict for pick_targets."""
    if image_to_stage is None:
        image_to_stage = [[1.0, 0.0], [0.0, 1.0]]
    seg = _make_segment_tile_output(mask, image)
    if image_size_px is not None:
        seg["image_size_px"] = image_size_px
    return {
        "segment_tile": seg,
        "input": {
            "tile_id": tile_id,
            "tile_stage_xy_um": tile_stage_xy_um,
            "tile_zwide_um": tile_zwide_um,
            "source_pixel_size_um": pixel_size,
            "source_image_size_px": (9999, 9999),
            "image_to_stage": image_to_stage,
            "n_picks": n_picks,
            "feature": feature,
        },
        "metadata": {"verbose": 0},
    }


# ---------------------------------------------------------------------------
# pick_targets tests
# ---------------------------------------------------------------------------

pick_targets = _load_step("pick_targets")


class TestRanking:
    """Top-N selection by area, descending."""

    def test_sorted_by_area_descending(self):
        mask = _make_mask(
            (10, 10, 20, 20, 1),
            (10, 100, 30, 30, 2),
            (10, 200, 10, 10, 3),
            (10, 300, 40, 40, 4),
            (10, 400, 15, 15, 5),
        )
        pd = _make_pipeline_data(mask, n_picks=5, feature="area")
        result = pick_targets.run(pd, {})
        picks = result["pick_targets"]["picks"]
        areas = [p["area_px"] for p in picks]
        assert areas == sorted(areas, reverse=True)

    def test_n_picks_limits_output(self):
        mask = _make_mask(
            (10, 10, 20, 20, 1),
            (10, 100, 30, 30, 2),
            (10, 200, 40, 40, 3),
        )
        pd = _make_pipeline_data(mask, n_picks=2, feature="area")
        result = pick_targets.run(pd, {})
        assert len(result["pick_targets"]["picks"]) == 2

    def test_n_picks_greater_than_n_cells(self):
        mask = _make_mask(
            (10, 10, 20, 20, 1),
            (10, 100, 30, 30, 2),
            (10, 200, 40, 40, 3),
        )
        pd = _make_pipeline_data(mask, n_picks=10, feature="area")
        result = pick_targets.run(pd, {})
        assert len(result["pick_targets"]["picks"]) == 3


class TestCentroidSwap:
    """skimage centroid is (row, col) — pick format is (col, row)."""

    def test_single_cell_centroid_order(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[90:110, 190:210] = 1
        pd = _make_pipeline_data(mask, n_picks=1)
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        col, row = pick["centroid_col_row_px"]
        assert 190 < col < 210
        assert 90 < row < 110


class TestPixelToStage:
    """Coordinate conversion with various matrices and pixel sizes."""

    def test_identity_matrix(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[146:166, 346:366] = 1
        pd = _make_pipeline_data(
            mask,
            tile_stage_xy_um=(1000.0, 2000.0),
            pixel_size=(0.5, 0.5),
            image_to_stage=[[1.0, 0.0], [0.0, 1.0]],
            n_picks=1,
        )
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        cx, cy = pick["cell_source_stage_xy_um"]
        assert cx == pytest.approx(1000.0 + (356 - 256) * 0.5, abs=1.0)
        assert cy == pytest.approx(2000.0 + (156 - 256) * 0.5, abs=1.0)

    def test_90_degree_rotation(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[146:166, 346:366] = 1
        pd = _make_pipeline_data(
            mask,
            tile_stage_xy_um=(25000.0, 20000.0),
            pixel_size=(0.65, 0.65),
            image_to_stage=[[0.0, -1.0], [1.0, 0.0]],
            n_picks=1,
        )
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        cx, cy = pick["cell_source_stage_xy_um"]
        col_offset = (356 - 256) * 0.65
        row_offset = (156 - 256) * 0.65
        expected_x = 25000.0 + (0.0 * col_offset + (-1.0) * row_offset)
        expected_y = 20000.0 + (1.0 * col_offset + 0.0 * row_offset)
        assert cx == pytest.approx(expected_x, abs=1.0)
        assert cy == pytest.approx(expected_y, abs=1.0)

    def test_non_square_pixels(self):
        mask = np.zeros((400, 800), dtype=np.int32)
        mask[190:210, 390:410] = 1
        pd = _make_pipeline_data(
            mask,
            tile_stage_xy_um=(5000.0, 3000.0),
            pixel_size=(0.5, 0.7),
            image_to_stage=[[1.0, 0.0], [0.0, 1.0]],
            n_picks=1,
        )
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        cx, cy = pick["cell_source_stage_xy_um"]
        assert cx == pytest.approx(5000.0 + (400 - 400) * 0.5, abs=1.0)
        assert cy == pytest.approx(3000.0 + (200 - 200) * 0.7, abs=1.0)


class TestBboxUm:
    """bbox_um must use pw for width (col axis) and ph for height (row axis)."""

    def test_non_square_pixels_bbox(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[100:140, 200:260] = 1
        pd = _make_pipeline_data(
            mask,
            pixel_size=(0.5, 0.7),
            image_to_stage=[[1.0, 0.0], [0.0, 1.0]],
            n_picks=1,
        )
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        w_um, h_um = pick["bbox_um"]
        assert w_um == pytest.approx(60 * 0.5, abs=0.01)
        assert h_um == pytest.approx(40 * 0.7, abs=0.01)


class TestMismatchedImageSize:
    """pick_targets must use segment_tile's image_size_px, not input's."""

    def test_uses_actual_image_size_not_submitted(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[246:266, 246:266] = 1
        pd = _make_pipeline_data(
            mask,
            tile_stage_xy_um=(0.0, 0.0),
            pixel_size=(1.0, 1.0),
            image_to_stage=[[1.0, 0.0], [0.0, 1.0]],
            n_picks=1,
        )
        pd["input"]["source_image_size_px"] = (2048, 1536)
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        cx, cy = pick["cell_source_stage_xy_um"]
        assert abs(cx) < 20.0
        assert abs(cy) < 20.0


class TestEmptyMask:

    def test_none_mask(self):
        dummy = np.zeros((512, 512), dtype=np.int32)
        pd = _make_pipeline_data(dummy)
        pd["segment_tile"]["n_cells"] = 0
        pd["segment_tile"]["masks"] = None
        result = pick_targets.run(pd, {})
        assert result["pick_targets"]["picks"] == []

    def test_zero_mask(self):
        mask = np.zeros((512, 512), dtype=np.int32)
        pd = _make_pipeline_data(mask)
        result = pick_targets.run(pd, {})
        assert result["pick_targets"]["picks"] == []


class TestNativeTypes:
    """All pick values must be native Python types — no numpy scalars."""

    def test_no_numpy_scalars(self):
        mask = _make_mask((50, 50, 30, 30, 1))
        pd = _make_pipeline_data(mask, n_picks=1)
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]

        for key, val in pick.items():
            if isinstance(val, tuple):
                for elem in val:
                    assert type(elem) in (int, float, str), \
                        f"pick[{key!r}] contains {type(elem).__name__}"
            else:
                assert type(val) in (int, float, str), \
                    f"pick[{key!r}] is {type(val).__name__}"


class TestFeatureValidation:

    def test_unknown_feature_raises(self):
        mask = _make_mask((10, 10, 20, 20, 1))
        pd = _make_pipeline_data(mask, feature="bogus")
        with pytest.raises(ValueError, match="Unknown feature"):
            pick_targets.run(pd, {})

    def test_mean_intensity_sorting(self):
        mask = _make_mask(
            (10, 10, 20, 20, 1),
            (10, 100, 30, 30, 2),
            (10, 200, 40, 40, 3),
        )
        pd = _make_pipeline_data(mask, n_picks=3, feature="mean_intensity")
        result = pick_targets.run(pd, {})
        picks = result["pick_targets"]["picks"]
        intensities = [p["mean_intensity"] for p in picks]
        assert intensities == sorted(intensities, reverse=True)


class TestZCarryThrough:

    def test_tile_zwide_preserved(self):
        mask = _make_mask((10, 10, 20, 20, 1))
        pd = _make_pipeline_data(mask, tile_zwide_um=1234.5, n_picks=1)
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        assert pick["tile_zwide_um"] == 1234.5


class TestPickId:

    def test_pick_id_extends_tile_id(self):
        mask = _make_mask((10, 10, 20, 20, 1))
        pd = _make_pipeline_data(mask, tile_id=("R0", 3, 7), n_picks=1)
        result = pick_targets.run(pd, {})
        pick = result["pick_targets"]["picks"][0]
        assert pick["pick_id"][:3] == ("R0", 3, 7)
        assert isinstance(pick["pick_id"][3], int)


# ---------------------------------------------------------------------------
# segment_tile tests (no cellpose — only _ensure_2d and _load_image)
# ---------------------------------------------------------------------------

segment_tile = _load_step("segment_tile")


class TestEnsure2d:

    def test_2d_passthrough(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        assert segment_tile._ensure_2d(img, 0).shape == (100, 200)

    def test_c_first(self):
        img = np.zeros((3, 100, 200), dtype=np.uint8)
        assert segment_tile._ensure_2d(img, 1).shape == (100, 200)

    def test_c_last(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert segment_tile._ensure_2d(img, 2).shape == (100, 200)

    def test_negative_channel_raises(self):
        img = np.zeros((3, 100, 200), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel must be >= 0"):
            segment_tile._ensure_2d(img, -1)

    def test_channel_out_of_range_c_first(self):
        img = np.zeros((2, 100, 200), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel=5"):
            segment_tile._ensure_2d(img, 5)

    def test_channel_out_of_range_c_last(self):
        img = np.zeros((100, 200, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel=3"):
            segment_tile._ensure_2d(img, 3)

    def test_ambiguous_shape_raises(self):
        img = np.zeros((10, 20, 30), dtype=np.uint8)
        with pytest.raises(ValueError, match="Cannot extract 2D"):
            segment_tile._ensure_2d(img, 0)


class TestSegmentTileResponseSchema:
    """Pin the post-D1 segment_tile response contract.

    Before D1, segment_tile.run returned an ``image_source`` key
    echoing which input branch (``"acquired"`` / ``"skimage_human_mitosis"``)
    produced the pixels. After D1 there is no branch -- the engine
    always reads from ``image_path`` -- so the key is gone from the
    contract. This test asserts the new shape structurally so a
    future contributor cannot re-introduce the field without breaking
    a named test.
    """

    def test_response_keys_exact(self, tmp_path):
        # Minimal real-file payload: write a tiny TIFF and let
        # segment_tile read it through the production file-read path.
        # _ensure_2d handles the single-plane shape; we don't need
        # cellpose for this schema test, so we stub it.
        import tifffile
        image_path = tmp_path / "tiny.ome.tiff"
        tifffile.imwrite(image_path, np.zeros((16, 16), dtype=np.uint8))

        pipeline_data = {
            "input": {"image_path": str(image_path)},
            "metadata": {"verbose": 0},
        }
        state = {"model": _StubCellposeModel()}
        result = segment_tile.run(pipeline_data, state)
        seg = result["segment_tile"]
        # Exact contract: these four keys, no others. A future
        # contributor adding back image_source -- or any other
        # provenance field -- breaks this assertion.
        assert set(seg.keys()) == {
            "image_2d", "masks", "n_cells", "image_size_px",
        }


class _StubCellposeModel:
    """Stub matching cellpose.models.CellposeModel's eval() signature
    used by segment_tile. Returns an empty mask plus the four-tuple
    cellpose returns (masks, flows, styles)."""
    def eval(self, image_2d, diameter=None):
        return np.zeros(image_2d.shape, dtype=np.int32), None, None


class TestSegmentTileIgnoresExtraInputKeys:
    """Engine accepts payloads with extra input keys it doesn't
    recognize -- doesn't branch, doesn't log, doesn't crash. Pins
    forward-compatibility for callers that pass through fields the
    engine has no contract for.

    Phrased generically rather than naming any specific historical
    field: the principle (extra keys are silently ignored) stands on
    its own, and the test doesn't fossilize a field name that the
    engine has since dropped from its contract.
    """

    def test_extra_input_keys_are_ignored_engine_reads_image_path(
        self, tmp_path,
    ):
        import tifffile
        # 16x16 sentinel image. If a hidden branch loaded any other
        # image source (e.g. skimage.data.human_mitosis is ~512x512),
        # the resulting image_size_px would not match.
        image_path = tmp_path / "tiny.ome.tiff"
        tifffile.imwrite(image_path, np.zeros((16, 16), dtype=np.uint8))

        pipeline_data = {
            "input": {
                "image_path": str(image_path),
                # Generic extra key. The engine must not branch on
                # *any* unrecognized input key; this is the
                # forward-compat contract.
                "an_unrecognized_field": "anything",
            },
            "metadata": {"verbose": 0},
        }
        state = {"model": _StubCellposeModel()}
        result = segment_tile.run(pipeline_data, state)
        seg = result["segment_tile"]
        # Engine read the 16x16 file -- not anything inferred from
        # the extra key. A hidden branch loading a different source
        # would yield a different shape; assertion catches it.
        assert seg["image_size_px"] == (16, 16)
        assert seg["image_2d"].shape == (16, 16)
        assert seg["n_cells"] == 0
        # Response schema stays clean even when the input carries
        # extra keys.
        assert set(seg.keys()) == {
            "image_2d", "masks", "n_cells", "image_size_px",
        }


# ---------------------------------------------------------------------------
# Real cellpose end-to-end (optional)
# ---------------------------------------------------------------------------


@pytest.mark.cellpose
@pytest.mark.slow
class TestCellposeEndToEnd:

    def test_human_mitosis_produces_picks(self, tmp_path):
        # Pre-D1 this test exercised the engine's
        # analysis_image_source="skimage_human_mitosis" branch, which
        # made segment_tile skip the file read and load the skimage
        # reference image directly. D1 removed that branch -- the
        # engine now always reads from image_path. Preserve the
        # useful end-to-end cellpose-on-real-image coverage by
        # writing human_mitosis() to a temp .ome.tiff and letting
        # segment_tile read it through the production file-read
        # path. Same coverage, exercises the same code path a real
        # LAS X acquisition would.
        import tifffile
        from skimage.data import human_mitosis

        image = human_mitosis()
        image_path = tmp_path / "human_mitosis.ome.tiff"
        tifffile.imwrite(image_path, image, photometric="minisblack")

        state = {}
        pipeline_data = {
            "input": {
                "image_path": str(image_path),
                "tile_id": ("0", 0, 0),
                "tile_stage_xy_um": (10000.0, 15000.0),
                "tile_zwide_um": 2500.0,
                "source_pixel_size_um": (0.65, 0.65),
                "source_image_size_px": (9999, 9999),
                "image_to_stage": [[0.0, -1.0], [1.0, 0.0]],
                "n_picks": 5,
                "feature": "area",
            },
            "metadata": {"verbose": 2},
        }

        seg_result = segment_tile.run(pipeline_data, state)
        assert seg_result["segment_tile"]["n_cells"] > 0
        ny, nx = seg_result["segment_tile"]["image_size_px"][1], \
                 seg_result["segment_tile"]["image_size_px"][0]
        assert (nx, ny) == (image.shape[1], image.shape[0])

        pick_result = pick_targets.run(seg_result, {})
        picks = pick_result["pick_targets"]["picks"]
        assert len(picks) > 0
        assert len(picks) <= 5

        for p in picks:
            assert isinstance(p["cell_source_stage_xy_um"], tuple)
            assert all(isinstance(v, float) for v in p["cell_source_stage_xy_um"])
            assert p["tile_zwide_um"] == 2500.0
            assert p["source_image_size_px"] == (nx, ny)
