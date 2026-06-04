"""Tests for shared segmentation channel selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _segmentation import select_channels  # noqa: E402


def test_2d_passthrough():
    img = np.zeros((10, 20), dtype=np.uint8)
    assert select_channels(img).shape == (10, 20)


def test_2d_rejects_nonzero_explicit_channel():
    img = np.zeros((10, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D image"):
        select_channels(img, channels=[1])


def test_channel_last_three_kept():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert select_channels(img).shape == (10, 20, 3)


def test_channel_first_normalized_to_last():
    img = np.zeros((3, 10, 20), dtype=np.uint8)
    assert select_channels(img).shape == (10, 20, 3)


def test_single_channel_returns_2d():
    img = np.zeros((10, 20, 1), dtype=np.uint8)
    assert select_channels(img).shape == (10, 20)


def test_more_than_three_uses_specified_indices():
    img = np.zeros((10, 20, 5), dtype=np.uint8)
    img[..., 2] = 7
    img[..., 4] = 9
    out = select_channels(img, channels=[2, 4])
    assert out.shape == (10, 20, 2)
    assert int(out[0, 0, 0]) == 7
    assert int(out[0, 0, 1]) == 9


def test_integer_channel_is_single_channel_shorthand():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 2] = 17
    out = select_channels(img, channels=2)
    assert out.shape == (10, 20)
    assert int(out[0, 0]) == 17


def test_channel_first_more_than_three_uses_specified_indices():
    img = np.zeros((5, 10, 20), dtype=np.uint8)
    img[1] = 11
    img[3] = 13
    out = select_channels(img, channels=[1, 3])
    assert out.shape == (10, 20, 2)
    assert int(out[0, 0, 0]) == 11
    assert int(out[0, 0, 1]) == 13


def test_auto_caps_at_three():
    img = np.zeros((10, 20, 5), dtype=np.uint8)
    assert select_channels(img).shape == (10, 20, 3)


def test_out_of_range_channel_raises():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        select_channels(img, channels=[0, 9])


def test_more_than_three_requested_raises():
    img = np.zeros((10, 20, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        select_channels(img, channels=[0, 1, 2, 3])


def test_empty_channel_list_raises_for_multichannel():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one"):
        select_channels(img, channels=[])


def test_four_dimensional_input_raises():
    img = np.zeros((2, 3, 10, 20), dtype=np.uint8)
    with pytest.raises(ValueError):
        select_channels(img)


def test_immunohistochemistry_keeps_three_channels():
    from skimage.data import immunohistochemistry

    try:
        image = immunohistochemistry()
    except Exception as exc:
        pytest.skip(f"skimage immunohistochemistry fixture unavailable: {exc}")

    out = select_channels(image)
    assert out.shape == (512, 512, 3)


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def eval(self, x, channel_axis=None, **kwargs):
        self.calls.append({"channel_axis": channel_axis, "kwargs": kwargs})
        masks = np.zeros(x.shape[:2], dtype=np.int32)
        masks[1:3, 1:3] = 1
        return masks, None, None


def test_segment_tiff_uses_channel_axis_for_multichannel(tmp_path):
    import tifffile
    from _segmentation import segment_tiff

    path = tmp_path / "rgb.tif"
    tifffile.imwrite(path, np.zeros((10, 12, 3), dtype=np.uint8))
    model = _RecordingModel()
    out = segment_tiff(path, {"model": model})

    assert model.calls[0]["channel_axis"] == -1
    assert "diameter" not in model.calls[0]["kwargs"]
    assert out["image_2d"].ndim == 2


def test_segment_tiff_no_channel_axis_for_2d(tmp_path):
    import tifffile
    from _segmentation import segment_tiff

    path = tmp_path / "gray.tif"
    tifffile.imwrite(path, np.zeros((10, 12), dtype=np.uint8))
    model = _RecordingModel()
    segment_tiff(path, {"model": model})

    assert model.calls[0]["channel_axis"] is None
