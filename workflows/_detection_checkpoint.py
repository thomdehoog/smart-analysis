"""Shared object-detection checkpoint helpers.

The checkpoint boundary is the contract between expensive segmentation and
re-runnable feature extraction. Keep the segmentation identity here so the
writer, loader, notebook helpers, and tests cannot drift.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _contracts import to_builtin


SEGMENTATION_IDENTITY_KEYS = (
    "channels",
    "cellprob_threshold",
    "flow_threshold",
    "niter",
    "diameter",
    "max_segmentation_size_px",
)


def segmentation_params(inp: dict, params: dict) -> dict:
    """Return params that define mask generation, excluding runtime details."""
    return {
        "channels": inp.get("channels", params.get("channels", None)),
        "cellprob_threshold": params.get("cellprob_threshold", None),
        "flow_threshold": params.get("flow_threshold", None),
        "niter": params.get("niter", None),
        "diameter": params.get("diameter", None),
        "max_segmentation_size_px": params.get("max_segmentation_size_px", None),
    }


def area_filter_params(inp: dict, params: dict) -> dict:
    """Return post-segmentation object-size filtering params."""
    return {
        "min_area_px": inp.get("min_area_px", params.get("min_area_px", None)),
        "max_area_px": inp.get("max_area_px", params.get("max_area_px", None)),
    }


def segmentation_params_hash(params: dict) -> str:
    """Stable hash of true segmentation identity params.

    This deliberately excludes GPU/CPU placement and area filters. GPU is an
    execution detail, while area filters are applied after Cellpose and can be
    retuned from persisted raw masks.
    """
    identity = {key: params.get(key, None) for key in SEGMENTATION_IDENTITY_KEYS}
    text = json.dumps(to_builtin(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return a SHA256 digest for a persisted artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
