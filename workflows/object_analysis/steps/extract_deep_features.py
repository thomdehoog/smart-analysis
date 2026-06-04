"""extract_deep_features -- optional DINOv2 embeddings for object crops."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import to_builtin  # noqa: E402
from _object_crops import extract_object_crops  # noqa: E402


METADATA = {
    "description": "Extract optional DINOv2 embeddings for object crops",
    "version": "1.0",
    "max_workers": 1,
    "environment": "SMART--object_analysis--vision",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    inp = pipeline_data["input"]
    backend = _setting(inp, params, "backend", "dinov2")
    crop_result = extract_object_crops(
        image=pipeline_data["detect_objects"].get(
            "image", pipeline_data["detect_objects"]["image_2d"]
        ),
        masks=pipeline_data["detect_objects"]["masks"],
        tile_id=inp["tile_id"],
        context_multiplier=float(
            _setting(inp, params, "context_multiplier", 1.5)
        ),
        min_crop_size_px=int(_setting(inp, params, "min_crop_size_px", 64)),
        mode=_setting(inp, params, "crop_mode", "neighborhood"),
        output_dir=_setting(inp, params, "output_dir", None),
    )
    objects = list(crop_result["objects"])
    labels = [int(obj["label"]) for obj in objects]

    if backend == "mock":
        vectors = _mock_embeddings(
            objects,
            dim=int(_setting(inp, params, "embedding_dim", 8)),
        )
        model_name = "mock"
    elif backend == "dinov2":
        model_name = _setting(inp, params, "model_name", "dinov2_vitb14")
        vectors = _dinov2_embeddings(
            objects,
            state,
            model_name=model_name,
            input_size_px=int(_setting(inp, params, "input_size_px", 518)),
            batch_size=int(_setting(inp, params, "batch_size", 8)),
            device=_setting(inp, params, "device", None),
            disable_xformers=bool(_setting(inp, params, "disable_xformers", True)),
        )
    else:
        raise ValueError("backend must be 'dinov2' or 'mock'.")

    pipeline_data["extract_deep_features"] = {
        "embeddings": to_builtin({
            "label": labels,
            "vectors": vectors,
            "model": model_name,
            "backend": backend,
            "crop_policy": crop_result.get("crop_policy", {}),
        }),
        "objects": to_builtin([
            {
                key: value
                for key, value in obj.items()
                if key not in {"crop_image", "crop_mask"}
            }
            for obj in objects
        ]),
        "tile_artifacts": to_builtin(crop_result.get("tile_artifacts", {})),
    }
    return pipeline_data


def _setting(inp: dict, params: dict, key: str, default):
    return inp[key] if key in inp else params.get(key, default)


def _mock_embeddings(objects: list[dict], *, dim: int) -> list[list[float]]:
    vectors = []
    for obj in objects:
        image = np.asarray(obj["crop_image"], dtype=float)
        mask = np.asarray(obj["crop_mask"], dtype=bool)
        values = image[mask] if mask.any() and image.ndim == 2 else image.reshape(-1)
        base = np.asarray([
            float(obj["label"]),
            float(values.mean()) if values.size else 0.0,
            float(values.std()) if values.size else 0.0,
            float(mask.sum()),
        ])
        if dim <= len(base):
            vector = base[:dim]
        else:
            vector = np.resize(base, dim)
        vectors.append(_l2_normalize(vector).tolist())
    return vectors


def _dinov2_embeddings(
    objects: list[dict],
    state: dict,
    *,
    model_name: str,
    input_size_px: int,
    batch_size: int,
    device,
    disable_xformers: bool,
) -> list[list[float]]:
    if not objects:
        return []

    if disable_xformers:
        import os

        # DINOv2 runs correctly without xFormers. On some Windows Torch
        # environments, importing xFormers can raise a DLL-level fatal error,
        # so keep the optional acceleration off unless explicitly requested.
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if input_size_px <= 0:
        raise ValueError("input_size_px must be > 0.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_key = (model_name, device)
    if state.get("dinov2_model_key") != model_key:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval().to(device)
        state["dinov2_model"] = model
        state["dinov2_model_key"] = model_key
    model = state["dinov2_model"]

    vectors = []
    with torch.no_grad():
        for start in range(0, len(objects), batch_size):
            batch_objects = objects[start:start + batch_size]
            batch = _crop_batch(batch_objects, input_size_px, torch).to(device)
            features = _forward_dinov2(model, batch, torch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            vectors.extend(features.cpu().numpy().astype(float).tolist())
    return vectors


def _crop_batch(objects: list[dict], input_size_px: int, torch):
    tensors = []
    for obj in objects:
        rgb = _as_rgb(np.asarray(obj["crop_image"]))
        rgb = _percentile_normalize(rgb)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(input_size_px, input_size_px),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensors.append((tensor - mean) / std)
    return torch.cat(tensors, dim=0)


def _forward_dinov2(model, batch, torch):
    if hasattr(model, "forward_features"):
        features = model.forward_features(batch)
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1)
        return features
    return model(batch)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3:
        if image.shape[0] <= 4 and image.shape[2] > 4:
            raise ValueError(
                f"Cannot convert crop with shape {image.shape} to RGB. "
                "Expected a 2D crop or a channel-last 2D+channels crop."
            )
        n_channels = image.shape[2]
        if n_channels >= 3:
            return image[:, :, :3]
        if n_channels >= 1:
            pad = np.repeat(image[:, :, -1:], 3 - n_channels, axis=2)
            return np.concatenate([image, pad], axis=2)
    raise ValueError(
        f"Cannot convert crop with shape {image.shape} to RGB. "
        "Expected a 2D crop or a channel-last 2D+channels crop."
    )


def _percentile_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image.astype(np.float32, copy=False)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(float)
    return (vector / norm).astype(float)
