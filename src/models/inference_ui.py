"""Pure helpers for persistent, threshold-aware inference presentation."""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from typing import Any

import cv2
import numpy as np


RESULT_STATE_KEY = "screening_inference_result"


def classify_pneumonia_probability(probability: float, threshold: float) -> bool:
    """Return True at or above the threshold; equality is positive by policy."""
    probability = float(probability)
    threshold = float(threshold)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return probability >= threshold


def image_fingerprint(image: np.ndarray) -> str:
    """Create a stable content identity for the standardized input image."""
    array = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def inference_identity(model_name: str, image: np.ndarray) -> str:
    """Identify the exact model/image pair that produced a result."""
    return f"{model_name}:{image_fingerprint(image)}"


def valid_persisted_result(
    state: MutableMapping[str, Any], identity: str
) -> dict[str, Any] | None:
    """Return a matching result or remove a stale model/image result."""
    result = state.get(RESULT_STATE_KEY)
    if not isinstance(result, dict) or result.get("identity") != identity:
        state.pop(RESULT_STATE_KEY, None)
        return None
    return result


def persist_result(state: MutableMapping[str, Any], result: dict[str, Any]) -> None:
    """Store a completed inference result for threshold-only reruns."""
    if not result.get("identity"):
        raise ValueError("Persisted inference results require an identity.")
    state[RESULT_STATE_KEY] = result


def gradcam_visuals(
    grayscale_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42
) -> tuple[np.ndarray, np.ndarray]:
    """Return an aligned RGB overlay and colorized raw Grad-CAM heatmap."""
    base = np.asarray(grayscale_image, dtype=np.float32)
    cam = np.asarray(heatmap, dtype=np.float32)
    if base.ndim != 2 or cam.ndim != 2:
        raise ValueError("Grad-CAM visualization expects 2-D grayscale and heatmap arrays.")
    if base.shape != cam.shape:
        cam = cv2.resize(cam, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")

    base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0)
    if base.max(initial=0.0) <= 1.0:
        base = base * 255.0
    base_u8 = np.clip(base, 0, 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)

    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    color_bgr = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha, color_rgb, alpha, 0)
    return overlay, color_rgb
