"""Regression tests for threshold-aware, persistent inference presentation."""

import numpy as np
import pytest

from src.models.inference_ui import (
    RESULT_STATE_KEY,
    classify_pneumonia_probability,
    gradcam_visuals,
    inference_identity,
    persist_result,
    valid_persisted_result,
)


def test_threshold_changes_class_not_probability() -> None:
    probability = 0.70
    assert classify_pneumonia_probability(probability, 0.50) is True
    assert classify_pneumonia_probability(probability, 0.90) is False
    assert probability == 0.70


def test_probability_equal_to_threshold_is_positive() -> None:
    assert classify_pneumonia_probability(0.50, 0.50) is True


@pytest.mark.parametrize("probability,threshold", [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)])
def test_threshold_policy_rejects_invalid_values(probability: float, threshold: float) -> None:
    with pytest.raises(ValueError):
        classify_pneumonia_probability(probability, threshold)


def test_result_persists_for_same_model_and_image() -> None:
    image = np.zeros((224, 224), dtype=np.float32)
    identity = inference_identity("resnet18", image)
    state: dict = {}
    result = {"identity": identity, "pneumonia_probability": 0.7}
    persist_result(state, result)
    assert valid_persisted_result(state, identity) is result


def test_model_switch_invalidates_stale_result() -> None:
    image = np.zeros((224, 224), dtype=np.float32)
    old_identity = inference_identity("resnet18", image)
    state = {RESULT_STATE_KEY: {"identity": old_identity, "pneumonia_probability": 0.7}}
    new_identity = inference_identity("hog_rbf_svm", image)
    assert valid_persisted_result(state, new_identity) is None
    assert RESULT_STATE_KEY not in state


def test_image_switch_invalidates_stale_result() -> None:
    first = np.zeros((224, 224), dtype=np.float32)
    second = first.copy()
    second[0, 0] = 1.0
    old_identity = inference_identity("resnet18", first)
    state = {RESULT_STATE_KEY: {"identity": old_identity, "pneumonia_probability": 0.7}}
    assert valid_persisted_result(state, inference_identity("resnet18", second)) is None
    assert RESULT_STATE_KEY not in state


def test_gradcam_visuals_are_aligned_color_images() -> None:
    image = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (64, 1))
    heatmap = np.zeros((16, 20), dtype=np.float32)
    heatmap[4:12, 5:15] = 1.0
    overlay, colorized = gradcam_visuals(image, heatmap)
    assert overlay.shape == (64, 80, 3)
    assert colorized.shape == (64, 80, 3)
    assert overlay.dtype == np.uint8
    assert colorized.dtype == np.uint8
    assert not np.array_equal(colorized[..., 0], colorized[..., 1])
    assert not np.array_equal(overlay[..., 0], overlay[..., 2])
