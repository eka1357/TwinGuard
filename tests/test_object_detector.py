"""Automated test suite for the TwinGuard neural object detector.

Verifies:
- Model checkpoint exists in checkpoints/object_detector.pt and loads cleanly.
- Forward pass tensor dimensions and bounds.
- 3D to 2D camera pinhole projection and ground-truth bounding box generation.
- Object detector inference on live rendered frames from MuJoCo cameras.
- Well-formed detection format (label, bbox [x1, y1, x2, y2], confidence).
- Synthetic dataset generation pipeline across randomized renders.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from simulation.simulator import TwinGuardSim
from perception.object_detector import (
    DEFAULT_CHECKPOINT_PATH,
    OBJECT_CLASSES,
    NUM_CLASSES,
    ObjectDetectorCNN,
    detect_objects,
    generate_synthetic_dataset,
    get_ground_truth_bboxes,
    project_point_to_camera,
)


@pytest.fixture
def sim():
    """Fixture providing a fresh TwinGuardSim instance."""
    sim_inst = TwinGuardSim()
    sim_inst.reset()
    yield sim_inst
    sim_inst.close()


def test_checkpoint_file_exists():
    """Verify that model weights file exists in checkpoints/ directory."""
    assert DEFAULT_CHECKPOINT_PATH.exists(), f"Checkpoint not found at {DEFAULT_CHECKPOINT_PATH}"
    assert DEFAULT_CHECKPOINT_PATH.stat().st_size > 100_000, "Checkpoint file unexpectedly small"


def test_model_architecture_forward():
    """Verify neural network forward pass tensor shapes and output ranges."""
    model = ObjectDetectorCNN(num_classes=NUM_CLASSES)
    model.eval()

    batch_size = 3
    dummy_input = torch.rand(batch_size, 3, 128, 128)

    with torch.no_grad():
        boxes, scores = model(dummy_input)

    assert boxes.shape == (batch_size, NUM_CLASSES, 4)
    assert scores.shape == (batch_size, NUM_CLASSES)

    # Sigmoid activations ensure outputs in [0, 1]
    assert (boxes >= 0.0).all() and (boxes <= 1.0).all()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_camera_projection_and_bbox_extraction(sim):
    """Verify 3D world to 2D image projection and bounding box generation."""
    width, height = 640, 480
    cam_name = "overview_cam"

    # Project table center [0.2, 0.0, 0.4]
    u, v = project_point_to_camera(sim, [0.2, 0.0, 0.4], camera_name=cam_name, width=width, height=height)
    assert 0 <= u < width
    assert 0 <= v < height

    # Ground truth bounding boxes
    gt_boxes = get_ground_truth_bboxes(sim, camera_name=cam_name, width=width, height=height, normalize=True)
    assert "plate" in gt_boxes
    assert "mug" in gt_boxes
    assert "drawer_handle" in gt_boxes

    for name, bbox in gt_boxes.items():
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert 0.0 <= x1 <= x2 <= 1.0, f"Invalid normalized bbox for {name}: {bbox}"
        assert 0.0 <= y1 <= y2 <= 1.0, f"Invalid normalized bbox for {name}: {bbox}"


def test_synthetic_dataset_generation(sim):
    """Verify synthetic dataset generator produces well-formed tensors."""
    num_samples = 3
    images, bboxes, labels = generate_synthetic_dataset(
        sim, num_samples=num_samples, img_size=(64, 64)
    )

    assert images.shape == (num_samples, 3, 64, 64)
    assert bboxes.shape == (num_samples, NUM_CLASSES, 4)
    assert labels.shape == (num_samples, NUM_CLASSES)
    assert images.dtype == torch.float32


def test_object_detector_inference_on_rendered_frames(sim):
    """Verify detector runs inference on rendered simulation frames and outputs well-formed detections."""
    # Test with overview camera
    img_overview = sim.get_camera_image("overview_cam")
    h_ov, w_ov = img_overview.shape[:2]

    res_overview = detect_objects(img_overview)
    assert "detections" in res_overview
    assert res_overview["image_size"] == [w_ov, h_ov]
    assert len(res_overview["detections"]) == NUM_CLASSES

    for det in res_overview["detections"]:
        assert det["label"] in OBJECT_CLASSES
        assert isinstance(det["confidence"], float)
        assert 0.0 <= det["confidence"] <= 1.0

        bbox = det["bbox"]
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert 0.0 <= x1 < x2 <= w_ov, f"Out of bounds x coordinates: {bbox}"
        assert 0.0 <= y1 < y2 <= h_ov, f"Out of bounds y coordinates: {bbox}"

    # Test with front camera
    img_front = sim.get_camera_image("front_cam")
    h_fr, w_fr = img_front.shape[:2]

    res_front = detect_objects(img_front)
    assert len(res_front["detections"]) == NUM_CLASSES
    for det in res_front["detections"]:
        assert det["label"] in OBJECT_CLASSES
        x1, y1, x2, y2 = det["bbox"]
        assert 0.0 <= x1 < x2 <= w_fr
        assert 0.0 <= y1 < y2 <= h_fr
