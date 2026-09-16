"""TwinGuard Perception Module.

Handles camera feeds, visual feature extraction, and workspace observation.
"""

from perception.object_detector import (
    OBJECT_CLASSES,
    ObjectDetectorCNN,
    detect_objects,
    get_ground_truth_bboxes,
    project_point_to_camera,
    train_and_save_detector,
)
from perception.scene_state import (
    ArmState,
    ObjectState,
    SceneState,
    format_scene_state,
    get_ground_truth_scene_state,
    get_scene_state_prompt,
)

__all__ = [
    "ArmState",
    "OBJECT_CLASSES",
    "ObjectDetectorCNN",
    "ObjectState",
    "SceneState",
    "detect_objects",
    "format_scene_state",
    "get_ground_truth_bboxes",
    "get_ground_truth_scene_state",
    "get_scene_state_prompt",
    "project_point_to_camera",
    "train_and_save_detector",
]
