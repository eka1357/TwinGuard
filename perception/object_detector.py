"""Lightweight neural object detector for TwinGuard camera observations.

Provides:
- ObjectDetectorCNN: Lightweight PyTorch CNN predicting bounding boxes and class
  confidences for scene objects (plate, mug, drawer_handle).
- Camera projection: Automatically computes 2D ground-truth bounding boxes from 3D
  MuJoCo positions without manual annotation.
- Synthetic training pipeline: Generates randomized synthetic renders and trains the model.
- Inference: Runs inference on camera frames rendered by simulator.get_camera_image().
- Checkpoint persistence: Saves/loads model weights to/from checkpoints/.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim

from simulation.simulator import TwinGuardSim


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "object_detector.pt"

# Detectable classes for dinner table manipulation
OBJECT_CLASSES = ("plate", "mug", "drawer_handle")
NUM_CLASSES = len(OBJECT_CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(OBJECT_CLASSES)}


# ------------------------------------------------------------------------------
# 1. Camera Projection & Auto-Labeling
# ------------------------------------------------------------------------------

def project_point_to_camera(
    sim: TwinGuardSim,
    point_world: Sequence[float],
    camera_name: str = "overview_cam",
    width: int = 640,
    height: int = 480,
) -> Tuple[float, float]:
    """Project a 3D world coordinate onto 2D image pixel coordinates (u, v).

    Args:
        sim: Active TwinGuardSim simulation instance.
        point_world: [x, y, z] coordinate in world frame.
        camera_name: Name of camera in MJCF.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        (u, v) pixel coordinates in image frame.
    """
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found in simulation model")

    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = sim.model.cam_fovy[cam_id]

    # Transform vector into camera coordinates
    p_world = np.asarray(point_world, dtype=np.float64)
    p_cam = cam_mat.T @ (p_world - cam_pos)

    # Focal length from vertical field of view
    f = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)

    # Pinhole projection (MuJoCo camera looks down negative Z)
    z_cam = -p_cam[2]
    if z_cam <= 1e-4:
        z_cam = 1e-4

    u = (width / 2.0) + f * (p_cam[0] / z_cam)
    v = (height / 2.0) - f * (p_cam[1] / z_cam)

    return float(u), float(v)


def get_ground_truth_bboxes(
    sim: TwinGuardSim,
    camera_name: str = "overview_cam",
    width: int = 640,
    height: int = 480,
    normalize: bool = True,
) -> Dict[str, List[float]]:
    """Compute 2D bounding boxes for plate, mug, and drawer_handle.

    Args:
        sim: Active TwinGuardSim simulation instance.
        camera_name: Camera viewpoint to project against.
        width: Rendered image width.
        height: Rendered image height.
        normalize: If True, coordinates are normalized to [0, 1].

    Returns:
        Dict mapping object class name to [x1, y1, x2, y2] bounding box.
    """
    bboxes: Dict[str, List[float]] = {}

    # Object geometry extents [dx, dy, dz] (half-sizes)
    extents = {
        "plate": (0.08, 0.08, 0.02),
        "mug": (0.04, 0.04, 0.05),
        "drawer_handle": (0.02, 0.03, 0.02),
    }

    for obj_name in OBJECT_CLASSES:
        if obj_name == "drawer_handle":
            sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site")
            if sid != -1:
                center = np.copy(sim.data.site_xpos[sid])
            else:
                bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
                center = np.copy(sim.data.xpos[bid]) if bid != -1 else np.array([0.26, 0.22, 0.46])
        else:
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if bid != -1:
                center = np.copy(sim.data.xpos[bid])
            else:
                continue

        dx, dy, dz = extents[obj_name]

        # Generate 8 corner vertices in 3D
        corners_3d = [
            center + np.array([sx * dx, sy * dy, sz * dz])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]

        # Project corners into 2D pixel coordinates
        proj_pixels = [
            project_point_to_camera(sim, corner, camera_name=camera_name, width=width, height=height)
            for corner in corners_3d
        ]

        us = [p[0] for p in proj_pixels]
        vs = [p[1] for p in proj_pixels]

        x1 = max(0.0, float(min(us)))
        y1 = max(0.0, float(min(vs)))
        x2 = min(float(width), float(max(us)))
        y2 = min(float(height), float(max(vs)))

        if normalize:
            bboxes[obj_name] = [
                round(x1 / width, 4),
                round(y1 / height, 4),
                round(x2 / width, 4),
                round(y2 / height, 4),
            ]
        else:
            bboxes[obj_name] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    return bboxes


# ------------------------------------------------------------------------------
# 2. Lightweight PyTorch CNN Model
# ------------------------------------------------------------------------------

class ObjectDetectorCNN(nn.Module):
    """Lightweight convolutional neural network for multi-class object detection."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (128, 4, 4)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Bounding box regression head: [B, num_classes, 4] normalized coordinates
        self.bbox_head = nn.Sequential(
            nn.Linear(128, num_classes * 4),
            nn.Sigmoid(),
        )

        # Class confidence score head: [B, num_classes] in [0, 1]
        self.conf_head = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor of shape [B, 3, H, W] normalized in [0, 1].

        Returns:
            boxes: [B, num_classes, 4] normalized [x1, y1, x2, y2].
            scores: [B, num_classes] confidence scores in [0, 1].
        """
        feats = self.features(x)
        embed = self.dense(feats)

        boxes = self.bbox_head(embed).view(-1, self.num_classes, 4)
        scores = self.conf_head(embed)

        return boxes, scores


# ------------------------------------------------------------------------------
# 3. Synthetic Dataset Generation & Training
# ------------------------------------------------------------------------------

def generate_synthetic_dataset(
    sim: TwinGuardSim,
    num_samples: int = 15,
    camera_name: str = "overview_cam",
    img_size: Tuple[int, int] = (128, 128),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic training dataset with ground-truth bounding box labels.

    Args:
        sim: Active TwinGuardSim simulation instance.
        num_samples: Number of randomized synthetic frames to render.
        camera_name: Camera viewpoint.
        img_size: Rendered image resolution (width, height).

    Returns:
        images: [N, 3, H, W] float32 tensor.
        bboxes: [N, num_classes, 4] float32 tensor.
        labels: [N, num_classes] float32 tensor (1.0 for visible objects).
    """
    w, h = img_size
    images_list = []
    bboxes_list = []
    labels_list = []

    # Backup original simulation state
    orig_qpos = np.copy(sim.data.qpos)
    orig_qvel = np.copy(sim.data.qvel)

    plate_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    plate_qpos_adr = sim.model.jnt_qposadr[plate_jid] if plate_jid != -1 else -1

    mug_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "mug_joint")
    mug_qpos_adr = sim.model.jnt_qposadr[mug_jid] if mug_jid != -1 else -1

    drawer_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
    drawer_qpos_adr = sim.model.jnt_qposadr[drawer_jid] if drawer_jid != -1 else -1

    np.random.seed(42)

    for i in range(num_samples):
        # Randomize object locations within reachable table surface
        if plate_qpos_adr != -1:
            px = np.random.uniform(0.20, 0.28)
            py = np.random.uniform(-0.06, 0.06)
            sim.data.qpos[plate_qpos_adr : plate_qpos_adr + 3] = [px, py, 0.435]

        if mug_qpos_adr != -1:
            mx = np.random.uniform(0.18, 0.26)
            my = np.random.uniform(-0.20, -0.10)
            sim.data.qpos[mug_qpos_adr : mug_qpos_adr + 3] = [mx, my, 0.465]

        if drawer_qpos_adr != -1:
            disp = np.random.uniform(0.0, 0.04)
            sim.data.qpos[drawer_qpos_adr] = disp

        mujoco.mj_forward(sim.model, sim.data)

        # Render camera image
        img_np = sim.get_camera_image(camera_name, width=w, height=h)
        # Convert uint8 [H, W, 3] -> float32 [3, H, W] normalized in [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Compute ground truth bounding boxes
        gt_boxes = get_ground_truth_bboxes(sim, camera_name=camera_name, width=w, height=h, normalize=True)

        boxes_tensor = torch.zeros(NUM_CLASSES, 4, dtype=torch.float32)
        conf_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for c_idx, c_name in enumerate(OBJECT_CLASSES):
            if c_name in gt_boxes:
                boxes_tensor[c_idx] = torch.tensor(gt_boxes[c_name], dtype=torch.float32)
                conf_tensor[c_idx] = 1.0

        images_list.append(img_tensor)
        bboxes_list.append(boxes_tensor)
        labels_list.append(conf_tensor)

    # Restore original simulation state
    sim.data.qpos[:] = orig_qpos
    sim.data.qvel[:] = orig_qvel
    mujoco.mj_forward(sim.model, sim.data)

    images = torch.stack(images_list, dim=0)
    bboxes = torch.stack(bboxes_list, dim=0)
    labels = torch.stack(labels_list, dim=0)

    return images, bboxes, labels


def train_and_save_detector(
    sim: Optional[TwinGuardSim] = None,
    epochs: int = 5,
    num_samples: int = 15,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> ObjectDetectorCNN:
    """Train the lightweight detector on synthetic simulation renders and save checkpoint.

    Args:
        sim: Optional TwinGuardSim instance. Created if None.
        epochs: Number of training epochs (default: 5).
        num_samples: Number of synthetic training images.
        checkpoint_path: Destination path for model weights.

    Returns:
        Trained ObjectDetectorCNN model instance.
    """
    owns_sim = sim is None
    if sim is None:
        sim = TwinGuardSim()
        sim.reset()

    save_p = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    save_p.parent.mkdir(parents=True, exist_ok=True)

    images, bboxes, labels = generate_synthetic_dataset(sim, num_samples=num_samples)

    model = ObjectDetectorCNN(num_classes=NUM_CLASSES)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    box_criterion = nn.SmoothL1Loss()
    conf_criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_boxes, pred_scores = model(images)

        loss_box = box_criterion(pred_boxes, bboxes)
        loss_conf = conf_criterion(pred_scores, labels)
        total_loss = loss_box + loss_conf

        total_loss.backward()
        optimizer.step()

    # Save trained checkpoint weights
    torch.save(model.state_dict(), str(save_p))

    if owns_sim:
        sim.close()

    model.eval()
    return model


# ------------------------------------------------------------------------------
# 4. Inference Interface
# ------------------------------------------------------------------------------

def detect_objects(
    image: np.ndarray,
    model: Optional[ObjectDetectorCNN] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    score_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Run object detection inference on a rendered camera frame.

    Args:
        image: uint8 numpy array of shape (height, width, 3) as returned by get_camera_image().
        model: Optional pre-loaded ObjectDetectorCNN. Loaded from checkpoint if None.
        checkpoint_path: Optional custom path to model weights checkpoint.
        score_threshold: Minimum confidence score threshold.

    Returns:
        Dict with "detections": List of {"label": str, "bbox": [x1, y1, x2, y2], "confidence": float}.
    """
    orig_h, orig_w = image.shape[:2]

    # Preprocess image into [1, 3, 128, 128] tensor
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_resized = torch.nn.functional.interpolate(
        img_tensor, size=(128, 128), mode="bilinear", align_corners=False
    )

    # Load model if not provided
    if model is None:
        save_p = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        model = ObjectDetectorCNN(num_classes=NUM_CLASSES)
        if save_p.exists():
            model.load_state_dict(torch.load(str(save_p), map_location="cpu", weights_only=True))
        else:
            # If checkpoint does not yet exist, train and save on the fly
            train_and_save_detector(checkpoint_path=save_p)

    model.eval()
    with torch.no_grad():
        boxes, scores = model(img_resized)

    boxes = boxes.squeeze(0).cpu().numpy()  # [num_classes, 4]
    scores = scores.squeeze(0).cpu().numpy()  # [num_classes]

    detections: List[Dict[str, Any]] = []

    for c_idx, c_name in enumerate(OBJECT_CLASSES):
        conf = float(scores[c_idx])
        norm_box = boxes[c_idx]

        # Scale normalized box coordinates back to original image resolution
        x1 = round(float(norm_box[0] * orig_w), 1)
        y1 = round(float(norm_box[1] * orig_h), 1)
        x2 = round(float(norm_box[2] * orig_w), 1)
        y2 = round(float(norm_box[3] * orig_h), 1)

        # Ensure valid box bounds
        x1 = max(0.0, min(float(orig_w), x1))
        y1 = max(0.0, min(float(orig_h), y1))
        x2 = max(x1 + 1.0, min(float(orig_w), x2))
        y2 = max(y1 + 1.0, min(float(orig_h), y2))

        detections.append({
            "label": c_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 3),
        })

    return {
        "image_size": [orig_w, orig_h],
        "detections": detections,
    }
