"""Visual Anomaly Detector for TwinGuard closed-loop manipulation.

Inspired by Intel Anomalib (v2.6.0), this module monitors off-screen camera frames
in real-time to detect physical execution anomalies (e.g. dropped objects, spills,
tipped containers, unexpected visual occlusions) directly from pixel data.
Complements physical coordinate verification with visual safety checks.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ANOMALY_WEIGHTS = PROJECT_ROOT / "checkpoints" / "visual_anomaly_detector.pt"
DEFAULT_OPENVINO_ANOMALY_DIR = PROJECT_ROOT / "checkpoints" / "openvino"


class VisualAnomalyDetector(nn.Module):
    """Lightweight convolutional feature embedding network for visual anomaly detection.

    Extracts spatial-semantic embeddings from camera frames (RGB 128x128) and
    computes embedding distance / reconstruction error against normal manipulation baselines.
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Feature extractor backbone (3 conv blocks with batch norm and LeakyReLU)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Projection head to compact embedding space
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized visual feature embedding vector.

        Args:
            x: Tensor of shape (B, 3, H, W) in range [0, 1].

        Returns:
            Tensor of shape (B, embedding_dim) normalized to unit hypersphere.
        """
        feats = self.encoder(x)
        emb = self.head(feats)
        return F.normalize(emb, p=2, dim=-1)


class VisualGuard:
    """Runtime visual anomaly detector verifying post-step scene integrity."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.45,
        device: str = "cpu",
        use_openvino: bool = False,
    ):
        self.threshold = threshold
        self.device = device
        self.use_openvino = use_openvino
        self.reference_embedding: Optional[np.ndarray] = None

        self.model = VisualAnomalyDetector(embedding_dim=64)
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_ANOMALY_WEIGHTS

        if self.weights_path.exists():
            try:
                self.model.load_state_dict(
                    torch.load(str(self.weights_path), map_location="cpu", weights_only=True)
                )
            except Exception:
                pass
        self.model.eval()

        # OpenVINO compiled model if requested
        self.ov_compiled = None
        if use_openvino:
            self._init_openvino()

    def _init_openvino(self) -> None:
        """Initialize OpenVINO execution engine for visual anomaly inference."""
        try:
            import openvino as ov
            core = ov.Core()
            xml_path = DEFAULT_OPENVINO_ANOMALY_DIR / "visual_anomaly_detector.xml"
            if xml_path.exists():
                model = core.read_model(xml_path)
                self.ov_compiled = core.compile_model(model, "CPU")
        except Exception:
            self.ov_compiled = None

    def preprocess_image(self, img_array: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
        """Preprocess uint8 RGB camera image into normalized PyTorch tensor [1, 3, H, W]."""
        h, w, c = img_array.shape

        # Simple manual bilinear downsample if cv2 not available
        if target_size != (h, w):
            try:
                import cv2
                resized = cv2.resize(img_array, target_size)
            except Exception:
                # Basic strided decimation
                sy = max(1, h // target_size[0])
                sx = max(1, w // target_size[1])
                resized = img_array[::sy, ::sx, :][:target_size[0], :target_size[1]]
        else:
            resized = img_array

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def extract_embedding(self, img_array: np.ndarray) -> np.ndarray:
        """Extract 64-dimensional normalized visual feature embedding."""
        tensor = self.preprocess_image(img_array)
        if self.ov_compiled is not None:
            inp = tensor.numpy()
            res = self.ov_compiled([inp])
            return res[0][0]

        with torch.no_grad():
            emb = self.model(tensor).squeeze(0).cpu().numpy()
        return emb

    def set_reference_state(self, baseline_img: np.ndarray) -> None:
        """Record the reference visual embedding of the initial clean scene."""
        self.reference_embedding = self.extract_embedding(baseline_img)

    def check_visual_anomaly(
        self,
        current_img: np.ndarray,
        expected_difference: bool = False,
    ) -> Dict[str, Any]:
        """Compute cosine visual distance to detect physical scene disruption.

        Args:
            current_img: Current RGB frame from camera.
            expected_difference: Whether the current step intentionally altered the scene.

        Returns:
            Dict containing 'anomaly_detected': bool, 'score': float, 'threshold': float.
        """
        curr_emb = self.extract_embedding(current_img)

        if self.reference_embedding is None:
            self.reference_embedding = curr_emb
            return {"anomaly_detected": False, "score": 0.0, "threshold": self.threshold}

        # Cosine distance = 1.0 - dot_product
        dot_product = float(np.dot(self.reference_embedding, curr_emb))
        cosine_dist = float(np.clip(1.0 - dot_product, 0.0, 2.0))

        # Adjust threshold when intentional physical transport/pour occurred
        effective_thresh = self.threshold * 1.8 if expected_difference else self.threshold
        anomaly = cosine_dist > effective_thresh

        return {
            "anomaly_detected": anomaly,
            "score": round(cosine_dist, 4),
            "threshold": round(effective_thresh, 4),
            "reason": f"Visual anomaly score {cosine_dist:.3f} exceeded threshold {effective_thresh:.3f}" if anomaly else "Visual inspection passed",
        }
