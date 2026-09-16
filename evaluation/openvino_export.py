"""Export neural object detector to OpenVINO Intermediate Representation (IR).

Converts trained PyTorch ObjectDetectorCNN checkpoint into OpenVINO IR format
(.xml and .bin), saving the artifacts to checkpoints/openvino/ for Intel hardware
inference and benchmarking.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openvino as ov
import torch

from perception.object_detector import (
    DEFAULT_CHECKPOINT_PATH,
    NUM_CLASSES,
    ObjectDetectorCNN,
)


DEFAULT_OPENVINO_DIR = PROJECT_ROOT / "checkpoints" / "openvino"
DEFAULT_XML_PATH = DEFAULT_OPENVINO_DIR / "object_detector.xml"


def export_to_openvino(
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    input_shape: Sequence[int] = (1, 3, 128, 128),
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """Convert PyTorch object detector model to OpenVINO IR format.

    Args:
        checkpoint_path: Path to PyTorch .pt weights file (defaults to checkpoints/object_detector.pt).
        output_dir: Destination directory for OpenVINO IR files (defaults to checkpoints/openvino/).
        input_shape: Expected input tensor shape (default: [1, 3, 128, 128]).
        verbose: Whether to print export progress messages.

    Returns:
        Tuple of (xml_path, bin_path) Path objects.
    """
    ckpt_p = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    out_d = Path(output_dir) if output_dir else DEFAULT_OPENVINO_DIR
    out_d.mkdir(parents=True, exist_ok=True)

    xml_path = out_d / "object_detector.xml"
    bin_path = out_d / "object_detector.bin"

    if not ckpt_p.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found at: {ckpt_p}")

    if verbose:
        print(f"[INFO] Loading PyTorch model from {ckpt_p}...")

    model = ObjectDetectorCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(str(ckpt_p), map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    if verbose:
        print(f"[INFO] Converting model to OpenVINO IR with input shape {list(input_shape)}...")

    ov_model = ov.convert_model(model, example_input=dummy_input)

    if verbose:
        print(f"[INFO] Saving OpenVINO IR model to {xml_path}...")

    ov.save_model(ov_model, str(xml_path))

    if verbose:
        print(f"[SUCCESS] OpenVINO model exported successfully:")
        print(f"  - Model XML: {xml_path} ({xml_path.stat().st_size} bytes)")
        print(f"  - Weights BIN: {bin_path} ({bin_path.stat().st_size} bytes)")

    return xml_path, bin_path


def main() -> None:
    """CLI entrypoint to convert PyTorch checkpoint to OpenVINO IR."""
    parser = argparse.ArgumentParser(description="Export TwinGuard Object Detector to OpenVINO IR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to trained PyTorch .pt checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OPENVINO_DIR),
        help="Target directory for OpenVINO IR artifacts",
    )
    args = parser.parse_args()

    export_to_openvino(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
