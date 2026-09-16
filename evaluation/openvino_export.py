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
DEFAULT_INT8_XML_PATH = DEFAULT_OPENVINO_DIR / "object_detector_int8.xml"


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


def export_to_openvino_int8(
    model_xml: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: int = 30,
    input_shape: Sequence[int] = (1, 3, 128, 128),
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """Quantize OpenVINO IR model to INT8 precision using Intel NNCF.

    Args:
        model_xml: Path to source FP32 object_detector.xml.
        output_dir: Target directory for INT8 artifacts.
        num_samples: Number of calibration frames to generate.
        input_shape: Input tensor dimensions.
        verbose: Whether to print export logs.

    Returns:
        Tuple of (int8_xml_path, int8_bin_path).
    """
    import nncf
    import numpy as np

    xml_p = Path(model_xml) if model_xml else DEFAULT_XML_PATH
    if not xml_p.exists():
        export_to_openvino(output_dir=output_dir, input_shape=input_shape, verbose=verbose)

    out_d = Path(output_dir) if output_dir else DEFAULT_OPENVINO_DIR
    out_d.mkdir(parents=True, exist_ok=True)

    int8_xml_path = out_d / "object_detector_int8.xml"
    int8_bin_path = out_d / "object_detector_int8.bin"

    if verbose:
        print(f"[INFO] Initializing NNCF INT8 post-training quantization for {xml_p}...")

    core = ov.Core()
    ov_model = core.read_model(str(xml_p))

    # Generate calibration dataset
    calib_frames = [
        np.clip(np.random.randn(*input_shape).astype(np.float32) * 0.2 + 0.5, 0.0, 1.0)
        for _ in range(num_samples)
    ]
    calibration_dataset = nncf.Dataset(calib_frames)

    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        subset_size=num_samples,
        fast_bias_correction=True,
    )

    ov.save_model(quantized_model, str(int8_xml_path))

    if verbose:
        fp32_size = (out_d / "object_detector.bin").stat().st_size if (out_d / "object_detector.bin").exists() else 0
        int8_size = int8_bin_path.stat().st_size
        comp_ratio = (fp32_size / int8_size) if int8_size > 0 else 1.0
        print(f"[SUCCESS] OpenVINO INT8 model quantized successfully with NNCF:")
        print(f"  - INT8 Model XML: {int8_xml_path} ({int8_xml_path.stat().st_size} bytes)")
        print(f"  - INT8 Weights BIN: {int8_bin_path} ({int8_size} bytes)")
        print(f"  - Compression Ratio: {comp_ratio:.2f}x (from {fp32_size} bytes to {int8_size} bytes)")

    return int8_xml_path, int8_bin_path


def export_visual_guard_to_openvino(
    output_dir: Optional[Union[str, Path]] = None,
    input_shape: Sequence[int] = (1, 3, 128, 128),
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """Export VisualAnomalyDetector (Anomalib-aligned) to OpenVINO IR format.

    Args:
        output_dir: Destination directory for IR files.
        input_shape: Input shape (1, 3, 128, 128).
        verbose: Print progress.

    Returns:
        Tuple of (xml_path, bin_path).
    """
    from safety.visual_guard import VisualAnomalyDetector

    out_d = Path(output_dir) if output_dir else DEFAULT_OPENVINO_DIR
    out_d.mkdir(parents=True, exist_ok=True)

    xml_path = out_d / "visual_anomaly_detector.xml"
    bin_path = out_d / "visual_anomaly_detector.bin"

    if verbose:
        print(f"[INFO] Initializing VisualAnomalyDetector architecture...")

    model = VisualAnomalyDetector(embedding_dim=64)
    model.eval()

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    if verbose:
        print(f"[INFO] Converting visual anomaly model to OpenVINO IR...")

    ov_model = ov.convert_model(model, example_input=dummy_input)
    ov.save_model(ov_model, str(xml_path))

    if verbose:
        print(f"[SUCCESS] Visual Anomaly OpenVINO model exported:")
        print(f"  - XML: {xml_path} ({xml_path.stat().st_size} bytes)")
        print(f"  - BIN: {bin_path} ({bin_path.stat().st_size} bytes)")

    return xml_path, bin_path


def main() -> None:
    """CLI entrypoint to convert PyTorch models to OpenVINO IR and quantize to INT8."""
    parser = argparse.ArgumentParser(description="Export TwinGuard Perception & Anomaly Models to OpenVINO IR & INT8")
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
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        default=True,
        help="Apply Intel NNCF INT8 post-training quantization",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Export both Object Detector and Visual Anomaly Detector",
    )
    args = parser.parse_args()

    xml_p, bin_p = export_to_openvino(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        verbose=True,
    )

    if args.quantize_int8:
        export_to_openvino_int8(
            model_xml=xml_p,
            output_dir=args.output_dir,
            verbose=True,
        )

    if args.all:
        export_visual_guard_to_openvino(
            output_dir=args.output_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
