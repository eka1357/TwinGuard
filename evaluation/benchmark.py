"""OpenVINO inference latency and throughput benchmark for TwinGuard.

Measures inference performance of the exported OpenVINO IR object detector model
on Intel-targeted execution hardware (CPU, iGPU, or NPU).

NOTE: The current development machine is an AMD Ryzen laptop.
This execution provides functional verification and pipeline correctness only,
NOT a valid graded benchmark number. The exact same benchmark script must be
executed on genuine Intel Core Ultra hardware (CPU/iGPU/NPU) for official challenge
submission metrics.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import yaml

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openvino as ov

from evaluation.openvino_export import DEFAULT_XML_PATH, export_to_openvino


# ------------------------------------------------------------------------------
# One-Line Device Selection
# Change DEVICE to "GPU" or "NPU" to target Intel Core Ultra accelerators
# ------------------------------------------------------------------------------
DEFAULT_DEVICE: str = "CPU"  # Options: "CPU", "GPU", "NPU"


def load_benchmark_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load benchmark parameters from configs/sim_config.yaml."""
    cfg_p = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "sim_config.yaml"
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            return full_cfg.get("evaluation", {}).get("benchmark", {})
    return {}


def run_benchmark(
    model_xml: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    num_iterations: Optional[int] = None,
    warmup_iterations: Optional[int] = None,
    input_shape: Sequence[int] = (1, 3, 128, 128),
    config_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute inference benchmark on OpenVINO IR model and return latency/throughput metrics.

    Args:
        model_xml: Path to object_detector.xml. Exported automatically if missing.
        device: Target execution device ('CPU', 'GPU', 'NPU'). Defaults to config or 'CPU'.
        num_iterations: Number of timed inference iterations.
        warmup_iterations: Warmup cycles before timing begins.
        input_shape: Dimensions of dummy test input tensor.
        config_path: Path to sim_config.yaml.
        verbose: Whether to print benchmark summary table to stdout.

    Returns:
        Dict containing mean, median (p50), p95, min, max latencies (ms) and throughput (FPS).
    """
    cfg = load_benchmark_config(config_path)

    # Determine configuration values with fallback defaults
    target_device = str(device or cfg.get("device", DEFAULT_DEVICE))
    n_iters = int(num_iterations or cfg.get("num_iterations", 100))
    n_warmup = int(warmup_iterations or cfg.get("warmup_iterations", 10))

    xml_p = Path(model_xml) if model_xml else DEFAULT_XML_PATH

    # If OpenVINO model has not yet been exported, export it automatically
    if not xml_p.exists():
        if verbose:
            print(f"[INFO] OpenVINO model not found at {xml_p}. Exporting now...")
        export_to_openvino(output_dir=xml_p.parent, input_shape=input_shape, verbose=verbose)

    # Initialize OpenVINO runtime
    core = ov.Core()

    # Query available devices
    available_devices = core.available_devices
    if verbose:
        print(f"[INFO] OpenVINO available devices: {available_devices}")
        print(f"[INFO] Selected target device: '{target_device}'")

    # Load and compile model onto selected device
    compiled_model = core.compile_model(str(xml_p), device_name=target_device)
    infer_request = compiled_model.create_infer_request()

    # Prepare dummy input tensor matching expected shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup phase (untimed)
    if verbose:
        print(f"[INFO] Running {n_warmup} warmup iterations...")
    for _ in range(n_warmup):
        infer_request.infer({0: dummy_input})

    # Timed benchmark loop
    if verbose:
        print(f"[INFO] Running {n_iters} benchmark iterations on {target_device}...")

    latencies_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        infer_request.infer({0: dummy_input})
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    # Compute statistical metrics
    arr = np.array(latencies_ms)
    mean_lat = float(np.mean(arr))
    median_lat = float(np.median(arr))  # P50
    p95_lat = float(np.percentile(arr, 95))  # P95
    min_lat = float(np.min(arr))
    max_lat = float(np.max(arr))
    std_lat = float(np.std(arr))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    metrics = {
        "device": target_device,
        "iterations": n_iters,
        "warmup": n_warmup,
        "mean_latency_ms": round(mean_lat, 3),
        "median_p50_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "throughput_fps": round(fps, 1),
    }

    if verbose:
        print("\n" + "=" * 65)
        print(" TWINGUARD OPENVINO INFERENCE BENCHMARK REPORT")
        print("=" * 65)
        print(f" Target Device:      {target_device}")
        print(f" Iterations:         {n_iters} (warmup: {n_warmup})")
        print(f" Input Shape:        {list(input_shape)}")
        print("-" * 65)
        print(f" Mean Latency:       {metrics['mean_latency_ms']:>8.3f} ms")
        print(f" Median (P50):       {metrics['median_p50_latency_ms']:>8.3f} ms")
        print(f" 95th Percentile:    {metrics['p95_latency_ms']:>8.3f} ms")
        print(f" Min Latency:        {metrics['min_latency_ms']:>8.3f} ms")
        print(f" Max Latency:        {metrics['max_latency_ms']:>8.3f} ms")
        print(f" Standard Dev:       {metrics['std_latency_ms']:>8.3f} ms")
        print(f" Throughput:         {metrics['throughput_fps']:>8.1f} FPS")
        print("=" * 65)
        print(" [NOTE] Development machine: AMD Ryzen laptop (functional test only).")
        print(" Official evaluation MUST be re-run on Intel Core Ultra hardware.")
        print("=" * 65 + "\n")

    return metrics


def main() -> None:
    """CLI entrypoint to run OpenVINO benchmark."""
    parser = argparse.ArgumentParser(description="TwinGuard OpenVINO Hardware Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'CPU', 'GPU', 'NPU' (defaults to sim_config.yaml)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of timed benchmark iterations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to OpenVINO object_detector.xml",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to sim_config.yaml",
    )
    args = parser.parse_args()

    run_benchmark(
        model_xml=args.model,
        device=args.device,
        num_iterations=args.iterations,
        config_path=args.config,
        verbose=True,
    )


if __name__ == "__main__":
    main()
