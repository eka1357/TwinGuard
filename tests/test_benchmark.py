"""Automated test suite for TwinGuard OpenVINO export and inference benchmark.

Verifies:
- OpenVINO IR model files (object_detector.xml, object_detector.bin) exist and are valid.
- OpenVINO runtime loads the compiled model on CPU and performs inference on dummy input without error.
- Inference output shapes and types match expected detector dimensions.
- Exporter utility function correctly exports to custom target directories.
- Benchmarking engine computes valid latency percentiles (mean, p50, p95) and throughput.
"""

from pathlib import Path
import numpy as np
import openvino as ov
import pytest

from evaluation.benchmark import (
    DEFAULT_DEVICE,
    load_benchmark_config,
    run_benchmark,
)
from evaluation.openvino_export import (
    DEFAULT_OPENVINO_DIR,
    DEFAULT_XML_PATH,
    export_to_openvino,
)
from perception.object_detector import NUM_CLASSES


def test_openvino_ir_files_exist():
    """Verify that OpenVINO IR XML and BIN files exist in checkpoints/openvino/."""
    bin_path = DEFAULT_OPENVINO_DIR / "object_detector.bin"

    assert DEFAULT_XML_PATH.exists(), f"OpenVINO XML file missing: {DEFAULT_XML_PATH}"
    assert bin_path.exists(), f"OpenVINO BIN file missing: {bin_path}"
    assert DEFAULT_XML_PATH.stat().st_size > 1000, "OpenVINO XML file unexpectedly small"
    assert bin_path.stat().st_size > 100_000, "OpenVINO BIN file unexpectedly small"


def test_openvino_model_loads_and_runs_dummy_input():
    """Verify OpenVINO model loads into runtime and executes inference on dummy input."""
    core = ov.Core()
    compiled_model = core.compile_model(str(DEFAULT_XML_PATH), device_name="CPU")
    infer_request = compiled_model.create_infer_request()

    # Model input shape is [1, 3, 128, 128]
    dummy_input = np.random.randn(1, 3, 128, 128).astype(np.float32)

    # Perform inference
    results = infer_request.infer({0: dummy_input})

    # Results should contain output tensors
    assert len(results) >= 2, "Expected at least 2 outputs (boxes, scores)"

    output_tensors = list(results.values())
    shapes = [t.shape for t in output_tensors]

    # Expected: (1, NUM_CLASSES, 4) for boxes, (1, NUM_CLASSES) for scores
    assert (1, NUM_CLASSES, 4) in shapes, f"Expected box shape (1, {NUM_CLASSES}, 4), got {shapes}"
    assert (1, NUM_CLASSES) in shapes, f"Expected score shape (1, {NUM_CLASSES}), got {shapes}"


def test_openvino_export_to_directory(tmp_path):
    """Verify export_to_openvino successfully converts model to a temporary directory."""
    xml_p, bin_p = export_to_openvino(output_dir=tmp_path, verbose=False)

    assert xml_p.exists() and xml_p.is_file()
    assert bin_p.exists() and bin_p.is_file()
    assert xml_p.name == "object_detector.xml"
    assert bin_p.name == "object_detector.bin"
    assert xml_p.stat().st_size > 0
    assert bin_p.stat().st_size > 0


def test_run_benchmark_metrics():
    """Verify run_benchmark executes on CPU and returns well-formed latency and throughput."""
    metrics = run_benchmark(
        model_xml=DEFAULT_XML_PATH,
        device="CPU",
        num_iterations=5,
        warmup_iterations=2,
        verbose=False,
    )

    assert metrics["device"] == "CPU"
    assert metrics["iterations"] == 5
    assert metrics["warmup"] == 2
    assert metrics["mean_latency_ms"] > 0.0
    assert metrics["median_p50_latency_ms"] > 0.0
    assert metrics["p95_latency_ms"] > 0.0
    assert metrics["min_latency_ms"] > 0.0
    assert metrics["max_latency_ms"] >= metrics["min_latency_ms"]
    assert metrics["throughput_fps"] > 0.0


def test_benchmark_config_loader():
    """Verify load_benchmark_config loads configured benchmark settings."""
    cfg = load_benchmark_config()
    assert isinstance(cfg, dict)
    # If sim_config.yaml has benchmark section, check expected keys
    if "num_iterations" in cfg:
        assert isinstance(cfg["num_iterations"], int)
        assert cfg["num_iterations"] > 0
    if "device" in cfg:
        assert cfg["device"] in ("CPU", "GPU", "NPU")


def test_openvino_int8_quantization_and_compare(tmp_path):
    """Verify NNCF INT8 quantization and benchmark comparison engine."""
    from evaluation.benchmark import compare_fp32_int8
    from evaluation.openvino_export import DEFAULT_INT8_XML_PATH, export_to_openvino_int8

    # Ensure INT8 files exist
    assert DEFAULT_INT8_XML_PATH.exists()

    # Compare metrics computation
    res = compare_fp32_int8(device="CPU", num_iterations=5)
    assert "fp32" in res
    assert "int8" in res
    assert res["compression_ratio"] >= 1.5
    assert res["fp32"]["mean_latency_ms"] > 0.0
    assert res["int8"]["mean_latency_ms"] > 0.0
