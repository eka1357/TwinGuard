# TwinGuard Environment Setup & Quickstart

This document details how to configure the development environment and run tests for Milestone 1 (M1) of the **TwinGuard** project.

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13 (Python 3.13.5 verified on Windows)
- Git

---

## 2. Virtual Environment Setup

From the repository root directory (`TwinGuard`):

### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### Linux / macOS (bash)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 3. Run Automated Tests

TwinGuard uses `pytest` for unit and regression testing:

```powershell
# Run the complete test suite (51 tests)
.venv\Scripts\pytest tests/ -v

# Or run specific test modules:
.venv\Scripts\pytest tests/test_primitives.py -v
.venv\Scripts\pytest tests/test_planner.py -v
.venv\Scripts\pytest tests/test_executor.py -v
.venv\Scripts\pytest tests/test_runner.py -v
.venv\Scripts\pytest tests/test_benchmark.py -v
```

Expected output:
```
======================= 51 passed in 26.93s =======================
```

---

## 4. Run Execution Pipelines & Demos

### End-to-End Pipeline Executor
```powershell
.venv\Scripts\python evaluation/executor.py
```

### 10-Seed Robustness Evaluation Runner
```powershell
.venv\Scripts\python evaluation/runner.py
.venv\Scripts\python evaluation/report.py
```

### OpenVINO Hardware Benchmark
```powershell
.venv\Scripts\python evaluation/openvino_export.py
.venv\Scripts\python evaluation/benchmark.py --device CPU
```

### Demo Video & Frame Capture
```powershell
.venv\Scripts\python scripts/record_demo.py --seeds 2 7 --save-render --make-gif
```

### Simulation Visualizer (Headless or Interactive 3D Viewer)
```powershell
# Run headless physics loop
.venv\Scripts\python scripts/run_dual_arm_sim.py

# Launch interactive 3D MuJoCo viewer window
.venv\Scripts\python scripts/run_dual_arm_sim.py --view
```

---

## 5. Directory Structure Overview

```
TwinGuard/
├── AGENTS.md                 # Project rules and milestone scopes
├── NOTICE.md                 # Open-source third-party credits and licenses
├── README.md                 # Comprehensive project deliverable and architecture summary
├── requirements.txt          # Python dependencies
├── simulation/               # MuJoCo simulation environment & models
│   ├── models/
│   │   ├── scene.xml         # Dual-arm bimanual scene (table, plate, mug, drawer)
│   │   ├── left_arm.xml      # Namespaced left_arm SO-101 definition
│   │   ├── right_arm.xml     # Namespaced right_arm SO-101 definition
│   │   └── scene_single.xml  # Preserved single-arm scene for M1 regression
│   └── simulator.py          # TwinGuardSim Python API wrapper
├── robotics/                 # Bimanual motion primitives (approach, grasp, lift, etc.)
├── perception/               # Ground-truth scene state & neural object detector
├── planning/                 # Task planner, Pydantic validation & dynamic replanner
├── safety/                   # Safety verifier, drop detector, collision checker
├── evaluation/               # End-to-end executor, 10-seed runner, report, OpenVINO benchmark
├── checkpoints/              # Model weights and OpenVINO IR (.xml + .bin)
├── scripts/                  # Runnable scripts (record_demo, run_dual_arm_sim, run_minimal_sim)
├── configs/                  # Central configuration (sim_config.yaml)
├── docs/                     # Project documentation (CHALLENGE.md, setup.md)
└── tests/                    # Pytest verification suite (51 tests)
```

