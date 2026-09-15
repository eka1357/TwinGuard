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
# Run dual-arm bimanual simulation tests
.venv\Scripts\pytest tests/test_dual_arm_scene.py -v

# Run single-arm regression tests
.venv\Scripts\pytest tests/test_simulation.py -v

# Run all tests
.venv\Scripts\pytest -v
```

Expected output:
```
============================= 14 passed in 2.70s ==============================
```

---

## 4. Run Simulation Examples

### Dual-Arm Bimanual Simulation (Headless or Interactive 3D Viewer)

Execute the dual-arm SO-101 simulation script:

```powershell
# Run headless physics loop with real-time telemetry (1000 steps)
.venv\Scripts\python scripts/run_dual_arm_sim.py

# Launch interactive 3D MuJoCo viewer window
.venv\Scripts\python scripts/run_dual_arm_sim.py --view
```

### Minimal Single-Arm Simulation

```powershell
.venv\Scripts\python scripts/run_minimal_sim.py
```

---

## 5. Directory Structure Overview

```
TwinGuard/
├── AGENTS.md                 # Project rules and milestone scopes
├── README.md                 # Project overview and quickstart
├── requirements.txt          # Python dependencies
├── simulation/               # MuJoCo simulation environment & models
│   ├── models/
│   │   ├── so101.xml         # SO-101 6-DOF robotic arm definition
│   │   ├── left_arm.xml      # Namespaced left_arm SO-101 definition
│   │   ├── right_arm.xml     # Namespaced right_arm SO-101 definition
│   │   ├── scene.xml         # Dual-arm workspace, table, plate & cameras
│   │   └── scene_single.xml  # Preserved single-arm scene for M1 regression
│   └── simulator.py          # TwinGuardSim Python API wrapper (dual-arm & single-arm)
├── robotics/                 # Robot hardware, kinematics & low-level control
├── perception/               # Visual observation interfaces
├── planning/                 # Task and trajectory planning
├── safety/                   # Safety limits and collision monitoring
├── evaluation/               # Benchmarking and metrics
├── scripts/                  # Executable entry points
│   ├── run_dual_arm_sim.py   # Dual-arm runner with optional interactive viewer
│   └── run_minimal_sim.py    # Single-arm simulation script
├── configs/                  # Environment and model parameters
│   └── sim_config.yaml       # Simulation & dual-arm configuration
├── docs/                     # Project documentation
│   └── setup.md              # Setup and execution guide
└── tests/                    # Pytest test suite
    ├── test_dual_arm_scene.py# Dual-arm scene & independent motion tests
    └── test_simulation.py    # Single-arm regression tests
```
