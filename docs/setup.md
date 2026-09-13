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
# Run all simulation tests
.venv\Scripts\pytest tests/test_simulation.py -v
```

Expected output:
```
tests/test_simulation.py::test_model_loads PASSED
tests/test_simulation.py::test_joint_names PASSED
tests/test_simulation.py::test_actuator_names PASSED
tests/test_simulation.py::test_physics_step_stability PASSED
tests/test_simulation.py::test_actuator_targets_apply PASSED
tests/test_simulation.py::test_simulation_reset PASSED

================ 6 passed in 0.86s ================
```

---

## 4. Run Minimal Simulation Example

Execute the minimal single-arm SO-101 simulation script:

```powershell
.venv\Scripts\python scripts/run_minimal_sim.py
```

This script:
1. Loads the SO-101 arm in the table scene (`simulation/models/scene.xml`).
2. Resets the simulation to the initial rest position.
3. Steps through 500 physics timesteps with position control commands.
4. Logs real-time joint positions and verifies numerical stability.

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
│   │   └── scene.xml         # Workspace, table, and lighting setup
│   └── simulator.py          # TwinGuardSim Python API wrapper
├── robotics/                 # Robot hardware, kinematics & low-level control
├── perception/               # Visual observation interfaces
├── planning/                 # Task and trajectory planning
├── safety/                   # Safety limits and collision monitoring
├── evaluation/               # Benchmarking and metrics
├── scripts/                  # Executable entry points
│   └── run_minimal_sim.py    # M1 runnable simulation script
├── configs/                  # Environment and model parameters
│   └── sim_config.yaml       # Simulation configuration
├── docs/                     # Project documentation
│   └── setup.md              # Setup and execution guide
└── tests/                    # Pytest test suite
    └── test_simulation.py    # Simulation verification tests
```
