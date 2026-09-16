# TwinGuard

TwinGuard is a reliable bimanual Physical AI system utilizing two simulated SO-101 robotic arms in MuJoCo, with multi-step manipulation, failure detection, recovery mechanisms, and hardware optimization.

---

## Current Milestone: M1 — Python & MuJoCo Environment Verification

The current scope strictly focuses on:
- Verified Python environment with MuJoCo 3.x.
- Clean, modular repository architecture.
- Minimal MJCF model for the 6-DOF SO-101 robotic arm.
- Automated tests and a minimal runnable physics simulation loop.

---

## Directory Structure

```
TwinGuard/
├── AGENTS.md                 # Development rules and milestone boundaries
├── README.md                 # Project overview and quickstart
├── requirements.txt          # Python dependencies (mujoco, numpy, pyyaml, pytest)
├── simulation/               # MuJoCo simulation environment & models
│   ├── models/
│   │   ├── so101.xml         # SO-101 6-DOF robotic arm MJCF
│   │   └── scene.xml         # Work table, lighting, cameras, ground plane
│   └── simulator.py          # TwinGuardSim physics wrapper class
├── robotics/                 # Robot hardware, kinematics & low-level control
├── perception/               # Visual observation interfaces
├── planning/                 # Task and trajectory planning
├── safety/                   # Safety limits and collision monitoring
├── evaluation/               # Benchmarking and metrics
├── scripts/                  # Runnable scripts
│   └── run_minimal_sim.py    # Smallest runnable simulation example
├── configs/                  # Environment and model parameters
│   └── sim_config.yaml       # Simulation configuration
├── docs/                     # Project documentation
│   └── setup.md              # Installation and run instructions
└── tests/                    # Pytest verification suite
    └── test_simulation.py    # Automated test cases
```

---

## Quickstart

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Tests

```powershell
# Run the complete test suite (dual-arm scene, motion primitives, single-arm)
.venv\Scripts\pytest tests/ -v
```

### 3. Run Simulations with Interactive 3D Visualizer

By default, scripts run headlessly in the console for fast physics benchmarks. Pass `--view` to open the native MuJoCo interactive 3D viewer window:

```powershell
# Interactive 3D viewer: Dual-arm bimanual scene (table, plate, mug, drawer)
.venv\Scripts\python scripts/run_dual_arm_sim.py --view

# Interactive 3D viewer: Minimal single-arm scene
.venv\Scripts\python scripts/run_minimal_sim.py --view

# Headless mode (fast console loop):
.venv\Scripts\python scripts/run_minimal_sim.py
```

For complete setup and troubleshooting details, see [docs/setup.md](docs/setup.md).
