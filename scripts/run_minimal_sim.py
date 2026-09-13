"""Minimal runnable simulation example for TwinGuard with one SO-101 arm.

Runs a physics simulation loop, commands the SO-101 joints, and validates stability.
"""

import math
import sys
from pathlib import Path

# Add project root to sys.path so simulation module can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.simulator import TwinGuardSim


def main() -> int:
    print("=" * 60)
    print(" TwinGuard: Minimal SO-101 MuJoCo Simulation")
    print("=" * 60)

    # 1. Initialize simulation
    sim = TwinGuardSim()
    print(f"[OK] MuJoCo model loaded: {sim.model_path.name}")
    print(f"     - Joints ({len(sim.joint_names)}): {', '.join(sim.joint_names)}")
    print(f"     - Actuators ({len(sim.actuator_names)}): {', '.join(sim.actuator_names)}")
    print(f"     - Timestep: {sim.get_timestep()}s")

    # 2. Reset to initial state
    sim.reset()
    print("[OK] Simulation state reset to zero configuration.")

    # 3. Run simulation loop for 500 steps (1.0 second of physics)
    total_steps = 500
    print(f"\nRunning simulation for {total_steps} steps...")

    for step in range(1, total_steps + 1):
        t = sim.get_time()

        # Generate smooth sinusoidal motion for base and shoulder
        base_target = 25.0 * math.sin(2.0 * math.pi * 0.5 * t)  # ±25 deg at 0.5 Hz
        shoulder_target = 15.0 * math.sin(2.0 * math.pi * 0.5 * t)  # ±15 deg
        elbow_target = -20.0 + 10.0 * math.cos(2.0 * math.pi * 0.5 * t)

        sim.set_joint_targets({
            "actuator_base_yaw": base_target,
            "actuator_shoulder_pitch": shoulder_target,
            "actuator_elbow_pitch": elbow_target,
        })

        sim.step(1)

        # Telemetry print every 100 steps
        if step % 100 == 0 or step == total_steps:
            qpos = sim.get_joint_positions()
            print(
                f"Step {step:3d} | Time: {t:5.3f}s | "
                f"Base Yaw: {qpos['joint_base_yaw']:+6.2f} deg | "
                f"Shoulder: {qpos['joint_shoulder_pitch']:+6.2f} deg | "
                f"Elbow: {qpos['joint_elbow_pitch']:+6.2f} deg | "
                f"Stable: {sim.is_stable()}"
            )

        # Check stability
        if not sim.is_stable():
            print(f"[ERROR] Simulation became unstable at step {step}!")
            return 1

    print("\n" + "=" * 60)
    print(" [SUCCESS] Simulation completed successfully with stable physics!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
