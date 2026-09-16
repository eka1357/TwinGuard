"""Minimal runnable simulation example for TwinGuard with one SO-101 arm.

Runs a physics simulation loop, commands the SO-101 joints, and validates stability.
"""

import argparse
import math
import sys
import time
from pathlib import Path

# Add project root to sys.path so simulation module can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.simulator import TwinGuardSim


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TwinGuard: Minimal SO-101 MuJoCo Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Launch the interactive MuJoCo 3D viewer window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps to run (default: continuous if --view, 500 headless)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" TwinGuard: Minimal SO-101 MuJoCo Simulation")
    print("=" * 60)

    # 1. Initialize simulation with single-arm scene
    sim = TwinGuardSim(model_path="simulation/models/scene_single.xml")
    print(f"[OK] MuJoCo model loaded: {sim.model_path.name}")
    print(f"     - Joints ({len(sim.joint_names)}): {', '.join(sim.joint_names)}")
    print(f"     - Actuators ({len(sim.actuator_names)}): {', '.join(sim.actuator_names)}")
    print(f"     - Timestep: {sim.get_timestep()}s")

    # 2. Reset to initial state
    sim.reset()
    print("[OK] Simulation state reset to zero configuration.")

    # 3. Run simulation loop
    total_steps = args.steps if args.steps is not None else (None if args.view else 500)
    if total_steps is not None:
        print(f"\nRunning simulation for {total_steps} steps...")
    else:
        print("\nRunning simulation continuously. Close the 3D viewer window to stop...")

    if args.view:
        print("[INFO] Launching MuJoCo 3D viewer window. Close the window to stop.")

    def run_loop(viewer=None) -> bool:
        step = 0
        while total_steps is None or step < total_steps:
            step += 1
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

            if viewer is not None:
                viewer.sync()
                time.sleep(sim.get_timestep())
                if not viewer.is_running():
                    print("\n[INFO] Viewer closed by user.")
                    break

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
                return False

        return True

    if args.view:
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                success = run_loop(viewer)
        except Exception as e:
            print(f"[WARN] Failed to launch interactive viewer: {e}")
            print("[INFO] Falling back to headless execution.")
            success = run_loop(None)
    else:
        success = run_loop(None)

    if not success:
        return 1

    print("\n" + "=" * 60)
    print(" [SUCCESS] Simulation completed successfully with stable physics!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
