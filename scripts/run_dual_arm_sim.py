"""Manual runnable script for TwinGuard dual-arm MuJoCo simulation.

Commands both SO-101 robotic arms independently (left_arm and right_arm)
facing a shared work table with a dynamic plate placeholder object.
Supports both headless simulation with real-time telemetry and
interactive 3D visualization via the native MuJoCo passive viewer.

Usage:
    # Run headless physics loop for 1000 steps with telemetry
    python scripts/run_dual_arm_sim.py

    # Launch interactive MuJoCo viewer showing independent arm motion
    python scripts/run_dual_arm_sim.py --view

    # Run for custom number of steps
    python scripts/run_dual_arm_sim.py --steps 500
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

import mujoco
from simulation.simulator import TwinGuardSim


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TwinGuard: Dual-Arm SO-101 MuJoCo Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Launch the interactive MuJoCo passive viewer window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps to execute (timestep = 0.002s)",
    )
    parser.add_argument(
        "--save-render",
        type=str,
        default=None,
        help="Optional path to save an overview camera image (e.g. docs/dual_arm_scene.png)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" TwinGuard: Bimanual Dual-Arm SO-101 MuJoCo Simulation")
    print("=" * 65)

    # 1. Initialize simulation from sim_config.yaml
    sim = TwinGuardSim()
    print(f"[OK] Model loaded: {sim.model_path.name}")
    print(f"     - Configured arms: {', '.join(sim.arm_names)}")
    print(f"     - Total joints: {len(sim.joint_names)} ({sim.model.nq} qpos, {sim.model.nv} DOFs)")
    print(f"     - Total actuators: {len(sim.actuator_names)}")
    print(f"     - Scene cameras: {', '.join(sim.get_camera_names())}")
    print(f"     - Physics timestep: {sim.get_timestep()}s")

    # 2. Reset simulation to initial resting state
    sim.reset()
    print("[OK] Simulation state reset to zero configuration.")

    plate_init = sim.get_joint_positions()["plate_joint"]
    print(f"[OK] Plate initial pos on table: x={plate_init[0]:.3f}, y={plate_init[1]:.3f}, z={plate_init[2]:.3f}")

    total_steps = args.steps
    print(f"\nExecuting simulation for {total_steps} steps ({total_steps * sim.get_timestep():.2f}s physics)...")
    if args.view:
        print("[INFO] Launching MuJoCo interactive viewer. Close window to stop.")

    # 3. Execution loop (with or without viewer)
    step_count = 0

    def step_simulation(viewer_instance=None):
        nonlocal step_count
        while step_count < total_steps:
            step_count += 1
            t = sim.get_time()

            # Independent motion trajectories:
            # Left arm: 0.6 Hz oscillation (yaw ±25°, shoulder pitch ±15°, elbow ±20°)
            left_yaw = 25.0 * math.sin(2.0 * math.pi * 0.6 * t)
            left_shoulder = -15.0 + 10.0 * math.sin(2.0 * math.pi * 0.4 * t)
            left_elbow = 20.0 + 15.0 * math.cos(2.0 * math.pi * 0.5 * t)
            left_wrist_p = 10.0 * math.sin(2.0 * math.pi * 0.7 * t)
            left_wrist_r = 30.0 * math.cos(2.0 * math.pi * 0.5 * t)
            # Left gripper alternates open/close at 0.3 Hz
            left_gripper = 20.0 + 20.0 * math.sin(2.0 * math.pi * 0.3 * t)

            # Right arm: Independent phase and frequencies
            right_yaw = -25.0 * math.cos(2.0 * math.pi * 0.5 * t)
            right_shoulder = -15.0 - 10.0 * math.cos(2.0 * math.pi * 0.4 * t)
            right_elbow = 20.0 - 15.0 * math.sin(2.0 * math.pi * 0.6 * t)
            right_wrist_p = -10.0 * math.cos(2.0 * math.pi * 0.7 * t)
            right_wrist_r = -30.0 * math.sin(2.0 * math.pi * 0.5 * t)
            # Right gripper opposite phase
            right_gripper = 20.0 - 20.0 * math.sin(2.0 * math.pi * 0.3 * t)

            sim.set_arm_joint_targets("left_arm", {
                "joint_base_yaw": left_yaw,
                "joint_shoulder_pitch": left_shoulder,
                "joint_elbow_pitch": left_elbow,
                "joint_wrist_pitch": left_wrist_p,
                "joint_wrist_roll": left_wrist_r,
                "joint_gripper": left_gripper,
            })

            sim.set_arm_joint_targets("right_arm", {
                "joint_base_yaw": right_yaw,
                "joint_shoulder_pitch": right_shoulder,
                "joint_elbow_pitch": right_elbow,
                "joint_wrist_pitch": right_wrist_p,
                "joint_wrist_roll": right_wrist_r,
                "joint_gripper": right_gripper,
            })

            sim.step(1)

            if viewer_instance is not None:
                viewer_instance.sync()
                # Sleep to approximate wall-clock real time
                time.sleep(sim.get_timestep())
                if not viewer_instance.is_running():
                    print("\n[INFO] Viewer closed by user.")
                    break

            # Print telemetry every 100 steps
            if step_count % 100 == 0 or step_count == total_steps:
                left_q = sim.get_arm_joint_positions("left_arm")
                right_q = sim.get_arm_joint_positions("right_arm")
                plate = sim.get_joint_positions()["plate_joint"]
                print(
                    f"Step {step_count:4d} | t={t:5.3f}s | "
                    f"Left [yaw:{left_q['joint_base_yaw']:+5.1f} deg, sh:{left_q['joint_shoulder_pitch']:+5.1f} deg, grip:{left_q['joint_gripper']:4.1f} deg] | "
                    f"Right [yaw:{right_q['joint_base_yaw']:+5.1f} deg, sh:{right_q['joint_shoulder_pitch']:+5.1f} deg, grip:{right_q['joint_gripper']:4.1f} deg] | "
                    f"Plate z={plate[2]:.3f}m | Stable: {sim.is_stable()}"
                )

            if not sim.is_stable():
                print(f"\n[ERROR] Simulation became unstable at step {step_count}!")
                return False

        return True

    if args.view:
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                success = step_simulation(viewer)
        except Exception as e:
            print(f"[WARN] Failed to launch interactive viewer: {e}")
            print("[INFO] Falling back to headless simulation execution.")
            success = step_simulation(None)
    else:
        success = step_simulation(None)

    if not success:
        return 1

    # Optional camera snapshot save
    if args.save_render:
        try:
            import zlib, struct, binascii
            out_p = Path(args.save_render)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            rgb = sim.get_camera_image("overview_cam")
            h, w, _ = rgb.shape
            raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))
            compress = zlib.compress(raw)
            def chunk(tag: bytes, data: bytes) -> bytes:
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
            png = (
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", compress)
                + chunk(b"IEND", b"")
            )
            out_p.write_bytes(png)
            print(f"[OK] Overview camera snapshot saved to: {out_p}")
        except Exception as e:
            print(f"[WARN] Could not save camera render: {e}")

    print("\n" + "=" * 65)
    print(" [SUCCESS] Dual-arm simulation completed with stable physics!")
    print("=" * 65)
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
