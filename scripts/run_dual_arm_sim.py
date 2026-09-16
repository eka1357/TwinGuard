"""Manual runnable script for TwinGuard dual-arm MuJoCo simulation.

Commands both SO-101 robotic arms independently (left_arm and right_arm)
facing a shared work table with plate, mug, and drawer objects.
Supports both headless simulation and interactive 3D visualization
via the native MuJoCo passive viewer.

Usage:
    # Launch interactive MuJoCo viewer showing coordinated bimanual manipulation
    python scripts/run_dual_arm_sim.py --view

    # Run headless physics loop for 1000 steps with telemetry
    python scripts/run_dual_arm_sim.py

    # Run sinusoidal joint range oscillation demo
    python scripts/run_dual_arm_sim.py --view --oscillate

    # Save camera snapshot
    python scripts/run_dual_arm_sim.py --save-render docs/dual_arm_scene.png
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path so simulation module can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
from simulation.simulator import TwinGuardSim
from robotics.primitives import MotionPrimitives
from safety.verifier import SafetyVerifier


def run_bimanual_task(sim: TwinGuardSim, viewer_instance=None) -> bool:
    """Execute realistic coordinated bimanual manipulation task across dual SO-101 arms."""
    primitives = MotionPrimitives(sim)
    verifier = SafetyVerifier(sim)

    print("\n[TASK] Starting coordinated bimanual manipulation sequence...")

    def check_stability(step_name: str) -> bool:
        if not sim.is_stable():
            print(f"[ERROR] Instability detected during '{step_name}'!")
            return False
        return True

    # 1. Arm A: Open drawer
    print("  -> Step 1: Left arm opening top drawer...")
    primitives.open_drawer("left_arm", "drawer")
    if not check_stability("open_drawer"):
        return False

    # 2. Arm A: Approach plate
    plate_pos = list(sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")])
    print(f"  -> Step 2: Left arm approaching plate at {plate_pos}...")
    primitives.approach("left_arm", [plate_pos[0], plate_pos[1], 0.50])
    primitives.approach("left_arm", [plate_pos[0], plate_pos[1], 0.440])
    if not check_stability("approach_plate"):
        return False

    # 3. Arm A: Grasp plate
    print("  -> Step 3: Left arm grasping plate...")
    primitives.grasp("left_arm", object_name="plate")
    if not check_stability("grasp_plate"):
        return False

    # 4. Arm A: Lift plate
    print("  -> Step 4: Left arm lifting plate...")
    primitives.lift("left_arm", height=0.07)
    if not check_stability("lift_plate"):
        return False

    # 5. Arm A: Transport plate to front-center table
    dest = [0.20, 0.08, 0.50]
    print(f"  -> Step 5: Left arm transporting plate to {dest}...")
    primitives.transport("left_arm", dest)
    if not check_stability("transport_plate"):
        return False

    # 6. Arm A: Release plate
    print("  -> Step 6: Left arm releasing plate...")
    primitives.release("left_arm")
    if not check_stability("release_plate"):
        return False

    # 7. Arm B: Approach mug
    mug_pos = list(sim.data.xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "mug")])
    print(f"  -> Step 7: Right arm approaching mug at {mug_pos}...")
    primitives.approach("right_arm", [mug_pos[0], mug_pos[1], 0.50])
    primitives.approach("right_arm", [mug_pos[0], mug_pos[1], 0.465])
    if not check_stability("approach_mug"):
        return False

    # 8. Arm B: Grasp mug
    print("  -> Step 8: Right arm grasping mug...")
    primitives.grasp("right_arm", object_name="mug")
    if not check_stability("grasp_mug"):
        return False

    # 9. Arm B: Lift mug
    print("  -> Step 9: Right arm lifting mug...")
    primitives.lift("right_arm", height=0.06)
    if not check_stability("lift_mug"):
        return False

    # 10. Arm A: Pour geometric proxy into mug
    print("  -> Step 10: Left arm tilting to pour water proxy into mug...")
    primitives.pour("left_arm", target_container=[0.22, -0.05, 0.52])
    if not check_stability("pour"):
        return False

    global_check = verifier.check_global_safety()
    print(f"\n[VERIFICATION] Global Safety: {'PASSED' if global_check.success else 'FAILED'} ({global_check.reason})")
    return bool(global_check.success)


def run_oscillation_demo(sim: TwinGuardSim, total_steps: Optional[int], viewer_instance=None) -> bool:
    """Run independent joint oscillation pattern to demo full joint ranges."""
    step_count = 0
    while total_steps is None or step_count < total_steps:
        step_count += 1
        t = sim.get_time()

        # Left arm: 0.6 Hz oscillation
        left_yaw = 25.0 * math.sin(2.0 * math.pi * 0.6 * t)
        left_shoulder = -15.0 + 10.0 * math.sin(2.0 * math.pi * 0.4 * t)
        left_elbow = 20.0 + 15.0 * math.cos(2.0 * math.pi * 0.5 * t)
        left_wrist_p = 10.0 * math.sin(2.0 * math.pi * 0.7 * t)
        left_wrist_r = 30.0 * math.cos(2.0 * math.pi * 0.5 * t)
        left_gripper = 20.0 + 20.0 * math.sin(2.0 * math.pi * 0.3 * t)

        # Right arm: Independent phase
        right_yaw = -25.0 * math.cos(2.0 * math.pi * 0.5 * t)
        right_shoulder = -15.0 - 10.0 * math.cos(2.0 * math.pi * 0.4 * t)
        right_elbow = 20.0 - 15.0 * math.sin(2.0 * math.pi * 0.6 * t)
        right_wrist_p = -10.0 * math.cos(2.0 * math.pi * 0.7 * t)
        right_wrist_r = -30.0 * math.sin(2.0 * math.pi * 0.5 * t)
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

        if viewer_instance is not None and not viewer_instance.is_running():
            print("\n[INFO] Viewer closed by user.")
            break

        if step_count % 100 == 0 or step_count == total_steps:
            left_q = sim.get_arm_joint_positions("left_arm")
            right_q = sim.get_arm_joint_positions("right_arm")
            plate = sim.get_joint_positions()["plate_joint"]
            print(
                f"Step {step_count:4d} | t={t:5.3f}s | "
                f"Left [yaw:{left_q['joint_base_yaw']:+5.1f} deg, sh:{left_q['joint_shoulder_pitch']:+5.1f} deg] | "
                f"Right [yaw:{right_q['joint_base_yaw']:+5.1f} deg, sh:{right_q['joint_shoulder_pitch']:+5.1f} deg] | "
                f"Plate z={plate[2]:.3f}m | Stable: {sim.is_stable()}"
            )

        if not sim.is_stable():
            print(f"\n[ERROR] Simulation became unstable at step {step_count}!")
            return False

    return True


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
        default=None,
        help="Number of simulation steps to execute (default: 1000 headless)",
    )
    parser.add_argument(
        "--oscillate",
        action="store_true",
        help="Run sinusoidal joint oscillation demo instead of manipulation task",
    )
    parser.add_argument(
        "--save-render",
        type=str,
        default=None,
        help="Optional path to save an overview camera image (e.g. recordings/scene.png)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" TwinGuard: Bimanual Dual-Arm SO-101 MuJoCo Simulation")
    print("=" * 65)

    sim = TwinGuardSim()
    print(f"[OK] Model loaded: {sim.model_path.name}")
    print(f"     - Configured arms: {', '.join(sim.arm_names)}")
    print(f"     - Total joints: {len(sim.joint_names)} ({sim.model.nq} qpos, {sim.model.nv} DOFs)")
    print(f"     - Total actuators: {len(sim.actuator_names)}")
    print(f"     - Scene cameras: {', '.join(sim.get_camera_names())}")
    print(f"     - Physics timestep: {sim.get_timestep()}s")

    sim.reset()
    print("[OK] Simulation state reset to zero configuration.")

    plate_init = sim.get_joint_positions()["plate_joint"]
    print(f"[OK] Plate initial pos on table: x={plate_init[0]:.3f}, y={plate_init[1]:.3f}, z={plate_init[2]:.3f}")

    viewer_ctx = None
    if args.view:
        try:
            import mujoco.viewer
            viewer_ctx = mujoco.viewer.launch_passive(sim.model, sim.data)
            sim.attach_viewer(viewer_ctx, realtime=True)
            print("[OK] Interactive MuJoCo viewer active. Watch the dual-arm manipulation live.")
        except Exception as e:
            print(f"[WARN] Failed to launch interactive viewer: {e}")

    try:
        if args.oscillate:
            print("\nExecuting joint oscillation demo...")
            total_steps = args.steps if args.steps is not None else (None if args.view else 1000)
            success = run_oscillation_demo(sim, total_steps, viewer_instance=viewer_ctx)
        elif args.steps is not None:
            print(f"\nExecuting {args.steps} physics steps...")
            success = run_oscillation_demo(sim, args.steps, viewer_instance=viewer_ctx)
        else:
            success = run_bimanual_task(sim, viewer_instance=viewer_ctx)
    finally:
        if viewer_ctx is not None:
            sim.detach_viewer()
            viewer_ctx.close()

    if not success:
        return 1

    # Optional camera snapshot save
    if args.save_render:
        try:
            from PIL import Image
            out_p = Path(args.save_render)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            rgb = sim.get_camera_image("overview_cam", width=1280, height=720)
            img = Image.fromarray(rgb)
            img.save(str(out_p))
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
