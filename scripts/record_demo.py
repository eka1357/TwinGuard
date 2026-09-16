"""Demo recording script for TwinGuard bimanual manipulation.

Executes designated seeds (e.g. Seed 2 and Seed 7 with controlled dynamic failure
recovery) and captures rendered camera frames at each step for assembling demo videos
or presentations.

Usage:
    python scripts/record_demo.py --seeds 2 7 --save-render --camera overview_cam
    python scripts/record_demo.py --seeds 2 --save-render --make-gif
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.simulator import TwinGuardSim
from evaluation.runner import (
    ControlledFailureInterceptor,
    create_perception_informed_planner,
    randomize_scene_for_seed,
)
from evaluation.executor import (
    WORKED_EXAMPLE_INSTRUCTION,
    execute_primitive_step,
    resolve_step_target,
)
from planning.planner import PlanStep, plan
from safety.verifier import SafetyVerifier

# Configure logger
logger = logging.getLogger("twinguard.record_demo")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "recordings" / "demo_frames"


def save_frame(
    sim: TwinGuardSim,
    camera_name: str,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Render and save an RGB camera frame as a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_array = sim.get_camera_image(camera_name=camera_name, width=width, height=height)
    img = Image.fromarray(img_array)
    img.save(str(output_path), format="PNG")
    return output_path


def create_animated_gif(image_paths: Sequence[Path], output_gif: Path, duration_ms: int = 600) -> Path:
    """Stitch rendered sequence of frames into an animated GIF."""
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p) for p in image_paths if p.exists()]
    if images:
        images[0].save(
            str(output_gif),
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
    return output_gif


def record_seed_demo(
    seed: int,
    output_dir: Path,
    camera_name: str = "overview_cam",
    save_render: bool = True,
    make_gif: bool = False,
    force_failure: bool = True,
    instruction: str = WORKED_EXAMPLE_INSTRUCTION,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute a single seed and capture rendered camera frames at key steps.

    Args:
        seed: Randomization seed (e.g. 2 or 7).
        output_dir: Destination folder for captured frames.
        camera_name: Scene camera ('overview_cam', 'front_cam', etc.).
        save_render: Whether to capture and save camera frames.
        make_gif: Whether to compile frames into an animated GIF.
        force_failure: Whether to inject controlled failure on this seed.
        instruction: Natural language task instruction.
        verbose: Whether to log execution details.

    Returns:
        Dict containing execution summary and list of saved image paths.
    """
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    sim = TwinGuardSim()
    sim.reset()

    # Deterministic scene randomization
    poses = randomize_scene_for_seed(sim, seed=seed)

    captured_frames: List[Path] = []

    if save_render:
        frame_p = seed_dir / "frame_00_initial_scene.png"
        save_frame(sim, camera_name, frame_p)
        captured_frames.append(frame_p)
        if verbose:
            logger.info(f"[RENDER] Saved initial state -> {frame_p.name}")

    planner_fn = create_perception_informed_planner(sim)
    verifier = SafetyVerifier(sim)
    from robotics.primitives import MotionPrimitives
    primitives = MotionPrimitives(sim)

    # Initial task plan
    raw_plan = planner_fn(f"TASK INSTRUCTION: {instruction}")
    steps_data = json.loads(raw_plan)
    initial_steps = [PlanStep(**s) for s in steps_data]

    target_arm = "left_arm" if seed % 2 == 0 else "right_arm"
    target_obj = "plate" if target_arm == "left_arm" else "mug"

    frame_counter = 1

    def execute_and_capture(step_obj: PlanStep, step_label: str) -> bool:
        nonlocal frame_counter
        target = resolve_step_target(sim, step_obj)
        ok = execute_primitive_step(primitives, sim, step_obj)
        ver = verifier.verify_step(step_obj, primitive_returned=ok, target=target)

        if save_render:
            action_tag = f"{step_obj.action}_{step_obj.arm}"
            fn = f"frame_{frame_counter:02d}_{step_label}_{action_tag}.png"
            fp = seed_dir / fn
            save_frame(sim, camera_name, fp)
            captured_frames.append(fp)
            if verbose:
                logger.info(f"[RENDER] Captured frame -> {fn}")
            frame_counter += 1

        return ver["success"]

    # Execute with failure interceptor if requested
    interceptor_ctx = (
        ControlledFailureInterceptor(target_seed=seed, target_arm=target_arm, target_object=target_obj)
        if force_failure
        else None
    )

    if verbose:
        print("\n" + "=" * 65)
        print(f" RECORDING DEMO FOR SEED {seed}")
        print(f" Camera:             {camera_name}")
        print(f" Controlled Failure: {'YES (' + target_arm + ' ' + target_obj + ')' if force_failure else 'None'}")
        print(f" Output Directory:   {seed_dir}")
        print("=" * 65 + "\n")

    if interceptor_ctx:
        interceptor_ctx.__enter__()

    execution_successful = True
    for idx, step in enumerate(initial_steps, 1):
        step_ok = execute_and_capture(step, f"step_{idx:02d}")

        if not step_ok:
            logger.warning(f"Step {idx} [{step.action}] failed verification! Initiating dynamic recovery...")
            # Re-observe scene and replan recovery
            recovery_prompt = (
                f"TASK INSTRUCTION: Recovery Attempt 1: Step '{step.action}' on '{step.object}' failed because: "
                f"Grasp failed: object '{step.object}' freejoint is not within gripper grasp zone. "
                f"Perform a recovery action to continue achieving: {instruction}"
            )
            raw_rec = planner_fn(recovery_prompt)
            rec_steps = [PlanStep(**s) for s in json.loads(raw_rec)]

            rec_success = True
            for r_idx, r_step in enumerate(rec_steps, 1):
                logger.info(f"  Executing recovery action {r_idx}: {r_step.action} ({r_step.arm})")
                r_ok = execute_and_capture(r_step, f"recovery_{idx:02d}")
                if not r_ok:
                    rec_success = False
                    break

            if rec_success:
                logger.info(f"[RECOVERY SUCCESS] Step {idx} [{step.action}] recovered successfully!")
            else:
                logger.error(f"[RECOVERY FAILED] Step {idx} [{step.action}] could not be recovered.")
                execution_successful = False
                break

    if interceptor_ctx:
        interceptor_ctx.__exit__(None, None, None)

    gif_path = None
    if make_gif and captured_frames:
        gif_path = seed_dir / f"demo_seed_{seed}.gif"
        create_animated_gif(captured_frames, gif_path, duration_ms=600)
        if verbose:
            print(f"[GIF] Generated animated demonstration: {gif_path}")

    sim.close()

    summary = {
        "seed": seed,
        "success": execution_successful,
        "camera": camera_name,
        "captured_frames_count": len(captured_frames),
        "frame_paths": [str(p) for p in captured_frames],
        "gif_path": str(gif_path) if gif_path else None,
    }

    # Save demo metadata
    with open(seed_dir / "demo_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """CLI entrypoint for demo recording."""
    parser = argparse.ArgumentParser(description="TwinGuard Demo Frame Recorder")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2, 7],
        help="Seeds to execute and record (default: 2 and 7 with dynamic recovery)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="overview_cam",
        help="Scene camera: 'overview_cam', 'front_cam', etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save rendered frames",
    )
    parser.add_argument(
        "--save-render",
        action="store_true",
        default=True,
        help="Capture and save camera frames at each step (default: True)",
    )
    parser.add_argument(
        "--make-gif",
        action="store_true",
        default=True,
        help="Assemble captured frames into an animated GIF",
    )
    parser.add_argument(
        "--no-forced-failure",
        action="store_true",
        help="Disable controlled failure injection",
    )
    args = parser.parse_args()

    out_d = Path(args.output_dir)
    force_fail = not args.no_forced_failure

    print("=" * 70)
    print(" TWINGUARD DEMO VIDEO RECORDER")
    print("=" * 70)
    print(f" Seeds:               {args.seeds}")
    print(f" Camera:             {args.camera}")
    print(f" Save Frames:        {args.save_render}")
    print(f" Compile GIF:        {args.make_gif}")
    print(f" Controlled Failure: {force_fail}")
    print(f" Output Folder:      {out_d}")
    print("=" * 70)

    for seed in args.seeds:
        record_seed_demo(
            seed=seed,
            output_dir=out_d,
            camera_name=args.camera,
            save_render=args.save_render,
            make_gif=args.make_gif,
            force_failure=force_fail,
            verbose=True,
        )


if __name__ == "__main__":
    main()
