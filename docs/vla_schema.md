# TwinGuard VLA Adapter Schema

## Status

This document proposes an adapter between the TwinGuard dual-arm MuJoCo simulator and a future LeRobot/SmolVLA policy. It is an interface proposal for project planning, not the confirmed schema of any specific SmolVLA checkpoint.

No LeRobot or SmolVLA dependency is required by this proposal.

## Current Simulator Observation API

The simulator entry point is `TwinGuardSim` from `simulation.simulator`.

The configured arm names are:

- `left_arm`
- `right_arm`

Each arm has six canonical joints:

1. `joint_base_yaw`
2. `joint_shoulder_pitch`
3. `joint_elbow_pitch`
4. `joint_wrist_pitch`
5. `joint_wrist_roll`
6. `joint_gripper`

Joint positions can be read with:

```python
positions = sim.get_arm_joint_positions("left_arm")
```

The returned mapping includes the arm's joint positions. Joint positions and joint target inputs use degrees by default.

Images can be read with:

```python
image = sim.get_camera_image("overview_cam")
```

Configured camera names include:

- `overview_cam` - 1280 x 720
- `front_cam` - 640 x 480
- `left_arm_wrist_cam` - 640 x 480
- `right_arm_wrist_cam` - 640 x 480

The camera API returns an RGB NumPy image. The expected array layout is height x width x 3, with `uint8` pixel values.

A parser instruction is available as a language string and can be produced from a user instruction before being passed to a policy adapter.

## Proposed Policy Observation Fields

The adapter could expose one observation dictionary with fields similar to the following:

| Field | Source | Proposed content |
| --- | --- | --- |
| `language_instruction` | Parser input/output | Natural-language instruction string |
| `observation.images.overview` | `get_camera_image("overview_cam")` | RGB image, H x W x 3 |
| `observation.images.front` | `get_camera_image("front_cam")` | RGB image, H x W x 3 |
| `observation.images.left_wrist` | `get_camera_image("left_arm_wrist_cam")` | RGB image, H x W x 3 |
| `observation.images.right_wrist` | `get_camera_image("right_arm_wrist_cam")` | RGB image, H x W x 3 |
| `observation.state.left_arm` | `get_arm_joint_positions("left_arm")` | Six joint positions in canonical order, degrees |
| `observation.state.right_arm` | `get_arm_joint_positions("right_arm")` | Six joint positions in canonical order, degrees |

The image field names above are proposed adapter names. They are not claims about the feature names expected by a particular LeRobot dataset or checkpoint.

A concrete adapter should normalize each joint mapping into this fixed order:

```text
[joint_base_yaw,
 joint_shoulder_pitch,
 joint_elbow_pitch,
 joint_wrist_pitch,
 joint_wrist_roll,
 joint_gripper]
```

## Proposed Action Vector

The proposed bimanual action is a 12-value vector in degrees:

```text
[
    left joint_base_yaw,
    left joint_shoulder_pitch,
    left joint_elbow_pitch,
    left joint_wrist_pitch,
    left joint_wrist_roll,
    left joint_gripper,
    right joint_base_yaw,
    right joint_shoulder_pitch,
    right joint_elbow_pitch,
    right joint_wrist_pitch,
    right joint_wrist_roll,
    right joint_gripper,
]
```

The first six values are applied to `left_arm`; the final six values are applied to `right_arm`. The adapter would split the vector into two six-value mappings using the canonical joint order and pass the mappings to the simulator's arm-target API.

This is a proposed project action convention. It is not yet a confirmed checkpoint action format.

## Units and Shapes

| Data | Proposed/current representation |
| --- | --- |
| Joint observations | Six values per arm, shape `(6,)`, degrees |
| Joint action | Twelve values, shape `(12,)`, degrees |
| Camera image | RGB array, shape `(height, width, 3)`, `uint8` |
| Language instruction | String |
| Arm names | String identifiers: `left_arm` and `right_arm` |

The simulator internally converts degree-valued position targets as needed for MuJoCo control. An adapter must not silently mix radians and degrees.

Image batching, channel order, resizing, cropping, and normalization remain adapter decisions until the target policy and checkpoint are confirmed.

## Parser Arm Label Conversion

The language parser uses labels `A` and `B`. The simulator uses explicit arm names:

| Parser label | Simulator name |
| --- | --- |
| `A` | `left_arm` |
| `B` | `right_arm` |

The adapter should perform this conversion at the boundary before reading observations or sending actions. It should reject unknown labels instead of guessing an arm.

## Important Scope Limitations

This schema is only an adapter proposal. It is not the confirmed SmolVLA checkpoint schema, and the proposed field names, action layout, image handling, and normalization may need to change.

Current pick, place, and handoff behavior is not yet fully implemented. A policy adapter must not imply that a parsed instruction or predicted action already provides reliable manipulation, grasping, recovery, or handoff behavior.

## Unknowns Before Training

The following details must be confirmed before collecting training data or connecting a policy checkpoint:

- Exact LeRobot version.
- Exact policy and checkpoint.
- Exact feature key names.
- Image layout and resolution expected by the policy.
- Action chunk format.
- Observation and action normalization.
- Whether the checkpoint supports bimanual 12-dimensional actions.

These unknowns should be resolved from the selected LeRobot version, dataset schema, policy implementation, and checkpoint configuration rather than inferred from this project proposal.
