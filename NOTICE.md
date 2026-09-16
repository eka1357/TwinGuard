# Notice and Third-Party Acknowledgements

TwinGuard incorporates open-source models, specifications, and software components. We gratefully acknowledge the creators and maintainers of the following projects:

---

## 1. Robotic Arm Model & MJCF Assets
- **Project**: [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **Author**: TheRobotStudio (Alexander Koch and contributors)
- **License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Description**: The kinematic structure, joint limits, and link geometries for the SO-101 6-DOF robotic arms used in TwinGuard (`simulation/models/left_arm.xml`, `simulation/models/right_arm.xml`, `simulation/models/so101.xml`) are adapted from the SO-ARM100 open-source hardware and simulation project, instantiated in dual-arm bimanual configuration.

---

## 2. Action and Observation Conventions
- **Project**: [LeRobot](https://github.com/huggingface/lerobot) & [SmolVLA](https://github.com/huggingface/lerobot)
- **Author**: Hugging Face and contributors
- **License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Description**: TwinGuard follows the LeRobot `bi_so101_follower` bimanual action and joint telemetry conventions (12-dimensional joint control: 6 joints × 2 arms) and the SmolVLA multimodal action chunking / VLA interface conventions.

---

## 3. Physics Simulation Engine
- **Project**: [MuJoCo (Multi-Joint dynamics with Contact)](https://github.com/google-deepmind/mujoco)
- **Author**: Google DeepMind
- **License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Description**: All physical contact dynamics, constraint solving, forward kinematics, and off-screen rendering are simulated with MuJoCo 3.x.

---

## 4. Hardware Optimization & Neural Runtime
- **Project**: [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino)
- **Author**: Intel Corporation
- **License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Description**: Neural object detector models are converted to OpenVINO Intermediate Representation (IR) and benchmarked for low-latency inference on Intel Core Ultra architectures (CPU, iGPU, NPU).

---

## 5. Machine Learning Framework
- **Project**: [PyTorch](https://github.com/pytorch/pytorch)
- **Author**: PyTorch Contributors
- **License**: [BSD 3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- **Description**: Used for the convolutional neural object detector training and export pipeline.
