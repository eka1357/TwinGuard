# TwinGuard — 5-Minute Video Presentation & Slide Deck Script

> **Challenge**: Intel Physical AI Online Challenge — *Bimanual VLA Manipulation with Multi-Modal Reasoning*
> **Platform**: lablab.ai × AI Infra Summit 2026
> **Project**: TwinGuard — Autonomous Bimanual Physical AI with Closed-Loop Recovery & OpenVINO Inference

---

## ⏱ Timing Overview (Total: 5:00)

| Time | Segment | Visual | Speaker Talking Points |
|:---|:---|:---|:---|
| **0:00 – 0:30** | Hook & Problem | Slide 1 (Cover Banner) | High-level bimanual manipulation challenge, why open-loop fails, why Physical AI needs closed-loop verifiers. |
| **0:30 – 1:30** | System Architecture | Slide 2 (Architecture Diagram) | Perception (Neural + GT) → LLM/VLA Planning → Motion Primitives → Safety Verification → Dynamic Recovery. |
| **1:30 – 3:00** | Live Execution Demo | Terminal Screen Recording | Running `executor.py` end-to-end; showing `report.py` summary table; running `benchmark.py` OpenVINO INT8 vs FP32. |
| **3:00 – 4:00** | Autonomous Recovery | Slide 3 + GIF Animations | Seed 2 & Seed 7 controlled grasp failure recovery; re-observation & replanning loop in action. |
| **4:00 – 4:30** | Hardware & OpenVINO | Slide 4 (Benchmark Table) | Sub-2ms edge inference, INT8 NNCF quantization, Intel Core Ultra NPU/iGPU readiness. |
| **4:30 – 5:00** | Conclusion & Impact | Slide 5 (Summary & Repro) | 100% test pass (51+ tests), zero collisions, reproducibility via Docker & clean repo. |

---

## 📑 Slide-by-Slide Content & Narration Script

### Slide 1: Title & Problem (0:00 – 0:30)
**Slide Visuals**:
- Title: **TwinGuard: Closed-Loop Bimanual Physical AI**
- Subtitle: *Autonomous Recovery, Neural Perception & OpenVINO Acceleration on Intel Hardware*
- Badges: MuJoCo Physics | OpenVINO 2025 | NNCF INT8 | SO-101 Bimanual

**Voiceover Script**:
> "Welcome to TwinGuard. Open-loop robot manipulation fails when the real world changes — an object slips, a grasp misses, or a drawer sticks. TwinGuard solves this for bimanual manipulation by pairing high-level multi-modal planning with closed-loop physical verification and autonomous recovery. Today, we demonstrate two SO-101 robotic arms coordinating to set a table, pour drinks, and recover from failures in real-time."

---

### Slide 2: End-to-End Architecture (0:30 – 1:30)
**Slide Visuals**:
```
  [ Natural Language Instruction ] + [ RGB Camera View ]
                  │
                  ▼
   ┌───────────────────────────────┐
   │ Neural Object Detector (CNN)  │ ──► [ OpenVINO IR / INT8 NNCF ]
   └──────────────┬────────────────┘
                  ▼
   ┌───────────────────────────────┐
   │  Multi-Modal LLM Task Planner │ ──► Pydantic Strict Schema Validation
   └──────────────┬────────────────┘
                  ▼
   ┌───────────────────────────────┐
   │ Bimanual Primitives Execution │ ──► IK Solvers (Damped Least Squares)
   └──────────────┬────────────────┘
                  ▼
   ┌───────────────────────────────┐
   │    SafetyVerifier Feedback    │ ──► Collision, Drops, Grasp Zone Check
   └──────────────┬────────────────┘
                  │ Failure Detected?
                  ├──► YES: Dynamic Re-observation & Replanning
                  └──► NO:  Proceed to Next Primitive
```

**Voiceover Script**:
> "Our architecture separates high-level reasoning from low-level physics. First, visual observations and scene state feed into our multi-modal planner, which decomposes instructions into validated primitive steps: approach, grasp, lift, transport, release, open drawer, and pour. Each step executes via damped least-squares inverse kinematics. Crucially, after every single primitive, our SafetyVerifier physically inspects the MuJoCo simulation. If a grasp fails or an object shifts, TwinGuard re-observes the scene, injects the failure reason back into the planner, and synthesizes an immediate recovery step."

---

### Slide 3: Live Terminal Demonstration (1:30 – 3:00)
**Terminal Commands to Screen-Record**:
1. **Full Pipeline Run**:
   ```bash
   python evaluation/executor.py
   ```
   *Highlight*: Steps 1 through 10 executing sequentially with verification logs.
2. **10-Seed Benchmark Report**:
   ```bash
   python evaluation/report.py
   ```
   *Highlight*: 10 randomized seeds, 0 unexpected collisions, high recovery rate.
3. **OpenVINO Inference Benchmark**:
   ```bash
   python evaluation/benchmark.py
   ```
   *Highlight*: FP32 vs INT8 latency (<2ms) and throughput (>500 FPS).

**Voiceover Script**:
> "Here is TwinGuard running live. As `executor.py` processes the command 'Open the top drawer, pick up the plate with arm A, place it on the table, pick up the mug with arm B, and pour water into the mug with arm A', notice how both arms operate in a unified workspace without collisions. In `report.py`, we see across 10 randomized seeds that TwinGuard achieves 0 collisions and robust recovery. Finally, `benchmark.py` proves our neural perception runs at sub-2 millisecond latency with OpenVINO."

---

### Slide 4: Autonomous Failure Recovery (3:00 – 4:00)
**Slide Visuals**:
- Side-by-side video/GIF clips of **Seed 2** and **Seed 7** (from `recordings/demo_seed_2.gif` and `recordings/demo_seed_7.gif`).
- Callout box showing the log output:
  `[WARNING] Step 8 [grasp] failed verification: object 'mug' freejoint is not within gripper grasp zone`
  `[INFO] Initiating dynamic recovery: re-observing scene state and replanning...`
  `[SUCCESS] Step 8 recovered on attempt 1!`

**Voiceover Script**:
> "To test real-world robustness, we injected controlled physical grasp failures on seeds 2 and 7. Watch seed 2: on the initial mug grasp, the gripper slips. An open-loop system would have continued blindly and failed the entire task. TwinGuard detects the missing object immediately, pauses, re-scans the table, and commands a recovery approach. The second grasp succeeds, and the bimanual pour completes smoothly without human intervention."

---

### Slide 5: Intel Hardware Optimization & Summary (4:00 – 5:00)
**Slide Visuals**:
- **OpenVINO Optimization Table**:
  - Model: ObjectDetectorCNN
  - Precision: FP32 vs INT8 (NNCF Quantized)
  - Latency: ~1.9 ms (FP32) / ~1.1 ms (INT8)
  - Throughput: >500 FPS
  - Memory Footprint: 4x reduction via INT8
  - Hardware Readiness: Intel Core Ultra CPU, iGPU, and NPU
- **Reproducibility Badges**:
  - 51/51 Pytest Tests Passing
  - Complete Dockerfile & Setup Guide
  - Clean Modular Codebase under Apache-2.0

**Voiceover Script**:
> "For edge deployment on Intel Core Ultra, we quantized our neural perception using Intel's NNCF framework to INT8 and compiled it with OpenVINO latency hints and model caching. This achieves sub-2 millisecond inference with a 4x reduction in model size, making it ideal for the integrated NPU and iGPU of Intel Core Ultra processors. TwinGuard comes with a full 51-test suite, reproducible Docker container, and complete open-source documentation. Thank you!"
