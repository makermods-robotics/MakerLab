<h1 align="center">🦾 MakerLab</h1>

<p align="center">
  <b>A browser interface for <a href="https://github.com/huggingface/lerobot">LeRobot</a>, built for SO-101 leader/follower arms.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

MakerLab puts the full LeRobot workflow—robot setup, calibration, teleoperation,
recording, training, inference, and replay—into one web application. Plug in the
arms, start MakerLab, and operate everything from the browser instead of moving
between LeRobot CLI commands and keyboard-driven flows.

MakerLab is a fork of Hugging Face's [LeLab](https://github.com/huggingface/leLab),
the graphical interface created by Team LeLab at the 2025 LeRobot Worldwide
Hackathon and maintained by the [LeRobot](https://huggingface.co/lerobot) team.
MakerLab extends it with hardware-safety guards, bimanual support, reusable robot
profiles, data and model libraries, and a guided training and deployment flow.

## What you can do

| Workflow | Capability |
| --- | --- |
| **Set up robots** | Create reusable single-arm or bimanual robot profiles with saved ports, calibrations, cameras, and power limits. |
| **Calibrate** | Run guided manual or automatic calibration without keyboard prompts. |
| **Teleoperate** | Drive one or two follower arms from their leaders with live joint visualization. |
| **Record** | Capture multi-camera episodes into LeRobot datasets and resume existing sessions. |
| **Manage data** | Inspect, import, download, upload, rename, delete, and merge local or Hub datasets. |
| **Train** | Run local or Hugging Face Jobs training and continue from checkpoints. |
| **Manage models** | Import, download, upload, inspect, and select trained policies. |
| **Run inference** | Bind the required cameras and deploy a selected checkpoint on single or bimanual robots. |
| **Replay** | Re-run a recorded episode on the robot. |

## What MakerLab adds

### Hardware safety

- **Arm-identity guard:** fingerprints each arm before energizing it, catching
  swapped leader and follower ports.
- **Hand-motion port detection:** identifies a serial port by watching an arm
  move while its motors remain unpowered. Gripper wiggle remains available as
  an alternative.
- **Graceful stops:** freezes the arm, returns toward the session's starting
  pose, and releases torque. Press Stop again for immediate release.
- **Motor-power limits:** stores a per-robot power limit and reports live supply
  voltage and session power telemetry.

### Robots and calibration

- **Reusable robot profiles:** one selected robot supplies the ports,
  calibrations, cameras, layout, and power settings used across the app.
- **Bimanual workflows:** four-arm calibration, bimanual teleoperation, dual-arm
  visualization, bimanual recording, and bimanual inference.
- **Named calibration library:** create, rename, import, export, and safely
  delete calibration files without repeatedly overwriting one slot.
- **Automatic calibration:** calibrate one arm or a selected group concurrently,
  with per-arm status and controlled cancellation.

### Datasets, models, and training

- **Dataset library:** local and Hub status, metadata, tasks, per-task episode
  counts, camera information, background transfers, visibility, and tags.
- **Dataset merging:** validates compatible datasets and runs LeRobot's
  aggregation flow from the UI.
- **Model library:** manages local runs, imported checkpoints, downloaded Hub
  policies, model metadata, aliases, and deployment selection.
- **Checkpoint continuation:** resumes local or cloud training while preserving
  lineage and stitching metrics into one history.
- **Guided configuration:** selects the policy and dataset before training,
  checks availability, and offers required policy extras when missing.

## Installation

MakerLab requires Python 3.12 or newer and supports macOS and Linux. NVIDIA
Jetson installations have additional CUDA and camera-driver requirements; read
the [Jetson station guide](jetson/README.md) before creating the environment on
a Jetson intended for GPU training or inference.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) and Git LFS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git lfs install
```

Git LFS is required before rebuilding the frontend. The SO-101 meshes under
`frontend/public/` are LFS objects; building from an unsmudged clone embeds
pointer files and breaks the 3D viewer.

### Install as a tool

Use a tool install when you only want to run MakerLab. It installs a global
`makerlab` command and uses the production frontend bundled in the package.

```bash
uv tool install "git+https://github.com/makermods-robotics/MakerLab"
makerlab
```

To install from an existing local checkout or update a tool installation:

```bash
uv tool install /path/to/MakerLab
uv tool install --reinstall /path/to/MakerLab
```

A tool installation is a frozen snapshot and does not include the frontend
source needed by `makerlab --dev`. It also creates a separate environment with
its own copy of PyTorch and LeRobot. Jetson systems that require CUDA PyTorch
should use the source installation described in the
[Jetson guide](jetson/README.md).

### Install from source

Use an editable source installation when developing MakerLab or when the host
needs control over the Python and PyTorch environment.

```bash
git clone https://github.com/makermods-robotics/MakerLab
cd MakerLab
git lfs install --local
git lfs pull
uv venv --python 3.12
uv pip install -e .
.venv/bin/makerlab
```

When working on the source, invoke `.venv/bin/makerlab` explicitly. A bare
`makerlab` may resolve to an older `uv tool install` snapshot. Check with:

```bash
which makerlab
readlink "$(which makerlab)"
```

Running either entry point from the source environment also attempts to place
non-destructive symlinks in `~/.local/bin`, making `makerlab` and
`makerlab-station` available outside the checkout. Set
`MAKERLAB_NO_PATH_LINK=1` to disable that convenience.

### Run modes

| Command | Behavior |
| --- | --- |
| `makerlab` | Serves the production UI on `127.0.0.1:8000` and opens a browser. |
| `makerlab --dev` | Runs Vite with hot reload on `:8080` and reloadable FastAPI on `:8000`. Requires a source checkout and Node.js. |
| `makerlab --lan` | Binds to `0.0.0.0:8000` without opening a browser. |
| `makerlab --offline` | Sets `HF_HUB_OFFLINE=1`; hardware and local-model workflows remain available while Hub operations fail fast. |
| `makerlab-station` | Equivalent to `makerlab --lan --offline`. |
| `makerlab --stop` | Stops a running MakerLab process tree and releases ports `8000` and `8080`. |

## Development

Install the development and test dependencies into the editable environment:

```bash
uv pip install -e ".[dev,test]"
.venv/bin/makerlab --dev
```

Useful checks:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The Vite frontend lives in `frontend/`. The production bundle in
`frontend/dist/` is committed and packaged into the Python wheel. Normal
frontend development uses Vite and does not require rebuilding `dist` by hand.

## Resources

- [Jetson station guide](jetson/README.md)
- [LeRobot](https://github.com/huggingface/lerobot), the underlying robotics framework
- [LeLab](https://github.com/huggingface/leLab), the upstream project
- [LeLab Space](https://huggingface.co/spaces/lerobot/LeLab), the upstream hosted UI
- [LeRobot Discord](https://discord.gg/q8Dzzpym3f)
- [Contributor architecture notes](CLAUDE.md)

<div align="center">
<sub>MakerLab is maintained by <a href="https://github.com/makermods-robotics">makermods-robotics</a>. Forked with ❤️ from <a href="https://github.com/huggingface/leLab">LeLab</a>.</sub>
</div>
