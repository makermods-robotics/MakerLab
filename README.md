<h1 align="center">🦾 MakerLab</h1>

<p align="center">
  <b>A web interface for <a href="https://github.com/huggingface/lerobot">LeRobot</a>, built for the SO-101 leader/follower arm.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

**MakerLab** is a web app that puts the full LeRobot workflow — calibrate, teleoperate, record, train, replay — into a single browser UI. Plug in your arm, open the app, and go. No CLI gymnastics, no keyboard prompts.

MakerLab is a fork of Hugging Face's **[LeLab](https://github.com/huggingface/leLab)** — the graphical interface for LeRobot, originally built by Team LeLab at the 2025 LeRobot Worldwide Hackathon 🏆 and maintained by the [LeRobot](https://huggingface.co/lerobot) team. MakerLab, by [makermods-robotics](https://github.com/makermods-robotics), extends it with hardware-safety guards, bimanual support, and a more guided setup and training flow.

## Quick Start

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Just want to run it? One line, no clone:

```bash
uv tool install "git+https://github.com/makermods-robotics/MakerLab"
makerlab            # serves the UI + API on :8000, opens your browser
```

Working on the code? Clone and do an editable install instead:

```bash
git clone https://github.com/makermods-robotics/MakerLab
cd MakerLab
uv venv --python 3.12
uv pip install -e .
.venv/bin/makerlab            # serves the UI + API on :8000, opens your browser
```

The package and CLI are named `makerlab`. See [INSTALL.md](INSTALL.md) for platform-specific setup (macOS, Jetson) and network tips.

## What you can do

The core LeRobot workflow, inherited from LeLab and extended throughout:

<div align="center">
  <table>
    <tr>
      <td>🎯 <b>Calibrate</b></td>
      <td>Guided web flow for both arms — manual or fully automatic, no keyboard prompts.</td>
    </tr>
    <tr>
      <td>🕹️ <b>Teleoperate</b></td>
      <td>Move the leader, the follower mirrors it. Live joint streaming into a 3D viewer.</td>
    </tr>
    <tr>
      <td>📹 <b>Record</b></td>
      <td>Capture episodes into a LeRobotDataset, with multiple cameras.</td>
    </tr>
    <tr>
      <td>🧠 <b>Train</b></td>
      <td>Kick off a LeRobot training job and watch the loss/lr chart live.</td>
    </tr>
    <tr>
      <td>🤖 <b>Run inference</b></td>
      <td>Execute a trained policy on the follower.</td>
    </tr>
    <tr>
      <td>⏪ <b>Replay</b></td>
      <td>Re-run any recorded episode.</td>
    </tr>
    <tr>
      <td>☁️ <b>Upload</b></td>
      <td>Push your dataset to the <a href="https://huggingface.co/">Hugging Face Hub</a> in one click.</td>
    </tr>
  </table>
</div>

## What MakerLab adds

**Hardware safety** — opinionated about not letting a wiring mistake break a servo:

- 🛡️ **Arm-identity guard** — fingerprints each arm's EEPROM before energizing, so a swapped leader/follower port is caught rather than driven.
- ✋ **Hand-motion port detection** — hit *Detect* and swing an arm's base to identify its serial port with no motor power. The legacy gripper-wiggle method is still available.
- 🛑 **Graceful stops** — teleop and auto-calibration freeze, return to the start pose, then release torque. Hit *Stop* twice for an instant release.
- 🔋 **Motor power limiting** — cap per-robot motor power, with a live supply-voltage readout and session power telemetry.

**Robots & calibration:**

- 🤝 **Robots as first-class objects** — create a robot through a dialog with an immutable arm layout (single or bimanual), and reuse it across every feature.
- 🦾 **Bimanual mode** — two leader/follower pairs: 4-arm calibration, bimanual teleoperation with a dual-arm 3D viewer, and bimanual dataset recording.
- 🏷️ **Named calibrations** — save calibrations under names instead of overwriting; deleting one in use unassigns it rather than blocking. A start-pose guard rejects calibrations that didn't begin from the middle pose, and <code>wrist_roll</code> is handled as a full turn to match upstream <code>lerobot-calibrate</code>.

**Datasets:**

- 🪪 **Dataset info cards** — episodes, cameras, and tasks with per-task episode counts, plus warnings on unusable datasets.
- 🔀 **Merge from the UI** — combine datasets (wraps LeRobot's <code>aggregate_datasets</code>), with legible errors and name validation.
- 🎥 **Preview before naming** — see all camera feeds before committing to a recording setup.

**Training:**

- 🧭 **Model-type-first entry** — pick the policy and dataset on the home page (availability-gated), frozen for the run thereafter; config guards, run names, and honest compute targets.
- ⏯️ **Continue from a checkpoint** — resume a saved run, with the lineage's loss chart stitched into one view and source checkpoints folded into the successor.
- 🗂️ **Job tooling** — checkpoint management, model display-name aliases, and idempotent imports with dedup.

## Resources

- **[LeRobot](https://github.com/huggingface/lerobot):** the underlying library — go here for everything beyond the UI.
- **[LeLab](https://github.com/huggingface/leLab):** the upstream project this is forked from; try its hosted UI on the [LeLab Space](https://huggingface.co/spaces/lerobot/LeLab).
- **[Discord](https://discord.gg/q8Dzzpym3f):** chat with the LeRobot community.
- **[CLAUDE.md](CLAUDE.md):** architecture rundown for contributors.

## Contribute

PRs welcome. Hot-reload mode for working on the code:

```bash
makerlab --dev
```

Vite on `:8080`, uvicorn `--reload` on `:8000`.

<div align="center">
<sub>MakerLab is maintained by <a href="https://github.com/makermods-robotics">makermods-robotics</a>. Forked with ❤️ from <a href="https://github.com/huggingface/leLab">LeLab</a>, originally hacked together by <a href="https://www.linkedin.com/posts/nicolas-rabault-_lerobot-hackathon-lerobot-ugcPost-7341065019368828930-jTnl/">Team LeLab at the 2025 LeRobot Worldwide Hackathon 🏆</a>.</sub>
</div>
