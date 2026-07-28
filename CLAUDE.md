# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## I10: real SIGTERM does not appear to gracefully stop a live, real-hardware teleoperation session

**Status: confirmed on real hardware (SO-101), root cause not found. No fix yet — this branch exists to track the investigation, not to ship a patch.**

PR #29 (I8) added `stop_and_wait()` to `teleoperate.py`/`record.py` and wired both into `shutdown_event()` so that a `SIGTERM` (plain `kill <pid>`, or a `uvicorn --reload` restart) gracefully stops an active teleoperation/recording session — return-to-rest, torque release, disconnect — before the process exits, instead of orphaning the in-process control-loop thread. All of I8's automated tests pass, but they only ever invoke `shutdown_event()` directly via `asyncio.run(...)`; none of them go through a real OS signal delivered to a live process that's actually driving hardware.

Tested that gap directly, against a real, connected SO-101 (single arm, leader `/dev/tty.usbmodem5B140311451` / follower `/dev/tty.usbmodem5B3E0901701`, calibration profile "autocalli robot"):

1. Started a real teleoperation session via `POST /move-arm` (real `SO101Follower`/`SO101Leader`, torque enabled, control loop running) and deliberately never called stop.
2. Sent a real `kill -TERM` to the running `makerlab` server process (prod mode, `_run_prod()`, no `--reload`).

**Result, twice in a row:** the process exited in ~0.2–0.3s with zero log output from `handle_stop_teleoperation()`/`stop_and_wait()` — no "Stop teleoperation triggered," no rest-pose return, no disconnect. A normal, uninterrupted stop of the same kind of session takes ~4s and logs every step. Both times the arm was left with unknown torque state; recovered by reconnecting and issuing a clean stop, confirmed released (`outcome: "ok"`) before continuing.

### What was ruled out (all via hardware-free repros against the real code, `SIGTERM`-tested each time)

- `teleop.stop_and_wait()` called directly with a fake background-thread worker — correct, full timing.
- The real `makerlab.server:app` (real `shutdown_event()`, real `asyncio.gather`/`asyncio.to_thread` wiring from I8) with a fake worker — correct, full timing.
- The real `server:app` with a fake worker doing **real pyserial reads** on the actual connected port at loop-speed cadence (no lerobot classes, no torque) — correct, full timing.
- Custom signal handlers inside lerobot's motor bus / SO101 robot classes — none exist; ruled out by source inspection.
- `timeout_graceful_shutdown=2` (set in `scripts/makerlab.py`'s `_run_prod()`) cutting the lifespan shutdown handler short — ruled out: a real 2s+ shutdown handler completes fully when hardware isn't in the loop (matches the existing shutdown-audit's own finding that this setting only bounds in-flight HTTP requests, not the lifespan call).

The only thing that reproduces the failure is the **real, pre-existing `teleoperation_worker()` closure inside `handle_start_teleoperation()`** (code that predates I8) actually driving the real servos. So this is not a logic bug in I8's own diff — `stop_and_wait()` and the `shutdown_event()` wiring both work correctly in isolation. The problem is that I8's fix depends on cooperation from the real hardware-driving loop, and that loop does not appear to respond the same way to a genuine OS-delivered `SIGTERM` as it does when `shutdown_event()` is invoked directly in-process.

### Not yet determined

- The exact mechanism (nothing pinned beyond the ruled-out list above — candidates not yet tested: whether a blocking Feetech-bus read/write syscall in the worker thread is somehow interrupted or left in a bad state by signal delivery, and whether that manifests as a silent exception path not reaching the loop's own `except`/`finally`).
- Whether recording (`record.py`'s side of the same I8 mechanism) has the same problem — blocked in this environment by an unrelated bug: any real recording session currently fails immediately with `AttributeError: 'DatasetRecordConfig' object has no attribute 'vcodec'` deep in lerobot's dataset/video pipeline (confirmed not caused by `create_record_config()` itself, which builds a valid config in isolation — likely a lerobot pin/version mismatch in this environment). Recording could not be tested end-to-end against real hardware until that's resolved separately.
- Whether I5 (inference) / I7 (auto-calibration) share this gap. Both use **subprocesses**, not in-process threads, so the shutdown mechanism differs enough that this may or may not generalize — genuinely open, not tested.

### Recommendation

Don't treat I8 as closing the real-world `kill <pid>` / `--reload` risk until this is root-caused and fixed. Next step is probably instrumenting the real worker loop directly (e.g., wrapping the Feetech bus calls to log entry/exit around each real read/write) and repeating the real-SIGTERM test, to catch exactly which call is in flight — or isn't returning normally — at the moment of signal delivery. That needs more live-SIGTERM cycles against real hardware; budget for arm wear/time accordingly and always follow with a clean reconnect-and-stop to confirm the arm ends up released.

## Repository purpose

MakerLab is a FastAPI + React web interface wrapping the [LeRobot](https://github.com/huggingface/lerobot) framework for the SO-101 leader/follower arm (single or bimanual). It exposes teleoperation, dataset recording, calibration, training, inference, and replay as HTTP/WebSocket endpoints, replacing LeRobot's CLI + keyboard-driven flows. It is a fork of Hugging Face's [leLab](https://github.com/huggingface/leLab), heavily extended by [makermods-robotics](https://github.com/makermods-robotics).

The frontend (React + Vite) lives in [`frontend/`](frontend/). The built bundle in `frontend/dist/` is committed and shipped inside the Python wheel as package data (`frontend.__init__.py` makes setuptools treat it as a package); [`makerlab/server.py`](makerlab/server.py) mounts it as `StaticFiles` at `/` so a single `makerlab` process serves both API and UI on `:8000`.

## Common commands

Install and run: see [README.md](README.md) Quick Start (uv editable install; `makerlab` / `makerlab --dev`). Requires Python ≥3.12. Use the repo `.venv` — pytest fails to collect under other interpreters because of the pinned lerobot.

`lerobot` is pinned to a specific commit on `huggingface/lerobot` `main` (see [pyproject.toml](pyproject.toml)) — no PyPI release yet exposes the `lerobot-rollout` script that [makerlab/rollout.py](makerlab/rollout.py) shells out to. Bump the SHA when you want newer upstream changes; expect import-path drift and adjust call sites.

When `frontend/**` (excluding `frontend/dist/**`) changes on `main`, [`.github/workflows/build_frontend.yml`](.github/workflows/build_frontend.yml) auto-rebuilds `frontend/dist/` and commits it back — don't rebuild it by hand for a PR. `makerlab --dev` serves from Vite, no rebuild needed.

Test with `pytest` (install `.[test]`), lint with `ruff check` / `ruff format` (config for both in [pyproject.toml](pyproject.toml)). Tests in [tests/](tests/) cover request schemas, pure helpers, and idle/mutex branches; subprocess/thread happy paths and HF Jobs integration are **deliberately** not unit-tested — don't add coverage there. There is no Python build step; for end-to-end validation, run `makerlab` and exercise endpoints.

Frontend checks (run from `frontend/`): `npm run lint`, `npx tsc --noEmit`, `npm run build`. Some type/lint errors pre-date any given change — record the baseline before you start and compare against it; don't fix pre-existing errors unrelated to your change.

## Architecture

### Backend module layout (`makerlab/`)

[server.py](makerlab/server.py) is the FastAPI router (~2600 lines; many request models are defined inline there). Each feature lives in its own module that owns its global state (module-level flags + per-feature locks) and exposes handler functions the router calls.

**Robot flows:**

- [record.py](makerlab/record.py) — dataset recording. The record loop is reimplemented as `record_with_web_events`, driven by a `web_events` dict (`exit_early` / `stop_recording` / `rerecord_episode`) that frontend buttons toggle — there is no keyboard listener.
- [teleoperate.py](makerlab/teleoperate.py) — leader→follower teleoperation.
- [calibrate.py](makerlab/calibrate.py) — step-by-step manual web calibration: `CalibrationManager` singleton with a `_step_complete` threading.Event.
- [auto_calibrate.py](makerlab/auto_calibrate.py) — automatic calibration: runs the vendored Feetech autocal ([vendor/feetech_autocal/](makerlab/vendor/feetech_autocal/)) as a subprocess that drives the arm under torque and **writes servo EEPROM**.
- [rollout.py](makerlab/rollout.py) — inference: runs a trained policy on the follower via a `lerobot-rollout` subprocess; single global session.

**Hardware safety (modules that guard or touch servos):**

- [arm_identity.py](makerlab/arm_identity.py) — fingerprints each arm via servo EEPROM before energizing; read-only, runs after bus connect, strictly before torque.
- [identify.py](makerlab/identify.py) — hand-motion port detection (watches raw positions while the user swings the arm); read-only, no torque. [wiggle.py](makerlab/wiggle.py) is the legacy variant that drives the gripper.
- [motor_power.py](makerlab/motor_power.py) — per-robot motor-power cap (used as autocal drive torque); `reset_torque_limit` un-throttles `Torque_Limit` so an earlier autocal doesn't silently limit the next session. **Writes servo registers.**
- [rest_pose.py](makerlab/rest_pose.py) — captures the start pose and gently returns the arm before torque release. Hand-mirrored twin of logic in the vendored autocal script — change one, check the other.
- [torque.py](makerlab/torque.py) — shared `force_disable_bus_torque` fallback (motor-by-motor release, loud on failure).

**Data & training:**

- [datasets.py](makerlab/datasets.py) / [models.py](makerlab/models.py) — local + Hub browsers (fan-out Hub listing with offline resilience; caches behind locks).
- [merge.py](makerlab/merge.py) — wraps lerobot's `aggregate_datasets` as a subprocess.
- [train.py](makerlab/train.py) / [jobs.py](makerlab/jobs.py) — local training subprocess lifecycle; `JobRunner`/`JobRegistry` persist run history to `outputs/train/`.
- [runners/hf_cloud.py](makerlab/runners/hf_cloud.py) — training on HF Jobs GPUs (replaces the image's bundled lerobot with MakerLab's pin in-container).

**utils/:**

- [utils/config.py](makerlab/utils/config.py) — shared paths and persistence. **Import shared constants from here, do not hardcode paths in feature modules.**
- [utils/robot_factory.py](makerlab/utils/robot_factory.py) — the single place `SO101LeaderConfig`/`SO101FollowerConfig`/`BiSO*Config` objects are assembled (`build_single_configs` / `build_bimanual_configs`); rollout.py builds CLI args separately.
- [utils/hf_auth.py](makerlab/utils/hf_auth.py) (cached `whoami`, offline detection), [utils/devices.py](makerlab/utils/devices.py) (force-close serial ports/cameras), [utils/errors.py](makerlab/utils/errors.py) (error-text → plain-language hints), [utils/system.py](makerlab/utils/system.py) (optional-extra pip installs as subprocess).

### State model & mutual exclusion

Each feature module owns module-level globals (`recording_active`, `teleoperation_active`, `inference_active`) protected by per-feature locks. Teleoperation, recording, and inference **are mutually exclusive, enforced in code** — not by a shared lock, but by reciprocal active-flag checks at each feature's start (e.g. `handle_start_teleoperation` refuses while recording or inference is active). New features that drive the robot must add the same reciprocal checks.

### WebSocket broadcast

server.py defines a single `ConnectionManager` with a background `_broadcast_worker` thread that drains a `queue.Queue` and forwards joint data to all `/ws/joint-data` clients via a thread-local asyncio loop. Feature modules get the manager passed in and call `manager.broadcast_joint_data_sync(data)` from their worker threads. Don't `await` from these threads — use the sync queue method.

### Persistent state on disk

All under `~/.cache/huggingface/lerobot/` (managed in [utils/config.py](makerlab/utils/config.py); writes are atomic):

- `calibration/teleoperators/so_leader/*.json`, `calibration/robots/so_follower/*.json` — named calibrations (leader = "teleop", follower = "robot")
- `robots/*.json` — per-robot records: arm layout (`mode: single|bimanual` with right-arm fields), ports, cameras, calibration names, `motor_power`
- `makerlab_biso/` — bimanual calibration staging
- `ports/{leader,follower}_port.txt` — last-used serial ports
- `dismissed_hub_jobs.json`, `saved_custom_{datasets,models}.json`, `hidden_{datasets,models}.json` — UI-level bookkeeping

`device_type` in API requests is `"teleop"` or `"robot"` (mapped to leader/follower paths). `robot_type` in port endpoints is `"leader"` or `"follower"`. Don't conflate the two vocabularies.

### Calibration files: dual-location pattern

`setup_calibration_files` ([utils/config.py](makerlab/utils/config.py)) copies user-selected configs into LeRobot's expected locations under `calibration/`. Recording and teleoperation call it before starting (replay uses `setup_follower_calibration_file`). New features that drive a robot must do the same.

## Frontend layout (`frontend/src/`)

React + Vite + TypeScript with shadcn/radix primitives. Four pages (`Launchpad`, `Teleoperation`, `Training`, `NotFound`); ~100 components grouped by feature area (`calibration/`, `control/`, `dialogs/`, `studio/`, `library/`, `recording/`, `jobs/`, `launchpad/`, … plus shared `ui/`); state via React contexts (`ApiContext`, `StudioContext`, `InferenceSessionContext`, `UrdfContext`, …) and ~19 data/session hooks (`useRobots`, `useDatasets`, `useRealTimeJoints`, …). No Redux/Zustand.

## UI verification scope

The UI is verified in **light mode only** and is a desktop tool driven from a laptop next to the arms — **skip mobile/responsive-breakpoint checks**. Caveat: `ThemeProvider` defaults to `system`, so OS-dark users do get the `dark` class with unaudited styling; don't polish dark mode unless asked.

## Hardware target

SO-101 leader/follower arms, single or **bimanual** (two leader/follower pairs via `BiSOLeaderConfig`/`BiSOFollowerConfig`). Robot config construction is centralized in [utils/robot_factory.py](makerlab/utils/robot_factory.py) — adding a robot type means extending the factory, plus calibrate.py and rollout.py which build their configs/args themselves.
