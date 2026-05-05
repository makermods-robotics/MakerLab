# Job Checkpoints + On-Robot Inference

## Goal

Make every checkpoint produced by a training job visible and runnable on the user's robot.
On the Landing page tile and the per-job page, show a checkpoint dropdown (latest pre-selected) and a Play button. Clicking Play opens a small "Configure Inference" modal — modelled on the existing Recording configuration modal — and starts a `lerobot-rollout` subprocess driving the SO-101 follower with the selected policy.

This is two pieces of work bundled into one user-facing feature:

1. Surface checkpoints (list endpoint, dropdown UI, play affordance).
2. Add an "inference mode" — a brand-new feature alongside teleoperation/recording/replay — that runs `lerobot-rollout` on the user's hardware.

Plus a third backend-only piece needed for parity:

3. For HF Cloud training jobs, push every intermediate checkpoint to the Hub during training (today only the final model is pushed).

## Non-goals

- Live URDF visualization while inference runs. The follower's serial bus is owned by the rollout subprocess and can't be opened twice. Drop the live joint stream during inference for v1; the running page shows status + elapsed + stop only.
- Inference history / per-run pages. Inference is a "mode" (one global session at a time) like teleoperation, not a "job" with a registry. No `/inference/<id>` route. The existing empty `/inference` route becomes the single running page.
- Configurable `policy.device`, Rerun visualization, `torch.compile`, or other rollout knobs. Defaults only for v1; auto-detect device server-side (cuda → mps → cpu).
- Modifying or replacing LeRobot's training script. The cloud sidecar runs alongside it, doesn't fork it.

## Architecture

### Inference is a mode, not a job

Reasons:

- It's interactive hardware-driven (like teleop, recording, replay), not batch (like training).
- Mutex naturally lives at module level alongside the existing `teleoperation_active` / `recording_active` flags.
- Avoids invasive changes to `app/jobs.py`, which is shaped around training subprocesses with metrics parsing.
- Matches the existing UI pattern: configure from a modal → land on a real-time page → stop returns home.

One inference session at a time, mutually exclusive with teleoperation and recording (they all drive the follower's serial bus).

### How `lerobot-rollout` is invoked

`python -m lerobot.scripts.lerobot_rollout` accepts `--policy.path=<local_dir or hub_repo>`, `--robot.type=so101_follower`, `--robot.port=...`, `--robot.cameras="{...}"`, `--task="..."`, `--duration=N`. We spawn it as a subprocess (not a thread) so cancellation is clean (`SIGTERM`, 5s grace, `SIGKILL`).

For Hub-checkpoint subdirectories (e.g. `checkpoints/05000/pretrained_model`), we resolve to a local directory at start-inference time via `huggingface_hub.snapshot_download(repo_id, allow_patterns=f"checkpoints/{step}/pretrained_model/*")` and pass the resolved local path to `--policy.path`. Repo-root checkpoints (final model only) can be passed as `--policy.path=user/repo` directly.

### How cloud jobs get every checkpoint into the Hub

LeRobot's training script only `push_to_hub`s the final model at end-of-training. To get parity with local jobs (where the dropdown grows live), we ship a small **sidecar uploader** in the same HF Jobs container.

Today, `HfCloudJobRunner.start()` calls:

```python
self._api.run_job(image=LEROBOT_IMAGE, command=argv, flavor=..., secrets={"HF_TOKEN": token})
```

Where `argv` is the lerobot CLI `argv`. Change this so `command` is a Python wrapper:

```python
command = ["python", "-c", WRAPPER_SOURCE, "--", *argv]
```

`WRAPPER_SOURCE` (a string constant in `app/runners/hf_cloud.py`) does:

1. Parses its own argv after `--` to recover the lerobot argv. Pulls `--output_dir` and `--policy.repo_id` out of it.
2. Spawns `python -m lerobot.scripts.lerobot_train ...` as a subprocess.
3. Runs a watcher thread that every 15s scans `<output_dir>/checkpoints/*/pretrained_model/config.json`. For each step folder not yet uploaded, calls `HfApi.upload_folder(folder_path=step_dir, repo_id=repo_id, path_in_repo=f"checkpoints/{step}", commit_message=f"checkpoint {step}")`. Tracks "seen" steps in an in-memory set.
4. Waits for the trainer subprocess; the watcher does one final pass on exit; wrapper returns the trainer's return code.

`HF_TOKEN` is already passed as a secret. `huggingface_hub` is already in `huggingface/lerobot-gpu:latest`. No image changes needed.

The existing end-of-training `push_model_to_hub` from lerobot's training script is left intact — it overwrites the repo root with the final model, leaving `checkpoints/<step>/` subfolders alone.

## Components

### Backend

#### `app/jobs.py` changes

- New `JobCheckpoint` Pydantic model: `step: int`, `source: Literal["local", "hub"]`, `ref: str` (the local dir path, or for hub: `f"{repo_id}@checkpoints/{step}"`).
- New method `JobRegistry.list_checkpoints(job_id) -> List[JobCheckpoint]`.
  - **Local**: scan `<output_dir>/checkpoints/*/pretrained_model/config.json`, parse step from the dir name (zero-padded), sort ascending.
  - **HF Cloud**: `HfApi.list_repo_files(record.hf_repo_id, repo_type="model")`, filter for paths matching `checkpoints/<int>/pretrained_model/config.json`, dedupe by step.
- New `checkpoint_count: int` field on `JobRecord`. Computed inside `list()` and `get()`. For local jobs it's a cheap readdir per call. For cloud jobs we add a per-job TTL cache (30s) so `GET /jobs` doesn't hammer `list_repo_files` when there are several cloud jobs listed.

#### `app/inferring.py` (new file)

Mirrors `app/teleoperating.py` in shape:

- Module globals: `inference_active`, `inference_proc`, `inference_started_at`, `inference_meta` (selected policy ref, source job id, step, duration).
- `InferenceRequest` Pydantic model:
  ```python
  follower_port: str
  follower_config: str
  policy_ref: str        # opaque ref returned by the checkpoints endpoint
  task: str
  cameras: dict          # same shape Recording uses
  duration_s: int = 60
  ```
- `handle_start_inference(request, websocket_manager) -> dict`:
  - **Mutex**: refuses if any of `teleoperation_active`, `recording_active`, `inference_active`. Tighten the existing `handle_start_teleoperation` and `handle_start_recording` to also check the other two flags (today they only check their own — see "Cross-cutting changes" below).
  - Calls `setup_calibration_files(...)` for the follower (same as teleop/recording do).
  - Resolves `policy_ref` to a usable `--policy.path` argument:
    - Local checkpoints: ref is already an absolute local path.
    - Hub checkpoints: `huggingface_hub.snapshot_download(repo_id, allow_patterns=f"checkpoints/{step}/pretrained_model/*", local_dir=<temp>)` and use the resolved subdir.
  - Auto-detects device (cuda → mps → cpu) via the same `torch.cuda.is_available()` / `torch.backends.mps.is_available()` checks lerobot uses internally.
  - Spawns `python -m lerobot.scripts.lerobot_rollout --strategy.type=base --policy.path=... --robot.type=so101_follower --robot.port=... --robot.cameras="{...}" --task="..." --duration=N` as a subprocess.
  - Captures stdout/stderr to a log file under `~/.cache/huggingface/lerobot/inference_logs/<timestamp>.log` (kept short — last N runs only).
- `handle_stop_inference()`: SIGTERM the subprocess, wait 5s, SIGKILL if still alive. Clear globals.
- `handle_inference_status() -> dict`: `{inference_active, started_at, elapsed_s, duration_s, policy_ref, job_id, step}`.

#### `app/main.py` route additions

- `GET /jobs/{job_id}/checkpoints`
- `POST /start-inference`
- `POST /stop-inference`
- `GET /inference-status`

#### `app/runners/hf_cloud.py` changes

- Add module-level `WRAPPER_SOURCE` string.
- Change `HfCloudJobRunner.start()` to submit `command=["python", "-c", WRAPPER_SOURCE, "--", *argv]` instead of raw `argv`.
- No other changes — log tailing, terminal status detection, etc. stay identical.

#### Cross-cutting changes

- Tighten the start-handlers in `app/teleoperating.py`, `app/recording.py`, `app/inferring.py` so each refuses if **any** of the three modes is active, not just its own.

### Frontend

#### New components

- **`frontend/src/components/landing/InferenceModal.tsx`** — cloned from `RecordingModal.tsx`, simplified.
  - Robot picker (reuses the existing `useRobots` hook + `RobotRecord`, same UX as Recording).
  - Checkpoint dropdown — populated from `GET /jobs/{job_id}/checkpoints`, latest pre-selected. The user can reach this modal pre-populated with a specific step if they came from the tile dropdown or the page dropdown; in that case the modal opens with that step pre-selected.
  - `CameraConfiguration` reused as-is.
  - Task description input.
  - Max duration input (seconds), default 60.
  - "Start Inference" button → `POST /start-inference` → on 200, navigate to `/inference`. On 409 (mutex conflict) keep the modal open and toast the error.

- **Inference running page** — replaces the empty body of `frontend/src/pages/Inference.tsx`. Cloned from `Recording.tsx` but trimmed:
  - Status pill: `RUNNING INFERENCE` / `COMPLETED`.
  - Elapsed timer + duration progress bar.
  - Stop button with confirm dialog.
  - No URDF widget.
  - Polls `GET /inference-status` every 1s.
  - Auto-navigates to `/` when the backend reports `inference_active=false`.

- **`frontend/src/components/jobs/CheckpointDropdown.tsx`** (new, shared) — compact dropdown showing the currently-selected step; popover lists all steps with their absolute step number. Used in three places: the JobCard tile, the MonitoringMode panel, and inside the InferenceModal.

#### `JobCard.tsx` changes

- Hide the progress bar when `job.state !== "running"` (i.e. done/failed/interrupted). Today it's always visible.
- When `job.checkpoint_count > 0`, render a row inside the card with `<CheckpointDropdown>` + a green Play icon button. Layout:
  - **Running with checkpoints**: card shows progress bar (top) and inference row (bottom).
  - **Done with checkpoints**: card shows only the inference row in place of the (now-hidden) progress bar.
  - **No checkpoints**: card shows only the existing state UI (progress bar while running, nothing else when done).
- The Play click `e.stopPropagation()`s (so the card-level navigate to `/training/<id>` doesn't fire) and calls a new `onPlay(job, selectedStep)` prop.

#### `JobsSection.tsx` changes

- Hoists `[modalOpen, modalJob, modalStep]` state.
- Renders one `<InferenceModal>` instance for the whole section.
- Passes `onPlay` to each `JobCard`.

#### `MonitoringMode` (in `Training.tsx`) changes

- Above `<TrainingLogs>`, add a "Run inference" panel containing `<CheckpointDropdown>` (latest preselected, fetched on mount + every 5s while job state is `running`) and a "Run on robot" button. Disabled if 0 checkpoints. Clicking opens its own `<InferenceModal>` instance with the selected step pre-populated.

#### New API client modules

- **`frontend/src/lib/checkpointsApi.ts`**: `listJobCheckpoints(baseUrl, fetcher, jobId)`.
- **`frontend/src/lib/inferenceApi.ts`**: `startInference`, `stopInference`, `getInferenceStatus`.

#### `JobRecord` type extension (in `frontend/src/lib/jobsApi.ts`)

Add `checkpoint_count: number` to the `JobRecord` interface — mirrors the new backend field.

## Data flow

### Listing checkpoints

```
[Landing page]                            [MonitoringMode]
JobsSection                               Training.tsx (jobId branch)
  └─ GET /jobs           (every 1–5s)       └─ GET /jobs/{id}        (every 1s)
       returns JobRecord[]                       returns JobRecord
       (incl. checkpoint_count)                  (incl. checkpoint_count)
  └─ JobCard (per record)                   └─ GET /jobs/{id}/checkpoints (every 5s while running)
      └─ if checkpoint_count > 0:                 returns JobCheckpoint[]
          └─ CheckpointDropdown                └─ CheckpointDropdown
              └─ GET /jobs/{id}/checkpoints
                 (lazy on mount)
```

The tile dropdown only fires `GET /jobs/{id}/checkpoints` when the dropdown actually mounts (i.e. when `checkpoint_count > 0`), so cost scales with "how many tiles can play" not "how many tiles".

### Starting inference

```
User clicks ▶ on tile          User clicks "Run on robot" on monitoring page
       │                                  │
       ▼                                  ▼
   onPlay(job, step) ─────────────► InferenceModal opens, pre-populated
                                          │
                                          ▼
                                   User confirms robot + cameras + task + duration
                                          │
                                          ▼
                                   POST /start-inference
                                          │
                                          ├─► resolves policy_ref (snapshot_download for hub refs)
                                          ├─► spawns lerobot-rollout subprocess
                                          ▼
                                   200 → navigate to /inference
                                   409 → toast, modal stays open
```

### Stopping inference

```
User clicks Stop on /inference     OR     duration elapses naturally
       │                                          │
       ▼                                          ▼
POST /stop-inference                      subprocess exits 0
       │                                          │
       └─► SIGTERM → wait 5s → SIGKILL            └─► watchdog notices via proc.poll()
       │                                          │
       └────────────────► clears globals ─────────┘
                                  │
                                  ▼
                          GET /inference-status returns inference_active=false
                                  │
                                  ▼
                          Inference page auto-navigates to /
```

## Edge cases

- **0 checkpoints**: Play button doesn't render on the tile or in the monitoring panel. Prevents the user from opening a useless modal.
- **Race**: tile shows "checkpoint exists" stale → user clicks → `GET /jobs/{id}/checkpoints` returns empty → modal shows the dropdown disabled with a "no checkpoints yet" message and the Start button disabled. Auto-closes after 3s if still empty.
- **Hub repo not yet created**: `list_repo_files` raises 404 → list_checkpoints returns `[]`, `checkpoint_count` is 0. No Play button. Caller doesn't see an error.
- **Hub snapshot_download is slow** (multi-hundred-MB checkpoint): `POST /start-inference` blocks for the duration of the download. Acceptable for v1 — frontend shows "Starting…" until the response comes back. Could later be made async with a "preparing checkpoint" status, but YAGNI.
- **Mutex conflict**: 409 with a clear message naming which mode is busy ("teleoperation is currently active").
- **Inference subprocess crashes**: stderr captured to log file. Status flips to `inference_active=false`; running page shows "Inference exited unexpectedly — check ~/.cache/huggingface/lerobot/inference_logs/<timestamp>.log" and offers a "Back" button.
- **Local job currently training and using GPU**: starting inference on the same machine triggers GPU contention. We don't refuse — let the user decide. (Most SO-101 inference is small-model-on-CPU/MPS anyway.)
- **Sidecar uploader fails to push** (e.g. transient Hub 5xx): swallow + log + retry on next 15s tick. Don't kill the training job. The "seen" set is in-memory only — if the wrapper process crashes, all checkpoints get re-considered, and `upload_folder` is idempotent on identical content (creates an empty commit).
- **Cloud job killed before any checkpoint pushed**: repo may be empty. `list_checkpoints` returns `[]`. No Play button.
- **User has no HF token but tries to inference a Hub checkpoint**: `snapshot_download` raises an auth error. Surface as a 401 from `/start-inference` with a clear message.

## Testing strategy

This repo has no test suite. Validate manually:

1. Start a local training job with a small `save_freq` (~50 steps). Confirm checkpoints appear in the tile dropdown and the monitoring page dropdown live as steps tick over.
2. Start an `hf_cloud` training job with the same small save_freq. Confirm intermediate `checkpoints/<step>/` folders appear in the Hub repo within ~30s of being saved.
3. From a finished local job, click ▶ on the tile, confirm modal pre-populates the latest checkpoint, run with cameras off and a tiny duration. Confirm the robot moves and stops cleanly when duration elapses.
4. From a finished cloud job, do the same (tests the snapshot_download path).
5. Mid-inference, click Stop; confirm the subprocess dies and the page returns to `/`.
6. While inference is running, try to start teleoperation from Landing → expect a clear 409.
7. Mid-training (local), click ▶ on the tile to confirm we don't refuse — and that the rollout succeeds (or fails gracefully with a clear hardware/GPU message).
