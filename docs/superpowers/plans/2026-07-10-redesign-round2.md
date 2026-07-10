# Redesign Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each worker reads **Global Constraints** plus their own workstream section only.

**Goal:** Fix the unclickable Home page, re-project the hero animation to a low-elevation isometric view, convert robot create/settings into dialog flows (with delete, local-calibration picking, and a max-torque slider replacing motor power), restore leLab's browser-based camera previews, and move the dataset picker into an expandable card library at the bottom of the Collect page.

**Architecture:** All work happens on branch `redesign` of `makermods-robotics/MakerLab`. The orchestrator (main session) keeps taste-critical and conflict-prone files; four workers run in isolated worktrees on disjoint file sets. Workers commit once on throwaway `wt/*` branches; the orchestrator squash-merges each back onto `redesign` with normal commits (Isaac approved commits 2026-07-10). **Nothing is pushed and no PRs are opened** — the branch is not ready to ship.

**Tech Stack:** React 18 + Vite 8 + shadcn/ui + Tailwind (frontend), FastAPI + Pydantic (backend), pytest (backend tests only — the frontend has no test runner; verification is grep + `npm run lint` + `npm run build` + Playwright behavioral QA).

## Global Constraints

- **No pushes, no PRs, ever** (branch not ready). Commits are fine: workers make exactly one commit on their own `wt/*` branch; the orchestrator commits merges on `redesign`.
- **Worktree base check:** your worktree may branch from a stale base. Before starting, run `git merge redesign` (fast-forward expected) and verify this plan file plus `frontend/src/components/robot/robotSettingsStore.ts` exist.
- **Do not touch `frontend/dist/**`** — CI rebuilds it on `main`. (`git status` must show no `dist` changes from workers.)
- **Do not modify these orchestrator-owned files:** `frontend/src/pages/Home.tsx`, `frontend/src/components/home/BoothHero.tsx`, `frontend/src/components/landing/CreateRobotDialog.tsx`, `frontend/src/contexts/ApiContext.tsx` — the orchestrator edits them concurrently in the main checkout. (`useRobots.ts`: W1 only, one-line type addition, see its section.)
- **Behavior-preservation contract:** this is a UI restructure. Existing hooks, handlers, effects, API calls, and WebSocket listeners are untouchable except where a task explicitly names them. Never "clean up" dead-looking code.
- **No new npm or Python dependencies.** `git diff frontend/package.json pyproject.toml` must be empty.
- **Grep gates (run before returning; all must hold at merge):**
  - `grep -c "BackendCameraStream" frontend/src/components/recording/CameraConfiguration.tsx` → `0` (W2)
  - `grep -rn "motor_power" frontend/src` → no hits (W1)
  - `grep -rn "ImportCalibrationButton" frontend/src --include="*.tsx" -l` → only the component file itself, no importers (W1)
  - `grep -rn "DatasetPicker" frontend/src/pages/Collect.tsx` → no hits (W3)
- **Verification commands** every frontend worker must pass from `frontend/`: `npm ci` (worktrees have no node_modules), then `npm run lint` and `npm run build` — baseline is 6 pre-existing lint errors / 10 warnings and a passing build; do not fix pre-existing failures, add none. Backend worker: `cd <worktree> && /Users/isaac/Documents/GitHub/MakerLab/.venv/bin/python -m pytest -q` (the main repo's venv has the pinned lerobot; cwd precedence makes it import the worktree's `makerlab`) — baseline 599 passed. Sanity-check with `python -c "import makerlab; print(makerlab.__file__)"` that the worktree copy is imported. Do NOT use the conda python (its PyPI lerobot 0.5.1 breaks collection).
- **Vocabulary guard:** `device_type` is `"teleop" | "robot"`; `robot_type` is `"leader" | "follower"`. Don't conflate.
- **Design tokens:** zinc-based theme, shadcn components from `frontend/src/components/ui/`, primary CTA variant is `brand`. Both light and dark themes must look right.

## Design decisions (locked by orchestrator — do not re-litigate)

1. **Home tiles (exactly 3):** `New robot` (opens CreateRobotDialog), `Make new skill` (→ `/collect` with the preferred robot selected; opens CreateRobotDialog when no robots exist), `Browse skills` (→ `/market`).
2. **Robot settings is a Dialog**, not a page: `RobotSettingsDialog`, openable from anywhere via a module-level store (`openRobotSettings(name)`), mounted once in `App.tsx`. The `/calibration` route stays and renders the same panel full-page (deep links keep working).
3. **Settings dialog layout — 3 boxes:** top row grid `lg:grid-cols-2`: left **Robot configuration** (device/arm/ports, detect/wiggle), right **Calibration** (auto/manual calibrate actions, local-calibration-file pickers via `CalibrationLibrary`, max-torque slider); bottom full-width **Cameras** (`CameraConfiguration` inline).
4. **Max torque:** robot-record field `max_torque_limit`, int, clamped 0–1000, **default 380** (matches auto-cal `DEFAULT_TORQUE_LIMIT`). Written raw to the Feetech RAM `Torque_Limit` register at session start (replacing the percent×10 `motor_power` scaling). ⚠️ Behavior change: recording torque cap default drops 1000 → 380.
5. **Camera preview** returns to leLab's browser `getUserMedia`-by-`deviceId` streams (hook `useCameraStream` already exists in-repo). The deviceId↔cv2-index name-matching and the index re-sync effect stay exactly as they are. `BackendCameraStream` (MJPEG) remains for live teleop/inference feeds elsewhere — it is removed **only** from `CameraConfiguration.tsx`.
6. **Dataset library:** the `DatasetPicker` combobox leaves the Record card; a full-width expandable **Dataset library** section at the bottom of Collect renders dataset cards filterable by **Local / Yours on Hub / Public / Private**, reusing `MarketListingCard`. The **Merge** button (GitMerge icon + visible "Merge" label) sits on the right of the library header and opens the existing `MergeDatasetsDialog`.

---

## Task 0: Baselines + foundation (orchestrator, main checkout, before fan-out)

**Files:** none modified — read-only capture, plus one new file:
- Create: `frontend/src/components/robot/robotSettingsStore.ts`

**Interfaces:**
- Produces: `openRobotSettings(name: string | null): void`, `closeRobotSettings(): void`, `useRobotSettingsState(): { open: boolean; robotName: string | null }` — W1 builds the dialog against this exact API; orchestrator wires Home against it at merge.

- [ ] **Step 1: Record baselines** (from repo root):

```bash
cd frontend && npm run lint 2>&1 | tail -5 > /tmp/baseline-lint.txt; npm run build 2>&1 | tail -5 > /tmp/baseline-build.txt; cd .. && pytest -q 2>&1 | tail -3 > /tmp/baseline-pytest.txt
```

- [ ] **Step 2: Write the store** (shared foundation so W1 and the orchestrator's Home wiring agree on one API):

```ts
// frontend/src/components/robot/robotSettingsStore.ts
import { useSyncExternalStore } from "react";

interface RobotSettingsState {
  open: boolean;
  robotName: string | null;
}

let state: RobotSettingsState = { open: false, robotName: null };
const listeners = new Set<() => void>();

const emit = () => listeners.forEach((l) => l());

export const openRobotSettings = (name: string | null) => {
  state = { open: true, robotName: name };
  emit();
};

export const closeRobotSettings = () => {
  state = { ...state, open: false };
  emit();
};

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

const getSnapshot = () => state;

export const useRobotSettingsState = () =>
  useSyncExternalStore(subscribe, getSnapshot);
```

- [ ] **Step 3:** `cd frontend && npx tsc --noEmit 2>&1 | head` — no new errors from the store file. This file is committed to no branch; workers get its content verbatim inside their prompts (their worktrees won't contain it — W1 recreates it byte-identical; at merge the orchestrator keeps a single copy).

---

## Workstream A (orchestrator, inline, main checkout): Home fixes + isometric hero

Taste-critical; owner is the main session. Files: `frontend/src/pages/Home.tsx`, `frontend/src/components/home/BoothHero.tsx`.

### Task A1: Fix the unclickable Home page

Root cause: the exit-animation sidebar overlay at `Home.tsx:298-301` is `absolute inset-0`, last in DOM order, and only `opacity-0` when idle — invisible but still catching every click over the tiles and robot rows.

- [ ] **Step 1:** In `Home.tsx:301`, change the overlay className:

```tsx
className="pointer-events-none absolute inset-0 px-4 pb-4 pt-5 opacity-0 transition-opacity delay-200 duration-300 data-[exiting=true]:pointer-events-auto data-[exiting=true]:opacity-100"
```

- [ ] **Step 2:** Verify in dev server: all three tiles and robot rows are clickable; the exit animation still plays when opening a robot.

### Task A2: The 3 action tiles

- [ ] **Step 1:** In `Home.tsx`, replace the icon import line 3 with `import { Grid2X2, Plus, Sparkles } from "lucide-react";` and replace the three `ActionCard`s (lines 212-226):

```tsx
<ActionCard
  icon={<Plus className="h-4 w-4" />}
  label="New robot"
  onClick={() => setCreateOpen(true)}
/>
<ActionCard
  icon={<Sparkles className="h-4 w-4" />}
  label="Make new skill"
  onClick={() => {
    if (robots.length === 0) {
      setCreateOpen(true);
      return;
    }
    const preferred = robots.find((r) => r.name === selectedName) ?? robots[0];
    openRobot(preferred);
  }}
/>
<ActionCard
  icon={<Grid2X2 className="h-4 w-4" />}
  label="Browse skills"
  to="/market"
/>
```

### Task A3: Isometric hero (BoothHero re-projection)

- [ ] Rework `BoothHero.tsx`'s scene from the current flat side-on framing to a corner/isometric framing with a **low elevation (pitch) angle**: 2:1-ish axonometric floor plane (diamond grid), booth and arms drawn from a corner three-quarter view, horizon high in the frame so the camera reads as low. Keep: the two-arm pass-the-block animation loop, the IK approach, `prefers-reduced-motion` static pose, current color tokens, `viewBox` aspect. Orchestrator does this by hand — no worker.

### Task A4 (merge-time): Wire Home + create flow to the settings dialog

Done **after** W1's dialog merges. In `Home.tsx`:

- [ ] **Step 1:** `import { openRobotSettings } from "@/components/robot/robotSettingsStore";`
- [ ] **Step 2:** Header "Settings" link (lines 198-204) becomes a button calling `openRobotSettings(selectedName)`; the Cmd+, handler (line 152-155) calls `openRobotSettings(selectedName)` instead of `navigate("/calibration", …)`.
- [ ] **Step 3:** Create-flow auto-open — `CreateRobotDialog` stays untouched; the handler in `Home.tsx:359-363` becomes:

```tsx
onCreateNew={async (name, mode) => {
  const ok = await createRobot(name, mode);
  if (ok) {
    selectRobot(name);
    openRobotSettings(name); // straight into calibration, as a dialog
  }
  return ok;
}}
```

---

## Workstream W1 (opus-4.8, worktree `wt/settings-dialog`): Robot Settings Dialog

**Files:**
- Create: `frontend/src/components/robot/robotSettingsStore.ts` (byte-identical copy from Task 0 — content provided in your prompt)
- Create: `frontend/src/components/robot/RobotSettingsPanel.tsx` (extracted from `pages/Calibration.tsx`)
- Create: `frontend/src/components/robot/RobotSettingsDialog.tsx`
- Modify: `frontend/src/pages/Calibration.tsx` (becomes a thin page wrapper around the panel)
- Modify: `frontend/src/App.tsx` (mount `<RobotSettingsDialog />` once, inside the Router)
- Modify: `frontend/src/components/shell/RobotsSidebar.tsx:90-97` (gear → `openRobotSettings(r.name)` instead of `navigate("/calibration", …)`)
- Modify: `frontend/src/components/calibration/CalibrationLibrary.tsx` (remove download buttons; keep local-file selection, rename, delete)
- **useRobots.ts exception:** you may add exactly one line to the `RobotRecord` interface — `max_torque_limit?: number;` — and change nothing else in that file.

**Interfaces:**
- Consumes: `openRobotSettings` / `useRobotSettingsState` from the store (Task 0); `deleteRobot(name)` from `useRobots` (already exists, `useRobots.ts:200-233`); `POST /robots/{name}` accepting `{ max_torque_limit: number }` (W4's contract — build the UI against it now; it merges independently).
- Produces: `<RobotSettingsDialog />` (no props, self-mounting via store) and `<RobotSettingsPanel robotName={string|null} variant="dialog"|"page" />`.

**Tasks:**

- [ ] **W1.1 — Extract the panel.** Move the body of `Calibration.tsx` (the two-card grid at :1103 onward: Configuration card, calibrate actions, calibration checklist/`CalibrationLibrary`, `CameraConfiguration`) into `RobotSettingsPanel.tsx`. The panel takes `robotName` as a prop instead of reading `location.state`. `Calibration.tsx` keeps only: `AppShell` + `location.state.robot_name` → `<RobotSettingsPanel robotName={…} variant="page" />`. All hooks/handlers/effects move verbatim — do not refactor logic.
- [ ] **W1.2 — Re-layout to 3 boxes.** Inside the panel: `div.grid.gap-4.lg:grid-cols-2` → Box 1 "Robot configuration" (device type / arm / port with Detect + Wiggle + rescan), Box 2 "Calibration" (Auto-calibrate / Calibrate manually actions, the per-slot local-calibration-file pickers from `CalibrationLibrary`, and the max-torque slider from W1.4). Below, full-width Box 3 "Cameras" (`CameraConfiguration` inline). Use `Card` from ui/. In `variant="dialog"` the boxes must fit a `max-w-5xl` dialog with internal scroll (`max-h-[85vh] overflow-y-auto`).
- [ ] **W1.3 — Remove upload/download.** Delete the `ImportCalibrationButton` usage (upload) and the per-config download buttons from the calibration section / `CalibrationLibrary.tsx`. Keep: selection of existing local calibration files per slot, rename, delete. Grep gate: no importer of `ImportCalibrationButton` remains.
- [ ] **W1.4 — Max-torque slider replaces motor power.** Delete the "Motor power" block (`Calibration.tsx:1372-1431`, `powerDraft`/`commitMotorPower`/`supplyDv` gearing). Add in Box 2:

```tsx
<div className="space-y-1.5">
  <div className="flex items-baseline justify-between">
    <Label htmlFor="max-torque">Max torque limit</Label>
    <span className="font-mono text-[12px] text-muted-foreground">{torqueDraft}</span>
  </div>
  <input
    id="max-torque"
    type="range"
    min={0}
    max={1000}
    step={10}
    value={torqueDraft}
    onChange={(e) => setTorqueDraft(Number(e.target.value))}
    onPointerUp={commitTorque}
    onBlur={commitTorque}
    className="w-full accent-primary"
  />
  <p className="text-xs text-muted-foreground">
    Servo torque cap (0–1000). Default 380 — the auto-calibration setting.
  </p>
</div>
```

with `torqueDraft` seeded from `record?.max_torque_limit ?? 380` and `commitTorque` POSTing `{ max_torque_limit: torqueDraft }` to `${baseUrl}/robots/${robotName}` (same pattern as the old `commitMotorPower`). Grep gate: `grep -rn "motor_power" frontend/src` → 0.
- [ ] **W1.5 — Dialog + delete.** `RobotSettingsDialog.tsx`: shadcn `Dialog` bound to the store (`open` → `useRobotSettingsState().open`, `onOpenChange(false)` → `closeRobotSettings()`), `DialogContent className="max-w-5xl"`, title = robot name, body = `<RobotSettingsPanel variant="dialog" robotName={robotName} />`, footer left = destructive-outline "Delete robot" button → `AlertDialog` confirm ("Delete <name>? Calibration files are kept on disk.") → `deleteRobot(name)` → `closeRobotSettings()`. Mount `<RobotSettingsDialog />` once in `App.tsx` inside `BrowserRouter`. Rewire `RobotsSidebar.tsx` gear to `openRobotSettings(r.name)`.
- [ ] **W1.6 — Verify + return.** `npm run lint`, `npm run build` (baseline-compare), grep gates from Global Constraints, manual dev-server check: `/calibration` still renders; gear in sidebar opens the dialog over Collect. **One commit** on `wt/settings-dialog`. Return contract: branch name, worktree path, files changed, verification output, deviations + why.

---

## Workstream W2 (gpt-5.6-sol via codex wrapper, worktree `wt/camera-preview`): Browser camera previews (leLab parity)

**Files:**
- Modify: `frontend/src/components/recording/CameraConfiguration.tsx` (only file with logic changes)
- Read for parity (do not copy blindly — MakerLab's versions already exist): `frontend/src/hooks/useCameraStream.ts`, `frontend/src/hooks/useAvailableCameras.ts`
- leLab reference source (read-only, local): `/Users/isaac/.local/share/uv/tools/lelab/lib/python3.13/site-packages/frontend/src/{hooks/useCameraStream.ts,hooks/useAvailableCameras.ts,components/recording/CameraConfiguration.tsx}`

**Interfaces:**
- Consumes: `useCameraStream(deviceId: string, paused: boolean)` (exists, currently unused, `useCameraStream.ts:26`); `AvailableCamera { index, name, deviceId, available }` from `useAvailableCameras`.
- Produces: unchanged `CameraConfig` shape and unchanged props for `CameraConfiguration` — no consumer of this component may need edits.

**Tasks:**

- [ ] **W2.1 — Diff MakerLab's `useCameraStream.ts`/`useAvailableCameras.ts` against leLab's.** If retry-on-`NotReadableError`, track release on `paused`, or `devicechange` re-enumeration are missing in MakerLab's copies, port those behaviors. If identical, change nothing.
- [ ] **W2.2 — Swap `CameraStreamBox` to browser streams.** Change its props from `cameraIndex?: number` to `deviceId?: string`; render a `<video autoPlay muted playsInline>` driven by `useCameraStream(deviceId, paused)`; when `deviceId` is empty show the fallback ("No browser match — rescan or reconnect the camera") reusing the existing `VideoOff` empty-state styling. Remove the `BackendCameraStream` import from this file. Call sites: pre-add preview (line ~282) passes `selectedCamera.deviceId`; per-camera `CameraPreview` (line ~401) passes `camera.device_id`.
- [ ] **W2.3 — Preserve semantics:** preview appears immediately on dropdown selection **before** naming (already true — keep it); `releaseStreamsRef` / `paused` must still stop all browser tracks before recording starts (cv2 needs exclusive access); the `device_id`→`camera_index` re-sync effect (lines 89-98) is untouchable; duplicate detection by index-or-deviceId is untouchable.
- [ ] **W2.4 — Verify + return.** Grep gate: `grep -c "BackendCameraStream" frontend/src/components/recording/CameraConfiguration.tsx` → 0; confirm `BackendCameraStream.tsx` itself and its other consumers (`CameraFeed`, `TeleopCameraPanel`, `InferenceModal`) are untouched. `npm run lint && npm run build`. One commit on `wt/camera-preview`. Same return contract as W1.

---

## Workstream W3 (gpt-5.6-sol via codex wrapper, worktree `wt/dataset-library`): Collect dataset library + merge button

**Files:**
- Create: `frontend/src/components/collect/DatasetLibrary.tsx`
- Modify: `frontend/src/pages/Collect.tsx`
- Reuse unchanged: `MarketListingCard`, `MergeDatasetsDialog`, `CreateDatasetDialog`, `useDatasets`, `useHfAuth`

**Interfaces:**
- Consumes: `DatasetItem { repo_id, last_modified, private, source: "local"|"hub"|"both" }` from `useDatasets`; current HF username from `useHfAuth` (as `Collect.tsx:94,174` already does); `MarketListingCard` props (`kind`, `source`, `meta`, `complete` — see `Market.tsx:321-350`).
- Produces: `<DatasetLibrary selectedRepoId={string|null} onSelect={(repoId: string) => void} onMerge={() => void} />`.

**Tasks:**

- [ ] **W3.1 — Record card slimdown.** In the Record dataset card (`Collect.tsx:441-589`): remove the `DatasetPicker` combobox (lines 459-482). In its place a compact row: selected `repo_id` (mono, truncated; placeholder "No dataset selected"), the existing "+" `CreateDatasetDialog` trigger, and a small "Choose…" ghost button that expands + scrolls to the library section (`document.getElementById("dataset-library")?.scrollIntoView({behavior:"smooth"})` after setting its expanded state up in Collect).
- [ ] **W3.2 — Library section.** New full-width section at the bottom of the Collect page (below the existing grid, `id="dataset-library"`): a `Collapsible` whose header row has: left — chevron + "Dataset library" + count badge; right — the **Merge** button:

```tsx
<Button variant="outline" size="sm" onClick={onMerge}>
  <GitMerge className="h-4 w-4" /> Merge
</Button>
```

Expanded body: filter chips `All · Local · Yours on Hub · Public · Private` (single-select pills; "Yours on Hub" = `source !== "local"` && repo namespace === current HF username; Public/Private filter on the `private` flag among hub-visible sets), then a responsive card grid (`sm:grid-cols-2 xl:grid-cols-3`) of `MarketListingCard`-based cards showing repo_id, source badge(s) (`local` / `hub` / `local copy`), private badge, and relative `last_modified`. Clicking a card calls `onSelect(repo_id)` and highlights the selected card (ring). Keep the existing Dataset info card behavior (`DatasetInfoCard`) fed by the same selection state.
- [ ] **W3.3 — Merge wiring.** The old merge buttons in the Dataset card (`Collect.tsx:598-606` icon-only and `:623-630` full-width) are removed; `onMerge` → `setShowMergeDialog(true)` reuses the existing `MergeDatasetsDialog` mount (`Collect.tsx:658-663`) unchanged.
- [ ] **W3.4 — Verify + return.** Grep gate: no `DatasetPicker` reference left in `Collect.tsx`. `npm run lint && npm run build`. Dev-server check: selection flows into the recording start payload exactly as before (same state variable — do not rename it). One commit on `wt/dataset-library`. Same return contract.

---

## Workstream W4 (opus-4.8, worktree `wt/torque-limit`): Backend `max_torque_limit`

**Files:**
- Modify: `makerlab/utils/config.py` (robot-record field + clamp + migration)
- Modify: `makerlab/motor_power.py` (raw-value apply)
- Modify: `makerlab/record.py` (request field + call site)
- Sweep: `grep -rn "motor_power" makerlab/` — every hit (`teleoperate.py`, `rollout.py`, `rest_pose.py`, `torque.py`, `server.py`) must be visited and either migrated to the new field or left with a stated reason in your report.
- Tests: `tests/test_utils_config.py`, `tests/test_motor_power.py`, `tests/test_record.py`

**Interfaces:**
- Produces (frontend contract for W1): robot-record JSON field `max_torque_limit: int`, clamped to `[0, 1000]`, default `380`; accepted by `POST /robots/{name}`; returned by `GET /robots` / `GET /robots/{name}`. `motor_power` disappears from API responses; unknown/legacy keys on disk are ignored, with migration `max_torque_limit = clamp(motor_power * 10)` when a legacy record has `motor_power` but no `max_torque_limit`.

**Tasks (TDD — write each test first, watch it fail, implement, watch it pass):**

- [ ] **W4.1 — config.py.** Test first in `tests/test_utils_config.py`:

```python
def test_max_torque_limit_clamp_and_default(tmp_path, monkeypatch):
    # new records default to 380
    rec = get_robot_record_after_save({"name": "r1"})
    assert rec["max_torque_limit"] == 380

def test_max_torque_limit_migrates_from_motor_power():
    # legacy record: motor_power 50 (%) -> 500 raw
    rec = get_robot_record_for_disk_json({"name": "r1", "motor_power": 50})
    assert rec["max_torque_limit"] == 500
    assert "motor_power" not in rec
```

(Adapt to the file's existing test helpers/fixtures — `tests/test_utils_config.py` already exercises `save_robot_record`/`get_robot_record`; follow its patterns exactly.) Implement: replace `motor_power` in the typed-field handling (`_ROBOT_*_FIELDS`, `_empty_record` at `config.py:429`, clamp helper at `config.py:393-398`) with `max_torque_limit` (`MAX_TORQUE_LIMIT_MIN = 0`, `MAX = 1000`, `DEFAULT = 380`, `clamp_max_torque_limit`), plus the read-time migration in `get_robot_record`.
- [ ] **W4.2 — motor_power.py.** Rename the public apply to `apply_torque_limit(device, value: int, label: str)` writing the raw clamped value (no ×10 scaling) to the `Torque_Limit` register (address 48) with the same retry/normalize args as today (`motor_power.py:105-125`). Keep `clear_goal_velocity` and `read_supply_voltage` untouched. Update `tests/test_motor_power.py` expectations (raw value, bounds).
- [ ] **W4.3 — record.py + sweep.** `RecordingRequest.motor_power: int = 100` → `max_torque_limit: int = 380`; the apply call at `record.py:1102-1106` passes the raw value. Visit every remaining `motor_power` grep hit across `makerlab/` (`teleoperate.py`, `rollout.py`, `rest_pose.py`, `torque.py`, `server.py`) — migrate the same way; where a hit is genuinely unrelated (e.g. a docstring about supply voltage) leave it and say so in the report. Update `tests/test_record.py` schema tests.
- [ ] **W4.4 — Verify + return.** `/Users/isaac/Documents/GitHub/MakerLab/.venv/bin/python -m pytest -q` from the worktree root fully green (599-passed baseline + your new tests); `grep -rn "motor_power" makerlab/ tests/` output included in the report with a one-line disposition per remaining hit (target: none, or documented exceptions). One commit on `wt/torque-limit`. Same return contract.

---

## Merge & final verification (orchestrator)

- [ ] **M1.** As each worker completes: independent spot-check its branch diff (`git diff redesign...wt/<branch> | grep -E "useEffect|fetch\(|addEventListener|WebSocket"`) against the behavior-preservation contract, then `git merge --squash wt/<branch>` and commit on `redesign` (one commit per workstream).
- [ ] **M2.** Task A4 wiring (Home → dialog store).
- [ ] **M3.** Run every grep gate from Global Constraints; `npm run lint && npm run build`; `pytest -q` — all compared against Task 0 baselines.
- [ ] **M4.** Behavioral QA with Playwright MCP against `makerlab --dev` (both themes, every touched route): Home tiles clickable; create robot → settings dialog auto-opens; sidebar gear → dialog; delete robot (create a scratch robot first — never delete real ones); `/calibration` deep link still renders; camera previews against the **two real cameras plugged in** (real enumeration + getUserMedia); dataset library expand/filter/select; Merge button opens the dialog. Real leader/follower arms are connected — port detection and wiggle are fair game, but do not start recording/teleop sessions or calibration runs that move the arms unattended.
- [ ] **M5.** Independent cross-model review: `codex exec -s read-only --skip-git-repo-check "diff the working tree of /Users/isaac/Documents/GitHub/MakerLab against HEAD; verify <contract summary>; report file:line findings, verdict PASS/FAIL"`. Judge findings against this plan (sanctioned changes are not drift), fix real ones, re-verify the fixes specifically.
- [ ] **M6.** Deliver as uncommitted local changes on `redesign`; clean up `wt/*` worktrees and branches only after Isaac approves. **No commit on `redesign`, no push, no PR.**
