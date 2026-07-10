# MakerLab shadcn redesign — implementation plan (branch `shadcn-app`)

Implement the approved design prototype (`reference-design/*.html`, view over HTTP for the
live animations) as the real webapp, preserving every feature that works on `main`.
Architect: fable (main session). Workers implement per the task sections below.

## Global constraints (ALL workers — hard rules)

1. **Behavior preservation.** When editing an existing file, only markup/classNames/copy
   change. Hooks, handlers, effects, polling loops, API calls, WebSocket wiring,
   `stoppedRef` + `pagehide` + sessionStorage safety patterns are UNTOUCHABLE. Never
   "clean up" code that looks unused.
2. **Storage keys are frozen:** `makerlab.apiBaseUrl`, `makerlab.selectedRobot`,
   `makerlab.selectedDataset`, `makerlab.training.policyType`, `vite-ui-theme`,
   `makerlab:teleop-stopped`, `makerlab:recording-stopped`.
3. **Router-state contracts are frozen:** `/recording` requires
   `state.recordingConfig`; `/upload` takes `state.datasetInfo`; `/training` takes
   `state.policyType|resume|finetune`; `/calibration` takes `state.robot_name`.
4. **Design tokens only.** No hex colors, no `gray-*` Tailwind classes; use the CSS-var
   tokens (`bg-background`, `text-muted-foreground`, `border-border`, `bg-card`,
   `bg-primary`, `text-primary-foreground`, `bg-secondary`, `ok`/`warn`/`destructive`).
   No new npm dependencies. Fonts are Inter (display+body) and JetBrains Mono (mono);
   never reference Chakra Petch / Space Grotesk.
5. **Robot data flows through `useRobots()`** (singleton store) — never fetch `/robots`
   directly, never cache selection locally.
6. **Camera previews** use `BackendCameraStream` (cv2 MJPEG) only — no
   `getUserMedia`/`useCameraStream` in new code. Any surface that starts recording or
   inference must unmount/pause previews first (the `releaseStreamsRef` /
   `streamsPaused` handshake), because the backend needs exclusive cv2 access.
   Previews are allowed during teleop (serial-bus only). NO live previews on the
   record-live or inference-live screens (the session owns the cameras) — render the
   session state instead.
7. **Hardware safety in dev/testing:** never call `/move-arm`, `/start-recording`,
   `/start-inference`, `/start-auto-calibration`, `/start-calibration`, or `/wiggle`
   while developing. Wiring the buttons is required; clicking them is not.
8. Verification every worker runs before reporting:
   `cd frontend && npx tsc --noEmit && npm run build` (must pass) and
   `npx eslint src/<files-you-touched>` (no new errors).
9. Do not run git commands; report changed files instead. Do not edit files outside
   your ownership list. Read the design HTML for your page before coding.

## Design language (already applied by foundation)

shadcn/ui zinc: `--radius: 0.5rem`, Inter + JetBrains Mono, black/white primary
buttons, pill badges, soft `ok/warn/destructive` badge tints, rounded-xl cards with
`shadow-sm`, muted dashed `.media-slot` placeholders. The old orange/notch/stencil
variants are re-mapped in `components/ui/` to zinc equivalents — keep using semantic
variants (`variant="brand"` renders as primary now); do not introduce new variants.

## IA / routes (foundation wires these; workers fill the pages)

| Route | Page | Chrome |
|---|---|---|
| `/` | `pages/Home.tsx` — robot picker + booth hero; selecting a robot morphs the panel left, then navigates to `/collect` | none (page owns it) |
| `/collect` | `pages/Collect.tsx` — teleop + record + datasets | StageLayout (robots sidebar + stage dock) |
| `/training` | `pages/TrainDeploy.tsx` — jobs + models board + new-run config | StageLayout |
| `/market` | `pages/Market.tsx` — Hub datasets & models | StageLayout |
| `/recording` | existing `Recording.tsx`, re-skinned record-live | AppShell (live, no logo link) |
| `/inference` | existing `Inference.tsx`, re-skinned inference-live | AppShell (live) |
| `/training/:jobId` | existing `Training.tsx` monitoring, light re-skin | AppShell |
| `/teleoperation`, `/calibration`, `/upload`, `/edit-dataset` | unchanged (token-driven restyle only) | AppShell |
| `/legacy` | old `Landing.tsx`, hidden fallback during transition | AppShell |

`StageLayout` (foundation): fixed 288px robots sidebar (from `useRobots`: rows with
status dot from `is_clean`, mode badge, gear → `/calibration` with
`state.robot_name`; `+ New robot` → `CreateRobotDialog`; footer: `HfAuthChip`,
`ThemeToggle`) + bottom stage dock (Robot `/`, Collect `/collect`, Train & Deploy
`/training`, Market `/market`) + `<Outlet/>` with `pl-[288px] pb-24`.

## Worker tasks (disjoint file ownership)

### W1 — Home (`pages/Home.tsx`, `components/home/*` — new files only)
Design: `reference-design/home-rig-picker.html`. Centered cluster: `BoothHero`
(foundation component), wordmark + "SO-101 workbench · Settings", three action cards
(New robot → CreateRobotDialog; Import from Hub → `/market`; Browse Market →
`/market`), Recent robots list from `useRobots` (name, mode badge, mono meta:
ports/cams or "needs calibration" from `is_clean`), keyboard hints. Selecting a robot:
`selectRobot(name)`, play the collapse/slide-left transition (booth max-height→0,
panel translates left, ~700ms), then `navigate('/collect')`. First-run (no robots):
same screen; list shows empty state + New robot primary.

### W2 — Collect (`pages/Collect.tsx`, `components/collect/*` — new files only)
Design: `reference-design/collect.html`. Left 2/3: `VisualizerPanel` (existing —
URDF + live joints via WS when teleop runs; bimanual renders two viewers) + camera
strip (`CameraFeed` per `selectedRecord.cameras`, backend MJPEG — fine during teleop).
Right 1/3 stacked cards: **Teleoperate** (Start/Stop; port the exact start logic from
`RobotConfigManager.handleTeleop` — `/move-arm` request incl. bimanual + motor_power —
into `components/collect/useTeleopSession.ts`; stop via `/stop-teleoperation`;
status line reflects `/teleoperation-status`), **Record dataset** (dataset picker =
`useSelectedDataset` + `useDatasets` + create-dataset dialog; task prompt, episodes,
episode/reset seconds, cameras summary chip + "Configure cameras" opening existing
`CameraConfiguration` in a dialog; Start recording = build `recordingConfig` exactly
like `Landing.tsx:284-314`, release previews first, `navigate('/recording',
{state:{recordingConfig}})`), **Dataset card** (existing `DatasetInfoCard` +
merge/create dialogs — upload/publish parity lives here). Handoff banner at top when
`location.state?.completedDataset` (set by W5's Recording exit): "N episodes recorded →
Train on this dataset" → `/training`. Disable teleop/record with a visible reason when
`!selectedRecord` or `!selectedRecord.is_clean`.

### W3 — Train & Deploy (`pages/TrainDeploy.tsx`, `components/train/*` new; owns
re-organizing `pages/Training.tsx` and re-skinning `components/jobs/JobsSection.tsx` +
`jobs/*.tsx` cards)
Design: `reference-design/train.html`. `/training` (no jobId) renders TrainDeploy:
page head + "New training run" button toggling a config panel = extract
ConfigurationMode out of `Training.tsx` into `components/train/TrainingConfigPanel.tsx`
(move code, do not rewrite logic: policy select, dataset select, local/hf_cloud
runner, hardware flavors, extras preflight, HF auth banner). Below: `JobsSection`
re-skinned to the design's card language (status badges ok/err/brand-dot, mono meta,
checkpoint select + primary "▶ Test on <selectedRobot>" which opens the existing
`InferenceModal`; Continue/Resume/Download ghost buttons; Models column cards with
Run/Fine-tune). All JobsSection logic (lineage nesting, lazy hub import, search,
websocket refresh) is untouchable. `Training.tsx` keeps ONLY monitoring mode
(`:jobId`), re-skinned: mono metrics, progress bar `bg-primary`, logs panel `.mono`.

### W4 — Market (`pages/Market.tsx`, `components/market/*` — new files only)
Design: `reference-design/market.html`. Real data, marketplace-lite: tabs All /
Datasets / Models. Datasets = `listDatasets()` filtered to `source: 'hub'|'both'`
(card: repo id, episode count if local info available, "Use in Collect" → select via
`useSelectedDataset` + toast). Models = Hub models from the existing jobs API
(`listHubJobs` / hub models listing used by JobsSection's model import; card: repo id,
"Import" → existing `ImportModelModal` flow, imported ⇒ "Imported ✓" ghost). Search
input filters client-side. `.media-slot` thumbnails with mono pill label (task
preview). Footnote: installed items appear in Collect / Train & Deploy.

### W5 — Live screens (owns `pages/Recording.tsx`, `pages/Inference.tsx`)
Designs: `reference-design/record-live.html`, `inference-live.html`. Re-skin ONLY
(the poll/phase/audio/safety logic is untouchable). Recording: back link "← Collect
(discard)", big mono elapsed timer, Chakra-free big "Episode N / M", phase badge
(`● recording` brand / `resetting` warn), episode progress, bottom control bar
(Re-record ghost, Skip ahead outline, Finish early primary). NO camera previews —
render per-camera status chips instead. On finish/stop navigate to
`/collect` with `state.completedDataset` (replaces the old `/upload` hop; `/upload`
route stays for parity but is no longer the default exit). Inference: back link,
"Running — <policy>" head, mono session line (checkpoint · robot · elapsed), progress
bar, Session card, Pause?—no (no backend for pause; omit), "Stop run" destructive →
confirm dialog (existing) → `/training`.

## Foundation (already in place when workers start)

- zinc tokens in `index.css` (HSL triples, light+dark), Inter fonts, radius 0.5rem;
  `.media-slot` utility; notch/grid-bg neutralized; `button.tsx`/`card.tsx`/
  `status-pill.tsx`/`eyebrow.tsx` re-mapped to zinc equivalents.
- `components/shell/StageLayout.tsx`, `RobotsSidebar.tsx`, `StageDock.tsx`.
- `components/home/BoothHero.tsx` (React port of the booth SVG animation).
- `App.tsx` routes per the table (worker pages stubbed until filled).

## Verification gates (fable runs)

grep sweeps (no hex brand orange, no chakra/space-grotesk, no new getUserMedia) →
`tsc --noEmit` + `npm run build` + `eslint` → `pytest` (599 baseline) → backend smoke
on SAFE endpoints only (`/robots` shows desk_isaac, `/available-ports` shows the two
usbmodem ports, `/available-cameras`, `/camera-preview/{i}`, `/datasets`, `/jobs`) →
codex computer-use user journey in real Chrome (do-not-click: Start teleop, Start
recording, Run/Test buttons, Calibrate, Detect/wiggle) → fix findings → re-verify.
