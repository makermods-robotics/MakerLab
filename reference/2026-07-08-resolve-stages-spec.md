# MakerLab redesign v2 — "Resolve stages" wireframe spec

Synthesis after Isaac reviewed concepts A/B/C: a **DaVinci Resolve-style staged
workspace**. One app, a fixed stage bar, each stage a purpose-built workspace.
Deliverable: one plain-HTML wireframe `reference/wireframes/concept-d-resolve.html`
(same grayscale conventions as v1 spec `2026-07-08-flow-redesign-wireframes.md`).

## Direction (Isaac's words, distilled)

- Like DaVinci Resolve: distinct stages of work, switched via a persistent stage bar.
- Stage 1 = robot config, like **opening a repo in Cursor**: choose/open a robot
  config first, with a settings panel on the side to configure it (robots need
  configuration). After opening, the robot hides in a corner chip + gear —
  out of sight, always accessible.
- Stage 2 = **teleop + recording on the SAME page** — users practice teleop before
  recording. The 3D arm animation (URDF viewer) lives on this page during teleop.
  Starting a dataset recording goes to a **dedicated full page** like current
  MakerLab: timer, episode counter, re-record episode, skip ahead.
- Stage 3 = **train + deploy on the SAME page** — jobs and models as cards
  (like the current MakerLab jobs/models board). Key mechanic: **deploy a
  checkpoint to test while training is still running** (per-job step dropdown + ▶).
- Stage 4 = **skills marketplace** — datasets AND finetuned policies for specific
  tasks, browse/install.

## Borrowed from leLab issue #49 (skills-first redesign)

Adopt:
- **"Skill" = task bundle (dataset + its trained policy versions).** The unifying
  noun for stage 3 grouping and the marketplace inventory.
- **Version history grouped by source dataset**: retraining after dataset edits
  keeps old versions runnable. Stage 3 groups job/model cards under their skill.
- **Plain-words glosses** on stage labels and empty states (e.g. "Record — show the
  robot the task by doing it").
- Dataset library with per-episode player: future Collect-stage drawer; out of
  wireframe scope, mention as a NOTE only.

Do not adopt: chat assistant surface; per-episode destructive dataset rewrite.

## Global chrome (all stages)

- Top bar: logo left; right corner = robot chip `● <name> ▾` + gear ⚙ (identical
  behavior to v1 spec: dropdown with Single|Bimanual tabs of saved configs,
  status dots, "+ New config"; gear opens Robot Settings slide-over with ports +
  Detect (wiggle), calibration per arm + Calibrate now, canonical cameras,
  motor power). Chip/gear hidden on live session pages.
- **Bottom stage bar, DaVinci Resolve style**: centered row of 4 stage buttons,
  always visible except during live sessions:
  `[① RIG] [② COLLECT] [③ TRAIN & DEPLOY] [④ MARKET]`
  Each has a one-word label + tiny plain-words gloss underneath
  (Rig "set up your robot" · Collect "teach by demonstration" ·
  Train & Deploy "turn demos into skills" · Market "get skills & data").
  Current stage highlighted. Stages never lock; unmet prerequisites show a
  one-line reason inside the stage itself.

## Stage ① RIG — open a robot like a repo

- First-run / no-robot state doubles as the stage content: a **project-picker**
  (Cursor open-repo feel). Left: "Recent rigs" list — cards with name, mode badge
  (single/bimanual), status line (calibrated · ports OK · 2 cameras), last-used.
  Plus "+ New rig" card and an "Import from Market" ghost row.
- Selecting a rig card opens a **settings drawer on the right side** (not a new
  page): ports per arm with Detect (wiggle), calibration file per arm +
  Calibrate now, cameras, motor power, Save + **"Open rig →"** primary action.
- "Open rig" activates it: app jumps to COLLECT, robot collapses into the corner
  chip. NOTE on wireframe: returning to RIG later shows the same picker
  (active rig highlighted) — switching rigs re-targets every stage instantly.

## Stage ② COLLECT — teleop practice + record setup, one page

- Layout: left ~2/3 = **3D arm visualizer panel** (URDF placeholder box, live
  during teleop) with a camera-feeds strip beneath (front / wrist placeholders).
  Right ~1/3, stacked:
  1. **Teleoperate card** — Start/Stop practice toggle; status line (joints
     streaming · nothing saved). Practicing animates the 3D panel (annotate).
  2. **Record card** — dataset picker + "+ New", task prompt text, episodes
     target, episode/reset seconds; primary **"Start recording"**.
- NOTE: teleop-before-record is the expected rhythm; both cards share the same
  hardware session, no reconnect between practice and record.
- **Start recording → dedicated live-record page** (stage bar + chrome hidden):
  big timer, `Episode 7 / 10`, phase indicator (recording / resetting), camera
  feeds, controls: **Re-record episode · Skip ahead (next episode) · Stop early ·
  mute**. Mirrors current MakerLab Recording page experience.
- After stop/finish → back to COLLECT with a **handoff banner**: "pick-cube
  +8 episodes → Train on this dataset" (deep-links to stage ③ pre-filled).

## Stage ③ TRAIN & DEPLOY — one page, skills-first cards

- Page head: skill/dataset selector + **"New training run"** (policy type
  dropdown + essentials, collapsed until clicked).
- Content grouped by **skill** (source dataset), each group a section with
  version history; inside, two card families side by side (Isaac's screenshot
  is the reference):
  - **Jobs column** — collapsible "Local jobs (1)" / "Online jobs (6)" groups.
    Job card: status chip (Done ✓ / Failed ✕ / Running ▸), policy + name,
    "ended 2h ago", and the key row: **step/checkpoint dropdown + ▶ test** —
    deploy that checkpoint to the active rig even while the job is still
    running — plus Continue/Resume and Download. Pencil/trash as small icons.
  - **Models column** — uploaded/finetuned model cards: "Uploaded" badge,
    repo id line, **▶ Run** + **Fine-tune** buttons, open-on-Hub + delete icons.
- NOTE: versions under one skill stay runnable after retrains (issue #49).
- **▶ (test checkpoint or Run model) → dedicated live-inference page** (chrome
  hidden): model + checkpoint + target rig line, camera feeds, elapsed/progress,
  single Stop. Exiting returns to stage ③.

## Stage ④ MARKET — skills marketplace

- Search field + filter tabs: **All · Skills · Datasets · Policies**.
- Grid of cards, each a **task-first skill**: task name ("fold-sock"), what it
  contains (dataset badge `120 ep` and/or policy badges `ACT` `SmolVLA`),
  robot compatibility (SO-101 single/bimanual), author, downloads;
  actions: **Get** (installs into your library) / "Installed ✓".
- A NOTE box: installed datasets appear in COLLECT's dataset picker; installed
  policies appear in stage ③ Models — the market feeds the other stages.

## Wireframe screen states (switcher strip, same conventions as v1)

1. `rig-picker` — first-run project picker (no rig open; chip area empty)
2. `rig-settings` — picker with a rig selected + right settings drawer open
3. `collect` — teleop + record page (rig open, chip in corner)
4. `live-record` — dedicated recording page, chrome hidden
5. `post-record` — collect page with handoff banner
6. `train-deploy` — skills-grouped jobs + models cards
7. `live-inference` — dedicated run page, chrome hidden
8. `market` — marketplace grid
9. `robot-dropdown` — any stage with chip dropdown open (Single|Bimanual tabs)

Hard rules: identical to v1 spec (self-contained file, grayscale only, system
font, vanilla JS switcher, dashed NOTE boxes, < ~800 lines, opens via file://).
Keep v1 files untouched; this is an additional wireframe.
