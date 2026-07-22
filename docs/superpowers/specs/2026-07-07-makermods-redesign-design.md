# MakerLab × MakerMods redesign — design spec

Date: 2026-07-07 · Branch: `redesign` · Status: approved by Isaac (visual preview reviewed in browser)

## Goal

Rebuild the MakerLab frontend's entire visual layer on the **MakerMods Design System** (claude.ai/design project `0bb9c567-88b8-4632-a6a4-b14687dfbc4f`, "MakerMods Design System") while preserving **100% of existing functionality**. Every route, dialog, hotkey, WebSocket stream, polling loop, and lifecycle behavior in the current app must survive unchanged; only presentation changes.

## Decisions (settled with user)

| Decision | Choice |
| --- | --- |
| Branding | Product stays **MakerLab** (MakerLab is a MakerMods product); logo uses the **MakerMods mark** from DS `assets/`, wordmark text "MAKERLAB" in Chakra Petch |
| Theme | **Light default** (paper `#FAFAFA`), full dark theme, toggle in shell; ThemeContext default changes from `system` to `light` |
| Scope | **Everything**: all 9 routes, all dialogs, all ~60 feature components, plus new app shell. EditDataset stays a stub but gets shell styling |
| Approach | **Foundation-first campaign**: Phase 1 foundation (tokens/fonts/primitives/shell) → Phase 2 agent fan-out per page → Phase 3 inspection |
| Functionality | **Hard constraint: no functionality removed or altered.** Re-skin only; all props, handlers, state, effects, API calls, hotkeys, audio cues, guards stay |

## 1 · Token layer

- Keep shadcn semantic CSS variable names (`--background`, `--foreground`, `--card`, `--primary`, `--muted`, `--destructive`, `--border`, `--ring`, `--radius`, …) so `ui/` primitives keep working. Re-value them from MakerMods tokens in `src/index.css`:
  - Light: paper `#FAFAFA` (background), surface `#FFFFFF` (card/popover), ink `#0A0A0A` (foreground/primary), line `#E5E5E5` (border), line-2 `#D4D4D4` (input), mute `#737373` (muted-foreground), danger `#DC2626` (destructive), ring = orange.
  - Dark (`.dark`): ink `#F5F5F5`, paper `#0A0A0A`, surface `#141414`, line `#262626`, mute `#A3A3A3`, danger `#EF4444`, orange `#FF7A40`.
- Add brand tokens: `--brand` (orange `#FF6B2C` / `#FF7A40`), `--brand-hover` (`#E55418` / `#FF9460`), `--brand-tint` (`#FFE3D2` / `#3A1F12`), `--brand-foreground` (always `#0A0A0A`), `--ok`, `--warn`, `--info`, notch clip-path vars (`--notch-sm/md/lg`), shadows (`--shadow-1`, `--shadow-2`).
- `tailwind.config.ts`: map `fontFamily.display/body/mono`, `boxShadow['1']/['2']`, brand/ok/warn/info colors, radius scale (sm 2px · DEFAULT 6px · lg 12px · xl 16px ceiling). Remove `sidebar-*` token group (unused template leftover).
- New utilities (in `index.css` `@layer utilities`): `.grid-bg` (24px graph-paper via `--border`), `.notch-sm/.notch-md/.notch-lg`, `.eyebrow` (Chakra Petch 11px, 600, `0.14em` tracking, uppercase, muted).

### Banned patterns (grep-enforced at inspection)

- Raw color utilities: `bg-black`, `bg-white`, `text-white`, `text-black`, `gray-*`, `slate-*`, `zinc-*`, `neutral-*`, raw `green-*`/`red-*`/`yellow-*`/`blue-*` for UI chrome (functional states go through `ok/warn/danger/info` tokens). Exception: none in `src/pages` and `src/components`; the 3D canvas and camera feeds are content, not chrome.
- Gradients (except the single highlighter-underline motif), glassmorphism, glow/colored shadows, translateY hover lifts, emoji in UI copy, Title Case in headings/buttons.
- Orange: **one static orange anchor per screen** (primary CTA *or* key stat). Orange focus rings and `[ NEW ]`-style stencil tags are allowed on top of that (per DS); orange is never a body-text color, full-bleed background, or state color. Notch: **max one stencil-cut element per screen** (badges/tags excluded).

## 2 · Typography & icons

- Families: **Chakra Petch** 500/600/700 (display: page titles, headings, buttons, eyebrows, big numbers), **Space Grotesk** 400/500/700 (body), **JetBrains Mono** 400/700 (ports, serial IDs, timers, joint values, logs, file paths, hotkey hints, dataset stats).
- **Self-hosted via `@fontsource/*` packages** imported in `main.tsx` — no Google CDN; the app must work on offline lab networks.
- Type scale per DS (`--t-eyebrow` 11px → `--t-h1` 48px; `display`/`mega` reserved for marketing-style moments, generally unused in product UI).
- Icons: `lucide-react` (already used), 24px default / 16px inline, stroke 1.5, `currentColor` only, never filled, no two-tone.

## 3 · App shell & navigation

New `src/components/shell/AppShell.tsx` used by every route:

- Sticky ~52px top bar, `backdrop-filter: saturate(140%) blur(8px)` over `background/80` — **the only blur in the app**. 1px bottom border.
- Left: MakerMods mark (from DS assets, copied to `frontend/public/` or `src/assets/`; light + dark variants) + "MAKERLAB" wordmark (Chakra Petch 700, letter-spaced). Links to `/`.
- Center/left slot: standardized `← back` ghost affordance on non-landing pages (replaces 5 hand-rolled headers); optional page eyebrow-title.
- Right slot: page-provided actions, HF auth chip, `UpdateNotice`, theme toggle (sun/moon, persists via existing ThemeContext).
- **Session screens** (Teleoperation, Recording, Inference): shell hosts the live `StatusPill` (phase + episode counter) and the session Stop control so state is visible while looking at the robot. The existing exit/stop lifecycle handlers move into shell-slot props — **behavior unchanged**.
- Hub-and-spoke navigation retained; no route menu. Footer stays Landing-only. Container: `max-w-[1440px]` product-wide.

## 4 · Component system

Restyle every `src/components/ui/` primitive on the new tokens; extend variant APIs (cva):

- **Button**: `default` (ink fill), `secondary` (surface + line border), `ghost`, `destructive` (outline danger), `link`, plus new `brand` (orange fill, dark text) and `notch` (stencil-cut, ink or brand fill). Sizes sm/default/lg/icon. Chakra Petch 600. Hover = opacity/background per DS; press = `scale(0.98)`, 80ms; no lifts.
- **Badge**: `default` (ink pill), `outline`, dot-status variants `ok`/`warn`/`danger` (12% tint bg + colored text + 6px dot, optional pulse), and `stencil` (`[ LABEL ]` clip-path tag, uppercase 700, 0.16em).
- **Card**: `default` (surface, 1px line, shadow-1), `flat`, `notch`, `inverted` (ink bg — reserved for terminal/log panels).
- **Input/Select/Textarea/Label**: 4px radius, line-2 border, focus = 2px orange outline; `Label` = uppercase Chakra Petch 11px; add a `mono` styling convention for technical values. Error = danger border + mono danger caption.
- **Dialog/AlertDialog**: 12px radius, shadow-2, solid `ink/60` overlay (no blur), Chakra Petch titles.
- **Toast**: consolidate to **one** system — keep the Radix toast (currently mounted), restyle to DS; remove `sonner` component + dependency usage. (If any code imports sonner's `toast`, migrate call sites to `use-toast` — call-site behavior preserved.)
- New primitives in `ui/`: `Eyebrow`, `StatNumber` (display-font number + mono label), `StatusPill` (session phases; REC uses danger red dot — red = recording convention, orange is never a state color), `PageHeader`.
- Progress bars: 4px, line track, ink fill (danger only for destructive contexts).
- Cleanup: remove `next-themes` dep (unused), remove sidebar tokens, standardize on semantic tokens in all feature components.

## 5 · Per-page treatment

All pages wrapped in `AppShell`; all existing logic, polling, hotkeys, and dialogs preserved.

- **Landing**: grid-bg hero strip (robot status line in mono), eyebrow-labeled sections `[ Robot ]` / `[ Dataset ]` / `[ Models ]`, DS cards, jobs as bordered mono rows with status badges. Orange: the record CTA (notch+brand).
- **Teleoperation**: viewer + cameras framed as bordered surface panels (dark 3D canvas is content); mono joint readouts; command bar as bottom toolbar; shell hosts session status + Done.
- **Recording**: centered notched card, mono timer (~56px), phase StatusPill in shell, ink progress bar, orange `End episode →` primary, ghost re-record/mute, mono hotkey hints. Audio cues unchanged.
- **Inference**: same HUD family as Recording (shared components where practical without behavior change).
- **Training**: config mode = two-column form, uppercase labels, mono for numeric/technical inputs; monitoring mode = 4 stat tiles (`StatNumber`, step count is the orange), inverted-card mono log terminal, checkpoint dropdown restyled.
- **Calibration**: two-column; mono-numbered stepper for calibration steps; ranges table in mono; motor-power slider and all dialogs restyled on tokens.
- **Upload**: shell + centered card confirmation.
- **EditDataset**: shell + `[ Under construction ]` stencil placeholder card.
- **NotFound**: shell + `[ 404 ]` stencil treatment, ghost link home.

## 6 · Execution plan

**Phase 1 — Foundation (Fable 5, this session):** tokens, tailwind config, fonts, utilities, all `ui/` primitives, `AppShell`, Logo swap, theme default. App must build and run after this phase with pages functional-but-mixed.

**Phase 2 — Page fan-out (subagents, parallel):** one agent per page/feature-area (Landing+landing components, Teleop+control, Recording+Inference, Training+jobs, Calibration, misc pages) doing the mechanical re-skin against this spec's rules.

- Claude workers: `model: 'opus'` (Opus 4.8) via Agent/Workflow.
- gpt-5.5 workers (optional, per user guide): thin Claude wrapper agent (`model: 'sonnet'`, `effort: 'low'`) whose prompt writes a self-contained codex prompt, runs `codex exec` via Bash, returns the report; label `gpt-5.5:<task>`; explicit Bash timeout or background+poll (codex can exceed 10-min default); parallel implementation agents use `isolation: 'worktree'`.
- Every worker prompt includes: the banned-pattern list, the variant APIs, "do not touch logic/props/handlers/effects/API calls", and the orange/notch budget.

**Phase 3 — Inspection (Fable 5):**

1. Grep sweeps for banned classes.
2. `npm run build` + lint clean.
3. Visual: dev server + browser screenshots of **every route in both themes**, checked against DS rules.
4. **Behavioral: not just visual.** Drive the real app (`makerlab --dev` or vite + backend) and verify flows still work — navigation, dialogs opening, teleop start/stop lifecycle, recording controls + hotkeys, training config form, calibration stepper, theme toggle persistence. Use browser automation and `codex exec -s read-only` with self-contained prompts for independent behavioral review; `codex review` for diff review of worker output.
5. Punch-list fixes applied by me.

## Non-goals

- No backend/API changes. No route changes. No new features. No removal of "unused" props or dead-looking code paths inside feature components (out of scope — too risky for behavior preservation). `frontend/dist/` rebuild handled by CI on main (or locally if we want to eyeball the production bundle).

## Risks & mitigations

- **Behavior regression while re-skinning 2000-line pages** (Calibration, Recording): workers are instructed to change only JSX classNames/markup structure and imports of primitives, never hooks/handlers; codex review + behavioral pass catches drift; per-area worktree isolation prevents cross-agent collisions.
- **Dark-canvas screens looking off in light theme**: viewer/camera panels are framed as bordered content panels — checked explicitly in inspection.
- **Font bundle size**: subset weights (3 families × 2-3 weights) via fontsource; acceptable for a local app.
