# Episode viewer: boundary-aware nav buttons + keyboard shortcuts — design spec

Date: 2026-07-27 · Branch: `feat/dataset-viewer-window` · Status: approved by user

## Goal

`EpisodeViewer` (inline in `frontend/src/components/dialogs/DatasetDetailDialog.tsx`) has Prev/Next episode buttons and a Play/Pause button. Two gaps:

1. Prev/Next are only disabled when there's no episode at all (`disabled={!episode}`), never at the actual start/end of the episode list — so a user can't tell by looking whether there's anywhere left to go.
2. No keyboard shortcuts — every transport action requires a mouse click.

## 1 · Boundary-aware disabling

Compute the current index once (`episodes.findIndex((e) => e.episode_index === selectedEpisode)`) and derive:

- `hasPrev = currentIndex > 0`
- `hasNext = currentIndex >= 0 && currentIndex < episodes.length - 1`

Prev button: `disabled={!episode || !hasPrev}`. Next button: `disabled={!episode || !hasNext}`.

No new styling needed — the shared `Button` component (`components/ui/button.tsx`) already applies `disabled:opacity-50 disabled:pointer-events-none`, which is exactly the "transparent, can't click" affordance requested.

## 2 · Keyboard shortcuts

Mirror the existing convention in `RecordingSessionDialog.tsx` (`window`-level `keydown` listener, guarded against text-input focus, `preventDefault()` on handled keys):

- `ArrowLeft` → `gotoEpisode(-1)`
- `ArrowRight` → `gotoEpisode(1)`
- `Space` → `handlePlayPause()`

Implementation: a `useEffect` inside `EpisodeViewer` that adds/removes the listener on mount/unmount — no extra "is this active" flag is needed because `EpisodeViewer` itself is only mounted while the dialog is open and episodes are loaded (see `DatasetDetailDialog`'s conditional render).

Guard, copied from the existing convention: skip entirely if `e.target` is an `INPUT`, `TEXTAREA`, or `isContentEditable` — relevant here because the dialog's side panel (`DatasetInfoCard`) contains a tag-input field. `gotoEpisode` already no-ops safely when there's nothing at `episodes[i + delta]`, so the handler doesn't need to duplicate the `hasPrev`/`hasNext` check.

## 3 · Testing

Per `CLAUDE.md`, this is a pure frontend UI change — no unit tests to add (no pure-helper logic beyond what's already inline). Verify manually via `makerlab --dev`: open a dataset with 3+ episodes, confirm Prev is greyed out on episode 0 and Next greyed out on the last episode, and confirm ArrowLeft/ArrowRight/Space work while focus is anywhere in the dialog except the tag-input field.
