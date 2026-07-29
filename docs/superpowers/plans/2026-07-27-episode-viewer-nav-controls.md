# Episode Viewer Nav Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the episode viewer's Prev/Next buttons visually reflect list boundaries, and add ArrowLeft/ArrowRight/Space keyboard shortcuts for episode navigation and play/pause.

**Architecture:** Both changes live entirely inside the existing `EpisodeViewer` component in `frontend/src/components/dialogs/DatasetDetailDialog.tsx`. No new files, no backend changes.

**Tech Stack:** React + TypeScript, existing `Button` component (`components/ui/button.tsx`) which already applies `disabled:opacity-50 disabled:pointer-events-none`.

## Global Constraints

- Frontend checks only: `npm run lint`, `npx tsc --noEmit` from `frontend/` (per `CLAUDE.md`). No `npm run build` — it can corrupt committed LFS assets elsewhere in the repo (unrelated to this change, but don't run it).
- No unit tests to add — this is inline UI logic, not a pure helper (per `CLAUDE.md`'s testing rules and the design spec's §3).
- Manual verification via `makerlab --dev`.

---

### Task 1: Boundary-aware Prev/Next disabling + keyboard shortcuts

**Files:**
- Modify: `frontend/src/components/dialogs/DatasetDetailDialog.tsx` (inside `EpisodeViewer`, roughly lines 246-345)

**Interfaces:**
- Consumes: existing `episodes: EpisodeSummary[]`, `selectedEpisode: number`, `gotoEpisode(delta: number)` (already defined at line 246), `handlePlayPause()` (already defined at line 163) — no signature changes to any of these.
- Produces: no new exports; this is a self-contained behavior change within the component.

- [ ] **Step 1: Add boundary computation next to `gotoEpisode`**

In `EpisodeViewer`, immediately after the existing `gotoEpisode` function (around line 250), add:

```tsx
  const currentIndex = episodes.findIndex((e) => e.episode_index === selectedEpisode);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < episodes.length - 1;
```

- [ ] **Step 2: Wire the computed flags into the Prev/Next buttons' `disabled` props**

Change (around line 322):
```tsx
          onClick={() => gotoEpisode(-1)}
          disabled={!episode}
          aria-label="Previous episode"
```
to:
```tsx
          onClick={() => gotoEpisode(-1)}
          disabled={!episode || !hasPrev}
          aria-label="Previous episode"
```

And (around line 341):
```tsx
          onClick={() => gotoEpisode(1)}
          disabled={!episode}
          aria-label="Next episode"
```
to:
```tsx
          onClick={() => gotoEpisode(1)}
          disabled={!episode || !hasNext}
          aria-label="Next episode"
```

- [ ] **Step 3: Add the keyboard shortcut effect**

Add a new `useEffect` inside `EpisodeViewer`, placed after the existing prefetch `useEffect` (after line 151, before the `setPlaying(false)` reset effect at line 153) — mirrors the guarded-`window`-listener convention already used in `frontend/src/components/recording/RecordingSessionDialog.tsx:556-582`:

```tsx
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        gotoEpisode(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        gotoEpisode(1);
      } else if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        handlePlayPause();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [gotoEpisode, handlePlayPause]);
```

Note: `gotoEpisode` and `handlePlayPause` are plain `const` closures re-created each render (not wrapped in `useCallback`), so this effect re-subscribes every render — that's fine, it's a cheap `addEventListener`/`removeEventListener` pair, and matches the codebase's existing tolerance for this pattern elsewhere.

- [ ] **Step 4: Type-check and lint**

Run from `frontend/`:
```bash
npx tsc --noEmit
npm run lint
```
Expected: no new errors beyond whatever pre-existing baseline `CLAUDE.md` notes (there shouldn't be any pre-existing errors touching this file — if there are, confirm they're unrelated to this change before proceeding).

- [ ] **Step 5: Manual verification via `makerlab --dev`**

Start the dev server, open a dataset with 3+ local episodes in the detail dialog:
- Confirm the Prev button is greyed out (visually dimmed, unclickable) on episode index 0, and re-enables after navigating forward.
- Confirm the Next button is greyed out on the last episode, and re-enables after navigating back.
- With focus somewhere in the dialog (not the tag-input field in the info card), press ArrowLeft/ArrowRight and confirm episode navigation; press Space and confirm play/pause toggles.
- Click into the tag-input field (in `DatasetInfoCard`, right panel) and confirm typing a literal space or arrow keys there does NOT trigger episode navigation or play/pause — it types normally.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/dialogs/DatasetDetailDialog.tsx
git commit -m "$(cat <<'EOF'
feat(datasets): boundary-aware episode nav buttons + keyboard shortcuts

Prev/Next now grey out at the actual start/end of the episode list
instead of only when no episode is selected at all. Adds ArrowLeft/
ArrowRight to navigate episodes and Space to play/pause, guarded
against text-input focus to match the existing convention in
RecordingSessionDialog.
EOF
)"
```
