# Local datasets in the dataset dropdown — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface local LeRobot datasets in every dataset dropdown, grouped separately from Hub datasets, and route Landing-page picks of local datasets to the Upload page.

**Architecture:** Backend extends `GET /datasets` to merge local-cache datasets (detected by `meta/info.json`) with the existing Hub listing into a single response with a `source` field. Frontend types/components add a Local / Hugging Face split, and Landing branches its click action on `source`.

**Tech Stack:** FastAPI + Pydantic backend, React + Vite + Tailwind frontend, `cmdk`-based `Command` primitive, react-router, LeRobot dataset cache at `~/.cache/huggingface/lerobot/` (or `$HF_LEROBOT_HOME`).

**Repo conventions:** No test suite, no linter, no build step (per [CLAUDE.md](../../../CLAUDE.md)). Validation is manual: run `lelab --dev`, hit endpoints with `curl`, exercise the UI in a browser. TDD-style "write failing test" steps are replaced with "write code, then verify with curl/UI" steps. **Frontend `lelab --dev` serves Vite directly — no `npm run build` needed during iteration.**

**Spec:** [docs/superpowers/specs/2026-05-06-local-datasets-in-dropdown-design.md](../specs/2026-05-06-local-datasets-in-dropdown-design.md)

---

## File Structure

**Backend (`app/`):**
- Modify [`app/dataset_browser.py`](../../../app/dataset_browser.py) — add `list_local_datasets()` and `list_all_datasets()` (merge function).
- Modify [`app/main.py`](../../../app/main.py) — `GET /datasets` calls `list_all_datasets()`.
- Modify [`app/runners/hf_cloud.py`](../../../app/runners/hf_cloud.py) — TODO comment about local-dataset handling.

**Frontend (`frontend/src/`):**
- Modify [`frontend/src/lib/replayApi.ts`](../../../frontend/src/lib/replayApi.ts) — extend `DatasetItem`, add `DatasetSource`.
- Modify [`frontend/src/components/replay/DatasetCombobox.tsx`](../../../frontend/src/components/replay/DatasetCombobox.tsx) — split into Local + Hugging Face groups, render "on Hub" badge.
- Modify [`frontend/src/components/landing/DatasetPicker.tsx`](../../../frontend/src/components/landing/DatasetPicker.tsx) — same group split; `onPickExisting` prop signature changes to receive the full `DatasetItem`.
- Modify [`frontend/src/pages/Landing.tsx`](../../../frontend/src/pages/Landing.tsx) — branch click action on `source`.

No new files. No file splits — each existing file's responsibility is unchanged, only extended.

---

## Task 1: Backend — `list_local_datasets()`

**Files:**
- Modify: [`app/dataset_browser.py`](../../../app/dataset_browser.py)

- [ ] **Step 1: Add the local-listing function**

Replace the entire contents of [`app/dataset_browser.py`](../../../app/dataset_browser.py) with:

```python
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, whoami
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

logger = logging.getLogger(__name__)


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _is_dataset_dir(path: Path) -> bool:
    """A directory is a LeRobot dataset iff <dir>/meta/info.json exists."""
    try:
        return (path / "meta" / "info.json").is_file()
    except OSError:
        return False


def _dir_mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Scan the LeRobot cache for local datasets (dirs containing meta/info.json).

    Walks one level deep: a top-level dataset dir is recorded as "<name>"; if a
    top-level dir is not itself a dataset, each subdir that is a dataset is
    recorded as "<top>/<sub>". Does not descend further.
    """
    root = _lerobot_cache_root()
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    try:
        top_entries = list(root.iterdir())
    except OSError as e:
        logger.warning(f"Could not read LeRobot cache root {root}: {e}")
        return []

    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _is_dataset_dir(top):
            out.append({
                "repo_id": top.name,
                "last_modified": _dir_mtime_iso(top),
                "private": False,
            })
            continue

        # Not a dataset itself — descend one level.
        try:
            sub_entries = list(top.iterdir())
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if not sub.is_dir():
                    continue
            except OSError:
                continue
            if _is_dataset_dir(sub):
                out.append({
                    "repo_id": f"{top.name}/{sub.name}",
                    "last_modified": _dir_mtime_iso(sub),
                    "private": False,
                })

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_user_datasets() -> list[dict[str, Any]]:
    try:
        info = whoami()
    except (LocalTokenNotFoundError, HfHubHTTPError, OSError):
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = HfApi()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for ds in api.list_datasets(author=author, filter="LeRobot", limit=200):
                if ds.id in seen:
                    continue
                seen.add(ds.id)
                out.append({
                    "repo_id": ds.id,
                    "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                    "private": bool(getattr(ds, "private", False)),
                })
        except HfHubHTTPError as e:
            logger.warning(f"list_datasets({author}) failed: {e}")

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_all_datasets() -> list[dict[str, Any]]:
    """Merged listing: Hub datasets + local cache, with `source` field.

    A repo_id present in both lists is collapsed to one entry with
    source="both" and last_modified set to the more recent of the two.
    """
    hub = list_user_datasets()
    local = list_local_datasets()

    merged: dict[str, dict[str, Any]] = {}
    for item in hub:
        merged[item["repo_id"]] = {**item, "source": "hub"}
    for item in local:
        rid = item["repo_id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # Keep the newer timestamp; ISO strings sort lexically.
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = {**item, "source": "local"}

    out = list(merged.values())
    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out
```

- [ ] **Step 2: Verify the function works against your real cache**

Run from the repo root:

```bash
python -c "from app.dataset_browser import list_local_datasets, list_all_datasets; import json; print(json.dumps(list_local_datasets()[:5], indent=2)); print('---'); print(json.dumps(list_all_datasets()[:5], indent=2))"
```

Expected: a JSON array of dicts, each with `repo_id`, `last_modified`, `private`, and (for `list_all_datasets`) `source`. Junk dirs without `meta/info.json` (e.g. `cvkjln`, `dgf`) MUST NOT appear. Real datasets like `Metabolik/cam_test` SHOULD appear if your local cache has them.

If it errors, fix before moving on. If it returns `[]` because your cache has no valid datasets, that's also acceptable — but verify by `ls ~/.cache/huggingface/lerobot/Metabolik/cam_test/meta/info.json`.

- [ ] **Step 3: Commit**

```bash
git add app/dataset_browser.py
git commit -m "feat(datasets): list local LeRobot cache datasets"
```

---

## Task 2: Backend — wire merged listing into `GET /datasets`

**Files:**
- Modify: [`app/main.py`](../../../app/main.py) (lines 337-340)

- [ ] **Step 1: Switch the endpoint to the merged function**

In [`app/main.py`](../../../app/main.py), find:

```python
@app.get("/datasets")
def datasets_list():
    """List datasets the logged-in HF user owns or shares with their orgs."""
    return dataset_browser.list_user_datasets()
```

Replace with:

```python
@app.get("/datasets")
def datasets_list():
    """List datasets available to the user — Hub-owned + local cache.

    Each entry carries a `source` field: "local", "hub", or "both".
    """
    return dataset_browser.list_all_datasets()
```

- [ ] **Step 2: Start the dev server and curl the endpoint**

In one terminal:

```bash
lelab --dev
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. In another terminal:

```bash
curl -s http://localhost:8000/datasets | python -m json.tool | head -40
```

Expected: a JSON array. Every item has a `source` key with value `"local"`, `"hub"`, or `"both"`. If you have local datasets, at least one entry has `"source": "local"` (or `"both"`).

If `whoami()` fails (no HF token), only `"local"` entries appear. To test the no-token path, temporarily move your token:

```bash
mv ~/.cache/huggingface/token ~/.cache/huggingface/token.bak
curl -s http://localhost:8000/datasets | python -m json.tool | head -20
mv ~/.cache/huggingface/token.bak ~/.cache/huggingface/token
```

Expected with token moved: only `"source": "local"` entries (or empty array if cache is empty).

- [ ] **Step 3: Stop `lelab` (Ctrl-C in the dev-server terminal) and commit**

```bash
git add app/main.py
git commit -m "feat(datasets): merge local + Hub in GET /datasets"
```

---

## Task 3: Backend — TODO comment for cloud-job local-dataset handling

**Files:**
- Modify: [`app/runners/hf_cloud.py`](../../../app/runners/hf_cloud.py)

This is a non-functional change documenting an open question that the spec calls out as out-of-scope.

- [ ] **Step 1: Add the comment near the top of the module**

In [`app/runners/hf_cloud.py`](../../../app/runners/hf_cloud.py), find the module docstring and the line `LEROBOT_IMAGE = "huggingface/lerobot-gpu:latest"` (around line 26). Just after that constant, add:

```python
# TODO(local-datasets): this runner assumes dataset_repo_id resolves on the Hub.
# Local-only datasets selected from the new dataset dropdown will fail at job
# start because the cloud GPU pod can't see ~/.cache/huggingface/lerobot.
# Either upload the dataset before submitting, or refuse the job with a clear
# message. See docs/superpowers/specs/2026-05-06-local-datasets-in-dropdown-design.md.
```

- [ ] **Step 2: Verify Python still imports cleanly**

```bash
python -c "from app.runners import hf_cloud; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add app/runners/hf_cloud.py
git commit -m "docs(hf-cloud): TODO for local-dataset handling in cloud jobs"
```

---

## Task 4: Frontend — extend `DatasetItem` type

**Files:**
- Modify: [`frontend/src/lib/replayApi.ts`](../../../frontend/src/lib/replayApi.ts)

- [ ] **Step 1: Replace the file contents**

Overwrite [`frontend/src/lib/replayApi.ts`](../../../frontend/src/lib/replayApi.ts) with:

```ts
export type DatasetSource = "local" | "hub" | "both";

export interface DatasetItem {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
  source: DatasetSource;
}

type Fetcher = (url: string, options?: RequestInit) => Promise<Response>;

export async function listDatasets(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<DatasetItem[]> {
  const r = await fetcher(`${baseUrl}/datasets`);
  if (!r.ok) throw new Error(`GET /datasets failed: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: exits 0 with no output (no type errors). If errors appear in `DatasetCombobox.tsx`, `DatasetPicker.tsx`, or `Landing.tsx`, that's expected — they're addressed in Tasks 5–7. **Only fail this step if errors appear elsewhere** (e.g. an unrelated file).

If errors are limited to the three files above, continue.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/replayApi.ts
git commit -m "feat(datasets): add source field to DatasetItem"
```

---

## Task 5: Frontend — group split in `DatasetCombobox`

**Files:**
- Modify: [`frontend/src/components/replay/DatasetCombobox.tsx`](../../../frontend/src/components/replay/DatasetCombobox.tsx)

This component is consumed by the **Replay page** and the **Training EssentialsCard**. The change is purely visual grouping — the `onChange(repoId)` callback signature stays the same.

- [ ] **Step 1: Replace the component**

Overwrite [`frontend/src/components/replay/DatasetCombobox.tsx`](../../../frontend/src/components/replay/DatasetCombobox.tsx) with:

```tsx
import React from "react";
import { Check, ChevronsUpDown, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { cn } from "@/lib/utils";
import { DatasetItem } from "@/lib/replayApi";

interface Props {
  datasets: DatasetItem[];
  loading: boolean;
  value: string | null;
  onChange: (repoId: string | null) => void;
}

const REPO_ID_RE = /^[\w.\-]+\/[\w.\-]+$/;

const DatasetCombobox: React.FC<Props> = ({ datasets, loading, value, onChange }) => {
  const [open, setOpen] = React.useState(false);
  const [customMode, setCustomMode] = React.useState(false);
  const [customValue, setCustomValue] = React.useState("");

  const submitCustom = () => {
    const v = customValue.trim();
    if (REPO_ID_RE.test(v)) {
      onChange(v);
      setCustomMode(false);
    }
  };

  const localDatasets = datasets.filter((d) => d.source === "local" || d.source === "both");
  const hubDatasets = datasets.filter((d) => d.source === "hub");

  const renderItem = (d: DatasetItem) => (
    <CommandItem
      key={d.repo_id}
      value={d.repo_id}
      onSelect={() => { onChange(d.repo_id); setOpen(false); }}
      className="text-white aria-selected:bg-gray-700"
    >
      <Check className={cn("mr-2 h-4 w-4", value === d.repo_id ? "opacity-100" : "opacity-0")} />
      <span className="flex-1 truncate">{d.repo_id}</span>
      {d.source === "both" && <span className="text-xs text-gray-400 mr-2">on Hub</span>}
      {d.private && <span className="text-xs text-amber-400">private</span>}
    </CommandItem>
  );

  if (customMode) {
    return (
      <div className="flex gap-2">
        <Input
          autoFocus
          value={customValue}
          onChange={(e) => setCustomValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") submitCustom(); }}
          placeholder="org/dataset-name"
          className="bg-gray-800 border-gray-600 text-white"
        />
        <Button onClick={submitCustom} disabled={!REPO_ID_RE.test(customValue.trim())}>
          Use
        </Button>
        <Button variant="ghost" onClick={() => setCustomMode(false)}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between bg-gray-800 border-gray-600 text-white hover:bg-gray-700"
        >
          {value ?? (loading ? "Loading datasets…" : "Select a dataset…")}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-gray-800 border-gray-700" align="start">
        <Command className="bg-gray-800 text-white">
          <CommandInput placeholder="Search datasets…" className="text-white" />
          <CommandList>
            <CommandEmpty>{loading ? "Loading…" : "No datasets."}</CommandEmpty>
            {localDatasets.length > 0 && (
              <CommandGroup heading="Local">
                {localDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {hubDatasets.length > 0 && (
              <CommandGroup heading="Hugging Face">
                {hubDatasets.map(renderItem)}
              </CommandGroup>
            )}
            <CommandGroup>
              <CommandItem
                onSelect={() => { setCustomMode(true); setOpen(false); }}
                className="text-purple-300 aria-selected:bg-gray-700"
              >
                <Pencil className="mr-2 h-4 w-4" />
                Use custom repo ID…
              </CommandItem>
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetCombobox;
```

- [ ] **Step 2: Type check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: 0 errors in this file. (Errors may still exist in `DatasetPicker.tsx` and `Landing.tsx`; ignore those for now.)

- [ ] **Step 3: Visual check in browser**

Start the dev server (`lelab --dev` from repo root) and open `http://localhost:8080`. Navigate to:

- **Replay page** (`/replay`): open the dataset combobox. Expected: two headings — "Local" first (if you have local datasets), then "Hugging Face". The "Use custom repo ID…" affordance is still present at the bottom.
- **Training page** (`/training`): the combobox in the Run Configuration card should show the same grouping.

If a dataset exists in both (`source: "both"`), it appears under Local with a small gray "on Hub" tag right of the name and left of any "private" tag.

If the lists look right, stop the dev server (Ctrl-C).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/replay/DatasetCombobox.tsx
git commit -m "feat(datasets): split DatasetCombobox into Local and Hub groups"
```

---

## Task 6: Frontend — group split in `DatasetPicker` + new `onPickExisting` signature

**Files:**
- Modify: [`frontend/src/components/landing/DatasetPicker.tsx`](../../../frontend/src/components/landing/DatasetPicker.tsx)

The `onPickExisting` prop signature changes from `(repoId: string) => void` to `(item: DatasetItem) => void` so Landing can branch on `source`. Landing.tsx is updated in Task 7 — there will be a TS error between Task 6 and Task 7, which is expected.

- [ ] **Step 1: Replace the component**

Overwrite [`frontend/src/components/landing/DatasetPicker.tsx`](../../../frontend/src/components/landing/DatasetPicker.tsx) with:

```tsx
import React, { useState } from "react";
import { Plus, ExternalLink } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { DatasetItem } from "@/lib/replayApi";

interface DatasetPickerProps {
  datasets: DatasetItem[];
  loading: boolean;
  onPickExisting: (item: DatasetItem) => void;
  onCreateNew: (name: string) => void;
  onOpenCustom: (repoId: string) => void;
  children: React.ReactNode;
}

const REPO_ID_RE = /^[\w.\-]+\/[\w.\-]+$/;
const NAME_RE = /^[A-Za-z0-9._-]+$/;

const DatasetPicker: React.FC<DatasetPickerProps> = ({
  datasets,
  loading,
  onPickExisting,
  onCreateNew,
  onOpenCustom,
  children,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const trimmed = query.trim();
  const matchesExisting = datasets.some(
    (d) => d.repo_id.toLowerCase() === trimmed.toLowerCase(),
  );
  const isRepoId = REPO_ID_RE.test(trimmed);
  const isName = NAME_RE.test(trimmed) && !trimmed.includes("/");
  const canCreate = trimmed.length > 0 && isName && !matchesExisting;
  const canOpenCustom = isRepoId && !matchesExisting;

  const localDatasets = datasets.filter((d) => d.source === "local" || d.source === "both");
  const hubDatasets = datasets.filter((d) => d.source === "hub");

  const reset = () => {
    setQuery("");
    setOpen(false);
  };

  const handlePick = (item: DatasetItem) => {
    onPickExisting(item);
    reset();
  };

  const handleCreate = () => {
    if (!canCreate) return;
    onCreateNew(trimmed);
    reset();
  };

  const handleOpenCustom = () => {
    if (!canOpenCustom) return;
    onOpenCustom(trimmed);
    reset();
  };

  const renderItem = (d: DatasetItem) => (
    <CommandItem
      key={d.repo_id}
      value={d.repo_id}
      onSelect={() => handlePick(d)}
      className="text-white aria-selected:bg-gray-700"
    >
      <span className="flex-1 truncate">{d.repo_id}</span>
      {d.source === "both" && (
        <span className="text-xs text-gray-400 mr-2">on Hub</span>
      )}
      {d.private && (
        <span className="text-xs text-amber-400">private</span>
      )}
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="w-[320px] p-0 bg-gray-800 border-gray-700 text-white"
        align="end"
      >
        <Command className="bg-gray-800">
          <CommandInput
            placeholder="Search, type a new name, or org/name…"
            value={query}
            onValueChange={(v) => setQuery(v.replace(/[^A-Za-z0-9._\-/]/g, "_"))}
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              if (canCreate) {
                e.preventDefault();
                handleCreate();
              } else if (canOpenCustom) {
                e.preventDefault();
                handleOpenCustom();
              }
            }}
            className="text-white"
          />
          <CommandList>
            {datasets.length === 0 && !canCreate && !canOpenCustom && (
              <CommandEmpty className="py-4 text-sm text-gray-400 text-center">
                {loading
                  ? "Loading datasets…"
                  : "No datasets yet. Type a name to create one."}
              </CommandEmpty>
            )}
            {localDatasets.length > 0 && (
              <CommandGroup heading="Local">
                {localDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {hubDatasets.length > 0 && (
              <CommandGroup heading="Hugging Face">
                {hubDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {canCreate && (
              <CommandGroup heading="New">
                <CommandItem
                  value={`__create__${trimmed}`}
                  onSelect={handleCreate}
                  className="text-white aria-selected:bg-gray-700"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Create &quot;{trimmed}&quot;
                </CommandItem>
              </CommandGroup>
            )}
            {canOpenCustom && (
              <CommandGroup heading="Custom repo">
                <CommandItem
                  value={`__open__${trimmed}`}
                  onSelect={handleOpenCustom}
                  className="text-white aria-selected:bg-gray-700"
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open &quot;{trimmed}&quot; in viewer
                </CommandItem>
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetPicker;
```

- [ ] **Step 2: Verify the only remaining TS error is in Landing.tsx**

```bash
cd frontend && npx tsc --noEmit
```

Expected: an error in `frontend/src/pages/Landing.tsx` about `onPickExisting`'s argument type (it currently passes `openDatasetInViewer` which takes `string`). Other files compile clean. **If errors appear elsewhere, stop and fix before continuing.**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/landing/DatasetPicker.tsx
git commit -m "feat(datasets): split DatasetPicker into Local and Hub groups"
```

---

## Task 7: Frontend — Landing branches click action on `source`

**Files:**
- Modify: [`frontend/src/pages/Landing.tsx`](../../../frontend/src/pages/Landing.tsx) (lines ~91-99 for `openDatasetInViewer`, ~199-214 for the `<DatasetPicker>` usage)

- [ ] **Step 1: Replace `openDatasetInViewer` with a source-aware handler**

In [`frontend/src/pages/Landing.tsx`](../../../frontend/src/pages/Landing.tsx), find:

```tsx
  const openDatasetInViewer = (repoId: string) => {
    const found = datasets.find((d) => d.repo_id === repoId);
    const needsAuth = !found || found.private;
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    const target = needsAuth
      ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
      : `https://huggingface.co${spacePath}`;
    window.open(target, "_blank", "noopener,noreferrer");
  };
```

Replace with:

```tsx
  const openHubViewer = (repoId: string, isPrivate: boolean) => {
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    const target = isPrivate
      ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
      : `https://huggingface.co${spacePath}`;
    window.open(target, "_blank", "noopener,noreferrer");
  };

  const handlePickExisting = (item: DatasetItem) => {
    if (item.source === "local" || item.source === "both") {
      navigate("/upload", {
        state: { datasetInfo: { dataset_repo_id: item.repo_id } },
      });
      return;
    }
    openHubViewer(item.repo_id, item.private);
  };

  const handleOpenCustom = (repoId: string) => {
    // Custom-typed repo IDs are always treated as Hub paths. We don't know
    // privacy, so route through the login redirect to be safe.
    openHubViewer(repoId, true);
  };
```

- [ ] **Step 2: Add the `DatasetItem` import and update `<DatasetPicker>` usage**

At the top of the same file, find the existing import line for `useDatasets`:

```tsx
import { useDatasets } from "@/hooks/useDatasets";
```

Add immediately below it:

```tsx
import { DatasetItem } from "@/lib/replayApi";
```

Then find the `<DatasetPicker ... />` usage (currently around lines 199-214):

```tsx
  const datasetTrigger = (
    <DatasetPicker
      datasets={datasets}
      loading={datasetsLoading}
      onPickExisting={openDatasetInViewer}
      onOpenCustom={openDatasetInViewer}
      onCreateNew={handleCreateDataset}
    >
```

Replace with:

```tsx
  const datasetTrigger = (
    <DatasetPicker
      datasets={datasets}
      loading={datasetsLoading}
      onPickExisting={handlePickExisting}
      onOpenCustom={handleOpenCustom}
      onCreateNew={handleCreateDataset}
    >
```

- [ ] **Step 3: Type check**

```bash
cd frontend && npx tsc --noEmit
```

Expected: exits 0 with no output.

- [ ] **Step 4: End-to-end browser check**

Start `lelab --dev` and navigate to `http://localhost:8080`. On the Landing page:

1. Click the dataset action's arrow button. The picker opens with grouped sections.
2. Click a **local** dataset (or a `"both"` dataset). Expected: the page navigates to `/upload`, the Upload page loads, and the dataset summary shows the repo_id you clicked. You should see "Upload to HuggingFace Hub" and "Skip Upload" buttons.
3. Go back to Landing. Click a **Hub-only** dataset. Expected: a new tab opens to `huggingface.co/spaces/lerobot/visualize_dataset?path=/<repo_id>` (or the login redirect if the dataset is private).
4. Type a custom org/name in the picker that you don't have, hit Enter. Expected: opens viewer with login redirect (existing behavior).
5. Type a fresh name and pick "Create". Expected: opens the recording modal (existing behavior, unchanged).

If any of these is wrong, fix before committing.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Landing.tsx
git commit -m "feat(landing): route local dataset picks to upload page"
```

---

## Task 8: End-to-end validation pass

**Files:** none modified.

This is a final manual smoke test. The repo has no automated tests, so this stands in for the test suite.

- [ ] **Step 1: Start fresh `lelab --dev` and run through the matrix**

```bash
lelab --dev
```

In a browser, verify each of these against `http://localhost:8080`:

| # | Action | Expected |
|---|--------|----------|
| 1 | Open Landing → click dataset arrow | Picker opens with Local + Hugging Face sections (one or both, depending on your cache and HF token) |
| 2 | Open Replay (`/replay`) → open combobox | Same two-group split |
| 3 | Open Training (`/training`) → open the Run Configuration combobox | Same two-group split |
| 4 | Pick a local dataset on Landing | Navigates to `/upload`; dataset summary loads; Upload-to-Hub and Skip buttons present |
| 5 | Pick a Hub-only dataset on Landing | Opens a new tab to the HF visualize_dataset Space |
| 6 | Pick a `"both"` dataset on Landing | Routes to `/upload` (treated as local; the small "on Hub" badge is visible in the dropdown) |
| 7 | Pick a local dataset on Replay | Selection sets; replay flow proceeds normally |
| 8 | Pick a local dataset in Training | `dataset_repo_id` field populates with the local repo_id |
| 9 | Move HF token aside (`mv ~/.cache/huggingface/token ~/.cache/huggingface/token.bak`), reload Landing | Only the Local group appears; no error toast about the dataset list |
| 10 | Restore HF token (`mv ~/.cache/huggingface/token.bak ~/.cache/huggingface/token`) | Both groups reappear after a reload |

- [ ] **Step 2: Sanity-check `curl /datasets` once more**

```bash
curl -s http://localhost:8000/datasets | python -c 'import json,sys; data=json.load(sys.stdin); kinds={}; [kinds.__setitem__(d["source"], kinds.get(d["source"],0)+1) for d in data]; print(kinds)'
```

Expected: a dict like `{'hub': N, 'local': M, 'both': K}` reflecting your cache state.

- [ ] **Step 3: Stop dev server, no commit needed for validation**

If everything passes, the feature is ready. If anything fails, return to the relevant task and fix.

---

## Self-review notes

- Spec coverage: each spec section is covered — §1 Local detection → Task 1; §2 API contract → Tasks 1 & 2; §3 Type → Task 4; §4 Grouping → Tasks 5 & 6; §5 Landing click → Task 7; §6 Replay/Training (no-op) → Task 8 validation only; HF cloud TODO → Task 3; edge cases → Tasks 1 & 8.
- Type consistency: `DatasetSource = "local" | "hub" | "both"` is defined once in Task 4 and reused identically in Tasks 5, 6, 7. The `source: "both"` filter rule (`d.source === "local" || d.source === "both"`) appears verbatim in both Task 5 and Task 6.
- No placeholders. Every code step shows the exact code to paste; every command shows expected output.
