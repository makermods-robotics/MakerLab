# LeRobot-Aligned Repo Layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the leLab Python package from `app/` → `lelab/` to mirror LeRobot's package layout, file naming, and tooling conventions, without changing runtime behavior.

**Architecture:** Mechanical rename. All intra-package imports are relative (verified by grep), so the moves preserve semantics. Two absolute uvicorn module strings and one pyproject script entry need updating. Tooling (ruff, pre-commit, smoke test, license headers) is added in dedicated tasks at the end so behavior changes and config changes don't mix.

**Tech Stack:** Python 3.12+, FastAPI, setuptools, ruff, pre-commit, pytest.

**Spec:** [`docs/superpowers/specs/2026-05-07-lerobot-aligned-repo-layout-design.md`](../specs/2026-05-07-lerobot-aligned-repo-layout-design.md)

---

## Pre-flight reference data (verified during planning)

**All intra-package relative imports that need rewriting after moves:**

| File (before move) | Line | Current import | Required new import |
|---|---|---|---|
| `app/main.py` | 15 | `from . import config` | `from .utils import config` |
| `app/main.py` | 19 | `from .recording import (...)` | `from .record import (...)` |
| `app/main.py` | 34 | `from .teleoperating import (...)` | `from .teleoperate import (...)` |
| `app/main.py` | 42 | `from .inferring import (...)` | `from .rollout import (...)` |
| `app/main.py` | 50 | `from .calibrating import CalibrationRequest, calibration_manager` | `from .calibrate import CalibrationRequest, calibration_manager` |
| `app/main.py` | 53 | `from .training import TrainingRequest` | `from .train import TrainingRequest` |
| `app/main.py` | 54 | `from .jobs import (...)` | unchanged |
| `app/main.py` | 62 | `from .system import (...)` | `from .utils.system import (...)` |
| `app/main.py` | 71 | `from .hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login` | `from .utils.hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login` |
| `app/main.py` | 72 | `from . import dataset_browser` | `from . import datasets as dataset_browser` |
| `app/main.py` | 122 | `from .config import (...)` | `from .utils.config import (...)` |
| `app/recording.py` | 18 | `from .config import (...)` | `from .utils.config import (...)` |
| `app/teleoperating.py` | 11 | `from .config import setup_calibration_files` | `from .utils.config import setup_calibration_files` |
| `app/inferring.py` | 23 | `from .config import setup_follower_calibration_file` | `from .utils.config import setup_follower_calibration_file` |
| `app/jobs.py` | 20 | `from .training import TrainingRequest` | `from .train import TrainingRequest` |
| `app/runners/__init__.py` | 7 | `from .hf_cloud import HfCloudJobRunner` | unchanged |
| `app/runners/hf_cloud.py` | 22 | `from ..hf_auth import cached_whoami` | `from ..utils.hf_auth import cached_whoami` |
| `app/runners/hf_cloud.py` | 23 | `from ..jobs import LogLine, TrainingMetrics, extract_wandb_run_url, parse_metrics_into` | unchanged |
| `app/runners/hf_cloud.py` | 24 | `from ..training import TrainingRequest, build_training_command` | `from ..train import TrainingRequest, build_training_command` |
| `scripts/backend.py` | 75 | `"app.main:app"` | `"lelab.server:app"` |
| `scripts/backend.py` | 113 | `"app.main:app"` | `"lelab.server:app"` |

**File mapping:**

| Current path | New path |
|---|---|
| `app/__init__.py` | overwritten by Task 2 (new content) |
| `app/main.py` | `lelab/server.py` |
| `app/recording.py` | `lelab/record.py` |
| `app/teleoperating.py` | `lelab/teleoperate.py` |
| `app/calibrating.py` | `lelab/calibrate.py` |
| `app/inferring.py` | `lelab/rollout.py` |
| `app/training.py` | `lelab/train.py` |
| `app/jobs.py` | `lelab/jobs.py` |
| `app/dataset_browser.py` | `lelab/datasets.py` |
| `app/config.py` | `lelab/utils/config.py` |
| `app/hf_auth.py` | `lelab/utils/hf_auth.py` |
| `app/system.py` | `lelab/utils/system.py` |
| `app/runners/__init__.py` | `lelab/runners/__init__.py` |
| `app/runners/hf_cloud.py` | `lelab/runners/hf_cloud.py` |
| `app/static/` | **DELETED** — empty, unused (StaticFiles mount targets `frontend/dist`) |
| `scripts/backend.py` | `lelab/scripts/lelab.py` |
| `scripts/__init__.py` | **DELETED** with the directory |

**Files needing license headers added** (every `.py` in `lelab/` and `tests/` after the moves):

```
lelab/__init__.py, lelab/__version__.py, lelab/types.py, lelab/server.py,
lelab/record.py, lelab/teleoperate.py, lelab/calibrate.py, lelab/rollout.py,
lelab/train.py, lelab/jobs.py, lelab/datasets.py,
lelab/utils/__init__.py, lelab/utils/config.py, lelab/utils/hf_auth.py, lelab/utils/system.py,
lelab/runners/__init__.py, lelab/runners/hf_cloud.py,
lelab/scripts/__init__.py, lelab/scripts/lelab.py,
tests/__init__.py, tests/test_smoke.py
```

`replaying.py` does **not** exist in the current tree (verified). The CLAUDE.md mention is stale documentation; it gets cleaned up in Task 12.

---

## Task 1: Pre-flight — confirm baseline behavior

**Files:** none modified. Establishes the regression target.

- [ ] **Step 1: Verify clean working tree before starting**

Run:
```bash
cd /Users/nicolasrabault/Projects/Hackathon/leLab
git status --short
```
Expected: only the pre-existing untracked files from before this work begins (no in-progress edits to `app/` or `scripts/`).

- [ ] **Step 2: Confirm the package installs and the server boots**

Run:
```bash
pip install -e . 2>&1 | tail -5
lelab &
LELAB_PID=$!
sleep 5
curl -s http://127.0.0.1:8000/health
echo
kill $LELAB_PID
wait $LELAB_PID 2>/dev/null
```
Expected: install succeeds; `/health` returns a JSON body (HTTP 200). Record the exact `/health` response — this is the post-rename equivalence check.

- [ ] **Step 3: Save baseline `/health` response and route count for later comparison**

Run:
```bash
python -c "from app.main import app; print('routes:', len(app.routes))"
```
Expected: prints e.g. `routes: 47` (a positive integer). Note the number — Task 14 must produce the same value.

---

## Task 2: Create the `lelab/` skeleton (new files only, no moves)

**Files:**
- Create: `lelab/__init__.py`
- Create: `lelab/__version__.py`
- Create: `lelab/types.py`
- Create: `lelab/utils/__init__.py`
- Create: `lelab/runners/__init__.py` *(will be overwritten in Task 4 by the moved file; create empty for now so the dir exists)*
- Create: `lelab/scripts/__init__.py`

- [ ] **Step 1: Create the directories**

Run:
```bash
mkdir -p lelab/utils lelab/runners lelab/scripts
```

- [ ] **Step 2: Create `lelab/__init__.py`**

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LeLab — FastAPI + React web interface around the LeRobot framework."""

from lelab.__version__ import __version__

__all__ = ["__version__"]
```

- [ ] **Step 3: Create `lelab/__version__.py`**

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""To enable `lelab.__version__`."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lelab")
except PackageNotFoundError:
    __version__ = "unknown"
```

- [ ] **Step 4: Create `lelab/types.py`** (stub)

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared types for the LeLab package. Populated as cross-module types emerge."""
```

- [ ] **Step 5: Create empty `__init__.py` for the new subpackages**

```bash
: > lelab/utils/__init__.py
: > lelab/runners/__init__.py
: > lelab/scripts/__init__.py
```

- [ ] **Step 6: Update `pyproject.toml` to discover `lelab*`** (alongside `app*` for now)

Edit `pyproject.toml`. Replace:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["app*", "scripts*", "frontend"]
```
With:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["app*", "lelab*", "scripts*", "frontend"]
```
Both packages installable in parallel during the migration so the old `app.main:app` keeps booting until Task 6 flips the entry point. We delete `app*` in Task 11.

- [ ] **Step 7: Reinstall and verify the new package imports**

Run:
```bash
pip install -e . 2>&1 | tail -3
python -c "import lelab; print(lelab.__version__)"
```
Expected: install succeeds; prints a version string (`0.1.0` from pyproject, or `unknown`).

- [ ] **Step 8: Confirm the old app still boots** (regression guard)

Run:
```bash
python -c "from app.main import app; print('OK', len(app.routes))"
```
Expected: prints `OK <N>` where `<N>` matches Task 1 step 3.

- [ ] **Step 9: Commit**

```bash
git add lelab/__init__.py lelab/__version__.py lelab/types.py lelab/utils/__init__.py lelab/runners/__init__.py lelab/scripts/__init__.py pyproject.toml
git commit -m "$(cat <<'EOF'
refactor(layout): scaffold lelab/ package alongside app/

Empty skeleton mirroring LeRobot's package layout. app/ remains the
runtime package until file moves and entry-point switch land.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Move feature modules with `git mv` (preserves history)

**Files:** moves only — no edits yet. Imports break temporarily; Task 5 fixes them.

- [ ] **Step 1: Move feature/server modules into `lelab/`**

Run:
```bash
git mv app/main.py lelab/server.py
git mv app/recording.py lelab/record.py
git mv app/teleoperating.py lelab/teleoperate.py
git mv app/calibrating.py lelab/calibrate.py
git mv app/inferring.py lelab/rollout.py
git mv app/training.py lelab/train.py
git mv app/jobs.py lelab/jobs.py
git mv app/dataset_browser.py lelab/datasets.py
```

- [ ] **Step 2: Move utilities into `lelab/utils/`**

Run:
```bash
git mv app/config.py lelab/utils/config.py
git mv app/hf_auth.py lelab/utils/hf_auth.py
git mv app/system.py lelab/utils/system.py
```

- [ ] **Step 3: Move runners into `lelab/runners/`** (overwrites the empty stub from Task 2)

Run:
```bash
rm lelab/runners/__init__.py
git mv app/runners/__init__.py lelab/runners/__init__.py
git mv app/runners/hf_cloud.py lelab/runners/hf_cloud.py
```

- [ ] **Step 4: Verify the moves**

Run:
```bash
ls lelab/ lelab/utils/ lelab/runners/
ls app/  # should now contain only __init__.py and possibly __pycache__/, static/
```
Expected: `lelab/` shows the renamed files; `app/` is nearly empty.

- [ ] **Step 5: Commit the moves (broken imports — fixed in next tasks)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(layout): move app/ contents to lelab/ via git mv

Preserves git history per file. Imports are temporarily broken;
fixed in the next commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Move the entry script into `lelab/scripts/`

**Files:**
- Move: `scripts/backend.py` → `lelab/scripts/lelab.py`

- [ ] **Step 1: Move the file**

Run:
```bash
git mv scripts/backend.py lelab/scripts/lelab.py
```

- [ ] **Step 2: Remove the now-stale `scripts/__init__.py`** (the new one in `lelab/scripts/` was created in Task 2)

Run:
```bash
git rm scripts/__init__.py
rmdir scripts/  # fails if not empty — check what's inside if so
```
Expected: `rmdir` succeeds (the directory is empty after removing the two files).

- [ ] **Step 3: Verify**

Run:
```bash
ls lelab/scripts/
test ! -d scripts/ && echo "scripts/ removed"
```
Expected: `lelab/scripts/` shows `__init__.py` and `lelab.py`; the message confirms `scripts/` is gone.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(layout): move scripts/backend.py into lelab/scripts/lelab.py

Drops the top-level scripts/ directory. Console-script entry still
broken — fixed in pyproject update.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fix relative imports across moved files

**Files:**
- Modify: `lelab/server.py`
- Modify: `lelab/record.py`
- Modify: `lelab/teleoperate.py`
- Modify: `lelab/rollout.py`
- Modify: `lelab/jobs.py`
- Modify: `lelab/runners/hf_cloud.py`

Use the import rewrite table at the top of this plan as the ground truth.

- [ ] **Step 1: Rewrite imports in `lelab/server.py`**

Apply these exact substitutions (the line numbers refer to the source before any edits — use grep to relocate after move if shifted):

```python
# Was:  from . import config
# Now:  from .utils import config

# Was:  from .recording import (
# Now:  from .record import (

# Was:  from .teleoperating import (
# Now:  from .teleoperate import (

# Was:  from .inferring import (
# Now:  from .rollout import (

# Was:  from .calibrating import CalibrationRequest, calibration_manager
# Now:  from .calibrate import CalibrationRequest, calibration_manager

# Was:  from .training import TrainingRequest
# Now:  from .train import TrainingRequest

# Was:  from .system import (
# Now:  from .utils.system import (

# Was:  from .hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login
# Now:  from .utils.hf_auth import cached_whoami, handle_hf_auth_status, handle_hf_login

# Was:  from . import dataset_browser
# Now:  from . import datasets as dataset_browser

# Was:  from .config import (
# Now:  from .utils.config import (

# (from .jobs import — UNCHANGED)
```

`from . import datasets as dataset_browser` keeps the local symbol name `dataset_browser` working at every call site without touching usages.

- [ ] **Step 2: Rewrite imports in the other moved files**

```python
# lelab/record.py
# Was:  from .config import (
# Now:  from .utils.config import (

# lelab/teleoperate.py
# Was:  from .config import setup_calibration_files
# Now:  from .utils.config import setup_calibration_files

# lelab/rollout.py
# Was:  from .config import setup_follower_calibration_file
# Now:  from .utils.config import setup_follower_calibration_file

# lelab/jobs.py
# Was:  from .training import TrainingRequest
# Now:  from .train import TrainingRequest

# lelab/runners/hf_cloud.py
# Was:  from ..hf_auth import cached_whoami
# Now:  from ..utils.hf_auth import cached_whoami
# Was:  from ..training import TrainingRequest, build_training_command
# Now:  from ..train import TrainingRequest, build_training_command
# (from ..jobs import — UNCHANGED)
```

`lelab/runners/__init__.py` (`from .hf_cloud import HfCloudJobRunner`) is unchanged.
`lelab/calibrate.py`, `lelab/train.py`, `lelab/datasets.py`, `lelab/utils/system.py`, `lelab/utils/hf_auth.py`, `lelab/utils/config.py` have no relative imports — leave them alone.

- [ ] **Step 3: Verify the new package imports cleanly**

Run:
```bash
python -c "import lelab.server; print('routes:', len(lelab.server.app.routes))"
```
Expected: prints `routes: <N>` matching Task 1 step 3. If it fails with `ModuleNotFoundError` or `ImportError`, the missing/wrong import is named in the error — fix and retry.

- [ ] **Step 4: Confirm the FastAPI app object is usable from the new path**

Run:
```bash
python -c "from lelab.server import app; print(type(app).__name__, len(app.routes))"
```
Expected: prints `FastAPI <N>`.

- [ ] **Step 5: Commit**

```bash
git add lelab/
git commit -m "$(cat <<'EOF'
refactor(layout): rewrite intra-package imports for lelab/ paths

Updates module references after the rename. server.py wraps the
datasets module under the alias dataset_browser to avoid touching
call sites.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update entry script's uvicorn module strings and pyproject `[project.scripts]`

**Files:**
- Modify: `lelab/scripts/lelab.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace both `app.main:app` strings in `lelab/scripts/lelab.py`**

There are two occurrences (originally lines 75 and 113 — relocate via grep after move):

```python
# Was:  uvicorn.run(
#           "app.main:app",
# Now:  uvicorn.run(
#           "lelab.server:app",

# Was:  [
#           sys.executable,
#           "-m",
#           "uvicorn",
#           "app.main:app",
# Now:  [
#           sys.executable,
#           "-m",
#           "uvicorn",
#           "lelab.server:app",
```

- [ ] **Step 2: Update `[project.scripts]` in `pyproject.toml`**

Edit:
```toml
[project.scripts]
lelab = "scripts.backend:main"
```
To:
```toml
[project.scripts]
lelab = "lelab.scripts.lelab:main"
```

- [ ] **Step 3: Reinstall to refresh the console-script shim**

Run:
```bash
pip install -e . 2>&1 | tail -3
which lelab
```
Expected: install succeeds; `which lelab` resolves to the venv's bin dir.

- [ ] **Step 4: Boot the server and confirm `/health`**

Run:
```bash
lelab &
LELAB_PID=$!
sleep 5
curl -s http://127.0.0.1:8000/health
echo
kill $LELAB_PID
wait $LELAB_PID 2>/dev/null
```
Expected: same `/health` response as Task 1 step 2.

- [ ] **Step 5: Commit**

```bash
git add lelab/scripts/lelab.py pyproject.toml
git commit -m "$(cat <<'EOF'
refactor(layout): point lelab CLI at lelab.server:app

Updates the console-script entry and the two uvicorn module strings
inside the launcher to the new package path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add the smoke test scaffold

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```python
```
(Empty file — `: > tests/__init__.py` after `mkdir -p tests`.)

- [ ] **Step 2: Create `tests/test_smoke.py`**

```python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke test — confirms the package is installed and importable.

Populated tests will land later. This file exists so the tests/ structure
is in place from day one.
"""

import lelab
from lelab.server import app


def test_lelab_has_version():
    assert isinstance(lelab.__version__, str)
    assert lelab.__version__


def test_server_app_has_routes():
    assert len(app.routes) > 0
```

- [ ] **Step 3: Add `[project.optional-dependencies]` to pyproject for dev tooling**

Edit `pyproject.toml` — add (after `dependencies = [...]`):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.6",
    "pre-commit>=3.0",
]
```

- [ ] **Step 4: Install pytest and run the smoke test**

Run:
```bash
pip install -e ".[dev]" 2>&1 | tail -3
pytest tests/ -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "$(cat <<'EOF'
test: add smoke test and dev extras

Two assertions: lelab.__version__ is a non-empty string, and the
FastAPI app exposes at least one route. Real test coverage to be
added in follow-up work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add ruff config to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Fetch the LeRobot pinned-SHA pyproject for reference**

Run:
```bash
curl -s "https://raw.githubusercontent.com/huggingface/lerobot/82dffde7fad11cba91f7916b050fbe7d7eea35ab/pyproject.toml" -o /tmp/lerobot-pyproject.toml
grep -A 30 "tool.ruff" /tmp/lerobot-pyproject.toml
```
Expected: prints LeRobot's `[tool.ruff]` and `[tool.ruff.lint]` sections.

- [ ] **Step 2: Copy LeRobot's ruff config into `pyproject.toml`**

Append (verbatim from `/tmp/lerobot-pyproject.toml` if it exists; otherwise use the skeleton below as a fallback):

Skeleton fallback (use only if step 1 returned nothing):
```toml
[tool.ruff]
line-length = 110
target-version = "py312"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "N", "B", "C4", "SIM", "UP"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
```

- [ ] **Step 3: Run ruff and observe**

Run:
```bash
ruff check lelab/ tests/ 2>&1 | tail -20
```
Expected: either clean, or a list of pre-existing issues. **Do not auto-fix or restructure code to satisfy ruff** in this PR — note the output and move on. (Style cleanup is its own project; the spec calls this explicitly out of scope.)

- [ ] **Step 4: Commit the config**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore(tooling): add ruff config matching LeRobot's pyproject

Style cleanup left for a follow-up PR; this commit only configures
the linter so future checks have a baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Add `.pre-commit-config.yaml` matching LeRobot

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Fetch LeRobot's pre-commit config**

Run:
```bash
curl -s "https://raw.githubusercontent.com/huggingface/lerobot/82dffde7fad11cba91f7916b050fbe7d7eea35ab/.pre-commit-config.yaml" -o .pre-commit-config.yaml
cat .pre-commit-config.yaml
```
Expected: writes the config; `cat` prints non-empty YAML with ruff hooks.

If the curl fetch fails or returns 404 (the SHA might predate the file or the file might not exist at that path), fall back to this skeleton:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 2: Install hooks (optional — safe even if user-side pre-commit isn't enabled)**

Run:
```bash
pre-commit install || true
pre-commit run --all-files 2>&1 | tail -20
```
Expected: hooks install if pre-commit picks up the config. Running over all files surfaces lint issues but does not block this commit (per the same scope decision as Task 8 step 3).

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "$(cat <<'EOF'
chore(tooling): add pre-commit config matching LeRobot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Add Apache-2.0 license headers to every `.py` file in the new layout

**Files:** all files listed under "Files needing license headers added" at the top of this plan.

- [ ] **Step 1: Define the canonical header**

Save to `/tmp/lelab-license-header.txt`:

```text
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

(Use `printf` or write via the editor — make sure the trailing newline is present.)

- [ ] **Step 2: Add the header to every `.py` file in `lelab/` and `tests/` that doesn't already have one**

Files created in Task 2 (`lelab/__init__.py`, `lelab/__version__.py`, `lelab/types.py`) and Task 7 (`tests/test_smoke.py`) already have the header. Files moved from `app/` and `scripts/` do not.

For each unheadered file, prepend the header followed by a blank line. Recommended approach (run from repo root):

```bash
HEADER=$(cat /tmp/lelab-license-header.txt)
for f in \
  lelab/server.py lelab/record.py lelab/teleoperate.py lelab/calibrate.py \
  lelab/rollout.py lelab/train.py lelab/jobs.py lelab/datasets.py \
  lelab/utils/config.py lelab/utils/hf_auth.py lelab/utils/system.py \
  lelab/runners/__init__.py lelab/runners/hf_cloud.py \
  lelab/scripts/lelab.py \
  lelab/utils/__init__.py lelab/scripts/__init__.py \
  tests/__init__.py
do
  if ! head -1 "$f" | grep -q "Copyright"; then
    { echo "$HEADER"; echo; cat "$f"; } > "$f.new" && mv "$f.new" "$f"
  fi
done
```

(For empty `__init__.py` files this leaves them with only the header, which is fine.)

- [ ] **Step 3: Verify every file in the package now has a header**

Run:
```bash
for f in $(find lelab tests -name "*.py"); do
  if ! head -1 "$f" | grep -q "Copyright"; then
    echo "MISSING: $f"
  fi
done
```
Expected: no output (no missing headers).

- [ ] **Step 4: Re-run the smoke test to confirm headers didn't break syntax**

Run:
```bash
pytest tests/ -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add lelab/ tests/
git commit -m "$(cat <<'EOF'
chore(license): add Apache-2.0 headers to lelab/ and tests/

Matches LeRobot's header text verbatim. Bare __init__.py files now
contain only the header.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Remove the old `app/` package and update pyproject discovery

**Files:**
- Delete: `app/__init__.py` and the `app/` directory (and any leftover `app/static/`, `app/__pycache__/`)
- Modify: `pyproject.toml`

- [ ] **Step 1: Confirm `app/` is empty of source**

Run:
```bash
find app/ -type f 2>/dev/null
```
Expected: only `app/__init__.py`. (If `app/static/` exists with files, abort and inspect — the spec assumed it was empty.)

- [ ] **Step 2: Remove the directory**

Run:
```bash
git rm app/__init__.py
rm -rf app/  # nukes any non-tracked __pycache__ and the empty static/
```

- [ ] **Step 3: Drop `app*` from pyproject's package discovery**

Edit `pyproject.toml`. Replace:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["app*", "lelab*", "scripts*", "frontend"]
```
With:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["lelab*", "frontend"]
```
(`scripts*` is also gone — the directory was deleted in Task 4.)

- [ ] **Step 4: Reinstall and run the smoke test**

Run:
```bash
pip install -e ".[dev]" 2>&1 | tail -3
pytest tests/ -v
```
Expected: install succeeds; 2 passed.

- [ ] **Step 5: Boot the server end-to-end**

Run:
```bash
lelab &
LELAB_PID=$!
sleep 5
curl -s http://127.0.0.1:8000/health
echo
kill $LELAB_PID
wait $LELAB_PID 2>/dev/null
```
Expected: `/health` response identical to Task 1.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(layout): remove app/ package now that lelab/ is the canonical home

pyproject.toml no longer discovers the deleted app* and scripts*
packages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update `CLAUDE.md` to reference the new layout

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: List the path references that need updating**

Run:
```bash
grep -n "app/" CLAUDE.md
```
Expected: 13 lines, all of which reference `app/<file>.py` paths inside the "Backend module layout" section and surrounding prose.

- [ ] **Step 2: Apply the replacements**

| Line content (before) | After |
|---|---|
| `[`app/main.py`](app/main.py)` | `[`lelab/server.py`](lelab/server.py)` |
| `[app/inferring.py](app/inferring.py)` | `[lelab/rollout.py](lelab/rollout.py)` |
| ``### Backend module layout (`app/`)`` | ``### Backend module layout (`lelab/`)`` |
| `[main.py](app/main.py)` | `[server.py](lelab/server.py)` |
| `[recording.py](app/recording.py)` | `[record.py](lelab/record.py)` |
| `[teleoperating.py](app/teleoperating.py)` | `[teleoperate.py](lelab/teleoperate.py)` |
| `[calibrating.py](app/calibrating.py)` | `[calibrate.py](lelab/calibrate.py)` |
| `[replaying.py](app/replaying.py)` | **delete the bullet entirely** — file does not exist |
| `[training.py](app/training.py)` | `[train.py](lelab/train.py)` |
| `[config.py](app/config.py)` | `[config.py](lelab/utils/config.py)` |
| `[main.py](app/main.py#L104-L206)` (in WebSocket section) | `[server.py](lelab/server.py#L104-L206)` |

After edits, also update any narrative sentences that say "Each feature module" / "feature modules" to remain accurate — only the path references change, not the architectural description.

- [ ] **Step 3: Verify no stale references remain**

Run:
```bash
grep -n "app/" CLAUDE.md
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update CLAUDE.md paths for lelab/ layout

Removes the stale replaying.py bullet; that module did not exist
in the tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update README and other user-facing docs (only if they reference moved paths)

**Files:** `README.md`, `HTTPS_SETUP.md`, `PHONE_CAMERA_SETUP.md`

- [ ] **Step 1: Search for stale references**

Run:
```bash
grep -n "app/\|app\." README.md HTTPS_SETUP.md PHONE_CAMERA_SETUP.md 2>/dev/null
```
Expected: empty (verified during planning — none of these reference the package path). If matches appear, edit them in place to the new paths.

- [ ] **Step 2: Search for references to `scripts/backend.py`**

Run:
```bash
grep -rn "scripts/backend\|scripts.backend" *.md 2>/dev/null
```
Expected: empty.

- [ ] **Step 3: If anything was edited, commit; otherwise mark this task done with no commit**

```bash
# Only if edits were needed:
git add README.md HTTPS_SETUP.md PHONE_CAMERA_SETUP.md
git commit -m "docs: update user-facing paths for lelab/ layout"
```

---

## Task 14: Final end-to-end verification

**Files:** none modified.

- [ ] **Step 1: Clean install from a fresh tree state**

Run:
```bash
git status --short
pip install -e ".[dev]" 2>&1 | tail -5
```
Expected: `git status` is clean (or shows only intended pre-existing untracked files); install succeeds.

- [ ] **Step 2: Smoke test passes**

Run:
```bash
pytest tests/ -v
```
Expected: 2 passed.

- [ ] **Step 3: Server boots and `/health` matches the Task 1 baseline**

Run:
```bash
lelab &
LELAB_PID=$!
sleep 5
curl -s http://127.0.0.1:8000/health
echo
python -c "from lelab.server import app; print('routes:', len(app.routes))"
kill $LELAB_PID
wait $LELAB_PID 2>/dev/null
```
Expected: `/health` body identical to Task 1; `routes:` value identical to Task 1 step 3.

- [ ] **Step 4: Dev mode boots both processes**

Run:
```bash
timeout 20 lelab --dev &
DEV_PID=$!
sleep 12
curl -s http://127.0.0.1:8080/ | head -1
curl -s http://127.0.0.1:8000/health
echo
kill $DEV_PID 2>/dev/null
wait $DEV_PID 2>/dev/null
```
Expected: HTML head from Vite (8080) and JSON `/health` from uvicorn (8000).

- [ ] **Step 5: Confirm git history was preserved on key files**

Run:
```bash
git log --follow --oneline lelab/server.py | head -5
git log --follow --oneline lelab/record.py | head -5
```
Expected: log shows commits predating this rename PR (i.e., from when these files lived under `app/`). If the log only shows the rename commit, `git mv` didn't preserve history — re-investigate Task 3.

- [ ] **Step 6: No final commit needed if Task 13 left nothing to do**

This task is verification only. If verification surfaces a bug, fix it in a new commit on this branch and re-run all of Task 14 before declaring done.

---

## Self-Review

**Spec coverage:**
- Goals 1 (mirror layout/naming) → Tasks 2, 3, 4, 5
- Goals 2 (tooling baseline) → Tasks 7, 8, 9, 10
- Goals 3 (mechanical, reviewable) → enforced by per-task commits
- Spec "File mapping (full)" table → Task 3 covers feature modules; Task 4 covers entry script; `app/static/` removal handled in Task 11; `replaying.py` open item resolved (does not exist) and removed from CLAUDE.md in Task 12
- Spec "Import-path updates" → Task 5 covers all internal; Task 6 covers the two uvicorn strings
- Spec "Affected files outside the package" → pyproject in Tasks 2/6/7/8/11; CLAUDE.md in Task 12; README/HTTPS/PHONE in Task 13; egg-info left alone (regenerates)
- Spec "Tooling baseline" → ruff (Task 8), pre-commit (Task 9), license headers (Task 10), tests (Task 7)
- Spec "Verification" — every concrete check from the spec maps to a step in Task 1, 6, 11, or 14

**Placeholder scan:** No "TBD"/"TODO"/"figure out" in any task step. Tasks 8 and 9 instruct fetching upstream config from a pinned SHA with explicit fallback skeletons — that's a documented contingency, not a placeholder.

**Type/name consistency:** All module names (`lelab.server`, `lelab.record`, `lelab.utils.config`, etc.) used identically across the import-rewrite table, the file-mapping table, and the per-task instructions. The console-script entry `lelab.scripts.lelab:main` matches between Task 6 and the spec.

**One nuance worth re-flagging during execution:** Task 5's `from . import datasets as dataset_browser` line is a deliberate compatibility alias — it lets the rest of `server.py` keep using the `dataset_browser` name without a sweep. If a reviewer prefers a clean rename, swap the alias for a follow-up where every `dataset_browser.X` usage in `server.py` is renamed to `datasets.X` (mechanical, no logic change).
