---
description: Review new upstream (huggingface/leLab) commits for knowledge, bugs, and features worth bringing into MakerMods Lab
argument-hint: "[optional: an upstream commit range to review instead of the ledger cursor]"
---

# Review upstream (leLab)

MakerMods Lab began as a fork of [huggingface/leLab](https://github.com/huggingface/leLab)
and has diverged substantially — different package name (`lelab/` → `makermodslab/`),
different feature set, far higher change velocity. Git-level integration with upstream
is no longer possible, so we track them deliberately instead: once a week, someone runs
this command, reads what upstream has learned, and decides what is worth having here.

The most valuable thing upstream produces for us is usually **not their code** — it is
knowledge about the shared substrate we both sit on: lerobot's APIs, SO-101 hardware
behaviour, the Hugging Face Hub. A twelve-line fix can encode a fact that cost them a
debugging session with real arms. Capture the fact even when we decline the patch.

Both repositories are Apache-2.0. Record the upstream SHA for anything we port, and when
we fix something that also affects their code, consider sending it back.

## Ground rules

- **Never `git merge`, `git rebase`, or `git cherry-pick` from `upstream`.** The rename
  means upstream commits touch paths that do not exist here; every backend file would
  conflict. Ports are deliberate re-implementations against our code, done separately
  from this review.
- **This command changes no application code.** It reads, classifies, and writes a report
  plus proposed ledger rows. Nothing under `makermodslab/`, `frontend/src/`, or `tests/`
  may be edited by this command or its subagents.
- **Do not commit, push, or open a PR.** Leave the report and ledger edits as local
  changes for the person who ran the command to review.
- **Path mapping:** upstream `lelab/<x>.py` corresponds to our `makermodslab/<x>.py`,
  but the contents often differ heavily. Always read our actual file before judging
  whether an upstream change applies — never assume the mapping implies similarity.
- **"We already solved this, differently" is a real verdict, and often the correct one.**
  Some of our code is ahead of upstream's. Recommending a downgrade is worse than
  recommending nothing.

## Steps

### 1. Set up and fetch

Teammates cloning fresh will not have the remote. Ensure it, then fetch:

```bash
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/huggingface/leLab.git
git fetch upstream
```

### 2. Determine the review range

Read `docs/upstream/ledger.md` and take the SHA from its `cursor:` line. That is the
last upstream commit already reviewed.

```bash
git log --oneline --no-merges <cursor>..upstream/main
```

If the user passed an explicit range as an argument, use that instead of the cursor.

Skip any SHA that already appears in the ledger — it has a recorded verdict and must not
be re-litigated. If the range is empty, say so and stop; there is nothing to review.

### 3. Fan out subagents

Group the commits into batches of at most 6, keeping related commits together (same
subsystem, or an original plus its follow-up review fixes — upstream often ships
"Address review feedback" commits separately). Run at most 6 subagents concurrently,
launched in a single message.

Give every subagent this instruction verbatim, with its own batch substituted in:

> You are reviewing commits from `huggingface/leLab`, the upstream project that
> MakerMods Lab forked from and has since diverged from substantially. Your job is to
> decide what, if anything, we should learn or take from each commit. You are read-only:
> do not edit any file.
>
> Prefer running the analysis through codex if it is available on this machine
> (`command -v codex`), since a second model reads our code without the biases of the
> one that wrote it:
> `codex exec -s read-only --skip-git-repo-check "<self-contained prompt>"`.
> If codex is not installed, or it errors, do the analysis yourself and say which path
> you took in your report.
>
> For each commit in your batch:
>
> 1. Read the full upstream diff: `git show <sha>`.
> 2. Find our corresponding code. Upstream `lelab/<x>.py` maps to `makermodslab/<x>.py`,
>    but the contents frequently differ — read our file, do not assume.
> 3. Classify it into exactly one bucket:
>    - **knowledge** — teaches a fact about lerobot, SO-101 hardware, or the HF Hub that
>      is worth knowing even if we never take a line of their code
>    - **bug** — a defect that plausibly also exists in our diverged copy
>    - **feature** — functionality worth having in MakerMods Lab
>    - **already-have** — we solved this; note whether our approach differs and why
>    - **n/a** — leLab-specific, or against code we have deleted or rewritten past
>      applicability
> 4. For `bug` and `feature`, estimate porting effort as **small** (isolated, new files,
>    little conflict surface), **medium**, or **large** (touches files we have
>    substantially rewritten), and name the files a port would touch.
>
> Ground every claim in evidence: cite our `file.py:line` for anything you assert about
> our code. If you could not verify a claim, mark it explicitly as unverified rather than
> stating it. Never recommend replacing our implementation with upstream's unless you
> have read both and can say concretely why theirs is better.
>
> Return, for each commit: SHA, subject, bucket, effort (if applicable), affected files,
> the knowledge or defect in one or two sentences, and your recommendation.

### 4. Write the report

Write the findings to `docs/upstream/reviews/<YYYY-MM-DD>.md` (use `date +%F`), ordered
by bucket: `bug`, then `knowledge`, then `feature`, then `already-have`, then `n/a`.
Include the range reviewed and the commit count.

Then summarize in chat: the counts per bucket, and the two or three items that actually
deserve someone's attention this week. Do not restate the whole report.

### 5. Propose the ledger update

Append one row per reviewed commit to the table in `docs/upstream/ledger.md`, and move
the `cursor:` line to the newest reviewed upstream SHA.

Leave every edit uncommitted. Tell the user which files changed and that the port work
itself is a separate task.
