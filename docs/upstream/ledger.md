# Upstream ledger — huggingface/leLab

MakerMods Lab forked from [leLab](https://github.com/huggingface/leLab) and has diverged
past the point where git-level integration works: the package rename (`lelab/` →
`makermodslab/`) means upstream commits touch paths that no longer exist here, and our
change volume is roughly twenty times theirs. We do not merge from upstream. We read it.

This file is the record of that reading. Every upstream commit we have looked at gets a
row and a verdict, so that:

- "have we already considered this?" is a `grep`, not a conversation;
- a settled **rejected** decision stays settled, instead of being rediscovered every
  few months by whoever next reads upstream's log;
- anything we port carries the upstream SHA, which is both good provenance and what
  Apache-2.0 attribution looks like in practice.

Update it by running `/review-upstream` — weekly is the intended cadence. Do not edit the
cursor by hand unless you are deliberately re-reviewing a range.

```
cursor: 8f8a50f
```

The cursor is the newest upstream commit that has been reviewed. `8f8a50f`
("Force-release camera/serial resources when a device disconnect fails", 2026-06-24) is
the last commit our history and upstream's share, so the first run reviews everything
they have shipped since the fork diverged in earnest.

## Verdicts

| Verdict | Meaning |
|---|---|
| `ported` | Re-implemented here. Note the PR or commit that did it. |
| `knowledge` | We took the lesson, not the code. Note where the lesson landed (CLAUDE.md, a code comment, a test). |
| `already-have` | We had solved it. Note if our approach differs, so nobody "fixes" it back. |
| `rejected` | Considered and declined. **The reason is the point of the row.** |
| `n/a` | Does not apply to our fork — leLab-specific, or against code we have rewritten. |
| `todo` | Worth having, not yet done. Should have a tracking issue. |

## Log

| Upstream SHA | Date | Subject | Verdict | Notes |
|---|---|---|---|---|
| `12ad202` | 2026-07-17 | fix(security): remediate workflow vulnerability in build_frontend.yml | `rejected` | Their vuln was a token interpolated into a push URL. We never had it — `build_frontend.yml` mints a short-lived GitHub App token, hands it to `actions/checkout` with `persist-credentials`, and pushes with a bare `git push origin`, so the token never reaches a command string. Their fix (`gh auth login --with-token`) also broke their dist push and was partly reverted in `a654148`. **Do not port; it would be a downgrade.** |
