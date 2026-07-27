# Task briefs and reports

Supporting material for [`../development-log.md`](../development-log.md). Same caveat
applies, more strongly: **this is a historical record, not documentation.** `CLAUDE.md`
is the only authoritative description of how the project currently works, and several
decisions described here were reversed by later tasks or by the final whole-branch review.

Each numbered task has a *brief* (what the implementer was asked to do, written before the
work) and usually a *report* (what they actually did, written after). Reading a brief and
its report together is the clearest surviving account of any single decision — the ledger
compresses each one to a line or two.

Two gaps worth knowing about:

- **There is no `task-8-report.md`.** Task 8 was executed inline rather than dispatched to
  a subagent, so no report was produced. The ledger's Task 8 entry is the only record.
- **`task-11-fix-brief.md`** is a second brief for Task 11, covering the fix pass after its
  first review, and is what the re-reviewer was given.

Frozen at commit `352b1a3` (2026-07-27) from `.superpowers/sdd/`, which was gitignored for
its whole life and existed only on one machine.

## Review diffs: not archived, regenerate on demand

The review packages were plain `git diff` output over commit ranges that are all still
reachable in this repository, so archiving them would have committed ~340K of derived
bytes. Regenerate any of them with the command below.

The last row is the final whole-branch review — the one that produced the Important
findings I1–I3, the `MissingRatingJoinError` guard, and the deferred-Minor triage that was
finally cleared on 2026-07-27.

| Original filename | Regenerate with | Size |
| --- | --- | --- |
| `review-bad39fd..4dd8bf4.diff` | `git diff bad39fd..4dd8bf4` | 8.0K |
| `review-4dd8bf4..1d92962.diff` | `git diff 4dd8bf4..1d92962` | 8.0K |
| `review-1d92962..b3b5108.diff` | `git diff 1d92962..b3b5108` | 12K |
| `review-1d92962..c6b9379.diff` | `git diff 1d92962..c6b9379` | 8.0K |
| `review-b3b5108..4df1e98.diff` | `git diff b3b5108..4df1e98` | 8.0K |
| `review-4df1e98..a6efaeb.diff` | `git diff 4df1e98..a6efaeb` | 8.0K |
| `review-a6efaeb..2bf6cba.diff` | `git diff a6efaeb..2bf6cba` | 12K |
| `review-5ddf154..2748b80.diff` | `git diff 5ddf154..2748b80` | 16K |
| `review-2748b80..814fdbd.diff` | `git diff 2748b80..814fdbd` | 20K |
| `review-814fdbd..8630478.diff` | `git diff 814fdbd..8630478` | 8.0K |
| `review-8630478..7b62ab9.diff` | `git diff 8630478..7b62ab9` | 12K |
| `review-7b62ab9..44fb9c1.diff` | `git diff 7b62ab9..44fb9c1` | 20K |
| `review-44fb9c1..aa6bf4e.diff` | `git diff 44fb9c1..aa6bf4e` | 44K |
| `review-c00f5df..bad39fd.diff` | `git diff c00f5df..bad39fd` | 8.0K |
| `review-c00f5df..130e7a3.diff` | `git diff c00f5df..130e7a3` | 136K |
