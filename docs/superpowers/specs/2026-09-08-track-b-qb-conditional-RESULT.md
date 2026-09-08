# Track B result: the QB block does NOT help on the games where it fires — REJECTED

**Run 2026-09-08** against the pre-registration in
`2026-09-08-track-b-qb-conditional-prereg.md`, committed at `0c6cd4f` before the
measurement script existed. Reproduce with
`./.venv/Scripts/python.exe scripts/track_b_qb_conditional.py`.

## The control, read before any candidate

**A0 reproduces the shipped Ridge-v1 baseline exactly**: pooled margin **10.273978**
against `CLAUDE.md`'s **10.273977625706554**, and totals **10.683825** against
**10.684**. The walk-forward here is reimplemented only because `GameModel` hardcodes
`FEATURE_COLS`; matching the published figure to six decimals is what makes the
candidate numbers below worth reading at all.

## Result

| arm | pooled margin MAE | change-subset MAE (n=302) |
| --- | ---: | ---: |
| A0 — shipped baseline | 10.273978 | 9.786236 |
| A1 — `+ qb_change_epa_diff` | 10.274482 | 9.790966 |
| A2 — `+ full C2` (8 cols) | 10.254631 | 9.832568 |

Change-subset margin MAE by season:

| arm | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 10.1828 | 8.0865 | 10.2910 | 10.0292 | 10.1498 |
| A1 | 10.1727 | 8.3503 | 10.1353 | 10.0547 | 10.0703 |
| A2 | 10.3875 | 8.3596 | 10.0670 | 10.0150 | 10.1633 |

Against the pre-registered rule:

| arm | change mean | lower90 | seasons improved | pooled delta | **ADOPT** |
| --- | ---: | ---: | ---: | ---: | --- |
| A1 | −0.008804 | −0.117521 | 3 of 5 | −0.000504 | **No** |
| A2 | −0.050593 | −0.185238 | 2 of 5 | +0.019347 | **No** |

All five seasons scored, none dropped — the common fold set held.

## The motivating hypothesis is falsified, and inverted

This track existed because the QB block had only ever been judged on **pooled** MAE,
and `qb_new_starter_any` fires on 22.2% of games. The premise was that a block right on
that 22% and noisy on the other 78% would lose pooled while being exactly what a bettor
wants.

Measured, the shape is the opposite. **A2 is slightly BETTER pooled (+0.019) and WORSE
on the change subset (−0.051).** Whatever the level features contribute, it is not
concentrated where the starter changed. The QB signal carries nothing the team ratings
do not already hold, even on the games where the starter demonstrably changed.

**Change games are also easier, not harder** — 9.786 against 10.274 pooled. The
intuition that a QB change makes a game less predictable is not supported here.

The prereg's falsification condition was "change-subset margin MAE unchanged or worse
for both arms", which is what happened. **This closes the question rather than deferring
it.**

## Secondary

- `corr(qb_epa_diff, net_rating_diff) = 0.589` — the redundancy mechanism proposed for
  A2's level features is real, not assumed. They partly re-describe the ratings.
- Totals, pooled: A0 10.683825, A1 10.689626, A2 10.715667. Both arms worse.

## Power, stated honestly

302 change games over 5 seasons. The lower90 spans roughly ±0.12, so this rules out
subset improvements larger than about 0.10 — not arbitrarily small ones. A real effect
below that floor would not be detected here, and would also be too small to act on.

## Consequence

`FEATURE_COLS` does not change. The shipped model stays as it is, and the decision to
build the QB advisory as **presentation only** — surfacing the starter without moving
the number — is the one this evidence supports.

The remaining Track B question, whether a pick published five days out could even see a
starter change, is now moot for model purposes. It would only matter if the signal had
value, and it does not.
