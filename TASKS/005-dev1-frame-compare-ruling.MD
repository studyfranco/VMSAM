# 005 — `frame_compare.py`: repair or replace, and say which before writing

**Agent:** `vmsam-dev-1`. **State:** open. **Do this before 006.**

## Goal

`src/frame_compare.py` does not compile — `frame_compare.py:109`, pre-existing
since `da1c191` — and **nothing in the live pipeline imports it or
`get_cut_time.py`**. Frame comparison is one of the two instruments that can
locate a cut. Decide whether to repair it or replace it, **and say which and why
before writing a line.**

The break is not subtle:

```python
costs =  * m          # line 109
hits  =  * m          # line 110
smoothed =  * m       # further down
```

A list literal was lost from three initialisers. So the *cost of repair* is
minutes; the ruling is not about that. It is about whether the method the file
implements — pHash on 32×32 grey frames, banded alignment, worst-cost window,
ffmpeg scene detection as a fallback — is the instrument we want, or whether it
should be replaced.

## What the ruling has to weigh

- `CAMPAIGNS/01-dev-AI/RESTART.MD` §10 argues **against** scene scores for this
  problem and **for** perceptual hashing, on the grounds that the same cut scores
  differently at two bitrates so two scene lists do not align. Note that
  `frame_compare.py` already *is* perceptual hashing, with scene detection only as
  a fallback — so §10 is closer to an argument for repairing it than against.
  That is my reading, not a ruling; make your own and give your reasons.
- `RESTART.MD` §10 is a campaign-1 opinion, and consulting it is not following it.
- The audio side may be the better instrument for a change point measured in
  points of 125 ms. Video is what pins it to a **frame**. Say which question you
  are answering.
- **Whatever you choose must be reachable.** A module nothing imports is exactly
  the campaign-1 failure. Name, in this file, the caller you intend — and note
  that the caller is dev-2's, so the two of you agree on the signature before
  either writes it.
- `get_cut_time.py` (565 lines, imports `frame_compare`) already contains
  `_refine_cut`, `_scene_refine_window`, `_refine_cut_audio`,
  `_build_chimeric_audios`. Read it before writing anything new. It is open, it
  has no importer, and it may be a better starting point than a blank file — or
  it may be dead weight. Rule on that too.

## Acceptance

The ruling, written here, **before** the first edit. Then: `python3 -m compileall
-q src -x '__pycache__'` fails on nothing, or fails only where you say it should
and you have said why.

## The ruling
*(write it here first)*

## What changed / measured / dropped / blocked
*(fill as you go)*
