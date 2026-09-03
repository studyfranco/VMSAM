# Subtitle codecs — what we can re-time, and what we cannot

Taken from **our own ffmpeg build** and **our installed pysubs2**, not from a web
list, because the only list that matters is what this container can do.

## The three classes

**TEXT — cues carry their own timestamps, so they can be re-timed by rewriting
them.** ffmpeg decodes all of these and can encode to `srt` or `ass`:

    ass  ssa  subrip  srt  text  mov_text  webvtt  ttml  microdvd  sami
    jacosub  mpl2  pjs  realtext  stl  subviewer  subviewer1  vplayer
    hdmv_text_subtitle

**BITMAP — images with presentation timestamps. No cues to rewrite.**

    dvd_subtitle (VobSub)   hdmv_pgs_subtitle (PGS)   dvb_subtitle   xsub

**CAPTION / TELETEXT — timing embedded in a transport stream rather than in
discrete cues.** A different problem again, and none appear in this corpus:

    arib_caption   dvb_teletext   eia_608   ivtv_vbi

## What pysubs2 1.8.0 reads

    ass  json  microdvd  sami  srt  ssa  tmp  ttml  vtt

**This list is NOT the constraint.** Subtitles are extracted with ffmpeg before
pysubs2 sees them, so the real support set is *"any text codec ffmpeg can decode,
re-encoded to srt or ass"* — which is the whole text class above.

## What the code actually accepts today

`merge_video_chimeric._extract_and_shift_subtitles` carries an **allow-list**:

    ass, ssa                          -> extract as ass
    subrip, srt, text, mov_text       -> extract as srt
    everything else                   -> skipped, logged as "bitmap subtitle"

**Two consequences, and the second is the defect:**

1. **The allow-list covers 6 codec names of ~19 text codecs.** A WebVTT, TTML,
   MicroDVD, SAMI, SubViewer or VPlayer track is dropped, although ffmpeg could
   convert every one of them to `srt` and pysubs2 could then re-time it.
2. **The message calls all of them "bitmap".** A reader chasing a missing WebVTT
   track is told it was an image. **The label is a guess dressed as a
   diagnosis.**

**Measured over 45 corpus files with subtitles**: subrip 413, ass 357,
dvd_subtitle 4, hdmv_pgs_subtitle 2. **So the allow-list happens to cover this
corpus — that is luck, not design**, and a single WebVTT file would lose its
subtitles silently.

## How each class would be adapted

**A constant delay needs no adaptation at all, for any codec.** A container-level
`--sync` or `-itsoffset` shifts bitmap and text alike. **Only PIECEWISE re-timing
is codec-bound**, and piecewise is the rarer case.

| class | piecewise re-timing |
| :--- | :--- |
| text | rewrite the cues — **already implemented** |
| VobSub | **tractable**: the `.idx` sidecar is text and carries the timestamps, so the same technique applies after extracting to idx/sub |
| PGS | **not tractable by cue rewriting**: presentation timestamps live inside binary segments; it would need splitting and remuxing per piece, or a dedicated tool |
| DVB / xsub | as PGS |
| caption / teletext | different problem; not present in this corpus |

**Recommended**: widen the allow-list to the text class and convert to `srt`;
refuse bitmap **by name** rather than by parse failure, and count a declined
codec separately from a failed extract. **Do not build PGS re-timing.**

## OCR of bitmap subtitles: measured benefit is ZERO today

Full corpus, metadata only: **41 of 300 files with subtitles carry a bitmap
track; 37 carry ONLY bitmap.** That looked like a real population. It is not,
and three measurements close it:

- **36 of the 37 masters already supply TEXT subtitles**, in many languages. The
  37th has **no master recorded at all** — and *absent master* and *master
  without subtitles* are different facts, only one of which is evidence.
- **Zero files end with no subtitle track.** But the denominator is the finding:
  **only 1 of the 37 has a published output at all.** The other 36 are the
  population Phase 2 closed — they fail on **7 to 126 seconds of unaccounted
  content** (§11). **They do not lose subtitles because they never produce a
  file.**
- **Zero carry a language the master lacks.** The only apparent case is the file
  with no master, so the comparison has no second term.

**And the 37 are 3 series-groups of 26 / 10 / 1 — not 37 independent
observations.**

**A PREMISE OF THIS DOCUMENT WAS WRONG.** The claim that bitmap subtitles are
skipped and the candidate contributes none holds for the **chimeric** path. **The
one file that did publish carries 53 subtitle streams, two of them
`hdmv_pgs_subtitle`** — so the plain merge path **passes bitmap through**. The
decline is path-specific and that had not been established.

**So OCR would benefit no file on today's evidence.** That is a statement about
the corpus **now**, not about the feature: if those 36 ever merge — they currently
cannot, for reasons unrelated to subtitles — **re-ask the question against their
OUTPUTS rather than their candidates.**
