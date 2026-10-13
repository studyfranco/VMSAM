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

## The classification lives in `tools`, and nowhere else — owner's ruling

**Refuse by exclusion, not by allow-list.** A codec absent from an allow-list is
dropped silently and reported as an image; a codec absent from an *exclusion* list is
attempted. The second fails loudly on a real defect, the first fails quietly on a
guess.

    tools.sub_type_not_encodable    the only bitmap set. Not attempted.
    tools.sub_type_near_srt         written back as srt.
    everything else                 written back as ass.

**Two parallel definitions exist today and they disagree in both directions.**
`merge_video_chimeric` declares its own sets rather than reading `tools`:

    knows dvb_subtitle and xsub, which tools does not     -> the merge path would
                                                             attempt a DVB subtitle
    lacks s_hdmv/pgs, pgs, vobsub, s_vobsub, which tools has  -> the repair path does
                                                             not recognise a stream
                                                             Matroska reports as s_vobsub

**The repair module reads `tools`; it does not keep a copy.** Two sets that must
agree and are maintained apart is the same defect as an unexamined constant copied
three times, and it will diverge again.

**`dvb_subtitle` and `xsub` are missing from `tools.sub_type_not_encodable`** and
adding them touches a frozen line — a finding for the owner rather than an edit.

## `to_convert_ffmpeg_type` — add on evidence, never on inventory

`webvtt` is converted before the calculations **because it caused a bug**, which is
the right reason. The text class holds seventeen codecs ffmpeg can convert, and
adding the rest speculatively is how a list stops being evidence. **Add one when a
codec is measured to break something**, and say which file showed it.

## OCR of bitmap subtitles — moved

**The measured OCR analysis (corpus counts, the path-specific decline, the premise that was wrong)
moved to `CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.** Its conclusion when taken: OCR
would have benefited no file — a statement about that corpus at that time, not about the feature.
If the population blocked on unaccounted content ever merges, re-ask the question against their
OUTPUTS rather than their candidates.
