#!/usr/bin/env python3
"""SPEC_ZONE_A §4d acceptance: judge the ARTEFACT, not the exit status.

WHY. A repair reported "7 audio and 24 subtitle track(s) rebuilt, 0 declined, 0 failed"
and shipped a file with nothing after 21:21. EVERY COUNTER SAID SUCCESS. A count of
rebuilt tracks is a statement about work done, not about a file -- and the instrument
that decides "repaired" is the one that would have to detect "damaged", so it never
looks at the output.

WHAT IT CHECKS, per track, against the master:
  - track present at all
  - track duration against the master's, as a RATIO and as a shortfall in seconds
  - the shortest track, which is where truncation shows first

RETURNS A NUMBER PER TRACK AND A VERDICT, and stores both. The verdict can be wrong;
the numbers underneath are what let someone overturn it without re-measuring. That has
already saved two of my results tonight.

COULD-NOT-MEASURE IS A VALUE. A file ffprobe cannot read is not a failed file.
"""
import json, subprocess, sys

TOL = 0.02          # 2 % shortfall tolerated: container rounding, not truncation


def streams(path):
    q = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=start_time,index,codec_type:stream_tags=language,DURATION",
                        "-of", "json", path], capture_output=True, timeout=120)
    try:
        return json.loads(q.stdout.decode()).get('streams', [])
    except Exception:
        return None


# ISO 639-2/B vs 639-1. THE MASTER AND THE PRODUCED FILE DO NOT AGREE ON CODE LENGTH.
# id 131's master tags its audio `fre` and `jpn`; the produced file tags the same content
# `fr` and `ja`. My attribution did `master_spans.get(produced_lang)` -- an EXACT match --
# so all seven languages returned None and every row read CANNOT_ATTRIBUTE.
#
# THE VERDICT WAS STILL `FULL_LENGTH` WITH ZERO SHORT TRACKS. A clean, confident answer
# from an instrument that had not made the comparison -- the same shape as the `0 of 14`
# encoder join, and I caught it only because CANNOT_ATTRIBUTE appeared SEVEN TIMES and
# seven is not a plausible number of missing languages.
#
# Bites only when the two files disagree on code length, which is why earlier files
# attributed correctly and this one did not.
_LANG3TO2 = {
    'fre': 'fr', 'fra': 'fr', 'jpn': 'ja', 'ger': 'de', 'deu': 'de', 'spa': 'es',
    'ita': 'it', 'por': 'pt', 'eng': 'en', 'cze': 'cs', 'ces': 'cs', 'hun': 'hu',
    'pol': 'pl', 'dut': 'nl', 'nld': 'nl', 'rus': 'ru', 'kor': 'ko', 'chi': 'zh',
    'zho': 'zh', 'ara': 'ar', 'tha': 'th', 'vie': 'vi', 'tur': 'tr', 'swe': 'sv',
    'dan': 'da', 'fin': 'fi', 'nor': 'no', 'heb': 'he', 'hin': 'hi', 'ind': 'id',
}


def norm_lang(v):
    """One key for one language, whichever code length the file used."""
    if not v:
        return v
    v = v.strip().lower()
    return _LANG3TO2.get(v, v)


def dur_of(s):
    """Duration in seconds from the stream's own tag, or None."""
    t = (s.get('tags') or {})
    v = t.get('DURATION') or t.get('duration')
    if not v:
        return None
    try:
        h, m, rest = v.split(':')
        return int(h) * 3600 + int(m) * 60 + float(rest)
    except Exception:
        try:
            return float(v)
        except Exception:
            return None


def check(out_path, master_path):
    o, m = streams(out_path), streams(master_path)
    if o is None or m is None:
        return {"verdict": "COULD_NOT_MEASURE", "why": "ffprobe failed"}
    # THE REFERENCE IS THE MASTER'S VIDEO STREAM DURATION, agreed with the repair's
    # author before either of us measured, so a units difference cannot masquerade as
    # a finding. Its reason is the decisive one: the plan's last segment end is
    # DERIVED FROM THE MEASUREMENT, so a check against it agrees with the defect when
    # the plan is itself short. The master's duration is the one reference neither
    # side computed.
    #
    # MEASURED, NOT ASSUMED, on 5 masters: ffprobe's VIDEO STREAM DURATION equals
    # mediainfo's video Duration exactly, 5 of 5 -- so the repair's number and this
    # one are the same number. ffprobe's format=duration differs from both by 31 to
    # 883 ms and IS A THIRD VOCABULARY NEITHER OF US SHOULD USE.
    vids = [x for x in m if x['codec_type'] == 'video' and dur_of(x)]
    mref = dur_of(vids[0]) if vids else 0
    if mref <= 0:
        return {"verdict": "COULD_NOT_MEASURE", "why": "no master duration"}
    rows, short = [], []
    for s in o:
        if s['codec_type'] not in ('audio', 'subtitle', 'video'):
            continue
        d = dur_of(s)
        try:
            _st = float(s.get('start_time'))
        except (TypeError, ValueError):
            _st = None
        rec = {"idx": s['index'], "type": s['codec_type'],
               "lang": norm_lang((s.get('tags') or {}).get('language')),
               # `dur_s` IS ROUNDED TO 2 dp FOR READABILITY AND MUST NOT BE THE INPUT
               # TO A MILLISECOND ARITHMETIC. It was, and it corrupted every deficit:
               # id 47's eng produced track is 1324.032 s; rounded to 1324.03 it yields
               # deficit 397.0 ms where the true value is 395.0, and fre's 1323.968 ->
               # 1323.97 yields -2.0 where the truth is 0.0 EXACTLY.
               #
               # `vmsam-forensic` reported the 2 ms disagreement between us and noted
               # it was NOT CONSTANT, so not a basis offset. Correct, and that is the
               # signature of rounding rather than a systematic error: the sign follows
               # whichever way each value rounds. The bound is +-5 ms.
               #
               # A QUANTITY REPORTED TO 0.1 ms COMPUTED FROM AN INPUT ROUNDED TO 10 ms.
               # The precision of the output said nothing about the precision of the
               # measurement, and it was mine that was wrong, not theirs.
               "dur_s": None if d is None else round(d, 2),
               "dur_s_exact": d,
               "start_s": None if _st is None else round(_st, 3),
               "ratio": None if d is None else round(d / mref, 4)}
        rows.append(rec)
        # SUBTITLES LEGITIMATELY END EARLY -- a subtitle track ends at its last cue,
        # not at the programme's end. The negative control proved this the hard way:
        # a known-good delivered file reported TRUNCATED on 24 short tracks, ALL OF
        # THEM SUBTITLES, while every audio track sat at ratio 1.0. Judging subtitles
        # on duration would flag every well-formed file in the corpus.
        # AUDIO AND VIDEO ARE CONTINUOUS MEDIA AND MUST RUN THE LENGTH. Truncation of
        # the kind that shipped -- nothing after 21:21 -- stops the AUDIO, and that is
        # what this fires on.
        if (s['codec_type'] in ('audio', 'video')
                and d is not None and d < mref * (1 - TOL)):
            short.append(rec)
    n_aud = sum(1 for r in rows if r['type'] == 'audio')
    n_sub = sum(1 for r in rows if r['type'] == 'subtitle')
    nodur = [r for r in rows if r['dur_s'] is None]
    subs_short = [r for r in rows if r['type'] == 'subtitle'
                  and r['ratio'] is not None and r['ratio'] < 1 - TOL]
    verdict = ("TRUNCATED" if short else
               "COULD_NOT_MEASURE" if len(nodur) == len(rows) else "FULL_LENGTH")
    return {"verdict": verdict, "master_dur_s": round(mref, 2),
            "n_audio": n_aud, "n_subtitle": n_sub,
            "n_without_duration": len(nodur),
            # REPORTING REQUIREMENT, not a footnote. A duration read from the
            # Matroska DURATION tag is exact; one read any other way carries a bias
            # measured at 42-85 ms per track and VARYING BETWEEN TRACKS OF ONE FILE,
            # so it does not subtract out. A bound that holds on paper and is quietly
            # halved in practice is the shape this campaign keeps finding, so the
            # COUNT travels with every row.
            "n_duration_from_tag": len(rows) - len(nodur),
            "n_duration_missing": len(nodur),
            "n_subtitle_short": len(subs_short),   # reported, NOT judged
            "shortest_av_ratio": min((r['ratio'] for r in rows
                                      if r['type'] in ('audio', 'video')
                                      and r['ratio'] is not None), default=None),
            "reference": "master video stream DURATION (== mediainfo video Duration)",
            "delta_ms_by_track": [
                {"idx": r['idx'], "type": r['type'], "lang": r['lang'],
                 "delta_ms": None if r.get('dur_s_exact') is None
                             else round((r['dur_s_exact'] - mref) * 1000, 1)}
                for r in rows],
            # UNIFORM MEANS "WITHIN A WINDOW", NOT "EQUAL". The positive control --
            # a container-level `-t` truncation, which stops everything at once by
            # construction -- gave A/V stops 220 ms apart, because audio frame
            # boundaries do not align with a cut instant. Exact equality would have
            # reported the one case we KNOW is uniform as non-uniform.
            # SPAN, NOT ENDPOINT -- AND MEASURED RATHER THAN ASSUMED. A peer retracted
            # a 1988 ms finding after discovering it compared LAST-PACKET times across
            # tracks that do not start together: a track starting 1.103 s late has an
            # early endpoint and has lost nothing. That number looked RIGHT -- it sat
            # inside a two-second band and agreed with two figures of mine.
            # So I checked whether mine inherit it. THE MATROSKA DURATION TAG IS A SPAN:
            # on a produced file the tag reads 1432.041 against last-first 1432.020,
            # NOT against last 1431.999. av_stop_spread and shortest_av_ratio therefore
            # compare spans and do not inherit the defect.
            # BUT every track on that file starts at ~0, so it CANNOT discriminate the
            # two. The start offsets now travel in the row, so a later reader can tell
            # whether span and endpoint would have differed ON THIS FILE instead of
            # trusting a check that was run on a different one.
            # THE MASTER'S OWN AUDIO SPANS, SO CAUSATION IS DECIDABLE FROM THE ROW.
            # A peer withdrew "the source is level, SO THE MERGE INTRODUCED IT" after
            # finding the claim assumed the produced track came from the CANDIDATE --
            # but the mechanism is FILL FROM THE MASTER, so a short produced track is
            # equally consistent with the merge faithfully copying a master track that
            # was ALREADY SHORT. On the file that settled it, that is exactly what it
            # was: master short by 1962 ms, produced short by 1988, ACTUALLY
            # UNEXPLAINED 26 ms rather than the 1081 first reported.
            # My rows compared produced tracks against the master's VIDEO duration and
            # never recorded the master's per-track AUDIO spans -- so they could show a
            # deficit and never say whether the pipeline caused it. Recording both ends
            # is the difference between an observation and an attribution.
            "master_audio_spans_s": [
                {"idx": x['index'], "lang": (x.get('tags') or {}).get('language'),
                 "dur_s": None if dur_of(x) is None else round(dur_of(x), 3)}
                for x in m if x['codec_type'] == 'audio'],
            "master_audio_spread_ms": (lambda v: None if len(v) < 2
                                       else round((max(v) - min(v)) * 1000, 1))(
                [dur_of(x) for x in m
                 if x['codec_type'] == 'audio' and dur_of(x) is not None]),
            # PER-LANGUAGE ATTRIBUTION, WHICH IS THE TEST A SPREAD CANNOT DO. When the
            # master has ONE audio track there is no spread to compare against, and the
            # attribution is against THAT TRACK. This is what the peer computed by hand
            # on the file that overturned its own finding: produced minus the matching
            # master span, per language. Positive means the pipeline LOST tail the
            # master had; ~0 means it faithfully copied a short source.
            # `unattributable_no_master_lang` is its own outcome, never a zero: a
            # produced language the master does not carry cannot be attributed at all.
            "deficit_vs_master_ms": (lambda mm: [
                {"idx": r['idx'], "lang": r['lang'],
                 "master_span_s": mm.get(r['lang']),
                 "deficit_ms": (None if (r.get('dur_s_exact') is None or mm.get(r['lang']) is None)
                                else round((mm[r['lang']] - r['dur_s_exact']) * 1000, 1)),
                 "unattributable_no_master_lang": r['lang'] not in mm}
                for r in rows if r['type'] == 'audio'])(
                {norm_lang((x.get('tags') or {}).get('language')): dur_of(x)
                 for x in m if x['codec_type'] == 'audio' and dur_of(x) is not None}),
            # THE MASTER'S OWN AUDIO/PICTURE DISAGREEMENT, PER LANGUAGE.
            # `vmsam-forensic` had to derive this BY HAND to validate id 47, because
            # the row carried all three numbers and no field that put them together.
            # A quantity a reader must reconstruct is a quantity most readers will not
            # reconstruct.
            #
            # POSITIVE = the master's audio for that language OVERHANGS the master's own
            # picture. The pipeline builds to the picture, so a produced deficit of the
            # same size is the REMOVAL OF AN OVERHANG, not lost content. On id 47:
            # master eng overhangs by 396 ms and the produced deficit was 397.
            "master_audio_overhang_ms": (lambda mm: [
                {"lang": k, "overhang_ms": (None if (v is None or not mref)
                                            else round((v - mref) * 1000, 1))}
                for k, v in mm.items()])(
                {norm_lang((x.get('tags') or {}).get('language')): dur_of(x)
                 for x in m if x['codec_type'] == 'audio' and dur_of(x) is not None}),
            # THE ATTRIBUTION ITSELF, COMPUTED. A note telling a reader how to combine
            # two fields is a rule the reader has to execute, and I executed my own
            # wrongly the first time: I compared deficit against overhang ONLY, and
            # flagged a benign fre track as a LOSS at deficit -2 / overhang -63.
            #
            # THERE ARE TWO BENIGN CASES AND THE COMMENT FORTY LINES UP ALREADY NAMED
            # BOTH -- "positive means the pipeline LOST tail the master had; ~0 means it
            # faithfully copied a short source". My rule implemented one of them.
            #
            #   deficit ~ overhang  the master's audio overhung its own picture and the
            #                       pipeline built to the picture. The tail removed was
            #                       never in the picture. (id 47 eng: 397 vs 396)
            #   deficit ~ 0         the pipeline reproduced the master's own track,
            #                       including its shortfall. (id 47 fre: -2, against a
            #                       master fre already 63 ms short of its own picture)
            #   neither             a real loss, and the only case worth an argument.
            #
            # Both readings use master, produced and picture on ONE basis, so neither
            # depends on the unadjudicated span-vs-endpoint question.
            "attribution_tolerance_ms": 10.0,
            "attribution": (lambda mm: [
                (lambda dfc, ovr: {
                    "lang": r['lang'],
                    "deficit_ms": dfc, "overhang_ms": ovr,
                    # FIVE OUTCOMES, NOT FOUR. When the deficit is small AND close to
                    # the overhang, BOTH benign explanations fit and an `if/elif` picks
                    # the first silently. Measured on ids 55 and 58: deficit -4.0 against
                    # overhang +6.0 -- |dfc-ovr| is EXACTLY 10.0 and |dfc| is 4.0, so both
                    # branches are true and the row asserted a mechanism it had not
                    # determined. Benign either way; the REASON was invented.
                    #
                    # And note what this exposes about the instrument: at a 10 ms
                    # tolerance against values of 4 to 6 ms, THE TOLERANCE IS LARGER THAN
                    # THE QUANTITY. The classifier cannot discriminate at this scale and
                    # should say so rather than choose.
                    # A NEGATIVE DEFICIT IS NOT A LOSS. `deficit = master - produced`,
                    # so a NEGATIVE value means the produced track is LONGER than its
                    # source. My rule called any |deficit| > tolerance a LOSS, and I even
                    # wrote a control asserting "produced longer than master -> LOSS" and
                    # accepted it.
                    #
                    # Measured on id 131's REFUSED artefact: fr -20.0 ms and ja -21.0 ms
                    # -- both LONGER than their sources, and ~21.33 ms is ONE AAC FRAME
                    # at 48 kHz, which `vmsam-forensic` established is what a re-encoded
                    # track carries. Calling that "LOSS" names the wrong fault and points
                    # at the wrong half of the pipeline.
                    #
                    # LOSS and EXCESS are different defects with different causes. One
                    # token for both is the two-valued read of a three-valued field, again.
                    "verdict": ("CANNOT_ATTRIBUTE" if (dfc is None or ovr is None) else
                                "BENIGN_AMBIGUOUS" if (abs(dfc - ovr) <= 10.0 and abs(dfc) <= 10.0) else
                                "BENIGN_MASTER_OVERHANG_REMOVED" if abs(dfc - ovr) <= 10.0 else
                                "BENIGN_FAITHFUL_SHORT_SOURCE" if abs(dfc) <= 10.0 else
                                "EXCESS_PRODUCED_LONGER" if dfc < 0 else
                                "LOSS")})(
                    (None if (r.get('dur_s_exact') is None or mm.get(r['lang']) is None)
                     else round((mm[r['lang']] - r['dur_s_exact']) * 1000, 1)),
                    (None if (mm.get(r['lang']) is None or not mref)
                     else round((mm[r['lang']] - mref) * 1000, 1)))
                for r in rows if r['type'] == 'audio'])(
                {norm_lang((x.get('tags') or {}).get('language')): dur_of(x)
                 for x in m if x['codec_type'] == 'audio' and dur_of(x) is not None}),
            # THIS NOTE WAS WRONG AND IT CONTRADICTED THE FIELD ABOVE IT.
            # It said: "a produced deficit is NOT attributable unless it exceeds the
            # master's own spread -- compare av_stop_spread_ms against
            # master_audio_spread_ms."
            #
            # `master_audio_spread_ms` is a disagreement BETWEEN TWO MASTER TRACKS.
            # `deficit_vs_master_ms` is a per-track change from master to produced.
            # THEY ARE DIFFERENT QUANTITIES, and the rule would have passed a REAL
            # 397 ms loss exactly as readily as the benign one it passed on id 47.
            # `vmsam-forensic` broke it on request and was right; it is the same shape
            # as their own retracted 1081 ms -- a statistic that cannot separate two
            # cases used to decide between them.
            #
            # Worse than wrong: the comment forty lines up already says a spread CANNOT
            # do this test, and the note then sent readers to the spread anyway.
            "causation_note": ("READ deficit_vs_master_ms, PER LANGUAGE. Do NOT use "
                               "master_audio_spread_ms for attribution -- it is a "
                               "disagreement between two master tracks, not a change "
                               "from master to produced, and it cannot separate a real "
                               "loss from a benign one. A deficit is BENIGN when it "
                               "matches master_audio_overhang_ms for the same language "
                               "(the master's audio overhung its own picture and the "
                               "pipeline built to the picture); it is a LOSS when it "
                               "does not. Both readings use the same basis, so neither "
                               "depends on the unadjudicated span-vs-endpoint question."),
            "start_time_by_track": [{"idx": r['idx'], "type": r['type'],
                                     "start_s": r.get('start_s')} for r in rows],
            "max_start_offset_ms": (lambda v: None if len(v) < 2
                                    else round((max(v) - min(v)) * 1000, 1))(
                [r['start_s'] for r in rows
                 if r['type'] in ('audio', 'video') and r.get('start_s') is not None]),
            "av_stop_spread_basis": "SPAN (Matroska DURATION tag), not endpoint",
            "av_stop_spread_ms": (lambda v: None if len(v) < 2
                                  else round((max(v) - min(v)) * 1000, 1))(
                [r['dur_s'] for r in rows
                 if r['type'] in ('audio', 'video') and r['dur_s'] is not None]),
            "shortest_ratio": min((r['ratio'] for r in rows
                                   if r['ratio'] is not None), default=None),
            "short_tracks": short, "tracks": rows}


if __name__ == '__main__':
    print(json.dumps(check(sys.argv[1], sys.argv[2]), indent=1))
