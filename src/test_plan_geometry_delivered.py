'''Tests for merge_plan_report.plan_geometry's lead track -- run: python3 src/test_plan_geometry_delivered.py
(or pytest). No media: the rows are the report's own, shaped as id 54's product (its only rebuilt eng track
dropped by the delivery gate, intact_same_language_wins; the delivered video, eng and jpn md5-identical to the
master), as id 718's product (lead language `ja` entirely dropped, `en` delivered), and as a pair where one
language's track is dropped and another is delivered.

A `fabricated_dropped` line's `stream=` is NOT the TRACK row's `track=` -- adb47eeb assumed they were the same
identifier and got id 718 wrong (the plan-mismatch watch re-captured it: `track=1` is `en`, kept,
`fabricated_kept stream=0`; `track=2` is `ja`, dropped, `fabricated_dropped stream=1` -- matching `stream=`
against `track=` directly excluded the DELIVERED `en` and left the DROPPED `ja` leading, the exact bug this
file exists to catch). `stream=` is the 0-based RANK of a track among the rebuilt audio tracks in build order:
`track=` is `report["stream_order"]`, the CANDIDATE file's own StreamOrder (video precedes the audios there);
`stream=` is `audio.get("StreamOrder")` read off the delivery gate's `repaired_obj`, which probes the file
`mux_repaired_file` just muxed from ONLY the rebuilt audio (and subtitle) tracks -- no video input at all
(merge_video_chimeric.py:1878-1910) -- so its audio tracks are numbered 0, 1, 2... in exactly the order the
TRACK rows were written (merge_plan_report.py's `delivery_drops` carries the full derivation, with line
numbers). `test_id_718_...` below uses id 718's own captured values (forensic/cases/718/LAST_RUN.json,
span_ms=1421420.0, tracks 1=en/2=ja) and the plan-mismatch watch's traced `stream=0`/`stream=1`.'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import merge_plan_report as mpr  # noqa: E402

PLAN = "PLAN master_end_ms=1452410"
REGIONS_TRACK1 = ["REGION track=1 kind=CANDIDATE master_start_ms=0 master_end_ms=500000 offset_ms=-1000",
                  "REGION track=1 kind=MASTER master_start_ms=500000 master_end_ms=501001 source=[master/en]",
                  "REGION track=1 kind=CANDIDATE master_start_ms=501001 master_end_ms=1452410 offset_ms=-2001"]
REGIONS_TRACK2 = ["REGION track=2 kind=CANDIDATE master_start_ms=0 master_end_ms=1452410 offset_ms=-1000"]


def regions_for(rows, track):
    '''REGIONS_TRACK1/2 retargeted at a different `track=` number, order preserved.'''
    source = "track=1" if rows is REGIONS_TRACK1 else "track=2"
    return [r.replace(source, f"track={track}") for r in rows]


def drop_line(lang, stream, cause="intact_same_language_wins", format_="E-AC-3"):
    '''A `repair: fabricated_dropped ...` line as merge_video_repair.py:931 (`where`,
    merge_video_repair.py:869-871) actually writes it, `stream=` the RANK
    delivery_drops derives, never the TRACK row's `track=`.'''
    return (f"UNPARSED line=[fabricated_dropped cause={cause} lang={lang} holder=audios "
            f"stream={stream} format={format_} marker=chimeric kept_master]")


def records(rows):
    return mpr.parse_rows(rows)


def test_id_54_no_delivered_track_is_no_geometry_and_says_so():
    # one TRACK, track=1: rank 0. Its only rebuilt track dropped -> stream=0.
    rows = ([PLAN, "TRACK track=1 kind=audio lang=en fill=[master/en] offset=measured"] + REGIONS_TRACK1
            + [drop_line("en", 0)])
    geometry = mpr.plan_geometry(records(rows))
    assert geometry["undelivered"] and geometry["dropped"] == ["en"], geometry
    assert "Aucune piste livrée" in mpr.render_plan_schematic(geometry)
    job = {"audios": {1: {"lang": "en"}}, "summary_counts": "", "subtitles": [], "delivery": []}
    assert "aucune piste livrée" in mpr.render_human_summary(job, geometry)


def test_the_delivered_track_leads_when_the_first_is_dropped():
    # track=1 (en, rank 0) dropped -> stream=0; track=2 (ja, rank 1) ships.
    rows = ([PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
             "TRACK track=2 kind=audio lang=ja offset=measured"] + REGIONS_TRACK1 + REGIONS_TRACK2
            + [drop_line("en", 0)])
    geometry = mpr.plan_geometry(records(rows))
    assert not geometry.get("undelivered") and geometry["lead"]["track"] == "2", geometry["lead"]
    assert len(geometry["cuts"]) == 0, geometry["cuts"]


def test_id_718_lead_language_entirely_dropped_the_delivered_sibling_leads():
    '''id 718 (Zetman E01), its own captured values: track=1 `en` (rank 0), track=2 `ja` (rank 1),
    span_ms=1421420.0 (forensic/cases/718/LAST_RUN.json). `ja` -- the master's own lead language --
    is entirely dropped by intact_same_language_wins (`fabricated_dropped ... lang=ja stream=1`,
    the RANK of the SECOND rebuilt track); `en` is kept (`fabricated_kept ... lang=en stream=0`,
    rank 0). adb47eeb matched `stream=1` against `track=1` and excluded `en` instead, leaving `ja`
    (the dropped track) leading -- this must fail on adb47eeb and pass here.'''
    rows = ([PLAN.replace("1452410", "1421420"),
             "TRACK track=1 kind=audio lang=en offset=measured",
             "TRACK track=2 kind=audio lang=ja offset=measured"]
            + REGIONS_TRACK1 + REGIONS_TRACK2
            + ["UNPARSED line=[fabricated_kept cause=no_intact_master_track lang=en holder=audios "
               "stream=0 format=FLAC marker=chimeric reason=the master carries no intact en track "
               "to race it against]",
               drop_line("ja", 1, format_="FLAC")])
    geometry = mpr.plan_geometry(records(rows))
    assert not geometry.get("undelivered"), geometry
    assert geometry["lead"]["track"] == "1" and geometry["lead"]["lang"] == "en", geometry["lead"]


def test_no_drop_keeps_the_first_measured_track():
    rows = [PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
            "TRACK track=2 kind=audio lang=ja offset=measured"] + REGIONS_TRACK1 + REGIONS_TRACK2
    geometry = mpr.plan_geometry(records(rows))
    assert geometry["lead"]["track"] == "1" and len(geometry["cuts"]) == 1, geometry["cuts"]


def test_one_drop_of_two_same_language_tracks_names_the_one_track():
    '''Two `en` tracks, track=1 and track=3 (ranks 0 and 1 -- ranking is by `track=` ascending, gaps
    do not matter). The drop line's `stream=0` names RANK 0, track=1, exactly; track=3 is never
    named, so it stays delivered and leads. Matching `stream=` against `track=` directly (adb47eeb)
    would instead have excluded nothing here (no TRACK carries `track=0`), a second, milder way the
    same mistake misreads this line.'''
    rows = ([PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
             "TRACK track=3 kind=audio lang=en offset=measured"] + REGIONS_TRACK1
            + regions_for(REGIONS_TRACK1, 3) + [drop_line("en", 0)])
    geometry = mpr.plan_geometry(records(rows))
    assert not geometry.get("undelivered"), geometry
    assert geometry["lead"]["track"] == "3", geometry["lead"]


if __name__ == "__main__":
    count = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            count += 1
            print("ok", name)
    print(f"ALL PASS ({count})")
