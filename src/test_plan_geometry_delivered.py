'''Tests for merge_plan_report.plan_geometry's lead track -- run: python3 src/test_plan_geometry_delivered.py
(or pytest). No media: the rows are the report's own, shaped as id 54's product (its only rebuilt eng track
dropped by the delivery gate, intact_same_language_wins; the delivered video, eng and jpn md5-identical to the
master), as id 718's product (lead language `ja` entirely dropped, `en` delivered), and as a pair where one
language's track is dropped and another is delivered.

The delivery gate names the track it drops by its `stream=` (StreamOrder), the SAME identifier the report's
TRACK row carries as `track=` (merge_video_chimeric.py:1342 vs merge_video_repair.py:926/1636): plan_geometry
matches a drop to the ONE track it names, never to a same-language sibling the line never mentioned.'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import merge_plan_report as mpr  # noqa: E402

PLAN = "PLAN master_end_ms=1452410"
REGIONS_EN = ["REGION track=1 kind=CANDIDATE master_start_ms=0 master_end_ms=500000 offset_ms=-1000",
              "REGION track=1 kind=MASTER master_start_ms=500000 master_end_ms=501001 source=[master/en]",
              "REGION track=1 kind=CANDIDATE master_start_ms=501001 master_end_ms=1452410 offset_ms=-2001"]
REGIONS_JA = ["REGION track=2 kind=CANDIDATE master_start_ms=0 master_end_ms=1452410 offset_ms=-1000"]
DROP_EN = ("UNPARSED line=[fabricated_dropped cause=intact_same_language_wins lang=en holder=audios stream=1 "
           "format=E-AC-3 marker=chimeric kept_master]")
DROP_JA = ("UNPARSED line=[fabricated_dropped cause=intact_same_language_wins lang=ja holder=audios stream=1 "
           "format=DTS marker=chimeric kept_master]")


def records(rows):
    return mpr.parse_rows(rows)


def test_id_54_no_delivered_track_is_no_geometry_and_says_so():
    rows = [PLAN, "TRACK track=1 kind=audio lang=en fill=[master/en] offset=measured"] + REGIONS_EN + [DROP_EN]
    geometry = mpr.plan_geometry(records(rows))
    assert geometry["undelivered"] and geometry["dropped"] == ["en"], geometry
    assert "Aucune piste livrée" in mpr.render_plan_schematic(geometry)
    job = {"audios": {1: {"lang": "en"}}, "summary_counts": "", "subtitles": [], "delivery": []}
    assert "aucune piste livrée" in mpr.render_human_summary(job, geometry)


def test_the_delivered_track_leads_when_the_first_is_dropped():
    rows = ([PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
             "TRACK track=2 kind=audio lang=ja offset=measured"] + REGIONS_EN + REGIONS_JA + [DROP_EN])
    geometry = mpr.plan_geometry(records(rows))
    assert not geometry.get("undelivered") and geometry["lead"]["track"] == "2", geometry["lead"]
    assert len(geometry["cuts"]) == 0, geometry["cuts"]


def test_id_718_lead_language_entirely_dropped_the_delivered_sibling_leads():
    '''id 718 (Zetman E01): `ja` is the lead language (track=1, master's own lead), every one of its
    rebuilt tracks dropped by intact_same_language_wins; `en` (track=2) ships. The lead must move to
    `en` -- the plan-mismatch watch found the lead staying on the dropped `ja` track instead.'''
    ja_track1 = [r.replace("track=2", "track=1") for r in REGIONS_JA]
    en_track2 = [r.replace("track=1", "track=2") for r in REGIONS_EN]
    rows = ([PLAN, "TRACK track=1 kind=audio lang=ja offset=measured",
             "TRACK track=2 kind=audio lang=en offset=measured"] + ja_track1 + en_track2 + [DROP_JA])
    geometry = mpr.plan_geometry(records(rows))
    assert not geometry.get("undelivered"), geometry
    assert geometry["lead"]["track"] == "2" and geometry["lead"]["lang"] == "en", geometry["lead"]


def test_no_drop_keeps_the_first_measured_track():
    rows = [PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
            "TRACK track=2 kind=audio lang=ja offset=measured"] + REGIONS_EN + REGIONS_JA
    geometry = mpr.plan_geometry(records(rows))
    assert geometry["lead"]["track"] == "1" and len(geometry["cuts"]) == 1, geometry["cuts"]


def test_one_drop_of_two_same_language_tracks_names_the_one_track():
    '''Two `en` tracks (1 and 3); the drop line's `stream=1` names track 1 exactly -- the SAME
    identifier as TRACK's `track=` (delivery_drops). Track 3 is never named, so it stays delivered
    and leads: the count-only rule used to treat this as ambiguous and leave the dropped track 1
    leading (the 718 bug's shape, one language, two tracks).'''
    rows = ([PLAN, "TRACK track=1 kind=audio lang=en offset=measured",
             "TRACK track=3 kind=audio lang=en offset=measured"] + REGIONS_EN
            + [r.replace("track=1", "track=3") for r in REGIONS_EN] + [DROP_EN])
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
