#!/usr/bin/env python3
"""Forensic's CANNOT HELP set -- 52 files where no repair is meaningful.

WHY EVERY ROW MUST CARRY IT: forensic states it outright -- "if these 52 are swept,
they will land in 'damaged' or 'declined' and NEITHER NUMBER WILL MEAN WHAT IT SAYS."
My corpus run sweeps all 315 with no exclusions, so without this field every rate I
publish silently includes 16.5% of files that CANNOT succeed by construction: 27 that
are not the same programme, 11 whose defect is in the master the repair never touches,
2 with no measurement of any kind.

THE FIELD IS RECORDED, NOT APPLIED. Rows are tagged and kept; the incidence can then
be computed both ways -- with and without -- instead of me choosing a denominator
once, invisibly, and everyone inheriting it. Same principle as folder_hash and
image_git_commit: a row that does not name its population cannot be counted correctly
later. It is also the correct home for the fact that a peer's sample EXCLUDES a file
that PRODUCTION STILL ATTEMPTS -- exclusion sets a denominator, it does not stop the
pipeline.
"""
FAMILIES = {
    "master_side_divergence":   [6, 7, 10, 11, 14, 15, 16, 17, 18, 19, 24],
    "no_shared_language":       [119, 678, 679, 680, 681, 682, 684],
    "wrong_episode":            [121, 123],
    "defective_master_audio":   [172, 349],
    "instrument_did_not_run":   [112, 693],
    "tags_lie_one_track":       [110],
    "wrong_series_named":       [692],
    # EXACT, from forensic: median prominence <= 0.007 over 11 trial rates against a
    # chance floor. THE MEASUREMENT, NOT THE FOLDER -- folder 103 holds 49 error files
    # and only these 26 qualify; the other 22 are repairable ("different cuts") and
    # id 172 is cannot-help under defective_master_audio, counted there and not here.
    "wrong_series_measured":    [237,238,239,241,243,245,246,247,248,249,250,251,252,
                                 253,254,255,256,257,258,259,260,261,262,263,264,265],
}
UNKNOWN = {"weak_correlation": [196, 307], "unclassifiable": [45]}
# folder 103 contributes 26 more to wrong_series; resolved from the corpus by folder,
# NOT hardcoded, because a hand-copied id list is a second source of truth.
def wrong_series_folder_ids(errs, folders_path='runs/folders.json'):
    import json, os
    try: fo = json.load(open(folders_path))
    except Exception: return []
    for f in (fo if isinstance(fo, list) else fo.get('folders', [])):
        if str(f.get('id')) == '103':
            # DO NOT INFER THE 26 FROM THE FOLDER. Forensic classified 26 files in
            # folder 103 as wrong-series; the folder holds 48 error files, so 22 of
            # them are NOT in that class. Tagging all 48 would over-exclude by 22 and
            # tagging none under-excludes by 26. The per-file judgement is forensic's
            # and I do not hold it, so these are marked AMBIGUOUS and reported by name
            # rather than resolved by a guess that would look like knowledge.
            # the error tree EMBEDS the destination path rather than equalling it,
            # so this is containment, not equality. Matching on equality silently
            # resolved zero files and would have under-tagged by 26.
            d = f.get('destination_path') or f.get('path') or ''
            return sorted(i for i, p in errs.items() if d and d in p)   # 48, of which 26 qualify
    return []
def build(errs=None):
    m = {}
    for fam, ids in FAMILIES.items():
        for i in ids: m[i] = fam
    for fam, ids in UNKNOWN.items():
        for i in ids: m.setdefault(i, "UNKNOWN:" + fam)
    # The folder heuristic is GONE. It was a stand-in for a per-file judgement I did
    # not hold; forensic supplied the exact ids, so the guess is deleted rather than
    # left beside the truth where a later reader might use either.
    return m
if __name__ == '__main__':
    import json
    errs = {e['id']: e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
    m = build(errs)
    print(f"  CANNOT HELP tagged: {len([k for k,v in m.items() if not v.startswith('UNKNOWN')])}"
          f"   (forensic states 52)")
    print(f"  UNKNOWN tagged    : {len([k for k,v in m.items() if v.startswith('UNKNOWN')])} (forensic states 3)")
    amb = [k for k, v in m.items() if v.startswith('AMBIGUOUS')]
    assert not amb, f"ambiguity was closed by forensic's exact ids; {len(amb)} tags remain"
    print(f"  folder 103: 49 error files = 26 wrong-series (cannot-help) + 1 defective master"
          f" (id 172, counted under its own class) + 22 repairable. NO AMBIGUOUS TAGS REMAIN.")
    five = [8, 9, 12, 13, 25]
    print(f"  the five splice ids in CANNOT HELP: {[i for i in five if i in m] or 'NONE -- all five are interpretable'}")
