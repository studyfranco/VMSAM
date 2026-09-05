#!/usr/bin/env python3
"""Audio tracks filled from a DIFFERENT language than they declare.

`vmsam-forensic` found it: two DELIVERED artefacts carry a third of their non-Japanese
audio as Japanese. A Spanish track plays Spanish for two thirds of the programme and
Japanese for the other third, in blocks.

    A TRACK FILLED WITH THE WRONG LANGUAGE HAS A PERFECTLY CORRECT DURATION.

So `FULL_LENGTH` is right, coverage is right, `deficit_vs_master_ms` is right, and NOT
ONE OF THEM CAN SEE IT. It is the third member of the family:

    a uniformly shifted subtitle track    correct duration   -- currently EMPTY
    an unnecessary interior fill          correct duration   -- OCCUPIED
    a track filled with the WRONG LANGUAGE correct duration  -- OCCUPIED, DELIVERED

TWO SOURCES, AND THEY DISAGREE BY AGE:

  `cross_language_fill=true` on the track line. dev-2 emits it -- but only in images
  from tonight. The two DELIVERED artefacts predate the field and declare 0/0.

  `repair: ADDED audio track N ... from=master/<lang>` against the track's own
  `lang=`. Present on EVERY record, old and new. This is the one that reaches back.

So: measure from ADDED, and report the declaration separately when it exists. A field
that only exists going forward cannot audit what already shipped.

WHY forensic DID NOT CATCH IT EARLIER, IN ITS OWN WORDS: it had written
"verified=2/7 is not under-coverage -- the 5 skipped are FOREIGN-LANGUAGE FILLS the
check cannot run on." THE PHRASE DID ITS JOB AS AN EXCUSE AND WAS NEVER READ AS A
DESCRIPTION. An explanation for a coverage gap, never asked what it described.
"""
import re, glob, sys, collections, json

def scan(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    plan = re.search(r'repair: plan [^\n]*?pieces=([^\n]*)', txt)
    segs = []
    if plan:
        for t in plan.group(1).split():
            m = re.match(r'([mc])(\d+)-(\d+)$', t)
            if m: segs.append((m.group(1), int(m.group(2)), int(m.group(3))))
    span = (segs[-1][2] - segs[0][1]) / 1000.0 if segs else None
    langs = {int(n): l for n, l in re.findall(r'repair: audio track (\d+) lang=(\w+)', txt)}
    declared = dict((int(n), v) for n, _l, v in
                    re.findall(r'repair: audio track (\d+) lang=(\w+)[^\n]*?cross_language_fill=(\w+)', txt))
    per = collections.defaultdict(float)
    src_of = collections.defaultdict(set)
    # THE FORMAT GREW A FIELD BETWEEN THE RANGE AND `from=`, AND MY REGEX REQUIRED
    # THEM ADJACENT. Tonight's image emits
    #     ADDED audio track 1 master 120000.0-220000.0 why=unreported(...) from=master/ja
    # so the pattern matched NOTHING and this detector returned 0 on every record from
    # the newest image -- a FALSE NEGATIVE ON EXACTLY THE FRESHEST DATA, which is the
    # worst direction.
    #
    # `vmsam-forensic` caught it by noticing my detector read 0 on a record whose OWN
    # DECLARATION said 5/5. A detector disagreeing with the declaration in the same file
    # is the shape both of us have been bitten by all night; here it was mine.
    #
    # `[^\n]*?` rather than a fixed gap: the format has grown once and will again.
    for n, a, b, src in re.findall(
            r'repair: ADDED audio track (\d+) master ([\d.]+)-([\d.]+)[^\n]*? from=master/(\w+)', txt):
        n = int(n)
        own = langs.get(n)
        if own and src[:2] != own[:2]:
            per[n] += (float(b) - float(a)) / 1000.0
            src_of[n].add(src)
    # PARSE COVERAGE. THREE TIMES TONIGHT I WROTE A PATTERN AGAINST THE FORMAT THAT
    # EXISTED WHEN I WROTE IT, and each time it returned a clean zero rather than an
    # error: `pieces=(\S+)` took the first of a space-separated list; a `quantum=`
    # pattern selected a subset and made an invariance look established; and this one
    # required `from=` adjacent to the range until the format grew `why=` between them.
    #
    # THE ANCHOR IS PRESENT AND THE STRUCTURED PATTERN MATCHED NOTHING = A PARSE
    # FAILURE, NOT AN ABSENCE. That distinction is the whole of `absent is never zero`
    # pointed at my own reader instead of at the data.
    anchor = len(re.findall(r'repair: ADDED audio track ', txt))
    matched = len(re.findall(
        r'repair: ADDED audio track (\d+) master ([\d.]+)-([\d.]+)[^\n]*? from=master/(\w+)', txt))
    parse_gap = anchor - matched
    # WAS THE CONDITION EVEN PRESENT? `vmsam-forensic`'s point, and it turns a clean
    # result into an untested one: F-54 arises only when the candidate supplies a
    # language THE MASTER CANNOT FILL FROM. Measured over this run's twelve jobs, that
    # was true of ZERO -- eight are single-language and the rest are covered.
    #
    #     NOT  "0 artefacts carry a foreign-language fill"
    #     BUT  "0 of N carry it; the condition was present in M of N jobs"
    #
    # With M = 0 the first line is a green light with nothing behind it. A CLEAN RESULT
    # WHOSE DENOMINATOR IS ZERO OPPORTUNITIES SHOULD READ *UNTESTED*, NOT *PASSED*.
    #
    # PROXY, AND THE LIMIT RUNS ONE WAY ONLY -- IT CANNOT MOVE A ZERO.
    # `from=master/<lang>` shows what the master DID fill from, not everything it could,
    # so `avail` UNDERSTATES the master. An understated `avail` makes `demand - avail`
    # LARGER, so this test reports risk where none exists: IT OVER-DETECTS.
    #
    #     A TEST BIASED TOWARD FALSE POSITIVES, RETURNING NONE, BOUNDS THE TRUE
    #     COUNT AT ZERO.
    #
    # MEASURED rather than argued: against the masters' actual audio languages via
    # ffprobe, the proxy understates in 11 of 12 jobs -- id 65's master carries SEVEN
    # audio languages where the proxy saw one. The bias is large and the test still
    # found nothing.
    #
    # `vmsam-forensic` supplied the direction and it corrects how I first wrote this up.
    # A LIMIT THAT WEAKENS A CONCLUSION AND A LIMIT THAT CANNOT ARE DIFFERENT THINGS,
    # and reporting both as caveats trains readers to discount the ones that matter.
    #
    # The stronger reading of the same fact: `demand - avail` empty on all twelve means
    # EVERY TRACK LANGUAGE APPEARED AS A FILL SOURCE IN ITS OWN JOB -- not "no risk
    # detected" but "every language was demonstrably fillable from the master".
    # EXPOSURE IS A COUNT OVER TRACKS, NOT JOBS. `vmsam-forensic`: the condition can
    # only arise on a REBUILT track -- a passthrough has no fill at all, so a candidate
    # may carry ten languages the master has never heard of and nothing happens.
    # Both F-54 artefacts rebuilt EVERY track they had, seven and two.
    #
    #     NOT  "the condition arose in M of N jobs"
    #     BUT  "N tracks rebuilt, of which K in a language the master lacks"
    #
    # `repair: audio track N lang=` is the repair's own per-track report, so its keys
    # are exactly the tracks the repair touched.
    rebuilt = sorted(langs)
    rebuilt_langs = [langs[n][:2] for n in rebuilt if langs.get(n)]
    demand = {l[:2] for l in langs.values() if l}
    supply = {s[:2] for s in re.findall(r'from=master/(\w+)', txt)}
    uncoverable = sorted(demand - supply) if supply else []
    return {"record": path.split('/')[-1],
            "parse_anchor_lines": anchor, "parse_matched": matched,
            "PARSE_FAILURE": parse_gap > 0,
            "refused": 'undelivered state=REFUSED' in txt,
            "programme_s": span,
            "track_langs": sorted(demand), "master_fills_from": sorted(supply),
            "tracks_rebuilt": len(rebuilt),
            "tracks_rebuilt_in_a_lang_the_master_lacks":
                sum(1 for l in rebuilt_langs if supply and l not in supply),
            "CONDITION_PRESENT": bool(uncoverable), "uncoverable_langs": uncoverable,
            "tracks": [{"track": n, "lang": langs.get(n), "filled_from": sorted(src_of[n]),
                        "foreign_s": round(per[n], 1),
                        "pct": round(100 * per[n] / span, 1) if span else None,
                        "declared_by_pipeline": declared.get(n)}
                       for n in sorted(per)]}

if __name__ == '__main__':
    # ONE ENTRY PER JOB. A record count made `tracks rebuilt` 131 where the job count
    # is lower -- my own redacted copies counted twice, for the fourth time tonight.
    import census_population
    files = [p for p, _k, _h, _d in census_population.records()]
    # COUNT JOBS, NOT RECORDS. The same job appears as a KEEP/*.error AND a
    # runs/decline-*.redacted, so a record count reported REFUSED 4 where there were
    # two jobs. `vmsam-forensic` caught it from a different population.
    #
    # AND THE FIRST DEDUPLICATION WAS WORSE THAN THE DOUBLE COUNT: it keyed on
    # `candidate_digest`, the two DELIVERED records predate that field, and `continue`
    # SKIPPED THEM -- reporting DELIVERED 0. A DEDUPLICATOR THAT DROPS WHAT IT CANNOT
    # KEY IS A FILTER PRETENDING TO BE A COUNTER.
    #
    # So: fall back to the plan string, and NEVER skip. A record with no key at all
    # keys on its own name -- that may OVER-count, which is the survivable direction.
    import hashlib, os
    hits_all = [scan(f) for f in files]
    hits = [r for r in hits_all if r['tracks']]
    paths = {os.path.basename(f): f for f in files}
    jobs = {}
    for r in hits:
        txt = open(paths[r['record']], encoding='utf-8', errors='replace').read()
        cd = re.search(r'repair: candidate_digest (\w+)', txt)
        pl = re.search(r'repair: plan [^\n]*?pieces=([^\n]*)', txt)
        if cd:   key, how = cd.group(1)[:12], 'candidate_digest'
        elif pl: key, how = hashlib.sha256(pl.group(1).encode()).hexdigest()[:12], 'plan-string'
        else:    key, how = 'unkeyed:' + r['record'], 'NO KEY -- counted separately'
        j = jobs.setdefault(key, dict(r, records=[], key_source=how))
        j['records'].append(r['record'])
    hits = list(jobs.values())
    dl = [r for r in hits if not r['refused']]
    rf = [r for r in hits if r['refused']]
    print(f"  {len(files)} records -> {len(hits)} DISTINCT JOBS "
          f"(keys: {', '.join(sorted({h['key_source'] for h in hits}))})\n")
    for r in hits:
        print(f"  {r['record'][:26]:28} {'REFUSED' if r['refused'] else 'DELIVERED':10} "
              f"programme {r['programme_s']} s")
        for t in r['tracks']:
            d = t['declared_by_pipeline']
            dtxt = f"declared={d}" if d else "NOT DECLARED (image predates the field)"
            print(f"      track {t['track']} lang={t['lang']:>3}  <- {'/'.join(t['filled_from'])}"
                  f"  {t['foreign_s']:>8} s = {t['pct']:>5} %   {dtxt}")
    broken = [r for r in (scan(f) for f in files) if r.get('PARSE_FAILURE')]
    if broken:
        print(f"\n  PARSE FAILURES -- the anchor is present and my pattern matched less:")
        for b in broken[:6]:
            print(f"    {b['record'][:28]:30} anchor {b['parse_anchor_lines']} lines, "
                  f"matched {b['parse_matched']}")
        print("    A CLEAN ZERO FROM A READER THAT COULD NOT READ IS NOT A MEASUREMENT.")
    else:
        print(f"\n  parse coverage: every ADDED line my anchor sees, my pattern parses")
    opp = [r for r in hits_all if r.get('CONDITION_PRESENT')]
    tr  = sum(r.get('tracks_rebuilt', 0) for r in hits_all)
    trk = sum(r.get('tracks_rebuilt_in_a_lang_the_master_lacks', 0) for r in hits_all)
    print(f"\n  EXPOSURE, over TRACKS rather than jobs:")
    print(f"    tracks rebuilt across all records          {tr}")
    print(f"    of those, in a language the master lacks   {trk}")
    print(f"  jobs where the condition could arise: {len(opp)} of {len(hits_all)}")
    if not opp:
        print("    -> ZERO OPPORTUNITIES. A clean result here is UNTESTED, not PASSED.")
    print(f"\n  DELIVERED carrying a foreign-language fill: {len(dl)}")
    print(f"  REFUSED   carrying one:                     {len(rf)}")
    if dl:
        worst = max((t['pct'] or 0) for r in dl for t in r['tracks'])
        print(f"  worst delivered fraction: {worst} % of a programme in the wrong language")
        # WHEN, NOT JUST WHETHER. Both delivered hits were captured at 15:14:53Z --
        # more than five hours before the current run began, and by an earlier image.
        # "2 delivered artefacts are wrong-language" and "this run is shipping
        # wrong-language artefacts" are different claims and the second is FALSE.
        #
        # A standing PROBLEMS line that cannot distinguish historical damage from
        # ongoing damage will be read as ongoing, because that is what a live check
        # normally means.
        import csv as _csv, os as _os
        cap = {}
        _led = '/config/output/KEEP/ci-preservation-ledger.tsv'
        if _os.path.exists(_led):
            for row in _csv.DictReader(open(_led), delimiter='\t'):
                cap[row.get('keep_name')] = row.get('captured_utc')
        print("  when each was captured -- historical damage is not ongoing damage:")
        for r in dl:
            for name in r['records']:
                print(f"    {name[:30]:32} captured {cap.get(name, 'unknown')}")
    json.dump(hits, open('runs/cross-language-fill.json', 'w'), indent=1)
    print("\n  written runs/cross-language-fill.json")
    sys.exit(1 if dl else 0)
