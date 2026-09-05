#!/usr/bin/env python3
"""Full-corpus merge run. The corpus IS production; the merges are the TEST.

Owner's instruction: validate the complete code on every error file. Not a sample.
Outputs are DISPOSABLE -- merge, verify, delete, next -- so disk is not a constraint.

ACCEPTANCE IS THE ARTEFACT, NOT THE EXIT STATUS (SPEC_ZONE_A §4d). A repair reported
"7 audio and 24 subtitle track(s) rebuilt, 0 declined, 0 failed" and shipped a file
with nothing after 21:21. Every counter said success. A count of rebuilt tracks is a
statement about work done, not about a file.

WAITING: poll for the ARTEFACT and the queue's status. NEVER /health is_running --
it reported JOB_DONE at t+271 s on a job that ran another 105 minutes.

REPORT WHAT FAILS, NOT WHAT PASSES. Every row is written either way; the summary
counts failures.
"""
import json, os, subprocess, sys, time

# WHICH CONTAINER, NOT JUST WHICH IMAGE. The owner is giving ci a SECOND container --
# `showgestionar-test2` -- from the next container update. Two containers on ONE image
# can diverge in queue state, output tree and elapsed time, so `image_git_commit` stops
# being a complete reference the moment the second exists.
#
# Parameterised BEFORE it exists rather than after, because the failure mode is silent:
# a probe aimed at the wrong container returns a VALID ANSWER ABOUT THE WRONG APPARATUS.
# Nothing errors. That is the image-label defect one substrate out, and I have already
# had to amend that correction twice tonight.
CONTAINER = os.environ.get("VMSAM_TEST_HOST", "showgestionar-test")
API = f"http://{CONTAINER}:8080"
OUT_ROOT = "/config/output/srv"
POLL, TIMEOUT = 30, 4 * 3600


def api(path, data=None):
    cmd = ["curl", "-s", "--max-time", "60", f"{API}{path}"]
    if data is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "-d", json.dumps(data)]
    try:
        return json.loads(subprocess.run(cmd, capture_output=True, timeout=90).stdout.decode())
    except Exception:
        return None


def image_now():
    """THE IMAGE AT THE MOMENT THE ROW IS WRITTEN, not at the moment the run started.

    `IMAGE` is read once in main(). That was enough for the failure it was built for
    -- a RESUME across a redeploy -- and not enough for a REDEPLOY DURING A RUN, which
    is what happened on 2026-09-04: this runner started at 22:49 on 278471a0, I
    recreated the container onto 0ea160d8 at 21:18Z, and id 44 ran on the NEW locator
    while its row named the OLD image. The comment beside the header field already
    said "a row that does not name the code that produced it is not a measurement of
    anything in particular" -- and the per-row value stayed cached anyway.

    Returns None on failure, and the caller records THAT rather than substituting a
    guess: an unreadable image and an unchanged image are different facts.
    """
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "20", f"{API}/health"],
                             capture_output=True, timeout=40).stdout.decode()
        return json.loads(out).get("git_commit")
    except Exception:
        return None


PREEXISTING = set()      # artefacts on disk before this run POSTed anything


def snapshot_preexisting():
    """Every artefact already on disk, recorded BEFORE the first POST.

    The matcher is time-based: "files newer than the moment I asked". That is sound
    with one client and a clean directory, and unsound the instant anything else has
    produced output. An ORPHANED job -- one POSTed by a client that died before it
    could collect -- writes its artefact whenever it finishes, and if that lands after
    my `since` I would claim another file's output as my own. That is the
    misattribution I retracted a CPU claim over, arriving through a different door.
    Excluding a recorded set is exact; a timestamp is a guess about ordering.
    """
    r = subprocess.run(["find", OUT_ROOT, "(", "-name", "*.mkv", "-o",
                        "-name", "*.log.error", "-o", "-name", "*.log", ")"],
                       capture_output=True, timeout=120)
    return {l for l in r.stdout.decode().splitlines()
            if l.strip() and not _is_refused(l)}


def _is_refused(path):
    """A GATE-DECLINED ARTEFACT IS PRESERVED BUT IS NOT A PRODUCED FILE.

    dev-2 suffixes a file the output gate refuses as `<name>.REFUSED.<ext>`, and the
    file STAYS ON DISK -- it is the only evidence of what the gate stopped. My preserver
    keeps it, in its own ledger class `declined_by_gate`, ranked with REFUTED so it is
    evicted last.

    But it must never be COUNTED as produced. With enforcing on, six of sixteen stop;
    if my discovery treats them as artefacts, every produced-file count silently
    absorbs the declines and the flip looks like it changed nothing.

    I first told dev-2 I would solve this by EXCLUDING them from the preserver. That
    was solving a counting problem by destroying the evidence class. Preserve, and
    count separately -- the two are different questions and only one of them is mine.
    """
    # TWO NON-PRODUCED CLASSES, AND THEY ASSERT DIFFERENT THINGS.
    #   .REFUSED.    the gate decided against it
    #   .NOVERDICT.  a tool fault escaped and NOBODY decided -- the file is on disk
    #                claimed by no report, and without a token my scans count it as
    #                PRODUCED. That is the untokened third state, one level out: not a
    #                missing field but a missing FILENAME token, and the default is the
    #                flattering one.
    return ".REFUSED." in path or ".NOVERDICT." in path


def artefacts(since):
    r = subprocess.run(["find", OUT_ROOT, "-newermt", since,
                        "(", "-name", "*.mkv", "-o", "-name", "*.log.error",
                         "-o", "-name", "*.log", ")"],   # *.log CARRIES THE GATE LINE
                       capture_output=True, timeout=120)
    return [l for l in r.stdout.decode().splitlines()
            if not _is_refused(l)
            if l.strip() and l not in PREEXISTING]


import hashlib, re as _re
from redact import redact_checked, BARE
from cannot_help import build as _ch_build


def _instrument():
    """THE PARAMETERS THAT SCOPE EVERY VERDICT IN THIS ROW.

    dev-1's rule, adopted: emit the instrument's own parameters as first-class keys
    rather than relying on anyone remembering the caveat. A reader cannot then quote a
    result at a precision the instrument does not have without contradicting a field in
    the same object.

    THE REASON IT IS A FIELD AND NOT A RULE: dev-1's `0.067 ms` was not caught by care
    or review. It was a plausible small figure that agreed with expectation, and it
    survived four hours and five recipients -- and I imported it into the very docstring
    that exists to stop my own number travelling bare, unchecked, because it was tighter
    than mine and therefore flattering to defer to. What caught it was an artefact whose
    existence made the omission visible. The dangerous errors are the ones that look
    reasonable; the only ones we can name are the ones that stopped looking reasonable.

    NOT `_`-prefixed, so a consumer resolving keys by name cannot skip them.
    """
    return {
        # check_output.TOL -- the shortfall tolerated before a track is called short.
        # THIS SCOPES EVERY `FULL_LENGTH` AND EVERY `short_tracks: []` IN THIS ROW.
        # 2 % of a 1480 s programme is 29.6 SECONDS: "FULL_LENGTH" means "not more than
        # 2 % short", NOT "not short". forensic's frozen prediction was scored against
        # this field, and the scoring is only as strong as the tolerance behind it.
        "short_track_tolerance_frac": 0.02,
        "short_track_tolerance_note": (
            "a track is 'short' only below (1 - 0.02) x master duration. On a 1480 s "
            "file that is 29.6 s of unflagged shortfall. Subtitles are EXCLUDED by "
            "design -- a subtitle ends at its last cue, and a known-good file reported "
            "24 short subtitle tracks with every audio track at ratio 1.0."),
        # Durations: exact from the Matroska DURATION tag, biased otherwise.
        "duration_basis": "Matroska DURATION tag where present; else ffprobe, biased",
        "duration_bias_note": (
            "a duration read any way other than the DURATION tag carries 42-85 ms of "
            "bias that VARIES BETWEEN TRACKS OF ONE FILE, so it does not subtract out. "
            "n_duration_from_tag / n_duration_missing say which applies."),
        # Coverage is now DERIVED, and the derivation has a positive control.
        "coverage_basis": "derived from per-track verify= lines; emitted verified=N/M kept beside it",
        "coverage_control": "derivation agreed with the emitted field on 7 of 7 logs carrying both",
        # The span/endpoint distinction that campaign 1 s156 contradicts.
        "duration_tag_semantics": "SPAN (last - first), not endpoint -- measured, contradicts campaign-1 s156",
    }


def _toolchain():
    """WHAT EXECUTED THE CODE, not which code it was.

    The Dockerfile installs ffmpeg and mkvtoolnix UNPINNED from Debian testing, after
    a dist-upgrade -- only pydantic is pinned. SO TWO IMAGES BUILT FROM THE SAME
    COMMIT ON DIFFERENT DAYS CAN CARRY DIFFERENT ffmpeg AND DIFFERENT mkvmerge.
    `image_git_commit` therefore identifies HALF a run: the code, never the tools.
    Recovered from a CONTAINER-PRODUCED artefact, never from this host -- the host
    and the container agreeing today is consistent with similar provisioning dates
    and says nothing about a later build.
    """
    import glob as _g
    ver = src_file = None
    for p in sorted(_g.glob('/config/output/KEEP/*.error') +
                    _g.glob('/config/output/KEEP/*.log'),
                    key=os.path.getmtime, reverse=True):
        try:
            txt = open(p, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        m = _re.search(r'ffmpeg version ([^\s,]+)', txt)
        if m:
            ver, src_file = m.group(1), os.path.basename(p)
            break
    return {"ffmpeg": ver,
            "ffmpeg_source": (("version banner in a container-produced artefact: "
                               + src_file) if src_file
                              else "NOT RECOVERABLE from any held artefact"),
            "mkvmerge": None,
            "mkvmerge_why_unknown": (
                "no artefact here was muxed by it: every produced .mkv carries "
                "ENCODER=Lavf* (libavformat). The REPAIR path muxes with ffmpeg; the "
                "FROZEN MERGE path (mergeVideo.py:1827) uses mkvmerge. A merge-path "
                "FULL_LENGTH would carry its writing_application; none exists yet. "
                "NOT taken from this host."),
            "unpinned_in_dockerfile": ["ffmpeg", "mkvtoolnix"],
            "consequence": ("a row reproduced on the same commit six weeks from now is "
                            "not necessarily reproduced on the same muxer")}


def _safe_text(s):
    """Any free text entering a row passes through here. A row is an artefact that
    travels; an exception message and an API echo are both untrusted for paths."""
    try:
        return redact_checked(s)
    except ValueError:
        return ("<REDACTED: carried a library path and could not be safely rendered; "
                "the fact is kept, the value is not>")
from out_of_contract import mark as _ooc_mark

ad = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
errs = json.load(open('runs/errors-7b83af4.json'))['incompatible_files']
ids = [e['id'] for e in errs]
# OPTIONAL ID RESTRICTION, so a series ask is the SAME runner and not a fork.
# A forked runner is a second implementation that drifts; every field this one
# earned tonight -- image_capabilities, folder_hash, cannot_help, the gate line,
# read-before-delete -- would have to be re-earned in the copy.
_only = os.environ.get('ONLY_IDS')
if _only:
    # HONOUR THE ORDER GIVEN, NOT THE CORPUS ORDER. Filtering the corpus list keeps
    # the API's order and DISCARDS the caller's -- which would have silently thrown
    # away the whole point of ordering this ask: the six-file control is first because
    # a queue's tail is what an interruption eats, and a filter would have put it back
    # wherever GET /errors happens to place it.
    _want = [int(x) for x in _only.split(',') if x.strip()]
    _have = set(ids)
    ids = [i for i in _want if i in _have]
paths = {e['id']: e['file_path'] for e in errs}
CANNOT_HELP = _ch_build(paths)      # RECORDED, NOT APPLIED -- see cannot_help.py


CLONE = '/home/vmsam/src/VMSAM_WIP/ci'


# MARKERS ARE SOURCE FRAGMENTS, NOT OUTPUT STRINGS. The first version searched the
# SOURCE for the string the LOG contains -- but `offset=BORROWED` is built by an
# f-string and never appears literally, so the probe reported "this image cannot
# emit it" about an image I had already verified emits it. A capability probe that
# false-negatives is worse than none: it would have stamped every row with a
# confident, wrong statement about what the code could tell us.
CAPABILITY_MARKERS = {
    "offset_provenance": ("src/merge_video_repair.py", "report.get('offset_measured')"),
    "fill_ambiguity":    ("src/merge_video_repair.py", "'(AMBIGUOUS among '"),
    "fill_narrowing":    ("src/merge_video_repair.py", "by measurement"),
    "offset_fidelity":   ("src/merge_video_repair.py", "'(fid '"),
    "weakest_correlation": ("src/merge_video_repair.py", "checked.get('weakest_correlation')"),
}


def image_emits(sha, marker, path='src/merge_video_repair.py'):
    """Can the DEPLOYED CODE emit this marker at all?

    A log-level absence is ambiguous: the marker may be missing because nothing was
    ambiguous, or because THIS VERSION DOES NOT WRITE IT. Those are not-measured and
    measured-zero, and one column cannot carry both -- the defect just caught in
    `ambiguous_fill`, which recorded 0 on an image where the marker did not exist.
    So the capability is read from the SOURCE OF THE DEPLOYED COMMIT, not inferred
    from the output. A row then says what its code COULD have told us, which is the
    only way a later reader can tell an empty column from an unasked question.
    """
    try:
        r = subprocess.run(['git', '-C', CLONE, 'show', f'{sha}:{path}'],
                           capture_output=True, timeout=60)
        return marker in r.stdout.decode('utf-8', 'replace')
    except Exception:
        return None                       # unknown is its own answer, never False


READ = set()          # every artefact path some reader has actually opened


def read_text(p):
    """The ONLY way an artefact's bytes enter this program. Records the path."""
    t = open(p, encoding='utf-8', errors='replace').read()
    READ.add(os.path.realpath(p))
    return t


def consume(paths):
    """Delete artefacts -- BUT REFUSE ANY FILE NO READER HAS OPENED.

    "Remember to read before deleting" is not a control; it is a comment, and I
    committed the same deletion defect twice tonight WITH the comment already in
    place. A control is a delete path that cannot silently discard evidence: a
    text artefact nobody opened is not deleted, it is reported and left on disk.
    .mkv is exempt -- it is the disposable output, read by check_output as a file
    rather than as text, and it is the thing this whole run is designed to throw away.
    """
    kept = []
    for f in paths:
        if f.endswith('.mkv') or os.path.realpath(f) in READ:
            try: os.remove(f)
            except Exception: pass
        else:
            kept.append(os.path.basename(f))
    return kept


def gate_line(logs):
    """The container's own verdict, from the success log. Without this the row says
    a file merged and nothing about whether the gate could MEASURE it -- and the
    §4d removal condition counts measured artefacts, not merges."""
    for p in logs:
        try: t = read_text(p)
        except Exception: continue
        if 'measured=' not in t: continue
        for pat, rep in BARE: t = _re.sub(pat, rep, t)
        t = redact_checked(t)
        ln = [l for l in t.splitlines() if 'measured=' in l][0]
        m = dict(_re.findall(r'(\w+)=([\w.()]+)', ln))
        # 'audio track' MATCHES THREE DIFFERENT LINE KINDS. My substring counted
        #   repair: audio track N lang=...        <- a TRACK
        #   repair: ADDED audio track N master... <- a PIECE OPERATION
        #   repair: CUT   audio track N candidate <- a PIECE OPERATION
        # and reported id 694 as 8 tracks with 1 verified. IT HAS ONE TRACK AND IT IS
        # VERIFIED. The seven "unverified tracks" I flagged to two agents were the cut
        # and fill operations ON that single track. A substring is not a line type --
        # the same defect as a substring not being a field test, one artefact along.
        au = [l for l in t.splitlines() if _re.search(r'repair: audio track \d+ lang=', l)]
        ops = [l for l in t.splitlines() if _re.search(r'repair: (ADDED|CUT) audio track', l)]
        cues = [(int(a), int(b)) for a, b in
                _re.findall(r'kept_cues=(\d+) dropped_cues=(\d+)', t)]
        # OFFSET PROVENANCE, new at bee9af4. Every track outside the measured
        # language is placed with ANOTHER language's offset -- 14 to 32 ms apart on
        # the files the author measured, UNDER the 100 ms tolerance, so it passes and
        # ships silently. Counting it per file turns "a tacit fallback nobody reads"
        # into a number.
        # AMBIGUOUS fill: two principal dubs under one language tag, correlating at
        # 0.63 -- the regime of two different LANGUAGES -- and 21.3 ms apart. Filling
        # from the wrong dub is silent and under tolerance. 0 of 290 tracks carry a
        # hyphenated tag, so no tag-based fix exists and counting is the only handle.
        # ZERO BECAUSE NONE, OR ZERO BECAUSE THE MARKER DOES NOT EXIST? On bee9af4
        # the fill marker is not emitted at all, so a plain 0 would read as "no
        # ambiguous fills found" when it means "not measured". That is the
        # could-not-measure-versus-measured-zero distinction, in a column, and it
        # is the same error as a verdict field that measured a file extension.
        fill_marker_present = ("(AMBIGUOUS among" in t) or ("(among" in t) or ("fill_choices" in t)
        ambiguous_fill = t.count("(AMBIGUOUS among") if fill_marker_present else None
        borrowed = t.count("offset=BORROWED")
        measured_off = t.count("offset=measured")
        # r_min = the WEAKEST correlation the verifier saw. This is what turns the
        # out-of-contract mark from a caution into something interrogable: if a file
        # whose streams reach no pair above 0.70 comes back REPAIRED, r_min says what
        # the verifier actually scored while doing it.
        rmin = _re.findall(r'r_min=([\d.]+)', t)
        fid = _re.findall(r'\(fid ([\d.]+)\)', t)
        return {"r_min": (min(float(x) for x in rmin) if rmin else None),
                "r_min_n": len(rmin),
                "ambiguous_fill": ambiguous_fill,
                "fill_marker_present": fill_marker_present,
                "offset_fidelity_min": (min(float(x) for x in fid) if fid else None),
                "offset_fidelity_n": len(fid),
                "offset_borrowed": borrowed, "offset_measured_tracks": measured_off,
                "offset_provenance_logged": bool(borrowed or measured_off),
                "measured": m.get('measured') == 'True',
                "would_refuse": m.get('would_refuse') == 'True',
                "enforcing": m.get('enforcing') == 'True',
                "tolerance_ms": m.get('tolerance_ms'),
                "verify_total": len(au),
                "verify_aligned": sum('verify=aligned' in l for l in au),
                "verify_skipped": sum('verify=skipped' in l for l in au),
                # 8 TRACKS, 1 ALIGNED, 0 SKIPPED -- and the 7 missing carry NO verify
                # verb at all. My verify_skipped=0 meant "no line said skipped", and a
                # reader takes it for "no track went unverified". THIRD TIME THIS EXACT
                # DEFECT HAS APPEARED IN MY OWN COLUMNS: could-not-measure read as
                # measured-zero. Naming the gap is the only fix that survives a reader.
                "verify_unaccounted": len(au) - sum(('verify=' in l) for l in au),
                "piece_operations": len(ops),
                # dev-2 EMITS THE COVERAGE ITSELF as verified=N/M. Prefer the emitter's
                # own number to my reconstruction of it wherever both exist.
                # ONE FACT, NOT N. `verified=N/M` is a FILE-LEVEL count repeated on
                # every track line, so findall returns it once per track and any reader
                # sorting on the column counts one fact seven times. A peer emitted
                # seven coverage findings for one file before catching it; my column
                # would have done the same thing one artefact along.
                "verified_field": sorted(set(_re.findall(r'verified=(\d+/\d+)', t))),
                # THE EMITTER'S FIELD IS MISSING EXACTLY WHERE IT MATTERS MOST.
                # dev-3 tabulated 11 logs: the four WITHOUT `verified=N/M` predate the
                # field, and THREE OF THOSE FOUR are 7-track files at 2 aligned / 5
                # skipped -- the 2/7 shape. So a gate reading only the emitted field
                # sees 1/1 and 2/2 on the small files and NOTHING on the ones that
                # would have said 71 % uncheckable. Its absence CORRELATES WITH THE
                # CASES THAT MATTER, which is the ffprobe negative again: the field is
                # missing from the reader's view, not from the world.
                #
                # So coverage is DERIVED from the per-track `verify=` lines, which are
                # present in all eleven, and the emitted field is kept beside it as an
                # independent reading. Two readings of one fact: a mismatch is a
                # finding, not a tie to be broken silently.
                "verified_derived": (lambda a, sk: None if (a + sk) == 0
                                     else f"{a}/{a + sk}")(
                    len(_re.findall(r'verify=aligned', t)),
                    len(_re.findall(r'verify=skipped', t))),
                # FOUR OUTCOMES, NOT TWO. Absent is not zero, and contradictory is not
                # absent -- the previous version collapsed both into None, so "the
                # pipeline never emitted it" and "the pipeline emitted two different
                # values" were the same token.
                "verified_coverage": (lambda v: v[0] if len(v) == 1
                                      else ("UNKNOWN_COVERAGE: field not emitted"
                                            if not v else
                                            f"CONTRADICTORY: {len(v)} distinct values"))(
                    sorted(set(_re.findall(r'verified=(\d+/\d+)', t)))),
                # A DISAGREEMENT BETWEEN THE TWO READINGS IS THE INTERESTING ROW.
                "coverage_readings_agree": (lambda em, a, sk: None
                                            if (not em or (a + sk) == 0)
                                            else em[0] == f"{a}/{a + sk}")(
                    sorted(set(_re.findall(r'verified=(\d+/\d+)', t))),
                    len(_re.findall(r'verify=aligned', t)),
                    len(_re.findall(r'verify=skipped', t))),
                # FOREIGN FILL vs VERIFICATION -- the pairing a peer resolved by hand.
                # A track filled from a language other than its own CANNOT be verified:
                # you cannot correlate Spanish against Japanese audio. So the skipped
                # tracks are exactly the tracks that received foreign-language content
                # -- the highest-risk object the pipeline makes -- and the verified ones
                # are those filled from their own language, the least likely to be wrong.
                # THE COVERAGE IS NOT AN EFFORT SHORTFALL. IT IS STRUCTURALLY IMPOSSIBLE
                # EXACTLY WHERE IT MATTERS MOST. Recorded per file so the claim is a
                # count across a hundred files rather than four resolved by hand.
                "fill_vs_lang": (lambda pairs: {
                    "n_tracks": len(pairs),
                    "own_language_fill": sum(1 for lg, fl, _ in pairs if fl == lg),
                    "foreign_fill": sum(1 for lg, fl, _ in pairs if fl and fl != lg),
                    "foreign_fill_langs": sorted({f"{lg}<-{fl}" for lg, fl, _ in pairs
                                                  if fl and fl != lg}),
                    "verified_and_own": sum(1 for lg, fl, v in pairs if v and fl == lg),
                    # same status: zero is structural, not observed. Non-zero would be
                    # a regression in the predicate, not an exception to a rule.
                    "verified_and_foreign": sum(1 for lg, fl, v in pairs
                                                if v and fl and fl != lg),
                    "unverified_and_foreign": sum(1 for lg, fl, v in pairs
                                                  if not v and fl and fl != lg),
                    # THIS IS AN IDENTITY IN THE CODE, NOT A CORRELATION IN THE DATA.
                    # find_fill_audio and verify_on_master_timeline ASK THE SAME
                    # QUESTION -- does the master carry this track's language? -- so
                    # "filled from its own language" and "a reference exists to verify
                    # against" are ONE PREDICATE. There can be no counter-example, and
                    # a hundred rows saying TRUE would be a hundred confirmations of
                    # something never in question.
                    # I built this field defending the rule that a field which can only
                    # report the expected value is not a measurement, and then wrote
                    # one. It is kept because it still has a job -- but a DIFFERENT job.
                    "skipped_iff_foreign_MEANS": (
                        "CODE IDENTITY INTACT, not an empirical result. TRUE is the only "
                        "possible value while one predicate decides both. A FALSE here "
                        "means the predicate has been SPLIT by a code change -- a claim "
                        "about the software, not about the corpus. Regression detector."),
                    "skipped_iff_foreign": (
                        all((fl != lg) for lg, fl, v in pairs if not v and fl)
                        and all((fl == lg) for lg, fl, v in pairs if v and fl))
                        if pairs else None})(
                    [(mm.group(1), (mm.group(2) or '').split('/')[-1] or None,
                      'verify=aligned' in l)
                     for l in au
                     for mm in [_re.search(r'lang=(\w+).*?fill=([\w/]+)', l)] if mm]),
                "verify_coverage": (f"{sum(('verify=' in l) for l in au)}/{len(au)}"
                                    if au else None),
                # dev-2's NEW ANNOTATION, first seen on this image. forensic computed
                # the third number BY HAND on id125 (907 short, 1988 lost, 1081
                # unexplained) and predicted the annotation would under-report by about
                # half. THE CONTAINER NOW EMITS ALL THREE PER TRACK, so the prediction
                # is checkable on every row instead of on one file.
                "fill_source_short_ms": [float(x) for x in
                    _re.findall(r'FILL SOURCE SHORT BY ([\d.-]+) ms', t)],
                "track_lost_ms": [float(x) for x in
                    _re.findall(r'TRACK LOST ([\d.-]+) ms', t)],
                "unexplained_ms": [float(x) for x in
                    _re.findall(r'UNEXPLAINED ([\d.-]+) ms', t)],
                "kept_cues_min": min((k for k, _ in cues), default=None),
                "n_sub_tracks_logged": len(cues)}
    return None


def decline_reason(logs, i):
    """READ THE REASON BEFORE DELETING IT. A crash and a principled refusal both write
    .log.error; without this the verdict field measures WHICH EXTENSION APPEARED."""
    for p in logs:
        try: t = read_text(p)
        except Exception: continue
        for pat, rep in BARE: t = _re.sub(pat, rep, t)
        t = redact_checked(t)
        # COMPUTE FIRST, OPEN SECOND. `open(p,'w')` truncates the moment it is
        # called, so `open(p,'w').write(f(x))` leaves a ZERO-BYTE FILE whenever f
        # raises -- and a zero-byte redacted log audits perfectly clean and cannot
        # fail any check. That is how I destroyed the id 214 log I had just
        # recovered: the guard refused to emit, and the refusal had already
        # emptied the file.
        _tmp = t
        with open(f'runs/decline-{i}.log.redacted', 'w') as _fh:
            _fh.write(_tmp)
        top = [l for l in t.splitlines() if 'Error processing file' in l]
        fr = _re.findall(r'line \d+, in (\w+)', t)
        # THREE CAUSES SHARE ONE EXTENSION. A peer read the three .error artefacts my
        # preserver retained and found: a CORRECT refusal (no_plan at fidelity 0.567,
        # barely above a 0.50 chance floor), a TOOLING FAILURE (ffmpeg rc 183, the
        # instrument did not run), and a PROBABLE FALSE REJECTION (a pair correlating
        # at 0.977 dropped because its delays span 2000 ms against a <127 ms
        # reconciliation window -- a progressive offset, which is what the resampler
        # exists for, reaching remove_not_compatible_video instead).
        # If `.error` ever becomes a bucket in a count it merges all three.
        #
        # AND THE MESSAGE NAMES THE CONSEQUENCE, NOT THE CAUSE: "Only N file left.
        # This is useless to merge files" tells a reader the merge was pointless and
        # never that a pair was rejected or why. The cause is recoverable -- deepest
        # frame plus the repair verb above it -- so it is EXTRACTED here rather than
        # left for whoever reads the row.
        _d = [float(x) for x in _re.findall(r'first_delay_test[^{]*\{([-\d, ]+)\}', t)
              for x in x.split(',') if x.strip().lstrip('-').isdigit()]
        _span = (max(_d) - min(_d)) if len(_d) > 1 else None
        if 'Invalid data found' in t or 'Error opening input' in t:
            _cause = "TOOLING: ffmpeg could not read an input -- the instrument did not run"
        elif 'no_plan' in t:
            _cause = "REFUSED_NO_PLAN: the locator declined to build a plan"
        elif _span is not None and _span > 127:
            _cause = ("DELAY_NOT_CONSTANT: delays span %.0f ms against the <127 ms "
                      "reconciliation window. A PROGRESSIVE OFFSET reaches this path "
                      "instead of the resampler. Candidate false rejection." % _span)
        elif 'useless to merge' in t:
            _cause = "REMOVED_LEFT_ONE: consequence recorded; underlying cause not extractable"
        else:
            _cause = "UNCLASSIFIED"          # reported by name, never folded into a default
        return {"error_cause": _cause,
                "delay_values_ms": sorted(set(_d)) or None,
                "delay_span_ms": _span,
                "reconciliation_window_ms": 127,
                "decline_msg": (top[0][:200] if top else None),
                "decline_stage": (fr[-1] if fr else None),
                "repair_failed": 'repair: failed' in t,
                "reason_saved": f'runs/decline-{i}.log.redacted'}
    return None


# EVERYTHING BELOW RUNS THE QUEUE. It was module-level, so `from merge_queue import
# gate_line` would have POSTED THE WHOLE CORPUS as a side effect of an import.
# That defect has cost this campaign four times tonight and once destroyed the raw
# data of a completed run; I then added two importable helpers to the file and made
# the temptation worse. Functions and constants above stay importable; the run does
# not start unless this file is the program.
if __name__ == '__main__':
    # RESUME FROM THE FILE THIS RUN WRITES TO, NOT A HARDCODED ONE. The output path
    # became a parameter and the resume path did not follow it, so a season run read
    # its `done` set from the CORPUS file. Nothing was skipped here only because the
    # id sets happen not to overlap -- with any overlap it would have silently
    # declared another run's rows complete and never asked for them. It also stamped
    # `resumed: true` on a fresh run, which is a false statement in the header field
    # I added an hour ago to make resumes visible.
    # A PID FILE, BECAUSE A NOTE IS NOT A CONTROL.
    # I killed my own shell TWICE tonight with `pgrep -f 'merge_queue.py'` -- the
    # pattern matches every command line CONTAINING it, including the one doing the
    # matching. After the first time I wrote "use pid-based handles" down, and the note
    # did not stop the second. A control is the thing that cannot be skipped: the runner
    # writes its pid, the killer reads the file, and no pattern is involved anywhere.
    try:
        with open('runs/merge_queue.pid', 'w') as _pf:
            _pf.write(str(os.getpid()) + "\n")
    except Exception:
        pass
    OUT_PATH = os.environ.get('OUT_FILE', 'runs/merge-queue.jsonl')
    done = set()
    if os.path.exists(OUT_PATH):
        for line in open(OUT_PATH):
            try:
                d = json.loads(line)
                if 'id' in d: done.add(d['id'])
            except Exception:
                pass

    out = open(OUT_PATH, 'a')          # APPEND: never truncate a result
    # PROBE UNCONDITIONALLY. Both bindings sat inside `if not done:` while two of the
    # four reads sit in the main loop, so EVERY RESUMED RUN raised NameError on its
    # first row. The file opens 'a' and `done` is built from it, so THE RESUME PATH IS
    # THE NORMAL PATH -- this fired on every run after the first, which for a 42-hour
    # queue is every run that matters. A peer found it by reading the file after I told
    # them to read it rather than trust my summary.
    #
    # AND IT MUST NOT DEFAULT TO A VALUE. "UNKNOWN" reads as "we asked and could not
    # tell"; on a resume the truth was "we never asked". That is not-measured against
    # measured-zero, in the very field I added to tell those apart.
    #
    # A resume is also exactly when the image may have changed underneath, which is the
    # case my own header comment describes -- so re-probing is the point, not a cost.
    # NB: this IS module scope (main-guard body), so plain assignment rebinds
    # the module global that artefacts() reads. A `global` statement here is
    # a SyntaxError -- and ast.parse ACCEPTED it while import did not, so
    # parsing is not a compile check and I have been treating it as one.
    PREEXISTING = snapshot_preexisting()
    if PREEXISTING:
        print(f"  {len(PREEXISTING)} pre-existing artefact(s) recorded and excluded",
              file=sys.stderr, flush=True)

    import subprocess as _sp
    try:
        _img = _sp.run(["curl","-s","--max-time","20",f"{API}/health"],
                       capture_output=True, timeout=40).stdout.decode()
        IMAGE = json.loads(_img)["git_commit"]
    except Exception:
        IMAGE = None
    CAPS = {k: image_emits(IMAGE, m, p) for k, (p, m) in CAPABILITY_MARKERS.items()} if IMAGE else None
    # HEADER ON EVERY RUN, RESUMES INCLUDED: the old one was written only when fresh,
    # so a resume across a redeploy left rows whose commit differed from the header's
    # while capabilities silently described the old image. Four `git show` per resume
    # costs nothing; per row it would be unaffordable.
    if True:
        out.write(json.dumps({"ask": "every error record, merged and verified as an artefact",
                              "resumed": bool(done), "n_already_done": len(done),
                              # THE DIGEST CARRIES ITS OWN DEFINITION. Two agents on
                              # this fleet emit `folder_hash` from DIFFERENT INPUTS:
                              # mine reads 00b243e5ba where two peers read 2c611366b7
                              # for the same folder. That is the codec-vocabulary
                              # defect in an IDENTIFIER, which is worse than in a
                              # value -- a disagreeing value gets argued about, but
                              # two digests under one name make two rows about ONE
                              # folder read as TWO folders, silently. A peer has
                              # already withdrawn a finding to it.
                              # I tried nine candidate definitions and reproduced
                              # NONE of theirs, which is the proof that a bare digest
                              # is unjoinable: it cannot be reverse-engineered, so it
                              # must arrive with its spec or not be cross-referenced.
                              "folder_hash_spec": ("sha256(dirname(error_file_path)).hexdigest()[:10] "
                                                   "-- error-tree path, NOT the master path, NOT the "
                                                   "library destination. Comparable ONLY with digests "
                                                   "carrying this same spec string."),
                              # A ROW THAT DOES NOT NAME THE CODE THAT PRODUCED IT IS NOT A
                              # MEASUREMENT OF ANYTHING IN PARTICULAR. The first 20 rows of
                              # this run were produced by an image seven hours stale, on the
                              # release most affected by the fixes it lacked, and the only
                              # way to find that out was for a peer to ask and for me to go
                              # and look. The artefact could not answer it.
                              "image_git_commit": IMAGE,
                          # WHAT THE DEPLOYED CODE CAN EVEN TELL US. Read from the
                          # source of the deployed commit, so an empty column is
                          # distinguishable from an unasked question. Probe validated
                          # on eight controls, both directions, before use.
                          "image_capabilities": CAPS,
                          "toolchain": _toolchain(),
                          "instrument": _instrument(),
                              "queue_order": "as returned by GET /errors -- NOT id order; "
                                             "the API groups by folder",
                              "asked_ids": ids, "n_asked": len(ids),
                              "acceptance": "SPEC_ZONE_A 4d: every track present, running to "
                                            "the master's VIDEO STREAM duration, no truncation",
                              "reference": "master video stream DURATION "
                                           "(== mediainfo video Duration; NOT format=duration)",
                              "outputs": "deleted after verification"}) + "\n")
        out.flush()

    todo = [i for i in ids if i not in done]
    print(f"asked {len(ids)}  done {len(done)}  to run {len(todo)}", file=sys.stderr, flush=True)
    fails = 0
    for n, i in enumerate(todo, 1):
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 5))
        r = api("/fusion", {"error_file_path": paths[i]})
        if not r or "queued" not in str(r.get("message", "")).lower():
            # THE API RESPONSE ECHOES error_file_path. Storing it raw put a full
            # library path into a row -- and rows travel. I scrubbed the LOG files
            # this morning and never the ROWS THAT QUOTE THEM: I fixed the artefact
            # I cared about and not the one that binds it, which is the same shape as
            # checking nlink on artefacts and never on logs.
            # REDACT AT THE BOUNDARY THE DATA CROSSES, not in the reader's care.
            rec = {"id": i, "verdict": "COULD_NOT_QUEUE",
                   "why": _safe_text(str(r)[:200])}
            # A ROW THAT DOES NOT NAME ITS POPULATION CANNOT BE COUNTED CORRECTLY LATER.
            rec["folder_hash"] = hashlib.sha256(os.path.dirname(paths[i]).encode()).hexdigest()[:10]
            # EMIT THE PEER'S DIGEST TOO rather than asking anyone to convert. A row
            # carrying both is joinable in either vocabulary; a row carrying one plus
            # a spec string still needs the reader to own the other implementation.
            # Mine is sha256 over the ERROR dirname, theirs sha1 over the MASTER
            # dirname -- I tried nine definitions and missed it by one cell of a
            # two-axis grid: sha1+error and sha256+master, never sha1+master.
            _mp = (ad.get(i) or {}).get("master_path")
            rec["folder_hash_master_sha1"] = (
                hashlib.sha1(os.path.dirname(_mp).encode()).hexdigest()[:10] if _mp else None)
            rec["cannot_help"] = CANNOT_HELP.get(i)          # None = interpretable
            _now = image_now()
        rec["container"] = CONTAINER
        rec["image_git_commit"] = _now or IMAGE
        rec["image_at_run_start"] = IMAGE
        rec["image_read_at_row"] = _now is not None
        if _now and IMAGE and _now != IMAGE:
            rec["image_changed_mid_run"] = True
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"[{n}/{len(todo)}] id={i} COULD_NOT_QUEUE", file=sys.stderr, flush=True)
            continue
        t0 = time.time(); found = []; seen_running = False
        while time.time() - t0 < TIMEOUT:
            time.sleep(POLL)
            found = artefacts(since)
            if found: break
            st = api("/fusion")
            # IDLE IS NOT A COMPLETION SIGNAL UNTIL MY FILE HAS BEEN SEEN RUNNING.
            # id 114 was recorded COULD_NOT_MEASURE because the container went briefly
            # idle between an ORPHANED job and mine -- before mine had started at all.
            # The runner read "idle, no current job" as "my merge finished and produced
            # nothing", when the true state was "my merge has not begun". A queue is
            # idle before your work as well as after it, and the two are the same
            # reading. So: only treat idle as completion once this file has actually
            # been observed as the current job.
            cj = (st or {}).get("current_job") or {}
            if cj.get("error_file_path") == paths[i]:
                seen_running = True
            if st and st.get("status") == "idle" and not st.get("current_job") and seen_running:
                found = artefacts(since)
                break
        if not found:
            # WHICH GAVE UP -- MY CLIENT, OR THE CONTAINER? Two rows carried
            # COULD_NOT_MEASURE for opposite reasons: id 114's client raced ahead of a
            # job that had not started, and id 127's client timed out while the
            # container was STILL WORKING on it, seven hours in. One token, two
            # findings. That is could-not-determine versus safe-to-delete arriving in
            # a verdict, and the row can tell them apart because the container says so.
            st = api("/fusion") or {}
            cj = st.get("current_job") or {}
            still_mine = cj.get("error_file_path") == paths[i]
            rec = {"id": i, "verdict": "COULD_NOT_MEASURE",
                   "why": f"no artefact within {int(time.time()-t0)}s",
                   "client_timed_out": True,
                   "container_still_on_this_file": bool(still_mine),
                   "container_status_at_giveup": st.get("status"),
                   "reading": ("THE CONTAINER DID NOT FAIL -- my client stopped waiting"
                               if still_mine else
                               "the container was not working on this file when I gave up")}
        else:
            mkv = [f for f in found if f.endswith('.mkv')]
            errlogs = [f for f in found if f.endswith('.log.error')]
            oklogs = [f for f in found
                      if f.endswith('.log') and not f.endswith('.log.error')]
            if not mkv:
                rec = {"id": i, "verdict": "DECLINED", "artefact": "log.error only"}
                r = decline_reason(errlogs, i)
                if r: rec.update(r)
            else:
                chk = subprocess.run([sys.executable, "check_output.py", mkv[0],
                                      ad[i]['master_path']], capture_output=True, timeout=300)
                try:
                    rec = {"id": i, **json.loads(chk.stdout.decode())}
                except Exception as e:
                    # an exception message can carry the path it failed to open
                    rec = {"id": i, "verdict": "COULD_NOT_MEASURE",
                           "why": _safe_text(f"probe: {e}")}
                g = gate_line(oklogs)
                if g: rec.update(g)
        # READ BEFORE DELETE. The probe was patched to FIND *.log and never to READ
        # it, so it collected the gate lines into the delete list and destroyed the
        # evidence the patch existed to capture -- the same defect as deleting the
        # decline reason, committed again after being fixed once.
        unread = consume(found)                          # outputs are disposable
        if unread:
            rec["artefacts_kept_unread"] = unread
        # A ROW THAT DOES NOT NAME ITS POPULATION CANNOT BE COUNTED CORRECTLY LATER.
        # folder_hash: three files whose figures matched to the millisecond turned out to
        # be one season sampled three times -- found by suspicion, not by the record.
        # cannot_help: forensic's 52 cannot succeed by construction. RECORDED, NOT APPLIED,
        # so the incidence is computable both ways instead of me choosing a denominator
        # once, invisibly, and everyone downstream inheriting it.
        # HOW LONG DID THIS FILE TAKE? I was asked whether a 42-minute job was wedged
        # and could not answer from my own artefacts -- no per-file duration was
        # recorded anywhere, so the only baseline available was two files someone
        # else had timed from outside my run.
        rec["elapsed_s"] = round(time.time() - t0, 1)
        # PRODUCED / CHECKED / VALIDATED ARE THREE DIFFERENT CLAIMS AND THIS ROW HAS
        # ONLY EVER SUPPORTED THE FIRST TWO. "Passed" without naming the property is
        # the count-instead-of-a-membership-test rule aimed at a verdict: ten files
        # passed a DURATION check, and nobody should be able to read that as ten
        # files being right. The case that founded the acceptance spec -- a file with
        # nothing after 21:21 -- PASSED EVERY COUNTER THE PIPELINE HAD. It was caught
        # by a person watching it.
        rec["checked"] = [
            "shortest audio/video stream duration vs the master video reference",
            "per-track duration source recorded (matroska tag vs missing)",
            "audio/video stop spread across tracks",
        ] if rec.get("verdict") in ("FULL_LENGTH", "TRUNCATED") else []
        rec["not_checked"] = [
            "content: no comparison of what the produced file CONTAINS against the master",
            "subtitle shortfall is REPORTED, never judged",
            "synchronisation beyond the container's own verifier probes",
        ]
        # AN EMPTY VALIDATED COLUMN IS THE FINDING; A MISSING ONE IS THE DEFECT.
        # Only a person who compared the file can fill this, and no automated column
        # in this fleet is entitled to.
        rec["validated_by"] = None
        rec["folder_hash"] = hashlib.sha256(os.path.dirname(paths[i]).encode()).hexdigest()[:10]
        _mp = (ad.get(i) or {}).get("master_path")
        rec["folder_hash_master_sha1"] = (
            hashlib.sha1(os.path.dirname(_mp).encode()).hexdigest()[:10] if _mp else None)
        rec["cannot_help"] = CANNOT_HELP.get(i)
        # A PASS FROM AN INSTRUMENT THAT CANNOT DETECT THE FAILURE MODE IS NOT A PASS,
        # and by the time anyone reads this row the reason is 400 rows upstream.
        rec.update(_ooc_mark(i))
        _now = image_now()
        rec["container"] = CONTAINER
        rec["image_git_commit"] = _now or IMAGE
        rec["image_at_run_start"] = IMAGE
        rec["image_read_at_row"] = _now is not None
        if _now and IMAGE and _now != IMAGE:
            rec["image_changed_mid_run"] = True
        out.write(json.dumps(rec) + "\n"); out.flush()
        bad = rec.get("verdict") not in ("FULL_LENGTH", "DECLINED")
        fails += bad
        print(f"[{n}/{len(todo)}] id={i} {rec.get('verdict')}"
              f"{'  <-- FAILURE' if bad else ''}", file=sys.stderr, flush=True)
    out.write(json.dumps({"summary": True, "n_asked": len(ids), "failures": fails}) + "\n")
    print(f"QUEUE DONE  failures {fails}", file=sys.stderr)
