"""Render the `.merge_plan` report: what was done to a merged file, for a human reader.

The report shows a timeline diagram, the lost regions, every offset used
(borrowed ones included), a per-region provenance table, and the same content
as sentences. This module only reads the `repair:` lines of a job log; a
missing field is reported as such, never filled in here.

Cell states distinguish why a value is missing:
    PRESENT       read from the log, by field name.
    DERIVED       computed from other emitted fields.
    COLLAPSED     emitted only as an aggregate coarser than the cell needs.
    ABSENT_FORMAT the log predates the field.
    NO_PRODUCER   no code emits the field.

Parsing notes: logs are split into lines and filtered by prefix (the first
`repair:` line follows `Logs:` without a blank line); a file is a job log only
if it carries a `repair: plan` line; fields are resolved by name, never by
position (e.g. `tolerance_ms` is not an absorption gate).
"""

from decimal import Decimal
from fractions import Fraction
import html
import hashlib
import re

def measure_corpus(log_paths, read=None):
    """Measure population statistics over the given job logs, at render time.

    Returns the dict rendered on the CORPUS row, so corpus-scale claims are
    measured rather than hard-coded. A "case" is a (master, candidate) pair;
    a log that does not identify its candidate is keyed by its master alone,
    and `key_basis` states which kind of key each case used.
    """
    reader = read or (lambda path: open(path, encoding="utf-8",
                                        errors="replace").read())
    keys, logs, rejected, basis = set(), 0, 0, {"digest": 0, "derived": 0,
                                                 "master_only": 0}
    # Distinct `repair: build` fingerprints the population spans.
    build_spread = {}
    # Widths of master fills by position.
    census = {"head": [], "interior": [], "tail": []}
    rejected_kind = {"died": 0, "merged": 0, "other": 0}
    # A production log carries the `Logs:` envelope; a lab replay starts at `repair:`.
    provenance = {"production_job_log": 0, "lab_replay": 0}
    # Fills contributed by replays, reported separately: most sit at the search
    # bound and cannot change the census conclusions.
    census_replay = {"head": [], "interior": [], "tail": []}
    # Files with no plan ("claimed") vs. those that also state a cause; a file
    # without a cause is counted as unexamined, not excluded.
    excluded = {"claimed": 0, "with_stated_cause": 0, "cause_anywhere": 0,
                "no_cause_stated": 0, "no_cause_field": 0, "unclassified_cause": 0,
                "tokened": 0, "untokened": 0, "tokens": {}}
    # `no_plan` lines have the form `repair: <outcome> cause=<token> for <path>: <prose>`.
    # The token is read by position, before the path, so the media path is never
    # parsed and a path segment cannot forge a `cause=` match.
    _TOKEN = re.compile(r"^\s*repair: \S+ cause=([A-Za-z0-9_]+) for ")
    # The `(unstated)` sentinel cannot match `[A-Za-z0-9_]+` because of its parentheses.
    _TOKEN_UNSTATED = re.compile(r"^\s*repair: \S+ cause=\(unstated\) for ")
    # Any `cause=` field, to count unrecognised tokens separately.
    _TOKEN_ANYFIELD = re.compile(r"^\s*repair: \S+ cause=\S+ for ")
    # Duration-gate state at production time; `would_refuse=True` marks a file
    # the enforcing gate would now decline.
    gate = {"enforcing_true": 0, "enforcing_false": 0, "would_refuse_true": 0}
    # `sources` digests every shipped module, so one `sources` value must map to
    # one `build` value; otherwise an emitter is broken.
    by_sources = {}
    brackets = {"compared": 0, "agree": 0, "disagree": 0,
                "emitted_true": 0, "emitted_false": 0, "incomparable": 0}
    for path in log_paths:
        text = reader(path)
        _job_for_exclusion = parse_job_log(text) if is_job_log(text) else {}
        _cause = _job_for_exclusion.get("declined")
        if _cause:
            excluded["cause_anywhere"] += 1
        # Match the prefix only: ` cause=<token>` sits between outcome and ` for `.
        _noplan = [ln for ln in text.splitlines() if "repair: no_plan" in ln]
        for ln in _noplan:
            hit = _TOKEN.search(ln)
            if hit:
                excluded["tokened"] += 1
                excluded["tokens"][hit.group(1)] = (
                    excluded["tokens"].get(hit.group(1), 0) + 1)
            elif _TOKEN_UNSTATED.search(ln):
                # Producer ran without a token: not a stated cause.
                excluded["untokened"] += 1
                excluded["no_cause_stated"] += 1
            elif _TOKEN_ANYFIELD.search(ln):
                # Unrecognised `cause=` token, counted separately.
                excluded["untokened"] += 1
                excluded["unclassified_cause"] += 1
            else:
                excluded["untokened"] += 1
                excluded["no_cause_field"] += 1
        if not (_job_for_exclusion.get("plan") or {}).get("pieces"):
            excluded["claimed"] += 1
            if _cause or any(_TOKEN.search(ln) for ln in _noplan):
                excluded["with_stated_cause"] += 1
        if not is_job_log(text):
            # Count non-plan logs by class so a repair that died before
            # planning stays in the denominator.
            rejected += 1
            rejected_kind["died" if "Traceback" in text else
                          "merged" if "first_delay_test" in text else
                          "other"] += 1
            continue
        logs += 1
        is_lab_replay = not any(line.startswith("Logs:")
                                for line in text.splitlines())
        if is_lab_replay:
            provenance["lab_replay"] += 1
        else:
            provenance["production_job_log"] += 1
        job = parse_job_log(text)
        if job.get("sources") and job.get("build"):
            key = job["sources"]["digest"]
            by_sources.setdefault(key, set()).add(
                tuple(sorted(job["build"].items())))
        check = job.get("output_check") or {}
        if check:
            if str(check.get("enforcing")).lower().startswith("true"):
                gate["enforcing_true"] += 1
            else:
                gate["enforcing_false"] += 1
            # The value can carry trailing prose: test the prefix.
            if str(check.get("would_refuse")).lower().startswith("true"):
                gate["would_refuse_true"] += 1
        pieces = (job.get("plan") or {}).get("pieces") or []
        for index, piece in enumerate(pieces):
            if piece.get("source") != "master":
                continue
            width = int(float(piece["master_end_ms"])) - int(float(piece["master_start_ms"]))
            position = ("head" if index == 0 else
                        "tail" if index == len(pieces) - 1 else "interior")
            census[position].append(width)
            if is_lab_replay:
                census_replay[position].append(width)
        # Compare the emitted `bound_only` with the width-derived signature.
        for entry in job.get("brackets") or []:
            width, flag = entry.get("width_ms"), entry.get("bound_only")
            if width is None or flag is None:
                brackets["incomparable"] += 1
                continue
            brackets["compared"] += 1
            emitted = str(flag).strip().lower().startswith("true")
            derived = abs(float(width) - float(SEARCH_BOUND_MS)) < 1e-9
            brackets["emitted_true" if emitted else "emitted_false"] += 1
            brackets["agree" if emitted == derived else "disagree"] += 1
        if job.get("build"):
            build_spread[tuple(sorted(job["build"].items()))] = (
                build_spread.get(tuple(sorted(job["build"].items())), 0) + 1)
        if job.get("candidate_digest"):
            basis["digest"] += 1
            candidate = ("digest", job["candidate_digest"])
        elif job.get("candidate_opaque_id"):
            basis["derived"] += 1
            candidate = ("derived", job["candidate_opaque_id"])
        else:
            basis["master_only"] += 1
            candidate = None
        keys.add((job.get("master_opaque_id"), candidate))
    # The reject split must partition the rejected count, and a distinct-count
    # cannot exceed its population.
    assert (rejected_kind["died"] + rejected_kind["merged"]
            + rejected_kind["other"] == rejected), (
        f"reject split does not partition: {rejected_kind} against {rejected} -- "
        f"the note would under-describe the class it claims to have measured")
    assert len(keys) <= logs, (
        f"n_distinct ({len(keys)}) exceeds n ({logs}): a distinct-count cannot "
        f"exceed its population")
    return {
        "logs": logs,
        "fills_above_the_bound": (
            f"{sum(1 for w in census['head'] + census['tail'] + census['interior'] if w > SEARCH_BOUND_MS)} "
            f"master fill(s) here exceed the 100 s search bound: "
            f"{sum(1 for w in census['head'] if w > SEARCH_BOUND_MS)} at the head, "
            f"{sum(1 for w in census['tail'] if w > SEARCH_BOUND_MS)} at the tail, "
            f"{sum(1 for w in census['interior'] if w > SEARCH_BOUND_MS)} INTERIOR"),
        # Every bracket width is a multiple of REFINE_STEP by construction, so
        # the grid is not a check; a width equal to PROBE_STEP + PROBE_WINDOW
        # marks a bracket that was never refined.
        "fill_census": (
            f"master fills by position: "
            f"head {len(census['head'])}, interior {len(census['interior'])}, "
            f"tail {len(census['tail'])}. "
            f"THE GRID PROPERTY IS NOT A TEST AND ITS OPPORTUNITY COUNT IS ZERO: "
            f"a refined bracket's width is k x {REFINE_STEP_MS} + "
            f"{REFINE_WINDOW_MS} ms and an unrefined one's is "
            f"k x {PROBE_STEP_MS} + {PROBE_WINDOW_MS} ms, and ALL FOUR CONSTANTS "
            f"ARE MULTIPLES OF {REFINE_STEP_MS} ms, so EVERY width lands on the "
            f"{REFINE_STEP_MS} ms grid BY CONSTRUCTION. m = 0, UNTESTED, NOT "
            f"PASSED. The {SEARCH_BOUND_MS} ms bound is "
            f"{PROBE_STEP_MS} + {PROBE_WINDOW_MS} -- the signature of a bracket "
            f"that was NEVER REFINED, which is the finding that survives: A FILL "
            f"REGION'S WIDTH IS A PROPERTY OF THE INSTRUMENT AND NOT OF THE "
            f"MEDIA. The distribution is reported below because it is "
            f"informative; it is not offered as a check that passed: "
            + "; ".join(
                (lambda at_bound, rest: (
                    f"{name} {len(widths)} fill(s), of which {len(at_bound)} "
                    f"sit exactly at the bound (never refined) and "
                    f"{len(rest)} do not"))(
                    [w for w in widths if w == SEARCH_BOUND_MS],
                    [w for w in widths if w != SEARCH_BOUND_MS])
                for name, widths in (("head", census["head"]),
                                     ("interior", census["interior"]),
                                     ("tail", census["tail"])))
            + ". distinct interior widths: "
            + " ".join(f"{w}x{census['interior'].count(w)}"
                       for w in sorted(set(census["interior"])))
            + ". tail widths: "
            + " ".join(str(w) for w in sorted(census["tail"]))
            + ((". OF THIS POPULATION, THE LAB REPLAYS CONTRIBUTE: "
                + "; ".join(
                    f"{name} {len(widths)} fill(s) of which "
                    f"{sum(1 for w in widths if w != SEARCH_BOUND_MS)} "
                    f"could have gone either way"
                    for name, widths in (("head", census_replay["head"]),
                                         ("interior", census_replay["interior"]),
                                         ("tail", census_replay["tail"])))
                + f". A LOG COUNT IS NOT AN OPPORTUNITY COUNT: these "
                  f"{provenance['lab_replay']} replays add "
                  f"{sum(len(v) for v in census_replay.values())} fill(s) and "
                  f"{sum(1 for v in census_replay.values() for w in v if w != SEARCH_BOUND_MS)} "
                  f"opportunit(y/ies). Ask of a corpus extension what the "
                  f"opportunity count asks of a check: how many of the files "
                  f"added could have changed the answer?")
               if provenance["lab_replay"] else "")),
        # Agreement is only meaningful when both `bound_only` values occur.
        "bracket_agreement": (
            f"{brackets['agree']} of {brackets['compared']} emitted brackets "
            f"agree with the derived signature `width == the {SEARCH_BOUND_MS} "
            f"ms search bound`"
            + (f"; OF WHICH COULD HAVE DISAGREED: {brackets['compared']} "
               f"({brackets['emitted_true']} emitted True, "
               f"{brackets['emitted_false']} emitted False, so the corpus "
               f"carries both values and the agreement is not a constant)"
               if brackets["compared"] and brackets["emitted_true"]
               and brackets["emitted_false"] else
               f"; OF WHICH COULD HAVE DISAGREED: 0 -- "
               + ("no emitted bracket in this corpus carries both `width_ms` "
                  "and `bound_only`, so the signature is UNTESTED here, not "
                  "confirmed"
                  if not brackets["compared"] else
                  "every emitted `bound_only` in this corpus carries the SAME "
                  "value, so agreement is what a constant reader would also "
                  "score and this corpus does not discriminate"))
            + (f". {brackets['disagree']} DISAGREE -- read the BRACKET rows: "
               f"the locator and this reader have parted company"
               if brackets["disagree"] else "")
            + (f". {brackets['incomparable']} bracket(s) carry only one of the "
               f"two fields and are not scored either way"
               if brackets["incomparable"] else "")),
        "build_spread": (
            f"{len(build_spread)} distinct `repair: build` fingerprint(s) across "
            f"{sum(build_spread.values())} log(s) that carry one; "
            f"{logs - sum(build_spread.values())} carry NONE and are undatable. "
            f"EVERY CORPUS-SCALE CLAIM BELOW SPANS THAT SPREAD -- it is not a "
            f"statement about one build, and this reader cannot say which build "
            f"produced which row: a build value is a content digest with no "
            f"ordering"),
        # The supplied logs are a preserved subset, not a full census.
        "population_is_a_preservation": (
            "the logs handed to this render are a PRESERVED SUBSET, not a "
            "census. They survive because something hard-linked them before the "
            "runner deleted its sources; a log never preserved leaves no trace "
            "here, not even a gap. This reader cannot state how many were "
            "produced, only how many it was given"),
        "excluded_claimed": excluded["claimed"],
        "excluded_with_stated_cause": excluded["with_stated_cause"],
        "excluded_no_cause_stated": excluded["no_cause_stated"],
        "excluded_no_cause_field": excluded["no_cause_field"],
        "excluded_unclassified_cause": excluded["unclassified_cause"],
        "exclusion_note": (
            f"AN ABSENT LINE IS NOT AN EXCLUSION. {excluded['claimed']} file(s) "
            f"handed to this render emitted NO PLAN -- these are the ones a merge "
            f"denominator would SUBTRACT -- and of them "
            f"{excluded['with_stated_cause']} carry a cause anyone can read. "
            + (f"THE OTHER {excluded['claimed'] - excluded['with_stated_cause']} "
               f"ARE SILENT AND THIS READER DOES NOT COUNT THEM AS EXCLUDED: they "
               f"are UNEXAMINED. `excluded because completely different` and "
               f"`declined for some other reason` are the same observable from "
               f"here, and only one of them is an exclusion. "
               if excluded["claimed"] != excluded["with_stated_cause"] else
               "The two counts are EQUAL on this material, which means this "
               "column had no opportunity to discriminate here -- m = 0 for the "
               "difference, and that is UNTESTED, not clean. ")
            + (f"`no_plan` CAUSE LINES, READ WITHOUT EVER PARSING THE PATH: "
               f"{excluded['tokened']} carry a `cause=` token"
               + (" (" + ", ".join(f"{k} x{v}" for k, v in
                                   sorted(excluded["tokens"].items())) + ")"
                  if excluded["tokens"] else "")
               + f" and {excluded['untokened']} carry a reason with NO token. "
               f"AN UNTOKENED REASON IS NOT A STATED CAUSE HERE: the constant it "
               f"carries asserts that no measurement was available on paths where "
               f"a measurement WAS available and said no, which is a false "
               f"qualifier and worse than a blank because it closes the question. "
               if (excluded["tokened"] or excluded["untokened"]) else "")
            + f"POSITIVE CONTROL: {excluded['cause_anywhere']} log(s) in this "
              f"population DO carry a declared cause, so the emitter and this "
              f"parser both fire; an absence below is a property of the file and "
              f"not of the instrument"),
        "rejected_by_structure": rejected,
        # Rejected logs split into: Traceback (died before planning), merge
        # path with `first_delay_test` (no repair), and other.
        "rejected_note": (
            f"handed to this render and NOT counted: they carry no `repair: "
            f"plan` line, so there is no plan in those bytes to describe. "
            f"MEASURED SPLIT of this class, not assumed: {rejected_kind['died']} "
            f"carry a Traceback -- a repair that died before planning, which "
            f"leaves no artefact -- and {rejected_kind['merged']} went through "
            f"the MERGE path with a delay test and NO repair attempted, WHICH "
            f"DO PRODUCE A FILE. Every claim below is blind to both classes, "
            f"and the second has artefacts this report will never describe at "
            f"all. AND A THIRD BUCKET EXISTS AND IS NAMED HERE EVEN WHEN EMPTY: "
            f"{rejected_kind['other']} match NEITHER pattern. "
            + ("Zero today, which is an UNEXERCISED BRANCH and not a "
               "reassurance -- m = 0 for this bucket. A log recording that the "
               "INSTRUMENT DID NOT RUN (a tool that could not read its input) "
               "carries no Traceback and no delay test, so it would land here "
               "and be counted as a refusal by anyone reading the two named "
               "classes as the whole split. `vmsam-ci` measures that class in "
               "its own corpus, where it is not zero: RELAYED, its figure, "
               "about its container substrate."
               if not rejected_kind["other"] else
               "THIS BUCKET IS NON-EMPTY: a decline class exists that neither "
               "named pattern describes, and it may be a NON-OBSERVATION "
               "counted as a refusal. Read those logs before quoting any "
               "decline-cause census over this corpus.")),
        "distinct_cases": len(keys),
        "unit": "(master, candidate) pair",
        "key_basis": f"{basis['digest']} from an emitted candidate_digest (read), "
                     f"{basis['derived']} from a path digest I derive myself "
                     f"(constructed), {basis['master_only']} from the master "
                     f"alone (INFERRED: two candidates merged toward one master "
                     f"would count as one)",
        "build_vs_sources": (
            "consistent: no `sources` digest carries two different `build` "
            "digests"
            if all(len(v) <= 1 for v in by_sources.values())
            else "CONTRADICTION: " + ", ".join(
                f"sources={k} carries {len(v)} different build digests"
                for k, v in by_sources.items() if len(v) > 1)
            + ". `build` cannot move while `sources` holds still -- the two "
              "repair modules are inside the 27 shipped files -- so one of the "
              "two emitters is broken")
            if by_sources else "not checkable: no artefact carries both lines",
        "gate_state": f"{gate['enforcing_false']} produced with the duration "
                      f"gate INERT (enforcing=False), {gate['enforcing_true']} "
                      f"with it enforcing. {gate['would_refuse_true']} carry "
                      f"would_refuse=True, so under the SHIPPING configuration "
                      f"they would be DECLINED and would not exist as produced "
                      f"files. THIS CORPUS IS NOT THE POPULATION THE CONTAINER "
                      f"NOW PRODUCES",
        "provenance": f"{provenance['production_job_log']} production job logs "
                      f"(carrying the `Logs:` envelope), "
                      f"{provenance['lab_replay']} lab replays (starting at "
                      f"`repair:`). NOT one population: a replay exercises the "
                      f"repair path and not a merge, and can emit a log without "
                      f"producing an artefact",
        "measured": "at render time, over the logs handed to this render",
        # Logs exist only where a repair completed, so the sample is not independent.
        "caveat": "NOT an independent sample: these logs exist only where a "
                  "repair COMPLETED -- which is not the same as `produced`. A "
                  "run that failed on a tool fault can leave a log and an "
                  "artefact under a NOVERDICT name; that class is countable "
                  "from `undelivered state=` and is not counted here. A COST ASYMMETRY IS MEASURED UPSTREAM -- "
                  "probe decoding is far dearer on lossless multichannel than "
                  "on EAC3 -- but NO LOSS IS ESTABLISHED: zero known instances "
                  "of a log missing for that reason, and the one expensive case "
                  "measured did finish. A mechanism is not a frequency. And "
                  "undatable from any artefact -- no build or timestamp field "
                  "is emitted anywhere",
    }


# Cell states.

PRESENT = "present"
DERIVED = "derived"
COLLAPSED = "collapsed"
ABSENT_FORMAT = "absent-from-this-format"
NO_PRODUCER = "no-producer"
# The quantity does not apply; another statistic decided (see `decided_by`).
NOT_DEFINED = "not-defined-here"
# The code path exists but this file did not trigger it.
NOT_EXERCISED = "not-exercised-here"
# The caller supplied no corpus population.
NOT_SUPPLIED = "not-supplied-to-this-render"
# The measurement was attempted but did not complete (e.g. a timeout).
NOT_MEASURED = "not-measured"

_STATE_MARK = {
    PRESENT: "",
    DERIVED: "~",           # computed, not read
    COLLAPSED: "^",         # emitted, but aggregated above this granularity
    ABSENT_FORMAT: "-",     # this format did not carry it
    NO_PRODUCER: "x",       # nothing emits it
    NOT_DEFINED: "\u00b7",   # not applicable here; decided elsewhere
    NOT_EXERCISED: "\u25cb",  # live branch, input not produced here
    NOT_SUPPLIED: "?",       # the caller supplied no population
    NOT_MEASURED: "\u2049",   # attempted, not completed -- re-measure
}

_STATE_WORD = {
    PRESENT: "read from the bytes by name",
    DERIVED: "derived from other emitted fields",
    COLLAPSED: "emitted, but aggregated above this granularity",
    ABSENT_FORMAT: "absent from this artefact's format",
    NO_PRODUCER: "no producer emits this",
    NOT_DEFINED: "not defined in this case; the decision was made elsewhere",
    NOT_EXERCISED: "the branch is live; this artefact does not produce its input",
    NOT_SUPPLIED: "the caller supplied no population for this render",
    NOT_MEASURED: "the measurement was attempted and did not complete; "
                  "re-measure, do not read as a negative",
}


class Cell:
    """A value with its cell state and an optional note."""

    __slots__ = ("value", "state", "note")

    def __init__(self, value, state, note=None):
        self.value = value
        self.state = state
        self.note = note

    def __bool__(self):
        return self.state in (PRESENT, DERIVED)

    def text(self):
        """Return the short form: the value with its state mark, or the mark alone."""
        if self.state in (PRESENT, DERIVED):
            return f"{_STATE_MARK[self.state]}{self.value}"
        return _STATE_MARK[self.state]

    def long(self):
        """Return the value (if any) followed by its state description and note."""
        if self.state in (PRESENT, DERIVED):
            return f"{self.value}  ({_STATE_WORD[self.state]}"\
                   f"{'; ' + self.note if self.note else ''})"
        return f"{_STATE_WORD[self.state]}"\
               f"{'; ' + self.note if self.note else ''}"


def absent(note=None):
    return Cell(None, ABSENT_FORMAT, note)


def no_producer(note):
    """Return a NO_PRODUCER Cell carrying `note`."""
    return Cell(None, NO_PRODUCER, note)


# Parsing.

def split_fields(text):
    """Parse `key=value` pairs whose values may contain spaces.

    Key starts are found at bracket depth zero and each value runs to the next
    key, so `source=master video Duration (mediainfo)` stays whole. The first
    occurrence of a key wins.
    """
    depth, starts = 0, []
    for index, character in enumerate(text):
        if character in "[(":
            depth += 1
            continue
        if character in "])":
            depth = max(0, depth - 1)
            continue
        if depth or character != "=":
            continue
        back = index
        while back > 0 and (text[back - 1].isalnum() or text[back - 1] == "_"):
            back -= 1
        name = text[back:index]
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if back > 0 and not text[back - 1].isspace():
            continue
        starts.append((back, name, index + 1))

    fields = {}
    for position, (back, name, value_start) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(text)
        fields.setdefault(name, text[value_start:end].strip())
    return fields


def _decimal(text):
    try:
        return Decimal(str(text))
    except Exception:
        return None


def _trim(value):
    """Format a Decimal without trailing zeros (1479979.000 -> 1479979)."""
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def is_job_log(text):
    """Return True when the text carries a `repair: plan` line."""
    return any(line.startswith("repair: plan ") for line in text.splitlines())


def parse_job_log(text):
    """Parse a job log's text into a job dict, line by line, by prefix.

    The first `repair:` line follows `Logs:` without a blank line, so the text
    is never split on blank-line separators.
    """
    lines = text.splitlines()
    job = {
        "master_line_present": False,
        "master_opaque_id": None,
        "master_path": None,
        "candidate_opaque_id": None,
        "candidate_path": None,
        "candidate_digest": None,
        "plan": None,
        # Every `repair: plan` line in order; `plan` is the one drawn, the
        # others become PLAN_DUPLICATE rows.
        "plan_lines": [],
        "audios": {},
        "subtitles": [],
        "regions_added": {},
        "regions_cut": {},
        "regions_used": {},
        "skipped": [],
        "refused": [],
        "declined": None,
        "failed": None,
        "build": None,
        "sources": None,
        "unparsed": [],
        "predicted_refusals": [],
        "prediction": None,
        "output_check": None,
        "summary_counts": None,
        "foreign_lines": [],
        "locator_measurements": [],
        "locator_notes": [],
        "brackets": [],
        # Delivery-gate verdicts (`fabricated_dropped|fabricated_kept`), for the
        # human summary; the lines also stay in `unparsed`.
        "delivery": [],
        "segments": [],
        "output_durations": None,
        # `picture_only_shift` runs: logged, never acted on.
        "picture_only_shifts": [],
        # `owner_judgment_pending` zones: audio ok, video differs -- logged, declined.
        "owner_judgment": [],
    }

    for line in lines:
        if not line.startswith("repair: "):
            # Lines outside the `repair:` stream are counted, not dropped.
            stripped_line = line.strip()

            # `[change_point_locator]` lines carry locator measurements; the
            # tag is tested on the dedented line (the logger indents with tabs).
            if stripped_line.startswith(_LOCATOR_TAG):
                rest = stripped_line[len(_LOCATOR_TAG):].strip()
                fields = split_fields(rest)
                if fields and "=" not in " ".join(fields.keys()):
                    job["locator_measurements"].append(fields)
                else:
                    # Free text may carry a media path: keep only a digest,
                    # length and path flag, never the text.
                    job["locator_notes"].append({
                        "digest": hashlib.md5(
                            stripped_line.encode("utf-8", "replace")).hexdigest()[:12],
                        "chars": len(stripped_line),
                        "carries_path": bool(_PATH.search(stripped_line)),
                        "tail": re.sub(r"[^a-z ]", "",
                                       stripped_line.lower()[-42:]).strip(),
                    })
                continue

            if stripped_line and not line.startswith(("Merged", "Logs:", "We was",
                                                      "Multiple delay")):
                # Free text may carry a media path that no pattern can redact:
                # record only a digest, length and path flag.
                stripped = line.strip()
                job["foreign_lines"].append({
                    "digest": hashlib.md5(
                        stripped.encode("utf-8", "replace")).hexdigest()[:12],
                    "chars": len(stripped),
                    "carries_path": bool(_PATH.search(stripped)),
                    "tail": re.sub(r"[^a-z ]", "", stripped.lower()[-42:]).strip(),
                })
            continue
        body = line[len("repair: "):]

        if body.startswith("master "):
            job["master_line_present"] = True
            # Keep the opaque id beside the path as a safe form to cite.
            job["master_path"] = body[len("master "):].strip()
            job["master_opaque_id"] = opaque_id(job["master_path"])
            continue

        if body.startswith("candidate_digest "):
            # Identifies the candidate without naming it.
            job["candidate_digest"] = body[len("candidate_digest "):].strip()
            continue

        if body.startswith("plan "):
            rest = body[len("plan "):]
            kind = rest.split(" ", 1)[0]
            fields = split_fields(rest)
            pieces_text = rest.partition("pieces=")[2]

            # Keep every plan field by name; normalise the known ones.
            plan = dict(fields)
            job["plan_lines"].append(plan)
            plan.update({
                "kind": kind,
                "language": fields.get("language"),
                "quantum_ms": fields.get("quantum"),
                "speed_margin": fields.get("speed_margin"),
                "speed_margin_absent_reason": fields.get("speed_margin_absent_reason"),
                "fidelity_margin": fields.get("fidelity_margin"),
                "decided_by": fields.get("decided_by"),
                "pieces": parse_pieces(pieces_text),
            })
            continue

        matched = re.match(r"audio track (\d+) (.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(2))
            # `residual=probes=4 worst=...` parses as `residual=probes=4`;
            # also expose the inner `probes` field by name.
            residual = fields.get("residual")
            if residual is not None and "=" in residual:
                inner, _, value = residual.partition("=")
                fields.setdefault(inner, value)
            job["audios"][order] = fields
            continue

        matched = re.match(r"ADDED audio track (\d+) (\w+) ([\d.]+)-([\d.]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(5))
            # Pass every field through, not a fixed list.
            entry = {name: value for name, value in fields.items()}
            entry.update({
                "kind": matched.group(2),
                "master_start_ms": _decimal(matched.group(3)),
                "master_end_ms": _decimal(matched.group(4)),
                "from": fields.get("from"),
            })
            job["regions_added"].setdefault(order, []).append(entry)
            continue

        # `USED`: candidate regions in the output with the applied offset.
        # `ADDED` is master fill, `CUT` is discarded candidate material.
        matched = re.match(r"USED audio track (\d+) master ([\d.-]+)-([\d.-]+) "
                           r"candidate ([\d.-]+)-([\d.-]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(6))
            used = dict(fields)
            used.update({
                "master_start_ms": _decimal(matched.group(2)),
                "master_end_ms": _decimal(matched.group(3)),
                "candidate_start_ms": _decimal(matched.group(4)),
                "candidate_end_ms": _decimal(matched.group(5)),
                # A negative offset is valid (the candidate leads the master).
                "offset_ms": _decimal(fields.get("offset_ms")),
                "offset_ms_present": "offset_ms" in fields,
            })
            job["regions_used"].setdefault(order, []).append(used)
            continue

        matched = re.match(r"CUT audio track (\d+) candidate ([\d.]+)-([\d.?]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(4))
            end_text = matched.group(3)
            cut = dict(fields)
            cut.update({
                "candidate_start_ms": _decimal(matched.group(2)),
                "candidate_end_ms": None if end_text == "?" else _decimal(end_text),
                "dropped_ms": (None if fields.get("dropped_ms") == "UNMEASURED"
                               else _decimal(fields.get("dropped_ms"))),
                "dropped_unmeasured": fields.get("dropped_ms") == "UNMEASURED",
                "where": fields.get("where"),
            })
            job["regions_cut"].setdefault(order, []).append(cut)
            continue

        matched = re.match(r"subtitle track (\d+) (.*)$", body)
        if matched:
            fields = split_fields(matched.group(2))
            fields["stream_order"] = int(matched.group(1))
            job["subtitles"].append(fields)
            continue

        if body.startswith("sources "):
            # Digest of all shipped `.py` files (`build` covers the repair modules only).
            rest = body[len("sources "):]
            digest = rest.split()[0] if rest.split() else None
            fields = split_fields(rest)
            job["sources"] = dict(fields)
            job["sources"]["digest"] = digest
            continue

        if body.startswith("build "):
            # `repair: build <module:sha12> ...`, one digest per module.
            job["build"] = {}
            for token in body[len("build "):].split():
                name, _, digest = token.rpartition(":")
                if name and digest:
                    job["build"][name] = digest
            continue

        # `FAILED` is a tool fault before any verdict; `DECLINED` is a gate decision.
        if body.startswith("FAILED"):
            job["failed"] = body[len("FAILED"):].strip(": ").strip() or "(no reason given)"
            continue

        if body.startswith("undelivered "):
            # Recognised but not rendered; keeps it out of UNPARSED.
            continue

        if body.startswith("DECLINED"):
            # The file was not produced.
            job["declined"] = body[len("DECLINED"):].strip(": ").strip() or "(no reason given)"
            continue

        if body.startswith("SKIPPED "):
            # A refused segment had a candidate that was dropped (invalid
            # offset); its bounds are parsed so the figure can place it.
            matched = re.match(r"SKIPPED segment master ([\d.]+)-([\d.]+)(.*)$", body)
            if matched:
                fields = split_fields(matched.group(3))
                job["refused"].append({
                    "master_start_ms": _decimal(matched.group(1)),
                    "master_end_ms": _decimal(matched.group(2)),
                    "dropped_ms": _decimal(fields.get("dropped_ms")),
                    "reason": matched.group(3).partition("DECLINED:")[2].strip()
                              or None})
            job["skipped"].append(body)
            continue

        # `repair: bracket <i> low_ms=... width_ms=... bound_only=...` and
        # `repair: segment <i> ...`: all fields are kept by name.
        if body.startswith("bracket "):
            rest = body[len("bracket "):]
            index, _, tail = rest.partition(" ")
            entry = split_fields(tail)
            entry["index"] = index
            job["brackets"].append(entry)
            continue
        if body.startswith("segment "):
            rest = body[len("segment "):]
            index, _, tail = rest.partition(" ")
            entry = split_fields(tail)
            entry["index"] = index
            job["segments"].append(entry)
            continue
        if body.startswith("picture_only_shift "):
            # `zone=<i> master_s=[a, b] residual_frames=<n> cuts=<k> audio=continuous`.
            fields = split_fields(body[len("picture_only_shift "):])
            job["picture_only_shifts"].append(fields)
            continue
        if body.startswith("picture_only_shift_summary ") or body.startswith(
                "picture_only_shift_unpaired "):
            # Recognised but not rendered; keeps it out of UNPARSED.
            continue
        if body.startswith("owner_judgment_pending "):
            # `zone=<i> reason=<...> master_start_s=<f> master_end_s=<f>
            # candidate_start_s=<f> candidate_end_s=<f> audio_cut_s=<f|None>
            # video_cut_s=<f|None> picture_shift_ms=<f|None> frames_compared=<int>`.
            fields = split_fields(body[len("owner_judgment_pending "):])
            job["owner_judgment"].append(fields)
            continue
        if body.startswith("owner_judgment_pending_summary "):
            # Recognised but not rendered; keeps it out of UNPARSED.
            continue
        # Refusals the gate expects, then whether the outcome agreed.
        if body.startswith("PREDICTED_REFUSAL "):
            job["predicted_refusals"].append(
                split_fields(body[len("PREDICTED_REFUSAL "):]))
            continue
        if body.startswith("prediction "):
            job["prediction"] = split_fields(body[len("prediction "):])
            continue
        if body.startswith("output durations "):
            job["output_durations"] = split_fields(
                body[len("output durations "):])
            continue

        if body.startswith("output file "):
            fields = split_fields(body)
            # `audio 1/1 subtitles 8/8` precedes the keys: parse it separately.
            counts = re.search(r"audio (\S+) subtitles (\S+)", body)
            if counts:
                fields.setdefault("audio_tracks", counts.group(1))
                fields.setdefault("subtitle_tracks", counts.group(2))
            job["output_check"] = fields
            continue

        # Unrecognised `repair:` lines are kept as UNPARSED.
        matched = re.match(r"repaired for (.*?): (.*)$", body)
        if matched:
            job["candidate_path"] = matched.group(1)
            job["candidate_opaque_id"] = opaque_id(matched.group(1))
            job["summary_counts"] = matched.group(2)
            continue

        matched = re.match(r"fabricated_(dropped|kept) (.*)$", body)
        if matched:
            gate = {"verdict": matched.group(1)}
            gate.update(split_fields(matched.group(2)))
            job["delivery"].append(gate)

        job["unparsed"].append(body[:120])

    job["plan"] = select_plan(job["plan_lines"])
    return job


def select_plan(plan_lines):
    """Choose the plan line to draw: the last one carrying `pieces`, else the last.

    The orchestrator also writes a geometry-less plan line, so position alone
    would lose the pieces. Other lines become PLAN_DUPLICATE rows.
    """
    if not plan_lines:
        return None
    with_geometry = [plan for plan in plan_lines if plan.get("pieces")]
    return (with_geometry or plan_lines)[-1]


def parse_pieces(text):
    """Parse `c0-160000 m160000-260000 ...` into the file's piece list.

    These are the fallback (track-independent) pieces; their master-side
    bounds are identical for every track, so they place per-track regions.
    """
    pieces = []
    for token in text.split():
        matched = re.fullmatch(r"([cm])(-?[\d.]+)-(-?[\d.]+)", token)
        if not matched:
            continue
        pieces.append({
            "source": {"c": "candidate", "m": "master"}[matched.group(1)],
            "master_start_ms": _decimal(matched.group(2)),
            "master_end_ms": _decimal(matched.group(3)),
        })
    return pieces


def opaque_id(path):
    """Return a stable opaque id for a path (`md5(path)[:16]`).

    Same construction as the working-directory key in
    `merge_video_repair.build_repaired_video_object`.
    """
    if not path:
        return None
    return hashlib.md5(str(path).strip().encode("utf-8", "replace")).hexdigest()[:16]


# Format generation.

def format_generation(job):
    """Return (generation, label) of the log format, detected by structure."""
    if job["plan"] is None:
        return 0, "pre-plan (no `repair: plan` line at all)"
    if not job["regions_cut"] and not job["regions_added"]:
        return 1, "per-track (track lines, no per-region ADDED/CUT lines)"
    if not job["master_line_present"]:
        return 2, "region-level (ADDED/CUT present, master not named)"
    return 3, "current (master named, region-level)"


# Offset recovery.

def recover_offsets(job, stream_order):
    """Derive the offset applied to each candidate piece of a track (list of Cells).

    Inverts `candidate_start = master_start + offset` using the CUT line
    bounds; each predicted piece end must land exactly on an emitted cut
    boundary, otherwise the remaining regions get no value. A track with no
    CUT line yields no offsets.
    """
    plan = job["plan"]
    if not plan:
        return []
    candidate_pieces = [p for p in plan["pieces"] if p["source"] == "candidate"]
    cuts = job["regions_cut"].get(stream_order) or []
    if not candidate_pieces:
        return []
    if not cuts:
        return [Cell(None, NO_PRODUCER,
                     "no CUT line for this track, so nothing to derive from: "
                     "the offset is emitted under no name at all")
                for _ in candidate_pieces]

    head = next((c for c in cuts if c["where"] == "head"), None)
    if head is None or head["candidate_end_ms"] is None:
        return [Cell(None, NO_PRODUCER,
                     "no head CUT line, so the first piece's candidate start is "
                     "not emitted and the chain has no anchor")
                for _ in candidate_pieces]

    results = []
    source_start = head["candidate_end_ms"]
    for index, piece in enumerate(candidate_pieces):
        offset = source_start - piece["master_start_ms"]
        results.append(Cell(_trim(offset), DERIVED,
                            "candidate_start_ms - master_start_ms"))
        source_end = source_start + (piece["master_end_ms"] - piece["master_start_ms"])
        if index + 1 >= len(candidate_pieces):
            break
        following = [c for c in cuts
                     if c["where"] in ("interior", "tail")
                     and c["candidate_start_ms"] == source_end]
        if not following or following[0]["candidate_end_ms"] is None:
            # Chain broken: mark this and the following regions.
            for _ in candidate_pieces[index + 1:]:
                results.append(Cell(None, NO_PRODUCER,
                                    "the CUT chain does not reach this piece: no "
                                    "emitted cut boundary matches the predicted "
                                    "source end"))
            break
        source_start = following[0]["candidate_end_ms"]
    return results


def track_regions(job, stream_order):
    """Return one row per region of a track, in master timeline order.

      CANDIDATE  taken from the candidate; offset from the USED line or derived.
      MASTER     filled from the master (ADDED line, `from=`).
      SILENCE    filled with silence (`from=silence`).
      MASTER?    a master piece with no matching ADDED line.
      LOST       candidate material not in the output (CUT line, candidate timeline).
    """
    plan = job["plan"]
    if not plan:
        return []
    # The emitted USED offset wins over the derived one; a disagreement is
    # noted (it can only reveal a format change or parser regression).
    derived = recover_offsets(job, stream_order)
    emitted = {}
    for region in job.get("regions_used", {}).get(stream_order) or []:
        if region.get("offset_ms_present"):
            emitted[_trim(region["master_start_ms"])] = region["offset_ms"]
    offsets = derived
    added = {(_trim(r["master_start_ms"]), _trim(r["master_end_ms"])): r
             for r in job["regions_added"].get(stream_order) or []}

    rows, candidate_index = [], 0
    for piece in plan["pieces"]:
        key = (_trim(piece["master_start_ms"]), _trim(piece["master_end_ms"]))
        if piece["source"] == "master":
            region = added.get(key)
            matched_by = None
            if region is None:
                # Fallback join by truncated bounds (plan `m0-983` vs ADDED
                # `0-983.54`); only a unique match is joined.
                near = [(ck, cr) for ck, cr in added.items()
                        if ck[0] is not None and ck[1] is not None
                        and str(int(float(ck[0]))) == str(key[0])
                        and str(int(float(ck[1]))) == str(key[1])]
                if len(near) == 1:
                    candidate_key, region = near[0]
                    matched_by = (f"joined by TRUNCATION, not exactly: the plan "
                                  f"says {key[0]}-{key[1]} and the ADDED line "
                                  f"says {candidate_key[0]}-{candidate_key[1]} "
                                  f"-- one boundary, two emitted "
                                  f"representations")
                elif len(near) > 1:
                    matched_by = (f"NOT JOINED: {len(near)} ADDED lines truncate "
                                  f"to {key[0]}-{key[1]} and this reader will "
                                  f"not guess between them")
            if region is None:
                rows.append({
                    "kind": "MASTER?",
                    "master_start_ms": piece["master_start_ms"],
                    "master_end_ms": piece["master_end_ms"],
                    "source": absent(
                        "no ADDED line matches this master piece, exactly or by "
                        "rounding. Expected on artefacts whose format carries no "
                        "ADDED lines at all; ON AN ARTEFACT THAT HAS THEM THIS "
                        "IS A FINDING, not an explanation"),
                    "offset": absent(
                        "this master piece has no matching ADDED line, so this "
                        "reader cannot say whether it read the candidate at all. "
                        "A FILLED region reads no candidate and would carry "
                        "`n/a`; THIS cell is not that -- it is the offset "
                        "question left unanswered because the region's source "
                        "line is missing"),
                })
                continue
            # `from=master/<lang>` carries a suffix: test the prefix.
            is_silence = str(region["from"] or "").startswith("silence")
            width = piece["master_end_ms"] - piece["master_start_ms"]
            rows.append({
                "fill_width": (
                    "equals the locator's UNREFINED SEARCH BOUND"
                    if width == SEARCH_BOUND_MS else
                    "equals the refine floor: the bracket WAS narrowed"
                    if width == REFINE_FLOOR_MS else None),
                "kind": "SILENCE" if is_silence else "MASTER",
                "master_start_ms": piece["master_start_ms"],
                "master_end_ms": piece["master_end_ms"],
                "source": Cell(region["from"], PRESENT),
                # A filled region has no offset ("n/a", not missing).
                "offset": Cell("n/a", PRESENT, "a filled region reads no candidate"),
                # `why=` (head_gap / interior_bracket / tail_gap) as emitted,
                # with the position derived here beside it.
                "matched_by": matched_by,
                "why": (region.get("why") or
                        absent("no `why=` on the ADDED line for this region")),
                # Older logs carry an incorrect `unreported(...)` fallback; annotate it.
                "why_note": (
                    "THIS FALLBACK IS KNOWN FALSE. It says the assembly predates "
                    "the field; measured cause: the value was attached to the "
                    "plan pieces while the ADDED line reads filled_regions -- two "
                    "objects. Never true at any occurrence (48/48 on the "
                    "producer's bytes, 19/19 on mine). Fixed by its producer in "
                    "`9a9d164` (the crossing) and `85a2614` (the sentence -- "
                    "the string now names no cause, because an old assembly and "
                    "an un-annotated region are the same absence from inside the "
                    "emitter); artefacts built before those carry this string"
                    if str(region.get("why") or "").startswith("unreported(")
                    else None),
                "derivation_agreement_proves": (
                    "that the emitted `ADDED` line and the emitted `pieces=` "
                    "geometry agree with each other -- NOT that either "
                    "classification is right. The producer labels by position "
                    "and so does this reader: SAME RULE, so agreement here is "
                    "largely definitional and a divergence would be a fact "
                    "about the log's internal consistency"),
                "position_in_plan_derived_by_this_reader": (
                    "head" if piece is plan["pieces"][0] else
                    "tail" if piece is plan["pieces"][-1] else "interior"),
            })
            continue

        offset = (offsets[candidate_index] if candidate_index < len(offsets)
                  else Cell(None, NO_PRODUCER, "no offset recoverable"))
        read = emitted.get(key[0])
        if read is not None:
            note = None
            if offset.state == DERIVED and str(offset.value) != _trim(read):
                # Keep the emitted value and note the derived one.
                note = (f"disagrees with the value derived from the CUT bounds "
                        f"({offset.value}); the emitted value is shown")
            offset = Cell(_trim(read), PRESENT, note)
        candidate_index += 1
        rows.append({
            "kind": "CANDIDATE",
            "master_start_ms": piece["master_start_ms"],
            "master_end_ms": piece["master_end_ms"],
            # Without a USED line, only the plan's `c<a>-<b>` token covers these regions.
            "source": (Cell("candidate", PRESENT) if emitted
                       else no_producer("no line type covers a kept candidate "
                                        "region; only the `c<a>-<b>` token on "
                                        "the plan line")),
            "offset": offset,
        })

    for cut in job["regions_cut"].get(stream_order) or []:
        rows.append({
            "kind": "LOST",
            "candidate_start_ms": cut["candidate_start_ms"],
            "candidate_end_ms": cut["candidate_end_ms"],
            "dropped_ms": cut["dropped_ms"],
            "dropped_unmeasured": cut["dropped_unmeasured"],
            "where": cut["where"],
        })
    return rows


# Redaction. With REDACT_MEDIA_NAMES on, every value passes through `redact`
# and the finished document is re-checked; a leak raises.

# At least two segments, so `</title>`, `verified=1/1` and `master/ja` are not paths.
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s'\"\]<>/]+){2,}/?")
_CATALOGUE = re.compile(r"[\[{（(]?\b(?:tvdb|tmdb|imdb|tvdbid|anidb)[-_ ]?\d+\b[\]})）]?",
                        re.IGNORECASE)
_MEDIA = re.compile(r"[^\s/\\]+\.(?:mkv|mp4|avi|m4v|ts|mka|mks|srt|ass|sub|idx)\b",
                    re.IGNORECASE)


def redact(text):
    """Return (redacted text, number of replacements).

    Absolute paths, catalogue ids and media filenames become stable opaque
    tokens, so repeated mentions stay correlatable.
    """
    if text is None:
        return None, 0
    working, hits = str(text), 0

    def token(match):
        nonlocal hits
        found = match.group(0)
        if len(found) < 3 or found.strip("/") == "":
            return found
        hits += 1
        return "<redacted:" + hashlib.md5(found.encode("utf-8", "replace"))\
                                      .hexdigest()[:8] + ">"

    working = _PATH.sub(token, working)
    working = _CATALOGUE.sub(token, working)
    working = _MEDIA.sub(token, working)
    return working, hits


class _Redactor:
    """Callable redactor that counts examined values and replacements for one render."""

    def __init__(self):
        self.hits = 0
        # Distinguishes "nothing matched" from "the control did not run".
        self.examined = 0

    def __call__(self, text):
        if not REDACT_MEDIA_NAMES:
            return text
        self.examined += 1
        clean, hits = redact(text)
        self.hits += hits
        return clean


def assert_no_leak(document):
    """Raise AssertionError if the finished document still carries a path or media name.

    A no-op when REDACT_MEDIA_NAMES is False. It never corrects the document.
    """
    if not REDACT_MEDIA_NAMES:
        return
    """DERNIER CONTROLE, sur le document fini. Il leve; il ne corrige pas.

    Un correctif silencieux ici rendrait la fuite suivante invisible. On
    prefere un plantage bruyant a un artefact qui voyage avec un titre dedans.
    """
    for pattern, label in ((_PATH, "an absolute path"),
                           (_CATALOGUE, "a catalogue id"),
                           (_MEDIA, "a media filename")):
        for match in pattern.finditer(document):
            found = match.group(0)
            if found.startswith("<redacted:") or len(found.strip("/")) < 3:
                continue
            if found.startswith("/") or pattern is not _PATH:
                raise AssertionError(
                    f"merge_plan report would have carried {label} "
                    f"({found[:40]!r}). Refusing to emit: this artefact travels "
                    f"and the log it is built from does not.")


# Rendering. The text rows (`KIND key=value ...`, resolved by name) carry every
# number; the HTML drawing is rendered from them. The page is self-contained
# (no remote font, image, script or library).

_STYLE = """
:root{--ink:#16181d;--paper:#fbfaf7;--rule:#c9c4b8;--faint:#6b6558;
--candidate:#2f6f9f;--master:#b8873a;--silence:#5a5a5a;--lost:#c0392b;--grid:#e4e0d6}
body{background:var(--paper);color:var(--ink);
font-family:ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;
font-size:13px;line-height:1.5;margin:0;padding:24px}
h1,h2{font-size:14px;font-weight:700;margin:26px 0 8px;letter-spacing:.04em}
h1{font-size:16px;margin-top:0}
pre{white-space:pre;overflow-x:auto;background:transparent;margin:0;
border-left:2px solid var(--rule);padding:6px 0 6px 12px}
.narrative{max-width:78ch;font-family:ui-sans-serif,"DejaVu Sans",system-ui,sans-serif;
font-size:14px}
.narrative p{margin:.5em 0}
.legend span{margin-right:14px;white-space:nowrap}
.sw{display:inline-block;width:11px;height:11px;vertical-align:-1px;margin-right:4px}
.diagram{overflow-x:auto}
.note{color:var(--faint)}
h2{font-size:15px;border-bottom:1px solid var(--rule);padding-bottom:4px}
h3{font-size:13px;font-weight:700;margin:20px 0 6px}
.schema{max-width:1100px;display:block}
.sw.dashed{border:1.5px dashed var(--lost);width:9px;height:9px}
.sw.cut{width:1px;height:12px;background:var(--ink)}
.badge{display:inline-block;background:var(--master);color:#fff;padding:2px 8px;
border-radius:3px;font-weight:700}
.resume{font-family:ui-sans-serif,"DejaVu Sans",system-ui,sans-serif;font-size:14px;
max-width:110ch;padding-left:18px}
.resume li{margin:.3em 0}
.resume li.table{list-style:none;margin-left:-18px}
table.cuts{border-collapse:collapse;margin:4px 0;font-variant-numeric:tabular-nums}
table.cuts th,table.cuts td{border-bottom:1px solid var(--grid);padding:3px 10px;
text-align:left;white-space:nowrap}
table.cuts th{color:var(--faint);font-weight:600}
"""


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def clock(ms):
    """Format milliseconds as `h:mm:ss.mmm`."""
    if ms is None:
        return "?"
    total = Decimal(str(ms))
    negative = total < 0
    total = abs(total)
    seconds, millis = divmod(int(total), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{'-' if negative else ''}{hours:d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _row(record_kind, /, _redactor=None, **fields):
    """Render one record as a single `KIND key=value ...` line.

    A Cell that is not present/derived is written as `<key>_state=...` plus
    `<key>_because=...` when it has a note, never as an empty value.
    """
    parts = [record_kind]
    for name, value in fields.items():
        if isinstance(value, Cell):
            if value.state in (PRESENT, DERIVED):
                split = _split_head(value.value)
                if split:
                    parts.append(f"{name}={split[0]}")
                    parts.append(f"{name}_detail="
                                 f"{_redactor(_wrap(split[1])) if _redactor else _maybe_redact(_wrap(split[1]))}")
                    if value.state == DERIVED:
                        parts.append(f"{name}_state=derived")
                    if value.note:
                        parts.append(
                            f"{name}_because="
                            f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
                    continue
                cleaned = _wrap(value.value)
                parts.append(f"{name}={_redactor(cleaned) if _redactor else _maybe_redact(cleaned)}")
                if value.state == DERIVED:
                    parts.append(f"{name}_state=derived")
                if value.note:
                    parts.append(
                        f"{name}_because="
                        f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
            else:
                parts.append(f"{name}_state={value.state}")
                if value.note:
                    parts.append(
                        f"{name}_because="
                        f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
            continue
        if value is None:
            continue
        text = _wrap(value)
        # `head(reason)` values: the head stays the value, the reason goes to `<key>_detail`.
        head = _split_head(value)
        if head:
            parts.append(f"{name}={head[0]}")
            detail = head[1]
            parts.append(f"{name}_detail="
                         f"{_redactor(_wrap(detail)) if _redactor else _maybe_redact(_wrap(detail))}")
            continue
        parts.append(f"{name}={_redactor(text) if _redactor else _maybe_redact(text)}")
    return " ".join(parts)


def _split_head(value):
    """Split `head(prose)` into `("head", "prose")`; return None otherwise.

    The greedy match up to the last parenthesis handles nested parentheses,
    and any head word is accepted.
    """
    if value is None:
        return None
    found = re.match(r"^([A-Za-z_]+)\((.*)\)$", str(value).strip())
    return (found.group(1), found.group(2)) if found else None


def _basename(path):
    """Return the last path component, or None."""
    if not path:
        return None
    return str(path).rstrip("/").rsplit("/", 1)[-1] or None


def _maybe_redact(text):
    return redact(text)[0] if REDACT_MEDIA_NAMES else text


def _wrap(value):
    """Wrap a value containing spaces or `=` in square brackets.

    Inner brackets become parentheses; `split_fields` ignores `=` inside
    brackets, so the value creates no phantom key.
    """
    text = str(value)
    if " " not in text and "=" not in text:
        return text
    if text.startswith("[") and text.endswith("]"):
        return text
    return "[" + text.replace("[", "(").replace("]", ")") + "]"


def plain(value):
    """Strip the outer square brackets added by `_wrap`, if any."""
    if value is None:
        return None
    text = str(value)
    if len(text) > 1 and text.startswith("[") and text.endswith("]"):
        return text[1:-1]
    return text


def applied_ratio(track_fields):
    """Return the applied speed ratio from `speed=` as a Decimal.

    Returns None when no ratio was proposed (`None` or `none(...)`) and
    "UNREADABLE" when the value does not parse.
    """
    value = plain(track_fields.get("speed"))
    if value is None:
        return None
    if value.lower().startswith("none"):
        return None
    try:
        return Decimal(value)
    except Exception:
        return "UNREADABLE"


def lost_reference_frame(ratio):
    """Describe the timeline `dropped_ms` is expressed in.

    CUT bounds are read on the already resampled candidate, which advances at
    the master's rate, so `dropped_ms` draws 1:1 on the master axis; the
    original candidate material discarded is `dropped_ms / ratio`.
    """
    if ratio in (None, "UNREADABLE"):
        return "resampled-candidate (advances at master rate; 1:1 with the axis)"
    return (f"resampled-candidate (advances at master rate; 1:1 with the axis) "
            f"ratio={ratio}")


def seconds_fr(ms, decimals, signed=True):
    """Format milliseconds as seconds quantised to `decimals` places.

    With `signed=True` the sign is inverted: a positive `offset_ms` (candidate
    read later than the master) is displayed as a negative delay.
    """
    if ms is None:
        return None
    value = Decimal(str(ms)) / Decimal("1000")
    if signed:
        value = -value
    quantised = value.quantize(Decimal("1." + "0" * decimals))
    return f"{quantised}"


def borrow_provenance(job, stream_order):
    """Infer which track a BORROWED track took its offsets from.

    The reference is the single track in the plan's `language=`; the
    inference is checked by comparing per-region offsets. Returns None when
    the plan names no language or this track is the reference.
    """
    plan = job.get("plan") or {}
    language = plan.get("language")
    if not language:
        return None
    reference = [order for order, fields in (job.get("audios") or {}).items()
                 if fields.get("lang") == language]
    if len(reference) != 1:
        # Undecidable; list the track codes since a zero may come from a
        # language-code mismatch (`fr` vs `fre`).
        codes = sorted({(f or {}).get("lang") for f in (job.get("audios") or {}).values()
                        if (f or {}).get("lang")})
        return {"language": language, "track": None,
                "agreement": f"undecidable: {len(reference)} tracks carry the "
                             f"plan language"
                             + ("" if reference else
                                f". THE PLAN'S CODE `{language}` IS NOT AMONG "
                                f"THE TRACK CODES {codes} -- if those look like "
                                f"the same languages in a different code set "
                                f"(ISO-639-1 against ISO-639-2, `fr` against "
                                f"`fre`), this zero is a fact about THE JOIN and "
                                f"not about the file, and a string-equality "
                                f"match would return zero rows on every file "
                                f"without ever erroring")}
    other = reference[0]
    if other == stream_order:
        return None
    mine = [r["offset_ms"] for r in (job.get("regions_used", {}).get(stream_order) or [])]
    theirs = [r["offset_ms"] for r in (job.get("regions_used", {}).get(other) or [])]
    if not mine or not theirs:
        mine = [c.value for c in recover_offsets(job, stream_order) if c.state == DERIVED]
        theirs = [c.value for c in recover_offsets(job, other) if c.state == DERIVED]
    if not mine or not theirs or len(mine) != len(theirs):
        agreement = "unverified: the two tracks do not expose comparable offsets"
    elif all(str(a) == str(b) for a, b in zip(mine, theirs)):
        agreement = f"offsets identical at {len(mine)} of {len(mine)} regions"
    else:
        agreement = (f"DISAGREES: {sum(1 for a, b in zip(mine, theirs) if str(a) != str(b))} "
                     f"of {len(mine)} regions differ from the reference track")
    return {"language": language, "track": other, "agreement": agreement}


def _speed_margin_cell(plan):
    """Return the `speed_margin` Cell of a plan.

    When a single hypothesis clears the fidelity gate the margin is undefined
    (not absent); the decision is then in `fidelity_margin`.
    """
    value = plain(plan.get("speed_margin"))
    if not value:
        # No key: no margin, or a line older than the field; indistinguishable here.
        return Cell(None, NO_PRODUCER,
                    "the plan line carries no `speed_margin` key. THIS READER "
                    "CANNOT DISTINGUISH two cases: the plan genuinely had no "
                    "margin, or this line predates the field. The producer "
                    "appends nothing in either case. `repair: build` would date "
                    "it, but it is present on only 12 of the 28 job logs here "
                    "and is a content digest with no ordering")
    if value.lower().startswith("absent("):
        return _plan_absent_cell(value, "speed_margin")
    return Cell(value, PRESENT)


def _plan_absent_cell(value, name):
    """Map `absent(<reason>)` on a plan field to a Cell.

    `absent(not_in_plan)` gives ABSENT_FORMAT; any other reason gives
    NOT_DEFINED pointing to `decided_by`.
    """
    reason = value[len("absent("):].rstrip(")")
    if reason.strip().lower() == "not_in_plan":
        return Cell(None, ABSENT_FORMAT,
                    f"the producer states this plan did not carry "
                    f"`{name}`. THIS IS THE PRODUCER SPEAKING, not this reader "
                    f"inferring: a line with no token at all leaves the same "
                    f"blank ambiguous")
    return Cell(None, NOT_DEFINED, reason + " -- see decided_by")


def _plan_field_cell(plan, name):
    """Return a plan field as a Cell, handling `absent(<reason>)`; None if missing."""
    value = plain(plan.get(name))
    if not value:
        return None
    if value.lower().startswith("absent("):
        return _plan_absent_cell(value, name)
    return Cell(value, PRESENT)


def plan_end_ms(job):
    """Return the master end of the plan's last piece, or None."""
    pieces = (job.get("plan") or {}).get("pieces") or []
    return pieces[-1]["master_end_ms"] if pieces else None


def build_rows(job, artefact_id, source_name, n_caveat, corpus=None):
    """Build the report's text rows, one record per line, carrying every number."""
    generation, description = format_generation(job)
    plan = job.get("plan") or {}
    redactor = _Redactor()
    rows = []

    rows.append(_row("MERGE_PLAN", schema="1", produced_by="merge_plan_report",
                     reads="repair-lines-of-one-job-log"))
    rows.append("# Every record below is one line: KIND key=value ... . Resolve BY NAME;")
    rows.append("# there are no columns and no positions. A value that is not present")
    rows.append("# carries <key>_state instead of <key>, with one of:")
    rows.append("#   derived                 computed from other emitted fields")
    rows.append("#   collapsed               emitted, but aggregated above this granularity")
    rows.append("#   absent-from-this-format this artefact predates the field")
    rows.append("#   no-producer             nothing emits it; this is the defect")
    rows.append("#   not-exercised-here      the branch is live and this artefact does not")
    rows.append("#                           produce its input -- NOT a negative answer.")
    rows.append("#                           ALONE AMONG THESE STATES IT IS A PROPERTY OF")
    rows.append("#                           THE DATA, NOT OF THE CODE: a new corpus file")
    rows.append("#                           turns it into `present` with nobody touching")
    rows.append("#                           producer or consumer. It is therefore scoped")
    rows.append("#                           to ONE artefact and it CANNOT BE DATED from")
    rows.append("#                           this record -- no build, commit or timestamp")
    rows.append("#                           field is emitted anywhere (see the")
    rows.append("#                           build_identity GAP row). Read it as of the")
    rows.append("#                           artefact named on the SOURCE row, never as a")
    rows.append("#                           standing fact about the pipeline.")
    rows.append("#   not-defined-here        the quantity does not exist in this case;")
    rows.append("#                           the decision was made elsewhere, see decided_by")
    if corpus:
        rows.append(_row("CORPUS", _redactor=redactor, **dict(corpus)))
    else:
        rows.append(_row("CORPUS", _redactor=redactor, state=NOT_SUPPLIED,
                         note="no population was supplied to this render, so "
                              "every corpus-scale claim below is undenominated. "
                              "IN A MULTI-ARTEFACT RENDER, call measure_corpus() "
                              "and pass the result. IN PRODUCTION THIS IS THE "
                              "CORRECT OUTPUT AND NOT A DEFECT: a single merge "
                              "has no population, and inventing one of 1 would "
                              "present a denominator that does not exist"))
    rows.append(_row("SOURCE", artefact=artefact_id, log=source_name,
                     format_generation=generation, format=description))
    if job.get("sources"):
        rows.append(_row("SOURCES", _redactor=redactor,
                         digest=job["sources"]["digest"],
                         files=job["sources"]["files"],
                         scope=job["sources"]["scope"],
                         manifest=job["sources"]["manifest"],
                         derivation="sha256 of the bytes on disk at call time, "
                                    "read per file",
                         covers="the .py files the image ships, AND NOTHING "
                                "ELSE: not the interpreter, not ffmpeg, not "
                                "mkvtoolnix -- all installed unpinned. Two "
                                "artefacts sharing this digest ran identical "
                                "Python; they did not necessarily run in "
                                "identical containers"))
    if job.get("build"):
        rows.append(_row("BUILD", _redactor=redactor,
                         **dict(job["build"]),
                         note="which version of each module produced this. A "
                              "verdict is a claim about a file AND about the "
                              "build that made it"))
    rows.append(_row("IDENTITY",
                     master=job.get("master_opaque_id") or "",
                     candidate=job.get("candidate_opaque_id") or "",
                     # Basenames only, never full paths.
                     master_name_local_only=(_basename(job.get("master_path"))
                                  if not REDACT_MEDIA_NAMES else None),
                     candidate_name_local_only=(_basename(job.get("candidate_path"))
                                     if not REDACT_MEDIA_NAMES else None),
                     construction="md5(path)[:16]",
                     quote_by=("the opaque ids above, never the `_local_only` "
                               "fields: those must not leave the output "
                               "directory"
                               if not REDACT_MEDIA_NAMES else None),
                     name_fidelity=("square brackets in a name are rendered as "
                                    "parentheses by this row grammar, so a NAME "
                                    "HERE IS NOT BYTE-EXACT -- use the id if you "
                                    "need to match, and the figure's title or "
                                    "the source log if you need the literal name"
                                    if not REDACT_MEDIA_NAMES else None),
                     note=("names carried BESIDE the ids, never instead: an id "
                           "removed is an id nobody can use again"
                           if not REDACT_MEDIA_NAMES else
                           "opaque ids only; no media name travels in this report")))
    if not job.get("master_line_present"):
        rows.append(_row("IDENTITY_LIMIT", master_state=NO_PRODUCER,
                         detail="this_format_does_not_name_the_master;"
                                "_content_correspondence_is_unmeasurable"))
    for caveat in n_caveat:
        rows.append(_row("CAVEAT", text=caveat))

    rows.append(_row("CONVENTION", _redactor=redactor,
                     displayed_offset="the negative of offset_ms, in seconds",
                     reason="a candidate read later than the master must be "
                            "advanced, which the figure writes as a negative delay",
                     decimal_separator="point"))
    rows.append(_row("PLAN", kind=plan.get("kind"), language=plan.get("language"),
                     quantum_ms=plan.get("quantum_ms"),
                     # `absent(<reason>)` means undefined, not missing; only a missing field
                     # renders as not emitted.
                     speed_margin=_speed_margin_cell(plan),
                     fidelity_margin=_plan_field_cell(plan, "fidelity_margin"),
                     decided_by=_plan_field_cell(plan, "decided_by"),
                     master_end_ms=_trim(plan_end_ms(job)),
                     master_end=clock(plan_end_ms(job)),
                     pieces=len(plan.get("pieces") or []),
                     # Pass other plan fields through by name (e.g. `dropped_segments`).
                     dropped_segments_absent_means=(
                         None if "dropped_segments" in plan else
                         "no `dropped_segments` on this plan line. On an "
                         "artefact from a build that emits it, that means ZERO "
                         "segments dropped AND the locator reported so. On an "
                         "older one it means the field did not exist. THIS "
                         "READER CANNOT TELL WHICH: `repair: build` is missing "
                         "from 16 of the 28 job logs here and carries no "
                         "ordering where it is present"),
                     **{name: value for name, value in plan.items()
                        if name not in _PLAN_FIELDS_RENDERED}))
    rows.append(_row("PLAN_GEOMETRY_LIMIT",
                     detail="the_pieces=_token_is_assembly[pieces],_the_one_"
                            "normalize_segments_call_made_with_no_stream_order,"
                            "_i.e._the_geometry_a_BORROWING_track_uses"))
    # Report each extra plan line by field names only (it may end with a path).
    plan_lines = job.get("plan_lines") or []
    for position, other in enumerate(plan_lines, 1):
        if other is plan:
            continue
        raw_names = sorted(name for name, value in other.items()
                           if value is not None
                           and name not in ("kind", "quantum_ms", "pieces"))
        rows.append(_row("PLAN_DUPLICATE", _redactor=redactor,
                         line=f"{position}/{len(plan_lines)}",
                         kind=other.get("kind"),
                         pieces=len(other.get("pieces") or []),
                         field_names=",".join(n for n in raw_names
                                              if re.fullmatch(r"[a-z_]+", n)),
                         carries_path=any(_PATH.search(str(v))
                                          for v in other.values()),
                         status="duplicate plan line -- ignored for geometry, "
                                "emitter to fix (one plan line per run)",
                         drawn_from=f"line {plan_lines.index(plan) + 1}/"
                                    f"{len(plan_lines)} (kind "
                                    f"{plan.get('kind')}, "
                                    f"{len(plan.get('pieces') or [])} pieces)"))

    # One staircase is drawn per geometry; report whether tracks share it.
    if len(job.get("audios") or {}) > 1:
        rows.append(_row(
            "SHARED_GEOMETRY_ASSERTION",
            asserts="this figure draws one staircase per GEOMETRY, not per "
                    "track. IT IS A CORRECT DRAWING OF WHAT THE PIPELINE DID "
                    "AND MUST NOT BE READ AS `THE TRACKS WERE FOUND TO AGREE`",
            answered_from_the_source="NOT unfalsifiable after all, and the "
                 "answer is NO. merge_video_chimeric.py:1611 -- the per-stream "
                 "pairing table covers ONLY the comparison language, so every "
                 "other language BORROWS, silently, with 14 to 32 ms of MEASURED "
                 "error, below the verifier's 100 ms tolerance. The tracks do "
                 "not share a measured geometry: THEY SHARE ONE MEASUREMENT. "
                 "Borrowing is structural because nothing else was ever "
                 "measured, not because two geometries were compared and found "
                 "equal",
            why_no_artefact_can_settle_it="there is no job in which two tracks "
                 "are INDEPENDENTLY measured, so the corpus case this cell "
                 "would need cannot exist under this code. That is the "
                 "finding, not a gap in the corpus -- and it is why "
                 "`0 logs carry more than one repair: plan line` is not a "
                 "logging defect either: ONE PLAN BECAUSE ONE MEASUREMENT",
            head="a correlation check between two tracks declared filled FROM "
                 "THE SAME master stream can confirm the declared fill is "
                 "PRESENT in the produced samples. It is NOT evidence that two "
                 "geometries agree: identical bytes copied to two tracks "
                 "correlate at 1 whatever the geometry",
            interior=Cell(None, NOT_MEASURED,
                          "MEASURED AND NON-DISCRIMINATING -- the divergence is "
                          "not unknown, it is 14 to 32 ms and it is hidden "
                          "under the verifier's tolerance BY DESIGN. "
                          "Measured and non-discriminating is a third outcome, "
                          "not a negative, and this row is printed at m = 0 "
                          "rather than dropped: a dropped m = 0 is how a reader "
                          "stops being able to tell `verified` from `never put "
                          "to the test`"),
            refuted_by_nothing_i_hold=(
                "0 logs carry more than one `repair: plan` line, so this reader "
                "cannot compare two per-track geometries even in principle"
                if len(job.get("plan_lines") or []) < 2 else
                f"THIS log carries {len(job['plan_lines'])} `repair: plan` "
                f"lines (see PLAN_DUPLICATE), of which "
                f"{sum(1 for p in job['plan_lines'] if p.get('pieces'))} carry "
                f"pieces. A plan line is emitted per run, not per track, so a "
                f"second line is a duplicated emission or a second repair call, "
                f"never a second per-track geometry to compare")))
    for order in sorted(job.get("audios") or {}):
        fields = job["audios"][order]
        rows.append(_row("TRACK", track=order, kind="audio",
                         lang=fields.get("lang"),
                         fill=fields.get("fill"),
                         filled_ms=fields.get("filled_ms"),
                         silence_ms=fields.get("silence_ms"),
                         head_pad_ms=fields.get("head_pad_ms"),
                         speed=fields.get("speed"),
                         offset=fields.get("offset"),
                         verify=fields.get("verify"),
                         probes=fields.get("probes"),
                         worst=fields.get("worst"),
                         r_min=fields.get("r_min"),
                         verified=fields.get("verified"),
                         # Pass any other field through by name.
                         **{name: value for name, value in fields.items()
                            if name not in _TRACK_FIELDS_RENDERED}))
        label = plain(fields.get("offset")) or ""
        if label.startswith("BORROWED"):
            borrow = borrow_provenance(job, order)
            if borrow:
                # The BORROWED reason comes from a generic fallback branch; render it as a
                # label, not a finding.
                rows.append(_row("BORROW", _redactor=redactor, track=order,
                                 plan_language=borrow["language"],
                                 from_track=borrow["track"],
                                 attribution="inferred, not stated by the log",
                                 printed_reason_is_not_checked_by_its_writer=(
                                     "the `BORROWED[...]` text on the TRACK row "
                                     "is the producer's and is rendered "
                                     "verbatim, never laundered -- but its "
                                     "final branch is a CATCH-ALL reached when "
                                     "the pairing table has no row for this "
                                     "stream, and nothing on that path inspects "
                                     "what it names. One such reason has been "
                                     "measured FALSE against the file: it said "
                                     "the master carried no stream in that "
                                     "language and the master carried one, at "
                                     "the index and codec the `fill=` field "
                                     "names. READ IT AS A LABEL, NOT AS A "
                                     "FINDING"),
                                 check=borrow["agreement"]))

        audios = job.get("audios") or {}
        pads = {order: (f or {}).get("head_pad_ms") for order, f in audios.items()}
        distinct = {str(v) for v in pads.values() if v is not None}
        plan_language = (job.get("plan") or {}).get("language")
        # Emit PLAN_ATTRIBUTION_LIMIT when the plan has no `language=` (attribution
        # undecidable) or when `head_pad_ms` differs across tracks (shared geometry wrong).
        if len(audios) > 1 and (not plan_language or len(distinct) > 1):
            attribution = borrow_provenance(job, order)
            own = [o for o, f in audios.items()
                   if plan_language and (f or {}).get("lang") == plan_language]
            if attribution is None and own == [order]:
                attribution = {"track": "THIS track: it carries the plan's own "
                                        "measurement language"}
            rows.append(_row("PLAN_ATTRIBUTION_LIMIT", track=order,
                             plan_lines_read=len(job.get("plan_lines") or []),
                             audio_tracks=len(audios),
                             plan_language=(plan_language or
                                            absent("this plan line carries no "
                                                   "`language=` key")),
                             reference_track=(
                                 (attribution or {}).get("track")
                                 if plan_language and attribution else
                                 absent("undecidable: no track can be "
                                        "identified as the one the plan was "
                                        "measured on")),
                             head_pad_ms_distinct_values=len(distinct),
                             head_pad_ms_seen=" ".join(sorted(distinct)) or None,
                             reads_as=("THIS ROW'S GEOMETRY IS DRAWN FROM THE "
                                       "ONE PLAN IN THIS LOG AND WAS NOT "
                                       "MEASURED FOR THIS TRACK"
                                       if len(distinct) > 1 else
                                       "drawn from the one plan in this log; "
                                       "no per-track geometry is emitted"),
                             evidence=("the tracks are NOT identical at the "
                                       "head: head_pad_ms takes "
                                       f"{len(distinct)} different values "
                                       "across them, so they cannot share the "
                                       "head geometry drawn for all of them"
                                       if len(distinct) > 1 else None),
                             state=NOT_MEASURED))

        # The log has no master durations, so a fill shortfall cannot be attributed
        # to the repair or to a short master.
        shortfall = re.search(r"FILL SOURCE SHORT BY ([\d.-]+) ms; "
                              r"TRACK LOST ([\d.-]+) ms; UNEXPLAINED ([\d.-]+) ms",
                              plain(fields.get("fill")) or "")
        if shortfall:
            rows.append(_row("SHORTFALL", _redactor=redactor, track=order,
                             fill_source_short_by_ms=shortfall.group(1),
                             track_lost_ms=shortfall.group(2),
                             unexplained_ms=shortfall.group(3),
                             attribution=NOT_MEASURED,
                             detail="a shortfall MEASURED here is not a "
                                    "shortfall ATTRIBUTED. This log names the "
                                    "master and carries none of its durations, "
                                    "so nothing here can say whether the repair "
                                    "lost this or inherited it. Fidelity to a "
                                    "source is not correctness of an output. "
                                    "THE NARROW FIX: the `repair: master` line "
                                    "names the master and emits none of its "
                                    "per-track durations. Three numbers on "
                                    "that line and this cell becomes "
                                    "measurable at render time",
                             # Context not read from the log; carries its source.
                             count_source="this report's own prose, message layer "
                                          "only: ZERO occurrences in MEASURING.MD, "
                                          "WRITE_ZONES.MD or AGENT.MD (checked). "
                                          "This report is currently its most "
                                          "durable carrier and there is no "
                                          "citation to follow back",
                             count_direction="can only be invalidated, never "
                                             "confirmed further: one introduced "
                                             "defect ends it, and a number that "
                                             "flatters is not re-checked",
                             count_dated=NOT_MEASURED))

        rate = applied_ratio(fields)
        speed_text = plain(fields.get("speed"))
        if isinstance(rate, Decimal):
            rows.append(_row("SPEED", _redactor=redactor, track=order,
                             ratio_applied=str(rate),
                             margin_state=NO_PRODUCER,
                             note="the winning transform's margin has no "
                                  "producer, so how decisively it won is unknown"))
        else:
            rows.append(_row("SPEED", _redactor=redactor, track=order,
                             ratio_applied_state=NO_PRODUCER,
                             rendering_of_an_accelerated_track=NOT_EXERCISED,
                             emitted=speed_text,
                             note="NOBODY ASKED. `no rate proposed by the "
                                  "measurement` is not a verdict of `no rate "
                                  "problem` -- no producer of a speed verdict "
                                  "exists in src/, so this track was never "
                                  "assessed for one"))

        # `head_pad=` states: none / padded / read_past / unreported; a missing field
        # is its own state, not zero.
        head_pad = fields.get("head_pad")
        if head_pad:
            counts = {}
            for piece in plain(head_pad).split(","):
                name, _, value = piece.partition("=")
                if name.strip():
                    counts[name.strip()] = value.strip()
            rows.append(_row("HEAD", _redactor=redactor, track=order,
                             head_pad_ms=fields.get("head_pad_ms"),
                             none=counts.get("none"), padded=counts.get("padded"),
                             read_past=counts.get("read_past"),
                             unreported=counts.get("unreported"),
                             note="read_past is a real container offset the plan "
                                  "reads over: it costs no padding and used to "
                                  "print the same zero as no offset at all"))
        else:
            rows.append(_row("HEAD", _redactor=redactor, track=order,
                             head_pad_ms=fields.get("head_pad_ms"),
                             breakdown_state=ABSENT_FORMAT,
                             note="this assembly predates the per-piece head "
                                  "breakdown, so head_pad_ms=0 here still covers "
                                  "three states with one digit"))

        # One label per track; flag when regions carry different offsets.
        rows.append(_row("TRACK_LABEL_LIMIT", track=order,
                         offset_label=fields.get("offset"),
                         offset_fidelity_state=COLLAPSED,
                         detail="offset_measured_is_all()_over_segments_and_"
                                "offset_fidelity_is_the_first_non-None;_both_"
                                "are_per-TRACK_over_per-REGION_values"))

        candidate_rows = []
        for index, region in enumerate(track_regions(job, order)):
            if region["kind"] == "CANDIDATE":
                candidate_rows.append(region)
            if region["kind"] == "LOST":
                ratio = applied_ratio(fields)
                original = None
                if isinstance(ratio, Decimal) and ratio and not region["dropped_unmeasured"] \
                        and region["dropped_ms"] is not None:
                    # The same loss in original candidate material (`dropped_ms / ratio`).
                    original = _trim((region["dropped_ms"] / ratio).quantize(Decimal("0.01")))
                rows.append(_row("LOST", track=order, seq=index,
                                 timeline=lost_reference_frame(ratio),
                                 original_candidate_ms=original,
                                 candidate_start_ms=_trim(region["candidate_start_ms"]),
                                 candidate_end_ms=_trim(region["candidate_end_ms"]),
                                 candidate_start=clock(region["candidate_start_ms"]),
                                 candidate_end=clock(region["candidate_end_ms"]),
                                 dropped_ms=("UNMEASURED" if region["dropped_unmeasured"]
                                             else _trim(region["dropped_ms"])),
                                 dropped_s=(None if region["dropped_unmeasured"]
                                            else seconds_fr(region["dropped_ms"], 1)),
                                 where=region["where"]))
                continue
            rows.append(_row("REGION", track=order, seq=index, kind=region["kind"],
                             master_start_ms=_trim(region["master_start_ms"]),
                             master_end_ms=_trim(region["master_end_ms"]),
                             master_start=clock(region["master_start_ms"]),
                             master_end=clock(region["master_end_ms"]),
                             source=region["source"], offset_ms=region["offset"],
                             fill_width=region.get("fill_width"),
                             # The duration the figure draws, also present in the rows.
                             duration=short_clock(region["master_end_ms"]
                                                  - region["master_start_ms"]),
                             offset_s=(seconds_fr(region["offset"].value, 3)
                                       if region["offset"].state in (PRESENT, DERIVED)
                                       and region["offset"].value not in (None, "n/a")
                                       else None),
                             # Pass other region fields through by name.
                             **{name: value for name, value in region.items()
                                if name not in _REGION_FIELDS_RENDERED}))

        # Offset step between consecutive candidate regions (the number drawn on
        # each cut rule).
        steps = []
        for previous, following in zip(candidate_rows, candidate_rows[1:]):
            if not (previous["offset"].state in (PRESENT, DERIVED)
                    and following["offset"].state in (PRESENT, DERIVED)):
                continue
            try:
                delta = Decimal(str(following["offset"].value)) - \
                        Decimal(str(previous["offset"].value))
            except Exception:
                continue
            steps.append((previous["master_end_ms"], delta))
            rows.append(_row("STEP", _redactor=redactor, track=order,
                             at_master_ms=_trim(previous["master_end_ms"]),
                             at_master=clock(previous["master_end_ms"]),
                             step_ms=_trim(delta), step_s=seconds_fr(delta, 1)))

    for fields in job.get("subtitles") or []:
        rows.append(_row("SUBTITLE", track=fields.get("stream_order"),
                         lang=fields.get("lang"), format=fields.get("format"),
                         kept_cues=fields.get("kept_cues"),
                         dropped_cues=fields.get("dropped_cues"),
                         shifts_ms=fields.get("shifts_ms"),
                         **{name: value for name, value in fields.items()
                            if name not in _SUBTITLE_FIELDS_RENDERED}))

    # A refused-segment mark that is never drawn must still be reported as
    # not exercised.
    if not (job.get("refused") or []):
        # `drop_unverified_segments` drops segments shorter than the 60 s probe
        # window; report this plan's shortest candidate segment against it.
        shortest = None
        for piece in plan.get("pieces") or []:
            if piece["source"] != "candidate":
                continue
            length = piece["master_end_ms"] - piece["master_start_ms"]
            if shortest is None or length < shortest:
                shortest = length
        rows.append(_row("REFUSED_NONE", _redactor=redactor,
                         count=0, state=NOT_EXERCISED,
                         producer="merge_video_repair.drop_unverified_segments "
                                  "then log_assembly `repair: SKIPPED segment`",
                         precondition="a plan segment SHORTER than the "
                                      "measurement's probe window "
                                      "(PROBE_WINDOW_SECONDS=60s)",
                         shortest_candidate_segment_ms=_trim(shortest),
                         note="this artefact refuses no candidate segment, so "
                              "the figure draws no dashed amber box. NOT a claim "
                              "that nothing is ever refused. The mark has never "
                              "been drawn against real refused material"))
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="segment-level: drop_unverified_segments never "
                                "fired", applies="YES on this artefact",
                         bound=f"measurable -- probe window 60 s, shortest "
                               f"candidate segment here "
                               f"{_trim(shortest) if shortest else '?'} ms"))
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="per-track: declined/failed needs SOME tracks to "
                                "succeed while OTHERS fail on one file",
                         applies=("YES on this artefact"
                                  if (job.get("audios") or {}) else "unknown"),
                         bound="not measurable from this artefact alone -- the "
                               "population most likely to produce mixed "
                               "success is a file where speed-mismatch causes "
                               "some tracks to succeed while others on the "
                               "same file do not; whether any real case has "
                               "ever produced a per-track SKIPPED is a census "
                               "question, not a per-job one"))
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="no tracks were attempted: the repair died "
                                "before any track was built",
                         applies="CANNOT APPEAR HERE",
                         bound="such a run emits no `repair: plan` line, so this "
                               "reader rejects it by structure and it is absent "
                               "from every count in this report -- see "
                               "rejected_by_structure on the CORPUS row"))

    if job.get("failed"):
        rows.append(_row("FAILED", _redactor=redactor, file_produced="NO",
                         reason=job["failed"],
                         note="a TOOL FAULT escaped before any verdict existed. "
                              "Not a decision about the media: nobody decided"))
    if job.get("declined"):
        # Declined: no file was produced, so the verdict comes first.
        rows.append(_row("DECLINED", _redactor=redactor,
                         file_produced="NO",
                         reason=job["declined"],
                         note="this report describes a repair that was REFUSED. "
                              "No file was produced. Everything below describes "
                              "what the plan WOULD have done, not what any "
                              "artefact contains"))
    if job.get("plan") and not job.get("output_check"):
        rows.append(_row("NO_OUTPUT_CHECK", _redactor=redactor,
                         reason=("the repair was DECLINED: the `output file` "
                                 "summary is written past the point where the "
                                 "gate raises, so it does not exist"
                                 if job.get("declined") else
                                 "no `output file` line on this artefact and no "
                                 "decline recorded either -- unexplained"),
                         state=(PRESENT if job.get("declined") else NOT_MEASURED)))
    for entry in job.get("unparsed") or []:
        rows.append(_row("UNPARSED", _redactor=redactor, line=entry,
                         # Lines this reader does not recognise yet.
                         reads_as="an alarm about THIS READER, not a fact about "
                                  "the artefact: the producer emitted a line "
                                  "this parser does not know yet",
                         note="kept rather than dropped, because a reader that "
                              "discards what it does not know discards exactly "
                              "what is new"))

    for entry in job.get("refused") or []:
        # Candidate material refused by the plan (dashed box), distinct from a master fill.
        rows.append(_row("REFUSED", _redactor=redactor,
                         master_start_ms=_trim(entry["master_start_ms"]),
                         master_end_ms=_trim(entry["master_end_ms"]),
                         master_start=clock(entry["master_start_ms"]),
                         master_end=clock(entry["master_end_ms"]),
                         dropped_ms=_trim(entry["dropped_ms"]),
                         reason=entry.get("reason")))

    for entry in job.get("skipped") or []:
        rows.append(_row("SKIPPED", _redactor=redactor,
                         detail=entry.replace(" ", "_")))

    check = job.get("output_check")
    if check:
        rows.append(_row("CHECK", **{k: v for k, v in check.items()}))
        # `audio_tracks` counts rebuilt tracks, not the tracks in the file.
        if check.get("audio_tracks"):
            rows.append(_row("CHECK_DENOMINATOR_LIMIT",
                             audio_tracks=check.get("audio_tracks"),
                             counts="tracks the repair REBUILT",
                             does_not_count="tracks present in the produced file",
                             measured="a file rendering audio_tracks=1/1 was "
                                      "measured to carry EIGHT audio streams",
                             reads_as="an n/n here is completeness about the "
                                      "WORK, never about the FILE",
                             state=COLLAPSED))
    # Predicted refusals vs outcome, with the count so a trivial agreement
    # on zero predictions is visible.
    prediction = job.get("prediction")
    if prediction:
        predicted = prediction.get("predicted")
        try:
            opportunities = int(str(predicted).strip())
        except (TypeError, ValueError):
            opportunities = None
        rows.append(_row(
            "PREDICTION",
            **dict(prediction),
            reads="the producer testing ITS OWN duration gate: it announces the "
                  "refusals it expects, then reports whether the outcome agreed",
            of_which_could_have_disagreed=(
                predicted if opportunities else
                "0 -- a prediction of no refusals AGREES with any run that "
                "refuses nothing, which a constant predictor would also score. "
                "UNTESTED on this artefact, not confirmed"
                if opportunities == 0 else
                Cell(NOT_MEASURED, "`predicted` is not a number this reader can "
                                   "read, so it cannot say what the agreement "
                                   "was worth"))))
    for entry in job.get("predicted_refusals") or []:
        rows.append(_row("PREDICTED_REFUSAL", **dict(entry),
                         reads="announced BEFORE the mux, by the producer, "
                               "naming the track and the shortfall that will "
                               "trip the gate"))
    if job.get("summary_counts"):
        rows.append(_row("SUMMARY_COUNTS", text=job["summary_counts"].replace(" ", "_"),
                         note="a_count_is_a_statement_about_work_done,_not_about_a_file"))

    for entry in job.get("brackets") or []:
        # Emitted `bound_only` beside the width-derived signature.
        width = entry.get("width_ms")
        try:
            derived = float(width) == float(SEARCH_BOUND_MS)
        except (TypeError, ValueError):
            derived = None
        rows.append(_row("BRACKET", _redactor=redactor,
                         agrees_with_my_derived_signature=(
                             None if derived is None else
                             str(derived) == str(entry.get("bound_only"))),
                         derived_signature=(
                             None if derived is None else
                             "width == the 100 s search bound" if derived else
                             "width != the search bound"),
                         note="the locator's OWN bracket for this change point. "
                              "`bound_only` is now emitted and this row prints "
                              "it; the derived signature beside it is kept so a "
                              "reader can watch the two agree",
                         **{k: v for k, v in entry.items()}))
    for entry in job.get("segments") or []:
        rows.append(_row("SEGMENT", _redactor=redactor,
                         note="the locator's own segmentation, with the offset "
                              "it measured for each -- per stream where the "
                              "producer gives it per stream",
                         **{k: v for k, v in entry.items()}))
    if job.get("output_durations"):
        rows.append(_row("OUTPUT_DURATIONS", _redactor=redactor,
                         note="measured on the PRODUCED FILE: container length, "
                              "longest a/v stream, and the length expected of "
                              "it. A per-FILE quantity -- still not per-track",
                         **{k: v for k, v in job["output_durations"].items()}))

    for entry in job.get("locator_measurements") or []:
        # All emitted fields, by name.
        rows.append(_row("LOCATOR", _redactor=redactor,
                         note="the locator's own measurement of this pair, "
                              "read by name from the line it emits",
                         # Spell out empty values (`offset_ms=` parses to '').
                         **{k: (v if v != "" else
                                "the producer emitted this key with nothing "
                                "after it")
                            for k, v in entry.items()}))
    for entry in job.get("locator_notes") or []:
        rows.append(_row("LOCATOR_NOTE", _redactor=redactor,
                         digest=entry["digest"], chars=entry["chars"],
                         carries_path=entry["carries_path"],
                         tail=entry["tail"] or None,
                         reads_as="a locator line that is prose and not "
                                  "`key=value` -- a decline, or a probe that "
                                  "failed",
                         note="content withheld: `probe at ...s failed: {error}` "
                              "interpolates an arbitrary ffprobe exception, and "
                              "those carry the input path"))

    for entry in job.get("foreign_lines") or []:
        # Only the fact that a foreign line exists, never its text.
        rows.append(_row("OUTSIDE_REPAIR", _redactor=redactor,
                         digest=entry["digest"], chars=entry["chars"],
                         carries_path=entry["carries_path"],
                         tail=entry["tail"] or None,
                         note="content withheld: free text outside the repair "
                              "vocabulary cannot be redacted with a guarantee"))

    # Self-controls report whether they had a chance to fire (counts are per render).
    for gap in blank_cells(job, corpus):
        # NOT_EXERCISED depends on the input file; scope it to this artefact.
        rows.append(_row("GAP", _redactor=redactor,
                         quantity=gap["quantity"], state=gap["state"],
                         addressed_to=gap["address"], detail=gap["detail"],
                         **({"observed_on": artefact_id,
                             "scope": "this artefact only; a property of the "
                                      "corpus and not of the code, and undatable "
                                      "from this record"}
                            if gap["state"] == NOT_EXERCISED else {})))
    # Written last so the redaction count covers every row.
    rows.append(_row("CONTROL", _redactor=redactor, name="redaction",
                     fired_on_this_render=redactor.hits,
                     # With redaction off, nothing is examined; `could_have_fired` says so.
                     values_examined=redactor.examined,
                     could_have_fired=(
                         redactor.examined if redactor.examined else
                         "ZERO. REDACT_MEDIA_NAMES is False on this build -- "
                         "media names are permitted in this report, so this "
                         "control DID NOT RUN. The zero above is the "
                         "instrument being off, not a clean result"),
                     checks="absolute paths, catalogue ids and media filenames "
                            "in emitted values, replaced by a stable opaque token",
                     limit="pattern-based, and a pattern cannot survive text it "
                           "does not own: a path containing spaces let a show "
                           "title through, measured. The real control is that "
                           "free text outside the repair vocabulary is never "
                           "reproduced at all"))
    rows.append(_row("CONTROL", _redactor=redactor, name="leak_assertion",
                     fired_on_this_render=0,
                     could_have_fired="the whole finished document, re-read",
                     checks="the finished document is re-read and the render "
                            "RAISES rather than corrects if anything survives",
                     limit="it raises, so a zero here is the only outcome a "
                           "reader can ever see -- this line records that the "
                           "control ran, not that there was nothing to catch"))
    rows.append(_row("CONTROL", _redactor=redactor, name="corpus_sanity",
                     fired_on_this_render=0 if corpus else None,
                     could_have_fired=("1 comparison: n_distinct against n"
                                       if corpus else
                                       "0 -- NO POPULATION, so this control had "
                                       "no opportunity to fire at all"),
                     state=None if corpus else NOT_SUPPLIED,
                     checks="a distinct-count can never exceed its population",
                     limit="it passes on a WRONG count as readily as a right "
                           "one: 15 and 16 both satisfy it, and 15 was the "
                           "wrong unit. NEVER FIRED"))

    return rows


# Plan fields rendered explicitly or consumed elsewhere; any other field is
# printed as-is.
_PLAN_FIELDS_RENDERED = frozenset((
    "kind", "language", "quantum", "quantum_ms", "speed_margin",
    "speed_margin_absent_reason", "fidelity_margin", "decided_by", "pieces",
))

_REGION_FIELDS_RENDERED = frozenset((
    "kind", "master_start_ms", "master_end_ms", "source", "offset",
    "fill_width", "from", "candidate_start_ms", "candidate_end_ms",
    "dropped_ms", "dropped_unmeasured", "where",
))

_TRACK_FIELDS_RENDERED = frozenset((
    "lang", "fill", "filled_ms", "silence_ms", "head_pad_ms", "speed", "offset",
    "verify", "probes", "worst", "r_min", "verified",
    # Carried by other rows:
    "residual", "quantum", "head_pad"))

# Subtitle fields rendered explicitly; others are passed through.
# `stream_order` is rendered as `track=`.
_SUBTITLE_FIELDS_RENDERED = frozenset((
    "stream_order", "lang", "format", "kept_cues", "dropped_cues", "shifts_ms",
))


def blank_cells(job, corpus=None):
    """Return the register of required quantities missing from this report.

    Each entry carries its cell state and the producer that would emit it.
    """
    generation, _ = format_generation(job)
    entries = []

    if not job.get("master_line_present"):
        entries.append({
            "quantity": "master_identity",
            "state": ABSENT_FORMAT if generation < 3 else PRESENT,
            "address": "merge_video_repair.log_assembly",
            "detail": "the master is not named, so a produced deficit cannot "
                      "be attributed to the merge or to the master"})

    used = bool(job.get("regions_used") or {})
    if not used:
        entries.append({
            "quantity": "per region offset by name",
            "state": ABSENT_FORMAT if generation >= 2 else NO_PRODUCER,
            "address": "merge_video_repair.log_assembly (USED_line)",
            "detail": "derived here from CUT bounds against plan bounds; "
                      "derivation is conditional on the plan having cut something"})

    entries.append({
        "quantity": "per region offset_fidelity",
        "state": COLLAPSED,
        "address": "merge_video_chimeric.assemble_on_master_timeline",
        "detail": "offset_fidelity is the first non-None across segments and "
                  "offset_measured is all() over them; per-segment values exist "
                  "upstream in candidate offset_fidelity by stream"})
    entries.append({
        "quantity": "kept candidate region provenance",
        "state": NO_PRODUCER if not used else PRESENT,
        "address": "merge_video_repair.log_assembly",
        # State and detail must agree.
        "detail": ("the USED line covers them: master bounds, candidate bounds "
                   "and the offset applied, per track and per region"
                   if used else
                   "no line type covers the regions the output takes from the "
                   "candidate, which are the majority of every file")})
    # Subtitles have no per-region provenance: subtitle repair shifts or drops
    # cues and only emits a per-track summary line.
    entries.append({
        "quantity": "subtitle region provenance",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly (subtitle track line)",
        "detail": "audio gets a REGION row per master-timeline span because "
                  "`log_assembly` emits ADDED/USED/CUT lines keyed to master "
                  "positions; subtitle repair shifts or drops CUES, not "
                  "timeline spans, and emits only the one per-track summary "
                  "line (lang, format, kept_cues, dropped_cues, shifts_ms) "
                  "the SUBTITLE row already carries in full. Which SOURCE "
                  "delivered which cue, or which cue-range came from where, "
                  "is not stated by anything this module emits -- that is a "
                  "gap in the emitter, not something this render can infer "
                  "or fabricate a row for"})
    # PRESENT when the log carries `repair: bracket` lines.
    emitted_brackets = len(job.get("brackets") or [])
    entries.append({
        "quantity": "gap_is_filled_because_unsure",
        "state": PRESENT if emitted_brackets else NO_PRODUCER,
        "address": "change_point_locator -> merge_video_repair "
                   "(repair: bracket line)",
        "detail": (
            f"EMITTED ON THIS ARTEFACT: {emitted_brackets} `repair: bracket` "
            f"line(s) carrying low_ms, high_ms, width_ms, bound_only, step_ms "
            f"and step_points. `bound_only=True` is the locator SAYING it could "
            f"not narrow the bracket, which is the question this report existed "
            f"to answer and could previously only infer. THE INFERENCE WAS "
            f"RIGHT, AND THE COUNT IS MEASURED AT RENDER TIME: "
            + (corpus["bracket_agreement"] if corpus else
               "NO POPULATION SUPPLIED to this render, so this report cannot "
               "say how often the derived signature and the emitted "
               "`bound_only` agree -- it can only show you both on the BRACKET "
               "rows of THIS artefact. The corpus-scale figure this sentence "
               "used to recite was measured on five logs and typed in; it is "
               "gone")
            + f". Both are printed on the BRACKET row so a reader can watch "
              f"them diverge"
            if emitted_brackets else
            "NOT ON THIS ARTEFACT. The locator computes it; this log carries no "
            "`repair: bracket` line, so for this file the width signature is "
            "all there is. Other artefacts in this corpus DO carry it -- an "
            "absence here is a fact about this record and not about the field")})
    entries.append({
        "quantity": "plateau_tolerance_ms",
        "state": NO_PRODUCER,
        "address": "change_point_locator.locate_change_points",
        "detail": "PLATEAU_TOLERANCE_MS=5.0 "
                  "(`grep 'PLATEAU_TOLERANCE_MS =' src/change_point_locator.py`"
                  " -> :230), and it IS returned in the plan dict "
                  "(`grep '\"plateau_tolerance_ms\"' src/change_point_locator.py`"
                  " -> :2588). "
                  "It is the gate that shifts the plateau mean, reaches the "
                  "plan and, like step_floor_ms below, is never printed to a "
                  "log line. NOT tolerance_ms=500, which is the "
                  "duration-enforcement tolerance on the CHECK row"})
    entries.append({
        "quantity": "step_floor_ms",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly",
        "detail": "MIN_STEP_MS=5.0 "
                  "(`grep 'MIN_STEP_MS =' src/change_point_locator.py` -> "
                  ":231). MIN_STEP_MS=5.0 IS returned by the locator "
                  "(`grep '\"step_floor_ms\"' src/change_point_locator.py` -> "
                  ":2578) and is never printed; the gate reaches the plan dict "
                  "and dies at the emitter"})
    entries.append({
        "quantity": "speed_margin",
        "state": NO_PRODUCER,
        # The locator does not put `speed_margin`, `fidelity_margin`,
        # `speed_margin_absent_reason` or `decided_by` into the plan dict, so these
        # keys never reach the log.
        "address": "NO WRITER IN change_point_locator.py, verified there "
                   "specifically as of a3ee9dbf (0 occurrences of all four "
                   "names -- `grep -c 'speed_margin\\|fidelity_margin\\|"
                   "decided_by' src/change_point_locator.py`, the same form "
                   "of check as merge_video_repair.py:1023-1027, with its own "
                   "firing control on `quantum_ms`). "
                   "merge_video_chimeric.py assigns and emits `decided_by` for "
                   "an unrelated extraction-bound decision (`decided_by=declared`"
                   "/`decided_by=packets`, `:2313,2331,2336` -- `grep -c "
                   "decided_by src/merge_video_chimeric.py` -> 6), a second, "
                   "different decision under the same name. Cross-file line "
                   "counts (`grep -c`, each named): `decided_by` chimeric:6 "
                   "(`grep -c decided_by src/merge_video_chimeric.py`) "
                   "repair:5 (`grep -c decided_by src/merge_video_repair.py`); "
                   "`speed_margin` repair:11 (`grep -c 'speed_margin\\b' "
                   "src/merge_video_repair.py` -- `\\b` needed: the bare "
                   "pattern also matches `speed_margin_absent_reason`'s own "
                   "lines, double-counting the same line into two rows); "
                   "`speed_margin_absent_reason` repair:2 (`grep -c "
                   "speed_margin_absent_reason src/merge_video_repair.py`); "
                   "`fidelity_margin` repair:5 (`grep -c fidelity_margin "
                   "src/merge_video_repair.py`). (A count OF this "
                   "file, STORED in this file, cannot be stated stably -- "
                   "writing it changes it, and every correction would be "
                   "another increment; not stated here for that reason. The "
                   "claim rests on the cross-file counts above, which this "
                   "text does not disturb.) `speed_margin`, "
                   "`speed_margin_absent_reason` and `fidelity_margin` outside "
                   "this module remain a READ (`plan.get(...)`), an EMISSION "
                   "(`merge_video_repair.py:1089,1092,1095`, "
                   "`parts.append(f\"...\")`) or a comment -- three "
                   "categories, not two; nothing else assigns them. "
                   "The plan comes from change_point_locator.locate_change_points "
                   "and its returned dict carries none of these keys. Addressed "
                   "THERE -- NOT to the emitter in merge_video_repair, which is "
                   "correct and simply never fed. THE TOKEN IS A MARKER OF AN "
                   "OPEN DECISION AND IT HAS AN OBSERVABLE END -- not a "
                   "permanent record. WHILE the question of whether the "
                   "locator should originate these keys is open, the token is "
                   "the only artefact-visible evidence that it IS open: as a "
                   "measurement of the plan it carries nothing, as a record it "
                   "is the sole trace that four keys were designed, consumed "
                   "and never originated, and cutting it leaves that fact only "
                   "in observers like this row, never in the produced record. "
                   "WHEN the question closes: if the keys are to be "
                   "originated, the fields FILL and there is nothing to cut; "
                   "if the design is dropped, the fields COME OUT and this "
                   "cell changes BEFORE the bytes do, so no artefact ever "
                   "renders a state this reader cannot account for. A "
                   "PERMANENT FIELD EMITTING ONE CONSTANT FOREVER WOULD BE THE "
                   "SAME DEFECT IN ANOTHER COSTUME, which is why the end "
                   "condition is written here rather than assumed",
        "detail": (f"0 occurrences across {corpus['logs']} logs / "
                   f"{corpus['distinct_cases']} distinct cases (see the CORPUS "
                   f"row: NOT an independent sample). "
                   if corpus else
                   "not seen on this artefact; NO POPULATION SUPPLIED, so this "
                   "report cannot say how many artefacts were looked at. ")
                  + "AND THE EMITTER WILL FAIL ON "
                  "ARRIVAL, PRECISELY WHERE IT MATTERS: log_assembly emits the "
                  "field CONDITIONAL ON ITS TRUTHINESS, and a margin that is "
                  "undefined is None, which is falsy. So on exactly the files "
                  "where the fidelity gate decided and no flatness margin "
                  "exists, the line prints nothing and this report would say NON "
                  "EMISE over a decision with a 0.3637 separation. "
                  "speed_margin_absent_reason, fidelity_margin and decided_by "
                  "need emitters OF THEIR OWN rather than riding on "
                  "speed_margin's truthiness -- otherwise the reason never "
                  "travels in the one case it exists to explain. NOTE for any "
                  "caption: fidelity_margin is winner MINUS best hypothesis that "
                  "did NOT clear the gate -- a separation across the selection "
                  "boundary, not a spread among the accepted, and wording it as "
                  "a spread would make it circular"})
    entries.append({
        "quantity": "verification_probe_positions",
        "state": COLLAPSED,
        "address": "merge_video_chimeric.verify_on_master_timeline",
        "detail": "per-probe master_position_ms lag_ms and correlation are built "
                  "and summarised to probes worst r_min; a max() has no position "
                  "so it cannot land on a timeline"})
    # This register only lists named properties; checking the produced file
    # itself would be needed for others.
    entries.append({
        "quantity": "audio_normalisation_gain_and_its_output_rate",
        "state": NO_PRODUCER,
        "address": "video.generate_normalised_file -- neither the gain nor the "
                   "output sample rate is emitted on a job that SUCCEEDS",
        "detail": (
            "The pipeline applies `highpass=f=60,lowpass=f=16000,"
            "volume=<gain>dB` whenever |gain| >= 0.5 dB (video.py:528-538); "
            "0.499 takes the `anull` path and 0.500 takes the filter -- A "
            "BRANCH CONDITION, NOT A PHYSICAL THRESHOLD. At 32000 Hz, "
            "`lowpass=f=16000` sits EXACTLY ON NYQUIST, which is the "
            "degenerate case this cell exists to flag: the filter has no "
            "headroom left to work with at that rate. "
            "THE OUTPUT RATE IS NOT ONE NUMBER; IT DEPENDS ON WHICH STAGE "
            "EXTRACTED THE STREAM. STAGE 2 (final normalisation) sets the "
            "rate to the MINIMUM OVER THE SELECTED LANGUAGE'S STREAMS ON "
            "BOTH SIDES, clamped only from above at 44100 "
            "(mergeVideo.py:583, video.py:268). STAGE 1 (comparison/delay "
            "extraction) SETS NO `-ar` AT ALL: `prepare_get_delay_sub` builds "
            "its parameter dict WITHOUT a `SamplingRate` key, "
            "`compare_video.__init__` takes a `.copy()` so stage 2's "
            "assignment cannot propagate back, and video.py appends `-ar` "
            "ONLY `if 'SamplingRate' in exportParam`. SO A STREAM EXTRACTED "
            "AT STAGE 1 KEEPS ITS OWN NATIVE RATE, and a natively-32000 one "
            "meets the filter at its own Nyquist with no pair, no minimum "
            "and no clamp involved. `extract_audio_in_part` iterates "
            "`self.audios[language]`, so LANGUAGE STILL GATES WHICH STREAMS "
            "REACH STAGE 1 AT ALL: the exposure condition is A STREAM IN THE "
            "SELECTED LANGUAGE THAT IS NATIVELY 32000 -- the other side's "
            "rate, the pair minimum and the clamp are irrelevant to it. "
            "TWO SILENT-FAILURE TRAPS ON THAT CONDITION, BOTH VERIFIABLE AT "
            "THE CURRENT TREE. (i) THE LANGUAGE KEY IS TWO-LETTER: "
            "`self.audios` is keyed by `Lang(data['Language']).pt1` (ISO-639-1 "
            "-- `fr`, `ja`), never the three-letter tag an artefact's own "
            "ffprobe output carries (`fre`/`jpn`); joining a raw container "
            "tag against this key returns zero rows and never errors. "
            "Resolve exposure from the LOG's language or by stream index, "
            "never by joining an unnormalised container tag against it. "
            "(ii) THE `compatible` GUARD ON BOTH STAGE-1 EXTRACTION LOOPS "
            "(`if audio[\"compatible\"]:`) IS INERT: there is exactly one "
            "write to that key anywhere, `data[\"compatible\"] = True`, and "
            "nothing ever sets it False -- it reads as a filter and is not "
            "one. `codec_param` is built ONCE and fed to both the EXTRACT "
            "command and the normaliser (`.copy()`), so the temporary file "
            "is ALREADY AT THE TARGET RATE before the filter ever runs -- the "
            "filter never resamples, it inherits. "
            "VISIBILITY LIMIT: this pair is only emitted inside an ECHOED "
            "FFMPEG COMMAND on a FAILED job; a job that SUCCEEDS emits "
            "neither the gain nor the rate, so on exactly the files that "
            "ship, this report can say nothing at all. `grid_hz=44100` on "
            "the locator line is NOT this rate -- it is the comparison grid, "
            "a different quantity, and reading one as the other is the "
            "wrong-slot error this register exists to make visible"),
    })
    entries.append({
        "quantity": "file_properties_nobody_named",
        "state": NO_PRODUCER,
        "address": "this register, and the spec it enumerates",
        "detail": "MEASURED INSTANCE: a produced file carried 4 chapters with "
                  "boundaries and titles identical to the master's. This report "
                  "does not mention chapters anywhere, no code path claims to "
                  "write them, and mergeVideo passes --no-chapters. A register "
                  "of empty cells enumerates what the SPEC names; it cannot "
                  "hold `no-producer` for a quantity nobody thought of. THIS "
                  "REPORT IS BLIND TO PROPERTIES OF THE FILE THAT NOBODY HAS "
                  "NAMED, and only an instrument reading the FILE rather than "
                  "the LOG can find them"})
    # No per-track duration is emitted; a rebuilt AAC track may be one frame
    # (21.33 ms) longer, below the 500 ms length tolerance.
    entries.append({
        "quantity": "produced_track_duration_or_frame_count",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": "AN AAC TRACK'S PACKET COUNT AND ITS DURATION TAG INCLUDE A "
                  "PRIMING FRAME THE DECODED AUDIO DOES NOT: the encoder's "
                  "priming frame at the head is declared as "
                  "initial_padding=1024 and start_time is 0.000000 on every "
                  "track, so a correct decoder discards it. CONTAINER PACKETS "
                  "AND DECODED AUDIO ARE DIFFERENT QUANTITIES -- a rebuilt AAC "
                  "track carrying one packet more than a passthrough sibling "
                  "is not, by itself, evidence the pipeline made the track "
                  "longer. A comparison against an E-AC-3 sibling cannot "
                  "settle it either way: E-AC-3 has no priming because it was "
                  "never encoded, so it agrees with both explanations and "
                  "discriminates neither. WHY THIS CELL EXISTS: no emitted key "
                  "carries a per-track duration or frame count, so this report "
                  "could not have told you either way -- and the only length "
                  "check is expected_ms against tolerance_ms=500, which 21.33 "
                  "ms clears by a factor of 23 by construction. A quantity "
                  "nobody emits was read wrong twice from outside the log"})
    entries.append({
        "quantity": "borrowed_placement_tag",
        "state": NO_PRODUCER,
        "address": "merge_video_chimeric.mux_repaired_file",
        "detail": "s4g requires a tag DISTINCT from VMSAM_FABRICATED; the marker "
                  "value is the single string chimeric on every marked track of "
                  "every produced file measured -- see the CORPUS row for "
                  "what that population is and is not"})
    # Two cells: the frame rate, and whether original and final rates differ.
    entries.append({
        "quantity": "video_frame_rate_disagreement",
        "state": (PRESENT if (job.get("output_check") or {}).get("frame_rate_original")
                  else NOT_EXERCISED if (job.get("output_check") or {}).get("frame_rate")
                  else ABSENT_FORMAT),
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": "frame_rate_original= is emitted ONLY when the original and "
                  "final rates differ. It has never fired: it can only fire on "
                  "files that are VFR, which are likely to be declined instead. "
                  "A blank here is not agreement between two rates -- it is one "
                  "rate and no second to compare it with"})
    entries.append({
        "quantity": "video_frame_rate",
        # Older logs lack the field.
        "state": (PRESENT if (job.get("output_check") or {}).get("frame_rate")
                  else NO_PRODUCER),
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": ("emitted on the output line as frame_rate=RATE(MODE). "
                   "Any statement about a delay landing on the video grid now "
                   "divides by a MEASURED rate rather than by an assumption -- "
                   "and the mode matters: two CFR files at 23.839 and 47.281 "
                   "take the snap branch and snap to a fabricated grid"
                   if (job.get("output_check") or {}).get("frame_rate") else
                   "no fps, frame_rate or FrameRate key on this artefact. Any "
                   "statement about a delay landing on the video grid -- "
                   "rounding, snapping, a drawn grid -- divides by a rate this "
                   "record does not carry, so it divides by an assumption")})
    entries.append({
        "quantity": "container_start_time",
        "state": NO_PRODUCER,
        "address": "merge_video_chimeric.get_stream_start_ms",
        # `head_pad_ms` is clamped at zero; `head_pad=` disambiguates, but the
        # container start time itself is not emitted.
        "detail": ("the AMBIGUITY is resolved: head_pad= now separates none, "
                   "padded and read_past, so a zero no longer covers three "
                   "states. The MAGNITUDE is still not emitted -- no start_time, "
                   "Delay or container-offset key exists -- so a constant "
                   "off-grid phase in the offsets still cannot be explained "
                   "from this record"
                   if any((job.get("audios") or {}).get(order, {}).get("head_pad")
                          for order in (job.get("audios") or {}))
                   else
                   "no start_time, Delay or container-offset key is emitted. "
                   "head_pad_ms is a one-sided derivative clamped at zero, so "
                   "head_pad_ms=0 means EITHER no container offset OR an offset "
                   "the plan reads past -- it cannot be inverted. A constant "
                   "off-grid phase in the offsets cannot be explained without it")})
    entries.append({
        "quantity": "build_identity",
        # Present when the `repair: build` line (or the sources digest) exists.
        "state": PRESENT if (job.get("build") or job.get("sources"))
                 else NO_PRODUCER,
        "address": ("merge_video_repair (repair: build line)" if job.get("build")
                    else "gestionar_show.fusion (job log header)"),
        "detail": ("CONTENT-DERIVED, confirmed by its producer and reproduced "
                   "independently: sha256 of the modules' own bytes read from "
                   "disk at call time, not a label passed in. So this is a "
                   "MEASUREMENT of which code ran, which is what the deploy "
                   "gate is not -- that verifies ARG VMSAM_GIT_COMMIT against "
                   "nothing. `build` covers 2 modules and `sources` the 27 .py "
                   "files the image ships; different scopes, NOT a refinement "
                   "of one another. Neither covers the interpreter, ffmpeg or "
                   "mkvtoolnix. And a digest is not a DATE, so the "
                   "`not-exercised-here` cells stay undatable"
                   if job.get("build") else
                   (f"no commit build version or image key occurs in any of the "
                    f"{corpus['logs']} logs; " if corpus else
                    "no commit build version or image key occurs on this "
                    "artefact, and no population was supplied to say how many "
                    "were checked; ")
                   + "the log records what was done and not which build did it")})
    return entries


def parse_rows(rows):
    """Parse the text rows back into records; the figures are drawn from these.

    Drawing from the rows guarantees the figure shows nothing the text lacks.
    """
    records = []
    for line in rows:
        if not line or line.startswith("#"):
            continue
        kind, _, rest = line.partition(" ")
        records.append((kind, split_fields(rest)))
    return records


def _x(value, span, width):
    if not span:
        return 0.0
    return float(Decimal(str(value)) / span) * width


def _text(x, y, content, fill, size=11, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{_escape(content)}</text>')


def _clip(text, limit):
    """Clip text for layout only, ending with an ellipsis; the full value stays on its row."""
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "\u2026"


def short_clock(ms):
    """Format milliseconds as `m:ss` below one hour, `h:mm:ss` above."""
    if ms is None:
        return "?"
    total = int(Decimal(str(ms)))
    seconds, _ = divmod(abs(total), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sign = "-" if total < 0 else ""
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{seconds:02d}"
    return f"{sign}{minutes}:{seconds:02d}"


def _geometry_key(regions):
    """Return a key equal for tracks whose per-region offsets are identical."""
    return tuple((f.get("master_start_ms"), f.get("offset_s") or f.get("offset_ms_state"))
                 for f in regions)


def render_svg(records):
    """Render the detailed SVG figure: master timeline, one staircase per distinct geometry.

    Cut rules mark each cut, bars show matched segments stepped by offset, and
    dashed boxes mark refused candidate material. Tracks borrowing another's
    offsets are drawn as named strips under that staircase.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    if not plan or not plan.get("master_end_ms"):
        return ('<p class="note">No plan geometry in this artefact, so there is '
                'no master timeline to draw. The rows above carry everything '
                'this format emitted.</p>')
    span = Decimal(plan["master_end_ms"])

    tracks, regions, lost, borrows, steps, refused = [], {}, {}, {}, {}, []
    for kind, fields in records:
        if kind == "TRACK" and fields.get("kind") == "audio":
            tracks.append(fields)
        elif kind == "REGION":
            regions.setdefault(fields["track"], []).append(fields)
        elif kind == "LOST":
            lost.setdefault(fields["track"], []).append(fields)
        elif kind == "BORROW":
            borrows[fields["track"]] = fields
        elif kind == "STEP":
            steps.setdefault(fields["track"], []).append(fields)
        elif kind == "REFUSED":
            # Per file: segments are refused by the plan, before tracks exist.
            refused.append(fields)

    # Group by geometry, in order of first appearance.
    groups = []
    for track in tracks:
        key = _geometry_key(regions.get(track["track"], []))
        for group in groups:
            if group["key"] == key:
                group["tracks"].append(track)
                break
        else:
            groups.append({"key": key, "tracks": [track]})

    ink, paper = "#c6ccd4", "#0f1115"
    salmon, blue, bar_blue, amber, faint = ("#e8836b", "#8fbcd4", "#a3cadd",
                                            "#c9a227", "#6f7681")
    left, right, width = 44, 44, 1080
    plot = width - left - right
    top = 48

    def x_of(value):
        return left + float(Decimal(str(value)) / span) * plot

    # Plan info line: a piecewise-constant plan steps, a constant plan is flat.
    body = []
    plan_line = (f"plan: {plan.get('kind')} \u00b7 measured on "
                 f"{plan.get('language')} \u00b7 quantum {plan.get('quantum_ms')} ms \u00b7 "
                 f"{plan.get('pieces')} pieces \u00b7 master duration "
                 f"{plain(plan.get('master_end')) or '?'}")
    if plan.get("decided_by"):
        plan_line += f" \u00b7 decided by {plan['decided_by']}"
    # "NOT EMITTED" only for NO_PRODUCER; NOT_DEFINED defers to `decided by`.
    if plan.get("speed_margin_state") == NO_PRODUCER:
        # Mark the value missing; a blank would read as a zero margin.
        plan_line += " \u00b7 speed margin: NOT EMITTED"
    elif plan.get("speed_margin_state") == NOT_DEFINED:
        plan_line += " \u00b7 flatness margin: NOT APPLICABLE here"
    elif plan.get("speed_margin"):
        plan_line += f" \u00b7 marge {plan['speed_margin']}"
    if plan.get("fidelity_margin"):
        plan_line += f" \u00b7 fidelity separation {plan['fidelity_margin']}"
    body.append(_text(left, top - 18, plan_line, faint, 10))
    y = top + 4

    for group in groups:
        lead = next((t for t in group["tracks"]
                     if not (plain(t.get("offset")) or "").startswith("BORROWED")),
                    group["tracks"][0])
        number = lead["track"]
        mine = regions.get(number, [])
        candidates = [f for f in mine if f.get("kind") == "CANDIDATE"]
        filled = [f for f in mine if f.get("kind") in ("MASTER", "SILENCE", "MASTER?")]
        others = [t for t in group["tracks"] if t["track"] != number]

        panel_top = y + 16
        step = 30 if len(candidates) < 8 else max(14, int(200 / max(1, len(candidates))))
        stair = max(1, len(candidates)) * step
        panel_bottom = panel_top + stair + 10
        master_bar_y = panel_bottom + 16
        panel_end = master_bar_y + 16 + (12 * len(others))

        # --- header: plan, track, and resampling state ---
        # A missing `speed=` field is its own state, not "resampled".
        speed = plain(lead.get("speed"))
        if speed is None:
            resampled, speed = None, ""
        else:
            resampled = not speed.lower().startswith("none")
        head = (f"track {number} · {lead.get('lang')} · "
                f"{'measured' if not (plain(lead.get('offset')) or '').startswith('BORROWED') else 'BORROWED'}")
        if resampled:
            # Only the numeric ratio goes in the header; prose stays on the TRACK row.
            ratio = speed.split("(")[0].strip()
            head += (f" \u00b7 RESAMPLED \u00d7{_clip(ratio, 12)}"
                     f" \u00b7 winning margin: NOT EMITTED")
        elif resampled is None:
            head += " \u00b7 rate: FIELD ABSENT from this artefact"
        elif "(" in speed and not speed.lower().startswith("none("):
            head += " \u00b7 speed: see the TRACK row"
        else:
            # `none(...)`: no rate was proposed; render as not measured.
            head += " \u00b7 rate: NOT MEASURED"
        if lead.get("verify"):
            head += f" · verify {_clip(plain(lead['verify']), 24)}"
        body.append(_text(left, y + 10, _clip(head, 132), faint, 10))

        # --- refused regions: dashed amber low band ---
        band = 18
        band_y = panel_top + max(0.0, (stair - band) / 2)
        for fields in refused:
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            body.append(f'<rect x="{x0:.1f}" y="{panel_top:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="{stair:.1f}" '
                        f'fill="{amber}" fill-opacity="0.05" stroke="none"/>')
            body.append(f'<rect x="{x0:.1f}" y="{band_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="{band}" '
                        f'fill="none" stroke="{amber}" stroke-width="1" '
                        f'stroke-dasharray="4 3"/>')
            if x1 - x0 > 108:
                body.append(_text((x0 + x1) / 2, band_y + 13,
                                  "no correspondence", amber, 9, "middle"))

        # --- cuts: full-height rules at the master insertion point ---
        # A cut removes candidate material, a zero-width event on the master timeline.
        for index, fields in enumerate(lost.get(number, [])):
            where = fields.get("where")
            if where == "head" and candidates:
                boundary = Decimal(candidates[0]["master_start_ms"])
            elif where == "tail" and candidates:
                boundary = Decimal(candidates[-1]["master_end_ms"])
            else:
                inner = [f for f in lost.get(number, []) if f.get("where") == "interior"]
                position = inner.index(fields) if fields in inner else 0
                boundary = (Decimal(candidates[position]["master_end_ms"])
                            if position < len(candidates) else span)
            x = x_of(boundary)
            body.append(f'<line x1="{x:.1f}" y1="{panel_top - 4:.1f}" x2="{x:.1f}" '
                        f'y2="{panel_bottom:.1f}" stroke="{salmon}" stroke-width="1.6"/>')
            # Head and tail cuts have no step: draw a small circle instead of a number.
            if not any(_trim(Decimal(f["at_master_ms"])) == _trim(boundary)
                       for f in steps.get(number, [])):
                body.append(f'<circle cx="{x:.1f}" cy="{panel_top - 14:.1f}" r="2" '
                            f'fill="none" stroke="{salmon}" stroke-width="1">'
                            f'<title>no step here: head or tail cut, '
                            f'nothing flanks it</title></circle>')

        # Label: offset step across the cut (difference of the adjacent offsets).
        for fields in steps.get(number, []):
            x = x_of(fields["at_master_ms"])
            anchor, at = "middle", x
            if x < left + 44:
                anchor, at = "start", left
            elif x > left + plot - 44:
                anchor, at = "end", left + plot
            body.append(_text(at, panel_top - 10, f"{fields['step_s']} s",
                              salmon, 11, anchor))

        # --- the staircase: one bar per matched segment, stepped down ---
        for index, fields in enumerate(candidates):
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            bar_y = panel_top + index * step + 4
            body.append(f'<rect x="{x0:.1f}" y="{bar_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="12" '
                        f'fill="{bar_blue}"/>')
            offset = fields.get("offset_s")
            if not offset:
                body.append(_text(x1 + 6, bar_y + 10, "decalage non emis", salmon, 10))
                continue
            # The `s` unit is written on the FIRST label of a series only.
            label = f"{offset} s" if index == 0 else offset
            span_px = x1 - x0
            estimate = len(label) * 6.4
            if span_px > estimate + 16:
                body.append(_text((x0 + x1) / 2, bar_y - 3, label, blue, 11, "middle"))
            elif x1 + 8 + estimate < left + plot:
                body.append(_text(x1 + 8, bar_y + 10, label, blue, 11))
            else:
                body.append(_text(x0 - 8, bar_y + 10, label, blue, 11, "end"))

        # --- master fills: bar labelled with source language, start, end, duration ---
        sources, filled_total, placed_marks = [], Decimal("0"), []
        for fields in filled:
            duration = (Decimal(fields["master_end_ms"])
                        - Decimal(fields["master_start_ms"]))
            filled_total += duration
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            body.append(f'<rect x="{x0:.1f}" y="{master_bar_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="7" fill="{amber}" '
                        f'fill-opacity="0.75"/>')
            # A fill exactly as wide as the search bound is outlined on the figure.
            if "UNREFINED SEARCH BOUND" in (plain(fields.get("fill_width")) or ""):
                body.append(f'<rect x="{x0:.1f}" y="{master_bar_y - 2:.1f}" '
                            f'width="{max(2.0, x1 - x0):.1f}" height="11" '
                            f'fill="none" stroke="{salmon}" stroke-width="1.4"/>')
                body.append(_text((x0 + x1) / 2, master_bar_y + 20,
                                  "search bound \u2014 bracket never narrowed",
                                  salmon, 9, "middle"))
            timing = (f"{short_clock(fields['master_start_ms'])} \u2192 "
                      f"{short_clock(fields['master_end_ms'])}  "
                      f"+{short_clock(duration)}")
            estimate = len(timing) * 5.2
            middle = (x0 + x1) / 2
            if not placed_marks or middle - estimate / 2 > placed_marks[-1] + 8:
                placed_marks.append(middle + estimate / 2)
                body.append(_text(middle, master_bar_y - 4, timing, amber, 9, "middle"))
            elif x1 - x0 > 34:
                body.append(_text(middle, master_bar_y - 4,
                                  f"+{short_clock(duration)}", amber, 9, "middle"))
            # Missing `from=`: show a sentence, not a state token.
            name = plain(fields.get("source"))
            if not name:
                name = ("source not recorded in this format"
                        if fields.get("source_state") == ABSENT_FORMAT
                        else "source not emitted")
            if name not in sources:
                sources.append(name)
        body.append(_text(left, master_bar_y + 20,
                          "added from the master: "
                          + (" \u00b7 ".join(sources) or "nothing")
                          + f" \u00b7 {len(filled)} region(s)"
                          + f" \u00b7 {short_clock(filled_total)}",
                          amber, 10))

        # --- the tracks that BORROW this geometry ----------------------------
        for index, other in enumerate(others):
            strip = master_bar_y + 28 + index * 12
            borrow = borrows.get(other["track"], {})
            # Borrowers whose offsets match exactly are drawn faint, others highlighted.
            check = plain(borrow.get("check")) or ""
            agrees = bool(re.match(r"offsets identical at (\d+) of \1 regions$",
                                   check))
            tone = faint if agrees else salmon
            body.append(f'<rect x="{left:.1f}" y="{strip:.1f}" width="{plot:.1f}" '
                        f'height="6" fill="none" stroke="{tone}" '
                        f'stroke-width="{1 if agrees else 1.4}" '
                        f'stroke-dasharray="{"3 3" if agrees else "none"}"/>')
            detail = (f"track {other['track']} · {other.get('lang')} · EMPRUNTE "
                      f"cette geometrie · verify "
                      f"{plain(other.get('verify')) or '?'}")
            if check:
                detail += f" · {check}"
            body.append(_text(left + 6, strip + 5.5, _clip(detail, 150), tone, 9))
        y = panel_end + 18

    # --- the axis: one thin line at the bottom, region boundaries as ticks ---
    axis = y + 6
    height = axis + 38
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;height:auto" role="img" '
           f'aria-label="master timeline, one staircase per distinct geometry">',
           '<title>merge_plan</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="{paper}"/>']
    out.extend(body)
    out.append(f'<line x1="{left}" y1="{axis:.1f}" x2="{left + plot}" '
               f'y2="{axis:.1f}" stroke="{faint}" stroke-width="1"/>')

    # Region boundary times go on the axis; labels above rules carry the step.
    boundaries = set()
    for group in groups:
        for fields in regions.get(group["tracks"][0]["track"], []):
            boundaries.add(Decimal(fields["master_start_ms"]))
            boundaries.add(Decimal(fields["master_end_ms"]))
    placed = []
    for position in sorted(boundaries):
        x = x_of(position)
        out.append(f'<line x1="{x:.1f}" y1="{axis:.1f}" x2="{x:.1f}" '
                   f'y2="{axis + 5:.1f}" stroke="{faint}" stroke-width="1"/>')
        label = clock(position)
        if label.endswith(".000"):
            label = label[:-4]
        # Skip labels that would overlap; ticks keep their position.
        half = len(label) * 3.2
        if placed and x - half < placed[-1] + 4:
            continue
        placed.append(x + half)
        out.append(_text(x, axis + 17, label, ink, 10, "middle"))
    out.append(_text(left, axis + 30, "master timeline", faint, 9))
    out.append('</svg>')
    return "\n".join(out)


def render_narrative(records):
    """Render the same facts as the figure, as sentences."""
    plan = next((f for k, f in records if k == "PLAN"), None)
    source = next((f for k, f in records if k == "SOURCE"), {})
    identity = next((f for k, f in records if k == "IDENTITY"), {})
    said = []

    declined = next((f for k, f in records if k == "DECLINED"), None)
    if declined:
        said.append(
            f"<p><b>THIS FILE WAS NOT PRODUCED.</b> The repair was "
            f"<b>REFUSED</b>: {_escape(plain(declined.get('reason', '?')))}. "
            f"Everything below describes what the plan WOULD have done, not the "
            f"contents of an artefact \u2014 there is no artefact.</p>")
    if plan:
        said.append(
            f"<p>Artefact <b>{_escape(plain(source.get('artefact', '?')))}</b> "
            f"was rebuilt on the master's timeline, which runs to "
            f"<b>{_escape(plain(plan.get('master_end', '?')))}</b>. The "
            f"measurement called the relationship "
            f"<b>{_escape(plan.get('kind', '?'))}</b> and took it on the "
            f"<b>{_escape(plan.get('language', '?'))}</b> track, at a quantum of "
            f"{_escape(plan.get('quantum_ms', '?'))} ms.</p>")
    if identity.get("master"):
        said.append(f"<p>The master is named in this log, as "
                    f"<b>{_escape(identity['master'])}</b>. That is what makes a "
                    f"deficit measured in the produced file <i>attributable</i> "
                    f"to the merge or to the master, rather than merely "
                    f"observable.</p>")
    else:
        said.append("<p><b>The master is not named in this artefact's format.</b> "
                    "A deficit measured in the produced file can therefore be "
                    "attributed to nobody: content correspondence to its source "
                    "is unmeasurable, and that is a property of the record and "
                    "not of the file.</p>")

    # Corpus paragraphs only when a population was supplied.
    corpus_supplied = not any(k == "CORPUS" and plain(f.get("state")) == NOT_SUPPLIED
                              for k, f in records)
    corpus_said = []
    bound_only = [f for k, f in records if k == "REGION"
                  and "UNREFINED SEARCH BOUND" in (plain(f.get("fill_width")) or "")]
    if bound_only:
        seen = {(f["master_start_ms"], f["master_end_ms"]) for f in bound_only}
        said.append(
            f"<p><b>{len(seen)} filled region(s) here are exactly 100 s wide "
            f"\u2014 the locator's UNREFINED SEARCH BOUND.</b> Its coarse pass "
            f"steps 40 s through a 60 s window, so a change point it never "
            f"managed to narrow stays bracketed by exactly 100 s \u2014 AND "
            f"THAT BRACKET IS WHAT GETS FILLED WITH MASTER AUDIO. A width "
            f"landing on the search constant to the millisecond is not a "
            f"measured hole; it is the measurement's uncertainty, delivered as "
            f"sound. EVIDENCE, NOT PROOF: a genuine 100 s hole would look the "
            f"same. But real boundaries in these files are numbers like 1442240 "
            f"and 152056, and never round.</p>")
        # Fills exactly at the 100 s search bound occur at head and interior but
        # not at the tail, so that width is the locator's bound, not content. The
        # counts come from the CORPUS row (`fill_census`).
        census = next((plain(f.get("fill_census")) for k, f in records
                       if k == "CORPUS" and f.get("fill_census")), None)
        corpus_said.append(
            "<p><b>The same files carry their own control.</b> The census is on "
            "the CORPUS row above and this sentence reads it rather than "
            "restating it: <code>" + (census or "not supplied") + "</code>. "
            "INTERIOR fills sit on the locator's refine grid; fills at the END "
            "of a plan do not, and their widths look like measured content. If "
            "100 s were a plausible width for real missing content, it would "
            "appear at the end of a plan too. It never does. That is the same "
            "files disagreeing with themselves depending on whether the locator "
            "had a bracket to close.</p>")
        # Interior fill widths fall on the locator's refine grid.
        corpus_said.append(
            "<p><b>Interior gap widths in this corpus are not measurements of "
            "the media. They are positions on the locator's search grid.</b> "
            "Every interior master fill is an exact multiple of the 4 s refine "
            "step and they take only a handful of distinct values, each one a "
            "locator constant \u2014 the 100 s un-narrowed search bound, and the "
            "12 s refine floor plus zero, one or two refine steps. The fills at "
            "the end of a plan take a DIFFERENT width every time and NOT ONE is "
            "a multiple of the refine step. The counts are on the CORPUS row; "
            "this sentence does not carry its own copy of them. Nothing in the "
            "pipeline quantises a master fill to 4 s \u2014 if it did, the ends "
            "of plans would be quantised too.</p>"
            "<p><b>And most of that interior count could not have gone the "
            "other way.</b> The 100 s search bound is 25 x 4 s exactly, so "
            "every fill that IS the bound sits on the refine grid by "
            "construction and tests nothing. Those are counted out on the "
            "CORPUS row, and the remainder — the widths the locator "
            "actually narrowed — is the population this claim rests on. "
            "It is a much smaller number and it is the honest one; a fill at "
            "the bound belongs to the OTHER claim, the one the emitted "
            "<code>bound_only</code> field now settles directly. The "
            "end-of-plan column is unaffected: not one of those fills is at "
            "the bound, so every one of them could have landed on the grid and "
            "none did.</p>")
        # Refused records are excluded from this population.
        corpus_said.append(
            "<p><b>And that four-value list is a property of files that "
            "SHIPPED.</b> Refusal records are not in this corpus. In the "
            "refused records on disk that carry a plan, interior fills of 900 s "
            "and 932 s were measured — 15 minutes of master audio in one "
            "record — and both are still exact multiples of the 4 s refine "
            "step. THAT COUNT IS NOT PRODUCED BY THIS RENDER: refusal records "
            "are outside the population handed in, so it was measured "
            "separately and is dated by that measurement rather than by this "
            "artefact. So the GRID claim holds across both populations and the "
            "FOUR-VALUE claim holds only where a file was delivered. The pattern's edge is in the files that did not "
            "ship, which is the argument for keeping refusals in a census "
            "rather than the argument for excluding them.</p>")
        # No interior fill exceeds the search bound; head and tail fills may.
        corpus_said.append(
            "<p><b>And no delivered file in this corpus has an interior fill "
            "wider than the search bound.</b> The count is on the CORPUS row: "
            "<code>" + (next((plain(f.get("fills_above_the_bound"))
                              for k, f in records
                              if k == "CORPUS" and f.get("fills_above_the_bound")),
                             None) or "not supplied") + "</code>. Fills DO "
            "exceed 100 s, and every one of them is at the start or the end of "
            "a plan, where widths are not on the grid at all. Interior fills "
            "stop at 100 s in every file that shipped.</p>")
    if corpus_supplied:
        said.extend(corpus_said)
    elif corpus_said:
        said.append(
            "<p><b>A corpus-scale finding about this file's gap widths exists "
            "and is NOT shown here, because no population was supplied to this "
            "render.</b> It concerns whether an interior gap width is a "
            "measurement of the media or a position on the locator's search "
            "grid, and it cannot be stated from one artefact: it needs a "
            "population, and the numbers behind it were measured elsewhere. "
            "Render this log together with others and the finding appears with "
            "its denominator. WHAT THIS REPORT CAN STILL TELL YOU ABOUT THIS "
            "FILE ALONE is above: the width of each filled region, and whether "
            "it equals a locator constant.</p>")
    # Say whether this log explains why each gap was filled (`bound_only`).
    bracket_score = next((plain(f.get("bracket_agreement")) for k, f in records
                          if k == "CORPUS" and f.get("bracket_agreement")), None)
    if [f for k, f in records if k == "BRACKET"]:
        said.append(
            "<p><b>And on this artefact it CAN tell you why.</b> The locator "
            "now emits its own bracket for each change point, with "
            "<code>bound_only</code> — its statement that it could not narrow "
            "the search. Read the BRACKET rows: where <code>bound_only=True</code>, "
            "the master audio filling that gap is the width of the "
            "measurement's uncertainty and not the width of a hole in the "
            "candidate. The derived signature this report used before the field "
            "existed is printed beside it"
            + ((", and across the population handed to this render: "
                + html.escape(bracket_score) + ".</p>") if bracket_score
               else ". No population was supplied to this render, so this "
                    "paragraph says nothing about how often the two agree "
                    "elsewhere.</p>"))
    else:
        said.append(
        "<p><b>This figure cannot tell you WHY a gap was filled.</b> A region "
        "taken from the master because the candidate genuinely had nothing "
        "there, and a region taken because the measurement could not pin down "
        "the change point, are drawn IDENTICALLY. The locator computes which is "
        "which and it does not reach this report; the width above is a derived "
        "signature standing in for it. See "
        "<code>gap_is_filled_because_unsure</code> for the address. OTHER "
        "ARTEFACTS IN THIS CORPUS DO CARRY IT: the field has shipped, and its "
        "absence here is a fact about this record.</p>")

    tracks = [f for k, f in records if k == "TRACK" and f.get("kind") == "audio"]
    for track in tracks:
        number = track["track"]
        regions = [f for k, f in records if k == "REGION" and f["track"] == number]
        lost = [f for k, f in records if k == "LOST" and f["track"] == number]
        borrow = next((f for k, f in records
                       if k == "BORROW" and f["track"] == number), None)
        kept = [f for f in regions if f.get("kind") == "CANDIDATE"]
        filled = [f for f in regions if f.get("kind") in ("MASTER", "SILENCE")]
        offsets = [f.get("offset_s") for f in kept if f.get("offset_s")]

        sentence = [f"<p><b>Track {_escape(number)} "
                    f"({_escape(track.get('lang', '?'))}).</b> "]
        if kept:
            sentence.append(
                f"It takes {len(kept)} region(s) from the candidate and fills "
                f"{len(filled)} from "
                f"{_escape(plain(track.get('fill', 'the master')))}. ")
        if offsets:
            unique = []
            for value in offsets:
                if value not in unique:
                    unique.append(value)
            if len(unique) > 1:
                sentence.append(
                    f"<b>It reads the candidate at {len(unique)} different "
                    f"offsets</b> \u2014 "
                    f"{', '.join(_escape(v) + ' s' for v in unique)} \u2014 "
                    f"while the track line carries a single label, "
                    f"<code>offset={_escape(plain(track.get('offset', '?')))}</code>. "
                    f"One token stands over {len(unique)} values. ")
            else:
                sentence.append(f"It reads the candidate at "
                                f"{_escape(unique[0])} s throughout. ")
        else:
            sentence.append("<b>No offset is recoverable for this track</b>: "
                            "this format emits none by name, and the plan cut "
                            "nothing to derive one from. ")
        if borrow:
            sentence.append(
                f"<b>It BORROWS</b> the geometry of track "
                f"{_escape(borrow.get('from_track', '?'))}, the one carrying the "
                f"measurement language: "
                f"{_escape(plain(borrow.get('check', '?')))}. The log states "
                f"this nowhere \u2014 it is an inference, verified rather than "
                f"asserted. ")
        if lost:
            total = sum(Decimal(f["dropped_ms"]) for f in lost
                        if f.get("dropped_ms") not in (None, "UNMEASURED"))
            places = ", ".join(sorted({_escape(f.get("where", "?")) for f in lost}))
            sentence.append(f"<b>{_escape(seconds_fr(total, 1, signed=False))} s "
                            f"of candidate material is not in the output</b>, "
                            f"across {len(lost)} region(s), at the {places}. ")
        verify = plain(track.get("verify"))
        if verify:
            sentence.append(f"Verification says <code>{_escape(verify)}</code>")
            if track.get("probes"):
                sentence.append(f" on {_escape(track['probes'])} probe(s), worst "
                                f"{_escape(track.get('worst', '?'))}")
            if track.get("verified"):
                sentence.append(f", and {_escape(track['verified'])} of the "
                                f"file's audio tracks were verified at all")
            sentence.append(". ")
        sentence.append("</p>")
        said.append("".join(sentence))

    shortfalls = [f for k, f in records if k == "SHORTFALL"]
    if shortfalls:
        said.append(
            "<p><b>This file declares a shortfall, and this report CANNOT say "
            "where it came from.</b> The track line quantifies how short the "
            "fill source was and what the track lost; it does not say whether "
            "the repair caused it or <i>inherited</i> it from an already-short "
            "master. This log names the master and carries none of its "
            "durations, so the question is not decidable here. To date, <b>all "
            "five defects measured on produced files were inherited</b> "
            "\u2014 a <i>borrowed</i> figure, not measured here and recorded in "
            "no reference document \u2014 and no instance of the pipeline "
            "introducing one has been confirmed. Fidelity to a source is not "
            "correctness of an output.</p>")

    speeds = [f for k, f in records if k == "SPEED"]
    if speeds and all(f.get("ratio_applied_state") for f in speeds):
        said.append(
            "<p><b>No track in this artefact received a rate correction</b>, "
            "and that is not the same as \u201cnone needed one\u201d. The "
            "measurement proposed no rate, and <b>no producer of a speed "
            "verdict exists</b>: nobody asked the question. The figure has "
            "therefore never been drawn for a resampled track \u2014 that "
            "display path has never run.</p>")

    refused = [f for k, f in records if k == "REFUSED"]
    none_row = next((f for k, f in records if k == "REFUSED_NONE"), None)
    if refused:
        total = sum(Decimal(f["dropped_ms"]) for f in refused
                    if f.get("dropped_ms"))
        said.append(
            f"<p><b>{len(refused)} candidate region(s) were REFUSED</b>, "
            f"{_escape(seconds_fr(total, 1, signed=False))} s in all: the plan "
            f"had a candidate there and discarded it. They carry the dashed "
            f"amber box on the figure. That is a different thing from a fill "
            f"taken from the master, and both marks coexist.</p>")
    elif none_row:
        shortest = none_row.get("shortest_candidate_segment_ms")
        said.append(
            "<p><b>No candidate region was refused in this artefact</b>, so the "
            "figure carries no dashed amber box. Read that as \u201cthis case "
            "did not occur here\u201d and not as \u201cnothing is ever "
            "refused\u201d."
            + (f" A segment is refused only if it is SHORTER than the probe "
               f"window, which is 60 s; the shortest segment in this plan is "
               f"{_escape(seconds_fr(shortest, 0, signed=False))} s. The "
               f"condition is therefore not met"
               + (", and not by much" if Decimal(str(shortest)) < 120000 else "")
               + "." if shortest else "")
            + " The mark has never yet been drawn against genuinely refused "
              "material.</p>")

    gaps = [f for k, f in records if k == "GAP"]
    missing = [f for f in gaps if f.get("state") == NO_PRODUCER]
    if missing:
        said.append(
            f"<p><b>{len(missing)} quantity(ies) this report must show have no "
            f"producer at all:</b> "
            f"{', '.join('<code>' + _escape(plain(f['quantity'])) + '</code>' for f in missing)}. "
            f"They are listed above with their addresses. An empty cell here is "
            f"a FINDING and not a formatting accident \u2014 and it is "
            f"deliberately distinguishable from a field this format simply did "
            f"not carry yet.</p>")
    return "\n".join(said)


class merge_plan_error(Exception):
    """Raised when a call cannot produce a correct report."""


def _job_contract():
    """Return the keys a job dict must carry, taken from `parse_job_log("")`."""
    return frozenset(parse_job_log("").keys())


def validate_job(job):
    """Raise merge_plan_error naming every key missing from `job`.

    Some keys are read with brackets only on some branches, so a missing key
    is refused up front instead of failing on an unlucky file.
    """
    if not isinstance(job, dict):
        raise merge_plan_error(
            f"job must be the dict `parse_job_log` returns, not "
            f"{type(job).__name__}. Build it with `parse_job_log(<the emitted "
            f"bytes>)` and never by hand: a hand-built dict tests a fixture and "
            f"not the output, which is the failure this zone exists for")
    missing = sorted(_job_contract() - set(job))
    if missing:
        raise merge_plan_error(
            f"job is missing {len(missing)} key(s) this report reads: "
            f"{', '.join(missing)}. Pass `parse_job_log(<the emitted bytes>)`. "
            f"NOTE: 13 of the 15 keys are read both with brackets and with "
            f"`.get` in different branches, so an absent key crashes ONLY ON "
            f"SOME INPUTS -- a call that works on ten files can die on the "
            f"eleventh. This refusal is deliberate and is not fixed by "
            f"defaulting the value: a report rendered with a silently absent "
            f"field is the same class as a truncated document that still opens")
    return job


# Page layout:
#   A. PLAN     schematic of the master timeline
#   B. SUMMARY  a few lines in French, one table row per cut
#   C. DETAIL   rows, findings, detailed figure and narrative
# A and B are views of C: `_assert_figure_says_nothing_new` checks that every
# number drawn is in the rows (axis graduations carry `data-scale`). The
# summary may also read the full merge log for the final mux's refusals.

def _fr_number(value, decimals=3):
    """Format a number with a decimal comma, truncated (not rounded) to `decimals`.

    Truncation keeps the label a prefix of the row value, so the figure check
    can find it.
    """
    if value is None:
        return "?"
    quantum = Decimal(1).scaleb(-decimals)
    text = format(Decimal(str(value)).quantize(quantum, rounding="ROUND_DOWN"), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text.replace(".", ",")


def _fr_clock(ms):
    """Format milliseconds as `mm:ss,mmm` (`h:mm:ss,mmm` past the hour), built from `clock`."""
    if ms is None:
        return "?"
    text = clock(ms)
    sign = "-" if text.startswith("-") else ""
    hours, _, rest = text.lstrip("-").partition(":")
    return sign + (rest if hours == "0" else f"{hours}:{rest}").replace(".", ",")


def _fr_duration(ms):
    """Format a duration for the summary table: `6,023 s`, `1 min 12,5 s`."""
    if ms is None:
        return "?"
    value = Decimal(str(ms)) / 1000
    if 0 < abs(value) < 1:
        return f"{_fr_number(ms, 3)} ms"
    if abs(value) < 60:
        return f"{_fr_number(value, 3)} s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)} min {_fr_number(seconds, 1)} s"


def _dec(value):
    try:
        return Decimal(str(plain(value)))
    except Exception:
        return None


def delivery_drops(records):
    """Read the delivery gate's drops from the `fabricated_dropped` rows.

    Returns:
        ({language: number of dropped tracks}, {0-based ranks dropped}).

    `stream=` on a drop line is the 0-based rank of the track among the
    rebuilt audio tracks in build order, not its `track=` StreamOrder. With
    TRACK rows sorted by `track=`, rank 0 is the lowest `track=`.
    """
    drops = {}
    dropped_ranks = set()
    for kind, fields in records:
        if kind != "UNPARSED":
            continue
        line = plain(fields.get("line")) or ""
        if line.startswith("fabricated_dropped "):
            rest = split_fields(line[len("fabricated_dropped "):])
            language = rest.get("lang")
            if language:
                drops[language] = drops.get(language, 0) + 1
            stream = rest.get("stream")
            if stream is not None:
                try:
                    dropped_ranks.add(int(stream))
                except (TypeError, ValueError):
                    pass
    return drops, dropped_ranks


def plan_geometry(records):
    """Return the geometry the schematic and table draw, read from the rows.

    The lead track is the first delivered, measured (not borrowed) track with
    regions; dropped tracks are identified by rank (see `delivery_drops`).
    When no such track is delivered, returns `{"undelivered": True, ...}`.
    A cut is every non-candidate region plus every seam between adjacent
    candidate pieces; LOST rows are attached to their cut through
    `candidate = master + offset_ms`, or paired by order (marked derived).
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    if not plan or not plan.get("master_end_ms"):
        return None
    span = Decimal(plan["master_end_ms"])
    tracks, regions, steps, lost, refused = [], {}, {}, {}, []
    for kind, fields in records:
        if kind == "TRACK" and fields.get("kind") == "audio":
            tracks.append(fields)
        elif kind == "REGION":
            regions.setdefault(fields["track"], []).append(fields)
        elif kind == "STEP":
            steps.setdefault(fields["track"], []).append(fields)
        elif kind == "LOST":
            lost.setdefault(fields["track"], []).append(fields)
        elif kind == "REFUSED":
            refused.append(fields)
    _drop_counts, dropped_ranks = delivery_drops(records)
    # A drop line's `stream=` is a 0-based rank among rebuilt tracks ordered by
    # `track=`; only that track is gone.
    ranked = sorted(tracks, key=lambda t: int(t["track"]))
    dropped_track_numbers = {ranked[rank]["track"] for rank in dropped_ranks
                             if 0 <= rank < len(ranked)}
    delivered = [t for t in tracks if t.get("track") not in dropped_track_numbers]
    # A language is gone only once EVERY track it had is gone.
    gone = ({t.get("lang") for t in tracks} -
            {t.get("lang") for t in delivered})
    measured = [t for t in delivered if regions.get(t["track"])
                and not (plain(t.get("offset")) or "").startswith("BORROWED")]
    with_regions = [t for t in delivered if regions.get(t["track"])]
    lead = (measured or with_regions or [None])[0]
    if lead is None:
        if any(regions.get(t["track"]) for t in tracks):
            return {"undelivered": True, "plan": plan, "span": span,
                    "dropped": sorted(gone), "tracks": tracks}
        return None
    number = lead["track"]
    regs = sorted(regions[number], key=lambda r: Decimal(r["master_start_ms"]))
    step_at = {Decimal(s["at_master_ms"]): s for s in steps.get(number, [])}

    cuts = []
    for index, region in enumerate(regs):
        start = Decimal(region["master_start_ms"])
        end = Decimal(region["master_end_ms"])
        kind = region.get("kind") or "?"
        where = ("head" if index == 0 else
                 "tail" if index == len(regs) - 1 else "interior")
        if kind != "CANDIDATE":
            source = plain(region.get("source")) or ""
            language = source.partition("/")[2] or "?"
            inserted = ("silence" if kind == "SILENCE" else
                        f"maître ({language})" + (" ~" if kind.endswith("?") else ""))
            cuts.append({"start": start, "end": end, "region": region,
                         "where": where, "inserted": inserted, "fill": kind,
                         "step": step_at.get(start), "lost": []})
        elif index and regs[index - 1].get("kind") == "CANDIDATE":
            cuts.append({"start": start, "end": start, "region": None,
                         "where": "interior", "inserted": "rien — coupe franche",
                         "fill": "NONE", "step": step_at.get(start), "lost": []})

    candidates = [r for r in regs if r.get("kind") == "CANDIDATE"]
    unplaced = []

    def cut_at(boundary):
        return next((c for c in cuts
                     if c["start"] == boundary or c["end"] == boundary), None)

    for item in lost.get(number, []):
        where = item.get("where")
        target = None
        if where == "head" and candidates:
            boundary = Decimal(candidates[0]["master_start_ms"])
            target = cut_at(boundary)
            if target is None:
                target = {"start": boundary, "end": boundary, "region": None,
                          "where": "head", "fill": "NONE",
                          "inserted": "rien — début du candidat coupé",
                          "short": "début du candidat coupé",
                          "step": None, "lost": []}
                cuts.insert(0, target)
        elif where == "tail" and candidates:
            boundary = Decimal(candidates[-1]["master_end_ms"])
            target = cut_at(boundary)
            if target is None:
                target = {"start": boundary, "end": boundary, "region": None,
                          "where": "tail", "fill": "NONE",
                          "inserted": "rien — fin du candidat coupée",
                          "short": "fin du candidat coupée",
                          "step": None, "lost": []}
                cuts.append(target)
        else:
            begin = _dec(item.get("candidate_start_ms"))
            for region in candidates:
                offset = _dec(region.get("offset_ms"))
                if offset is None or begin is None:
                    continue
                if abs(Decimal(region["master_end_ms"]) + offset - begin) < 1:
                    target = cut_at(Decimal(region["master_end_ms"]))
                    break
        if target is None:
            unplaced.append(item)
        else:
            target["lost"].append((item, False))
    interior_free = [c for c in cuts if c["where"] == "interior" and not c["lost"]]
    interior_left = [i for i in unplaced if i.get("where") == "interior"]
    if interior_left and len(interior_left) == len(interior_free):
        for cut, item in zip(interior_free, interior_left):
            cut["lost"].append((item, True))
        unplaced = [i for i in unplaced if i not in interior_left]
    cuts.sort(key=lambda c: (c["start"], c["end"]))
    geometries = len({_geometry_key(regions.get(t["track"], []))
                      for t in with_regions})
    return {"plan": plan, "span": span, "lead": lead, "regions": regs,
            "candidates": candidates, "cuts": cuts, "refused": refused,
            "unplaced": unplaced, "tracks": tracks, "geometries": geometries}


_SCHEMA_COLOURS = {"CANDIDATE": "var(--candidate)", "MASTER": "var(--master)",
                   "MASTER?": "var(--master)", "SILENCE": "var(--silence)"}


def render_plan_schematic(geometry):
    """Render the plan schematic (part A): cuts, the file bar and the offset staircase."""
    if geometry is None:
        return ('<p class="note">Pas de géométrie de plan dans cet artefact : '
                'rien à dessiner. Le détail pour l\'IA porte tout ce que le '
                'journal a émis.</p>')
    if geometry.get("undelivered"):
        return ('<p class="note">Aucune piste livrée : la porte de livraison a '
                'écarté toutes les pistes reconstruites ('
                + _escape(", ".join(geometry["dropped"])) + ') ; le fichier porte '
                'l\'audio du maître, sans coupe. Rien à dessiner.</p>')
    span = geometry["span"]
    width, left, right = 1100, 60, 60
    plot = width - left - right

    def x_of(value):
        return left + float(Decimal(str(value)) / span) * plot

    # --- labels above the bar, staggered so neighbours never overwrite ----
    level_gap, label_w = 30, 118
    levels_last = []
    placed = []
    for cut in geometry["cuts"]:
        x = x_of(cut["start"])
        for level, last in enumerate(levels_last):
            if x - last >= label_w:
                levels_last[level] = x
                break
        else:
            if len(levels_last) < 4:
                level = len(levels_last)
                levels_last.append(x)
            else:
                level = min(range(len(levels_last)), key=lambda i: levels_last[i])
                levels_last[level] = x
        placed.append((cut, level))
    top = 14 + max(1, len(levels_last)) * level_gap
    bar_y, bar_h = top + 6, 30
    axis_y = bar_y + bar_h + 12

    body = []
    # Fills drawn on top, at least 5 px wide so short ones stay visible; exact
    # bounds are in the title, the table and the REGION rows.
    ordered = sorted(geometry["regions"],
                     key=lambda r: r.get("kind") != "CANDIDATE")
    for region in ordered:
        x0, x1 = x_of(region["master_start_ms"]), x_of(region["master_end_ms"])
        colour = _SCHEMA_COLOURS.get(region.get("kind"), "var(--faint)")
        opacity = ' fill-opacity="0.55"' if region.get("kind") == "MASTER?" else ""
        minimum = 1.5 if region.get("kind") == "CANDIDATE" else 5.0
        if x1 - x0 < minimum and x0 + minimum > left + plot:
            x0 = left + plot - minimum
        body.append(
            f'<rect x="{x0:.1f}" y="{bar_y}" width="{max(minimum, x1 - x0):.1f}" '
            f'height="{bar_h}" fill="{colour}"{opacity}><title>'
            f'{_escape(region.get("kind"))} {_escape(_fr_clock(region["master_start_ms"]))}'
            f' → {_escape(_fr_clock(region["master_end_ms"]))}</title></rect>')
    for fields in geometry["refused"]:
        x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
        body.append(f'<rect x="{x0:.1f}" y="{bar_y - 4}" '
                    f'width="{max(3.0, x1 - x0):.1f}" height="{bar_h + 8}" '
                    f'fill="none" stroke="var(--lost)" stroke-width="1.5" '
                    f'stroke-dasharray="5 3"><title>matériau du candidat refusé'
                    f'</title></rect>')

    for cut, level in placed:
        x = x_of(cut["start"])
        label_y = 14 + level * level_gap
        body.append(f'<line x1="{x:.1f}" y1="{label_y + 16}" x2="{x:.1f}" '
                    f'y2="{bar_y + bar_h + 4}" stroke="var(--ink)" '
                    f'stroke-width="1"/>')
        anchor = "start" if x < width - left - label_w else "end"
        dx = 3 if anchor == "start" else -3
        body.append(_text(x + dx, label_y, _fr_clock(cut["start"]),
                          "var(--ink)", 11, anchor, ' font-weight="600"'))
        if cut["step"] is not None:
            second = (("coupe franche · " if cut["fill"] == "NONE" else "")
                      + f"saut {_fr_number(cut['step']['step_ms'])} ms")
        elif cut["fill"] == "NONE":
            second = cut.get("short", "coupe franche")
        else:
            second = "rempli " + cut["inserted"].split(" ")[0]
        body.append(_text(x + dx, label_y + 12, second, "var(--faint)", 10, anchor))

    # --- the axis: graduations are SCALE (data-scale), the end is DATA ------
    body.append(f'<line x1="{left}" y1="{axis_y}" x2="{left + plot}" '
                f'y2="{axis_y}" stroke="var(--rule)" stroke-width="1"/>')
    seconds = float(span) / 1000
    tick = next((t for t in (30, 60, 120, 300, 600, 900, 1800, 3600)
                 if seconds / t <= 9), 3600)
    value = 0
    while value * 1000 < float(span) - tick * 250:
        x = x_of(value * 1000)
        body.append(f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" '
                    f'y2="{axis_y + 5}" stroke="var(--rule)"/>')
        minutes, secs = divmod(int(value), 60)
        body.append(_text(x, axis_y + 17, f"{minutes:02d}:{secs:02d}",
                          "var(--faint)", 10, "middle", ' data-scale="1"'))
        value += tick
    end_x = x_of(span)
    body.append(f'<line x1="{end_x:.1f}" y1="{axis_y}" x2="{end_x:.1f}" '
                f'y2="{axis_y + 5}" stroke="var(--ink)"/>')
    body.append(_text(end_x, axis_y + 17, _fr_clock(span), "var(--ink)", 10,
                      "end"))

    # --- the staircase: one level per distinct candidate offset -------------
    stair_top = axis_y + 48
    offsets = sorted({_dec(c.get("offset_ms")) for c in geometry["candidates"]
                      if _dec(c.get("offset_ms")) is not None}, reverse=True)
    # Proportional to the offset, not by rank.
    row_gap = 18
    depth = min(110, row_gap * max(1, len(offsets) - 1))
    high, low = (offsets[0], offsets[-1]) if offsets else (0, 0)
    level_of = {offset: (float((high - offset) / (high - low)) * depth / row_gap
                         if high != low else 0.0)
                for offset in offsets}
    unknown_y = stair_top + (depth + row_gap if offsets else 0)
    body.append(_text(left, stair_top - 16,
                      "décalage du candidat (ms) — une marche par saut",
                      "var(--faint)", 10))
    previous = None
    for region in geometry["candidates"]:
        offset = _dec(region.get("offset_ms"))
        y = (round(stair_top + level_of[offset] * row_gap, 1)
             if offset is not None else unknown_y)
        x0, x1 = x_of(region["master_start_ms"]), x_of(region["master_end_ms"])
        dash = "" if offset is not None else ' stroke-dasharray="3 3"'
        label = (f"{_fr_number(offset)} ms" if offset is not None
                 else "décalage non émis")
        body.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" '
                    f'stroke="var(--candidate)" stroke-width="3"{dash}>'
                    f'<title>{_escape(label)}</title></line>')
        if previous is not None:
            body.append(f'<line x1="{previous[0]:.1f}" y1="{previous[1]}" '
                        f'x2="{x0:.1f}" y2="{y}" stroke="var(--faint)" '
                        f'stroke-width="1" stroke-dasharray="2 2"/>')
        if x1 - x0 > 70:
            body.append(_text(x0 + 3, y - 4, label, "var(--ink)", 10))
        previous = (x1, y)
    missing = any(_dec(c.get("offset_ms")) is None for c in geometry["candidates"])
    height = int((unknown_y if missing else stair_top + depth) + 14)
    lead = geometry["lead"]
    title = (f"Plan sur la timeline du maître, piste {lead.get('track')} "
             f"({lead.get('lang')})")
    return (f'<svg class="schema" viewBox="0 0 {width} {height}" width="100%" '
            f'role="img" aria-label="{_escape(title)}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'font-family="ui-sans-serif,DejaVu Sans,system-ui,sans-serif">'
            f'<title>{_escape(title)}</title>' + "".join(body) + "</svg>")


def _resample_badge(job):
    """Return the applied speed ratio as an exact fraction and percentage, or ''."""
    applied = []
    for order in sorted(job.get("audios") or {}):
        ratio = applied_ratio(job["audios"][order])
        if ratio not in (None, "UNREADABLE"):
            applied.append((order, ratio))
    if not applied:
        return ""
    order, ratio = applied[0]
    fraction = Fraction(ratio).limit_denominator(100000)
    percent = (ratio - 1) * 100
    sign = "+" if percent >= 0 else ""
    tracks = ", ".join(str(o) for o, _ in applied)
    return (f'<p class="badge">Rééchantillonné ×{_escape(_fr_number(ratio, 6))} '
            f'= {fraction.numerator}/{fraction.denominator} '
            f'({sign}{_escape(_fr_number(percent, 3))} %) — piste(s) {tracks}</p>')


# `tools.logs` lines of the final mux, matched by search on the dedented line.
_MUX_DROPS = (
    (re.compile(r"Track (commentary|descriptive) (\d+) not added from (.*?)\.?$"),
     None),
    (re.compile(r"Skip the element (\d+) not added for (\S+) from (.*?)\. It seems to be empty"),
     "piste vide"),
    (re.compile(r"Track (\d+) with md5 \S+ not added for (\S+?)(?: from (.*?))?\. "
                r"It have the same md5 as other track added"),
     "doublon (même md5 qu'une piste déjà ajoutée)"),
    (re.compile(r"Track (\d+) with md5 \S+ not added for (\S+?)(?: from (.*?))?\. "
                r"It is not keep"),
     "non retenue par les règles de langue"),
    (re.compile(r"^Track (\d+) not added for (\S+) from (.*?)\.?$"),
     "langue à retirer complètement"),
)


def mux_drops(merge_log):
    """Return {reason: [(stream, language, source path or None)]} from the full merge log."""
    drops = {}
    for line in (merge_log or "").splitlines():
        text = line.strip()
        for pattern, reason in _MUX_DROPS:
            matched = pattern.search(text)
            if not matched:
                continue
            if reason is None:
                kind = matched.group(1)
                drops.setdefault("commentaire" if kind == "commentary"
                                 else "audiodescription", []).append(
                    (matched.group(2), None, matched.group(3)))
            else:
                drops.setdefault(reason, []).append(
                    (matched.group(1), matched.group(2),
                     matched.group(3) if matched.lastindex >= 3 else None))
            break
    return drops


def _source_word(path, job):
    if not path:
        return None
    name = path.rsplit("/", 1)[-1]
    if job.get("master_path") and name == job["master_path"].rsplit("/", 1)[-1]:
        return "maître"
    if (job.get("candidate_path") and name == job["candidate_path"].rsplit("/", 1)[-1]) \
            or "_repaired" in name:
        return "candidat"
    return "autre source"


def render_human_summary(job, geometry, merge_log=None):
    """Render the human summary (part B): a short French list with decimal commas."""
    said = []
    if job.get("declined"):
        said.append("<li><b>Réparation REFUSÉE</b> — aucun fichier produit ; "
                    "ce qui suit décrit ce que le plan AURAIT fait.</li>")

    # 1. resample
    applied = [ratio for ratio in (applied_ratio(job["audios"][order])
                                   for order in sorted(job.get("audios") or {}))
               if ratio not in (None, "UNREADABLE")]
    if applied:
        ratio = applied[0]
        fraction = Fraction(ratio).limit_denominator(100000)
        percent = (ratio - 1) * 100
        said.append(f"<li><b>Rééchantillonnage : oui</b> — ×{_fr_number(ratio, 6)} "
                    f"= {fraction.numerator}/{fraction.denominator} "
                    f"({'+' if percent >= 0 else ''}{_fr_number(percent, 3)} %).</li>")
    elif job.get("audios"):
        said.append("<li><b>Rééchantillonnage : non</b> — vitesse 1 (la mesure "
                    "n'a proposé aucun facteur).</li>")
    else:
        said.append("<li><b>Rééchantillonnage :</b> inconnu — aucune piste audio "
                    "dans ce journal.</li>")

    # 2. the cuts
    table = ""
    if geometry is None:
        said.append("<li><b>Coupes :</b> pas de géométrie de plan dans ce journal.</li>")
    elif geometry.get("undelivered"):
        said.append("<li><b>Coupes : aucune piste livrée</b> — la porte de livraison a "
                    "écarté toutes les pistes reconstruites ("
                    + _escape(", ".join(geometry["dropped"])) + ") : le fichier "
                    "livré porte l'audio intact du maître, aucune coupe.</li>")
    else:
        cuts = geometry["cuts"]
        pieces = len(geometry["candidates"])
        said.append(
            f"<li><b>Coupes : {len(cuts)}</b> — {pieces} morceau(x) du candidat "
            f"recollé(s) sur la timeline du maître, qui dure "
            f"{_fr_clock(geometry['span'])} (piste {geometry['lead'].get('track')}, "
            f"{_escape(geometry['lead'].get('lang'))}"
            + (f" ; {geometry['geometries']} géométries différentes entre pistes, "
               f"voir le détail" if geometry["geometries"] > 1 else "")
            + ").</li>")
        lines = ["<table class=\"cuts\"><thead><tr><th>#</th><th>où</th>"
                 "<th>début (maître)</th><th>fin</th><th>durée</th>"
                 "<th>inséré</th><th>saut de décalage</th>"
                 "<th>retiré du candidat</th></tr></thead><tbody>"]
        where_fr = {"head": "tête", "tail": "fin", "interior": "milieu"}
        for index, cut in enumerate(cuts, 1):
            removed = " + ".join(
                ("~" if derived else "") + _fr_duration(item.get("dropped_ms"))
                for item, derived in cut["lost"]) or "—"
            step = (f"{_fr_number(cut['step']['step_ms'])} ms"
                    if cut["step"] is not None else "—")
            lines.append(
                f"<tr><td>{index}</td><td>{where_fr.get(cut['where'], '?')}</td>"
                f"<td>{_fr_clock(cut['start'])}</td><td>{_fr_clock(cut['end'])}</td>"
                f"<td>{_fr_duration(cut['end'] - cut['start'])}</td>"
                f"<td>{_escape(cut['inserted'])}</td><td>{step}</td>"
                f"<td>{removed}</td></tr>")
        lines.append("</tbody></table>")
        table = "".join(lines)
        if geometry["unplaced"]:
            said.append("<li>Retiré du candidat sans point de coupe identifiable : "
                        + ", ".join(_fr_duration(i.get("dropped_ms"))
                                    for i in geometry["unplaced"]) + ".</li>")

    # 2.5. picture-only shifts: logged, nothing changed
    shifts = job.get("picture_only_shifts") or []
    for entry in shifts:
        span = re.match(r"\[\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\]", entry.get("master_s") or "")
        where = (f"{_fr_clock(_decimal(span.group(1)) * 1000)} – "
                f"{_fr_clock(_decimal(span.group(2)) * 1000)}") if span else "?"
        said.append(
            f"<li>Image décalée, son continu (zone {_escape(entry.get('zone'))}, "
            f"{where}) : {_escape(entry.get('residual_frames'))} image(s) sur "
            f"{_escape(entry.get('cuts'))} coupure(s) confirmée(s) — rien n'a été "
            f"modifié, c'est une information.</li>")

    # 3. what was rebuilt and what the delivery gate did with it
    marker = re.search(r"marker '([^']+)'", job.get("summary_counts") or "")
    tag = marker.group(1) if marker else "?"
    audios = job.get("audios") or {}
    audio_text = ", ".join(
        f"{_escape(f.get('lang'))} (trous remplis : {_escape(plain(f.get('fill')) or '?')})"
        for _, f in sorted(audios.items())) or "aucune"
    subs = job.get("subtitles") or []
    sub_langs = ", ".join(sorted({str(s.get("lang")) for s in subs}))
    said.append(f"<li><b>Reconstruit</b> (tag <code>{_escape(tag)}</code>) : "
                f"audio {len(audios)} — {audio_text} ; sous-titres {len(subs)}"
                + (f" ({_escape(sub_langs)})" if subs else "") + ".</li>")
    delivery = job.get("delivery") or []
    # One line per verdict and cause, naming its tracks.
    causes = {
        "intact_same_language_wins": "même contenu qu'une piste intacte du "
                                     "maître, l'intacte gagne",
        "no_intact_master_track": "aucune piste intacte du maître dans cette "
                                  "langue",
        "commentary_tagged": "piste de commentaire",
        "different_version": "version différente de celle du maître",
    }
    grouped = {}
    for gate in delivery:
        grouped.setdefault((gate["verdict"], gate.get("cause")), []).append(gate)
    for (verdict, cause), gates in grouped.items():
        names = ", ".join(
            f"{_escape(g.get('lang'))} {_escape(g.get('format'))}"
            + (f" → maître {_escape(g.get('kept_master_stream'))} "
               f"{_escape(g.get('kept_master_format'))}"
               if g.get("kept_master_stream") else "")
            + (f" (langue réelle {_escape(g.get('cross_lang'))})"
               if g.get("cross_lang") else "")
            for g in gates)
        said.append(f"<li>Livraison : {len(gates)} piste(s) reconstruite(s) "
                    f"<b>{'écartée(s)' if verdict == 'dropped' else 'gardée(s)'}</b>"
                    f" — {_escape(causes.get(cause, cause))} : {names}.</li>")
    dropped_audio = sum(1 for g in delivery if g["verdict"] == "dropped"
                        and g.get("holder") in ("audios", "commentary", "audiodesc"))
    if audios and dropped_audio >= len(audios):
        said.append("<li><b>→ Aucune piste audio du candidat n'entre dans le "
                    "fichier</b> : toutes ont été écartées par la porte de "
                    "livraison.</li>")
        dropped_subs = sum(1 for g in delivery if g["verdict"] == "dropped"
                           and g.get("holder") == "subtitles")
        if dropped_subs >= len(subs):
            said.append("<li><b>Aucune piste ajoutée, le master l'emporte</b> — la "
                        "fusion a lieu quand même et l'erreur se ferme : le candidat "
                        "n'apportait rien.</li>")

    # 4. the final mux's refusals, from the full log when it was passed
    if merge_log is None:
        said.append("<li>Merge final : les lignes « not added » (pistes non "
                    "ajoutées) sont émises après la réparation et n'ont pas été "
                    "transmises à ce rapport.</li>")
    else:
        drops = mux_drops(merge_log)
        if not drops:
            said.append("<li>Merge final : aucune piste signalée « not added » "
                        "dans le journal.</li>")
        for reason, items in drops.items():
            languages = ", ".join(sorted({str(l) for _, l, _ in items if l}))
            sources = {}
            for _, _, path in items:
                word = _source_word(path, job)
                if word:
                    sources[word] = sources.get(word, 0) + 1
            origin = ", ".join(f"{w} {n}" for w, n in sorted(sources.items()))
            said.append(f"<li>Merge final — non ajoutée(s), {_escape(reason)} : "
                        f"{len(items)} piste(s)"
                        + (f" ({_escape(languages)})" if languages else "")
                        + (f" — {_escape(origin)}" if origin else "") + ".</li>")

    # 5. final length vs the master's video
    durations = job.get("output_durations") or {}
    container = _dec(durations.get("container_ms"))
    expected = _dec(durations.get("expected_ms"))
    if container is not None and expected is not None:
        gap = container - expected
        tolerance = (job.get("output_check") or {}).get("tolerance_ms")
        outside = (tolerance is not None and _dec(tolerance) is not None
                   and abs(gap) > _dec(tolerance))
        # The produced file is the repaired candidate handed to the mux, not the
        # final .mkv.
        said.append(f"<li><b>Longueur du fichier réparé</b> : "
                    f"{_fr_clock(container)} · vidéo maître {_fr_clock(expected)}"
                    f" · écart {'+' if gap >= 0 else ''}{_fr_number(gap)} ms"
                    + (f" (tolérance {_escape(tolerance)} ms)" if tolerance else "")
                    + (" — <b>HORS TOLÉRANCE</b>" if outside else "") + ".</li>")
    else:
        said.append("<li><b>Longueur du fichier réparé</b> : non mesurée dans "
                    "ce journal.</li>")

    if table:
        cut_line = next(i for i, s in enumerate(said) if "<b>Coupes" in s)
        said.insert(cut_line + 1, f'<li class="table">{table}</li>')
    return f'<ul class="resume" lang="fr">{"".join(said)}</ul>'


def render_report(job, artefact_id, source_name, caveats=(), corpus=None,
                  merge_log=None):
    """Render the whole HTML report page from a parsed job.

    Parts: A. plan schematic, B. French summary, C. detail (rows, findings,
    detailed figure, narrative). A and B derive from the rows and the job;
    every SVG is checked to draw no number absent from the text.

    Args:
        merge_log: optional full job log, used by the summary for the final
            mux's "not added" lines.
    """
    validate_job(job)
    rows = build_rows(job, artefact_id, source_name, list(caveats), corpus)
    records = parse_rows(rows)
    generation, description = format_generation(job)
    geometry = plan_geometry(records)

    document = [
        # The doctype must come first (otherwise quirks mode).
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<!-- VMSAM merge_plan report. SPEC_ZONE_A.MD s4g.",
        "     THE ROWS BELOW ARE THE REPORT. The diagram is rendered from them and",
        "     adds no quantity of its own: `grep`, `cat` and `diff` give every",
        "     number in this file with no browser. Resolve fields BY NAME.",
        f"     artefact={artefact_id} format_generation={generation} ({description})",
        "     Opaque ids only: no media filename, title or catalogue id appears here.",
        "-->",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>merge_plan {_escape(artefact_id)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>merge_plan — artefact {_escape(artefact_id)}</h1>",
        ('<p class="note">Ce fichier contient des noms de médias : ne pas le '
         'copier hors du dossier de sortie.</p>' if not REDACT_MEDIA_NAMES else ''),
        "<h2>Le plan</h2>",
        '<p class="legend">'
        '<span><i class="sw" style="background:var(--candidate)"></i>candidat</span>'
        '<span><i class="sw" style="background:var(--master)"></i>rempli depuis le maître</span>'
        '<span><i class="sw" style="background:var(--silence)"></i>silence</span>'
        '<span><i class="sw dashed"></i>candidat refusé</span>'
        '<span><i class="sw cut"></i>coupe : heure maître + saut</span></p>',
        _resample_badge(job),
        f'<div class="diagram">{render_plan_schematic(geometry)}</div>',
        "<h2>Résumé pour l'humain</h2>",
        render_human_summary(job, geometry, merge_log),
        "<h2>Détail pour l'IA</h2>",
        ('<p class="note"><b>This file carries media names.</b> It is generated '
         'by VMSAM, written beside the produced file in the output directory, '
         'and never enters the repository. <b>Do not copy it, or lines from it, '
         'outside that directory</b> — quote the opaque id on the IDENTITY row '
         'instead, which is carried beside every name for exactly that purpose. '
         'The reader of this file is the one most likely to be tempted to cite '
         'it.</p>' if not REDACT_MEDIA_NAMES else ''),
        '<p class="note">Every record is one line, <code>KIND key=value …</code>. '
        'Resolve by name; there are no columns. A value that is not present carries '
        '<code>&lt;key&gt;_state</code> instead, so a field this format predates is '
        'never confused with a field nothing emits.</p>',
        "<h3>Rows</h3>", "<pre>",
    ]
    document.extend(_escape(row) for row in rows)
    document.append("</pre>")

    document.append("<h3>Timeline</h3>")
    document.append(
        '<p class="note legend">'
        '<span><i class="sw" style="background:var(--candidate)"></i>from the candidate</span>'
        '<span><i class="sw" style="background:var(--master)"></i>filled from the master</span>'
        '<span><i class="sw" style="background:var(--silence)"></i>filled with silence</span>'
        '<span><i class="sw" style="background:var(--lost)"></i>lost — candidate material '
        'not in the output</span></p>')
    document.append(
        '<p class="note">The axis is the <b>master</b> timeline. A lost region lives on '
        'the <b>candidate</b> timeline and is drawn at its insertion point on the master, '
        'at the same scale, so its extent compares with what was kept — a length drawn on '
        'an axis that is not its own, said rather than assumed. A <code>~</code> before an '
        'offset means it was <b>derived</b> from other emitted fields, not read.</p>')
    document.append(
        '<p class="note">Dashed amber marks <b>candidate material that was '
        'refused</b> \u2014 the plan had a candidate there and dropped it. That '
        'is a different thing from the master-contribution bar under the '
        'staircase, which shows what came <b>from the master</b>; a region can '
        'be either, both or neither. <b>If no dashed box appears, this artefact '
        'refused nothing</b> \u2014 a case not exercised here, and not a claim '
        'that nothing is ever refused.</p>')
    document.append(f'<div class="diagram">{render_svg(records)}</div>')

    document.append("<h3>What was done to this file</h3>")
    document.append(f'<div class="narrative">{render_narrative(records)}</div>')

    # Every number drawn must appear in the text rows; raises otherwise.
    _assert_figure_says_nothing_new(document)
    document.append("</body></html>")
    rendered = "\n".join(document)
    assert_no_leak(rendered)
    return rendered


def report_for_log(text, artefact_id, source_name, caveats=(), corpus=None):
    """Render the report for job-log text; raise if it is not a job log."""
    if not is_job_log(text):
        raise ValueError(
            f"{source_name} carries no `repair: plan` line, so it is not a job "
            f"log. Rejected by STRUCTURE and not by name: a `.log` suffix is not "
            f"evidence, and a denominator defended by a filename is defended "
            f"until the next filename.")
    return render_report(parse_job_log(text), artefact_id, source_name,
                         caveats, corpus)


# Destination: the report is written beside the produced file; a log entry
# (`transport_entry`) is only a copy for readers without disk access.

# Locator constants, copied from `change_point_locator.py` (keep in sync):
#   100 000 ms = PROBE_STEP 40 000 + PROBE_WINDOW 60 000  (unrefined search bound)
#    12 000 ms = REFINE_WINDOW 8 000 + REFINE_STEP 4 000   (refined floor)
# A filled gap exactly as wide as the search bound reflects the locator's
# uncertainty, not measured content.
_LOCATOR_TAG = "[change_point_locator]"

SEARCH_BOUND_MS = 100000
# Locator refine step, in ms.
REFINE_STEP_MS = 4000
REFINE_WINDOW_MS = 8000
PROBE_STEP_MS = 40000
PROBE_WINDOW_MS = 60000

REFINE_FLOOR_MS = 12000

TRANSPORT_KEYWORD = "MERGE_PLAN_HTML"

# Media names are allowed: the report stays in the output directory beside
# the media. Set to True to redact them.
REDACT_MEDIA_NAMES = False


def _assert_figure_says_nothing_new(document):
    """Raise if a number visible in a figure is absent from the text rows."""
    # Check every figure against the text outside all figures; `data-scale`
    # graduations are exempt.
    text = "".join(document)
    figures = re.findall(r"<svg.*?</svg>", text, re.S)
    if not figures:
        return
    rows = re.sub(r"<svg.*?</svg>", " ", text, flags=re.S)
    drawn = set()
    for figure in figures:
        for label in re.findall(r"<text(?![^>]*data-scale)[^>]*>([^<]*)</text>",
                                figure):
            drawn |= set(re.findall(r"\d+[.,]?\d*", html.unescape(label)))
    plain_rows = html.unescape(re.sub(r"<[^>]+>", " ", rows))
    missing = sorted(n for n in drawn
                     if n not in plain_rows and n.replace(",", ".") not in plain_rows)
    if missing:
        raise merge_plan_error(
            "the figure draws numbers that are in no row: "
            + ", ".join(missing[:12])
            + ". This report's construction is that the drawing shows nothing "
              "the grep-able text does not; a render that breaks it is not "
              "emitted")


def transport_entry(document, artefact_id):
    """Return the document as one log entry prefixed by its length in UTF-8 bytes.

    A length prefix (not a delimiter) lets the reader slice exactly n bytes;
    the length counts bytes because the French labels contain multi-byte
    characters.
    """
    payload = document.encode("utf-8")
    return (f"{TRANSPORT_KEYWORD} {artefact_id} bytes={len(payload)}\n"
            + document + "\n")


def report_path(produced_file_path):
    """Return `<produced file name>.merge_plan.html`, beside the produced file."""
    return str(produced_file_path) + ".merge_plan.html"


# Call site: `mergeVideo.py` fills `merge_plan` with the emitted log bytes
# when a repaired version exists; `fusion.py` calls `parse_job_log`, then
# `write_report` with the published output path. A repair that refuses
# produces no report.
def write_report(job, artefact_id, source_name, produced_file_path, caveats=(),
                 corpus=None, merge_log=None):
    """Write the report beside the produced file and return (path, transport_entry).

    Does not write to `tools.logs`; the caller decides whether to log the entry.
    """
    validate_job(job)
    document = render_report(job, artefact_id, source_name, caveats, corpus,
                             merge_log)
    destination = report_path(produced_file_path)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(document)
    return destination, transport_entry(document, artefact_id)
