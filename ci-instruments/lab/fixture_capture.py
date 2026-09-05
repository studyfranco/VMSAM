"""Capture the live arguments of remove_not_compatible_video.

LAB TOOL. NEVER COMMITTED TO THE REPOSITORY. Applied to a SEPARATE image at build
time; the image serving the hundred-file run is not touched, because a capture hook
in it would make every remaining row a result from a build nobody specified.

WHERE THE FIXTURES GO AND WHY: /config/output/fixtures, NOT the repo. The arguments
are library paths, and the standing rule is that a library path never enters a tracked
file. Outside the repo they are in the same privacy domain as the media itself.

WHAT IT RECORDS, AND THE RULE THAT SHAPES IT:
  AN UNSERIALISABLE FIELD THAT VANISHES IS A FIELD NO TEST WILL EVER EXERCISE, and
  its absence looks exactly like a field the objects do not have. So every attribute
  is either serialised or NAMED IN `unserialisable` with its type. Nothing is dropped
  silently.

FIELDS BY NAME, NEVER BY POSITION -- WRITE_ZONES section 7. This becomes a cross-agent
artefact the moment anyone else reads it, and positional reads have cost this campaign
a 2.7 GB near-eviction and a truncation figure wrong by a factor of thirteen.
"""
import datetime, json, os, traceback

OUT = os.environ.get('FIXTURE_DIR', '/config/output/fixtures')
_seq = [0]


def _describe(obj, depth=0):
    """Serialise what a JSON encoder accepts; NAME what it does not."""
    if depth > 3:
        return {"_truncated_at_depth": depth}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_describe(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _describe(v, depth + 1) for k, v in obj.items()}
    out, unser = {}, {}
    for name in sorted(dir(obj)):
        if name.startswith('_'):
            continue
        try:
            val = getattr(obj, name)
        except Exception as e:
            unser[name] = f"<getattr raised {type(e).__name__}>"
            continue
        if callable(val):
            continue
        try:
            json.dumps(val)
            out[name] = val                      # SERIALISED DIRECTLY: a real value
        except (TypeError, ValueError):
            # NOT DIRECTLY SERIALISABLE. Recursion may still describe it -- a file
            # handle yields {mode, name, closed, ...} -- and that description is JSON
            # and looks exactly like data. A test author reading it would think the
            # object HAS those fields meaningfully. So a described object is marked
            # as described, and the reader can tell a value from a rendering.
            try:
                desc = _describe(val, depth + 1)
                json.dumps(desc)
                if isinstance(desc, dict):
                    desc['_described_not_serialised'] = True
                out[name] = desc
                unser[name] = f"{type(val).__name__} (DESCRIBED, not a value)"
            except Exception:
                out.pop(name, None)
                unser[name] = f"{type(val).__name__} (DROPPED: undescribable)"
    if unser:
        out['_unserialisable'] = unser      # NAMED, not dropped
    out['_type'] = type(obj).__name__
    return out


def capture(call_site, list_not_compatible_video, dict_file_path_obj, best_video,
            compare_objs=None):
    """Write one fixture. Never raises into the pipeline."""
    try:
        _seq[0] += 1
        os.makedirs(OUT, exist_ok=True)
        rec = {
            "schema": "remove_not_compatible_video/1",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc)
                                    .strftime('%Y-%m-%dT%H:%M:%SZ'),
            "call_site": call_site,
            "seq": _seq[0],
            # THE THREE ARGUMENTS, AS THE FUNCTION CAN ACTUALLY REACH THEM
            "list_not_compatible_video": list(list_not_compatible_video or []),
            "list_not_compatible_video_len": len(list_not_compatible_video or []),
            "dict_file_path_obj": {str(k): _describe(v)
                                   for k, v in (dict_file_path_obj or {}).items()},
            "best_video": _describe(best_video),
            # SITE 1 ONLY. compareObjs is the SOLE SURVIVOR of a tournament -- the
            # loop exits at len <= 1 -- so there is no index to record. It is captured
            # because a one-element list is a fact worth recording rather than an
            # assumption worth trusting: if it is EVER not one element, that is a
            # finding, and the row is where it would show.
            "compareObjs_len": (None if compare_objs is None else len(compare_objs)),
            "compareObjs_paths": (None if compare_objs is None else
                                  [getattr(o, 'filePath', None) for o in compare_objs]),
            # the empty case raises IndexError at site 1 and is reachable only through
            # the caller. Recorded so the fixture stock can carry the failing path.
            "empty_list_case": not (list_not_compatible_video or []),
        }
        path = os.path.join(OUT, f"rnc_{rec['captured_utc'].replace(':', '')}"
                                 f"_{call_site}_{_seq[0]:04d}.json")
        tmp = path + '.part'
        with open(tmp, 'w') as fh:          # COMPUTE FIRST, RENAME SECOND: a truncated
            json.dump(rec, fh, indent=1)    # fixture that audits clean is worse than none
        os.replace(tmp, path)
    except Exception:
        try:
            with open(os.path.join(OUT, 'capture-errors.log'), 'a') as fh:
                fh.write(traceback.format_exc() + "\n---\n")
        except Exception:
            pass
