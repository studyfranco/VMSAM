"""Verify audio language tags by content rather than trusting them.

A track tagged with the comparison language may carry another language's content, which would
end the repair on low similarity while hiding the real cause.

Runs in `repair_orchestrator.repair` when the candidate has no comparison-language track, or
when every comparison-language couple reads as different content.

Procedure:
  1. fingerprint every audio track of both files once (reusing the prime's fingerprints);
  2. compare candidate tracks with the master reference tracks, and tracks within each file:
     60 s chromaprint windows at 25/50/75 %, best offset within +/-300 s, pair similarity is the
     lowest window (>= 2 windows), same content at >= `file_conformity.FP_SAME_CONTENT`;
  3. decide:
       * a candidate track tagged another language matches -> `audio_tag_conflict`: it is
         re-tagged (marked `VMSAM_tag_corrected`) and the repair runs again on the corrected pair;
       * no candidate track matches -> `no_common_language_after_tag_check`;
       * a comparison-language track matches -> tags are consistent, nothing changes.
  Same content under different tags within one file is always logged.

No language identification is done: content comparison only.
"""

import time

import numpy as np

import file_conformity as fc
import tools

AUDIO_TAG_CONFLICT = "audio_tag_conflict"
NO_COMMON_LANGUAGE = "no_common_language_after_tag_check"
SAME_CONTENT = fc.FP_SAME_CONTENT              # 0.90
WINDOW_S = fc.FP_WINDOW_S                      # 60 s
FRACTIONS = (0.25, 0.5, 0.75)
SHIFT_S = 300.0
MIN_WINDOWS = 2
# A reference window whose fingerprint barely varies is silence or a steady tone and would
# match any other silence, so it is skipped.
MIN_DISTINCT_FRACTION = 0.5
HOLDERS = ("audios", "audiodesc", "commentary")


def tracks_of(video_obj):
    """[(holder, language, audio dict)] of every audio track of the object."""
    out = []
    for holder in HOLDERS:
        for language, audios in (getattr(video_obj, holder, None) or {}).items():
            for audio in audios:
                out.append((holder, language, audio))
    return out


def pair_similarity(reference, other, hop_ms):
    """Content similarity of two whole-track chromaprint lists.

    60 s windows of `reference` at FRACTIONS are searched in `other` within +/-SHIFT_S.

    Returns:
        (similarity, windows): the lowest window similarity (None under MIN_WINDOWS
        windows) and a list of (fraction, similarity, shift_s).
    """
    if reference is None or other is None:
        return None, []
    ref = np.asarray(reference, dtype=np.int64).astype(np.uint32)
    oth = np.asarray(other, dtype=np.int64).astype(np.uint32)
    n = int(round(WINDOW_S * 1000.0 / hop_ms))
    shift = int(round(SHIFT_S * 1000.0 / hop_ms))
    windows = []
    for fraction in FRACTIONS:
        i0 = int(len(ref) * fraction) - n // 2
        if i0 < 0 or i0 + n > len(ref):
            continue
        window = ref[i0:i0 + n]
        if len(np.unique(window)) < MIN_DISTINCT_FRACTION * n:
            continue
        lo = max(0, i0 - shift)
        segment = oth[lo:i0 + n + shift]
        if len(segment) < n:
            continue
        sim, s = fc._fp_similarity(window, segment, max_shift=len(segment) - n, min_overlap=n)
        if sim is None:
            continue
        windows.append((fraction, round(sim, 4), round((lo + s - i0) * hop_ms / 1000.0, 1)))
    if len(windows) < MIN_WINDOWS:
        return None, windows
    return min(w[1] for w in windows), windows


def _label(side, holder, language, audio):
    return f"{side}#{audio['StreamOrder']}({language}{'' if holder == 'audios' else '/' + holder})"


def _remove(path_):
    import os
    try:
        os.remove(path_)
    except OSError:
        pass


def fingerprint_every_track(video_obj, side, primed, work_dir, deadline, fingerprint=None,
                            keep_if=None):
    """Fingerprint every audio track of `video_obj`, reusing the prime's fingerprints.

    Returns `(fingerprints, unmeasured, ran_out)`: `{stream_order: points}`, the tracks that
    could not be fingerprinted, and whether the deadline passed. With `keep_if(points) ->
    score|None`, the WAV of the best-scoring track is kept in `primed["tag_check_wav"]` for
    reuse; every other WAV is deleted."""
    import repair_orchestrator as ro
    fingerprint = fingerprint or ro.fingerprint_track
    fps, unmeasured = {}, []
    for holder, language, audio in tracks_of(video_obj):
        order = audio["StreamOrder"]
        key = (side, order)
        if key in primed.get("fingerprints", {}):
            fps[str(order)] = primed["fingerprints"][key][0]
            continue
        if deadline is not None and time.monotonic() > deadline:
            return fps, unmeasured, True
        duration = ro._track_duration_seconds(video_obj, language, order)
        if duration is None:
            unmeasured.append((str(order), "no_duration"))
            continue
        if side == "master":
            video_ms = ro._video_duration_ms(video_obj)
            if video_ms is not None:
                duration = min(duration, float(video_ms) / 1000.0)
        measures = {}
        try:
            points, quantum_ms = fingerprint(video_obj, language, order, side, work_dir,
                                             primed.get("sample_rate"), duration,
                                             measures=measures, keep_wav=keep_if is not None)
        except Exception as error:                                       # noqa: BLE001
            unmeasured.append((str(order), type(error).__name__))
            continue
        if not points:
            unmeasured.append((str(order), "no_fingerprint"))
            continue
        fps[str(order)] = points
        wav = measures.get("wav")
        if wav:
            score = keep_if(points) if keep_if is not None else None
            kept = primed.get("tag_check_wav")
            if score is not None and (kept is None or score > kept["score"]):
                if kept is not None:
                    _remove(kept["path"])
                primed["tag_check_wav"] = {"path": wav, "stream": order, "duration_s": duration,
                                           "rate": primed.get("sample_rate"), "score": score}
            else:
                _remove(wav)
        if side == "candidate":
            # Stored in the prime's shape so a re-run reuses it.
            primed.setdefault("fingerprints", {})[key] = (points, quantum_ms, duration)
            primed.setdefault("content_end", {})[key] = measures.get("content_end_s")
    return fps, unmeasured, False


def content_check(master_obj, candidate_obj, language, primed, work_dir, deadline=None,
                  fingerprint=None):
    """Compare audio track contents to check the language tags.

    Returns:
        dict with `verdict` and `reason`. Verdict is AUDIO_TAG_CONFLICT (with
        `track`, `tag`, `master_stream`, `similarity`, `moves`), NO_COMMON_LANGUAGE
        (with `best`), 'tags_consistent', 'repair_budget_exceeded' or 'unmeasured'.
    """
    import repair_orchestrator as ro
    t0 = time.time()
    hop = ro.CHROMAPRINT_HOP_MS
    path = candidate_obj.filePath
    master_fp, m_un, m_out = fingerprint_every_track(master_obj, "master", primed, work_dir,
                                                     deadline, fingerprint)
    refs0 = [str(m) for m, _ in (primed.get("couples") or [])]

    def keep_if(points):
        sims = [pair_similarity(master_fp.get(m), points, ro.CHROMAPRINT_HOP_MS)[0]
                for m in refs0]
        sims = [x for x in sims if x is not None]
        return max(sims) if sims and max(sims) >= SAME_CONTENT else None
    cand_fp, c_un, c_out = (({}, [], True) if m_out else fingerprint_every_track(
        candidate_obj, "candidate", primed, work_dir, deadline, fingerprint, keep_if=keep_if))
    if m_out or c_out:
        drop_kept_wav(primed)
        return {"verdict": "repair_budget_exceeded",
                "reason": "the repair's budget ran out while every audio track was fingerprinted "
                          "for the content check (ADDENDUM 28)"}
    master_tracks = tracks_of(master_obj)
    cand_tracks = tracks_of(candidate_obj)
    references = [str(m) for m, _ in (primed.get("couples") or [])]
    references = [m for i, m in enumerate(references) if m not in references[:i]]
    if not references:
        references = [str(a["StreamOrder"]) for _, l, a in master_tracks if l == language]

    for side, tracks, fps in (("master", master_tracks, master_fp),
                              ("candidate", cand_tracks, cand_fp)):
        for i, (hi, li, ai) in enumerate(tracks):
            for hj, lj, aj in tracks[i + 1:]:
                if li == lj:
                    continue
                sim, windows = pair_similarity(fps.get(str(ai["StreamOrder"])),
                                               fps.get(str(aj["StreamOrder"])), hop)
                if sim is not None and sim >= SAME_CONTENT:
                    tools.log_always(
                        f"repair: {AUDIO_TAG_CONFLICT} within={side} tracks="
                        f"{_label(side, hi, li, ai)},{_label(side, hj, lj, aj)} tags={li}/{lj} "
                        f"similarity={sim:.4f} windows={windows} -- the same content under two "
                        f"language tags: one tag is false, for {path}\n")

    rows = []
    for hc, lc, ac in cand_tracks:
        for hm, lm, am in master_tracks:
            sim, windows = pair_similarity(master_fp.get(str(am["StreamOrder"])),
                                           cand_fp.get(str(ac["StreamOrder"])), hop)
            rows.append({"candidate": str(ac["StreamOrder"]), "candidate_tag": lc,
                         "candidate_holder": hc, "master": str(am["StreamOrder"]),
                         "master_tag": lm, "similarity": sim, "windows": windows,
                         "reference": str(am["StreamOrder"]) in references})
    summary = [(f"c#{r['candidate']}({r['candidate_tag']})", f"m#{r['master']}({r['master_tag']})",
                r["similarity"]) for r in rows]
    tools.log_always(f"repair: content_check language={language} references={references} "
                     f"pairs={summary} unmeasured_master={m_un} unmeasured_candidate={c_un} "
                     f"cost_s={round(time.time() - t0, 1)} for {path}\n")
    measured = [r for r in rows if r["reference"] and r["similarity"] is not None]
    matches = sorted([r for r in measured if r["similarity"] >= SAME_CONTENT],
                     key=lambda r: -r["similarity"])
    best = max(measured, key=lambda r: r["similarity"]) if measured else None
    base = {"rows": rows, "references": references, "unmeasured": m_un + c_un,
            "best": best, "cost_s": round(time.time() - t0, 1)}
    kept = primed.get("tag_check_wav")
    if not (matches and kept is not None and kept["stream"] == matches[0]["candidate"]
            and matches[0]["candidate_tag"] != language):
        drop_kept_wav(primed)
    if not measured:
        return dict(base, verdict="unmeasured",
                    reason="no candidate track could be compared with the master's "
                           f"{language} tracks {references} (unmeasured: {m_un + c_un})")
    if any(r["candidate_tag"] == language for r in matches):
        return dict(base, verdict="tags_consistent",
                    reason="a candidate track tagged " + language + " matches the reference: "
                           "the tags tell the truth, the low similarity has another cause")
    if matches:
        hit = matches[0]
        moves = _moves_for(candidate_obj, language, hit, cand_fp, references, rows, hop)
        return dict(base, verdict=AUDIO_TAG_CONFLICT, track=hit["candidate"],
                    tag=hit["candidate_tag"], holder=hit["candidate_holder"],
                    master_stream=hit["master"], similarity=hit["similarity"], moves=moves,
                    reason=(f"the candidate track {hit['candidate']} tagged "
                            f"{hit['candidate_tag']} carries the master's {language} content "
                            f"(master track {hit['master']}, similarity "
                            f"{hit['similarity']:.4f}): the tag lies"))
    return dict(base, verdict=NO_COMMON_LANGUAGE,
                reason=(f"no candidate track matches the master's {language} tracks "
                        f"{references}: best similarity {best['similarity']:.4f} "
                        f"(candidate #{best['candidate']} tagged {best['candidate_tag']} "
                        f"vs master #{best['master']}), under the {SAME_CONTENT} of "
                        f"same content -- the pair has no common language"))


def drop_kept_wav(primed):
    """Delete the WAV kept in `primed["tag_check_wav"]`, if any."""
    kept = primed.pop("tag_check_wav", None)
    if kept is not None:
        _remove(kept["path"])


def _moves_for(candidate_obj, language, hit, cand_fp, references, rows, hop):
    """List the re-tagging moves for an `audio_tag_conflict`.

    The matched track moves to `language`; each other `language` track that matches no
    reference moves to the language of the master or candidate track it duplicates, else 'und'."""
    moves = [{"stream": hit["candidate"], "from": hit["candidate_tag"], "to": language,
              "holder": hit["candidate_holder"], "similarity": hit["similarity"],
              "matched": f"master#{hit['master']}"}]
    matched_any = {r["candidate"] for r in rows
                   if r["reference"] and r["similarity"] is not None
                   and r["similarity"] >= SAME_CONTENT}
    tracks = tracks_of(candidate_obj)
    for holder, lang, audio in tracks:
        order = str(audio["StreamOrder"])
        if lang != language or order in matched_any:
            continue
        to, sim_to, matched = "und", None, None
        for r in rows:
            if (r["candidate"] == order and r["master_tag"] != language
                    and r["similarity"] is not None and r["similarity"] >= SAME_CONTENT
                    and (sim_to is None or r["similarity"] > sim_to)):
                to, sim_to, matched = r["master_tag"], r["similarity"], f"master#{r['master']}"
        for h2, l2, a2 in ([] if matched else tracks):
            if l2 == language or str(a2["StreamOrder"]) == order:
                continue
            sim, _ = pair_similarity(cand_fp.get(order), cand_fp.get(str(a2["StreamOrder"])), hop)
            if sim is not None and sim >= SAME_CONTENT and (sim_to is None or sim > sim_to):
                to, sim_to = l2, sim
                matched = f"candidate#{a2['StreamOrder']}"
        moves.append({"stream": order, "from": lang, "to": to, "holder": holder,
                      "similarity": sim_to, "matched": matched})
    return moves


def apply_correction(candidate_obj, check):
    """Apply the re-tagging moves to the candidate's audio holders.

    Each moved track is marked `VMSAM_tag_corrected` = '<from>-><to> similarity=<x> matched=<m>'."""
    for move in check["moves"]:
        holder = getattr(candidate_obj, move["holder"])
        found = None
        for audio in holder.get(move["from"], []):
            if str(audio["StreamOrder"]) == move["stream"]:
                found = audio
        if found is None:
            continue
        holder[move["from"]].remove(found)
        if not holder[move["from"]]:
            del holder[move["from"]]
        sim = move["similarity"]
        found["VMSAM_tag_corrected"] = (f"{move['from']}->{move['to']} similarity="
                                        f"{'unmeasured' if sim is None else f'{sim:.4f}'}"
                                        f" matched={move['matched']}")
        holder.setdefault(move["to"], []).append(found)
        tools.log_always(f"repair: tag_corrected stream={move['stream']} from={move['from']} "
                         f"to={move['to']} similarity={sim} matched={move['matched']} for "
                         f"{candidate_obj.filePath}\n")
    candidate_obj.tag_checked = True
