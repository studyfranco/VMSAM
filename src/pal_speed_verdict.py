"""
Stage 2+3 orchestration for the PAL/speed family design
(DESIGN_PAL_SPEED_FAMILY_20260916.MD): discriminate on duration, then
confirm with BOTH independent instruments (pitch, rate-corrected log-mel
NCC). Produces a verdict; never applies anything -- Stage 4
(`merge_video_resample.build_resampled_candidate`) is a separate, later call
the caller makes only on a `confirmed` verdict.

Reuses `merge_video_resample.build_speed_filter_chain` for the undo filter
rather than re-deriving the asetrate arithmetic -- that module already
states the convention as an equation (`speed_ratio = duree_maitre /
duree_candidat`) matching `pal_speed_discriminator`'s own convention exactly,
and already handles the integer-`asetrate` rounding correctly (the effective
ratio, not the requested one).
"""

import pal_speed_discriminator
import pal_pitch_confirmer
import pal_rate_corrected_ncc


def _candidate_sample_rate(candidate_video_obj):
    """The candidate's own audio sample rate, read the same way
    `merge_video_repair.get_marker_value_for` already does -- ffprobe's
    field first, MediaInfo's as the fallback."""
    for holder in (candidate_video_obj.audios, candidate_video_obj.commentary,
                   candidate_video_obj.audiodesc):
        for language, tracks in holder.items():
            for audio in tracks:
                rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
                if rate:
                    return int(float(rate))
    return None


def determine_speed_verdict(master_video_obj, candidate_video_obj, language,
                             probe_start_seconds, probe_window_seconds=180.0):
    """Full Stage 2+3 for one pair. Returns a dict with `verdict` one of:

      "confirmed"    -- both confirmers agree; `speed_ratio`/`effective_ratio`/
                        `undo_filter` are ready for Stage 4.
      "out_of_scope" -- near-unity duration ratio: undecidable by duration
                        alone, not this family's problem (design's own blind
                        spot).
      "declined"     -- `cause` names which stage refused and why.

    NEVER forces: any confirmer refusing ends the chain at `declined` with
    ITS OWN named cause, never a downgraded acceptance.
    """
    discriminator_result, disc_err = pal_speed_discriminator.discriminate_from_videos(
        master_video_obj, candidate_video_obj, language)
    if disc_err is not None:
        return {"verdict": "declined", "cause": "locator_module_absent",
                "reason": disc_err, "discriminator": discriminator_result}

    band = discriminator_result["band"]
    if band is None:
        return {"verdict": "declined", "cause": "duration_unmeasurable",
                "reason": discriminator_result["hypothesis"],
                "discriminator": discriminator_result}
    if band == "near_unity":
        return {"verdict": "out_of_scope", "cause": "near_unity",
                "reason": discriminator_result["hypothesis"],
                "discriminator": discriminator_result}
    if band == "no_band":
        return {"verdict": "declined", "cause": "wrong_content_suspected",
                "reason": discriminator_result["hypothesis"],
                "discriminator": discriminator_result}

    predicted_ratio = discriminator_result["speed_ratio"]
    master_path = master_video_obj.filePath
    candidate_path = candidate_video_obj.filePath

    pitch_result = pal_pitch_confirmer.confirm_pitch(
        master_path, candidate_path, probe_start_seconds, float(predicted_ratio),
        window_seconds=probe_window_seconds)
    if pitch_result["refusal"] is not None:
        return {"verdict": "declined", "cause": pitch_result["refusal"],
                "reason": pitch_result["reason"],
                "discriminator": discriminator_result, "pitch": pitch_result}

    sample_rate = _candidate_sample_rate(candidate_video_obj)
    if sample_rate is None:
        return {"verdict": "declined", "cause": "no_rate_relation",
                "reason": "no sampling rate on the candidate: cannot build the undo filter",
                "discriminator": discriminator_result, "pitch": pitch_result}

    import merge_video_resample
    try:
        undo_filter, effective_ratio, _, _ = merge_video_resample.build_speed_filter_chain(
            sample_rate, predicted_ratio)
    except merge_video_resample.resample_error as error:
        return {"verdict": "declined", "cause": "no_rate_relation",
                "reason": f"could not build the undo filter: {error}",
                "discriminator": discriminator_result, "pitch": pitch_result}

    ncc_result = pal_rate_corrected_ncc.confirm_rate_correction(
        master_path, candidate_path, probe_start_seconds, probe_window_seconds,
        undo_filter=undo_filter)
    if ncc_result["refusal"] is not None:
        return {"verdict": "declined", "cause": ncc_result["refusal"],
                "reason": ncc_result["reason"], "discriminator": discriminator_result,
                "pitch": pitch_result, "ncc": ncc_result}

    return {"verdict": "confirmed", "cause": None, "reason": None,
            "speed_ratio": float(predicted_ratio), "effective_ratio": float(effective_ratio),
            "undo_filter": undo_filter, "discriminator": discriminator_result,
            "pitch": pitch_result, "ncc": ncc_result}
