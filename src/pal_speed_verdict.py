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

M3 (RULING_20260921_NTSC_KNIFE_EDGE.MD; VMSAM_HELP_AI/dev-pal/
012-ntsc-knife-edge.MD): duration cannot carry the NTSC signal -- a real cut
confounds it by 40-60x its own size, measured on 8 real production episodes
that duration routes to `pal_inverse` (4/8, coincidentally inside the PAL
band) or `no_band` (4/8), never anywhere near NTSC. So the NTSC recognizer
here is Stage-3-shaped and UNCONDITIONAL: it runs regardless of what the
duration band guessed, on the reasoning that a duration-band guess pitch
already refused must not consume the pair's only chance at the rate family.
TWO DURATION-LAYER BEHAVIOURS CHANGED, both stated where they used to be
unconditional assertions:
  - `no_band` no longer asserts `wrong_content_suspected` before pitch has
    actually looked for a rate relation and found none.
  - a `pal_direct`/`pal_inverse` guess that pitch refuses gets a second,
    independent attempt at NTSC before the chain declines.
EVERYTHING ELSE IS UNCHANGED: `band is None`, `band == "near_unity"`, and a
genuine PAL match (duration guesses right, pitch agrees) never call
`confirm_ntsc` at all -- verified by a monkeypatch that raises if called.
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


def _confirm_via_ncc(master_path, candidate_path, probe_start_seconds, probe_window_seconds,
                      confirmed_ratio, candidate_video_obj, discriminator_result, pitch_result):
    """Shared tail once SOME confirmer has agreed on `confirmed_ratio` --
    build the undo filter and run the same rate-corrected NCC confirmer PAL
    already uses ("route to the same Stage-3 confirmers", M3's own
    instruction). `confirmed_ratio` is WHATEVER ratio the caller's confirmer
    just measured -- the duration-predicted one for a genuine PAL match, the
    NTSC nominal (or its reciprocal) for an NTSC recovery. NEVER the
    discriminator's own `speed_ratio` for an NTSC recovery: M3 measured that
    value to be unusable there (a real cut confounds it by 40-60x)."""
    sample_rate = _candidate_sample_rate(candidate_video_obj)
    if sample_rate is None:
        return {"verdict": "declined", "cause": "no_rate_relation",
                "reason": "no sampling rate on the candidate: cannot build the undo filter",
                "discriminator": discriminator_result, "pitch": pitch_result}

    import merge_video_resample
    try:
        undo_filter, effective_ratio, _, _ = merge_video_resample.build_speed_filter_chain(
            sample_rate, confirmed_ratio)
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
            "speed_ratio": float(confirmed_ratio), "effective_ratio": float(effective_ratio),
            "undo_filter": undo_filter, "discriminator": discriminator_result,
            "pitch": pitch_result, "ncc": ncc_result}


def determine_speed_verdict(master_video_obj, candidate_video_obj, language,
                             probe_start_seconds, probe_window_seconds=180.0):
    """Full Stage 2+3 for one pair. Returns a dict with `verdict` one of:

      "confirmed"    -- some confirmer agreed; `speed_ratio`/`effective_ratio`/
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

    master_path = master_video_obj.filePath
    candidate_path = candidate_video_obj.filePath

    if band in ("pal_direct", "pal_inverse"):
        predicted_ratio = discriminator_result["speed_ratio"]
        pitch_result = pal_pitch_confirmer.confirm_pitch(
            master_path, candidate_path, probe_start_seconds, float(predicted_ratio),
            window_seconds=probe_window_seconds)
        if pitch_result["refusal"] is None:
            # UNCHANGED PATH: duration guessed right, pitch agrees. Byte-identical
            # to before M3 -- confirm_ntsc is never even imported into this branch.
            return _confirm_via_ncc(master_path, candidate_path, probe_start_seconds,
                                     probe_window_seconds, predicted_ratio,
                                     candidate_video_obj, discriminator_result, pitch_result)

        # M3: a coincidental duration-band guess that pitch refuses must not
        # consume the pair's only chance at the rate family -- measured on 4
        # of the 8 M3 census ids, which land `pal_inverse` by coincidence of
        # where a real cut puts the ratio, never because anything is PAL.
        ntsc_result = pal_pitch_confirmer.confirm_ntsc(
            master_path, candidate_path, probe_start_seconds,
            window_seconds=probe_window_seconds)
        if ntsc_result["refusal"] is None:
            return _confirm_via_ncc(master_path, candidate_path, probe_start_seconds,
                                     probe_window_seconds, ntsc_result["predicted_ratio"],
                                     candidate_video_obj, discriminator_result, ntsc_result)
        # Neither the duration-predicted PAL ratio nor the NTSC nominal
        # pitch-confirmed. `pitch` stays the ORIGINAL PAL attempt (unchanged
        # field, unchanged emitted line); the NTSC attempt is recorded
        # separately for full transparency without touching what the
        # existing wiring reads.
        return {"verdict": "declined", "cause": pitch_result["refusal"],
                "reason": pitch_result["reason"], "discriminator": discriminator_result,
                "pitch": pitch_result, "ntsc_attempt": ntsc_result}

    # band == "no_band": M3's other duration-layer change. "Outside named
    # duration bands" is the fact; `wrong_content_suspected` becomes a
    # verdict only after pitch has actually looked for a rate relation and
    # found none -- never asserted from duration alone, which measured 4 of
    # the 8 M3 census ids into exactly this branch despite a genuine,
    # tightly-measurable NTSC relation buried under a cut.
    ntsc_result = pal_pitch_confirmer.confirm_ntsc(
        master_path, candidate_path, probe_start_seconds,
        window_seconds=probe_window_seconds)
    if ntsc_result["refusal"] is None:
        return _confirm_via_ncc(master_path, candidate_path, probe_start_seconds,
                                 probe_window_seconds, ntsc_result["predicted_ratio"],
                                 candidate_video_obj, discriminator_result, ntsc_result)
    # `pitch` is now populated with the NTSC attempt that found nothing --
    # NEW, on a path that carried no pitch evidence at all before M3. This is
    # the arm that proves the assertion was removed, not the verdict: a
    # genuine content mismatch still declines the SAME cause, with pitch
    # having actually run first.
    return {"verdict": "declined", "cause": "wrong_content_suspected",
            "reason": discriminator_result["hypothesis"],
            "discriminator": discriminator_result, "pitch": ntsc_result}
