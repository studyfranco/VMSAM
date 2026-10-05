"""Owner-judgment-pending: the shared logging contract for a disputed audio/video zone.

A repair that measures the audio as aligned but finds the video disagreeing does not decide
between them. Two callers raise this: `repair_orchestrator.audio_transitions`, when a
resolved cut's video fill width contradicts the audio's own fill by more than a frame, and
`picture_only_shift.scan_zones`, when a confirmed picture-only-shift run sits inside an
audio-continuous zone. Both log every disputed zone here and decline the repair with this
module's cause; the owner judges case by case.
"""

import tools

CAUSE = "owner_judgment_pending"


def _f(value):
    """Format a float field to 3 decimals, or the literal `None`."""
    return "None" if value is None else f"{float(value):.3f}"


def log_pending(zone_index, reason, master_start_s, master_end_s, candidate_start_s,
                candidate_end_s, audio_cut_s, video_cut_s, picture_shift_ms, frames_compared):
    """Log one disputed zone. Exact format, shared with the ledger maintainer."""
    tools.log_always(
        f"repair: {CAUSE} zone={zone_index} reason={reason} "
        f"master_start_s={_f(master_start_s)} master_end_s={_f(master_end_s)} "
        f"candidate_start_s={_f(candidate_start_s)} candidate_end_s={_f(candidate_end_s)} "
        f"audio_cut_s={_f(audio_cut_s)} video_cut_s={_f(video_cut_s)} "
        f"picture_shift_ms={_f(picture_shift_ms)} frames_compared={int(frames_compared)}\n")


def log_summary(n_zones, lang):
    """Log the one-line summary after every disputed zone of a repair has been logged."""
    tools.log_always(f"repair: {CAUSE}_summary zones={int(n_zones)} lang={lang}\n")
