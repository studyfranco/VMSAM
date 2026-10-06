'''
Rebuild the tracks of a candidate that has no single delay, on the master's timeline.

The module measures nothing: it receives a plan of pieces tiling the master
timeline (built by `repair_orchestrator.apply_plan`, one set per track) and
produces a file. Sign convention:

    candidate_time_ms = master_time_ms + candidate_offset_ms

Master intervals the candidate lacks are filled from the master when it has the
same language, otherwise with silence; another language is never pasted in.

A speed change is applied before the cutting
(`[0:candidate] -> speed_chain -> asplit -> pieces -> concat`), so slice times
are read on the resampled candidate. The produced tracks carry the Matroska tag
`VMSAM_FABRICATED`, which survives the later ffmpeg and mkvmerge passes.
'''

from decimal import Decimal
from fractions import Fraction
from os import path, replace as replace_file
import re
import subprocess
import sys
import time

import tools
import repair_log
from frame_compare import parse_positive_rate
import video

# Audio codecs re-encoded within their own family. A lossy codec never becomes
# lossless: `keep_best_audio` would then prefer the rebuilt track over the
# original it sits beside.
audio_encoder_by_codec = {
    "flac": ("flac", "lossless"),
    "pcm_s16le": ("flac", "lossless"),
    "pcm_s24le": ("flac", "lossless"),
    "pcm_s32le": ("flac", "lossless"),
    "truehd": ("flac", "lossless"),
    "mlp": ("flac", "lossless"),
    "aac": ("aac", "lossy"),
    "ac3": ("ac3", "lossy"),
    "eac3": ("eac3", "lossy"),
    "mp3": ("libmp3lame", "lossy"),
    "opus": ("libopus", "lossy"),
    "vorbis": ("libvorbis", "lossy"),
}

# Samples per frame, by codec, used only to bound how far a track's last frame
# may run past the end of the content. `None`: the codec has no fixed frame size
# (Opus, Vorbis, FLAC, ...), so no tolerance is derived. AAC uses the LC size;
# HE-AAC (2048) then gets a stricter bound, never a looser one.
audio_codec_frame_samples = {
    "aac": 1024,
    "ac3": 1536,
    "eac3": 1536,
    "mp3": 1152,
    "opus": None,
    "vorbis": None,
    "flac": None,
    "truehd": None,
    "mlp": None,
}

# Subtitle codec classification is read from `tools`. Bitmap codecs are refused
# by exclusion, not allow-list, so an unknown codec is attempted and fails loudly
# instead of being dropped silently. Known gap: `dvb_subtitle` and `xsub` are
# bitmap but not listed in `tools`, so they are attempted as text and fail.


class chimeric_error(Exception):
    '''The plan cannot be executed: an explicit refusal, never a fallback.

    `cause` is a stable token set at the raise site (e.g.
    `delivery_offset_exceeds_tolerance`, `candidate_admission_window_exceeded`)
    so callers can classify the refusal without parsing the message; `None`
    when the site sets none.
    '''

    def __init__(self, message, cause=None):
        super().__init__(message)
        self.cause = cause


def _refuse_plan_shape(cause, message, numbers):
    """Log and raise a `chimeric_error` for a plan whose pieces do not tile the master timeline."""
    tools.log_always(f"chimeric: plan_shape_refused cause={cause} {numbers}\n")
    raise chimeric_error(message, cause=cause)


def delay_in_ms(track):
    """mediainfo `Delay` in milliseconds, its unit checked against ffprobe `start_time`.

    mediainfo reports `Delay` in seconds; a build reporting milliseconds would
    be off by 1000. Raises `chimeric_error` when the value only matches ffprobe
    read as milliseconds; without `start_time` the seconds unit is assumed.
    """
    raw = track.get("Delay", 0) or 0
    try:
        seconds = Decimal(str(raw))
    except Exception:
        return Decimal("0")
    probe = (track.get("ffprobe") or {}).get("start_time")
    if probe not in (None, "", "N/A"):
        try:
            start = Decimal(str(probe))
        except Exception:
            return seconds * Decimal("1000")
        # At zero both readings agree, so the check is meaningless.
        if seconds != 0 or start != 0:
            as_seconds = abs(seconds - start)
            as_ms = abs(seconds / Decimal("1000") - start)
            if as_ms < as_seconds:
                raise chimeric_error(
                    f"mediainfo Delay {raw} matches ffprobe start_time {probe} only "
                    f"if it is in MILLISECONDS, not seconds. This module multiplies "
                    f"by 1000 and would be wrong by that factor. THIS IS A STATEMENT "
                    f"ABOUT THE mediainfo BUILD, not about the media")
    return seconds * Decimal("1000")


def preferred_with_disagreement(name, preferred, other, preferred_tool, other_tool):
    """Return `(value, note)`: the preferred tool's value and a note if the other tool disagrees.

    `note` names both values, both tools and the one used; `None` when they
    agree or one is missing. Both tools can be right (coded vs decoded channels
    for HE-AAC v2, header vs decode rate for Opus), so the note does not arbitrate.
    """
    if preferred == None:
        return other, None
    if other == None or str(preferred) == str(other):
        return preferred, None
    return preferred, (f"{name} {preferred_tool}={preferred} "
                       f"{other_tool}={other} used={preferred_tool}")


def encoding_feature_context(audio):
    """Raw encoding-feature fields that explain a tool disagreement, as `key=value` text.

    The disagreement follows the encoding feature, not the codec (e.g. HE-AAC v2
    Parametric Stereo, visible only in mediainfo `Format_AdditionalFeatures`).
    Spaces become `_` to keep the log line space-separated.
    """
    parts = []
    profile = audio.get("ffprobe", {}).get("profile")
    features = audio.get("Format_AdditionalFeatures")
    if profile not in (None, ""):
        parts.append(f"ffprobe_profile={str(profile).replace(' ', '_')}")
    if features not in (None, ""):
        parts.append(f"mediainfo_features={str(features).replace(' ', '_')}")
    return " ".join(parts)


def get_audio_stream_parameters(audio):
    '''Sample rate, channels and layout of the track as ffmpeg names them, plus disagreement notes (empty list if none).'''
    ffprobe_data = audio.get("ffprobe", {})
    sample_rate, rate_note = preferred_with_disagreement(
        "sample_rate", ffprobe_data.get("sample_rate"), audio.get("SamplingRate"),
        "ffprobe", "mediainfo")
    channels, channel_note = preferred_with_disagreement(
        "channels", ffprobe_data.get("channels"), audio.get("Channels"),
        "ffprobe", "mediainfo")
    layout = ffprobe_data.get("channel_layout")
    if sample_rate == None or channels == None:
        raise chimeric_error(
            f"track {audio.get('StreamOrder')} has no sampling rate or channel count")
    if layout == None or layout == "" or "unknown" in str(layout):
        layout = f"{int(channels)}c"
    context = encoding_feature_context(audio)
    notes = [note for note in (rate_note, channel_note) if note != None]
    if context:
        notes = [f"{note} {context}" for note in notes]
    return str(sample_rate), int(channels), str(layout), notes


def get_stream_start_ms(audio):
    """The stream's container `start_time` in ms, as ffprobe (and so ffmpeg) reads it; 0 if unknown."""
    if audio == None:
        return Decimal("0")
    value = (audio.get("ffprobe") or {}).get("start_time")
    if value == None:
        return Decimal("0")
    try:
        return Decimal(str(value)) * Decimal("1000")
    except Exception:
        return Decimal("0")


def build_audio_filtergraph(pieces, candidate_stream_order, master_stream_order,
                            sample_rate, layout, speed_chain=None,
                            candidate_start_ms=None, master_start_ms=None, splices=None):
    '''Build the ffmpeg filtergraph that assembles one audio track from its pieces.

    Each source is read once, forward, and pieces are joined by `concat`; every
    piece gets `aformat` because concat refuses mixed formats.
    Returns (filtergraph, total head padding ms, per-piece head decisions).
    '''
    chains = []
    labels = []
    candidate_pieces = [p for p in pieces if p["source"] == "candidate"]
    master_pieces = [p for p in pieces if p["source"] == "master"]

    # Cut on the sample clock (`asetpts=N/SR/TB+STARTPTS`): plan positions are
    # sample positions, and packet-timestamp jumps would otherwise shift every
    # later piece. The speed chain runs before the cutting.
    candidate_entry = "[cin]"
    chains.append(f"[0:{candidate_stream_order}]asetpts=N/SR/TB+STARTPTS"
                  + (f",{speed_chain}" if speed_chain != None else "") + "[cin]")

    candidate_split = []
    if len(candidate_pieces) > 1:
        candidate_split = [f"cs{i}" for i in range(len(candidate_pieces))]
        chains.append(f"{candidate_entry}asplit={len(candidate_pieces)}"
                      + "".join(f"[{label}]" for label in candidate_split))
    # Master fills are cut on the sample clock too, so packet-timestamp jitter
    # does not move fill edges.
    master_entry = "[min]"
    if master_stream_order != None and len(master_pieces):
        chains.append(f"[1:{master_stream_order}]asetpts=N/SR/TB+STARTPTS{master_entry}")
    master_split = []
    if master_stream_order != None and len(master_pieces) > 1:
        master_split = [f"ms{i}" for i in range(len(master_pieces))]
        chains.append(f"{master_entry}asplit={len(master_pieces)}"
                      + "".join(f"[{label}]" for label in master_split))

    # A stream starting after zero yields a head piece shorter than its slot, which
    # would shift every later piece early; the head is padded with silence.
    pads = []
    # Head outcomes: "unmeasured" (no stream start), "read_past" (stream starts
    # after zero but the plan reads past it), "none" (starts at zero), "padded".
    head_decisions = []

    def head_pad(source_start_ms, stream_start_ms, sink, piece_ms):
        if stream_start_ms == None:
            head_decisions.append({"outcome": "unmeasured",
                                   "stream_start_ms": None, "missing_ms": None})
            return ""
        # The pad never exceeds the piece width.
        missing = min(Decimal(str(stream_start_ms)) - Decimal(str(source_start_ms)),
                      Decimal(str(piece_ms)))
        if missing <= 0:
            head_decisions.append({
                "outcome": "read_past" if Decimal(str(stream_start_ms)) > 0 else "none",
                "stream_start_ms": str(stream_start_ms), "missing_ms": str(missing)})
            return ""
        sink.append(missing)
        head_decisions.append({"outcome": "padded",
                               "stream_start_ms": str(stream_start_ms),
                               "missing_ms": str(missing)})
        return f",adelay={int(missing)}:all=1"

    candidate_index = 0
    master_index = 0
    for i, piece in enumerate(pieces):
        label = f"p{i}"
        width_ms = piece["master_end_ms"] - piece["master_start_ms"]
        duration = width_ms / Decimal("1000")
        if piece["source"] == "candidate":
            start = piece["source_start_ms"] / Decimal("1000")
            end = start + duration
            # Before a crossfaded candidate join, read 10 ms more for the crossfade.
            if ((splices or {}).get(i) or {}).get("fade_right") and \
                    i + 1 < len(pieces) and pieces[i + 1]["source"] == "candidate":
                end += Decimal("0.010")
            if len(candidate_split):
                entry = f"[{candidate_split[candidate_index]}]"
            else:
                entry = candidate_entry
            candidate_index += 1
            chains.append(f"{entry}atrim=start={start:.6f}:end={end:.6f},"
                          f"asetpts=PTS-STARTPTS"
                          f"{head_pad(piece['source_start_ms'], candidate_start_ms, pads, width_ms)},"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        elif piece["source"] == "master" and master_stream_order != None:
            start = piece["source_start_ms"] / Decimal("1000")
            end = start + duration
            # A fill reads 10 ms more on each crossfaded side and carries its own gain.
            splice = (splices or {}).get(i) or {}
            margin = Decimal("0.010")
            if splice.get("fade_left"):
                start -= margin
            if splice.get("fade_right"):
                end += margin
            volume = ""
            if splice.get("gain") is not None:
                import splice_hygiene
                volume = splice_hygiene.volume_filter(splice["gain"], float(end - start))
            if len(master_split):
                entry = f"[{master_split[master_index]}]"
            else:
                entry = master_entry
            master_index += 1
            chains.append(f"{entry}atrim=start={start:.6f}:end={end:.6f},"
                          f"asetpts=PTS-STARTPTS"
                          f"{head_pad(piece['source_start_ms'], master_start_ms, pads, width_ms)}"
                          f"{volume},"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        else:
            chains.append(f"anullsrc=r={sample_rate}:cl={layout},"
                          f"atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS,"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        labels.append(label)

    fades = set()
    for i in range(len(labels) - 1):
        left, right = (splices or {}).get(i) or {}, (splices or {}).get(i + 1) or {}
        if left.get("fade_right") or right.get("fade_left"):
            fades.add(i)
    if not fades:
        chains.append("".join(f"[{label}]" for label in labels)
                      + f"concat=n={len(labels)}:v=0:a=1[aout]")
    else:
        # 10 ms triangular crossfade at marked joins, plain concat elsewhere.
        current = labels[0]
        for i in range(len(labels) - 1):
            joined = "aout" if i == len(labels) - 2 else f"j{i}"
            if i in fades:
                # acrossfade's output timestamps drift from its sample count;
                # `asetpts=N/SR/TB` restates them.
                chains.append(f"[{current}][{labels[i + 1]}]"
                              f"acrossfade=d=0.010:o=1:c1=tri:c2=tri,asetpts=N/SR/TB[{joined}]")
            else:
                chains.append(f"[{current}][{labels[i + 1]}]concat=n=2:v=0:a=1[{joined}]")
            current = joined
    return (";".join(chains), sum(pads) if len(pads) else Decimal("0"),
            head_decisions)


def resolve_source_bitrate(audio, source_path, timeout=120):
    '''Return (bitrate, origin) of the source track.

    Tries metadata, then StreamSize/Duration, then a `-c copy` pass. Raises
    `chimeric_error` rather than let the encoder pick its default, which can
    exceed the source and make `keep_best_audio` prefer the rebuilt track.
    '''
    try:
        bitrate = video.get_bitrate(audio)
        if bitrate != None and str(bitrate).isdigit() and int(bitrate) > 0:
            return int(bitrate), "video.get_bitrate"
    except Exception:
        pass

    # `video.get_bitrate` already covers ffprobe.bit_rate, BitRate and
    # BitRate_Nominal; BitRate_Maximum remains.
    for key in ("BitRate_Maximum",):
        value = audio.get(key)
        if value != None and str(value).isdigit() and int(value) > 0:
            return int(value), f"mediainfo.{key}"

    stream_size = audio.get("StreamSize")
    duration = audio.get("Duration")
    if stream_size != None and duration != None:
        try:
            computed = int(Decimal(str(stream_size)) * 8 / Decimal(str(duration)))
            if computed > 0:
                return computed, "StreamSize/Duration"
        except Exception:
            pass

    # Last route: a `-c copy` pass to null reads the stream size without decoding.
    command = [tools.software["ffmpeg"], "-nostdin", "-hide_banner", "-i", source_path,
               "-map", f"0:{int(audio['StreamOrder'])}", "-c", "copy", "-f", "null", "-"]
    tools.dev_log(f"chimeric: resolve_source_bitrate starting "
                  f"file={source_path} stream_order={audio.get('StreamOrder')}\n")
    try:
        with repair_log.announced("chimeric", "ffmpeg", source_path) as call:
            stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
                command, 1, timeout)
            call["exit"] = exit_code
    except Exception as error:
        raise chimeric_error(
            f"the source bitrate of track {audio.get('StreamOrder')} could not "
            f"be measured: {error}")
    text = stderror.decode("utf-8", errors="ignore")
    match = re.search(r"audio:\s*(\d+)\s*([KMG])iB", text)
    if match != None and duration != None:
        scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
        try:
            measured = int(Decimal(match.group(1)) * scale * 8 / Decimal(str(duration)))
            if measured > 0:
                return measured, "measured by a copy pass"
        except Exception:
            pass

    raise chimeric_error(
        f"the source bitrate of track {audio.get('StreamOrder')} could not be "
        f"determined by any of the four routes: track declined rather than "
        f"encoded at whatever default the encoder picks")


def get_encoder_arguments(audio, codec_name, source_path=None):
    '''Encoder arguments in the source's codec family, never above the source bitrate.

    Returns (arguments, family, bitrate_origin).
    '''
    if codec_name not in audio_encoder_by_codec:
        raise chimeric_error(
            f"no encoder kept for codec {codec_name}: track declined by name")
    encoder, family = audio_encoder_by_codec[codec_name]
    arguments = ["-c:a", encoder]
    bitrate_origin = None
    if family == "lossless":
        if encoder == "flac":
            arguments.extend(["-compression_level", "8"])
    else:
        bitrate, bitrate_origin = resolve_source_bitrate(audio, source_path)
        arguments.extend(["-b:a", str(bitrate)])
    return arguments, family, bitrate_origin


def find_master_audio_for_language(master_obj, language, reference_stream=None):
    '''The master track of `language` that fills the gaps, or None; never a commentary.

    Among several same-language tracks, `reference_stream` (the one the plan
    was measured against) is preferred, since such tracks can be offset from
    each other; otherwise `pick_best_master_audio`.
    '''
    for holder in (master_obj.audios, master_obj.audiodesc):
        tracks = holder.get(language)
        if tracks == None or not len(tracks):
            continue
        if len(tracks) == 1:
            return tracks[0]
        # The reference stream is known to be aligned with the plan; quality
        # ranking is only the fallback.
        if reference_stream != None:
            for track in tracks:
                if str(track.get("StreamOrder")) == str(reference_stream):
                    return track
        return pick_best_master_audio(tracks)
    return None


def same_language_principal_count(master_obj, language):
    """How many principal (non-commentary, non-audiodesc) tracks the master carries in this language.

    More than one means the fill was chosen between tracks (e.g. two regional
    dubs) that the language tag does not separate; the count is logged.
    """
    tracks = (master_obj.audios or {}).get(language) or []
    return len(tracks)


def find_fill_audio(master_obj, language, reference_stream=None,
                    comparison_language=None):
    """Return (track, fill_language): the master track that fills this track's gaps.

    Same language first, else the comparison language, else (None, None) for
    silence. Kept apart from `find_master_audio_for_language`, which the
    verifier also uses and where a cross-language fallback would be meaningless.
    """
    same = find_master_audio_for_language(master_obj, language, reference_stream)
    if same != None:
        return same, language
    if comparison_language in (None, "", language):
        return None, None
    other = find_master_audio_for_language(master_obj, comparison_language,
                                           reference_stream)
    if other != None:
        return other, comparison_language
    return None, None


def pick_best_master_audio(tracks):
    """The best of several same-language tracks, as ranked by `keep_best_audio`.

    `keep_best_audio` mutates its input, so it runs on deep copies and the
    survivor is mapped back by StreamOrder. Falls back to the first track when
    the rules are unavailable, the ranking raises, or it does not leave exactly
    one survivor.
    """
    import copy
    try:
        import mergeVideo
        rules = mergeVideo.decript_merge_rules(tools.mergeRules['audio'])
    except Exception:
        return tracks[0]
    candidates = []
    for track in tracks:
        clone = copy.deepcopy(track)
        clone["keep"] = True
        candidates.append(clone)
    try:
        mergeVideo.keep_best_audio(candidates, rules)
    except Exception:
        return tracks[0]
    survivors = [c for c in candidates if c.get("keep")]
    if len(survivors) != 1:
        return tracks[0]
    order = survivors[0].get("StreamOrder")
    for track in tracks:
        if track.get("StreamOrder") == order:
            return track
    return tracks[0]


def split_master_fill_shortfall(pieces, fill_source_ms):
    '''How far this track's master pieces reach past the fill source's end, split by reason.

    A shortfall from the `tail_gap` piece is exempt from refusal; any other
    reason is not. At most one `tail_gap` piece exists (adjacent master pieces
    are merged). Returns `(non_tail_shortfall_ms, tail_shortfall_ms)`, each a
    `Decimal` or `None` when there is no shortfall of that class.
    '''
    non_tail_ends = [p["master_end_ms"] for p in pieces
                     if p["source"] == "master" and p.get("reason") != "tail_gap"]
    tail_ends = [p["master_end_ms"] for p in pieces
                if p["source"] == "master" and p.get("reason") == "tail_gap"]
    furthest_non_tail = max(non_tail_ends, default=None)
    furthest_tail = max(tail_ends, default=None)
    non_tail_shortfall = (furthest_non_tail - fill_source_ms
                          if furthest_non_tail != None and furthest_non_tail > fill_source_ms
                          else None)
    tail_shortfall = (furthest_tail - fill_source_ms
                      if furthest_tail != None and furthest_tail > fill_source_ms
                      else None)
    return non_tail_shortfall, tail_shortfall


def build_one_audio_track(candidate_obj, master_obj, audio, language, pieces,
                          out_path, timeout, speed_ratio=None,
                          reference_stream=None, comparison_language=None,
                          track_bound_ms=None, speed_engine="asetrate"):
    '''Build one chimeric audio track. Returns a report dict.

    `track_bound_ms` is this track's own extent (not the file's), used as the
    source end for the tail cut.
    '''
    tools.dev_log(f"chimeric: build_one_audio_track starting "
                  f"candidate={candidate_obj.filePath} "
                  f"stream_order={audio.get('StreamOrder')} language={language} "
                  f"out_path={out_path}\n")
    codec_name = audio.get("ffprobe", {}).get("codec_name", "").lower()
    encoder_arguments, family, bitrate_origin = get_encoder_arguments(
        audio, codec_name, candidate_obj.filePath)
    sample_rate, channels, layout, tool_disagreements = get_audio_stream_parameters(audio)

    master_audio, fill_language = find_fill_audio(
        master_obj, language, reference_stream, comparison_language)
    master_stream_order = None
    if master_audio != None:
        master_stream_order = int(master_audio["StreamOrder"])
    needs_master = any(piece["source"] == "master" for piece in pieces)
    fill = "none"
    if needs_master:
        fill = "master" if master_stream_order != None else "silence"
    # The fill title is recorded only: regional variants share one language key,
    # and the title is the only hint left.
    fill_title = master_audio.get("Title") if fill == "master" and master_audio != None else None
    fill_choices = (same_language_principal_count(master_obj, fill_language)
                    if fill == "master" and fill_language else 0)
    # Probe whether the master fill track has sound in the head gap (tracks of one
    # master do not share a duration, so this is measured, not inferred).
    head_piece = next((p for p in pieces
                       if p["source"] == "master" and p["master_start_ms"] == 0), None)
    head_source = None
    if head_piece != None and fill == "master" and master_audio != None:
        span = min(Decimal("20000"),
                   head_piece["master_end_ms"] - head_piece["master_start_ms"])
        try:
            samples = read_mono_samples(
                master_obj.filePath, f"0:{int(master_audio['StreamOrder'])}",
                Decimal("0"), span, verify_probe_rate)
            # The filtergraph takes one master stream for all fills, so a silent
            # head is still taken from it and marked `NO-HEAD(...)`.
            head_source = ("master/" + str(fill_language)
                           if get_rms(samples) >= verify_min_rms
                           else f"NO-HEAD(taken from master/{fill_language} "
                                f"anyway -- fall-through not implemented, "
                                f"this head is silent)")
        except Exception:
            head_source = "unprobed"
    elif head_piece != None:
        head_source = "silence"

    fill_source_ms = None
    fill_short_by_ms = None
    fill_short_by_ms_tail_exempt = None
    if fill == "master" and master_audio != None and "Duration" in master_audio:
        try:
            # The fill's end position on the master timeline is Duration + Delay.
            delay_ms = delay_in_ms(master_audio)
            fill_source_ms = (Decimal(str(master_audio["Duration"])) * Decimal("1000")
                              + delay_ms)
            # Only the non-tail shortfall is refused by `verify_output_file`.
            fill_short_by_ms, fill_short_by_ms_tail_exempt = (
                split_master_fill_shortfall(pieces, fill_source_ms))
            if fill_short_by_ms_tail_exempt != None:
                # Plan-stage prediction; `verify_output_file` confirms on the output.
                tools.log_line(
                    f"chimeric: fill_short_tail_exempt "
                    f"stream_order={audio['StreamOrder']} language={language} "
                    f"exempted_ms={fill_short_by_ms_tail_exempt} reason=tail_gap "
                    f"residual_non_exempt_ms="
                    f"{fill_short_by_ms if fill_short_by_ms != None else '0'}\n")
        except Exception:
            fill_source_ms = None
    # True when the fill track is the stream the plan was measured against,
    # rather than one chosen by quality ranking.
    fill_by_reference = bool(
        fill == "master" and master_audio != None and reference_stream != None
        and str(master_audio.get("StreamOrder")) == str(reference_stream))
    if fill != "master":
        fill_language = None

    speed_chain = None
    applied_ratio = None
    if speed_ratio != None:
        import merge_video_resample
        # asetrate changes speed and pitch; atempo changes tempo only.
        speed_chain, applied_ratio = merge_video_resample.build_transform_chain(
            sample_rate, speed_ratio, speed_engine)
    # After resampling, the candidate's start time scales by the ratio.
    candidate_start_ms = get_stream_start_ms(audio)
    if speed_ratio != None:
        candidate_start_ms = candidate_start_ms * Decimal(str(speed_ratio))
    # Gain and crossfade decisions for every splice, applied in the same graph.
    splices = (plan_splices(candidate_obj, audio, master_obj, master_audio, pieces,
                            speed_chain, speed_ratio)
               if fill == "master" and master_audio != None else {})
    for index, join in plan_candidate_joins(candidate_obj, audio, pieces, speed_chain,
                                            speed_ratio).items():
        splices.setdefault(index, {}).update(join)
    filtergraph, head_pad_ms, head_decisions = build_audio_filtergraph(
        pieces, int(audio["StreamOrder"]), master_stream_order, sample_rate, layout,
        speed_chain, candidate_start_ms, get_stream_start_ms(master_audio), splices)

    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath]
    if master_stream_order != None:
        command.extend(["-i", master_obj.filePath])
    else:
        # Input 1 is kept even when unused so input indices stay fixed.
        command.extend(["-i", master_obj.filePath])
    # `-map_chapters -1`: chapters are added later, retimed, by `mux_chapters`.
    command.extend(["-filter_complex", filtergraph, "-map", "[aout]", "-map_chapters", "-1"])
    command.extend(encoder_arguments)
    command.extend(["-ar", sample_rate, "-vn", "-sn", "-dn",
                    "-max_muxing_queue_size", "16384", out_path])
    tools.dev_log(f"chimeric: build_one_audio_track ffmpeg build call "
                  f"candidate={candidate_obj.filePath} out_path={out_path}\n")
    with repair_log.announced("chimeric", "ffmpeg", candidate_obj.filePath) as call:
        tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
        call["exit"] = 0

    bitrate = None
    if "-b:a" in encoder_arguments:
        bitrate = encoder_arguments[encoder_arguments.index("-b:a") + 1]
    # Per-region report: master-filled regions, cut candidate regions, and used
    # candidate regions with their offset.
    filled = Decimal("0")
    filled_regions = []
    cut_regions = []
    used_regions = []
    previous_candidate_end = None
    for piece in pieces:
        if piece["source"] == "master":
            filled += piece["master_end_ms"] - piece["master_start_ms"]
            filled_regions.append({
                "master_start_ms": str(piece["master_start_ms"]),
                "master_end_ms": str(piece["master_end_ms"]),
                "source_start_ms": str(piece.get("source_start_ms")),
                "reason": piece.get("reason"),
                "source": "silence" if fill == "silence" else "master",
                "language": None if fill == "silence" else fill_language,
                "fill_source_class": (
                    "silence" if fill == "silence"
                    else "same_language_master" if fill_language == language
                    else "comparison_language_master"),
                # Frame-tier decline reason and evidence; None unless it declined.
                "frame_tier_declined_reason": (
                    piece["frame_tier"]["reason"]
                    if piece.get("frame_tier") and piece["frame_tier"].get("declined")
                    else None),
                "frame_tier_declined_evidence": (
                    piece["frame_tier"].get("evidence")
                    if piece.get("frame_tier") and piece["frame_tier"].get("declined")
                    else None),
                # None when no interior frame tier ran on this piece.
                "frame_tier_similarity": (
                    piece["frame_tier"].get("similarity")
                    if piece.get("frame_tier") else None),
                "frame_tier_margin": (
                    piece["frame_tier"].get("margin")
                    if piece.get("frame_tier") else None)})
            continue
        if piece["source"] == "candidate":
            source_start = piece["source_start_ms"]
            source_end = source_start + (piece["master_end_ms"]
                                         - piece["master_start_ms"])
            if previous_candidate_end == None and source_start > 0:
                cut_regions.append({
                    "candidate_start_ms": "0",
                    "candidate_end_ms": str(source_start),
                    "dropped_ms": str(source_start), "where": "head"})
            if previous_candidate_end != None and source_start > previous_candidate_end:
                cut_regions.append({
                    "candidate_start_ms": str(previous_candidate_end),
                    "candidate_end_ms": str(source_start),
                    "dropped_ms": str(source_start - previous_candidate_end),
                    "where": "interior"})
            used_regions.append({
                "master_start_ms": str(piece["master_start_ms"]),
                "master_end_ms": str(piece["master_end_ms"]),
                "candidate_start_ms": str(source_start),
                "candidate_end_ms": str(source_end),
                "offset_ms": str(source_start - piece["master_start_ms"])})
            previous_candidate_end = source_end
    if previous_candidate_end != None and track_bound_ms != None:
        tail = Decimal(str(track_bound_ms)) - previous_candidate_end
        if tail > 0:
            cut_regions.append({
                "candidate_start_ms": str(previous_candidate_end),
                "candidate_end_ms": str(track_bound_ms),
                "dropped_ms": str(tail), "where": "tail"})
    elif previous_candidate_end != None and track_bound_ms == None:
        cut_regions.append({"candidate_start_ms": str(previous_candidate_end),
                            "candidate_end_ms": None, "dropped_ms": None,
                            "where": "tail", "unmeasured": True})
    total = pieces[-1]["master_end_ms"] - pieces[0]["master_start_ms"]
    silence_ms = filled if fill == "silence" else Decimal("0")
    return {"stream_order": int(audio["StreamOrder"]), "language": language,
            "tag_corrected": audio.get("VMSAM_tag_corrected"),
            "tool_disagreements": tool_disagreements,
            "codec": codec_name, "encoder": encoder_arguments[1],
            "family": family, "gap_fill": fill, "fill_language": fill_language,
            "fill_title": fill_title, "fill_choices": fill_choices,
            "head_source": head_source,
            "fill_stream_order": master_stream_order,
            "fill_source_ms": str(fill_source_ms) if fill_source_ms != None else None,
            "fill_short_by_ms": str(fill_short_by_ms) if fill_short_by_ms != None else None,
            "fill_short_by_ms_tail_exempt": (
                str(fill_short_by_ms_tail_exempt)
                if fill_short_by_ms_tail_exempt != None else None),
            "fill_by_reference": fill_by_reference,
            "path": out_path,
            "bitrate": bitrate, "bitrate_origin": bitrate_origin,
            "speed_ratio_requested": str(speed_ratio) if speed_ratio != None else None,
            "speed_ratio_applied": str(applied_ratio) if applied_ratio != None else None,
            "gap_filled_ms": str(filled),
            "filled_regions": filled_regions, "cut_regions": cut_regions,
            "splices": {str(k): v for k, v in splices.items()},
            "used_regions": used_regions,
            # Silence added at the head because the source does not start at zero.
            "head_pad_ms": str(head_pad_ms),
            "head_decisions": head_decisions,
            "silence_filled_ms": str(silence_ms),
            "silence_fraction": str((silence_ms / total).quantize(Decimal("0.000001")))
                                if total > 0 else "0",
            "title": audio.get("Title"), "kind": None}


def classify_subtitle(codec_name):
    """Classify a subtitle codec as "bitmap", "srt" or "ass", using the sets in `tools`."""
    name = (codec_name or "").lower()
    if name in tools.sub_type_not_encodable:
        return "bitmap"
    if name in tools.sub_type_near_srt:
        return "srt"
    return "ass"


def retime_subtitle_file(subtitle_path, pieces, speed_ratio=None):
    '''Rewrite a subtitle file's cue timestamps onto the master timeline, in place.

    Each cue is mapped through every candidate piece it overlaps and keeps only
    those parts. Parts that meet on the master timeline stay one cue; parts
    separated by a master piece become two (`split_across_master_span`); a
    removed part is logged (`clipped_to_kept_span`). A cue overlapping no
    candidate piece is dropped (`dropped_master_filled_span` or
    `dropped_no_offset_evidence`); no offset is borrowed from a neighbour. A cue
    ending past the timeline end is clamped (`clamped_to_timeline_end`), since a
    Matroska segment lasts until its last block, subtitles included.

    Returns:
        (kept, dropped, applied_shifts, decisions). Each decision has outcome,
        cue_count, source_start_ms, source_end_ms, shift_ms, piece_reason, gap_ms;
        consecutive drops of one kind are grouped.
    '''
    import pysubs2
    subtitles = pysubs2.load(subtitle_path)
    if speed_ratio != None:
        # Speed first, then cutting, in the same order as the audio.
        import merge_video_resample
        merge_video_resample.retime_subtitle_events_by_ratio(subtitles, speed_ratio)
    candidate_pieces = [p for p in pieces if p["source"] == "candidate"]
    piece_index = {id(p): i for i, p in enumerate(pieces)}
    # The plan's pieces end exactly at the master duration.
    timeline_end_ms = max(Decimal(str(p["master_end_ms"])) for p in pieces)

    def bordering_master(piece, direction):
        '''The master piece just before (-1) or after (+1) `piece`, else None.'''
        i = piece_index[id(piece)] + direction
        if 0 <= i < len(pieces) and pieces[i]["source"] == "master":
            return pieces[i]
        return None

    def find_gap_piece(event_start):
        '''The master piece an uncovered source time falls in, used only to name the drop.'''
        if not candidate_pieces:
            return None
        prev_piece, next_piece = None, None
        for p in candidate_pieces:
            source_start = p["source_start_ms"]
            source_end = source_start + (p["master_end_ms"] - p["master_start_ms"])
            if source_end <= event_start:
                prev_piece = p
            elif next_piece is None and source_start > event_start:
                next_piece = p
        if next_piece is not None:
            return bordering_master(next_piece, -1)
        if prev_piece is not None:
            return bordering_master(prev_piece, +1)
        return None

    kept_events = []
    kept_meta = {}
    dropped_master_filled_span = 0
    dropped_no_offset_evidence = 0
    dropped_degenerate_duration = 0
    applied = {}
    decisions = []
    pending_span = None

    def flush_span():
        nonlocal pending_span
        if pending_span is not None:
            pending_span.pop("_key")
            decisions.append(pending_span)
            pending_span = None

    def record(outcome, original_start, original_end, shift, master_piece):
        nonlocal pending_span
        span_key = (outcome, str(shift), id(master_piece) if master_piece else None)
        if pending_span is not None and pending_span["_key"] == span_key:
            pending_span["cue_count"] += 1
            pending_span["source_end_ms"] = str(original_end)
        else:
            flush_span()
            pending_span = {
                "_key": span_key, "outcome": outcome, "cue_count": 1,
                "source_start_ms": str(original_start),
                "source_end_ms": str(original_end),
                "shift_ms": str(shift) if shift is not None else None,
                "piece_reason": master_piece.get("reason") if master_piece else None,
                "gap_ms": (str(master_piece["master_end_ms"] - master_piece["master_start_ms"])
                          if master_piece else None)}

    def kept_spans(start, end):
        '''Kept parts of cue [start, end) as `(piece, master_start, master_end, shift)`, in source order.'''
        spans = []
        for piece in candidate_pieces:
            source_start = piece["source_start_ms"]
            source_end = source_start + (piece["master_end_ms"] - piece["master_start_ms"])
            piece_shift = piece["master_start_ms"] - source_start
            if end <= start:
                if source_start <= start < source_end:
                    spans.append((piece, start + piece_shift, end + piece_shift, piece_shift))
                continue
            lo, hi = max(start, source_start), min(end, source_end)
            if hi > lo:
                spans.append((piece, lo + piece_shift, hi + piece_shift, piece_shift))
        return sorted(spans, key=lambda span: span[1] - span[3])

    def master_groups(spans):
        '''Merge kept parts that meet on the master timeline (within 1 ms) into one cue.'''
        groups = []
        for span in spans:
            if groups and abs(span[1] - groups[-1][2]) <= 1:
                groups[-1] = (groups[-1][0], groups[-1][1], max(groups[-1][2], span[2]),
                              groups[-1][3])
            else:
                groups.append(span)
        return groups

    def keep_event(event, piece, original_start, original_end, shift):
        '''Clamp a retimed cue to the timeline end, drop it if degenerate, else keep it.'''
        nonlocal dropped_degenerate_duration
        if Decimal(str(event.end)) > timeline_end_ms:
            # Clamp before the degenerate guard, so a cue clamped to zero length
            # is then dropped.
            overhang_ms = Decimal(str(event.end)) - timeline_end_ms
            flush_span()
            decisions.append({
                "outcome": "clamped_to_timeline_end", "cue_count": 1,
                "source_start_ms": str(original_start),
                "source_end_ms": str(original_end),
                "shift_ms": str(shift),
                "piece_reason": piece.get("reason"),
                "gap_ms": str(overhang_ms)})
            event.end = int(timeline_end_ms)
        if event.end <= event.start:
            dropped_degenerate_duration += 1
            record("dropped_degenerate_duration", original_start, original_end, shift, None)
            return
        flush_span()  # drop spans must not merge across a kept cue
        kept_events.append(event)
        kept_meta[id(event)] = (piece, original_start, original_end, shift)

    for event in subtitles.events:
        original_start, original_end = event.start, event.end
        shift = None
        matched_piece = None
        spans = kept_spans(Decimal(str(event.start)), Decimal(str(event.end)))
        if spans:
            matched_piece, shift = spans[0][0], spans[0][3]
        if shift == None:
            gap_piece = find_gap_piece(Decimal(str(event.start)))
            if gap_piece is not None:
                dropped_master_filled_span += 1
                record("dropped_master_filled_span", original_start, original_end, None, gap_piece)
            else:
                dropped_no_offset_evidence += 1
                record("dropped_no_offset_evidence", original_start, original_end, None, None)
            continue
        applied[str(shift)] = applied.get(str(shift), 0) + 1
        groups = master_groups(spans)
        clipped_ms = (Decimal(str(original_end)) - Decimal(str(original_start))
                      - sum(span[2] - span[1] for span in spans))
        if clipped_ms > 0 or len(groups) > 1:
            flush_span()
        if clipped_ms > 0:
            decisions.append({
                "outcome": "clipped_to_kept_span", "cue_count": 1,
                "source_start_ms": str(original_start), "source_end_ms": str(original_end),
                "shift_ms": str(shift), "piece_reason": matched_piece.get("reason"),
                "gap_ms": str(clipped_ms)})
        if len(groups) > 1:
            # gap_ms: the master span between the parts.
            decisions.append({
                "outcome": "split_across_master_span", "cue_count": len(groups),
                "source_start_ms": str(original_start), "source_end_ms": str(original_end),
                "shift_ms": str(shift), "piece_reason": matched_piece.get("reason"),
                "gap_ms": str(sum(groups[k + 1][1] - groups[k][2]
                                  for k in range(len(groups) - 1)))})
        for rank, (group_piece, master_start, master_end, group_shift) in enumerate(groups):
            if rank:
                event = event.copy()
            event.start = int(master_start)
            event.end = int(master_end)
            keep_event(event, group_piece, original_start, original_end, group_shift)
    flush_span()
    kept_events, hygiene = splice_cue_hygiene(kept_events, kept_meta)
    decisions.extend(hygiene)
    dropped_at_splice = sum(1 for d in hygiene if d["outcome"].startswith("dropped"))
    subtitles.events = kept_events
    subtitles.save(subtitle_path)
    return (len(kept_events),
            dropped_master_filled_span + dropped_no_offset_evidence + dropped_degenerate_duration
            + dropped_at_splice,
            applied, decisions)


# Cues compared on each side of a splice to find a duplicate: a line read twice
# across a splice lands within a few neighbours.
SPLICE_DEDUP_NEIGHBOURS = 3


def normalized_cue_text(event):
    """A cue's plain text, lower-cased, without punctuation and with collapsed whitespace."""
    import unicodedata
    text = event.plaintext.lower()
    kept = "".join(ch if not unicodedata.category(ch).startswith("P") else " "
                   for ch in text)
    return " ".join(kept.split())


def splice_cue_hygiene(events, meta):
    """Remove duplicates and overlaps between cues that come from different plan pieces.

    A cue whose normalised text equals a nearby cue from another piece and
    touches it in time is dropped (`dropped_duplicate_at_splice`); a cue running
    into the next cue from another piece is truncated
    (`truncated_before_next_cue`) or dropped if nothing remains. Cues from the
    same piece keep their source relationship.

    Args:
        events: retimed cues.
        meta: `id(event) -> (piece, source_start, source_end, shift)`.

    Returns:
        (kept cues in original order, decisions).
    """
    order = sorted(range(len(events)), key=lambda i: (events[i].start, events[i].end))
    dropped = set()
    decisions = []

    def decide(outcome, index, gap_ms):
        piece, source_start, source_end, shift = meta[id(events[index])]
        decisions.append({
            "outcome": outcome, "cue_count": 1,
            "source_start_ms": str(source_start), "source_end_ms": str(source_end),
            "shift_ms": str(shift), "piece_reason": (piece or {}).get("reason"),
            "gap_ms": str(gap_ms)})

    def piece_of(index):
        return id(meta[id(events[index])][0])

    for rank, i in enumerate(order):
        if i in dropped:
            continue
        text = normalized_cue_text(events[i])
        if not text:
            continue
        seen = 0
        for j in order[rank + 1:]:
            if j in dropped:
                continue
            seen += 1
            if seen > SPLICE_DEDUP_NEIGHBOURS:
                break
            if piece_of(j) == piece_of(i):
                continue
            if (events[j].start <= events[i].end
                    and normalized_cue_text(events[j]) == text):
                dropped.add(j)
                decide("dropped_duplicate_at_splice", j, events[i].end - events[j].start)

    for rank, i in enumerate(order):
        if i in dropped:
            continue
        following = next((j for j in order[rank + 1:]
                          if j not in dropped and piece_of(j) != piece_of(i)
                          and events[j].start >= events[i].start), None)
        if following is None or events[i].end <= events[following].start:
            continue
        overlap = events[i].end - events[following].start
        events[i].end = events[following].start
        if events[i].end <= events[i].start:
            dropped.add(i)
            decide("dropped_degenerate_after_truncation", i, overlap)
        else:
            decide("truncated_before_next_cue", i, overlap)
    return [event for index, event in enumerate(events) if index not in dropped], decisions


def build_one_subtitle_track(candidate_obj, subtitle, language, pieces, work_dir,
                             index, timeout, speed_ratio=None):
    '''Extract and retime one subtitle track; return its report dict or raise `chimeric_error`.'''
    tools.dev_log(f"chimeric: build_one_subtitle_track starting "
                  f"candidate={candidate_obj.filePath} "
                  f"stream_order={subtitle.get('StreamOrder')} language={language} "
                  f"work_dir={work_dir}\n")
    codec_name = subtitle.get("ffprobe", {}).get("codec_name", "").lower()
    target = classify_subtitle(codec_name)
    if target == "bitmap":
        raise chimeric_error(
            f"codec {codec_name} is a bitmap subtitle: its timestamps live "
            f"inside binary segments, cue rewriting cannot reach them")

    out_path = path.join(work_dir, f"sub_{index}.{target}")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath,
               "-map", f"0:{int(subtitle['StreamOrder'])}", "-map_chapters", "-1",
               "-c:s", target, out_path]
    tools.dev_log(f"chimeric: build_one_subtitle_track ffmpeg extract call "
                  f"candidate={candidate_obj.filePath} out_path={out_path}\n")
    with repair_log.announced("chimeric", "ffmpeg", candidate_obj.filePath) as call:
        tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
        call["exit"] = 0
    if not path.getsize(out_path):
        # Empty at the source; checked before pysubs2, which cannot load an empty file.
        raise chimeric_error(
            "the source subtitle track carries no cue at all: nothing was "
            "dropped, there was never anything to drop")
    kept, dropped, shifts_applied, decisions = retime_subtitle_file(out_path, pieces, speed_ratio)
    for decision in decisions:
        tools.log_line(
            f"chimeric: subtitle stream_order={subtitle['StreamOrder']} "
            f"language={language} decision={decision['outcome']} "
            f"cue_count={decision['cue_count']} "
            f"source_span_ms=[{decision['source_start_ms']},{decision['source_end_ms']}) "
            f"shift_ms={decision['shift_ms']} piece_reason={decision['piece_reason']} "
            f"gap_ms={decision['gap_ms']}\n")
    if not kept:
        # An empty .srt would make the final mux fail; refuse the track here.
        raise chimeric_error(
            f"every cue fell outside the pieces kept from the candidate "
            f"({dropped} dropped, 0 kept): the track has no content on the "
            f"master timeline")
    return {"stream_order": int(subtitle["StreamOrder"]), "language": language,
            "codec": codec_name, "format": target, "path": out_path,
            "kept_cues": kept, "dropped_cues": dropped,
            "shifts_applied_ms": shifts_applied,
            "subtitle_decisions": decisions,
            "title": subtitle.get("Title")}


def log_prediction_outcome(predicted, would_refuse):
    """Log whether the predicted refusals matched the gate's verdict.

    Logs `held`, `BROKEN(...)` in either direction, or `unknown` when the gate
    reached no verdict.
    """
    if would_refuse == None:
        verdict = "unknown(no verdict reached)"
    elif bool(predicted) == bool(would_refuse):
        verdict = "held"
    elif predicted:
        verdict = "BROKEN(predicted a refusal, the gate passed it)"
    else:
        verdict = "BROKEN(no refusal predicted, the gate refused)"
    tools.log_line(f"repair: prediction predicted={len(predicted)} "
                      f"would_refuse={would_refuse} agreement={verdict}\n")


def mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout, job_start_utc, chapters_path=None):
    '''Mux the produced tracks and set the VMSAM_FABRICATED, VMSAM_ERA and VMSAM tags.

    This is the only mux site of the repair path, so every repaired file carries
    the tags. VMSAM_ERA records the git commit and `job_start_utc`.
    '''
    tools.dev_log(f"chimeric: mux_repaired_file starting out_path={out_path} "
                  f"n_audio={len(audio_reports)} n_subtitle={len(subtitle_reports)} "
                  f"chapters={chapters_path}\n")
    command = [tools.software["ffmpeg"], "-y", "-nostdin"]
    for report in audio_reports:
        command.extend(["-i", report["path"]])
    for report in subtitle_reports:
        command.extend(["-i", report["path"]])

    for i in range(len(audio_reports) + len(subtitle_reports)):
        command.extend(["-map", f"{i}:0"])
    command.extend(["-map_chapters", "-1", "-c", "copy"])

    era_value = f"git_commit={tools.get_git_commit()} job_start_utc={job_start_utc}"
    build = repair_log.build_sha()
    # VMSAM_FABRICATED is never empty: ffmpeg drops an empty tag, and the track
    # would then read as intact.
    for i, report in enumerate(audio_reports):
        command.extend([f"-metadata:s:a:{i}",
                        f"VMSAM_FABRICATED={report.get('marker') or marker_value or REBUILT_MARKER}"])
        command.extend([f"-metadata:s:a:{i}", f"VMSAM_ERA={era_value}"])
        command.extend([f"-metadata:s:a:{i}", f"VMSAM={build}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:a:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:a:{i}", f"title={report['title']}"])
        if report.get("tag_corrected"):
            command.extend([f"-metadata:s:a:{i}",
                            f"VMSAM_tag_corrected={report['tag_corrected']}"])
    for i, report in enumerate(subtitle_reports):
        command.extend([f"-metadata:s:s:{i}",
                        f"VMSAM_FABRICATED={report.get('marker') or marker_value or REBUILT_MARKER}"])
        command.extend([f"-metadata:s:s:{i}", f"VMSAM_ERA={era_value}"])
        command.extend([f"-metadata:s:s:{i}", f"VMSAM={build}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:s:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:s:{i}", f"title={report['title']}"])

    command.extend(["-max_muxing_queue_size", "16384", out_path])
    tools.dev_log(f"chimeric: mux_repaired_file ffmpeg mux call "
                  f"out_path={out_path}\n")
    with repair_log.announced("chimeric", "ffmpeg", out_path) as call:
        tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
        call["exit"] = 0
    if chapters_path is not None:
        mux_chapters(out_path, chapters_path, timeout)


def mux_chapters(out_path, chapters_path, timeout):
    '''Set a chapters XML on the produced file with a `mkvmerge --chapters` copy remux.

    mkvmerge exit 1 is a warning and counts as success; any other failure raises
    `chimeric_error`.
    '''
    remuxed = out_path + ".chapters.mkv"
    command = [tools.software["mkvmerge"], "-q", "-o", remuxed,
               "--chapters", chapters_path, out_path]
    tools.dev_log(f"chimeric: mux_chapters mkvmerge call out_path={out_path} "
                  f"chapters={chapters_path}\n")
    with repair_log.announced("chimeric", "mkvmerge", out_path) as call:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        call["exit"] = completed.returncode
    if completed.returncode not in (0, 1) or not path.exists(remuxed):
        raise chimeric_error(
            f"mkvmerge could not set the chapters on the produced file (exit "
            f"{completed.returncode}): {(completed.stdout + completed.stderr).strip()[-300:]}")
    replace_file(remuxed, out_path)


def extract_chapters_xml(file_path, out_path, timeout=120):
    """Extract a file's chapters with mkvextract.

    Returns:
        (ElementTree root or None, reason). reason is `extracted`, `no_chapters`,
        `mkvextract_not_configured`, `mkvextract_exit_<n>`, `timeout` or
        `unparseable(<type>)`.
    """
    import xml.etree.ElementTree as ElementTree
    tool = tools.software.get("mkvextract")
    if not tool:
        return None, "mkvextract_not_configured"
    command = [tool, file_path, "chapters", out_path]
    tools.dev_log(f"chimeric: extract_chapters_xml mkvextract call file={file_path} "
                  f"out={out_path}\n")
    try:
        with repair_log.announced("chimeric", "mkvextract", file_path) as call:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=timeout)
            call["exit"] = completed.returncode
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if completed.returncode not in (0, 1):
        return None, f"mkvextract_exit_{completed.returncode}"
    if not path.exists(out_path) or not path.getsize(out_path):
        return None, "no_chapters"
    try:
        root = ElementTree.parse(out_path).getroot()
    except Exception as error:
        return None, f"unparseable({type(error).__name__})"
    if root.find("EditionEntry") is None:
        return None, "no_chapters"
    return root, "extracted"


CHAPTER_TIME_PATTERN = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$")


def parse_chapter_time_ms(text):
    """Parse `HH:MM:SS.nnnnnnnnn` into exact milliseconds (Decimal), or None."""
    match = CHAPTER_TIME_PATTERN.match(text or "")
    if match is None:
        return None
    hours, minutes, seconds, fraction = match.groups()
    nanoseconds = int((fraction or "0").ljust(9, "0"))
    return (Decimal(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * Decimal(1000)
            + Decimal(nanoseconds) / Decimal(1000000))


def format_chapter_time(ms):
    """Format milliseconds as a Matroska `HH:MM:SS.nnnnnnnnn` time."""
    nanoseconds = int((Decimal(str(ms)) * Decimal(1000000)).to_integral_value())
    seconds, nanoseconds = divmod(max(0, nanoseconds), 1000000000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{nanoseconds:09d}"


def map_candidate_time_to_master(equivalent_ms, candidate_pieces):
    """Map a candidate time onto the master timeline through the plan's pieces.

    Returns:
        (master_ms or None, how): `mapped` (inside a piece),
        `snapped_to_next_piece` (inside cut content: start of the next piece) or
        `past_last_piece` (None).
    """
    following = None
    for piece in candidate_pieces:
        start = Decimal(str(piece["source_start_ms"]))
        length = Decimal(str(piece["master_end_ms"])) - Decimal(str(piece["master_start_ms"]))
        if start <= equivalent_ms < start + length:
            return (Decimal(str(piece["master_start_ms"])) + (equivalent_ms - start),
                    "mapped")
        if start > equivalent_ms and (following is None
                                      or start < Decimal(str(following["source_start_ms"]))):
            following = piece
    if following is not None:
        return Decimal(str(following["master_start_ms"])), "snapped_to_next_piece"
    return None, "past_last_piece"


def build_delivered_chapters(master_path, candidate_path, reference_pieces, time_scale,
                             timeline_ms, work_dir):
    """Build the produced file's chapters XML.

    Master editions are kept as is. Candidate editions are retimed (speed factor,
    then `map_candidate_time_to_master`, clamped to the timeline end) and lose
    their UIDs so mkvmerge assigns fresh ones; ordered editions are not delivered.

    Returns:
        (xml_path or None when no chapters remain, decisions).
    """
    import xml.etree.ElementTree as ElementTree
    decisions = []
    master_root, master_reason = extract_chapters_xml(
        master_path, path.join(work_dir, "chapters_master.xml"))
    candidate_root, candidate_reason = extract_chapters_xml(
        candidate_path, path.join(work_dir, "chapters_candidate.xml"))
    decisions.append({"source": "master", "extraction": master_reason})
    decisions.append({"source": "candidate", "extraction": candidate_reason})
    out_root = ElementTree.Element("Chapters")
    master_editions = list(master_root.findall("EditionEntry")) if master_root is not None else []
    for number, edition in enumerate(master_editions):
        out_root.append(edition)
        decisions.append({"source": "master", "edition": number, "decision": "kept_as_is",
                          "atoms": len(list(edition.iter("ChapterAtom")))})
    candidate_pieces = [piece for piece in reference_pieces if piece["source"] == "candidate"]
    scale = Decimal(str(time_scale)) if time_scale is not None else Decimal(1)
    for number, edition in enumerate(candidate_root.findall("EditionEntry")
                                     if candidate_root is not None else []):
        if (edition.findtext("EditionFlagOrdered") or "0").strip() == "1":
            decisions.append({"source": "candidate", "edition": number,
                              "decision": "not_delivered_ordered_edition"})
            continue
        for tag in ("EditionUID",):
            for element in edition.findall(tag):
                edition.remove(element)
        if len(master_editions):
            flag = edition.find("EditionFlagDefault")
            if flag is None:
                flag = ElementTree.SubElement(edition, "EditionFlagDefault")
            flag.text = "0"
        dropped = []
        for parent in [edition] + list(edition.iter("ChapterAtom")):
            for atom in list(parent.findall("ChapterAtom")):
                for element in atom.findall("ChapterUID"):
                    atom.remove(element)
                title = atom.findtext("ChapterDisplay/ChapterString")
                start_in = parse_chapter_time_ms(atom.findtext("ChapterTimeStart"))
                if start_in is None:
                    dropped.append((parent, atom))
                    decisions.append({"source": "candidate", "edition": number,
                                      "atom": title, "decision": "dropped_unreadable_start"})
                    continue
                start_out, start_how = map_candidate_time_to_master(start_in * scale,
                                                                    candidate_pieces)
                if start_out is None or start_out >= timeline_ms:
                    dropped.append((parent, atom))
                    decisions.append({"source": "candidate", "edition": number,
                                      "atom": title, "start_in_ms": str(start_in),
                                      "decision": f"dropped_{start_how}"
                                      if start_out is None else "dropped_past_timeline"})
                    continue
                atom.find("ChapterTimeStart").text = format_chapter_time(start_out)
                end_element = atom.find("ChapterTimeEnd")
                end_in = end_out = None
                end_how = "absent"
                if end_element is not None:
                    end_in = parse_chapter_time_ms(end_element.text)
                    if end_in is None:
                        atom.remove(end_element)
                        end_how = "removed_unreadable"
                    else:
                        end_out, end_how = map_candidate_time_to_master(end_in * scale,
                                                                        candidate_pieces)
                        if end_out is None or end_out > timeline_ms:
                            end_out, end_how = timeline_ms, f"clamped_to_timeline_end({end_how})"
                        if end_out < start_out:
                            end_out, end_how = start_out, f"raised_to_start({end_how})"
                        end_element.text = format_chapter_time(end_out)
                decisions.append({"source": "candidate", "edition": number, "atom": title,
                                  "start_in_ms": str(start_in), "start_out_ms": str(start_out),
                                  "start": start_how,
                                  "end_in_ms": None if end_in is None else str(end_in),
                                  "end_out_ms": None if end_out is None else str(end_out),
                                  "end": end_how})
        for parent, atom in dropped:
            parent.remove(atom)
        if edition.find("ChapterAtom") is None:
            decisions.append({"source": "candidate", "edition": number,
                              "decision": "not_delivered_no_atom_left"})
            continue
        out_root.append(edition)
        decisions.append({"source": "candidate", "edition": number,
                          "decision": "delivered_retimed"})
    if out_root.find("EditionEntry") is None:
        return None, decisions
    out_path = path.join(work_dir, "chapters_delivered.xml")
    ElementTree.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, decisions


def probe_delivered_durations(file_path, timeout=300):
    """Probe the produced file's durations with ffprobe.

    Returns a dict with the container duration, each stream's duration and the
    last subtitle cue end, in ms; unreadable values are None, never 0.
    """
    import json as _json
    result = {"container_ms": None, "streams": [], "max_cue_end_ms": None}
    command = [tools.software["ffprobe"], "-v", "error", "-show_entries",
               "format=duration:stream=index,codec_type,duration:stream_tags=DURATION",
               "-of", "json", file_path]
    tools.dev_log(f"chimeric: probe_delivered_durations ffprobe call file={file_path}\n")
    with repair_log.announced("chimeric", "ffprobe", file_path) as call:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        call["exit"] = completed.returncode
    data = _json.loads(completed.stdout or "{}")
    try:
        result["container_ms"] = str(Decimal(str(data["format"]["duration"])) * 1000)
    except Exception:
        pass
    subtitle_indices = []
    for stream in data.get("streams") or []:
        duration = stream.get("duration") or (stream.get("tags") or {}).get("DURATION")
        value = None
        if duration not in (None, "N/A"):
            try:
                value = (str(Decimal(str(duration)) * 1000) if ":" not in str(duration)
                         else str(parse_chapter_time_ms(str(duration))))
            except Exception:
                value = None
        result["streams"].append({"index": stream.get("index"),
                                  "type": stream.get("codec_type"), "duration_ms": value})
        if stream.get("codec_type") == "subtitle":
            subtitle_indices.append(stream.get("index"))
    cue_ends = []
    for index in subtitle_indices:
        command = [tools.software["ffprobe"], "-v", "error", "-select_streams", str(index),
                   "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0",
                   file_path]
        with repair_log.announced("chimeric", "ffprobe", file_path) as call:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=timeout)
            call["exit"] = completed.returncode
        last = None
        for line in completed.stdout.splitlines():
            parts = line.split(",")
            try:
                end = (Decimal(parts[0]) + Decimal(parts[1])) * 1000
            except Exception:
                continue
            last = end if last is None or end > last else last
        if last is not None:
            cue_ends.append(last)
    if cue_ends:
        result["max_cue_end_ms"] = str(max(cue_ends))
    return result


def iterate_candidate_audios(candidate_obj):
    """Yield `(language, audio)` for every candidate track the repair may rebuild.

    Covers normal, audio-description and commentary tracks, skips tracks marked
    `dropped_corrupt`, and yields each StreamOrder once (under its first
    language), since `video.py` may list one track under two language keys.
    """
    seen = {}
    for holder in (candidate_obj.audios, candidate_obj.audiodesc, candidate_obj.commentary):
        for language, audios in holder.items():
            for audio in audios:
                if audio.get("dropped_corrupt"):
                    continue
                order = str(audio.get("StreamOrder"))
                first_language = seen.get(order)
                if first_language is not None:
                    tools.dev_log(f"repair: track_dedup stream={order} "
                                  f"langs={first_language},{language}\n")
                    continue
                seen[order] = language
                yield language, audio


def get_master_timeline_length_ms(master_obj):
    '''Length in ms of the output timeline: the master's video duration, as imposed by `generate_new_file`.'''
    return Decimal(str(master_obj.video["Duration"])) * Decimal("1000")


def get_master_container_length_ms(master_obj):
    '''The master's container duration in ms, or None when it is not declared.

    It can exceed the video duration (a Matroska segment lasts until its last
    block on any track), so it is the reference for the produced container's
    duration. No fallback to the video duration.
    '''
    try:
        for track in master_obj.mediadata["media"]["track"]:
            if track.get("@type") == "General" and "Duration" in track:
                return Decimal(str(track["Duration"])) * Decimal("1000")
    except Exception:
        pass
    return None


def get_track_audio_length_ms(audio):
    '''This track's declared duration in ms, or None when it has no `Duration`.

    Tracks of one file differ in length, so the bound is per track, not the
    file's longest audio track.
    '''
    if "Duration" not in audio:
        return None
    return Decimal(str(audio["Duration"])) * Decimal("1000")


def measure_track_extent_ms(file_path, stream_order, timeout=300):
    '''Measure a track's real extent from its last packet end.

    The declared `Duration` can understate the stream by up to ~120 ms; this is
    run only after the declared bound has refused.

    Returns:
        (extent_ms or None, reason), reason being `measured`, `binary-absent`,
        `probe-exit-<code>`, `timeout`, `probe-failed(<type>)` or `no-packets`.
    '''
    try:
        probe = tools.software["ffprobe"]
    except Exception:
        return None, "binary-absent"
    command = [probe, "-v", "error", "-select_streams", str(stream_order),
               "-show_entries", "packet=pts_time,duration_time",
               "-of", "csv=p=0", file_path]
    tools.dev_log(f"chimeric: measure_track_extent_ms starting "
                  f"file={file_path} stream_order={stream_order}\n")
    try:
        with repair_log.announced("chimeric", "ffprobe", file_path) as call:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=timeout)
            call["exit"] = result.returncode
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as error:
        return None, f"probe-failed({type(error).__name__})"
    if result.returncode != 0:
        return None, f"probe-exit-{result.returncode}"
    last = None
    for line in result.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            continue
        try:
            last = (Decimal(parts[0].strip()) + Decimal(parts[1].strip())) \
                * Decimal("1000")
        except Exception:
            continue
    if last == None:
        return None, "no-packets"
    return last, "measured"


def describe_candidate_track(audio, language):
    '''Log label for a track: stream order, language and codec (language alone is ambiguous).'''
    codec = (audio.get("ffprobe", {}).get("codec_name")
             or audio.get("Format") or "unknown")
    return (f"stream_order={audio.get('StreamOrder')} "
            f"language={language or 'unknown'} codec={codec}")


def resolve_master_grid(frame_rate_mode, frame_rate, frame_rate_original):
    '''The master's frame rate as an exact rational, or None when not measurable.

    Mode "CFR" uses `FrameRate`; any other or absent mode requires a parseable
    `FrameRate_Original`.
    '''
    if frame_rate_mode == "CFR":
        return parse_positive_rate(frame_rate)
    return parse_positive_rate(frame_rate_original)


# Marker for a re-encoded track that is neither chimeric nor resampled; every
# rebuilt track needs a non-empty marker, or it would pass for an intact one.
REBUILT_MARKER = "rebuilt"


def delivered_marker(base_marker, factor, engine="asetrate"):
    """The marker a rebuilt track is muxed with: `compose_marker`'s, or `REBUILT_MARKER`."""
    return compose_marker(base_marker, factor, engine) or REBUILT_MARKER


def compose_marker(base_marker, factor, engine="asetrate"):
    '''Compose a track marker such as `chimeric+resampled:1001/1000`.

    Args:
        base_marker: the plan-level marker (e.g. `chimeric`), or empty.
        factor: the requested speed ratio, or None when no speed change applies.
        engine: `atempo` writes `atempo:<ratio>`, otherwise `resampled:<ratio>`.

    The ratio is written as the nearest rational with denominator <= 10**6, so
    every track of one product names the same exact factor.
    '''
    parts = [base_marker] if base_marker else []
    if factor is not None:
        exact = Fraction(str(factor)).limit_denominator(10 ** 6)
        verb = "atempo" if engine == "atempo" else "resampled"
        parts.append(f"{verb}:{exact.numerator}/{exact.denominator}")
    return "+".join(parts)


def assemble_on_master_timeline(candidate_obj, master_obj, track_plans, reference_pieces,
                                work_dir, out_path, marker_value, job_start_utc,
                                timeout=3600, verify=True, verify_tolerance_ms=15,
                                verify_search_ms=30000, max_silence_fraction=None,
                                speed_ratio=None, reference_stream=None,
                                comparison_language=None, chapters_path=None,
                                deadline=None, speed_engine="asetrate"):
    '''Build the repaired file on the master timeline from per-track plans; measures nothing.

    Args:
        track_plans: `{StreamOrder: {"pieces", "extent_ms", "extent_source",
            "offset_measured", "borrow_reason", "offset_sources"}}`; each track
            has its own pieces because it has its own offset.
        reference_pieces: the comparison track's pieces, used for subtitles and
            by the verifier.
        chapters_path: retimed chapters XML to set at mux, or None.
        job_start_utc: job start time, written into the VMSAM_ERA tag.

    Returns:
        A report of built, declined and failed tracks.

    Raises:
        chimeric_error: the plan or master is unusable, nothing could be built,
            or the repair budget ran out.
    '''
    tools.dev_log(f"chimeric: assemble_on_master_timeline starting "
                  f"candidate={candidate_obj.filePath} "
                  f"master={master_obj.filePath} work_dir={work_dir} "
                  f"out_path={out_path}\n")
    # Preconditions are checked before any build, so a refusal costs no encode.
    frame_rate = None
    frame_rate_mode = None
    frame_rate_original = None
    try:
        frame_rate = master_obj.video.get("FrameRate")
        frame_rate_mode = master_obj.video.get("FrameRate_Mode")
        original = master_obj.video.get("FrameRate_Original")
        if original != None and str(original) != str(frame_rate):
            frame_rate_original = original
    except Exception:
        pass
    # Piece boundaries are master frames, so the master's frame grid must be known.
    if resolve_master_grid(frame_rate_mode, frame_rate, frame_rate_original) is None:
        raise chimeric_error(
            f"the master's frame grid is not measurable -- FrameRate_Mode="
            f"{frame_rate_mode!r} FrameRate={frame_rate!r} "
            f"FrameRate_Original={frame_rate_original!r}: this candidate's "
            f"segment boundaries are not expressed on a measured grid")

    master_duration_ms = get_master_timeline_length_ms(master_obj)
    # Every plan must tile the master timeline from 0 to its end, without gap or overlap.
    for label, pieces_to_check in [("reference", reference_pieces)] + [
            (f"stream {order}", plan["pieces"]) for order, plan in track_plans.items()]:
        cursor = Decimal("0")
        for piece in pieces_to_check:
            if Decimal(str(piece["master_start_ms"])) != cursor:
                _refuse_plan_shape(
                    "plan_not_contiguous",
                    f"the {label} plan is not contiguous on the master timeline: a "
                    f"piece starts at {piece['master_start_ms']} ms where the previous "
                    f"one ended at {cursor} ms",
                    f"plan={label} piece_start_ms={piece['master_start_ms']} "
                    f"previous_end_ms={cursor}")
            cursor = Decimal(str(piece["master_end_ms"]))
            if cursor <= Decimal(str(piece["master_start_ms"])):
                _refuse_plan_shape(
                    "plan_piece_empty_or_inverted",
                    f"the {label} plan carries an empty or inverted piece "
                    f"[{piece['master_start_ms']},{piece['master_end_ms']})",
                    f"plan={label} piece_start_ms={piece['master_start_ms']} "
                    f"piece_end_ms={piece['master_end_ms']}")
        if cursor != master_duration_ms:
            _refuse_plan_shape(
                "plan_end_not_master_timeline",
                f"the {label} plan ends at {cursor} ms, not at the master's "
                f"timeline end {master_duration_ms} ms",
                f"plan={label} plan_end_ms={cursor} master_timeline_ms={master_duration_ms}")

    tools.make_dirs(work_dir)
    audio_reports = []
    subtitle_reports = []
    declined = []
    failed = []

    # The track build counts against the repair budget: the deadline is checked
    # before each track and bounds each ffmpeg call; a half-built set is never muxed.
    import time as _time
    timeline_s = float(master_duration_ms) / 1000.0
    built = []

    def build_bound():
        bound = min(float(timeout), tools.decoder_timeout_for(timeline_s))
        if deadline is not None:
            left = deadline - _time.monotonic()
            if left <= 0:
                tools.log_always(
                    f"repair: partial_plan cause=repair_budget_exceeded placed={built} "
                    f"-- the repair's budget ran out during the track build, nothing is muxed "
                    f"for {candidate_obj.filePath}\n")
                raise chimeric_error(
                    f"the repair's budget ran out during the track build after {len(built)} "
                    f"track(s) {built} -- nothing is muxed; declined, retried at the next run",
                    cause="repair_budget_exceeded")
            bound = min(bound, left)
        return bound

    def budget_stopped(error):
        """A build that failed because the budget's bound cut it is the budget's refusal."""
        if deadline is not None and _time.monotonic() >= deadline:
            build_bound()                    # raises repair_budget_exceeded with the partial plan
        return error

    index = 0
    for language, audio in iterate_candidate_audios(candidate_obj):
        track_path = path.join(work_dir, f"audio_{index}.mka")
        index += 1
        bound = build_bound()
        try:
            stream_order = int(audio["StreamOrder"])
            track_plan = track_plans.get(stream_order)
            track_label = describe_candidate_track(audio, language)
            if track_plan is None:
                raise chimeric_error(
                    f"the plan carries no pieces for {track_label}: no offset was "
                    f"established for this stream, and none is borrowed silently")
            track_bound_ms = track_plan["extent_ms"]
            tools.log_line(
                f"chimeric: extraction bound {track_label} "
                f"bound_ms={track_bound_ms} source={track_plan['extent_source']}\n")
            with repair_log.announced("chimeric", f"track_build:audio:{stream_order}",
                                      candidate_obj.filePath, media_s=timeline_s) as call:
                report = build_one_audio_track(
                    candidate_obj, master_obj, audio, language, track_plan["pieces"],
                    track_path, bound, speed_ratio, reference_stream, comparison_language,
                    track_bound_ms, speed_engine)
                call["exit"] = 0
            built.append(f"audio:{stream_order}")
            report["extraction_bound_ms"] = str(track_bound_ms)
            report["extraction_bound_source"] = track_plan["extent_source"]
            report["extraction_bound_track"] = track_label
            report["offset_measured"] = track_plan["offset_measured"]
            report["borrow_reason"] = track_plan.get("borrow_reason")
            report["offset_sources"] = track_plan.get("offset_sources")
            report["offset_fidelity"] = None
            # The marker names the requested ratio; the asetrate-quantised one
            # stays in `speed_ratio_applied`.
            report["marker"] = delivered_marker(
                marker_value,
                speed_ratio if report.get("speed_ratio_applied") is not None else None,
                speed_engine)
            audio_reports.append(report)
        except chimeric_error as error:
            budget_stopped(error)
            declined.append({"kind": "audio",
                             "stream_order": int(audio["StreamOrder"]),
                             "language": language, "reason": str(error)})
        except Exception as error:
            budget_stopped(error)
            failed.append({"kind": "audio",
                           "stream_order": int(audio["StreamOrder"]),
                           "language": language, "reason": str(error)})
            tools.log_line(f"chimeric: audio track {audio['StreamOrder']} failed: {error}\n")

    index = 0
    for language, subtitles in candidate_obj.subtitles.items():
        for subtitle in subtitles:
            # Set by a caller that compared this track's own text against a same-language
            # master subtitle (`merge_video_visual_fallback._drop_duplicate_candidate
            #_subtitles`): a retimed copy of a subtitle the master already carries never
            # changes the delivered content, only its stream MD5 (the merge's own dedup key,
            # which a retime always defeats).
            if subtitle.get("dropped_duplicate"):
                tools.log_line(
                    f"chimeric: subtitle stream_order={subtitle['StreamOrder']} "
                    f"language={language} decision=dropped_duplicate "
                    f"reason={subtitle['dropped_duplicate']}\n")
                index += 1
                continue
            bound = build_bound()
            try:
                with repair_log.announced(
                        "chimeric", f"track_build:subtitle:{subtitle['StreamOrder']}",
                        candidate_obj.filePath) as call:
                    report = build_one_subtitle_track(
                        candidate_obj, subtitle, language, reference_pieces, work_dir, index,
                        bound, speed_ratio)
                    call["exit"] = 0
                built.append(f"subtitle:{subtitle['StreamOrder']}")
                report["marker"] = delivered_marker(
                    marker_value, Decimal(str(speed_ratio)) if speed_ratio is not None else None,
                    speed_engine)
                subtitle_reports.append(report)
            except chimeric_error as error:
                budget_stopped(error)
                declined.append({"kind": "subtitle",
                                 "stream_order": int(subtitle["StreamOrder"]),
                                 "language": language, "reason": str(error)})
            except Exception as error:
                budget_stopped(error)
                failed.append({"kind": "subtitle",
                               "stream_order": int(subtitle["StreamOrder"]),
                               "language": language, "reason": str(error)})
                tools.log_line(f"chimeric: subtitle track {subtitle['StreamOrder']} failed: {error}\n")
            index += 1

    if max_silence_fraction != None:
        kept = []
        for report in audio_reports:
            if Decimal(report["silence_fraction"]) > Decimal(str(max_silence_fraction)):
                declined.append({"kind": "audio", "stream_order": report["stream_order"],
                                 "language": report["language"],
                                 "reason": f"{report['silence_filled_ms']} ms of the track "
                                           f"would be silence "
                                           f"({report['silence_fraction']} of it), over the "
                                           f"configured budget {max_silence_fraction}"})
            else:
                kept.append(report)
        audio_reports = kept

    if not len(audio_reports) and not len(subtitle_reports):
        raise chimeric_error(
            f"nothing could be rebuilt: {len(declined)} track(s) declined, "
            f"{len(failed)} failed")

    # Predict refusals from fill shortfalls before the mux; the file is still
    # built so the verification gate's verdict can be compared with the prediction.
    predicted_refusals = []
    for report in audio_reports:
        short = report.get("fill_short_by_ms")
        if short in (None, "", "0"):
            continue
        if abs(Decimal(str(short))) <= output_duration_tolerance_ms:
            continue
        predicted_refusals.append(report)
        # `reason=` is a stable [A-Za-z0-9_]+ token: log readers parse key=value fields.
        tools.log_line(
            f"repair: PREDICTED_REFUSAL track={report.get('stream_order')} "
            f"fill_short_by_ms={short} "
            f"tolerance_ms={output_duration_tolerance_ms} "
            f"reason=fill_source_too_short\n")

    file_marker = delivered_marker(
        marker_value, Decimal(str(speed_ratio)) if speed_ratio is not None else None,
        speed_engine)
    mux_bound = build_bound()
    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      mux_bound, job_start_utc, chapters_path=chapters_path)

    # The duration check runs before alignment: alignment probes on a truncated
    # track would still read "aligned". A refused output is kept on disk, renamed
    # `<name>.REFUSED.mkv` (or the no-verdict equivalent) by `mark_output`.
    try:
        # Container duration is compared to the master's container duration;
        # `master_duration_ms` is the timeline the assembly targets.
        output_check = verify_output_file(out_path, master_duration_ms, audio_reports,
                                          subtitle_reports,
                                          output_duration_tolerance_ms,
                                          get_master_container_length_ms(master_obj))
        log_prediction_outcome(predicted_refusals, output_check.get("would_refuse"))

        verification = None
        if verify:
            verification = verify_on_master_timeline(
                out_path, master_obj, audio_reports, reference_pieces, verify_tolerance_ms,
                verify_search_ms, reference_stream, deadline=deadline,
                envelope=(speed_engine == "atempo" and speed_ratio is not None),
                comparison_language=comparison_language)
            dropped_orders = {r["track"] for r in verification if r.get("drop")}
            if dropped_orders:
                for r in verification:
                    if not r.get("drop"):
                        continue
                    declined.append({"kind": "audio", "stream_order": r["track"],
                                     "language": r["language"],
                                     "reason": f"dropped at verification: outcome={r['outcome']} "
                                               f"-- its own content does not carry the file's "
                                               f"geometry, so it is dropped by name instead of "
                                               f"failing every other track"})
                audio_reports[:] = [r for r in audio_reports
                                    if r["stream_order"] not in dropped_orders]
                if audio_reports or subtitle_reports:
                    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                                      build_bound(), job_start_utc, chapters_path=chapters_path)
                else:
                    raise chimeric_error(
                        f"nothing could be rebuilt: every candidate track was dropped at "
                        f"verification ({sorted(dropped_orders)})")

        # Fill-content verification always runs and only records; only a budget
        # refusal propagates.
        fill_content = []
        try:
            fill_content = verify_fill_content(
                out_path, master_obj, audio_reports, master_duration_ms, deadline=deadline)
        except Exception as error:
            if getattr(error, "cause", None) == "repair_budget_exceeded":
                raise
            tools.log_line(
                f"chimeric: fill_content_verification_failed "
                f"{type(error).__name__}: {error}\n")
        if fill_content:
            if verification is None:
                verification = []
            by_track = {entry.get("track"): entry for entry in verification}
            for fc in fill_content:
                entry = by_track.get(fc["track"])
                if entry is None:
                    entry = {"track": fc["track"], "produced_index": fc["produced_index"],
                            "outcome": "skipped",
                            "reason": "no AV-alignment result recorded for this track"}
                    verification.append(entry)
                    by_track[fc["track"]] = entry
                entry["fill_content"] = fc["regions"]
                # A `skipped` alignment outcome is replaced by the content verdict;
                # a real sync verdict is never overwritten.
                if entry.get("outcome") == "skipped":
                    outcomes = [r["outcome"] for r in fc["regions"]]
                    if any(o == "content_mismatch" for o in outcomes):
                        entry["outcome"] = "content_mismatch"
                    elif any(o == "content_indiscriminate" for o in outcomes):
                        entry["outcome"] = "content_indiscriminate"
                    elif any(o == "content_verified" for o in outcomes):
                        entry["outcome"] = "content_verified"
    except Exception as error:
        log_prediction_outcome(predicted_refusals,
                               (getattr(error, "output_check", None) or {}).get(
                                   "would_refuse"))
        marking = (OUTPUT_REFUSED if isinstance(error, chimeric_error)
                   else OUTPUT_NO_VERDICT)
        error.undelivered_state = marking[0]
        error.undelivered_path = mark_output(out_path, marking)
        error.undelivered_in_place = out_path
        # The partial assembly rides on the exception so the caller can still log it.
        error.partial_assembly = {
            "path": out_path, "pieces": reference_pieces, "audios": audio_reports,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": file_marker, "base_marker": marker_value,
            "verification": None}
        raise

    return {"path": out_path, "pieces": reference_pieces, "audios": audio_reports,
            "master_frame_rate": frame_rate,
            "master_frame_rate_mode": frame_rate_mode,
            "master_frame_rate_original": frame_rate_original,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": file_marker, "base_marker": marker_value,
            "output_check": output_check,
            "verification": verification}


# --------------------------------------------------------------------------
# Verification: is the produced track on the master's timeline?
# --------------------------------------------------------------------------

# Two 20 s probes per candidate piece cover roughly 5-11 % of the duration, so
# `aligned` means the probed positions are aligned: persistent shifts are
# detected, a short transient one only if a probe lands on it.
verify_window_seconds = 20
verify_probe_rate = 8000
# Minimum window RMS: near-silence (~1e-5) correlates noise with false
# confidence, while content reads 2e-3 and above.
verify_min_rms = 1e-4
# Minimum correlation peak (same as `audio_walk.MATCH_NCC`): matched tracks read
# above 0.8, unrelated content below 0.2.
verify_min_correlation = 0.5


def choose_probe_positions(pieces, window_seconds):
    """Choose probe positions inside candidate pieces: two per piece when it fits two windows.

    Master-filled gaps are never probed (they match the master by construction).
    Two spaced probes in one piece disagree when an unknown boundary lies
    between them. Every piece holding at least one window is probed.

    Returns:
        A list of (piece_index, start_ms), sorted by position.
    """
    window_ms = Decimal(str(window_seconds)) * Decimal("1000")
    indexed = [(i, p) for i, p in enumerate(pieces) if p["source"] == "candidate"
               and (p["master_end_ms"] - p["master_start_ms"]) >= window_ms]
    if not len(indexed):
        return []

    by_length = sorted(indexed, key=lambda x: x[1]["master_end_ms"] - x[1]["master_start_ms"],
                       reverse=True)
    by_position = sorted(indexed, key=lambda x: x[1]["master_start_ms"])
    # All pieces: short ones are the most likely to carry a wrong offset.
    chosen = by_position
    per_piece = 2
    positions = []
    for index, piece in chosen:
        span = piece["master_end_ms"] - piece["master_start_ms"] - window_ms
        if span < 0:
            span = Decimal("0")
        count = per_piece if span >= window_ms else 1
        divisor = Decimal(max(1, count - 1))
        for i in range(count):
            offset = span * Decimal(i) / divisor if count > 1 else span / 2
            positions.append((index, piece["master_start_ms"] + offset))
    return sorted(positions, key=lambda x: x[1])


# Pre-roll of the hybrid seek: an input seek lands this many seconds before the
# window and an output seek cuts the rest, so a probe decodes ~25 s instead of
# the file from its start, with sub-0.1 ms difference from a pure output seek.
PROBE_PREROLL_S = Decimal("5")


def _mono_command(file_path, stream_specifier, input_seek_s, output_seek_s, duration_s, rate):
    command = [tools.software["ffmpeg"], "-v", "error", "-nostdin"]
    if input_seek_s > 0:
        command += ["-ss", f"{input_seek_s:.3f}"]
    return command + ["-i", file_path, "-map", stream_specifier,
                      "-ss", f"{output_seek_s:.3f}", "-t", f"{duration_s:.3f}",
                      "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                      "-ar", str(rate), "-"]


def _run_bounded(command, media_seconds, deadline, file_path, stream_specifier):
    """Run one ffmpeg reader call under the decoder timeout and the repair deadline.

    `deadline` is a `time.monotonic()` instant or None. Raises `chimeric_error`
    (`repair_budget_exceeded`) when the budget stops the call, else
    `tools.decoder_timeout` on timeout.
    """
    timeout = tools.decoder_timeout_for(media_seconds)
    budget_bound = False
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise chimeric_error(
                f"the repair's budget ran out before the reader could decode {stream_specifier} "
                f"of {path.basename(file_path)}", cause="repair_budget_exceeded")
        if remaining < timeout:
            timeout, budget_bound = remaining, True
    try:
        with repair_log.announced("chimeric", "ffmpeg", file_path,
                                  media_s=media_seconds) as call:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=timeout)
            call["exit"] = process.returncode
    except subprocess.TimeoutExpired:
        if budget_bound:
            raise chimeric_error(
                f"the repair's budget ran out while the reader decoded {stream_specifier} of "
                f"{path.basename(file_path)}", cause="repair_budget_exceeded")
        raise tools.decoder_timeout("read_mono_samples", timeout,
                                    f"file={file_path} stream={stream_specifier}")
    return process


def read_splice_window(file_path, stream_specifier, start_s, duration_s, audio_filter=None,
                       scale=None):
    """Read a mono window at `splice_hygiene.MEASURE_RATE` for splice measurements.

    With `audio_filter` / `scale` the window is read on the speed-corrected clock
    (corrected second t is raw second t / scale). Returns None when nothing was
    read; the splice is then hard-cut.
    """
    import numpy
    import splice_hygiene
    scale = float(scale) if scale else 1.0
    start_s, duration_s = max(0.0, float(start_s)), float(duration_s)
    if duration_s <= 0:
        return None
    raw_start = start_s / scale
    preroll = min(float(PROBE_PREROLL_S), raw_start)
    command = [tools.software["ffmpeg"], "-v", "error", "-nostdin"]
    if raw_start - preroll > 0:
        command += ["-ss", f"{raw_start - preroll:.6f}"]
    command += ["-i", file_path, "-map", stream_specifier]
    if audio_filter:
        command += ["-af", audio_filter]
    command += ["-ss", f"{preroll * scale:.6f}", "-t", f"{duration_s:.6f}", "-f", "f32le",
                "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(splice_hygiene.MEASURE_RATE), "-"]
    try:
        process = _run_bounded(command, preroll + duration_s, None, file_path, stream_specifier)
    except tools.decoder_timeout:
        return None
    if process.returncode != 0 or not process.stdout:
        return None
    return numpy.frombuffer(process.stdout, dtype=numpy.float32).astype(numpy.float64)


def plan_splices(candidate_obj, audio, master_obj, master_audio, pieces, speed_chain, speed_ratio):
    """Measure the gain and join type of every master fill piece at its candidate splices.

    Returns:
        `{piece_index: {"gain", "fade_left", "fade_right", "edges"}}`; each splice
        and each piece is logged.
    """
    import splice_hygiene
    plan = {}
    if master_audio is None:
        return plan
    candidate_spec = f"0:{int(audio['StreamOrder'])}"
    master_spec = f"0:{int(master_audio['StreamOrder'])}"
    scale = float(speed_ratio) if speed_ratio is not None else None
    window = splice_hygiene.SPLICE_MEASURE_S
    edge = splice_hygiene.FADE_S
    for index, piece in enumerate(pieces):
        if piece["source"] != "master":
            continue
        start_s = float(piece["master_start_ms"]) / 1000.0
        end_s = float(piece["master_end_ms"]) / 1000.0
        offset_s = float(piece["source_start_ms"]) / 1000.0 - start_s     # master source clock
        entry = {"gain": None, "fade_left": False, "fade_right": False, "edges": []}
        readings = {}
        for side, neighbour_index in (("A", index - 1), ("B", index + 1)):
            if not 0 <= neighbour_index < len(pieces) or \
                    pieces[neighbour_index]["source"] != "candidate":
                continue
            neighbour = pieces[neighbour_index]
            n_start = float(neighbour["master_start_ms"]) / 1000.0
            n_end = float(neighbour["master_end_ms"]) / 1000.0
            n_offset = float(neighbour["source_start_ms"]) / 1000.0 - n_start
            at = start_s if side == "A" else end_s
            lo, hi = ((max(n_start, at - window), at) if side == "A"
                      else (at, min(n_end, at + window)))
            receiving = read_splice_window(candidate_obj.filePath, candidate_spec,
                                           lo + n_offset, hi - lo, speed_chain, scale)
            source = read_splice_window(master_obj.filePath, master_spec, lo + offset_s, hi - lo)
            d, blocks = (splice_hygiene.edge_gain(receiving, source)
                         if receiving is not None and source is not None else (None, 0))
            source_edge = read_splice_window(master_obj.filePath, master_spec,
                                             at + offset_s - edge, 2 * edge)
            rate = splice_hygiene.MEASURE_RATE
            receiving_edge = (None if receiving is None else
                              receiving[-int(edge * rate):] if side == "A"
                              else receiving[:int(edge * rate)])
            join = splice_hygiene.splice_join(receiving_edge, source_edge)
            margin_ok = (at + offset_s - edge >= 0) if side == "A" else True
            if join == "crossfade" and not margin_ok:
                join = "hard_cut"
            readings[side] = d
            entry["fade_left" if side == "A" else "fade_right"] = join == "crossfade"
            entry["edges"].append({"side": side, "master_s": round(at, 3), "gain_db": d,
                                   "coherent_blocks": blocks, "join": join,
                                   "measure_s": [round(lo, 3), round(hi, 3)]})
            tools.log_always(
                f"repair: splice track={audio['StreamOrder']} fill_piece={index} side={side} "
                f"master_s={round(at, 3)} join={join}"
                f"{' fade_ms=' + str(round(edge * 1000, 1)) if join == 'crossfade' else ''} "
                f"edge_gain_db={None if d is None else round(d, 3)} coherent_blocks={blocks} "
                f"measure_s=[{round(lo, 3)}, {round(hi, 3)}] for {candidate_obj.filePath}\n")
        if not entry["edges"]:
            continue
        gain = splice_hygiene.fill_gain(readings.get("A"), readings.get("B"))
        entry["gain"] = gain
        tools.log_always(
            f"repair: fill_gain track={audio['StreamOrder']} fill_piece={index} "
            f"master_s=[{round(start_s, 3)}, {round(end_s, 3)}] mode={gain['mode']} "
            f"gain_a_db={gain['gain_a_db']} gain_b_db={gain['gain_b_db']}"
            f"{' fill_gain_unmeasurable' if gain['unmeasurable'] else ''} "
            f"for {candidate_obj.filePath}\n")
        plan[index] = entry
    return plan


def plan_candidate_joins(candidate_obj, audio, pieces, speed_chain, speed_ratio):
    """Choose crossfade or hard cut at each join between two candidate pieces.

    A join where both sides carry sound gets a 10 ms crossfade; a side in digital
    silence stays a hard cut. Returns `{piece_index: {"fade_right": True}}` for
    crossfaded joins; every join is logged.
    """
    import splice_hygiene
    joins = {}
    spec = f"0:{int(audio['StreamOrder'])}"
    scale = float(speed_ratio) if speed_ratio is not None else None
    edge = splice_hygiene.FADE_S
    for index in range(len(pieces) - 1):
        left, right = pieces[index], pieces[index + 1]
        if left["source"] != "candidate" or right["source"] != "candidate":
            continue
        at = float(left["master_end_ms"]) / 1000.0
        left_source = float(left["source_start_ms"]) / 1000.0 + at \
            - float(left["master_start_ms"]) / 1000.0
        right_source = float(right["source_start_ms"]) / 1000.0
        if abs(left_source - right_source) < 0.0005:
            continue                                    # the same material continues: no join
        left_edge = read_splice_window(candidate_obj.filePath, spec, left_source - edge, edge,
                                       speed_chain, scale)
        right_edge = read_splice_window(candidate_obj.filePath, spec, right_source, edge,
                                        speed_chain, scale)
        join = splice_hygiene.splice_join(left_edge, right_edge)
        if join == "crossfade":
            joins[index] = {"fade_right": True}
        tools.log_always(
            f"repair: splice track={audio['StreamOrder']} candidate_join={index} "
            f"master_s={round(at, 3)} join={join}"
            f"{' fade_ms=' + str(round(edge * 1000, 1)) if join == 'crossfade' else ''} "
            f"for {candidate_obj.filePath}\n")
    return joins


def read_mono_samples(file_path, stream_specifier, start_ms, duration_ms, rate, deadline=None):
    """Mono samples of one window, mean removed.

    Uses the hybrid seek (input seek `PROBE_PREROLL_S` before, output seek for
    the rest); a short hybrid read is retried with an output seek alone, which
    works on files without a usable index. Raises `chimeric_error` when ffmpeg
    fails or less than one second is read.
    """
    import numpy
    start_s = start_ms / Decimal("1000")
    duration_s = duration_ms / Decimal("1000")
    preroll = min(PROBE_PREROLL_S, start_s)
    tools.dev_log(f"chimeric: read_mono_samples starting file={file_path} "
                  f"stream={stream_specifier} start_ms={start_ms} "
                  f"duration_ms={duration_ms}\n")
    hybrid = _mono_command(file_path, stream_specifier, start_s - preroll, preroll, duration_s,
                           rate)
    try:
        process = _run_bounded(hybrid, float(preroll + duration_s), deadline, file_path,
                               stream_specifier)
    except tools.decoder_timeout:
        # One retry: host load can make a short read overrun once.
        tools.dev_log(f"chimeric: read_mono_samples retry=decoder_timeout file={file_path} "
                      f"stream={stream_specifier} start_ms={start_ms}\n")
        process = _run_bounded(hybrid, float(preroll + duration_s), deadline, file_path,
                               stream_specifier)
    expected = int(duration_s * rate) - rate // 100
    if preroll > 0 and (process.returncode != 0 or len(process.stdout) // 4 < expected):
        tools.dev_log(f"chimeric: read_mono_samples seek=output_fallback file={file_path} "
                      f"stream={stream_specifier} start_ms={start_ms} hybrid_samples="
                      f"{len(process.stdout) // 4} expected={expected}\n")
        # Bounded by what it decodes: from the file's start to the window's end.
        process = _run_bounded(
            _mono_command(file_path, stream_specifier, Decimal(0), start_s, duration_s, rate),
            float(start_s + duration_s), deadline, file_path, stream_specifier)
    # A tool failure is reported as such, not as missing audio.
    if process.returncode != 0:
        raise chimeric_error(
            f"the reader FAILED on {stream_specifier} at {start_ms} ms: ffmpeg "
            f"exited {process.returncode}. THIS IS A STATEMENT ABOUT THE TOOL, "
            f"not about the media: "
            f"{(process.stderr or b'').decode('utf-8', 'replace').strip()[-300:]}")
    samples = numpy.frombuffer(process.stdout, dtype=numpy.float32).astype(numpy.float64)
    if len(samples) < rate:
        raise chimeric_error(
            f"read only {len(samples)} samples from {stream_specifier} at "
            f"{start_ms} ms: the track carries no audio to compare there",
            cause="probe_reads_no_audio")
    return samples - samples.mean()


def read_reference_window(video_obj, audio, start_ms, duration_ms, rate, deadline=None):
    """A master window on the master's sample clock, resampled to `rate`, mean removed.

    Sliced from the shared whole-track read (`audio_walk.read_on_file_clock`), the
    clock the plan was measured on; a seek would read on the packet clock, which
    can jitter by several ms.
    """
    import numpy
    import scipy.signal
    import audio_walk
    samples = audio_walk.read_on_file_clock(video_obj, audio, deadline=deadline)
    ratio = Fraction(int(audio_walk.WALK_RATE), int(rate))
    # Start on an output-grid sample: a sub-sample shift lowers the correlation peak.
    first = int(round(round(float(start_ms) * int(rate) / 1000.0) * ratio))
    count = int(round(float(duration_ms) * audio_walk.WALK_RATE / 1000.0))
    window = numpy.asarray(samples[max(0, first):max(0, first + count)], dtype=numpy.float64)
    if ratio != 1 and len(window):
        window = scipy.signal.resample_poly(window, ratio.denominator, ratio.numerator)
    if len(window) < rate:
        raise chimeric_error(
            f"read only {len(window)} samples from stream {audio.get('StreamOrder')} at "
            f"{start_ms} ms: the track carries no audio to compare there",
            cause="probe_reads_no_audio")
    return window - window.mean()


def read_track_samples(file_path, stream_order, rate, audio_filter=None, timeout=900,
                       deadline=None):
    """Decode a whole track once as mono float32 samples at `rate` Hz.

    Sample `i` is at stream time `start_time + i / rate`; with `audio_filter`
    (a speed chain) times are on the corrected timeline. With `deadline` (a
    `time.monotonic()` instant) the remaining budget replaces `timeout`.

    Raises:
        chimeric_error: ffmpeg failed or timed out (`repair_budget_exceeded`
            when the budget stopped it).
    """
    import numpy
    if deadline is not None:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise chimeric_error(
                f"the repair's budget ran out before the whole-track read of stream "
                f"{stream_order} of {path.basename(file_path)}", cause="repair_budget_exceeded")
    command = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
               "-i", file_path, "-map", f"0:{int(stream_order)}", "-vn", "-sn", "-dn"]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                    "-ar", str(rate), "-"])
    tools.dev_log(f"chimeric: read_track_samples starting file={file_path} "
                  f"stream_order={stream_order} rate={rate} filter={audio_filter}\n")
    try:
        with repair_log.announced("chimeric", "ffmpeg", file_path) as call:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=timeout)
            call["exit"] = process.returncode
    except subprocess.TimeoutExpired:
        if deadline is not None:
            raise chimeric_error(
                f"the repair's budget ran out during the whole-track read of stream "
                f"{stream_order} of {path.basename(file_path)} ({round(timeout, 1)} s were left)",
                cause="repair_budget_exceeded")
        raise chimeric_error(
            f"the whole-track read of stream {stream_order} did not finish in "
            f"{timeout} s: a statement about the tool, not about the media")
    if process.returncode != 0:
        raise chimeric_error(
            f"the whole-track read of stream {stream_order} FAILED: ffmpeg exited "
            f"{process.returncode}: "
            f"{(process.stderr or b'').decode('utf-8', 'replace').strip()[-300:]}")
    return numpy.frombuffer(process.stdout, dtype=numpy.float32)


def get_rms(samples):
    """Root mean square of the samples; 0.0 when empty."""
    import numpy
    return float(numpy.sqrt((samples ** 2).mean())) if len(samples) else 0.0


def measure_lag_ms(reference, produced, rate, search_ms):
    """Lag of `produced` against `reference` within +/- `search_ms`, by FFT cross-correlation.

    Returns:
        (lag_ms, normalised correlation at the peak).
    """
    import numpy
    size = 1 << int(numpy.ceil(numpy.log2(len(reference) + len(produced))))
    spectrum = numpy.fft.rfft(reference, size) * numpy.conj(numpy.fft.rfft(produced, size))
    correlation = numpy.fft.irfft(spectrum, size)
    correlation = numpy.concatenate((correlation[-(len(produced) - 1):],
                                     correlation[:len(reference)]))
    lags = numpy.arange(-(len(produced) - 1), len(reference))
    keep = numpy.abs(lags) <= int(search_ms * rate / 1000)
    correlation, lags = correlation[keep], lags[keep]
    peak = int(numpy.argmax(correlation))
    norm = numpy.linalg.norm(reference) * numpy.linalg.norm(produced)
    return (float(lags[peak]) * 1000.0 / rate,
            float(correlation[peak] / norm) if norm > 0 else 0.0)


# Audio track duration tolerance against the target: correct tracks land within
# ~+42 ms, defective ones off by seconds, so 500 ms separates the two.
output_duration_tolerance_ms = Decimal("500")

# When True `verify_output_file` refuses a failing file; when False it only
# records the verdict (`would_refuse`).
output_check_enforcing = True


def probe_output_streams(file_path):
    """Probe the produced file's streams (type, codec, rate, language, duration).

    Returns:
        (streams, container duration in ms or None).
    """
    import json as _json
    command = [tools.software["ffprobe"], "-v", "error",
               "-show_entries", "stream=index,codec_type,codec_name,"
                                "sample_rate,r_frame_rate,initial_padding:"
                                "stream_tags=language,DURATION:format=duration",
               "-of", "json", file_path]
    tools.dev_log(f"chimeric: probe_output_streams starting file={file_path}\n")
    with repair_log.announced("chimeric", "ffprobe", file_path) as call:
        data = _json.loads(subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                          timeout=tools.DECODER_TIMEOUT_BASE_S).stdout)
        call["exit"] = 0
    ends = None
    streams = []
    for entry in data.get("streams", []):
        index = entry.get("index")
        tags = entry.get("tags") or {}
        # In Matroska the stream duration is the DURATION tag (`stream=duration`
        # is N/A); the last packet is a fallback that under-reads the end.
        duration_ms, source = parse_duration_tag(tags.get("DURATION")), "matroska tag"
        if duration_ms == None:
            if ends == None:
                ends = last_audio_packet_ms(file_path)
            duration_ms = ends.get(index)
            source = "last packet (under-reads by one packet)" if duration_ms != None else None
        streams.append({"index": index,
                        "codec_type": entry.get("codec_type"),
                        "codec_name": entry.get("codec_name"),
                        "sample_rate": entry.get("sample_rate"),
                        "frame_rate": entry.get("r_frame_rate"),
                        # Declared codec delay in samples (Matroska CodecDelay).
                        "initial_padding": entry.get("initial_padding"),
                        "language": tags.get("language"),
                        "duration_ms": duration_ms,
                        "duration_source": source})
    container = (data.get("format") or {}).get("duration")
    return streams, (Decimal(str(container)) * 1000 if container not in (None, "N/A") else None)


def parse_duration_tag(value):
    """Parse a DURATION tag such as `00:23:51.993000000` into ms; None when absent or invalid."""
    if value in (None, "", "N/A"):
        return None
    try:
        hours, minutes, seconds = str(value).split(":")
        return ((Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds))
                * 1000)
    except Exception:
        return None


def last_audio_packet_ms(file_path):
    """Map each audio stream index to its last packet time in ms (empty dict on failure).

    Fallback when DURATION tags are missing; reads only the end of the file and
    under-reads the track end by one packet duration.
    """
    command = [tools.software["ffprobe"], "-v", "error", "-select_streams", "a",
               "-show_entries", "packet=stream_index,pts_time",
               "-of", "csv=p=0", "-read_intervals", "99%", file_path]
    tools.dev_log(f"chimeric: last_audio_packet_ms starting file={file_path}\n")
    try:
        with repair_log.announced("chimeric", "ffprobe", file_path) as call:
            output = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                    timeout=tools.DECODER_TIMEOUT_BASE_S).stdout.decode()
            call["exit"] = 0
    except Exception:
        return {}
    ends = {}
    for line in output.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or parts[1] in ("", "N/A"):
            continue
        try:
            index, value = int(parts[0]), Decimal(parts[1]) * 1000
        except Exception:
            continue
        if index not in ends or value > ends[index]:
            ends[index] = value
    return ends


def _digest_of_loaded_source():
    """Short SHA-256 of this module's source, taken at import so it matches the loaded code."""
    import hashlib
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except Exception:
        return "unreadable"


LOADED_SOURCE_DIGEST = _digest_of_loaded_source()


# Undelivered output states: REFUSED (the gate decided against the file) and
# NOVERDICT (a tool fault escaped before any verdict).
OUTPUT_REFUSED = ("REFUSED", "the gate DECIDED against it")
OUTPUT_NO_VERDICT = ("NOVERDICT", "NOBODY decided -- a tool fault escaped before "
                                  "any verdict existed")


def mark_output(out_path, marking):
    """Rename an undelivered output to `<name>.<TOKEN>.<ext>` and return the new path.

    The token precedes the extension so the file stays openable but is not
    counted as produced. Returns None, logging it, if the file is gone or the
    rename fails; a rename failure never masks the caller's exception.
    """
    token, why = marking
    if not path.exists(out_path):
        return None
    base, extension = path.splitext(out_path)
    marked = f"{base}.{token}{extension}"
    try:
        replace_file(out_path, marked)
    except OSError as error:
        sys.stderr.write(f"repair: the undelivered artefact could NOT be renamed "
                         f"and is still at its produced name: {error}\n")
        tools.log_line("repair: an undelivered artefact kept its produced name\n")
        return None
    sys.stderr.write(f"repair: the artefact was renamed to *.{token}{extension} "
                     f"-- {why} -- so it is inspectable and NOT counted as "
                     f"produced\n")
    return marked


def stable_case_key(candidate_path):
    """A stable 16-hex key derived from the candidate path (shared with `merge_video_repair`)."""
    import hashlib
    return hashlib.md5(candidate_path.encode()).hexdigest()[:16]


def apply_tail_exemption(delta_ms, exempted_ms):
    '''Deduct a stream's tail-gap exemption from its measured shortfall.

    Args:
        delta_ms: produced duration minus target duration (negative = short).
        exempted_ms: the stream's tail-gap shortfall, or None.

    Returns:
        (residual_delta_ms, deduction_ms). The deduction is capped at the
        measured shortfall; it is None when nothing was deducted (no exemption,
        or the stream is not short).
    '''
    if exempted_ms is None or delta_ms >= 0:
        return delta_ms, None
    deduction = min(exempted_ms, abs(delta_ms))
    return delta_ms + deduction, deduction


def container_grid_tolerance_ms(streams):
    '''How far the produced container duration may exceed the master's without extra content.

    A muxer never splits a frame, so the last block overruns the content by at
    most one frame: the tolerance is the larger of one video frame (exact
    `r_frame_rate`) and one audio frame plus the declared codec delay
    (`initial_padding`), which the DURATION tag also counts. Codecs without a
    fixed frame size contribute nothing.

    Returns:
        (tolerance_ms or None when nothing is measurable, detail string).
    '''
    video_frame_ms, audio_frame_ms = None, None
    audio_delay_ms = Decimal(0)
    audio_from = None
    for stream in streams:
        if stream.get("codec_type") == "video" and video_frame_ms is None:
            grid = parse_positive_rate(stream.get("frame_rate"))
            if grid is not None:
                video_frame_ms = Decimal(1000 * grid.denominator) / Decimal(grid.numerator)
        elif stream.get("codec_type") == "audio":
            samples = audio_codec_frame_samples.get(
                (stream.get("codec_name") or "").lower())
            rate = parse_positive_rate(stream.get("sample_rate"))
            if samples is None or rate is None:
                continue
            frame_ms = Decimal(1000 * samples * rate.denominator) / Decimal(rate.numerator)
            # Encoder priming samples sit before zero and are counted by the
            # DURATION tag, so the declared delay is added to the frame.
            delay_ms = Decimal(0)
            padding = str(stream.get("initial_padding") or "0")
            if padding.isdigit() and int(padding) > 0:
                delay_ms = (Decimal(1000 * int(padding) * rate.denominator)
                            / Decimal(rate.numerator))
            if audio_frame_ms is None or frame_ms + delay_ms > audio_frame_ms + audio_delay_ms:
                audio_frame_ms, audio_delay_ms, audio_from = (
                    frame_ms, delay_ms, stream.get("codec_name"))
    audio_bound_ms = (audio_frame_ms + audio_delay_ms) if audio_frame_ms is not None else None
    measured = [value for value in (video_frame_ms, audio_bound_ms) if value is not None]
    # Values contain no spaces: the log line is parsed as key=value tokens.
    detail = (f"video_frame_ms={video_frame_ms} "
              f"audio_frame_ms={audio_frame_ms} "
              f"audio_codec={audio_from or 'none_with_codec_fixed_frame_size'} "
              f"audio_codec_delay_ms={audio_delay_ms}")
    return (max(measured) if measured else None), detail


def verify_output_file(out_path, master_duration_ms, audio_reports,
                       subtitle_reports, tolerance_ms, master_container_ms):
    """Check the produced file itself: track counts, audio durations and container duration.

    Each audio track must be within `tolerance_ms` of `master_duration_ms` (the
    master's video duration, the assembly's target), after any tail-gap
    exemption. The container duration is compared to `master_container_ms` with
    a one-frame tolerance (`container_grid_tolerance_ms`), which catches e.g. a
    subtitle cue lengthening the file. Raises `chimeric_error` when enforcing
    and a check fails.
    """
    tools.dev_log(f"chimeric: verify_output_file starting out_path={out_path}\n")
    streams, container_ms = probe_output_streams(out_path)
    audio = [s for s in streams if s["codec_type"] == "audio"]
    subtitle = [s for s in streams if s["codec_type"] == "subtitle"]
    problems = []
    if len(audio) != len(audio_reports):
        problems.append(f"{len(audio_reports)} audio track(s) were built and "
                        f"{len(audio)} are in the file")
    if len(subtitle) != len(subtitle_reports):
        problems.append(f"{len(subtitle_reports)} subtitle track(s) were built and "
                        f"{len(subtitle)} are in the file")
    short, unmeasured, tail_exempted = [], [], []
    for position, stream in enumerate(audio):
        duration = stream["duration_ms"]
        if duration == None:
            # The container duration is not substituted: it is the maximum over
            # all streams and can be far from this track's.
            unmeasured.append({"index": stream["index"],
                               "language": stream["language"],
                               "reason": "the stream declares no duration; the "
                                         "container's is a different quantity "
                                         "and is not substituted"})
            continue
        delta = duration - Decimal(str(master_duration_ms))
        # The tail-gap exemption comes only from this stream's own report.
        exempted_ms = None
        if position < len(audio_reports):
            _tail_exempt = audio_reports[position].get("fill_short_by_ms_tail_exempt")
            if _tail_exempt not in (None, "", "0"):
                exempted_ms = Decimal(str(_tail_exempt))
        residual_delta, deduction_ms = apply_tail_exemption(delta, exempted_ms)
        if tolerance_ms != None and abs(residual_delta) > Decimal(str(tolerance_ms)):
            entry = {"index": stream["index"], "language": stream["language"],
                     "duration_ms": str(duration), "delta_ms": str(delta)}
            if deduction_ms != None:
                entry["residual_delta_ms"] = str(residual_delta)
                entry["tail_exempt_ms"] = str(deduction_ms)
            # Attach the fill source's own shortfall, if any, so a short source is
            # not read as lost content. Streams match reports by position (mux order).
            if position < len(audio_reports):
                fill_short = audio_reports[position].get("fill_short_by_ms")
                if fill_short not in (None, "", "0"):
                    entry["fill_short_by_ms"] = fill_short
            short.append(entry)
        elif deduction_ms != None:
            # Confirms on the produced file the exemption predicted at plan stage.
            tail_exempted.append({
                "index": stream["index"], "language": stream["language"],
                "delta_ms": str(delta), "exempted_ms": str(deduction_ms),
                "residual_delta_ms": str(residual_delta)})
            tools.log_line(
                f"chimeric: output_check_tail_exempt stream={stream['index']} "
                f"language={stream['language']} delta_ms={delta} "
                f"exempted_ms={deduction_ms} residual_delta_ms={residual_delta}\n")
    if len(short):
        problems.append("track(s) not running to the master's duration: "
                        + "; ".join(f"stream {s['index']} ({s['language']}) "
                                    f"{s.get('residual_delta_ms', s.get('delta_ms', s.get('reason')))}"
                                    + (f" [of measured {s['delta_ms']}, "
                                       f"{s['tail_exempt_ms']} ms tail-gap-exempted]"
                                       if s.get("tail_exempt_ms") else "")
                                    + (f" [fill source itself short by "
                                       f"{s['fill_short_by_ms']} ms]"
                                       if s.get("fill_short_by_ms") else "")
                                    for s in short))
    if len(unmeasured):
        problems.append("track(s) whose duration the file does not state: "
                        + "; ".join(f"stream {u['index']} ({u['language']})"
                                    for u in unmeasured))
    # The container must not run longer than the master's (shortness is handled
    # per track above); with either duration missing nothing is concluded.
    container_overshoot_ms, container_tolerance_ms = None, None
    container_refused = False
    tolerance_detail = ("video_frame_ms=None audio_frame_ms=None audio_codec=not_reached "
                        "audio_codec_delay_ms=None")
    if container_ms != None and master_container_ms != None:
        container_tolerance_ms, tolerance_detail = container_grid_tolerance_ms(streams)
        container_overshoot_ms = container_ms - Decimal(str(master_container_ms))
        if container_tolerance_ms != None and container_overshoot_ms > container_tolerance_ms:
            container_refused = True
            problems.append(
                f"the produced container runs {container_overshoot_ms} ms past "
                f"the master's own container ({container_ms} vs "
                f"{master_container_ms}), more than the {container_tolerance_ms} ms "
                f"a last indivisible block and its declared codec delay can explain "
                f"({tolerance_detail})")
    tools.log_line(
        f"chimeric: output_container_check container_ms={container_ms} "
        f"master_container_ms={master_container_ms} "
        f"overshoot_ms={container_overshoot_ms} "
        f"tolerance_ms={container_tolerance_ms} {tolerance_detail}\n")
    report = {"unmeasured": unmeasured,
              "tail_exempted": tail_exempted,
              "expected_duration_ms": str(master_duration_ms),
              "expected_duration_source": "master video Duration (mediainfo)",
              # The container duration is the max over all streams, subtitles
              # included, so the longest audio/video stream is reported beside it.
              "container_duration_ms": str(container_ms) if container_ms != None else None,
              "max_av_stream_duration_ms": (
                  str(max([s["duration_ms"] for s in streams
                           if s["duration_ms"] != None
                           and s["codec_type"] in ("video", "audio")] or [0]))
                  if streams else None),
              "master_container_duration_ms": (str(master_container_ms)
                                               if master_container_ms != None else None),
              "container_overshoot_ms": (str(container_overshoot_ms)
                                         if container_overshoot_ms != None else None),
              "container_tolerance_ms": (str(container_tolerance_ms)
                                         if container_tolerance_ms != None else None),
              "container_tolerance_detail": tolerance_detail,
              "audio_built": len(audio_reports), "audio_in_file": len(audio),
              "subtitles_built": len(subtitle_reports), "subtitles_in_file": len(subtitle),
              "tolerance_ms": str(tolerance_ms) if tolerance_ms != None else None,
              "streams": [{"index": s["index"], "codec_type": s["codec_type"],
                           "language": s["language"],
                           "duration_ms": str(s["duration_ms"]) if s["duration_ms"] != None else None}
                          for s in streams],
              "problems": problems}
    report["would_refuse"] = bool(len(problems))
    report["enforcing"] = bool(output_check_enforcing)
    report["measured"] = bool(len(streams)) and not len(unmeasured) and bool(
        [x for x in streams if x["codec_type"] == "audio"])
    if len(problems) and output_check_enforcing:
        # Cause token, when nothing but fill shortfalls is wrong:
        #   every measured fill source short by the same value
        #       -> master_audio_complement_short
        #   some short and some not, or short by different amounts
        #       -> master_duration_sources_disagree
        # otherwise -> output_check_mismatch.
        def _short_beyond_tolerance(fill_report):
            value = fill_report.get("fill_short_by_ms")
            if value in (None, "", "0"):
                return False
            if tolerance_ms == None:
                return True
            return abs(Decimal(str(value))) > Decimal(str(tolerance_ms))

        measured_fill_reports = [r for r in audio_reports
                                 if r.get("fill_source_ms") not in (None, "")]
        short_fill_reports = [r for r in measured_fill_reports
                              if _short_beyond_tolerance(r)]
        agreeing_fill_reports = [r for r in measured_fill_reports
                                 if not _short_beyond_tolerance(r)]
        nothing_else_wrong = (
            not len(unmeasured) and len(audio) == len(audio_reports)
            and len(subtitle) == len(subtitle_reports)
            and not container_refused)

        cause = "output_check_mismatch"
        if nothing_else_wrong and short_fill_reports:
            # Exact equality, no tolerance band; a single short report also counts.
            tied_values = {Decimal(str(r["fill_short_by_ms"]))
                           for r in short_fill_reports}
            if not agreeing_fill_reports and len(tied_values) == 1:
                cause = "master_audio_complement_short"
            else:
                cause = "master_duration_sources_disagree"
        error = chimeric_error("the produced file does not match what was built: "
                               + "; ".join(problems),
                               cause=cause)
        error.output_check = report
        raise error
    return report


def verify_on_master_timeline(out_path, master_obj, audio_reports, pieces,
                              tolerance_ms, search_ms, reference_stream=None, deadline=None,
                              envelope=False, comparison_language=None):
    """Probe each rebuilt track against the master and raise if it is off the master's timeline.

    Catches plan errors (e.g. a wrong offset sign) that per-piece checks cannot
    see. Per track the outcome is `aligned`, `misaligned`, `uncorrelated`,
    `inconsistent` or `skipped` (no master track in that language, or no signal).

    A non-comparison-language track that misaligns or fails to correlate carries
    the file's own geometry (one picture-led geometry per file, owner ruling):
    its own content, not the plan, is suspect, so it is flagged `"drop"` in its
    result entry instead of failing every other track. Only a misaligned or
    uncorrelated COMPARISON-language track still raises, since that is the
    plan-wide error this check exists to catch.

    Raises:
        chimeric_error: `alignment_contradicts_plan` or
            `delivery_offset_exceeds_tolerance` (comparison-language track only).
    """
    tools.dev_log(f"chimeric: verify_on_master_timeline starting "
                  f"out_path={out_path} master={master_obj.filePath}\n")
    started = time.monotonic()
    results = _verify_on_master_timeline(out_path, master_obj, audio_reports, pieces,
                                         tolerance_ms, search_ms, reference_stream, deadline,
                                         envelope, comparison_language)
    tools.log_line(f"chimeric: verify_on_master_timeline done seconds="
                   f"{round(time.monotonic() - started, 1)} tracks={len(results)} "
                   f"out_path={out_path}\n")
    return results


def _verify_on_master_timeline(out_path, master_obj, audio_reports, pieces, tolerance_ms,
                               search_ms, reference_stream, deadline, envelope=False,
                               comparison_language=None):
    """Body of `verify_on_master_timeline`.

    With `envelope`, windows are compared on their speech envelopes: an atempo
    delivery's waveform no longer correlates, its speech does.
    """
    probe_plan = choose_probe_positions(pieces, verify_window_seconds)
    positions = [start for _, start in probe_plan]
    if not len(positions):
        return [{"track": None, "outcome": "skipped",
                 "reason": "no candidate-sourced piece long enough to probe"}]

    window_ms = Decimal(str(verify_window_seconds)) * Decimal("1000")
    results = []
    produced_index = 0
    # Inconsistent pieces are decided after all tracks are measured: the
    # disagreement may be the master's own (`master_intertrack_explains`).
    deferred = []
    master_tracks = {}
    for report in audio_reports:
        language = report["language"]
        master_audio = find_master_audio_for_language(master_obj, language,
                                                     reference_stream)
        master_tracks[produced_index] = master_audio
        if master_audio == None:
            results.append({"track": report["stream_order"], "language": language,
                        "produced_index": produced_index,
                            "outcome": "skipped",
                            "reason": "the master has no track in this language"})
            produced_index += 1
            continue
        probes = []
        # A probe before the master stream's start is moved to that start if the
        # window still fits in the piece, otherwise recorded as `no_reference`.
        master_start_ms = get_stream_start_ms(master_audio)
        for piece_index, start in probe_plan:
            probe_start = start
            if master_start_ms > start:
                piece_end = pieces[piece_index]["master_end_ms"]
                shifted = Decimal(str(master_start_ms))
                if shifted + window_ms <= Decimal(str(piece_end)):
                    probe_start = shifted
                else:
                    probes.append({"master_position_ms": str(start),
                                   "piece": piece_index, "outcome": "no_reference",
                                   "reason": f"the master's track starts at "
                                             f"{master_start_ms} ms, after this probe"})
                    continue
            start = probe_start
            reference = read_reference_window(master_obj, master_audio, start, window_ms,
                                              verify_probe_rate, deadline=deadline)
            produced = read_mono_samples(out_path, f"0:a:{produced_index}",
                                         start, window_ms, verify_probe_rate,
                                         deadline=deadline)
            reference_rms = get_rms(reference)
            produced_rms = get_rms(produced)
            if min(reference_rms, produced_rms) < verify_min_rms:
                probes.append({"master_position_ms": str(start), "piece": piece_index,
                               "outcome": "no_signal",
                               "reference_rms": reference_rms,
                               "produced_rms": produced_rms})
                continue
            if envelope:
                import audio_walk
                reference = audio_walk.speech_envelope(reference, verify_probe_rate)
                produced = audio_walk.speech_envelope(produced, verify_probe_rate)
                reference, produced = reference - reference.mean(), produced - produced.mean()
            lag, score = measure_lag_ms(reference, produced, verify_probe_rate, search_ms)
            probes.append({"master_position_ms": str(start), "piece": piece_index,
                           "lag_ms": lag, "correlation": score,
                           "outcome": ("measured" if score >= verify_min_correlation
                                       else "below_correlation_floor"),
                           "reference_rms": reference_rms, "produced_rms": produced_rms})
        measured = [p for p in probes if p.get("outcome") == "measured"]
        weak = [p for p in probes if p.get("outcome") == "below_correlation_floor"]
        if len(weak):
            # With no probe over the floor: a borrowed offset is unmeasurable here,
            # while a measured one that no longer correlates is a delivery fault.
            verdict = ("partial" if len(measured)
                       else "unmeasurable" if report.get("offset_measured") == False
                       else "uncorrelated")
            tools.log_line(f"chimeric: verify track {report['stream_order']} ({language}) "
                           f"probe_below_correlation_floor probes={len(weak)}/{len(probes)} "
                           f"r={[round(p['correlation'], 4) for p in weak]} "
                           f"lags_ms={[p['lag_ms'] for p in weak]} "
                           f"floor={verify_min_correlation} "
                           f"offset={'measured' if report.get('offset_measured') != False else 'borrowed'}"
                           f"{'[' + str(report['borrow_reason']) + ']' if report.get('borrow_reason') else ''} "
                           f"verdict={verdict}\n")
            if verdict == "unmeasurable":
                results.append({"track": report["stream_order"], "language": language,
                                "produced_index": produced_index,
                                "outcome": "skipped",
                                "reason": "probe_below_correlation_floor: no probe correlates "
                                          "with the master's track and the offset was "
                                          "borrowed; the track is unverified, not verified",
                                "weakest_correlation": round(float(min(
                                    p["correlation"] for p in weak)), 4),
                                "correlation_floor": verify_min_correlation,
                                "probes": probes})
                produced_index += 1
                continue
            if verdict == "uncorrelated":
                results.append({"track": report["stream_order"], "language": language,
                                "produced_index": produced_index,
                                "outcome": "uncorrelated",
                                "strongest_correlation": round(float(max(
                                    p["correlation"] for p in weak)), 4),
                                "correlation_floor": verify_min_correlation,
                                "probes": probes})
                produced_index += 1
                continue
        if not len(measured):
            results.append({"track": report["stream_order"], "language": language,
                        "produced_index": produced_index,
                            "outcome": "skipped",
                            "reason": "no probe window carried signal; the track "
                                      "is unverified, not verified",
                            "probes": probes})
            produced_index += 1
            continue
        worst = max(abs(probe["lag_ms"]) for probe in measured)
        weakest = min(probe["correlation"] for probe in measured)
        # Quietest probe RMS relative to the floor: shows how close the kept
        # probes came to being discarded as `no_signal`.
        quietest = min(min(probe.get("reference_rms", float("inf")),
                           probe.get("produced_rms", float("inf")))
                       for probe in probes
                       if probe.get("reference_rms") != None
                       or probe.get("produced_rms") != None)
        rms_over_floor = (quietest / verify_min_rms
                          if quietest not in (None, float("inf")) else None)
        # Probes of one piece that disagree reveal a change point the plan missed.
        inconsistent = []
        for index in sorted({p["piece"] for p in measured}):
            same = [p["lag_ms"] for p in measured if p["piece"] == index]
            if len(same) > 1 and (max(same) - min(same)) > tolerance_ms:
                inconsistent.append({"piece": index, "spread_ms": max(same) - min(same),
                                     "lags_ms": same,
                                     "correlations": [round(p["correlation"], 4)
                                                      for p in measured
                                                      if p["piece"] == index]})
        if len(inconsistent):
            # The spread shows that a boundary exists, not its size; correlations
            # are shown so a weak probe is visible.
            detail = "; ".join(
                f"piece {c['piece']} probes disagree by {c['spread_ms']:.1f} ms "
                f"{c['lags_ms']} r={c['correlations']} (SPREAD, NOT THE SIZE OF "
                f"THE MISSED STEP: a "
                f"window straddling a boundary returns a displaced peak, so this "
                f"establishes THAT a boundary lies between the probes and not "
                f"how large it is)" for c in inconsistent)
            error = chimeric_error(
                f"the plan says one alignment holds across a piece and the track "
                f"says otherwise: {detail}. That is a change point the measurement "
                f"missed, not a splice error -- the track may be correctly aligned "
                f"on both sides of a boundary nobody modelled",
                cause="alignment_contradicts_plan")
            error.verification = results + [
                {"track": report["stream_order"], "language": language,
                 "produced_index": produced_index,
                 "outcome": "inconsistent", "inconsistent": inconsistent,
                 "probes": probes}]
            error.audios = audio_reports
            deferred.append({"error": error, "index": len(results),
                             "master_audio": master_audio, "measured": measured,
                             "entry": error.verification[-1]})
            results.append(error.verification[-1])
            produced_index += 1
            continue
        outcome = "aligned" if worst <= tolerance_ms else "misaligned"
        results.append({"track": report["stream_order"], "language": language,
                        "produced_index": produced_index,
                        "outcome": outcome, "worst_lag_ms": worst,
                        "weakest_correlation": round(float(weakest), 4),
                        "probes_measured": len(measured),
                        "probes_without_signal": len([p for p in probes
                                                      if p.get("outcome") == "no_signal"]),
                        "probes_below_correlation_floor": len(weak),
                        "quietest_probe_rms": quietest,
                        "rms_over_floor": (round(float(rms_over_floor), 2)
                                           if rms_over_floor != None else None),
                        "rms_floor": verify_min_rms,
                        "probes": probes})
        produced_index += 1

    for pending in deferred:
        resolved = master_intertrack_explains(master_obj, pending, results, master_tracks,
                                              reference_stream, window_ms, search_ms,
                                              tolerance_ms, deadline, envelope)
        if resolved is None:
            error = pending["error"]
            error.verification = results
            raise error
        results[pending["index"]] = resolved

    misaligned = [r for r in results if r["outcome"] in ("misaligned", "uncorrelated")]
    if len(misaligned):
        # The comparison-language track carries the plan's own geometry: its misalignment
        # means the plan itself is wrong (a sign error, a missed step) and stays fatal. Any
        # other track's misalignment is that track's own content, not the file's single
        # picture-led geometry -- it is dropped by name instead of failing the whole build.
        plan_wide = [r for r in misaligned
                    if comparison_language is not None and r["language"] == comparison_language]
        if plan_wide:
            detail = "; ".join(
                f"track {r['track']} ({r['language']}) off by {r['worst_lag_ms']:.1f} ms"
                if r["outcome"] == "misaligned" else
                f"track {r['track']} ({r['language']}) correlates nowhere with the master "
                f"track its offset was measured on (strongest r={r['strongest_correlation']}, "
                f"floor {r['correlation_floor']})" for r in plan_wide)
            error = chimeric_error(
                f"the rebuilt track is not on the master's timeline: {detail}. "
                f"Tolerance {tolerance_ms} ms. The plan is wrong, not the splice: a "
                f"uniform offset means the base offset carries the wrong sign, and a "
                f"residual that changes at a change point means a step was missed",
                cause="delivery_offset_exceeds_tolerance")
            error.verification = results
            error.audios = audio_reports
            raise error
        for r in misaligned:
            r["drop"] = True
            tools.log_always(
                f"chimeric: track_dropped_misaligned stream={r['track']} lang={r['language']} "
                f"outcome={r['outcome']} tolerance_ms={tolerance_ms} -- not the comparison "
                f"language, so the file's own geometry is not in doubt; dropped by name, the "
                f"rest of the build is unaffected\n")
    return results


def _log_mel_z(samples, rate, n_fft=2048, hop=441, n_mels=40):
    '''Log-mel spectrogram (n_mels x frames), z-scored per band.'''
    import numpy
    import scipy.signal
    _, _, spectrum = scipy.signal.stft(samples, fs=rate, nperseg=n_fft,
                                       noverlap=n_fft - hop, boundary=None)
    magnitude = numpy.abs(spectrum)

    def hz_to_mel(hz):
        return 2595 * numpy.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_points = numpy.linspace(hz_to_mel(50), hz_to_mel(rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = numpy.floor((n_fft + 1) * hz_points / rate).astype(int)
    bank = numpy.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, centre):
            if centre > left:
                bank[i - 1, k] = (k - left) / (centre - left)
        for k in range(centre, right):
            if right > centre:
                bank[i - 1, k] = (right - k) / (right - centre)
    log_mel = numpy.log1p(bank @ magnitude)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-8
    return (log_mel - mean) / std


def _ncc_at_zero_offset(a, b):
    '''Normalised cross-correlation of two spectrograms at zero offset; None if too short or silent.

    No lag search: a master fill is at its own master position by construction.
    '''
    import numpy
    length = min(a.shape[1], b.shape[1])
    if length < 2:
        return None
    flat_a, flat_b = a[:, :length].flatten(), b[:, :length].flatten()
    norm_a, norm_b = numpy.linalg.norm(flat_a), numpy.linalg.norm(flat_b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    return float(numpy.dot(flat_a, flat_b) / (norm_a * norm_b))


fill_content_ncc_floor = 0.85  # log-mel NCC floor for a matching fill, fixed, not derived per run.
fill_content_control_min_offset_ms = Decimal("60000")  # minimum distance of the control window.


def _fill_control_window_ms(start_ms, span_ms, master_duration_ms):
    '''Start of a deliberately mismatched control window of the master, far from `start_ms`.'''
    offset = max(fill_content_control_min_offset_ms, span_ms * 4)
    forward = start_ms + offset
    if forward + span_ms <= master_duration_ms:
        return forward
    backward = start_ms - offset
    if backward >= 0:
        return backward
    if start_ms > (master_duration_ms - (start_ms + span_ms)):
        return Decimal("0")
    return max(Decimal("0"), master_duration_ms - span_ms)


def master_intertrack_explains(master_obj, pending, results, master_tracks, reference_stream,
                               window_ms, search_ms, tolerance_ms, deadline, envelope=False):
    """Check whether a piece's disagreeing probes come from a step between the master's own tracks.

    An aligned sibling track gives, per probe, the master's relation between that
    track and this one; each lag is re-read as `lag + (relation - relation_anchor)`
    (anchor: the probe closest to zero). If every re-read lag is within
    `tolerance_ms`, the master's track moved and the track is aligned.

    Returns:
        The replacement `aligned` result (with `master_intertrack_step`), or None
        when untestable or not explained, so the refusal stands.
    """
    own = pending["master_audio"]
    entry = pending["entry"]
    siblings = [r for r in results
                if r.get("outcome") == "aligned"
                and master_tracks.get(r.get("produced_index")) != None
                and str(master_tracks[r["produced_index"]].get("StreamOrder"))
                != str(own.get("StreamOrder"))]
    if reference_stream != None:
        siblings.sort(key=lambda r: str(master_tracks[r["produced_index"]].get("StreamOrder"))
                      != str(reference_stream))
    if not len(siblings):
        tools.log_line(f"chimeric: verify track {entry['track']} ({entry['language']}) "
                       f"master_intertrack_step not_testable reason=no_aligned_sibling\n")
        return None
    sibling = siblings[0]
    sibling_audio = master_tracks[sibling["produced_index"]]
    sibling_probes = {p["master_position_ms"]: p for p in sibling.get("probes", [])
                      if p.get("outcome") == "measured"}
    rows = []
    for probe in pending["measured"]:
        if probe["master_position_ms"] not in sibling_probes:
            tools.log_line(f"chimeric: verify track {entry['track']} ({entry['language']}) "
                           f"master_intertrack_step not_testable reason=sibling_probe_missing "
                           f"at={probe['master_position_ms']}\n")
            return None
        start = Decimal(probe["master_position_ms"])
        theirs = read_mono_samples(master_obj.filePath, f"0:{int(sibling_audio['StreamOrder'])}",
                                   start, window_ms, verify_probe_rate, deadline=deadline)
        ours = read_mono_samples(master_obj.filePath, f"0:{int(own['StreamOrder'])}",
                                 start, window_ms, verify_probe_rate, deadline=deadline)
        if min(get_rms(theirs), get_rms(ours)) < verify_min_rms:
            tools.log_line(f"chimeric: verify track {entry['track']} ({entry['language']}) "
                           f"master_intertrack_step not_testable reason=no_signal "
                           f"at={probe['master_position_ms']}\n")
            return None
        if envelope:
            import audio_walk
            theirs = audio_walk.speech_envelope(theirs, verify_probe_rate)
            ours = audio_walk.speech_envelope(ours, verify_probe_rate)
            theirs, ours = theirs - theirs.mean(), ours - ours.mean()
        relation, score = measure_lag_ms(theirs, ours, verify_probe_rate, search_ms)
        rows.append((probe, relation, score))
    anchor = min(rows, key=lambda row: abs(row[0]["lag_ms"]))
    reread = [row[0]["lag_ms"] + (row[1] - anchor[1]) for row in rows]
    worst = max(abs(value) for value in reread)
    step = {"sibling_track": sibling["track"], "sibling_language": sibling["language"],
            "master_sibling_stream": int(sibling_audio["StreamOrder"]),
            "master_stream": int(own["StreamOrder"]),
            "anchor_position_ms": anchor[0]["master_position_ms"],
            "positions_ms": [row[0]["master_position_ms"] for row in rows],
            "lags_ms": [row[0]["lag_ms"] for row in rows],
            "relation_ms": [row[1] for row in rows],
            "relation_correlations": [round(row[2], 4) for row in rows],
            "reread_lags_ms": [round(value, 3) for value in reread]}
    verdict = "explained" if worst <= tolerance_ms else "not_explained"
    tools.log_always(f"chimeric: verify track {entry['track']} ({entry['language']}) "
                     f"master_intertrack_step {verdict} sibling={sibling['track']} "
                     f"({sibling['language']}) positions_ms={step['positions_ms']} "
                     f"lags_ms={step['lags_ms']} master_relation_ms={step['relation_ms']} "
                     f"r={step['relation_correlations']} reread_lags_ms={step['reread_lags_ms']} "
                     f"tolerance_ms={tolerance_ms}\n")
    if verdict != "explained":
        return None
    measured = pending["measured"]
    return {"track": entry["track"], "language": entry["language"],
            "produced_index": entry["produced_index"],
            "outcome": "aligned", "worst_lag_ms": worst,
            "weakest_correlation": round(float(min(p["correlation"] for p in measured)), 4),
            "probes_measured": len(measured),
            "probes_without_signal": len(entry["probes"]) - len(measured),
            "master_intertrack_step": step,
            "inconsistent_against_master_track": entry["inconsistent"],
            "probes": entry["probes"]}


def verify_fill_content(out_path, master_obj, audio_reports, master_duration_ms,
                        deadline=None):
    '''Check that each master-filled region of the output carries the master's content there.

    Only records; raises nothing except a budget refusal. Each region's log-mel
    NCC against the master at the same position is compared with a mismatched
    control window, using `fill_content_ncc_floor`:
    control >= floor -> `content_indiscriminate`; reading >= floor ->
    `content_verified`; otherwise `content_mismatch`. Failures give
    `skipped_unmeasurable`, silence `skipped_silent`.
    '''
    tools.dev_log(f"chimeric: verify_fill_content starting out_path={out_path} "
                  f"master={master_obj.filePath}\n")
    results = []
    for produced_index, report in enumerate(audio_reports):
        if report.get("gap_fill") != "master":
            continue
        fill_stream_order = report.get("fill_stream_order")
        if fill_stream_order is None:
            continue
        regions = []
        for region in report.get("filled_regions") or []:
            if region.get("source") != "master":
                continue
            try:
                start_ms = Decimal(str(region["master_start_ms"]))
                end_ms = Decimal(str(region["master_end_ms"]))
            except Exception:
                regions.append({"master_start_ms": region.get("master_start_ms"),
                               "master_end_ms": region.get("master_end_ms"),
                               "outcome": "skipped_unmeasurable",
                               "reason": "region bounds unreadable"})
                continue
            span_ms = end_ms - start_ms
            if span_ms <= 0:
                regions.append({"master_start_ms": str(start_ms), "master_end_ms": str(end_ms),
                               "outcome": "skipped_unmeasurable", "reason": "degenerate span"})
                continue
            control_start_ms = _fill_control_window_ms(start_ms, span_ms, master_duration_ms)
            entry = {"master_start_ms": str(start_ms), "master_end_ms": str(end_ms),
                    "fill_source_class": region.get("fill_source_class")}
            try:
                produced_samples = read_mono_samples(
                    out_path, f"0:a:{produced_index}", start_ms, span_ms, verify_probe_rate,
                    deadline=deadline)
                reading_samples = read_mono_samples(
                    master_obj.filePath, f"0:{fill_stream_order}", start_ms, span_ms,
                    verify_probe_rate, deadline=deadline)
                control_samples = read_mono_samples(
                    master_obj.filePath, f"0:{fill_stream_order}", control_start_ms, span_ms,
                    verify_probe_rate, deadline=deadline)
            except Exception as error:
                if getattr(error, "cause", None) == "repair_budget_exceeded":
                    raise
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": f"extraction failed: {type(error).__name__}"})
                regions.append(entry)
                continue
            if (get_rms(produced_samples) < verify_min_rms
                   or get_rms(reading_samples) < verify_min_rms):
                entry["outcome"] = "skipped_silent"
                regions.append(entry)
                continue
            try:
                reading_ncc = _ncc_at_zero_offset(
                    _log_mel_z(produced_samples, verify_probe_rate),
                    _log_mel_z(reading_samples, verify_probe_rate))
                control_ncc = _ncc_at_zero_offset(
                    _log_mel_z(produced_samples, verify_probe_rate),
                    _log_mel_z(control_samples, verify_probe_rate))
            except Exception as error:
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": f"NCC computation failed: {type(error).__name__}"})
                regions.append(entry)
                continue
            if reading_ncc is None or control_ncc is None:
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": "degenerate window for NCC"})
                regions.append(entry)
                continue
            if control_ncc >= fill_content_ncc_floor:
                outcome = "content_indiscriminate"
            elif reading_ncc >= fill_content_ncc_floor:
                outcome = "content_verified"
            else:
                outcome = "content_mismatch"
            entry.update({"outcome": outcome, "reading_ncc": round(reading_ncc, 4),
                         "control_ncc": round(control_ncc, 4),
                         "separation": round(reading_ncc - control_ncc, 4)})
            regions.append(entry)
        if regions:
            results.append({"track": report["stream_order"], "produced_index": produced_index,
                           "regions": regions})
    return results
