'''
Audio conversion options for ffmpeg, and the view of a re-encoded track that
keep_best_audio compares.

generate_audio_ffmpeg_conversion_cmd appends, for one audio track:
- lossless source (MediaInfo Compression_Mode) -> FLAC, unless
  config.json's encode_lossless_in_flac is false;
- negative delay, a forced conversion or an extra filter -> pick_target's
  answer for the source codec (which may itself be FLAC, when the matched
  leaf says so);
- otherwise -> copy.

config.json's `audio_reencode` table:

    "audio_reencode": {
        "encode_lossless_in_flac": true,
        "flac_args": ["-compression_level", "12", "-exact_rice_parameters", "1"],
        "targets": {"<encoder>": {
            "sample_rates": [8000, 12000, 16000, 24000, 48000],
            "filters": {"<channel count>": "<ffmpeg filter>"}
        }},
        "sources": {"<ffprobe codec_name>": {
            "target": "<encoder>",
            "bitrate": {"<threshold>": {
                "sample_rate": {"<threshold>": {
                    "channels": {"<threshold>": {
                        "target": "<encoder, optional, overrides the source's>",
                        "bitrate": "192k" | "source",
                        "args": [...]
                    }}
                }}
            }}
        }}
    }

Four stages per lossy source: codec, bitrate, sample rate, channel count.
Each stage is a {"threshold": ...} dict: the rule picked is the one under
the highest threshold at or below the measured value, the lowest threshold
when the measurement is missing (select_by_threshold never raises).

`sample_rates` on a target names the rates its encoder accepts without a
resample; a measured rate outside that list gets one `aresample=<rate>`
filter (ffmpeg's default resampler -- soxr precision 28 measured worse on a
real null test, see AUDIO_FLAC_VS_OPUS_20261019.md). A channel-count-keyed
filter cannot tell two layouts of the same count apart (e.g. "5.1" from
"5.1(side)"); measured safe on real 5.1(side) tracks, see the commit log.
'''

import sys

import tools
import video

DEFAULT_FLAC_ARGS = ["-compression_level", "12", "-exact_rice_parameters", "1"]
NEGATIVE_DELAY_FILTER = "aresample=async=1:first_pts=0"

_checked_table = None


def parse_bitrate(value):
    '''Return a bitrate ("192k", "1.5M", 192000) in bit/s, or None.'''
    if value is None:
        return None
    text = str(value).strip().lower()
    scale = {"k": 1000, "m": 1000000}.get(text[-1:], 1)
    try:
        number = int(float(text.rstrip("km")) * scale)
    except ValueError:
        return None
    return number if number >= 0 else None


def select_by_threshold(stage, measured):
    '''Return the entry of `stage` (a {"threshold": entry} dict) whose key is
    the highest number at or below `measured`, or the lowest key's entry
    when `measured` is missing or below every key. None for an empty stage.
    '''
    if not stage:
        return None
    try:
        measured_value = float(measured)
    except (TypeError, ValueError):
        measured_value = None
    best_key, best_entry = None, None
    if measured_value is not None:
        for key, entry in stage.items():
            try:
                threshold = float(key)
            except (TypeError, ValueError):
                continue
            if threshold <= measured_value and (best_key is None or threshold > float(best_key)):
                best_key, best_entry = key, entry
    if best_key is None:
        best_key = min(stage, key=lambda k: float(k))
        best_entry = stage[best_key]
    return best_entry


def audio_rules():
    """Return the `audio_reencode` table of tools.configuration ({} when unset)."""
    global _checked_table
    table = (getattr(tools, "configuration", None) or {}).get("audio_reencode") or {}
    _checked_table = table
    return table


def is_lossless(audio):
    '''Return True when the source stream is lossless.

    A track already carrying an ORIGINAL_ENCODE tag went through a lossy
    re-encode and can never be lossless again. Otherwise, MediaInfo's
    Compression_Mode decides. A track with neither reads as not lossless.
    '''
    if original_encode_tag(audio) is not None:
        return False
    return audio.get("Compression_Mode") == "Lossless"


def build_resample_filter(target, measured_rate):
    '''Return an `aresample=<rate>` filter when `measured_rate` is outside
    `target`'s "sample_rates", else None.'''
    allowed = target.get("sample_rates")
    if not allowed or measured_rate is None:
        return None
    try:
        measured_rate = int(float(measured_rate))
    except (TypeError, ValueError):
        return None
    if measured_rate in allowed:
        return None
    higher = sorted(rate for rate in allowed if rate >= measured_rate)
    target_rate = higher[0] if higher else max(allowed)
    return f"aresample={target_rate}"


def resolve_rule_stages(stage, measurements):
    '''Recursively resolve a nested {"threshold": {...}} table and return the
    leaf parameter dict.

    `measurements` is the ordered list of (measured value, next stage's key)
    still to resolve, e.g. [(bitrate, "sample_rate"), (rate, "channels"),
    (channels, None)]. select_by_threshold resolves this level; a next-stage
    name recurses into it, None means this level already is the leaf.
    '''
    measured, next_name = measurements[0]
    entry = select_by_threshold(stage, measured) or {}
    if next_name is None:
        return entry
    return resolve_rule_stages(entry.get(next_name, {}), measurements[1:])


def pick_target(audio, path=None):
    '''Return (codec, bitrate, extra args, filter) to convert a lossy track.

    The single entry point for the table's codec -> bitrate -> sample rate
    -> channels selection. The matched leaf's own "target" overrides the
    source's usual one (e.g. a near-lossless DTS leaf naming "flac"). path
    is the file holding the track, for video.get_bitrate.

    A codec absent from the table is returned unchanged:
    (codec, source bitrate, [], None).
    '''
    table = audio_rules()
    codec = video.get_codec(audio)
    try:
        source_bitrate = video.get_bitrate(audio, path)
    except Exception:
        source_bitrate = None
    sample_rate = video.get_sample_rate(audio)
    channels = video.get_channel(audio)
    source_entry = table.get("sources", {}).get(codec)
    if not source_entry:
        return codec, source_bitrate, [], None

    params = resolve_rule_stages(source_entry.get("bitrate", {}),
                                 [(source_bitrate, "sample_rate"), (sample_rate, "channels"), (channels, None)])
    encoder = params.get("target", source_entry.get("target"))
    target = table.get("targets", {}).get(encoder, {})

    bitrate = source_bitrate if params.get("bitrate", "source") == "source" else parse_bitrate(params.get("bitrate"))
    args = list(params.get("args", []))
    channel_filter = target.get("filters", {}).get(str(channels)) if channels is not None else None
    resample_filter = build_resample_filter(target, sample_rate)
    table_filter = ",".join(f for f in (channel_filter, resample_filter) if f) or None
    return encoder, bitrate, args, table_filter


def original_encode_tag(audio):
    '''Return the track's EXISTING ORIGINAL_ENCODE tag, or None.

    Reads only a tag already on record, never derives one -- None means no
    lossy re-encode has happened to this track yet. measured_encode_tag
    derives the value a first lossy re-encode writes.
    '''
    return (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG) or None


def measured_encode_tag(audio, path=None):
    '''Return "codec:bitrate:sample_rate" from the track's current measured
    stream, or None when codec or sample rate cannot be measured.'''
    try:
        codec = video.get_codec(audio)
        sample_rate = video.get_sample_rate(audio)
        bitrate = video.get_bitrate(audio, path)
        if not codec or sample_rate is None:
            raise ValueError(f"codec={codec!r} sample_rate={sample_rate!r}")
        return f"{codec}:{int(bitrate)}:{int(sample_rate)}"
    except Exception as e:
        sys.stderr.write(f"No ORIGINAL_ENCODE for track {audio.get('StreamOrder')}: {e}\n")
        return None


def audio_for_keep(audio):
    """Return the audio dict keep_best_audio compares.

    A tagged track is returned as a copy carrying its original Format,
    bitrate (scaled by config.json's reencode_keep_handicap, 1.0 when unset)
    and SamplingRate. A lossless or untagged track is returned unchanged.
    """
    if is_lossless(audio):
        return audio
    configuration = getattr(tools, "configuration", None) or {}
    tag = original_encode_tag(audio)
    if tag is None:
        return audio
    try:
        codec, bitrate, sampling_rate = str(tag).split(":")
        handicap = float(configuration.get("reencode_keep_handicap", 1.0))
        bitrate = str(int(round(int(float(bitrate)) * handicap)))
        sampling_rate = str(int(float(sampling_rate)))
    except (KeyError, ValueError, TypeError):
        return audio
    codec = codec.strip().lower()
    if not codec:
        return audio
    compared = dict(audio)
    if codec.startswith("pcm_"):
        compared["Format"] = "PCM"
    else:
        compared["Format"] = configuration.get("original_encode_formats", {}).get(codec, codec.upper())
    compared["BitRate"] = bitrate
    compared.pop("BitRate_Nominal", None)
    compared["SamplingRate"] = sampling_rate
    if isinstance(audio.get("ffprobe"), dict):
        compared["ffprobe"] = dict(audio["ffprobe"], bit_rate=bitrate)
    compared["BitRate_resolved"] = int(bitrate)
    compared["BitRate_source"] = tools.ORIGINAL_ENCODE_TAG
    return compared


def default_stream_specifier(audio):
    """Return the output stream specifier of an audio track (":a:N")."""
    return f":a:{int(audio.get('@typeorder', 1)) - 1}"


def flac_options(audio, args):
    '''Return the -c flac options for this track: compression args plus the
    sample format its BitDepth requires.'''
    sample_fmt = "s16" if audio.get("BitDepth") == "16" else "s32"
    return ["-c", "flac", "-sample_fmt", sample_fmt] + list(args)


def build_flac_conversion(audio, table):
    '''Return (options, filter=None) to re-encode a lossless track to FLAC.
    No ORIGINAL_ENCODE tag: no loss happens here.'''
    return flac_options(audio, table.get("flac_args") or DEFAULT_FLAC_ARGS), None


def build_lossy_conversion(audio, path):
    '''Return (options, filter, tag) for pick_target's answer.

    When the leaf names FLAC (a config override, e.g. near-lossless DTS),
    this is a FLAC conversion and gets no tag either -- otherwise the
    ORIGINAL_ENCODE tag is written (an existing one kept, else measured).
    '''
    codec, bitrate, args, rule_filter = pick_target(audio, path)
    if codec == "flac":
        return flac_options(audio, args), rule_filter, None
    options = ["-c", codec] + (["-b", str(bitrate)] if bitrate else []) + args
    tag = original_encode_tag(audio) or measured_encode_tag(audio, path)
    return options, rule_filter, tag


def stream_qualify(options, stream):
    '''Return `options` with every "-flag" token suffixed by `stream`
    ("-c" -> "-c:a:0"); a plain value token is left alone.'''
    return [f"{token}{stream}" if str(token).startswith("-") else str(token) for token in options]


def build_filter_chain(rule_filter, delay_to_put, other_filter_param):
    '''Return the comma-joined filter chain (table filter, the negative-delay
    aresample, the caller's own filter), or None when all three are empty.'''
    filters = [rule_filter, NEGATIVE_DELAY_FILTER if delay_to_put < 0 else None, other_filter_param]
    return ",".join(f for f in filters if f) or None


def generate_audio_ffmpeg_conversion_cmd(base_cmd, audio, delay_to_put, other_filter_param="",
                                         other_param="", force_conversion=False,
                                         filters_inline=True, stream=None, path=None):
    """Append to base_cmd the options that copy or convert one audio track.

    other_param (str or list) is appended as given. stream is the output
    stream specifier, ":a:<@typeorder - 1>" by default. path is the file
    holding the track, for a bitrate the metadata lacks.

    filters_inline=True appends the filter chain as one -filter<stream> and
    returns True when the track is converted. filters_inline=False (for a
    filtergraph output, which refuses -filter) leaves it out and returns
    (args, filter_string) instead: the options appended to base_cmd, and the
    chain for the caller's own filtergraph (None when empty).
    """
    stream = stream or default_stream_specifier(audio)
    table = audio_rules()
    tag = None
    if is_lossless(audio) and table.get("encode_lossless_in_flac", True):
        options, rule_filter = build_flac_conversion(audio, table)
    elif delay_to_put < 0 or force_conversion or other_filter_param:
        options, rule_filter, tag = build_lossy_conversion(audio, path)
    elif filters_inline:
        return False
    else:
        return [], None

    args = stream_qualify(options, stream)
    joined = build_filter_chain(rule_filter, delay_to_put, other_filter_param)
    if joined and filters_inline:
        args.extend([f"-filter{stream}", joined])
    if tag:
        args.extend([f"-metadata:s{stream}", f"{tools.ORIGINAL_ENCODE_TAG}={tag}"])
    if isinstance(other_param, str):
        args.extend(other_param.split())
    elif other_param:
        args.extend(other_param)
    base_cmd.extend(args)
    if filters_inline:
        return True
    return args, joined
