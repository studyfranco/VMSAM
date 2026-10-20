'''
Audio conversion options for ffmpeg, and the view of a re-encoded track that
keep_best_audio compares.

One entry point, generate_audio_ffmpeg_conversion_cmd, decides for one audio
track whether to copy or convert it, and appends the per-stream options:

- lossless source -> the `lossless` codec of the table (FLAC);
- negative delay or forced conversion -> the target of the source codec in
  the table (a codec absent from `sources` takes `default`: its own codec at
  the source bitrate);
- otherwise -> copy.

The table is the `audio_reencode` key of config.json, read from
tools.configuration (set by the loaders). Layout:

    "audio_reencode": {
        "lossless": {"codec": "flac", "args": [...]},
        "sources": {
            "<ffprobe codec_name>": {"target": "<encoder>", "args": [...], "rules": [
                {"channels": [min, max], "source_bitrate": ["min", "max"],
                 "bitrate": "192k" | "source" | {"floor": "160k", "ceiling": "192k"},
                 "args": [...], "filters": {"<ffprobe channel_layout>": "<filter>"}}]}},
        "default": {"target": "same", "rules": [{"bitrate": "source"}]}
    }

`channels` is inclusive; `source_bitrate` is min inclusive, max exclusive,
null max = no limit; a missing bound = no limit. The first matching rule wins.
`{"floor", "ceiling"}` clamps the source bitrate (an unknown one takes the
floor, else the ceiling). The rules of a source must cover 1 to 8 channels and
0 to infinity; a track no rule covers keeps the source bitrate.
'''

import math
import re
import sys

import tools
import video

# master's `video.py` is frozen and carries neither `get_channel` nor the `path`-aware
# `get_bitrate(data, path)` dev-AI's video.py has; these two small, side-effect-free
# compatibility shims stand in for them here (an open module), so the frozen file is
# never touched. `_get_bitrate_compat` drops the `copy_pass` (probe-the-file) fallback
# tier dev-AI's version adds via `path` -- a real but minor loss: the metadata-only
# tiers (ffprobe/BitRate/BitRate_Nominal/StreamSize/BPS) still run.

def _get_bitrate_compat(audio, path=None):
    '''`video.get_bitrate`, tolerant of a `video.py` without the `path` parameter.'''
    try:
        return video.get_bitrate(audio, path)
    except TypeError:
        return video.get_bitrate(audio)


def _get_channel_compat(audio):
    '''Number of channels of an audio track, or None -- `video.get_channel` reimplemented
    here because master's `video.py` does not carry it (same logic, same field order).'''
    if hasattr(video, "get_channel"):
        return video.get_channel(audio)
    ffprobe = audio.get('ffprobe') or {}
    for value in (audio.get('Channels'), audio.get('Channel(s)'), audio.get('Channels_Original'),
                  ffprobe.get('channels')):
        try:
            return int(str(value).split('/')[0])
        except (TypeError, ValueError):
            pass
    layout = audio.get('ChannelLayout') or ffprobe.get('channel_layout')
    if not layout:
        return None
    layout = str(layout).split('(')[0].strip().lower()
    named = {'mono': 1, 'stereo': 2, 'downmix': 2, 'quad': 4, 'hexagonal': 6, 'octagonal': 8}
    if layout in named:
        return named[layout]
    if re.fullmatch(r"\d+(\.\d+)+", layout):
        # ffprobe "5.1", "7.1.4": the numbers add up
        return sum(int(number) for number in layout.split('.'))
    # MediaInfo "L R C LFE Ls Rs": one position per channel
    return len(layout.split())


LOSSLESS_FORMATS = ("flac", "pcm")
NEGATIVE_DELAY_FILTER = "aresample=async=1:first_pts=0"
ORIGINAL_ENCODE_TAG = "ORIGINAL_ENCODE"

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
    if number < 0:
        return None
    return number


def _bounds(rule):
    '''Return ((min, max) channels, (min, max) source bitrate) of a rule.'''
    low_ch, high_ch = rule.get("channels", [1, None])
    low_br, high_br = rule.get("source_bitrate", [0, None])
    if high_ch is None:
        high_ch = math.inf
    if high_br is None:
        high_br = math.inf
    else:
        high_br = parse_bitrate(high_br)
    return (int(low_ch), high_ch), (parse_bitrate(low_br), high_br)


def _check_coverage(name, rules):
    '''Raise ValueError when `rules` leave a gap over 1-8 channels and 0-infinity.'''
    spans = [_bounds(rule) for rule in rules]
    for channels in range(1, 9):
        reach = 0
        for low, high in sorted(br for ch, br in spans if ch[0] <= channels <= ch[1]):
            if low is None or high is None or low > reach:
                break
            reach = max(reach, high)
        if reach != math.inf:
            raise ValueError(f"audio_reencode: source {name}: {channels} channels "
                             f"not covered from {reach} bit/s")


def audio_rules():
    """Return the `audio_reencode` table of tools.configuration, its coverage checked once.

    Without a configuration or a table, every codec keeps its own.
    """
    global _checked_table
    table = (getattr(tools, "configuration", None) or {}).get("audio_reencode") or {}
    if table is not _checked_table:
        for name, entry in table.get("sources", {}).items():
            _check_coverage(name, entry.get("rules", []))
        if "default" in table:
            _check_coverage("default", table["default"].get("rules", []))
        _checked_table = table
    return table


def is_lossless(audio):
    '''Return True for a FLAC or PCM track, or a MediaInfo Compression_Mode "Lossless".'''
    if str(audio.get("Format", "")).lower() in LOSSLESS_FORMATS:
        return True
    return audio.get("Compression_Mode") == "Lossless"


def pick_target(audio, path=None):
    '''Return (codec, bitrate, extra args, filter) to convert a lossy track.

    path is the file holding the track, to measure a bitrate the metadata lacks.

    bitrate is in bit/s, or None to let the encoder choose; filter is None
    when the matching rule has none for this channel layout.
    '''
    table = audio_rules()
    ffprobe = audio.get("ffprobe") or {}
    codec = str(ffprobe.get("codec_name", "")).lower()
    try:
        source = _get_bitrate_compat(audio, path)
    except Exception as e:
        tools.dev_log(f"audio_reencode: no source bitrate for track {audio.get('StreamOrder')}: {e}\n")
        source = None
    entry = table.get("sources", {}).get(codec, table.get("default", {}))
    target = entry.get("target", "same")
    if target == "same":
        target = codec
    channels = _get_channel_compat(audio)
    # An unknown source bitrate matches the rule without upper bound.
    compared = sys.maxsize if source is None else source
    rule = {}
    for candidate in entry.get("rules", []):
        (low_ch, high_ch), (low_br, high_br) = _bounds(candidate)
        if channels is not None and low_ch <= channels <= high_ch and low_br <= compared < high_br:
            rule = candidate
            break
    bitrate = rule.get("bitrate", "source")
    if bitrate == "source":
        bitrate = source
    elif isinstance(bitrate, dict):
        floor = parse_bitrate(bitrate.get("floor"))
        ceiling = parse_bitrate(bitrate.get("ceiling"))
        bitrate = source
        if bitrate is None:
            bitrate = floor or ceiling
        if floor is not None and bitrate is not None and bitrate < floor:
            bitrate = floor
        if ceiling is not None and bitrate is not None and bitrate > ceiling:
            bitrate = ceiling
    else:
        bitrate = parse_bitrate(bitrate)
    args = list(entry.get("args", [])) + list(rule.get("args", []))
    rule_filter = rule.get("filters", {}).get(ffprobe.get("channel_layout"))
    return target, bitrate, args, rule_filter


def original_encode_tag(audio, path=None):
    '''Return `codec:bitrate:sample_rate` of the track, or None when unknown.

    A track already tagged keeps its first value.
    '''
    value = (audio.get("extra") or {}).get(ORIGINAL_ENCODE_TAG)
    if value:
        return value
    try:
        codec = audio["ffprobe"]["codec_name"].lower()
        sample_rate = audio.get("SamplingRate") or audio["ffprobe"]["sample_rate"]
        return f"{codec}:{_get_bitrate_compat(audio, path)}:{int(float(sample_rate))}"
    except Exception as e:
        sys.stderr.write(f"No ORIGINAL_ENCODE for track {audio.get('StreamOrder')}: {e}\n")
        tools.logs.append(f"No ORIGINAL_ENCODE for track {audio.get('StreamOrder')}: {e}\n")
        return None


def audio_for_keep(audio):
    """Return the audio dict keep_best_audio compares.

    A track tagged ORIGINAL_ENCODE=<codec>:<bitrate>:<sample_rate> is returned as a
    copy with the Format, bitrate and SamplingRate of its original encoding, the
    bitrate multiplied by `reencode_keep_handicap` of config.json (1.0 when
    unset) and the Format named by `original_encode_formats`. Any other track is
    returned unchanged.
    """
    configuration = getattr(tools, "configuration", None) or {}
    try:
        codec, bitrate, sampling_rate = str((audio.get("extra") or {})[ORIGINAL_ENCODE_TAG]).split(":")
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
    # video.get_bitrate reads its cache first.
    compared["BitRate_resolved"] = int(bitrate)
    compared["BitRate_source"] = ORIGINAL_ENCODE_TAG
    return compared


def generate_audio_ffmpeg_conversion_cmd(base_cmd, audio, delay_to_put, other_filter_param="",
                                         other_param="", force_conversion=False,
                                         filters_inline=True, stream=None, path=None):
    """Append to base_cmd the options that copy or convert one audio track.

    Lossless -> the lossless codec; negative delay, force_conversion or a
    filter -> the target of the source codec; otherwise nothing (copy). Every
    conversion gets the ORIGINAL_ENCODE tag. other_param (str or list) is
    appended as given. stream is the output stream specifier,
    ":a:<@typeorder - 1>" by default. path is the file holding the track: a
    bitrate the metadata lacks is measured there, so the target and
    ORIGINAL_ENCODE always get one.

    filters_inline=True appends the filter chain as one -filter<stream> and
    returns True when the track is converted. filters_inline=False (for a
    filtergraph output, which refuses -filter) leaves it out and returns
    (args, filter_string): the options appended to base_cmd, and the chain for
    the caller to put at the end of its filtergraph (None when empty).
    """
    if stream is None:
        stream = f":a:{int(audio.get('@typeorder', 1)) - 1}"
    lossless = audio_rules().get("lossless")
    if lossless and is_lossless(audio):
        sample_fmt = "s16" if audio.get("BitDepth") == "16" else "s32"
        options = ["-c", lossless["codec"], "-sample_fmt", sample_fmt] + list(lossless.get("args", []))
        rule_filter = None
    elif delay_to_put < 0 or force_conversion or other_filter_param:
        codec, bitrate, args, rule_filter = pick_target(audio, path)
        options = ["-c", codec]
        if bitrate:
            options.extend(["-b", str(bitrate)])
        options.extend(args)
    elif filters_inline:
        return False
    else:
        return [], None
    args = []
    for token in options:
        if str(token).startswith("-"):
            args.append(f"{token}{stream}")
        else:
            args.append(str(token))
    filters = [rule_filter, NEGATIVE_DELAY_FILTER if delay_to_put < 0 else None, other_filter_param]
    joined = ",".join(f for f in filters if f) or None
    if joined and filters_inline:
        args.extend([f"-filter{stream}", joined])
    tag = original_encode_tag(audio, path)
    if tag:
        args.extend([f"-metadata:s{stream}", f"{ORIGINAL_ENCODE_TAG}={tag}"])
    if isinstance(other_param, str):
        args.extend(other_param.split())
    elif other_param:
        args.extend(other_param)
    base_cmd.extend(args)
    if filters_inline:
        return True
    return args, joined
