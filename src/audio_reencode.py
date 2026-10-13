'''
Choose how a rebuilt audio track is encoded, from the `audio_reencode` table of
config.json.

Workflow for one track:

1. lossless source (Format in `lossless_formats`, or MediaInfo
   `Compression_Mode == "Lossless"`) -> the `lossless` target (FLAC);
2. otherwise the target named by the `sources` entry of the source codec
   (ffprobe `codec_name`), encoded by the first of its `rules` that matches;
3. codec absent from `sources` -> the same codec at the source bitrate; only
   the encoder name, sample rate, filters and args of the target named after
   the codec apply.

Table layout (all keys optional; a missing table means rule 3 for every lossy
track):

    "audio_reencode": {
        "lossless_formats": ["flac", "pcm"],
        "lossless": "flac",
        "sources": {"<codec_name>": "<target>"},
        "targets": {"<target>": {
            "encoder": "libopus",
            "sample_rate": "48000",
            "sample_fmt": {"<BitDepth>": "s16", "default": "s32"},
            "max_bitrate_per_channel": "256k",
            "filters": {"<ffprobe channel_layout>": "<filter>"},
            "args": ["-option", "value"],
            "rules": [{
                "codecs": ["ac3"],
                "channels": 2 | {"min": 3, "max": 8},
                "source_bitrate": {"min": "0", "max": "128k"},
                "bitrate": "256k" | "source"
                           | {"factor": 1.0, "floor": "192k", "ceiling": "512k"},
                "args": ["-mapping_family", "1"],
                "filters": {"5.1(side)": "channelmap=channel_layout=5.1"}}]}}
    }

Rule conditions (each optional, absent = any): `codecs`, the source codecs the
rule is restricted to; `channels`, a count or an inclusive range;
`source_bitrate`, `min` inclusive and `max` exclusive, either open (null).
The first rule whose conditions all hold wins. Its `bitrate` is fixed, or
clamp(source x factor, floor, ceiling); "source" is factor 1 without bounds.
Rule `args` follow the target's; a rule `filters` entry replaces the target's
for that layout. `max_bitrate_per_channel` caps every result (encoder limit).

Coverage: the rules of a target without a `codecs` condition must cover every
channel count from 1 to 8 and every source bitrate from 0 to infinity; the
loader raises `ConfigError` otherwise. A track no rule covers (more than 8
channels, unknown count, target without rules) gets `DEFAULT_RULE`.

Unknown source bitrate: the first rule matching codec and channels with no
upper bitrate bound is used at its floor (its fixed bitrate), and a line is
written to stderr.
'''

import json
import math
import sys

CONFIG_PATH = "config.json"
CONFIG_KEY = "audio_reencode"
COVERED_CHANNELS = range(1, 9)
DEFAULT_RULE = {"bitrate": "source"}
RULE_KEYS = {"codecs", "channels", "source_bitrate", "bitrate", "args", "filters"}

_cache = {}
_UNSET = object()


class ConfigError(ValueError):
    '''The `audio_reencode` table of config.json is malformed or leaves a case uncovered.'''


def rules(config_path=CONFIG_PATH):
    '''Return the validated `audio_reencode` table of `config_path`, read once and cached.

    A missing file or key gives an empty table; a table that fails `validate`
    raises `ConfigError`.
    '''
    if config_path not in _cache:
        try:
            with open(config_path) as handle:
                table = json.load(handle).get(CONFIG_KEY) or {}
        except (OSError, ValueError):
            table = {}
        validate(table)
        _cache[config_path] = table
    return _cache[config_path]


def parse_bitrate(value):
    '''Return `value` ("256k", "1.5M", "256000", 256000) in bit/s as an int, or None.'''
    if value is None:
        return None
    text = str(value).strip().lower()
    scale = 1
    if text.endswith("k"):
        scale, text = 1000, text[:-1]
    elif text.endswith("m"):
        scale, text = 1000000, text[:-1]
    try:
        number = int(float(text) * scale)
    except ValueError:
        return None
    return number if number > 0 else None


def _bound(value, where):
    '''A bitrate bound in bit/s: None stays None (open), 0 is allowed; else ConfigError.'''
    if value is None:
        return None
    if str(value).strip() in ("0", "0k"):
        return 0
    number = parse_bitrate(value)
    if number is None:
        raise ConfigError(f"audio_reencode: {where}: bad bitrate {value!r}")
    return number


def _channel_range(rule, where="rule"):
    '''(low, high) inclusive channel bounds of a rule; (1, inf) when absent.'''
    spec = rule.get("channels")
    if spec is None:
        return 1, math.inf
    if isinstance(spec, dict):
        low, high = spec.get("min", 1), spec.get("max")
    else:
        low = high = spec
    try:
        low = int(low)
        high = math.inf if high is None else int(high)
    except (TypeError, ValueError):
        raise ConfigError(f"audio_reencode: {where}: bad channels {spec!r}")
    if low > high:
        raise ConfigError(f"audio_reencode: {where}: channels min above max {spec!r}")
    return low, high


def _bitrate_range(rule, where="rule"):
    '''(low, high) source bitrate bounds of a rule, low inclusive, high exclusive.'''
    spec = rule.get("source_bitrate") or {}
    if not isinstance(spec, dict):
        raise ConfigError(f"audio_reencode: {where}: source_bitrate must be {{min, max}}")
    low = _bound(spec.get("min"), where) or 0
    high = _bound(spec.get("max"), where)
    high = math.inf if high is None else high
    if low >= high:
        raise ConfigError(f"audio_reencode: {where}: source_bitrate min not below max {spec!r}")
    return low, high


def _target_spec(rule, where="rule"):
    '''("fixed", bit/s) or ("factor", k, floor, ceiling) for a rule's `bitrate`.'''
    spec = rule.get("bitrate", "source")
    if isinstance(spec, dict):
        try:
            factor = float(spec.get("factor", 1))
        except (TypeError, ValueError):
            raise ConfigError(f"audio_reencode: {where}: bad factor {spec!r}")
        floor = _bound(spec.get("floor"), where)
        ceiling = _bound(spec.get("ceiling"), where)
        if factor <= 0 or (floor is not None and ceiling is not None and floor > ceiling):
            raise ConfigError(f"audio_reencode: {where}: bad factor target {spec!r}")
        return ("factor", factor, floor, ceiling)
    if str(spec) == "source":
        return ("factor", 1.0, None, None)
    fixed = parse_bitrate(spec)
    if fixed is None:
        raise ConfigError(f"audio_reencode: {where}: bad bitrate {spec!r}")
    return ("fixed", fixed)


def coverage_gaps(rule_list):
    '''List the uncovered cases of `rule_list` as (channels, low, high) bit/s triples.

    Only rules without a `codecs` condition count, since they alone hold for
    every source codec.
    '''
    gaps = []
    general = [rule for rule in rule_list if not rule.get("codecs")]
    for channels in COVERED_CHANNELS:
        spans = sorted(_bitrate_range(rule) for rule in general
                       if _channel_range(rule)[0] <= channels <= _channel_range(rule)[1])
        reach = 0
        for low, high in spans:
            if low > reach:
                gaps.append((channels, reach, low))
            reach = max(reach, high)
        if reach != math.inf:
            gaps.append((channels, reach, math.inf))
    return gaps


def _show(bitrate):
    return "infinity" if bitrate == math.inf else f"{bitrate} bit/s"


def validate(table):
    '''Raise `ConfigError` when a target's `rules` are malformed or leave a case uncovered.'''
    targets = table.get("targets", {})
    for codec, entry in table.get("sources", {}).items():
        name = entry.get("to") if isinstance(entry, dict) else entry
        if not isinstance(name, str) or not name:
            raise ConfigError(f"audio_reencode: source {codec!r} names no target")
    for name, target in targets.items():
        if "rules" not in target:
            continue
        rule_list = target["rules"]
        if not isinstance(rule_list, list) or not rule_list:
            raise ConfigError(f"audio_reencode: target {name!r}: rules must be a non-empty list")
        for index, rule in enumerate(rule_list):
            where = f"target {name!r} rule {index}"
            if not isinstance(rule, dict):
                raise ConfigError(f"audio_reencode: {where}: not an object")
            unknown = set(rule) - RULE_KEYS
            if unknown:
                raise ConfigError(f"audio_reencode: {where}: unknown keys {sorted(unknown)}")
            _channel_range(rule, where)
            _bitrate_range(rule, where)
            _target_spec(rule, where)
        gaps = coverage_gaps(rule_list)
        if gaps:
            listed = "; ".join(f"{channels} ch from {_show(low)} to {_show(high)}"
                               for channels, low, high in gaps)
            raise ConfigError(f"audio_reencode: target {name!r}: rules leave cases "
                              f"uncovered: {listed}")


def is_lossless(audio, table=None):
    '''Return True when the track is lossless.

    Format in `lossless_formats` (default flac and PCM), or MediaInfo
    `Compression_Mode` equal to "Lossless".
    '''
    table = rules() if table is None else table
    formats = [name.lower() for name in table.get("lossless_formats", ["flac", "pcm"])]
    if str(audio.get("Format", "")).lower() in formats:
        return True
    return audio.get("Compression_Mode") == "Lossless"


def source_codec(audio):
    '''Return the track's ffprobe codec name, lower case ("" when unknown).'''
    return str((audio.get("ffprobe") or {}).get("codec_name") or "").lower()


def channel_count(audio):
    '''Return the track's channel count (ffprobe first, then MediaInfo), or None.'''
    for value in ((audio.get("ffprobe") or {}).get("channels"), audio.get("Channels")):
        try:
            return int(str(value).split("/")[0].strip())
        except (TypeError, ValueError):
            continue
    return None


def channel_layout(audio):
    '''Return the track's ffprobe channel layout ("" when unknown).'''
    return str((audio.get("ffprobe") or {}).get("channel_layout") or "")


def _candidates(target, codec, channels):
    '''Rules of `target` whose codec and channel conditions hold, then `DEFAULT_RULE`.'''
    found = []
    for rule in target.get("rules", []):
        if rule.get("codecs") and codec not in [str(c).lower() for c in rule["codecs"]]:
            continue
        low, high = _channel_range(rule)
        if channels is None:
            if rule.get("channels") is not None:
                continue
        elif not low <= channels <= high:
            continue
        found.append(rule)
    return found + [DEFAULT_RULE]


def match_rule(target, codec, channels, source_bitrate):
    '''Return the rule of `target` that encodes this track.

    The first candidate whose bitrate bounds hold; with an unknown
    `source_bitrate` (None), the first candidate without an upper bound.
    '''
    for rule in _candidates(target, codec, channels):
        low, high = _bitrate_range(rule)
        if source_bitrate is None:
            if high == math.inf:
                return rule
        elif low <= source_bitrate < high:
            return rule
    return DEFAULT_RULE


def rule_bitrate(rule, source_bitrate):
    '''Bitrate in bit/s a rule gives for `source_bitrate` (None when unknown).

    Fixed: that value. Factor: clamp(source x factor, floor, ceiling); with an
    unknown source, the floor (None when the rule has none).
    '''
    spec = _target_spec(rule)
    if spec[0] == "fixed":
        return spec[1]
    _, factor, floor, ceiling = spec
    if source_bitrate is None:
        return floor
    bitrate = int(round(source_bitrate * factor))
    if floor is not None:
        bitrate = max(bitrate, floor)
    if ceiling is not None:
        bitrate = min(bitrate, ceiling)
    return bitrate


def _depends_on_source(target, codec, channels):
    '''Whether the source bitrate can change the rule or bitrate chosen for this track.'''
    for rule in _candidates(target, codec, channels):
        if rule.get("source_bitrate"):
            return True
        return _target_spec(rule)[0] == "factor"
    return False


def _target_name(entry):
    return entry.get("to") if isinstance(entry, dict) else entry


def choose(audio, source_bitrate=_UNSET, table=None):
    '''Return the encoding decision for one track as a dict.

    Keys: "rule" ("lossless", "table" or "same_codec"), "target" (table target
    name or the source codec), "encoder" (None when the table names no encoder
    for a same-codec track), "bitrate" (bit/s or None), "uses_source_bitrate"
    (whether `source_bitrate` can change the bitrate), "sample_rate",
    "sample_fmt", "filter" (or None) and "args" (a list).

    Leaving `source_bitrate` out is a probe: an unknown bitrate is not logged.
    '''
    probe = source_bitrate is _UNSET
    source_bitrate = parse_bitrate(None if probe else source_bitrate)
    table = rules() if table is None else table
    targets = table.get("targets", {})
    codec = source_codec(audio)
    channels = channel_count(audio)
    layout = channel_layout(audio)
    matched = {}
    uses_source = False
    if is_lossless(audio, table):
        name = table.get("lossless", "flac")
        target = targets.get(name, {})
        rule = "lossless"
        bitrate = None
    elif codec in table.get("sources", {}):
        name = _target_name(table["sources"][codec])
        target = targets.get(name, {})
        rule = "table"
        matched = match_rule(target, codec, channels, source_bitrate)
        bitrate = rule_bitrate(matched, source_bitrate)
        uses_source = _depends_on_source(target, codec, channels)
        per_channel_max = parse_bitrate(target.get("max_bitrate_per_channel"))
        if per_channel_max is not None and channels and bitrate is not None:
            bitrate = min(bitrate, per_channel_max * channels)
        if source_bitrate is None and uses_source and not probe:
            sys.stderr.write(f"audio_reencode: source bitrate of a {codec} {channels} ch "
                             f"track unknown: {name} at the rule floor {bitrate}\n")
    else:
        name = codec
        target = targets.get(codec, {})
        rule = "same_codec"
        bitrate = source_bitrate
        uses_source = True
    encoder = target.get("encoder")
    if encoder is None and rule == "lossless":
        encoder = name
    sample_fmt = None
    if target.get("sample_fmt"):
        sample_fmt = target["sample_fmt"].get(str(audio.get("BitDepth")),
                                              target["sample_fmt"].get("default"))
    filters = dict(target.get("filters") or {})
    filters.update(matched.get("filters") or {})
    return {"rule": rule, "target": name, "encoder": encoder, "bitrate": bitrate,
            "uses_source_bitrate": uses_source,
            "sample_rate": target.get("sample_rate"), "sample_fmt": sample_fmt,
            "filter": filters.get(layout),
            "args": list(target.get("args", [])) + list(matched.get("args", []))}


def join_filters(*filters):
    '''Join filter strings into one chain, skipping empty ones; None when all are empty.

    ffmpeg keeps only the last `-filter:a:N` given for a stream, so every
    filter of one stream goes into a single string.
    '''
    kept = [str(f) for f in filters if f]
    return ",".join(kept) if kept else None


def stream_options(options, stream):
    '''Suffix each "-option" of a flat option list with the stream specifier `stream`.'''
    return [f"{token}{stream}" if str(token).startswith("-") else str(token)
            for token in options]


def encoder_options(decision, stream=":a", filters=(), include_filter=True):
    '''Return the ffmpeg output options of one decision for stream specifier `stream`.

    `filters` are joined ahead of the table filter into one `-filter` option;
    `include_filter=False` leaves the filter out (for a filtergraph output,
    which cannot take `-filter`).
    '''
    options = ["-c", decision["encoder"]]
    if decision.get("bitrate") is not None:
        options.extend(["-b", str(decision["bitrate"])])
    if decision.get("sample_fmt"):
        options.extend(["-sample_fmt", decision["sample_fmt"]])
    if decision.get("sample_rate"):
        options.extend(["-ar", str(decision["sample_rate"])])
    options.extend(decision.get("args", []))
    joined = join_filters(*filters, decision.get("filter")) if include_filter else None
    if joined:
        options.extend(["-filter", joined])
    return stream_options(options, stream)


def second_pass_options(audio, relative_id, filters, bitrate_of):
    '''Options of one shifted lossy track in `mergeVideo`'s second pass.

    `relative_id` is the per-type suffix (":N"), `filters` the filters the
    caller already applies to that stream, `bitrate_of(audio)` the source
    bitrate (its errors leave the bitrate out). A codec the table does not
    name keeps its own ffmpeg name as encoder.
    '''
    try:
        source_bitrate = bitrate_of(audio)
    except Exception:
        source_bitrate = None
    decision = choose(audio, source_bitrate)
    if decision["encoder"] is None:
        decision["encoder"] = source_codec(audio) or audio["ffprobe"]["codec_name"]
    return encoder_options(decision, f":a{relative_id}", filters)
