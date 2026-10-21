'''ffmpeg options that copy or convert one audio track, from the config.json audio_reencode table (decision: audio-reencode).'''

import sys

import tools
import video

DEFAULT_FLAC_ARGS = ["-compression_level", "12", "-exact_rice_parameters", "1"]
NEGATIVE_DELAY_FILTER = "aresample=async=1:first_pts=0"
LOSSLESS_CODECS = {"truehd", "mlp", "flac", "alac"}


def is_lossless(audio):
    if (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG):
        return False
    codec = video.get_codec(audio)
    if codec in LOSSLESS_CODECS or codec.startswith("pcm_"):
        return True
    if codec == "dts" and "MA" in str((audio.get("ffprobe") or {}).get("profile") or audio.get("Format_Profile") or "").upper():
        return True
    return audio.get("Compression_Mode") == "Lossless"


def audio_for_keep(audio):
    """Return the audio dict keep_best_audio compares: a re-encoded track as its original encoding."""
    if is_lossless(audio):
        return audio
    configuration = tools.configuration or {}
    tag = (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG)
    if not tag:
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


def generate_audio_ffmpeg_conversion_cmd(base_cmd, audio, delay_to_put, other_filter_param="",
                                         other_param="", force_conversion=False,
                                         filters_inline=True, stream=None, path=None):
    """Append to base_cmd the options that copy or convert one audio track; True (or (args, filter)) when converted."""
    stream = stream or f":a:{int(audio.get('@typeorder', 1)) - 1}"
    table = (tools.configuration or {}).get("audio_reencode") or {}
    tag = None
    if is_lossless(audio) and table.get("encode_lossless_in_flac", True):
        encoder, args, rule_filter = "flac", table.get("flac_args") or DEFAULT_FLAC_ARGS, None
    elif delay_to_put < 0 or force_conversion or other_filter_param:
        encoder = video.get_codec(audio)
        try:
            bitrate = video.get_bitrate(audio, path)
        except Exception:
            bitrate = None
        sample_rate = video.get_sample_rate(audio)
        channels = video.get_channel(audio)
        args, rule_filter = [], None
        source_entry = table.get("sources", {}).get(encoder)
        if source_entry:
            params = source_entry.get("bitrate", {})
            for measured, next_stage in ((bitrate, "sample_rate"), (sample_rate, "channels"), (channels, None)):
                entry = {}
                if params:
                    try:
                        measured_value = float(measured)
                    except (TypeError, ValueError):
                        measured_value = None
                    best_key = None
                    for key in params:
                        try:
                            threshold = float(key)
                        except (TypeError, ValueError):
                            continue
                        if measured_value is not None and threshold <= measured_value and (best_key is None or threshold > float(best_key)):
                            best_key = key
                    if best_key is None:
                        best_key = min(params, key=float)
                    entry = params[best_key] or {}
                params = entry.get(next_stage, {}) if next_stage else entry
            encoder = params.get("target", source_entry.get("target"))
            target = table.get("targets", {}).get(encoder, {})
            if params.get("bitrate", "source") != "source":
                text = str(params.get("bitrate")).strip().lower()
                try:
                    bitrate = int(float(text.rstrip("km")) * {"k": 1000, "m": 1000000}.get(text[-1:], 1))
                except ValueError:
                    bitrate = None
                if bitrate is not None and bitrate < 0:
                    bitrate = None
            args = list(params.get("args", []))
            filters = [target.get("filters", {}).get(str(channels)) if channels is not None else None]
            allowed = target.get("sample_rates")
            if allowed and sample_rate is not None and sample_rate not in allowed:
                higher = sorted(rate for rate in allowed if rate >= sample_rate)
                filters.append(f"aresample={higher[0] if higher else max(allowed)}")
            rule_filter = ",".join(f for f in filters if f) or None
        if encoder != "flac":
            tag = (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG)
            if not tag:
                try:
                    codec = video.get_codec(audio)
                    measured_rate = video.get_sample_rate(audio)
                    measured_bitrate = video.get_bitrate(audio, path)
                    if not codec or measured_rate is None:
                        raise ValueError(f"codec={codec!r} sample_rate={measured_rate!r}")
                    tag = f"{codec}:{int(measured_bitrate)}:{int(measured_rate)}"
                except Exception as e:
                    sys.stderr.write(f"No ORIGINAL_ENCODE for track {audio.get('StreamOrder')}: {e}\n")
    elif filters_inline:
        return False
    else:
        return [], None

    if encoder == "flac":
        options = ["-c", "flac", "-sample_fmt", "s16" if audio.get("BitDepth") == "16" else "s32"] + list(args)
    else:
        options = ["-c", encoder] + (["-b", str(bitrate)] if bitrate else []) + args
    args = [f"{token}{stream}" if str(token).startswith("-") else str(token) for token in options]
    filters = [rule_filter, NEGATIVE_DELAY_FILTER if delay_to_put < 0 else None, other_filter_param]
    joined = ",".join(f for f in filters if f) or None
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
