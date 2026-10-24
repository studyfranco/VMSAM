'''ffmpeg options that copy or convert one audio track, from the config.json audio_reencode table (decision: audio-reencode).'''

import sys

import tools
import video

DEFAULT_FLAC_ARGS = ["-compression_level", "12", "-exact_rice_parameters", "1"]
NEGATIVE_DELAY_FILTER = "aresample=async=1:first_pts=0"
LOSSLESS_CODECS = {"truehd", "mlp", "flac", "alac"}

audio_encode_parameters = (tools.configuration or {}).get("audio_reencode") or {}

def is_lossless(audio):
    if (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG):
        return False
    codec = video.get_codec(audio)
    if codec in LOSSLESS_CODECS or codec.startswith("pcm_"):
        return True
    if codec == "dts" and "MA" in str((audio.get("ffprobe") or {}).get("profile") or audio.get("Format_Profile") or "").upper():
        return True
    return audio.get("Compression_Mode") == "Lossless"

def get_orginal_encode_data(audio):
    original_encode_param = []
    try:
        original_encode = (audio.get("extra") or {}).get(tools.ORIGINAL_ENCODE_TAG, '')
        if original_encode:
            codec, bitrate, sampling_rate = str(original_encode).split(":")
            original_encode_param = [codec, int(float(bitrate)), int(float(sampling_rate))]
    except Exception as e:
        tools.log_always(f"Error during get_orginal_encode_data: {e}")
    return original_encode_param
    

def audio_for_keep(audio):
    """Return the audio dict keep_best_audio compares: a re-encoded track as its original encoding."""
    configuration = tools.configuration or {}
    original_encode_param = get_orginal_encode_data(audio)
    if not original_encode_param:
        return audio
    try:
        handicap = float(configuration.get("reencode_keep_handicap", 1.0))
        bitrate = str(int(round(int(float(original_encode_param[1]) * handicap))))
        sampling_rate = str(int(float(original_encode_param[2])))
    except (KeyError, ValueError, TypeError):
        return audio
    codec = original_encode_param[0].strip().lower()
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

def get_parameters_lossless(audio, audio_position):
    parameters = [f"-c:a:{audio_position}", "flac", f"-compression_level:a:{audio_position}", "12"]
    # impossible to use because: parameters.extend(table.get("flac_args") or DEFAULT_FLAC_ARGS) can't have a proper way to find audio. We have to create a way to specify the stream.
    if "BitDepth" in audio:
        if audio["BitDepth"] == "16":
            parameters.extend([f"-sample_fmt:a:{audio_position}", "s16"])
        else:
            parameters.extend([f"-sample_fmt:a:{audio_position}", "s32"])
    else:
        parameters.extend([f"-sample_fmt:a:{audio_position}", "s32"])
    parameters.extend([f"-exact_rice_parameters:a:{audio_position}", "1"])
    return parameters

def all_lossless_in_flac():
    return audio_encode_parameters.get("encode_lossless_in_flac", True)

# This case is when we don't want reencode lossless in flac 
def get_parameters_in_reencode(audio, audio_position_in_file, file_path, filter_param_to_add=''):
    param_to_send = []
    original_encode_param = get_orginal_encode_data(audio)
    if original_encode_param:
        encoder = original_encode_param[0]
        bitrate = original_encode_param[1]
        sample_rate = original_encode_param[2]
    else:
        encoder = video.get_codec(audio)
        try:
            bitrate = video.get_bitrate(audio, file_path)
        except Exception:
            bitrate = 0
        sample_rate = video.get_sample_rate(audio)
    channels = video.get_channel(audio)

    param_reencode = reencode_rules_writed_in_table(encoder,bitrate,sample_rate,channels)

    if param_reencode:
        """ Example:
        {"target": "libopus","bitrate": "source", "filter": "my_special_arg", "args": {"-mapping_family": "1", "-compression_level": "12"}}
        """
        target_map = audio_encode_parameters.get("targets", {}).get(param_reencode["target"], {}) # The encoder must be set. Raise an error if it missing.
        if param_reencode["target"] == "flac":
            param_to_send.extend(get_parameters_lossless(audio, audio_position_in_file))
        else:
            param_to_send.extend([f"-c:a:{audio_position_in_file}", param_reencode["target"]])

            if bitrate != 0:
                param_to_send.extend([f"-metadata:s:a:{audio_position_in_file}", f"{tools.ORIGINAL_ENCODE_TAG}={encoder}:{int(bitrate)}:{int(sample_rate)}"])
            else:
                tools.log_always(f"No {tools.ORIGINAL_ENCODE_TAG} for audio track {audio.get('StreamOrder')} of {file_path}: no bitrate\n")

            bitrate_param = str(param_reencode.get("bitrate", "source"))
            if bitrate_param != "source":
                bitrate_param = bitrate_param.strip().lower()
                bitrate = int(float(bitrate_param.rstrip("km")) * {"k": 1000, "m": 1000000}.get(bitrate_param[-1:], 1))
            if bitrate > 0:
                param_to_send.extend([f"-b:a:{audio_position_in_file}", str(bitrate)])

        sample_rates_allowed = target_map.get("sample_rates", [])
        input_sample_rate = video.get_sample_rate(audio)
        if sample_rates_allowed and input_sample_rate and input_sample_rate not in sample_rates_allowed:
            higher_sample_rates = sorted([rate for rate in sample_rates_allowed if rate >= input_sample_rate])
            if higher_sample_rates:
                param_to_send.extend([f"-ar:a:{audio_position_in_file}", str(higher_sample_rates[0])])
            else:
                param_to_send.extend([f"-ar:a:{audio_position_in_file}", str(max(sample_rates_allowed))])

        args_param = param_reencode.get("args", [])
        if isinstance(args_param, dict):
            for option, value in args_param.items():
                param_to_send.extend([f"{option}:a:{audio_position_in_file}", str(value)])
        else:
            for arg in args_param:
                if str(arg).startswith("-"):
                    param_to_send.append(f"{arg}:a:{audio_position_in_file}")
                else:
                    param_to_send.append(str(arg))

        list_filter = [target_map.get("filters", {}).get(str(channels), ""), param_reencode.get("filter", ""), filter_param_to_add]
        filter_param = ",".join([filter_one for filter_one in list_filter if filter_one])
        if filter_param:
            param_to_send.extend([f"-filter:a:{audio_position_in_file}", filter_param])
    else:
        param_to_send.extend([f"-c:a:{audio_position_in_file}", audio["ffprobe"]["codec_name"]])
        if filter_param_to_add:
            param_to_send.extend([f"-filter:a:{audio_position_in_file}", filter_param_to_add])
        if bitrate != 0:
            param_to_send.extend([f"-b:a:{audio_position_in_file}", str(bitrate)])

    return param_to_send

def reencode_rules_writed_in_table(encoder,bitrate,sample_rate,channels):
    """ Example:
    "aac": {
        "bitrate": {
            "0": {
                "sample_rate": {
                    "0": {
                        "channels": {
                            "0": {"target": "libopus","bitrate": "source"},
                            "2": {"target": "libopus","bitrate": "160k"},
                            "3": {"target": "libopus","bitrate": "source", "filter": "my_special_arg", "args": {"-mapping_family": "1", "-compression_level": "12"}},
                            "6": {"target": "libopus","bitrate": "640k", "args": ["-mapping_family", "1"]},
                            "7": {"target": "libopus","bitrate": "source", "args": ["-mapping_family", "1"]}
                        }
                    }
                }
            },
    """
    param_find = {}
    source_entry = audio_encode_parameters.get("sources", {}).get(encoder)
    if source_entry:
        bitrate_map = source_entry.get("bitrate", {})
        if bitrate_map:
            sample_rate_map = select_lowest_value(bitrate_map, bitrate).get("sample_rate", {})
            if sample_rate_map:
                channels_map = select_lowest_value(sample_rate_map, sample_rate).get("channels", {})
                if channels_map:
                    param_find = select_lowest_value(channels_map, channels)
    return param_find

def select_lowest_value(mapping, target):
    valid_keys = [int(k) for k in mapping.keys() if int(k) <= target]
    if valid_keys:
        return mapping[str(max(valid_keys))]
    else:
        return {}