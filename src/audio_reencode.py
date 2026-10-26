'''ffmpeg options that copy or convert one audio track, from the config.json audio_reencode table (decision: audio-reencode).'''

import sys

import tools
import video

DEFAULT_FLAC_ARGS = ["-compression_level", "12", "-exact_rice_parameters", "1"]
NEGATIVE_DELAY_FILTER = "aresample=async=1:first_pts=0"
LOSSLESS_CODECS = {"truehd", "mlp", "flac", "alac"}

audio_encode_parameters = (tools.configuration or {}).get("audio_reencode") or {}

def get_audio_encode_parameters():
    global audio_encode_parameters
    if not audio_encode_parameters:
        audio_encode_parameters = (tools.configuration or {}).get("audio_reencode") or {}
    return audio_encode_parameters

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
    return get_audio_encode_parameters().get("encode_lossless_in_flac", True)

# This case is when we don't want reencode lossless in flac 
def get_parameters_in_reencode(audio, audio_position_in_file, file_path, filter_param_to_add=''):
    param_to_send = []
    original_encode_param = get_orginal_encode_data(audio)
    if original_encode_param:
        codec = original_encode_param[0]
        bitrate = original_encode_param[1]
        sample_rate = original_encode_param[2]
    else:
        codec = video.get_codec(audio)
        try:
            bitrate = video.get_bitrate(audio, file_path)
        except Exception:
            bitrate = 0
        sample_rate = video.get_sample_rate(audio)
    channels = video.get_channel(audio)

    param_reencode = reencode_rules_writed_in_table(codec,bitrate,sample_rate,channels)

    if param_reencode:
        """ Example:
        {"target": "libopus","bitrate": "source", "sample_rate": "48000", "filter": "my_special_arg,my_second_special_param", "args": {"-mapping_family": "1", "-compression_level": "12"}}
        """

        param_to_send.extend([f"-c:a:{audio_position_in_file}", param_reencode["target"]]) # The encoder must be set. Raise an error if it missing.

        param_to_send.extend([f"-metadata:s:a:{audio_position_in_file}", f"{tools.ORIGINAL_ENCODE_TAG}={codec}:{int(bitrate)}:{int(sample_rate)}"])
        if int(bitrate) == 0:
            tools.log_always(f"No bitrate found for audio track {audio.get('StreamOrder')} of {file_path}. {tools.ORIGINAL_ENCODE_TAG} will be at 0. The stream will certainely remove.\n")

        bitrate_param = str(param_reencode.get("bitrate", "source"))
        if bitrate_param != "source":
            bitrate_param = bitrate_param.strip().lower()
            bitrate = int(float(bitrate_param.rstrip("km")) * {"k": 1000, "m": 1000000}.get(bitrate_param[-1:], 1))
        if int(bitrate) > 0 and param_reencode["target"] != "flac":
            param_to_send.extend([f"-b:a:{audio_position_in_file}", str(bitrate)])

        sample_rate_param = str(param_reencode.get("sample_rate", "")).strip().lower()
        if sample_rate_param:
            param_to_send.extend([f"-ar:a:{audio_position_in_file}", str(int(float(sample_rate_param.rstrip("km")) * {"k": 1000, "m": 1000000}.get(sample_rate_param[-1:], 1)))])

        filter_param = param_reencode.get("filter", "")
        if filter_param and filter_param_to_add:
            filter_param = filter_param + "," + filter_param_to_add
        elif filter_param_to_add:
            filter_param = filter_param_to_add
        if filter_param:
            param_to_send.extend([f"-filter:a:{audio_position_in_file}", filter_param])

        args_param = param_reencode.get("args", [])
        if args_param:
            for option, value in args_param.items():    
                param_to_send.extend([f"{option}:a:{audio_position_in_file}", str(value)])
    else:
        param_to_send.extend([f"-c:a:{audio_position_in_file}", codec])
        if filter_param_to_add:
            param_to_send.extend([f"-filter:a:{audio_position_in_file}", filter_param_to_add])
        if codec == "flac": # 7. 2- search the encoder use in the table to compare.
            param_to_send.extend(get_parameters_lossless(audio, audio_position_in_file))
        elif int(bitrate) != 0:
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
                            "6": {"target": "libopus","bitrate": "640k", "args": {"-mapping_family": "1"}},
                            "7": {"target": "libopus","bitrate": "source", "args": {"-mapping_family": "1"}}
                        }
                    }
                }
            },
    """
    param_find = {}
    source_entry = get_audio_encode_parameters().get("sources", {}).get(encoder)
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