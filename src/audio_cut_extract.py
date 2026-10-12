'''Audio cuts for the delay comparison: one ffmpeg decode per track, one file per cut.

Each cut keeps the seek of the original per-cut command: `-copyts` with `-ss` and `-t`
as output options, so the cut starts at the exact sample of the container time and every
cut is sample-identical to a separate `ffmpeg ... -ss <begin> -t <length> cut.wav`.
The track is decoded (and converted) once and split to one output per cut, instead of
one decode from the start of the track for every cut.
'''
import struct
from os import path


MIN_LENGTH_RATIO = 0.9


class ExtractionTooShort(Exception):
    '''An extracted cut is missing or holds less than MIN_LENGTH_RATIO of its expected audio.

    Built by too_short(); it carries only its message so it survives the pool's pickling.
    '''


def too_short(out_path, file_path, stream_order, cut_index, cut, got_seconds, expected_seconds):
    '''Return the ExtractionTooShort naming the output, the source track and the cut.'''
    where = "full track" if cut is None else f"cut {cut_index} (begin {cut[0]}, length {cut[1]})"
    return ExtractionTooShort(f"Audio extraction of {file_path} stream {stream_order} {where} gave "
                              f"{got_seconds:.3f} s in {out_path}, expected at least "
                              f"{MIN_LENGTH_RATIO * expected_seconds:.3f} s of {expected_seconds:.3f} s")


def parse_time_us(text):
    '''Parse an ffmpeg duration ("[-][HH:]MM:SS[.f]" or "[-]S[.f]") into integer microseconds.

    Digits after the sixth decimal are dropped, as av_parse_time does.

    Raises ValueError on any other form.
    '''
    text = str(text).strip()
    sign = 1
    if text.startswith("-"):
        sign = -1
        text = text[1:]
    parts = text.split(":")
    if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
        raise ValueError(f"Not an ffmpeg time: {text!r}")
    whole = 0
    for p in parts[:-1]:
        if not p.isdigit():
            raise ValueError(f"Not an ffmpeg time: {text!r}")
        whole = whole * 60 + int(p)
    sec = parts[-1]
    if "." in sec:
        sec_int, frac = sec.split(".", 1)
    else:
        sec_int, frac = sec, ""
    if not (sec_int.isdigit() or sec_int == "") or not (frac.isdigit() or frac == "") or (sec_int == "" and frac == ""):
        raise ValueError(f"Not an ffmpeg time: {text!r}")
    frac_us = int((frac + "000000")[:6])  # ffmpeg keeps six digits and drops the rest
    return sign * ((whole * 60 + int(sec_int or "0")) * 1000000 + frac_us)


def time_us_to_sample(time_us, sample_rate):
    '''Return the sample index of a time, rounded to nearest as ffmpeg's trim does.'''
    num = time_us * sample_rate
    return (num + 500000) // 1000000 if num >= 0 else -((-num + 500000) // 1000000)


def cut_sample_offset(begin_text, span_begin_text, sample_rate):
    '''Sample offset of a cut inside a decode that starts at span_begin_text.

    Both begins round to their own sample, so the offset is the difference of the two
    rounded samples, not the rounded difference of the times.
    '''
    return (time_us_to_sample(parse_time_us(begin_text), sample_rate)
            - time_us_to_sample(parse_time_us(span_begin_text), sample_rate))


PCM_SAMPLE_FORMATS = {"pcm_s16le": "s16", "pcm_s32le": "s32", "pcm_f32le": "flt"}
CHANNEL_LAYOUTS = {"1": "mono", "2": "stereo"}


def shared_conversion_filter(codec_param):
    '''Return the aformat filter ffmpeg itself inserts before a PCM output, or None.

    Converting once before an asplit gives each cut the very samples a separate output
    converts on its own. Only PCM codecs and mono/stereo layouts are mapped; anything
    else returns None and every output keeps its own conversion.
    '''
    codec = rate = channels = None
    for i in range(0, len(codec_param) - 1):
        opt, val = codec_param[i], str(codec_param[i + 1])
        if opt in ("-c:a", "-acodec"):
            codec = val
        elif opt == "-ar":
            rate = val
        elif opt == "-ac":
            channels = val
    if codec not in PCM_SAMPLE_FORMATS or (channels is not None and channels not in CHANNEL_LAYOUTS):
        return None
    parts = ["sample_fmts=" + PCM_SAMPLE_FORMATS[codec]]
    if rate is not None:
        if not rate.isdigit():
            return None
        parts.append("sample_rates=" + rate)
    if channels is not None:
        parts.append("channel_layouts=" + CHANNEL_LAYOUTS[channels])
    return "aformat=" + ":".join(parts)


def build_cuts_command(ffmpeg, file_path, stream_order, codec_param, cuts, out_paths, threads=3):
    '''Return one ffmpeg command that decodes the stream once and writes every cut.

    With a PCM output the stream is decoded and converted once, then split to one output
    per cut; otherwise each output maps the stream and converts it itself. Each output
    keeps its own output-side -ss/-t, so each cut is the one a separate command writes.

    Args:
        ffmpeg: ffmpeg executable.
        file_path: Source media file.
        stream_order: Absolute stream index of the audio track.
        codec_param: Output codec options, repeated for every output.
        cuts: List of [begin, length] ffmpeg times, or None for the whole track.
        out_paths: One output path per cut (one path when cuts is None).
        threads: ffmpeg -threads value.
    '''
    if cuts is None:
        if len(out_paths) != 1:
            raise ValueError("A full-track extraction writes exactly one file")
    elif len(cuts) != len(out_paths) or not cuts:
        raise ValueError("One output path is needed per cut")
    cmd = [ffmpeg, "-y", "-analyzeduration", "1000M", "-probesize", "1000M", "-threads", str(threads),
           "-nostdin", "-i", file_path, "-copyts"]
    conversion = shared_conversion_filter(codec_param) if len(out_paths) > 1 else None
    if conversion is not None:
        labels = "".join("[cut%d]" % i for i in range(len(out_paths)))
        cmd.extend(["-filter_complex", "[0:%s]%s,asplit=%d%s" % (stream_order, conversion, len(out_paths), labels)])
    for i, out_path in enumerate(out_paths):
        if conversion is None:
            cmd.extend(["-vn", "-dn", "-sn"])
        cmd.extend(codec_param)
        cmd.extend(["-map", "[cut%d]" % i if conversion is not None else "0:" + str(stream_order)])
        if cuts is not None:
            cmd.extend(["-ss", str(cuts[i][0]), "-t", str(cuts[i][1])])
        cmd.append(out_path)
    return cmd


def wav_duration_seconds(wav_path):
    '''Return the audio duration of a RIFF/WAVE file from its header, or 0.0 if it holds none.

    A data size left unset by a non-seekable write (0 or 0xFFFFFFFF) falls back to the
    bytes present after the data header.
    '''
    if not path.isfile(wav_path):
        return 0.0
    size = path.getsize(wav_path)
    with open(wav_path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[0:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return 0.0
        block_align = rate = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                return 0.0
            cid, csize = chunk[0:4], struct.unpack("<I", chunk[4:8])[0]
            if cid == b"fmt ":
                fmt = f.read(csize)
                if len(fmt) < 16:
                    return 0.0
                rate = struct.unpack("<I", fmt[4:8])[0]
                block_align = struct.unpack("<H", fmt[12:14])[0]
                f.seek(csize % 2, 1)
            elif cid == b"data":
                if not rate or not block_align:
                    return 0.0
                remaining = size - f.tell()
                if csize in (0, 0xFFFFFFFF) or csize > remaining:
                    csize = remaining
                return (csize // block_align) / rate
            else:
                f.seek(csize + csize % 2, 1)


def check_cut_length(out_path, file_path, stream_order, cut_index, cut, track_seconds=None):
    '''Raise ExtractionTooShort when a cut holds less than MIN_LENGTH_RATIO of its expected audio.

    The expected audio is the requested length, shortened when the track ends before the
    cut does (track_seconds, if known). A full-track extraction (cut None) must not be empty.
    Only WAV outputs are measured; other formats only have to exist.
    '''
    if not out_path.lower().endswith(".wav"):
        if not path.isfile(out_path) or path.getsize(out_path) == 0:
            raise too_short(out_path, file_path, stream_order, cut_index, cut, 0.0, 0.0)
        return
    got = wav_duration_seconds(out_path)
    if cut is None:
        if got <= 0.0:
            raise too_short(out_path, file_path, stream_order, cut_index, cut, got, 0.0)
        return
    expected = parse_time_us(cut[1]) / 1e6
    if track_seconds is not None:
        expected = min(expected, max(float(track_seconds) - parse_time_us(cut[0]) / 1e6, 0.0))
    if got <= 0.0 or got < MIN_LENGTH_RATIO * expected:
        raise too_short(out_path, file_path, stream_order, cut_index, cut, got, expected)


def extract_track_cuts(cmd, file_path, stream_order, cuts, out_paths, track_seconds=None, timeout=2400):
    '''Run one built command (pool job) and check every cut it wrote.'''
    import tools
    tools.launch_cmdExt_with_timeout_reload(cmd, 3, timeout)
    if cuts is None:
        check_cut_length(out_paths[0], file_path, stream_order, 0, None, track_seconds)
    else:
        for i, out_path in enumerate(out_paths):
            check_cut_length(out_path, file_path, stream_order, i, cuts[i], track_seconds)
