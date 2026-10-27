'''Perceptual quality of an audio encode against its source, measured by ViSQOL and kept as a track tag (decision: audio-quality).'''

import json
import re
from os import path
from xml.etree import ElementTree

import tools

VISQOL_TAG = "VMSAM_VISQOL"
VISQOL_WINDOW_POSITIONS = [0.05, 0.10, 0.25, 0.50, 0.65, 0.85, 0.90, 0.95]
VISQOL_WINDOW_DURATION = 30
VISQOL_EDGE_GUARD = 1
VISQOL_MIN_WINDOWS = 4

class AudioQualityError(Exception):
    pass

def get_visqol_mos(reference_file, degraded_file, delay_ms, tmp_file_base):
    """Return the mean ViSQOL audio-mode MOS-LQO of degraded_file against reference_file over 8 windows of the audio both play, degraded_file being reference_file shifted by delay_ms."""
    """ Example:
    reference_file: start 0.000 s, end 1465.789 s, 32000 Hz, 2 channels; degraded_file: start 0.000 s, end 1465.669 s; delay_ms = -120
    source content at t is at t - 0.120 s in degraded_file, so the windows lie in [0.120 + 1, 1465.669 + 0.120 - 1 - 30] of the source
    window 5 %: source 73.289 s to 103.289 s against degraded 73.169 s to 103.169 s, the degraded WAV resampled to 32000 Hz
    each file is sought 1 s before its window and the last second is decoded and dropped: a seek on the input alone lands up to 350 ms off on some AC-3 tracks
    result: mean of the 8 window MOS-LQO
    """
    list_stream = []
    for audio_file in [reference_file, degraded_file]:
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["ffprobe"], "-v", "error", "-select_streams", "a:0", "-show_entries",
                                                                              "stream=start_time,sample_rate,channels:format=start_time,duration", "-of", "json", audio_file], 2, 60)
        probe = json.loads(stdout.decode("utf-8", errors="ignore"))
        list_stream.append([float(probe["streams"][0]["start_time"]), float(probe["format"]["start_time"]) + float(probe["format"]["duration"]),
                            probe["streams"][0]["sample_rate"], int(probe["streams"][0]["channels"])])
    delay = float(delay_ms) / 1000
    window_first = max(list_stream[0][0], list_stream[1][0] - delay) + VISQOL_EDGE_GUARD
    window_last = min(list_stream[0][1], list_stream[1][1] - delay) - VISQOL_EDGE_GUARD - VISQOL_WINDOW_DURATION

    list_window_start = []
    for position in VISQOL_WINDOW_POSITIONS:
        window_start = list_stream[0][0] + position * (list_stream[0][1] - list_stream[0][0])
        window_start_valid = min(max(window_start, window_first), window_last)
        if window_last < window_first or any(abs(window_start_valid - window_kept) < VISQOL_WINDOW_DURATION for window_kept in list_window_start):
            tools.dev_log(f"[ViSQOL] window at {position:.0%} of {degraded_file} skipped: no free {VISQOL_WINDOW_DURATION} s left in the audio both files play\n")
        else:
            if window_start_valid != window_start:
                tools.dev_log(f"[ViSQOL] window at {position:.0%} of {degraded_file} moved from {window_start:.3f} s to {window_start_valid:.3f} s of the source\n")
            list_window_start.append(window_start_valid)
    if len(list_window_start) < VISQOL_MIN_WINDOWS:
        raise AudioQualityError(f"only {len(list_window_start)} ViSQOL windows fit in the audio both {reference_file} and {degraded_file} play, {VISQOL_MIN_WINDOWS} needed")

    list_file_wav = [f"{tmp_file_base}_reference.wav", f"{tmp_file_base}_degraded.wav"]
    list_mos = []
    try:
        for window_start in list_window_start:
            cmd_reference = [tools.software["ffmpeg"], "-y", "-nostdin", "-v", "error", "-ss", f"{window_start - list_stream[0][0] - VISQOL_EDGE_GUARD:.3f}", "-i", reference_file,
                             "-ss", str(VISQOL_EDGE_GUARD), "-t", str(VISQOL_WINDOW_DURATION)]
            cmd_degraded = [tools.software["ffmpeg"], "-y", "-nostdin", "-v", "error", "-ss", f"{window_start + delay - list_stream[1][0] - VISQOL_EDGE_GUARD:.3f}", "-i", degraded_file,
                            "-ss", str(VISQOL_EDGE_GUARD), "-t", str(VISQOL_WINDOW_DURATION), "-ar", list_stream[0][2]]
            if list_stream[0][3] > 2:
                cmd_reference.extend(["-ac", "2"])
                cmd_degraded.extend(["-ac", "2"])
            cmd_reference.append(list_file_wav[0])
            cmd_degraded.append(list_file_wav[1])
            tools.launch_cmdExt_with_timeout_reload(cmd_reference, 2, 600)
            tools.launch_cmdExt_with_timeout_reload(cmd_degraded, 2, 600)
            stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["visqol"], "--reference_file", list_file_wav[0], "--degraded_file", list_file_wav[1],
                                                                                  "--similarity_to_quality_model", path.join(path.dirname(tools.software["visqol"]), "libsvm_nu_svr_model.txt")], 2, 600)
            mos_lqo = re.search(r"MOS-LQO:\s*([0-9.]+)", stdout.decode("utf-8", errors="ignore"))
            if mos_lqo is None:
                raise AudioQualityError(f"ViSQOL gave no MOS-LQO for {degraded_file} against {reference_file} at {window_start:.3f} s: {stderror.decode('utf-8', errors='ignore').strip()}")
            list_mos.append(float(mos_lqo.group(1)))
    finally:
        for tmp_file_wav in list_file_wav:
            if path.exists(tmp_file_wav):
                tools.file_remove(tmp_file_wav)
    tools.dev_log(f"[ViSQOL] {degraded_file}: window MOS-LQO {list_mos} at {[round(window_start, 3) for window_start in list_window_start]} s of the source\n")
    return round(sum(list_mos) / len(list_mos), 5)

def add_visqol_tag(degraded_file, degraded_track, mos_lqo, tmp_file_base):
    """Append mos_lqo to the JSON list tag VMSAM_VISQOL of degraded_file's audio track, in place, and return the list."""
    """ Example:
    degraded_track["extra"] = {"ORIGINAL_ENCODE": "ac3:192000:32000", "VMSAM_ALREADY_TRANSFORMED": "1467.1", "VMSAM_VISQOL": "[3.38914]"}
    written track tags: ORIGINAL_ENCODE=ac3:192000:32000, VMSAM_ALREADY_TRANSFORMED=1467.1, VMSAM_VISQOL=[3.38914, 4.72626]
    """
    extra = dict(degraded_track.get("extra") or {})
    list_mos = json.loads(extra.get(VISQOL_TAG, "[]"))
    list_mos.append(mos_lqo)
    extra[VISQOL_TAG] = json.dumps(list_mos)

    tags = ElementTree.Element("Tags")
    track_tag = ElementTree.SubElement(tags, "Tag")
    ElementTree.SubElement(track_tag, "Targets")
    for name, value in extra.items():
        simple = ElementTree.SubElement(track_tag, "Simple")
        ElementTree.SubElement(simple, "Name").text = name
        ElementTree.SubElement(simple, "String").text = str(value)
    tmp_file_tags = f"{tmp_file_base}_tags.xml"
    ElementTree.ElementTree(tags).write(tmp_file_tags, encoding="utf-8", xml_declaration=True)
    try:
        tools.launch_cmdExt_with_timeout_reload([tools.software["mkvpropedit"], degraded_file, "--tags", f"track:a1:{tmp_file_tags}"], 2, 120)
    finally:
        tools.file_remove(tmp_file_tags)
    return list_mos
