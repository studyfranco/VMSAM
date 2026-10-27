'''Perceptual quality of an audio encode against its source, measured by ViSQOL (decision: audio-quality).'''

import re
from os import path

import tools

VISQOL_WINDOW_DURATION = 30

class AudioQualityError(Exception):
    pass

def get_visqol_mos(reference_file, degraded_file):
    """Return the ViSQOL audio-mode MOS-LQO of degraded_file's audio against reference_file's, on 30 s taken in the middle of degraded_file."""
    stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["ffprobe"], "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", degraded_file], 2, 60)
    window_start = max(0, float(stdout.decode("utf-8", errors="ignore").strip()) / 2 - VISQOL_WINDOW_DURATION / 2)

    list_file_wav = []
    try:
        for audio_file in [reference_file, degraded_file]:
            tmp_file_wav = path.join(tools.tmpFolder, f"{path.splitext(path.basename(audio_file))[0]}_visqol.wav")
            list_file_wav.append(tmp_file_wav)
            tools.launch_cmdExt_with_timeout_reload([tools.software["ffmpeg"], "-y", "-nostdin", "-v", "error", "-ss", f"{window_start:.3f}", "-t", str(VISQOL_WINDOW_DURATION),
                                                     "-i", audio_file, "-ar", "48000", "-ac", "1", tmp_file_wav], 2, 600)

        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["visqol"], "--reference_file", list_file_wav[0], "--degraded_file", list_file_wav[1],
                                                                              "--similarity_to_quality_model", path.join(path.dirname(tools.software["visqol"]), "libsvm_nu_svr_model.txt")], 2, 600)
        mos_lqo = re.search(r"MOS-LQO:\s*([0-9.]+)", stdout.decode("utf-8", errors="ignore"))
        if mos_lqo is None:
            raise AudioQualityError(f"ViSQOL gave no MOS-LQO for {degraded_file} against {reference_file}: {stderror.decode('utf-8', errors='ignore').strip()}")
    finally:
        for tmp_file_wav in list_file_wav:
            if path.exists(tmp_file_wav):
                tools.file_remove(tmp_file_wav)
    return float(mos_lqo.group(1))
