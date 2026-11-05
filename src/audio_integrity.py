'''The check run on every re-encoded audio track: no silence gained against its source, no decode error.'''

import integrity
import tools

import math
import subprocess
import numpy as np


class AudioReencodeIntegrityError(Exception):
    pass


def check_reencoded_track(source_path, source_index, produced_path, delay_ms=0, produced_index=0):
    """Raise AudioReencodeIntegrityError when produced_path's stream gained a silence source_path's stream plays, or does not decode cleanly."""
    label = f"re-encode of track {source_index} of {source_path} into {produced_path}"
    findings = []
    report = integrity.silence_report(source_path, source_index, produced_path, produced_index, delay_ms)
    for gained in report["b_only"]:
        findings.append(f"{label}: gained a silence {gained['start_s']}-{gained['end_s']} s the source plays")
    stdout, stderror, exit_code = tools.launch_cmdExt_no_test(
        [tools.software["ffprobe"], "-v", "error", "-err_detect", "crccheck+bitstream+buffer",
         "-select_streams", str(produced_index), "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", produced_path])
    errors = stderror.decode("utf-8", errors="ignore").strip().splitlines() or ([f"ffprobe exit {exit_code}"] if exit_code else [])
    if errors:
        findings.append(f"{label}: decode errors: {' | '.join(errors[:3])}")
    if findings:
        raise AudioReencodeIntegrityError("\n".join(findings))

AUDIO_LENGTH_GAP = 0.01
AUDIO_STEREO_CORRELATION_MIN = 0.2
AUDIO_CLIPPED_SAMPLES_MAX = 1000
AUDIO_FULL_SCALE = 0.9999
AUDIO_LOSSY_CUTOFF_HZ = 21500
AUDIO_LOSSY_WALL_DB = 25
AUDIO_SPECTRUM_FFT = 8192

def remove_bad_audio(video_obj,list_audio):
    bad = {}
    for i,audio_a in enumerate(list_audio):
        for audio_b in list_audio[i+1:]:
            if audio_a['keep'] and audio_b['keep']:
                try:
                    silence_compare = integrity.silence_report(video_obj.filePath, audio_a['StreamOrder'], video_obj.filePath, audio_b['StreamOrder'])
                    if silence_compare['result'] == "A":
                        bad[id(audio_a)] = f"silent where track {audio_b['StreamOrder']} plays {silence_compare['a_only'][:2]}"
                    if silence_compare['result'] == "B":
                        bad[id(audio_b)] = f"silent where track {audio_a['StreamOrder']} plays {silence_compare['b_only'][:2]}"
                except Exception as e:
                    tools.log_always(f"[silence WARN] tracks {audio_a['StreamOrder']} and {audio_b['StreamOrder']} of {video_obj.filePath} not compared: {e}\n")
    """ Example:
        indicator       measure                                         out of standard
        silence         5 s silent where another track of video_obj plays
        length          decoded samples / sample rate vs video length   AUDIO_LENGTH_GAP apart
        stereo image    correlation of the front left and right         under AUDIO_STEREO_CORRELATION_MIN
        clipping        samples at or above 0 dBFS (AUDIO_FULL_SCALE)   over AUDIO_CLIPPED_SAMPLES_MAX
        fake lossless   lossless codec: highest frequency within 60 dB  under AUDIO_LOSSY_CUTOFF_HZ with a drop
                        of the 1-4 kHz level (median spectrum)          of AUDIO_LOSSY_WALL_DB over 500 Hz
        pt tracks [Brazilian: all pass, European: silent 576-583 s] -> European keep = False
        pt tracks [Brazilian: clipped, European: clipped] -> both kept: no track passes every test
    """
    try:
        video_length = float(video_obj.video['Duration'])
        probe_data = integrity.probe(video_obj.filePath)
        streams_by_index = {int(stream['index']): stream for stream in probe_data['streams']}
        for audio in list_audio:
            if audio['keep'] and id(audio) not in bad:
                stream = streams_by_index[int(audio['StreamOrder'])]
                channels, rate = int(stream['channels']), float(stream['sample_rate'])
                process = subprocess.Popen([tools.software["ffmpeg"], "-nostdin", "-v", "error", "-i", video_obj.filePath, "-map", f"0:{audio['StreamOrder']}",
                                            "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_f32le", "-f", "f32le", "-"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                samples, clipped, sums, spectra = 0, 0, np.zeros(5), []
                window = np.hanning(AUDIO_SPECTRUM_FFT)
                chunk = process.stdout.read(channels * 4 * 262144)
                while chunk:
                    pcm = np.frombuffer(chunk[:len(chunk) - len(chunk) % (channels * 4)], dtype=np.float32).reshape(-1, channels).astype(np.float64)
                    samples += len(pcm)
                    clipped += int((np.abs(pcm) >= AUDIO_FULL_SCALE).sum())
                    if channels >= 2:
                        sums += [pcm[:,0].sum(), pcm[:,1].sum(), (pcm[:,0]*pcm[:,0]).sum(), (pcm[:,1]*pcm[:,1]).sum(), (pcm[:,0]*pcm[:,1]).sum()]
                    if stream['codec_name'] in tools.AUDIO_LOSSLESS_CODECS:
                        mid = pcm.mean(axis=1)
                        for start in range(0, len(mid) - AUDIO_SPECTRUM_FFT, int(rate / 2)):
                            frame = mid[start:start + AUDIO_SPECTRUM_FFT] * window
                            if np.sqrt((frame * frame).mean()) > 10 ** (-50 / 20):
                                spectra.append((10 * np.log10(np.abs(np.fft.rfft(frame)) ** 2 + 1e-20)).astype(np.float32))
                    chunk = process.stdout.read(channels * 4 * 262144)
                process.wait()
                length_gap = abs(samples / rate - video_length) / video_length
                correlation = None
                if channels >= 2 and samples:
                    mean_l, mean_r = sums[0] / samples, sums[1] / samples
                    power = (sums[2] / samples - mean_l * mean_l) * (sums[3] / samples - mean_r * mean_r)
                    correlation = (sums[4] / samples - mean_l * mean_r) / math.sqrt(power) if power > 0 else None
                cutoff, wall = None, None
                if len(spectra) >= 10:
                    frequencies = np.fft.rfftfreq(AUDIO_SPECTRUM_FFT, 1 / rate)
                    bin_hz = frequencies[1]
                    level = np.convolve(np.median(np.array(spectra), axis=0), np.ones(int(150 / bin_hz)) / int(150 / bin_hz), mode="same")
                    audible = np.where((level - np.median(level[(frequencies >= 1000) & (frequencies <= 4000)]) > -60) & (frequencies < rate / 2 - 300))[0]
                    last = audible[-1] if len(audible) else 0
                    cutoff = float(frequencies[last])
                    wall = float(level[max(0, last - int(500 / bin_hz))] - level[min(len(level) - 1, last + int(500 / bin_hz))])
                audio["metrics"] = f"[keep INFO] track {audio['StreamOrder']} of {video_obj.filePath}: length_gap={round(length_gap,5)} correlation={None if correlation is None else round(correlation,3)} clipped={clipped} cutoff_hz={cutoff} wall_db={None if wall is None else round(wall,1)}\n"
                if length_gap > AUDIO_LENGTH_GAP:
                    bad[id(audio)] = f"length {round(samples / rate,3)} s against video {video_length} s"
                elif correlation is not None and correlation < AUDIO_STEREO_CORRELATION_MIN:
                    bad[id(audio)] = f"left/right correlation {round(correlation,3)}"
                elif clipped > AUDIO_CLIPPED_SAMPLES_MAX:
                    bad[id(audio)] = f"{clipped} samples at or above 0 dBFS"
                elif cutoff is not None and cutoff < AUDIO_LOSSY_CUTOFF_HZ and wall >= AUDIO_LOSSY_WALL_DB:
                    bad[id(audio)] = f"{stream['codec_name']} with a lossy spectrum: cutoff {round(cutoff)} Hz, wall {round(wall,1)} dB"
        if any(audio['keep'] and id(audio) not in bad for audio in list_audio):
            for audio in list_audio:
                if audio['keep']:
                    if id(audio) in bad:
                        audio['keep'] = False
                        tools.log_always(f"[keep WARN] track {audio['StreamOrder']} of {video_obj.filePath} not kept: {bad[id(audio)]}\n")
                        if "metrics" in audio:
                            tools.log_always(audio["metrics"])
                    else:
                        tools.dev_log(audio["metrics"])
    except Exception as e:
        tools.dev_log(f"[remove_bad_audio ERROR] We get this error during detection stream quality metrics{e}")