"""
Stage 3 confirmer 2 (RATE-CORRECTED RE-MEASURE) of the PAL/speed family
design (DESIGN_PAL_SPEED_FAMILY_20260916.MD).

Ported from VMSAM_HELP_AI/tools/tool_logmel_ncc.py ("the crown jewel of
inv-fidelity" per TOOLS.MD -- "ready", no hardcoded paths). Core algorithm
only (mel filterbank / log-mel spectrogram / NCC slide search), adapted to
take decoded PCM arrays directly instead of requiring pre-cut WAV files on
disk, and to extract its own audio via `tools.software["ffmpeg"]` rather
than a bare CLI invocation on two file paths. Owes nothing to
`audioCorrelation.py`/chromaprint, which is the entire point: it is the
SECOND, independent metric that makes a before/after comparison non-circular
-- exactly the property `t119`'s pitch confirmer also has, by a different
route.

Apply the undo at the measured ratio to a probe extract, re-measure with
log-mel NCC. Expected shape (measured 4/4 on the real PAL trio, per the
design): NCC rises decisively, 0.26-0.42 before -> 0.84-0.97 after. A
confirmer that contradicts the discriminator declines with a named cause --
never forces. Refusal token: `no_rate_relation` (the design's own naming
rule states the PITCH confirmer's refusal is `no_pitch_relation`, "never
no_rate_relation" -- implying `no_rate_relation` names THIS confirmer's
failure instead; read as an interpretation of the naming scheme, not an
explicit assignment in the design text, and flagged as such for the
Architect to correct if wrong).
"""

import numpy as np
import scipy.signal
import tools

SR = 44100
N_FFT, HOP, N_MELS = 2048, 441, 40
NCC_FLOOR = 0.80  # "0.84-0.97 measured" (design) -- 0.80 is a margin below
                  # the measured floor, not the measured floor itself;
                  # FLAGGED, same shape as the other two modules' open
                  # constants: not derived from a population, a defensible
                  # starting point cited against the design's own numbers.


def _pcm(path, start_s, dur_s, audio_filter=None, sample_rate=SR):
    """Decode mono PCM. None on ANY failure -- ported convention from
    `pal_pitch_confirmer._pcm`: a decode that did not happen is not
    silence."""
    cmd = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", path,
           "-vn", "-ac", "1", "-ar", str(sample_rate)]
    if audio_filter:
        cmd += ["-af", audio_filter]
    cmd += ["-f", "s16le", "-"]
    stdout, stderror, exit_code = tools.launch_cmdExt_no_test(cmd)
    if exit_code != 0 or len(stdout) < sample_rate:
        return None
    return np.frombuffer(stdout, dtype="<i2").astype(np.float64) / 32768.0


def _mel_filterbank(sr, n_fft, n_mels, fmin=50, fmax=None):
    fmax = fmax or sr / 2

    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mmin, mmax = hz2mel(fmin), hz2mel(fmax)
    mel_pts = np.linspace(mmin, mmax, n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l:
                fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[i - 1, k] = (r - k) / (r - c)
    return fb


def _mel_spec(x, sr, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """Per-band z-scored log-mel spectrogram. None if too short to frame."""
    if x is None or len(x) < n_fft:
        return None
    f, t, spectrum = scipy.signal.stft(x, fs=sr, nperseg=n_fft,
                                       noverlap=n_fft - hop, boundary=None)
    mag = np.abs(spectrum)
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = fb @ mag
    logmel = np.log1p(mel)
    mu = logmel.mean(axis=1, keepdims=True)
    sd = logmel.std(axis=1, keepdims=True) + 1e-8
    return (logmel - mu) / sd


def _ncc_search(cand_z, master_z):
    """Slide `cand_z` over `master_z` (both n_mels x frames); best NCC and
    its offset (in master frames). None if either side has no usable frames
    or the candidate window is wider than the master window it searches."""
    if cand_z is None or master_z is None:
        return None, None
    n_cand = cand_z.shape[1]
    n_master = master_z.shape[1]
    if n_cand > n_master:
        return None, None
    best_ncc, best_off = -2.0, None
    cand_flat = cand_z.flatten()
    cand_norm = np.linalg.norm(cand_flat)
    for off in range(0, n_master - n_cand + 1):
        seg = master_z[:, off:off + n_cand].flatten()
        seg_norm = np.linalg.norm(seg)
        if seg_norm < 1e-6 or cand_norm < 1e-6:
            continue
        ncc = np.dot(cand_flat, seg) / (cand_norm * seg_norm)
        if ncc > best_ncc:
            best_ncc, best_off = ncc, off
    return best_off, (None if best_off is None else float(best_ncc))


def measure_ncc(master_path, candidate_path, start_seconds, window_seconds,
                 candidate_audio_filter=None, sample_rate=SR):
    """NCC of the candidate window (optionally filtered -- e.g. the ratio
    correction under test) against the BEST-matching position inside a
    WIDER master window, so an ordinary timing offset does not masquerade as
    a content mismatch. Returns the NCC, or None if either side could not be
    measured."""
    master_pcm = _pcm(master_path, max(0.0, start_seconds - window_seconds),
                       window_seconds * 3, sample_rate=sample_rate)
    candidate_pcm = _pcm(candidate_path, start_seconds, window_seconds,
                          audio_filter=candidate_audio_filter,
                          sample_rate=sample_rate)
    master_z = _mel_spec(master_pcm, sample_rate)
    cand_z = _mel_spec(candidate_pcm, sample_rate)
    _, ncc = _ncc_search(cand_z, master_z)
    return ncc


def confirm_rate_correction(master_path, candidate_path, start_seconds,
                             window_seconds, undo_filter, ncc_floor=NCC_FLOOR):
    """Stage 3, confirmer 2. Measures NCC on the SAME candidate window
    before and after applying `undo_filter` (an ffmpeg `-af` string built by
    the caller from the discriminator's measured ratio, e.g. via
    `merge_video_resample.build_speed_filter_chain`), against the SAME
    master window both times.

    Returns {ncc_before, ncc_after, rose_decisively, refusal, reason}.
    `refusal`, when not None, is `no_rate_relation` -- a DECLINE, never a
    forced acceptance: the design's own rule, "a confirmer that contradicts
    the discriminator declines with a named cause."
    """
    ncc_before = measure_ncc(master_path, candidate_path, start_seconds, window_seconds)
    ncc_after = measure_ncc(master_path, candidate_path, start_seconds, window_seconds,
                            candidate_audio_filter=undo_filter)
    if ncc_before is None or ncc_after is None:
        return {"ncc_before": ncc_before, "ncc_after": ncc_after,
                "rose_decisively": False, "refusal": "no_rate_relation",
                "reason": "could not measure NCC before and/or after correction"}
    rose = ncc_after > ncc_before and ncc_after >= ncc_floor
    if not rose:
        return {"ncc_before": round(ncc_before, 4), "ncc_after": round(ncc_after, 4),
                "rose_decisively": False, "refusal": "no_rate_relation",
                "reason": (f"NCC did not rise decisively after correction "
                           f"({ncc_before:.4f} -> {ncc_after:.4f}, floor "
                           f"{ncc_floor}): the discriminator's ratio does not "
                           f"hold up under this independent, non-chromaprint "
                           f"metric")}
    return {"ncc_before": round(ncc_before, 4), "ncc_after": round(ncc_after, 4),
            "rose_decisively": True, "refusal": None, "reason": None}
