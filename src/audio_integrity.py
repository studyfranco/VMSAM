'''The check run on every re-encoded audio track: no silence gained against its source, no decode error.'''

import subprocess

import integrity
import tools


class AudioReencodeIntegrityError(Exception):
    pass


def check_reencoded_track(source_path, source_index, produced_path, delay_ms=0, produced_index=0):
    """Raise AudioReencodeIntegrityError when produced_path's stream gained a silence source_path's stream plays, or does not decode cleanly."""
    label = f"re-encode of track {source_index} of {source_path} into {produced_path}"
    findings = []
    report = integrity.silence_report(source_path, source_index, produced_path, produced_index, delay_ms)
    for gained in report["b_only"]:
        findings.append(f"{label}: gained a silence {gained['start_s']}-{gained['end_s']} s the source plays")
    decode = subprocess.run(
        [tools.software["ffprobe"], "-v", "error", "-err_detect", "crccheck+bitstream+buffer",
         "-select_streams", str(produced_index), "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", produced_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    errors = decode.stderr.strip().splitlines() or ([f"ffprobe exit {decode.returncode}"] if decode.returncode else [])
    if errors:
        findings.append(f"{label}: decode errors: {' | '.join(errors[:3])}")
    if findings:
        raise AudioReencodeIntegrityError("\n".join(findings))
