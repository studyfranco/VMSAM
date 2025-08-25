# frame_compare.py
# Created for alignment with project style (tools + Thread + ffmpeg pipelines)

from threading import Thread
from sys import stderr
import math
import struct
import tools

class FrameComparer:
    """
    Compare frames between two videos around a time window to find a gap region.
    Uses ffmpeg to pipe low-res gray raw frames, builds 32x32 aHash per frame,
    then searches the best alignment by minimizing Hamming distances within a band.
    """

    def __init__(self, ref_path, tgt_path, start_sec, end_sec, fps=10, band_width=20, max_search_frames=50, debug=False):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.fps = int(max(1, fps))
        self.band_width = int(max(1, band_width))
        self.max_search_frames = int(max(8, max_search_frames))
        self.side = 32
        self.debug = debug

    def _read_hashes(self, path, start_sec, dur_sec):
        # ffmpeg: scale to 32x32, gray, fps=N, output rawvideo (1 byte per pixel)
        ffmpeg = tools.software["ffmpeg"]
        w = h = self.side
        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur_sec}",
            "-i", path,
            "-vf", f"scale={w}:{h},fps={self.fps},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
        ]
        # Use no-test to stream; we handle status ourselves
        p_stdout, p_stderr, rc = tools.launch_cmdExt_no_test(cmd)
        if rc not in (0,):
            # Note: ffmpeg often returns 0 while streaming all frames; if not 0, we still try to parse what we received
            pass

        data = p_stdout  # bytes of concatenated frames (w*h per frame)
        frame_size = w * h
        n_frames = len(data) // frame_size
        hashes = []
        for i in range(n_frames):
            frame = data[i*frame_size:(i+1)*frame_size]
            # aHash: average threshold over 32*32=1024 bytes
            avg = sum(frame) / float(len(frame))
            h64 = 0
            # Use first 64 samples to keep cost low; or sample grid 8x8
            # Here: 8x8 down-sample by stride to build 64-bit hash
            stride = self.side // 8
            bitpos = 0
            for yy in range(0, self.side, stride):
                for xx in range(0, self.side, stride):
                    idx = yy * self.side + xx
                    if idx < len(frame) and frame[idx] >= avg:
                        h64 |= (1 << bitpos)
                    bitpos += 1
                    if bitpos >= 64:
                        break
                if bitpos >= 64:
                    break
            hashes.append(h64)
        if self.debug:
            stderr.write(f"[frame_compare] Read {len(hashes)} frames from {path}\n")
        return hashes

    @staticmethod
    def _popcount64(x: int) -> int:
        return int(x).bit_count()

    def _align_and_find_gap(self, ref_hashes, tgt_hashes):
        """
        Align sequences within a band around diagonal, find contiguous region with large mismatch.
        Returns (start_idx, end_idx) in target indices or None.
        """
        n = len(ref_hashes)
        m = len(tgt_hashes)
        if n == 0 or m == 0:
            return None

        # Limit comparison frames
        n = min(n, self.max_search_frames)
        m = min(m, self.max_search_frames)
        ref_hashes = ref_hashes[:n]
        tgt_hashes = tgt_hashes[:m]

        # Dynamic-band search: for each ref idx i, compare tgt j in [i-band, i+band]
        # Accumulate cost and detect a run of high costs (candidate gap)
        band = self.band_width
        costs = *m
        matches = *m

        for i in range(n):
            j0 = max(0, i - band)
            j1 = min(m-1, i + band)
            best_j = None
            best_d = 1e9
            for j in range(j0, j1+1):
                d = self._popcount64(ref_hashes[i] ^ tgt_hashes[j])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                costs[best_j] += best_d
                matches[best_j] += 1

        # Smooth cost to highlight a “gap window”
        window = max(3, self.fps // 3)
        smoothed = *m
        run = 0
        for j in range(m):
            run += costs[j]
            if j >= window:
                run -= costs[j-window]
            smoothed[j] = run

        # Locate the worst (max) smoothed cost region: candidate gap centered there
        if m == 0:
            return None
        center = max(range(m), key=lambda j: smoothed[j])
        # Heuristic gap half-width
        half = max(2, window // 2)
        start = max(0, center - half)
        end = min(m-1, center + half)
        if self.debug:
            stderr.write(f"[frame_compare] gap frames in tgt: {start}..{end} (center={center}, window={window})\n")
        return (start, end)

    def find_scene_gap_requirements(self, before_common=2, after_common=3):
        """
        Main entry:
        - Extract low-res frame hashes on both videos over [start_sec, end_sec]
        - Find high-cost region in target sequence
        - Return frames/time boundaries in target coordinates
        """
        dur = max(0.5, self.end_sec - self.start_sec)
        # small padding for stability
        pad = min(4.0, dur/2.0)
        start = max(0.0, self.start_sec - pad)
        dur2 = dur + 2*pad

        ref = self._read_hashes(self.ref_path, start, dur2)
        tgt = self._read_hashes(self.tgt_path, start, dur2)
        gap = self._align_and_find_gap(ref, tgt)
        if not gap:
            return None

        s_idx, e_idx = gap
        # Convert indices to target frame numbers at fps (relative to 'start')
        start_time = start + (s_idx / float(self.fps))
        end_time = start + (e_idx / float(self.fps))
        start_frame = int(round(start_time * self.fps))
        end_frame = int(round(end_time * self.fps))
        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time
        }