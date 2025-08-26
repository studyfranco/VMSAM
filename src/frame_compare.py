# frame_compare.py
# Amélioré: pHash DCT 64‑bits + alignement à bande + repli scène ffmpeg

from threading import Thread
from sys import stderr
import math
import io
import struct
import numpy as np
from scipy.fft import dct
import tools

class FrameComparer:
    """
    Compare des cadres entre deux vidéos dans une fenêtre temporelle pour localiser une zone de rupture.
    1) Extrait des cadres basses résolutions 32x32 gris via ffmpeg
    2) Calcule un pHash DCT 64‑bits par cadre
    3) Aligne dans une bande et agrège un coût de dissimilarité pour trouver la pire zone (supposée rupture)
    Repli: détection de scène ffmpeg dans la même fenêtre.
    """

    def __init__(self, ref_path, tgt_path, start_sec, end_sec,
                 fps=10, band_width=20, max_search_frames=50, debug=False,
                 scene_threshold=0.30):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.fps = int(max(1, fps))
        self.band_width = int(max(1, band_width))
        self.max_search_frames = int(max(8, max_search_frames))
        self.side = 32
        self.debug = debug
        self.scene_threshold = float(scene_threshold)

    @staticmethod
    def _popcount64(x: int) -> int:
        return int(x).bit_count()

    def _ffmpeg_raw_frames(self, path, start_sec, dur_sec):
        # ffmpeg: scale 32x32 gray, fps=N, rawvideo
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
        # Une lecture complète suffit (fenêtres courtes)
        stdout, stderr_out, rc = tools.launch_cmdExt_no_test(cmd)
        if rc not in (0,):
            # on tente quand même de parser ce qu’on a reçu
            if self.debug:
                stderr.write(f"[frame_compare] ffmpeg returned {rc}, partial data used\n")
        return stdout

    def _phash64_frames(self, blob_bytes):
        # Chaque frame = side*side octets
        s = self.side
        frame_size = s * s
        n_frames = len(blob_bytes) // frame_size
        hashes = []
        if n_frames == 0:
            return hashes
        # base DCT 2D: DCT-II ligne puis colonne
        # Pour performance, on évite de realouer trop
        for i in range(n_frames):
            block = blob_bytes[i*frame_size : (i+1)*frame_size]
            arr = np.frombuffer(block, dtype=np.uint8).astype(np.float32)
            arr = arr.reshape((s, s))
            # DCT 2D
            dct_rows = dct(arr, norm='ortho', axis=0)
            dct_2d = dct(dct_rows, norm='ortho', axis=1)
            # Top-left 8x8
            d8 = dct_2d[:8, :8].copy()
            # Option: ignorer DC [0,0] dans le seuillage médian
            flat = d8.flatten()
            median = np.median(flat[1:]) if flat.size >= 2 else np.median(flat)
            h = 0
            bit = 0
            for r in range(8):
                for c in range(8):
                    v = d8[r, c]
                    if v > median:
                        h |= (1 << bit)
                    bit += 1
            hashes.append(h)
        if self.debug:
            stderr.write(f"[frame_compare] pHash frames: {len(hashes)}\n")
        return hashes

    def _align_and_find_gap(self, ref_hashes, tgt_hashes):
        """
        Agrège un coût minimal par indice cible dans une bande autour de la diagonale.
        Lisse le coût et extrait la zone max (supposée rupture).
        Retourne (start_idx, end_idx) en indices de la séquence cible ou None.
        """
        n = min(len(ref_hashes), self.max_search_frames)
        m = min(len(tgt_hashes), self.max_search_frames)
        if n == 0 or m == 0:
            return None
        ref = ref_hashes[:n]
        tgt = tgt_hashes[:m]

        band = self.band_width
        costs =  * m
        hits =  * m

        # Pour chaque i (ref), chercher le j (tgt) dans la bande [i-band, i+band] minimisant la distance
        for i in range(n):
            j0 = max(0, i - band)
            j1 = min(m - 1, i + band)
            best_j = None
            best_d = 1_000_000
            r = ref[i]
            for j in range(j0, j1 + 1):
                d = self._popcount64(r ^ tgt[j])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                costs[best_j] += best_d
                hits[best_j] += 1

        # lisser par fenêtre glissante (~0.33 s)
        window = max(3, self.fps // 3)
        smoothed =  * m
        run = 0
        for j in range(m):
            run += costs[j]
            if j >= window:
                run -= costs[j - window]
            smoothed[j] = run

        if m == 0:
            return None

        center = max(range(m), key=lambda j: smoothed[j])
        half = max(2, window // 2)
        start = max(0, center - half)
        end = min(m - 1, center + half)

        if self.debug:
            stderr.write(f"[frame_compare] gap tgt frames: {start}..{end} (center={center}, window={window})\n")
        return (start, end)

    def _scene_gap_fallback(self, start_sec, end_sec):
        """
        Recherche des timestamps de rupture par ffmpeg scene detection dans la fenêtre,
        renvoie une petite fenêtre autour de la valeur médiane détectée si dispo.
        """
        ffmpeg = tools.software["ffmpeg"]
        dur = max(0.5, end_sec - start_sec)
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur}",
            "-i", self.tgt_path,
            "-vf", f"select='gt(scene,{self.scene_threshold})',showinfo",
            "-f", "null", "-"
        ]
        # showinfo écrit sur stderr
        out, err, rc = tools.launch_cmdExt_no_test(cmd)
        text = err.decode("utf-8", errors="ignore")
        import re
        times = []
        # showinfo… pts_time:123.456
        for m in re.finditer(r"pts_time:([0-9]+\.[0-9]+)", text):
            ts = float(m.group(1))
            # convertir vers temps global (cmd déjà -ss)
            times.append(ts + start_sec)
        if not times:
            return None
        c = times[len(times) // 2]
        band = max(0.2, min(2.0, dur * 0.2))
        return (max(start_sec, c - band), min(end_sec, c + band))

    def find_scene_gap_requirements(self, before_common=2, after_common=3):
        """
        Entrée principale:
          - extrait les pHash des cadres sur [start_sec,end_sec]
          - aligne ref/tgt dans une bande
          - renvoie frames/temps en coordonnées cible
        """
        dur = max(0.5, self.end_sec - self.start_sec)
        # léger pad pour stabilité
        pad = min(4.0, dur / 2.0)
        start = max(0.0, self.start_sec - pad)
        dur2 = dur + 2 * pad

        ref_blob = self._ffmpeg_raw_frames(self.ref_path, start, dur2)
        tgt_blob = self._ffmpeg_raw_frames(self.tgt_path, start, dur2)
        ref_hashes = self._phash64_frames(ref_blob)
        tgt_hashes = self._phash64_frames(tgt_blob)

        gap = self._align_and_find_gap(ref_hashes, tgt_hashes)
        if not gap:
            # Repli: scène ffmpeg
            fb = self._scene_gap_fallback(start, start + dur2)
            if not fb:
                return None
            s_time, e_time = fb
            s_idx = int(round((s_time - start) * self.fps))
            e_idx = int(round((e_time - start) * self.fps))
        else:
            s_idx, e_idx = gap

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