# frame_compare.py
# Amélioré: pHash DCT 64‑bits + alignement à bande + repli scène ffmpeg

from fractions import Fraction
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
    1) Extrait des cadres basses résolutions 32x32 gris via ffmpeg, À LA CADENCE NATIVE
       (aucun filtre `fps=`, donc aucun cadre dupliqué ni perdu par ré-échantillonnage).
    2) Calcule un pHash DCT 64‑bits par cadre
    3) Aligne dans une bande et agrège un coût de dissimilarité pour trouver la pire zone (supposée rupture)
    Repli: détection de scène ffmpeg dans la même fenêtre.

    La cadence est portée en RATIONNEL EXACT (fps_num/fps_den, p.ex. r_frame_rate
    ffprobe ou FrameRate_Original). Jamais un flottant arrondi, jamais une valeur
    par défaut: un appelant qui ne peut pas mesurer la cadence ne doit pas
    construire cet objet — c'est un déclin, pas une estimation à 25.0.
    Chaque index de cadre renvoyé voyage avec sa grille (`fps_num`/`fps_den`):
    un numéro de cadre sans sa grille est la même erreur qu'un compte sans sa
    surface.
    """

    def __init__(self, ref_path, tgt_path, start_sec, end_sec,
                 fps_num, fps_den,
                 band_width_sec=2.0, max_search_sec=5.0, debug=False,
                 scene_threshold=0.30):
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)

        fps_num = int(fps_num)
        fps_den = int(fps_den)
        if fps_num <= 0 or fps_den <= 0:
            # Un déclin, pas une estimation: aucune cadence mesurée ne peut être
            # non positive, et il n'existe pas de valeur "par défaut" légitime
            # ici (voir ARCH_FRAME_ACCURATE.MD, défaut 3).
            raise ValueError(
                f"FrameComparer requires an exact positive frame-rate rational, "
                f"got fps_num={fps_num} fps_den={fps_den}")
        self.fps_num = fps_num
        self.fps_den = fps_den
        self.fps_frac = Fraction(fps_num, fps_den)
        # Flottant dérivé, JAMAIS la source de vérité: seulement pour les appels
        # ffmpeg (-ss/-t attendent des secondes flottantes) et l'affichage.
        self.fps = float(self.fps_frac)

        # EN SECONDES, PAS EN CADRES, ET CONVERTI PAR LA GRILLE.
        # `band_width`/`max_search_frames` étaient des COMPTES DE CADRES fixés
        # à une lecture forcée `fps=10` (défauts 20/50 -> ±2.0 s / 5.0 s
        # couverts). Lire maintenant à la cadence native aurait changé ce que
        # ces mêmes nombres couvrent en temps — sur 24000/1001, ±0.83 s / 2.09 s
        # au lieu de ±2.0 s / 5.0 s — SANS QUE RIEN DANS LE DIFF NE LE DISE
        # (finding du Lead, 2026-09-15). Les défauts ci-dessous reproduisent la
        # couverture EN TEMPS de l'ancien `fps=10, band_width=20,
        # max_search_frames=50`, et restent cette couverture quelle que soit la
        # grille réelle.
        self.band_width = max(1, self._round_frac(Fraction(band_width_sec).limit_denominator(10**6) * self.fps_frac))
        self.max_search_frames = max(8, self._round_frac(Fraction(max_search_sec).limit_denominator(10**6) * self.fps_frac))
        self.side = 32
        self.debug = debug
        self.scene_threshold = float(scene_threshold)

    @staticmethod
    def _popcount64(x: int) -> int:
        return int(x).bit_count()

    @staticmethod
    def _round_frac(frac: Fraction) -> int:
        """Arrondi UNE FOIS, sur le rationnel exact — jamais un flottant
        arrondi ni une troncature (`int()` sur une `Fraction` tronque vers
        zéro, ce n'est pas un arrondi: trouvé par le Lead sur la fenêtre de
        lissage, qui valait 7 cadres au lieu des 8 que le commentaire
        promettait)."""
        return int(frac + Fraction(1, 2))

    def _frame_index(self, seconds: float) -> int:
        """Index de cadre absolu à `seconds`, sur LA grille exacte de cet objet.

        Arrondi une seule fois, sur le rationnel exact — jamais une cadence
        arrondie en entier au préalable (défaut 1: `int(max(1, fps))`).
        """
        return self._round_frac(Fraction(seconds).limit_denominator(10**9) * self.fps_frac)

    def _ffmpeg_raw_frames(self, path, start_sec, dur_sec):
        # ffmpeg: scale 32x32 gray, rawvideo, À LA CADENCE NATIVE DE LA SOURCE.
        # AUCUN filtre `fps=N`: ce filtre RESAMPLE et duplique/perd des cadres
        # dès que N diverge de la cadence réelle — précisément le défaut 1
        # (23.976 -> 24 duplique un cadre par fenêtre de 30 s, sur tout le NTSC).
        # Lire ne doit pas changer la grille sur laquelle les cadres sont posés.
        ffmpeg = tools.software["ffmpeg"]
        w = h = self.side
        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur_sec}",
            "-i", path,
            "-vf", f"scale={w}:{h},format=gray",
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
        costs = [0] * m
        hits = [0] * m

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

        # lisser par fenêtre glissante (~0.33 s), calculée sur le rationnel exact
        # (arrondie, jamais tronquée -- voir `_round_frac`)
        window = max(3, self._round_frac(self.fps_frac / 3))
        smoothed = [0] * m
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
          - extrait les pHash des cadres sur [start_sec,end_sec], À LA CADENCE NATIVE
          - aligne ref/tgt dans une bande
          - renvoie frames/temps en coordonnées cible, AVEC LA GRILLE (fps_num/fps_den)
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

        # Index de cadre ABSOLU (depuis t=0 du fichier cible) du début de fenêtre,
        # calculé UNE SEULE FOIS sur le rationnel exact. Les indices locaux
        # (s_idx/e_idx, des positions RÉELLEMENT décodées, pas des multiples
        # d'une cadence forcée) s'y additionnent ensuite sans arrondi
        # supplémentaire — évite la dérive qu'un second `round(time * fps)`
        # réintroduirait (défaut 1's other face: reconvertir temps -> cadre
        # après être déjà passé par cadre -> temps).
        base_frame = self._frame_index(start)

        gap = self._align_and_find_gap(ref_hashes, tgt_hashes)
        if not gap:
            # Repli: scène ffmpeg
            fb = self._scene_gap_fallback(start, start + dur2)
            if not fb:
                return None
            s_time, e_time = fb
            start_frame = self._frame_index(s_time)
            end_frame = self._frame_index(e_time)
            start_time = s_time
            end_time = e_time
        else:
            s_idx, e_idx = gap
            start_time = start + (s_idx / self.fps)
            end_time = start + (e_idx / self.fps)
            start_frame = base_frame + s_idx
            end_frame = base_frame + e_idx

        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
        }
