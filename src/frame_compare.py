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


# ============================================================================
# THE F1 PRODUCER -- REFINES an already-INDICATED bracket to a frame index.
# NEVER SEARCHES FOR ONE: `offset_before_ms`/`offset_after_ms` are the
# adjacent segments' OWN measured offsets (change_point_locator.py already
# ran the coarse audio comparison; that is what makes a 100 s bracket
# tractable here -- the question is WHERE inside it the known before-state
# stops matching and the known after-state starts, never a third unknown
# offset). Contract: RULING_20260915_FRAME_INDEXED_BOUNDARY.MD ("F1").
#
# Wired from merge_video_chimeric.py (SPEC_ZONE_A.MD S4h: the chimeric
# builder narrows what the comparison stage only bracketed; it does not
# adopt the bracket). change_point_locator.py only carries the bracket as
# `following_bracket` transport on the segment before it -- it does not call
# this function itself.
#
# THREE NAMED SUB-STAGES, each with its own verdict, per the Lead's ruling
# 2026-09-15 on the design question this module's first live caller raised:
#   1. two_hypothesis_vote  -- cheap, works when the step is unambiguous
#      (high-motion content, no repetitive/degenerate zone in the way).
#   2. landmark_cutmatch    -- scene-cut cross-matching, REFINES inside the
#      bracket only, never searches for it; requires >=2 independent
#      landmarks with an EXACT 0-frame residual on EACH side, and a UNIQUE
#      winning shift (a tie among candidate shifts is TASKS/013 S1's alias
#      defect -- the peak and the second peak tying exactly -- so a tie is
#      treated as NO confirmation, never an arbitrary tie-break).
#   3. uniform_run          -- a held near-black card is DEGENERATE for both
#      of the above (every frame in it matches every other, at any offset),
#      so its own run length is used directly, only when exactly one such
#      run exists on each side (id 12 window A: TASK 0's hand-built target).
# None may promote itself to searching the whole bracket blind: all three read
# ONLY the two known offset hypotheses (plus a small +/-frame slack on stage
# 2, because the audio tier's own ms value can round to the wrong integer --
# measured on id 12 window B, TASK 0 report to the Lead 2026-09-15: -1412.34
# ms rounds to -34 frames, the exact frame-matching value is -33).
#
# RULE 7 (never silence): every call returns a dict. `declined` False means
# every F1 field is present; True means a NAMED reason and the evidence that
# produced it, never an empty/absent result standing for "no difference".
# ============================================================================

# Placeholders until TASKS/016 rules the boundary-validation window and
# floor -- named, not bare literals (WRITE_ZONES.MD: "no magic numbers").
# Overridable via config.ini `[frame_boundary]` (addition-only, WRITE_ZONES.MD
# S3); nobody has added that section yet, so today this always falls back to
# the defaults below.
BOUNDARY_VALIDATION_PROBE_FRAMES_DEFAULT = 50
BOUNDARY_VALIDATION_HAMMING_THRESHOLD_DEFAULT = 12

# Scene-cut spike threshold: a frame-to-frame Hamming distance at or above
# this is a real edit, not decode noise. CORRECTED from an initial 18 after
# target-first testing this producer against id 12 window A (TASK 0's own
# hand-built target): 18 fired ~400 times across a 106 s bracket that holds
# a high-contrast, fast-cutting VFX sequence, and `_confirm_shift`'s exact
# 0-residual match started tying by sheer combinatorial chance across that
# density -- a second, larger-scale sighting of the SAME alias-defect shape
# as `_confirm_shift`'s tie guard, this time from noise volume rather than
# periodicity. 30 measured clean on the SAME bracket: only genuine high-
# magnitude edits (the black-card entry/exit among them) survive it, and it
# still cleared window B's ordinary scene cuts (distances up to 36) with
# margin. TASKS/016 may recalibrate; this is not that ticket's ground truth.
SCENE_CUT_HAMMING_THRESHOLD = 30

# A run of NEAR-UNIFORM brightness (a black card, a held logo) is DEGENERATE
# for pHash -- every frame in it hashes near a constant pattern and matches
# ANY other frame from ANY such run, at ANY offset -- so per-frame Hamming
# distance carries no information inside one. The run's OWN LENGTH is the
# signal instead (TASK 0, id 12 window A: master holds 18 black frames,
# candidate only 1 -- the 17-frame difference IS the located boundary).
# Brightness is 0-255 (raw 8-bit gray mean); measured black card read 0.06,
# ordinary scene brightness 15-65 on the same file.
UNIFORM_RUN_BRIGHTNESS_THRESHOLD = 5.0
# A run this short is as likely to be a single dark frame in ordinary motion
# as a deliberate held card; TASK 0's real black card was 18 frames long.
UNIFORM_RUN_MIN_FRAMES = 3

# Lead's ruling 2026-09-15: "one landmark is not evidence" -- the bar
# demonstrated on real media was 15 and 6 independent confirming landmarks;
# this is the FLOOR, not the target.
LANDMARK_MIN_CONFIRMATIONS = 2

# How many frames either side of the audio tier's ms-derived nominal shift
# to search for the EXACT frame-matching integer. The audio tier's own
# quantum (~124-142 ms) can round to the wrong frame by roughly one quantum;
# a handful of frames of slack covers that without turning this into a blind
# search over the whole bracket.
LANDMARK_SHIFT_SEARCH_RADIUS = 4


def _boundary_validation_config():
    try:
        section = tools.config_loader(tools.config_file, "frame_boundary")
    except Exception:
        return (BOUNDARY_VALIDATION_PROBE_FRAMES_DEFAULT,
                BOUNDARY_VALIDATION_HAMMING_THRESHOLD_DEFAULT)
    probe = int(section.get("boundary_probe_frames",
                            BOUNDARY_VALIDATION_PROBE_FRAMES_DEFAULT))
    threshold = int(section.get("boundary_hamming_threshold",
                                BOUNDARY_VALIDATION_HAMMING_THRESHOLD_DEFAULT))
    return probe, threshold


def _nominal_shift_frames(offset_ms, fps_num, fps_den):
    frame_ms = 1000.0 * fps_den / fps_num
    return int(round(offset_ms / frame_ms))


def _extract_hashes(comparer, path, start_s, dur_s):
    start_s = max(0.0, start_s)
    blob = comparer._ffmpeg_raw_frames(path, start_s, max(0.0, dur_s))
    hashes = comparer._phash64_frames(blob)
    base = comparer._frame_index(start_s)
    return base, hashes


def _detect_cuts(base, hashes, threshold=SCENE_CUT_HAMMING_THRESHOLD):
    '''Frame-to-frame Hamming spikes -- real edits, master/candidate each on
    their OWN native timeline, no offset assumed here.'''
    cuts = {}
    for i in range(1, len(hashes)):
        d = FrameComparer._popcount64(hashes[i - 1] ^ hashes[i])
        if d >= threshold:
            cuts[base + i] = d
    return cuts


def _extract_brightness(comparer, path, start_s, dur_s):
    start_s = max(0.0, start_s)
    blob = comparer._ffmpeg_raw_frames(path, start_s, max(0.0, dur_s))
    side = comparer.side
    frame_size = side * side
    n = len(blob) // frame_size
    base = comparer._frame_index(start_s)
    out = []
    for i in range(n):
        block = blob[i * frame_size:(i + 1) * frame_size]
        out.append(float(np.frombuffer(block, dtype=np.uint8).mean()))
    return base, out


def _detect_uniform_runs(base, brightness, threshold=UNIFORM_RUN_BRIGHTNESS_THRESHOLD,
                         min_frames=UNIFORM_RUN_MIN_FRAMES):
    '''Contiguous runs of near-uniform (near-black) brightness, as
    (start_frame, end_frame_inclusive) pairs. See the module constants above
    for why this exists as its OWN landmark type rather than folding into
    `_detect_cuts`.'''
    runs = []
    start = None
    for i, v in enumerate(brightness):
        if v < threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_frames:
                runs.append((base + start, base + i - 1))
            start = None
    if start is not None and len(brightness) - start >= min_frames:
        runs.append((base + start, base + len(brightness) - 1))
    return runs


def _confirm_shift(master_cut_frames, candidate_cut_set, nominal_shift, radius):
    '''Which INTEGER shift in `nominal_shift +/- radius` puts the most MASTER
    cuts EXACTLY on a CANDIDATE cut (`master_frame - shift == candidate_frame`,
    zero residual, never a nearest-neighbour match)?

    A TIE AT THE TOP IS NOT A CONFIRMATION. TASKS/013 S1's alias defect is
    exactly this shape -- "the peak and the second peak tie exactly, and the
    tie-break picks the wrong one" -- so a tie here returns count 0 rather
    than arbitrarily choosing between the tied shifts. Two independent
    sightings of this defect landed the same day this function was written
    (TASK 0 report to the Lead, 2026-09-15); this is the mechanised guard
    against it, not a fix to the defect itself -- TASKS/013 stays open.

    Returns (best_shift, confirmation_count). count 0 means "not confirmed":
    either nothing matched, or the top matched and tied.
    '''
    counts = {}
    for delta in range(-radius, radius + 1):
        shift = nominal_shift + delta
        counts[shift] = sum(1 for m in master_cut_frames
                            if (m - shift) in candidate_cut_set)
    best = max(counts.values())
    if best == 0:
        return nominal_shift, 0
    winners = [shift for shift, count in counts.items() if count == best]
    if len(winners) != 1:
        return nominal_shift, 0
    return winners[0], best


def _validate_boundary(fps_num, fps_den, master_path, candidate_path,
                       start_frame, end_frame, before_shift, after_shift):
    '''The 50-frame / Hamming-12 boundary check named in the brief --
    placeholder until TASKS/016 rules the real window and floor. Checks that
    frames just BEFORE `start_frame` still match the candidate under
    `before_shift`, and frames just AFTER `end_frame` match under
    `after_shift`. Returns (similarity, margin): similarity is the matched
    fraction across both sides combined, margin is threshold minus the worst
    single-frame distance observed (negative means the validation FAILED,
    not that the caller should clamp it -- the caller decides what to do
    with a failing margin, this function only measures).
    '''
    probe_frames, threshold = _boundary_validation_config()
    frame_s = fps_den / fps_num
    comparer = FrameComparer(master_path, candidate_path, 0, 1, fps_num, fps_den)

    def side(anchor_s, shift_frames, frame_range):
        m_base, m_hashes = _extract_hashes(
            comparer, master_path, anchor_s, probe_frames * frame_s * 1.5)
        c_base, c_hashes = _extract_hashes(
            comparer, candidate_path, anchor_s + shift_frames * frame_s,
            probe_frames * frame_s * 1.5)
        ok = total = worst = 0
        for f in frame_range:
            mi, ci = f - m_base, f - c_base
            if 0 <= mi < len(m_hashes) and 0 <= ci < len(c_hashes):
                d = FrameComparer._popcount64(m_hashes[mi] ^ c_hashes[ci])
                total += 1
                worst = max(worst, d)
                if d <= threshold:
                    ok += 1
        return ok, total, worst

    before_first = max(0, start_frame - probe_frames)
    ok_b, tot_b, worst_b = side(before_first * frame_s, before_shift,
                                range(before_first, start_frame))
    ok_a, tot_a, worst_a = side(end_frame * frame_s, after_shift,
                                range(end_frame, end_frame + probe_frames))
    total = tot_b + tot_a
    ok = ok_b + ok_a
    similarity = round(ok / total, 4) if total else 0.0
    margin = threshold - max(worst_b, worst_a)
    return similarity, margin


def _monotone_split(pairs):
    '''`pairs`: [(d_before, d_after), ...] across the bracket, one entry per
    master frame, comparing it against ITS TWO KNOWN candidate hypotheses.

    Finds the TIGHTEST [lo, hi) such that no frame before `lo` ever has the
    AFTER hypothesis strictly winning, and no frame from `hi` on ever has the
    BEFORE hypothesis strictly winning. `lo == hi` is the ideal case: an
    exact, unambiguous single-frame flip. `lo < hi` is a genuine tie zone
    (both hypotheses match equally, e.g. a near-uniform run) and is reported
    as the located interval itself, not resolved further by this stage.

    Returns None when the data is NOT monotone at all -- some frame where
    AFTER strictly wins precedes some frame where BEFORE strictly wins,
    meaning the two-fixed-hypothesis model does not fit this bracket
    (measured on id 12 window A: dense, self-similar content upstream of the
    real transition aliases this stage outright; TASK 0 report, 2026-09-15).
    That is this stage's OWN decline, not a bug -- the caller falls through
    to landmark matching.
    '''
    n = len(pairs)
    after_ever_before = [False] * (n + 1)
    for i in range(n):
        db, da = pairs[i]
        after_ever_before[i + 1] = after_ever_before[i] or (
            db is not None and da is not None and da < db)
    before_ever_from = [False] * (n + 1)
    for i in range(n - 1, -1, -1):
        db, da = pairs[i]
        before_ever_from[i] = before_ever_from[i + 1] or (
            db is not None and da is not None and db < da)
    candidates = [i for i in range(n + 1)
                 if not after_ever_before[i] and not before_ever_from[i]]
    if not candidates:
        return None
    return min(candidates), max(candidates)


def _f1_payload(fps_num, fps_den, start_frame, end_frame, offset_after_frames,
                similarity, margin, method, evidence):
    frame_ms = 1000.0 * fps_den / fps_num
    return {
        "declined": False,
        "grid": {"num": fps_num, "den": fps_den},
        "master_start_frame": start_frame,
        "master_end_frame": end_frame,
        # RULE 8 (RULING_20260915...): the module's own sign convention --
        # negative = candidate missing content the master has. The AFTER
        # offset, because that is what the next piece will actually read the
        # candidate at.
        "candidate_offset_frames": offset_after_frames,
        "similarity": similarity,
        "margin": margin,
        # OPTIONAL, DERIVED, NEVER AUTHORITATIVE (rule 1) -- display only.
        "derived_ms": {
            "master_start_ms": round(start_frame * frame_ms, 2),
            "master_end_ms": round(end_frame * frame_ms, 2),
        },
        "method": method,
        "evidence": evidence,
    }


def locate_bracket_boundary(master_path, candidate_path, fps_num, fps_den,
                            bracket_low_ms, bracket_high_ms,
                            offset_before_ms, offset_after_ms, pad_sec=3.0,
                            debug=False):
    '''THE F1 producer, entry point. See the module-level block above for the
    design. Always returns a dict -- `declined` True or False, never neither.
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        # RULE 6: unrepresentable, not merely rejected. Mirrors
        # FrameComparer's own constructor guard, restated here because this
        # function may be called before any FrameComparer exists yet.
        return {"declined": True, "reason": "grid_unmeasured",
               "evidence": f"fps_num={fps_num} fps_den={fps_den}"}

    m_start_s = bracket_low_ms / 1000.0
    m_end_s = bracket_high_ms / 1000.0
    if m_end_s <= m_start_s:
        return {"declined": True, "reason": "empty_bracket",
               "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms"}

    comparer = FrameComparer(master_path, candidate_path, m_start_s, m_end_s,
                             fps_num, fps_den, debug=debug)

    off_lo_s = min(offset_before_ms, offset_after_ms) / 1000.0
    off_hi_s = max(offset_before_ms, offset_after_ms) / 1000.0

    m_base, m_hashes = _extract_hashes(
        comparer, master_path, m_start_s - pad_sec,
        (m_end_s - m_start_s) + 2 * pad_sec)
    c_base, c_hashes = _extract_hashes(
        comparer, candidate_path, m_start_s + off_lo_s - pad_sec,
        (m_end_s - m_start_s) + (off_hi_s - off_lo_s) + 2 * pad_sec)

    if not m_hashes or not c_hashes:
        return {"declined": True, "reason": "frames_unextractable",
               "evidence": f"master_frames={len(m_hashes)} "
                          f"candidate_frames={len(c_hashes)}"}

    before_shift = _nominal_shift_frames(offset_before_ms, fps_num, fps_den)
    after_shift = _nominal_shift_frames(offset_after_ms, fps_num, fps_den)

    m_first = comparer._frame_index(m_start_s)
    m_last = comparer._frame_index(m_end_s)

    def hash_at(base, hashes, frame):
        i = frame - base
        return hashes[i] if 0 <= i < len(hashes) else None

    pairs = []
    frames_in_bracket = []
    for m in range(m_first, m_last):
        mh = hash_at(m_base, m_hashes, m)
        if mh is None:
            continue
        chb = hash_at(c_base, c_hashes, m + before_shift)
        cha = hash_at(c_base, c_hashes, m + after_shift)
        db = FrameComparer._popcount64(mh ^ chb) if chb is not None else None
        da = FrameComparer._popcount64(mh ^ cha) if cha is not None else None
        pairs.append((db, da))
        frames_in_bracket.append(m)

    if not pairs:
        return {"declined": True, "reason": "bracket_unreadable",
               "evidence": f"no master frame in [{m_first},{m_last})"}

    # --- STAGE 1: two-hypothesis monotone vote --------------------------
    split = _monotone_split(pairs)
    if split is not None:
        lo, hi = split
        start_frame = (frames_in_bracket[lo] if lo < len(frames_in_bracket)
                       else frames_in_bracket[-1] + 1)
        end_frame = (frames_in_bracket[hi] if hi < len(frames_in_bracket)
                    else frames_in_bracket[-1] + 1)
        similarity, margin = _validate_boundary(
            fps_num, fps_den, master_path, candidate_path,
            start_frame, end_frame, before_shift, after_shift)
        return _f1_payload(
            fps_num, fps_den, start_frame, end_frame, after_shift,
            similarity, margin, method="two_hypothesis_vote",
            evidence=f"clean monotone split, {hi - lo} ambiguous frame(s)")

    # --- STAGE 2: landmark cross-matching --------------------------------
    m_cuts = _detect_cuts(m_base, m_hashes)
    c_cuts = _detect_cuts(c_base, c_hashes)
    c_cut_set = set(c_cuts)
    before_conf, before_count = _confirm_shift(
        list(m_cuts), c_cut_set, before_shift, LANDMARK_SHIFT_SEARCH_RADIUS)
    after_conf, after_count = _confirm_shift(
        list(m_cuts), c_cut_set, after_shift, LANDMARK_SHIFT_SEARCH_RADIUS)

    if before_count >= LANDMARK_MIN_CONFIRMATIONS and after_count >= LANDMARK_MIN_CONFIRMATIONS:
        before_landmarks = [f for f in m_cuts
                            if m_first <= f < m_last and (f - before_conf) in c_cut_set]
        after_landmarks = [f for f in m_cuts
                           if m_first <= f < m_last and (f - after_conf) in c_cut_set]
        last_before = max(before_landmarks, default=None)
        candidates_after = [f for f in after_landmarks
                            if last_before is None or f > last_before]
        first_after = min(candidates_after, default=None)
        if last_before is not None and first_after is not None and first_after >= last_before:
            similarity, margin = _validate_boundary(
                fps_num, fps_den, master_path, candidate_path,
                last_before, first_after, before_conf, after_conf)
            return _f1_payload(
                fps_num, fps_den, last_before, first_after, after_conf,
                similarity, margin, method="landmark_cutmatch",
                evidence=(f"{before_count} landmark(s) confirm shift "
                         f"{before_conf} before, {after_count} confirm "
                         f"{after_conf} after"))

    # --- STAGE 3: uniform-run landmark (a held card, degenerate for pHash) -
    # Ordinary cuts carry NO signal inside a near-uniform run (every frame in
    # it matches every other, at any shift), which is exactly why stage 2
    # cannot resolve id 12 window A's black card -- and cannot be MADE to by
    # raising its threshold, because the run is not a cut, it is an absence
    # of one. Only fires when each side has EXACTLY ONE such run: more than
    # one is ambiguous (which run?), and this stage does not guess.
    m_bright_base, m_brightness = _extract_brightness(
        comparer, master_path, m_start_s - pad_sec, (m_end_s - m_start_s) + 2 * pad_sec)
    c_bright_base, c_brightness = _extract_brightness(
        comparer, candidate_path, m_start_s + off_lo_s - pad_sec,
        (m_end_s - m_start_s) + (off_hi_s - off_lo_s) + 2 * pad_sec)
    m_runs = [r for r in _detect_uniform_runs(m_bright_base, m_brightness)
             if m_first <= r[0] < m_last]
    c_runs = _detect_uniform_runs(c_bright_base, c_brightness)
    # EXACTLY one on the master side (mandatory: more than one is ambiguous,
    # which run?). At most one on the candidate side, and ZERO is the common
    # case, not an exclusion: TASK 0 measured candidate's own transition as a
    # single frame -- below `UNIFORM_RUN_MIN_FRAMES`, so it never forms a run
    # at all. That absence IS the signal (the card is almost entirely trimmed
    # in the candidate), not a reason to skip this stage.
    if len(m_runs) == 1 and len(c_runs) <= 1:
        run_start, run_end = m_runs[0]
        # THE MASTER'S OWN RUN IS THE LOCATED INTERVAL. The candidate's run
        # need not be the same length -- a shorter one (or none) is exactly
        # the trimmed-card case (TASK 0: master 18 frames, candidate 1) and a
        # LONGER one is the mirror (candidate holds a card master trims).
        # Either way the master frames inside its own run have no reliable
        # per-frame candidate counterpart, which is what makes this interval
        # the boundary rather than something inside it.
        c_run_frames = (c_runs[0][1] - c_runs[0][0] + 1) if c_runs else 0
        similarity, margin = _validate_boundary(
            fps_num, fps_den, master_path, candidate_path,
            run_start, run_end + 1, before_shift, after_shift)
        return _f1_payload(
            fps_num, fps_den, run_start, run_end + 1, after_shift,
            similarity, margin, method="uniform_run",
            evidence=(f"master near-uniform run [{run_start},{run_end}] "
                     f"({run_end - run_start + 1} frames), candidate's own "
                     f"run {c_run_frames} frame(s)"))

    # --- DECLINE: structure present, could not narrow --------------------
    return {"declined": True, "reason": "structure_present_could_not_narrow",
           "evidence": (f"vote non-monotone; landmarks before={before_count} "
                       f"(need {LANDMARK_MIN_CONFIRMATIONS}) "
                       f"after={after_count} (need {LANDMARK_MIN_CONFIRMATIONS}); "
                       f"uniform runs master={len(m_runs)} candidate={len(c_runs)} "
                       f"(need exactly 1 each) "
                       f"in bracket [{bracket_low_ms},{bracket_high_ms}] ms"),
           "bracket_low_ms": bracket_low_ms, "bracket_high_ms": bracket_high_ms}
