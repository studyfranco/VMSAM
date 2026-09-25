# frame_compare.py
# Amélioré: pHash DCT 64‑bits + alignement à bande + repli scène ffmpeg

from fractions import Fraction
import subprocess
from sys import stderr
import numpy as np
from scipy.fft import dct
import tools
import repair_log

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
                 scene_threshold=0.30, crop_filters=None, time_scales=None):
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
        # GEOMETRY NORMALISATION, PER PATH, OPTIONAL AND ADDITIVE (edge
        # single-anchor ruling, 2026-09-22; ANALYSIS_edge_single_anchor.md
        # finding 2). `{path: "crop=w:h:x:y"}` -- a filter inserted BEFORE
        # `scale` for that path only. `None` (the default, and what every
        # existing caller passes by omission) rebuilds the EXACT ffmpeg
        # command this class shipped before, filter string included, so no
        # existing measurement moves by a bit.
        #
        # WHY IT BELONGS HERE AND NOT IN A SUBCLASS: the pHash instrument is
        # geometry-blind and fails SILENTLY. MEASURED on id 33 (Fallout
        # S01E05, master 1920x1080 vs candidate 1920x800): at master
        # t=1200 s, on provably identical content, Hamming<=6 gives 0/193
        # matches raw and 193/193 once the master is cropped to
        # `1920:800:0:140`. A caller that must normalise has exactly one
        # thing to change -- the filter chain in `_ffmpeg_raw_frames` -- and
        # duplicating that command in another module to add four characters
        # is how the two copies drift apart.
        self.crop_filters = dict(crop_filters) if crop_filters else {}
        # A RATE RELATION, PER PATH, OPTIONAL AND ADDITIVE (orchestrator stage
        # 4, 2026-09-24). `{path: Fraction r}` says: this file's content plays
        # `r` times FASTER than the reference's, so its RAW time `t` shows what
        # the reference shows at `t * r` -- the relation the audio speed sweep
        # measured (PAL 1001/960, NTSC 1001/1000). Every second this class is
        # handed for such a path is then read as REFERENCE-EQUIVALENT time:
        # the seek and the duration are divided by `r` before ffmpeg sees
        # them, and `_on_comparer_grid` re-indexes the decoded frames through
        # `native_rate / (grid_rate * r)` instead of `native_rate / grid_rate`.
        # WHY: without it a speed-changed candidate is placed on the grid by
        # WALL time, and its content drifts `r - 1` frames per frame -- MEASURED
        # arithmetic on errid-70 (25 fps against 24000/1001, r = 1001/960):
        # 0.427 s, ten frames, across one +/-10 s anchor window, which no
        # constant shift can absorb. On an exact speed-up pair the corrected
        # ratio is EXACTLY 1 (25 == 24000/1001 * 1001/960), so the candidate's
        # frame k simply faces the reference's frame k -- which is what a
        # speed-up physically is. `None` (the default, and every existing
        # caller) keeps the class bit-identical to before.
        self.time_scales = {}
        for scaled_path, scale in (time_scales or {}).items():
            if scale is None:
                continue
            scale = Fraction(scale)
            if scale <= 0:
                raise ValueError(f"FrameComparer time scale must be positive, "
                                 f"got {scale} for {scaled_path}")
            if scale != 1:
                self.time_scales[scaled_path] = scale

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
        # Le recadrage précède `scale`: normaliser la GÉOMÉTRIE ACTIVE avant
        # la réduction 32x32, jamais après (après, les bandes noires ont déjà
        # contaminé chaque bloc DCT). Absent par défaut -> chaîne identique à
        # celle d'avant, au caractère près.
        crop = self.crop_filters.get(path)
        vf = f"scale={w}:{h},format=gray" if not crop else f"{crop},scale={w}:{h},format=gray"
        # REFERENCE-EQUIVALENT SECONDS -> THIS FILE'S RAW SECONDS (see
        # `time_scales` in `__init__`). Absent for every path by default.
        scale = self.time_scales.get(path)
        if scale is not None:
            start_sec = float(Fraction(start_sec).limit_denominator(10**9) / scale)
            dur_sec = float(Fraction(dur_sec).limit_denominator(10**9) / scale)
        # A WINDOW OF NO LENGTH IS NEVER DECODED (ADDENDUM 26.1, measured): `-t 0.0` is not "zero
        # seconds" to ffmpeg, it is "no limit" -- a negative candidate window clamped to 0 here
        # decoded errid-84's WHOLE candidate (31,686 frames, 182 s) and id 691's three times.
        if dur_sec <= 0:
            tools.dev_log(f"frame_compare: _ffmpeg_raw_frames window_empty file={path} "
                          f"start_sec={start_sec} dur_sec={dur_sec} -- not decoded\n")
            return b""
        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur_sec}",
            "-i", path,
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
        ]
        # Une lecture complète suffit (fenêtres courtes)
        # BOUNDED (ADDENDUM 26.3: a timeout on EVERY decoder call): `tools.decoder_timeout_for`
        # the window's own length; past it the process is killed and `tools.decoder_timeout`
        # names the refusal -- a job never blocks a container.
        timeout = tools.decoder_timeout_for(dur_sec)
        tools.dev_log(f"frame_compare: _ffmpeg_raw_frames starting file={path} "
                      f"start_sec={start_sec} dur_sec={dur_sec} timeout_s={timeout}\n")
        try:
            with repair_log.announced("frame_compare", "ffmpeg", path) as call:
                done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      timeout=timeout)
                call["exit"] = done.returncode
        except subprocess.TimeoutExpired:
            raise tools.decoder_timeout("ffmpeg_raw_frames", timeout,
                                        f"file={path} start_sec={start_sec} dur_sec={dur_sec}")
        stdout, stderr_out, rc = done.stdout, done.stderr, done.returncode
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


def _nominal_shift_frames(offset_ms, fps_num, fps_den):
    frame_ms = 1000.0 * fps_den / fps_num
    return int(round(offset_ms / frame_ms))


# ===========================================================================
# NATIVE DECODE RATE VS LABEL GRID -- the layer BENEATH scene_anchor's own
# grid fix (`_probe_frame_rate` / `_frame_on_grid` there, landed a6fe40f5).
#
# THE DEFECT, measured on errid 5 (23.976-fps master against a 29.97-fps
# candidate): `_extract_hashes` labelled its array with
# `comparer._frame_index(start_s)` -- an index on the COMPARER's grid, which
# is the MASTER's by ruling (F1) -- while `_ffmpeg_raw_frames` deliberately
# decodes AT THE FILE'S OWN NATIVE RATE (no `fps=` filter; that filter IS
# defect 1). Element `i` was therefore CLAIMED to be grid frame `base + i`
# while it actually held the frame playing at `start_s + i / native_rate`.
# Every consumer reads the pair the same way and so inherits the same drift:
#
#   scene_anchor._frames_match   `ci = c_frame - c_base`
#   _validate_boundary           `ci = f - c_base`
#   _hamming_series              `ci = (m + shift_frames) - c_base`
#   locate_bracket_boundary      `hash_at(c_base, c_hashes, m + shift)`
#   _detect_cuts / _detect_stable_runs / _detect_uniform_runs
#                                `base + i` emitted as a grid frame number
#
# At 30000/1001 against 23.976 that is 25% of a frame of error PER FRAME --
# a whole frame every four -- so the owner's ">= 3 consecutive identical
# frames" anchor check cannot pass anywhere in the window, however right the
# window is. THE MASTER SIDE IS THE SAME DEFECT whenever the caller-supplied
# grid differs from the container's own rate, which errid 5 also does:
# MediaInfo hands `2997/125` while the container declares `24000/1001`.
#
# THE FIX, applied ONCE HERE rather than at each of the six comparison sites
# (four of which live in `scene_anchor.py` and would each need the rate
# threaded to them): a frame index that leaves this module is a MASTER-grid
# index by ruling, so an array carrying such a label must STEP on that same
# grid. Each natively-decoded array is projected onto the comparer's grid at
# the single point where it acquires its label -- output element `k` is the
# decoded frame playing `k` GRID frames into the window, not `k` NATIVE
# frames into it. Same exact-`Fraction` conversion as
# `scene_anchor._frame_on_grid` (an index carried between two grids through
# the instant it names), run on window-relative indices, at the one place
# the two coordinate systems meet.
#
# THIS IS NOT A RESAMPLE OF THE DECODE. `_ffmpeg_raw_frames` still reads
# every native frame, unfiltered, exactly as before -- defect 1 stays fixed.
# What is re-indexed is the LABELLING, and only when the two rates actually
# differ (see `_on_comparer_grid`'s equal-rate branch).
# ===========================================================================

# Per-path, successes only. `_extract_hashes` runs up to ten times per
# bracket over the same two files (`_validate_boundary` alone calls it four
# times) and a container's declared rate does not change under us mid-repair.
# A FAILURE IS NEVER CACHED: a transient ffprobe failure must not be frozen
# into a permanent decline for the rest of the process.
_NATIVE_RATE_CACHE = {}


def _parse_positive_rate(value):
    '''Exact positive rational, or `None`. SAME CONTRACT, deliberately NOT
    IMPORTED, as `scene_anchor._parse_positive_rate` -- which says the same
    of `merge_video_chimeric.parse_positive_rate`, and for the same reason
    one level down: `scene_anchor` imports THIS module at its own module
    level (`from frame_compare import FrameComparer, _extract_hashes`), so
    importing it back here would be a cycle.

    Every way of failing to read a rate -- absent, blank, non-positive,
    `"0/0"`, `"inf"`/`"nan"`, a malformed `"num/den"`, a wrong type --
    collapses to the SAME `None`: "I could not measure it" is a property of
    the value, not of its absence.
    '''
    if value is None:
        return None
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if rate > 0 else None


def _native_frame_rate(path):
    '''THE RATE `_ffmpeg_raw_frames` ACTUALLY DECODED THIS FILE AT, as an
    EXACT RATIONAL -- `(Fraction, None)` on success, `(None, reason)` when it
    could not be measured.

    `r_frame_rate` is the exact rational the container declares, read as the
    string ffprobe prints ("30000/1001"), never a float rounding of it --
    the same source and the same discipline this module's own class docstring
    already requires of every rate it carries ("Jamais un flottant arrondi").
    '''
    cached = _NATIVE_RATE_CACHE.get(path)
    if cached is not None:
        return cached, None
    try:
        cmd = [tools.software["ffprobe"], "-v", "error",
               "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
    except KeyError:
        return None, "ffprobe_not_configured"
    # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22): every
    # external tool call says which file it is on before it can hang.
    tools.dev_log(f"frame_compare: _native_frame_rate calling ffprobe "
                  f"file={path}\n")
    try:
        with repair_log.announced("frame_compare", "ffprobe", path) as call:
            stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
                cmd, max_restart=3, timeout=60)
            call["exit"] = exit_code
    except Exception as exc:
        return None, f"ffprobe_raised:{type(exc).__name__}"
    if exit_code != 0:
        return None, f"ffprobe_exit:{exit_code}"
    lines = stdout.decode("utf-8", "replace").strip().splitlines()
    raw = lines[0].strip() if lines else ""
    rate = _parse_positive_rate(raw)
    if rate is None:
        return None, f"unparseable_r_frame_rate:{raw!r}"
    _NATIVE_RATE_CACHE[path] = rate
    return rate, None


def _on_comparer_grid(comparer, path, values):
    '''Re-index one natively-decoded per-frame series onto the comparer's
    grid, so that element `k` really is what plays `k` GRID frames into the
    decoded window rather than `k` NATIVE frames into it. Returns
    `(values_on_grid, None)`, or `(None, reason)` when the file's own rate
    could not be measured.

    `values` is per-decoded-frame and rate-agnostic: pHashes
    (`_extract_hashes`) and brightness means (`_extract_brightness`) both go
    through here, so those two can never drift apart from each other either.

    THE CONVERSION IS THE RATIO OF THE TWO RATES, NOTHING ELSE -- element `k`
    comes from native element `round(k * native_rate / grid_rate)`, the same
    arithmetic as `scene_anchor._frame_on_grid` (a frame index carried
    between grids through the instant it names) applied to WINDOW-RELATIVE
    indices. Written this way on purpose, rather than as
    `round(((base + k) / grid_rate - start_s) * native_rate)`: that form is
    algebraically the same conversion PLUS the residue of `base`'s own
    rounding of `start_s`, and that residue lands element 0 on the wrong
    native frame about half the time -- MEASURED before this form was
    chosen, on a synthetic 24000/1001-vs-30000/1001 pair whose two sides
    hold identical content: the instant form matched 50% of frames, this one
    matches 100%.

    EQUAL RATES ARE THE IDENTITY, BY THE ARITHMETIC AND NOT ONLY BY THE
    EARLY RETURN: `round(k * 1)` is `k` for every `k`, with no residue that
    could sit on a rounding tie. The early return below is therefore an
    optimisation (it skips the probe's cost and the loop's), not the thing
    that makes same-rate behaviour bit-identical to the pre-fix code.

    THE HALF-FRAME `base` ALREADY CARRIED IS NEITHER FIXED NOR WORSENED
    HERE. Element 0 of a decoded window is the first frame ffmpeg emitted at
    or after `start_s`, and `base` is `start_s` ROUNDED onto the grid, so the
    two can disagree by up to half a grid frame. That was true before this
    function existed and stays exactly as true after it: element 0 still maps
    to element 0. Fixing it needs the window's real first PTS, which is a
    different measurement from the one this function makes.
    '''
    if not values:
        return values, None
    native_rate, reason = _native_frame_rate(path)
    if native_rate is None:
        return None, reason
    grid_rate = comparer.fps_frac
    # A SPEED-CHANGED PATH IS RE-INDEXED ON ITS CORRECTED GRID: one reference
    # grid frame of equivalent time is `1 / (grid_rate * r)` of this file's raw
    # time. `getattr` so a comparer built before `time_scales` existed (a
    # pickled or duck-typed one) keeps the old arithmetic exactly.
    scale = getattr(comparer, "time_scales", {}).get(path)
    if scale is not None:
        grid_rate = grid_rate * scale
    if native_rate == grid_rate:
        return values, None
    # Exact rationals throughout, rounded ONCE per output element, through
    # this module's own single rounding helper (`_round_frac`: round, never
    # `int()`'s truncation -- see its docstring).
    ratio = native_rate / grid_rate
    n_native = len(values)
    out = []
    k = 0
    while True:
        j = FrameComparer._round_frac(Fraction(k) * ratio)
        if j >= n_native:
            break
        out.append(values[j])
        k += 1
    return out, None


def _extract_hashes(comparer, path, start_s, dur_s):
    start_s = max(0.0, start_s)
    if dur_s <= 0:
        # NEVER `max(0.0, dur_s)` INTO ffmpeg (ADDENDUM 26.1): a window of no length is an
        # empty series, and every caller's own `frames_unextractable` decline takes it.
        return comparer._frame_index(start_s), []
    blob = comparer._ffmpeg_raw_frames(path, start_s, dur_s)
    hashes = comparer._phash64_frames(blob)
    base = comparer._frame_index(start_s)
    on_grid, reason = _on_comparer_grid(comparer, path, hashes)
    if on_grid is None:
        # DECLINE, NEVER GUESS THE GRID. Assuming the file decodes at the
        # comparer's rate is exactly the defect this block exists to remove,
        # and it is the silent kind: a wrong boundary reaches a destructive
        # splice, while an empty series reaches every caller's own
        # `frames_unextractable` decline and stops the repair. The evidence
        # those callers build cannot name THIS cause (they never see it), so
        # the true reason is written here, immediately, where a production
        # log will carry it next to the ffprobe call that produced it.
        tools.dev_log(f"frame_compare: _extract_hashes declining "
                      f"file={path} reason=native_rate_unmeasured:{reason} "
                      f"start_sec={start_s} decoded_frames={len(hashes)}\n")
        return base, []
    return base, on_grid
