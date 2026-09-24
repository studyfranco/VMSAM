# frame_compare.py
# Amélioré: pHash DCT 64‑bits + alignement à bande + repli scène ffmpeg

from fractions import Fraction
from sys import stderr
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
        cmd = [
            ffmpeg, "-v", "error", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur_sec}",
            "-i", path,
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
        ]
        # Une lecture complète suffit (fenêtres courtes)
        # IMMEDIATELY-PRE-CALL (owner's order via the Lead, 2026-09-22):
        # `launch_cmdExt_no_test` is genuinely unbounded, reached from the
        # repair path (merge_video_chimeric.py's frame-tier calls).
        tools.dev_log(f"frame_compare: _ffmpeg_raw_frames starting file={path} "
                      f"start_sec={start_sec} dur_sec={dur_sec}\n")
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
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
            cmd, max_restart=3, timeout=60)
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
    blob = comparer._ffmpeg_raw_frames(path, start_s, max(0.0, dur_s))
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


def _detect_cuts(base, hashes, threshold=SCENE_CUT_HAMMING_THRESHOLD):
    '''Frame-to-frame Hamming spikes -- real edits, master/candidate each on
    its OWN timeline, no offset assumed here. "Own timeline" meant "own
    NATIVE decode rate" when this was written; since `_on_comparer_grid`
    both series arrive already labelled on the COMPARER's grid, so `base + i`
    really is the grid frame number the emitted cut claims to be -- which is
    what lets `_confirm_shift` below subtract a grid-frame shift from one
    side's cuts and look them up in the other's.'''
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
    # SAME GRID DEFECT, SAME FIX (see `_on_comparer_grid` above):
    # `_detect_uniform_runs` emits `base + i` as a frame number and stage 3
    # of `locate_bracket_boundary` compares those numbers against `m_first`/
    # `m_last`, which are grid indices. A brightness series stepping at the
    # native rate under a grid label drifts exactly as the hash series did.
    on_grid, reason = _on_comparer_grid(comparer, path, out)
    if on_grid is None:
        tools.dev_log(f"frame_compare: _extract_brightness declining "
                      f"file={path} reason=native_rate_unmeasured:{reason} "
                      f"start_sec={start_s} decoded_frames={n}\n")
        return base, []
    return base, on_grid


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


# STAGE 4 -- static-run length-differencing (owner's technique, verbatim:
# "c'est pour cela que tu peux utiliser le changement de scene. faire un
# pHash sur toute une sequence meme quelques frames apres des changements
# de scenes permet de savoir quand une image static est garde 10 ou 15
# frames des plus" -- RULING_20260916_MINIMAL_DESTRUCTION_NORTH_STAR.MD,
# "The static-span class, dissolved by the owner"). Inside a static span
# no frame matches any frame better than another (RULE stated already at
# stage 3's own comment above) -- but the span's ENTRY and EXIT are what
# `_detect_cuts` already finds. Anchor on those, COUNT the run length on
# each side, and the delta IS the divergence -- measured by counting, not
# by matching.
#
# PARAMETERIZED, NOT CHOSEN (dev-tiergate mission, 2026-09-17, Architect's
# ruling, same discipline as `BOUNDARY_VALIDATION_*` until TASKS/016): the
# triage's own static-run distribution calibrates these when it lands.
STATIC_RUN_STABILITY_HAMMING_THRESHOLD_DEFAULT = 4
STATIC_RUN_MIN_FRAMES_DEFAULT = 3


def _static_run_config():
    try:
        section = tools.config_loader(tools.config_file, "frame_boundary")
    except Exception:
        return (STATIC_RUN_STABILITY_HAMMING_THRESHOLD_DEFAULT,
                STATIC_RUN_MIN_FRAMES_DEFAULT)
    stability = int(section.get("static_run_stability_hamming_threshold",
                                STATIC_RUN_STABILITY_HAMMING_THRESHOLD_DEFAULT))
    min_frames = int(section.get("static_run_min_frames",
                                 STATIC_RUN_MIN_FRAMES_DEFAULT))
    return stability, min_frames


def _detect_stable_runs(base, hashes, threshold=None, min_frames=None):
    '''GENERALIZED run detector -- inter-frame pHash near-identity, not
    brightness. `_detect_uniform_runs` (stage 3, above) gates on
    near-zero BRIGHTNESS (`v < threshold` against a raw 0-255 mean): a
    bright static logo, a freeze-frame, or a non-dark held title card
    NEVER registers there, confirmed at that function's own source line
    (forensic advisory, 2026-09-17), regardless of how long it is held --
    the owner's "une image statique" reads broader than brightness alone.

    This detector keys on STABILITY instead: consecutive-frame Hamming
    distance at or below `threshold`. Near-black content is a SPECIAL
    CASE of this, not a separate one -- a solid dark frame is trivially
    stable frame-to-frame too, so this detector strictly generalizes
    stage 3's population rather than replacing its criterion with an
    unrelated one.

    Returns `[(start_frame, end_frame_inclusive), ...]`, the SAME shape as
    `_detect_uniform_runs`, so either can serve a caller expecting a run
    list. A single frame is not a run by itself (`min_frames` floor,
    same rule as stage 3): a lone stable pair proves nothing about a HELD
    span.
    '''
    cfg_threshold, cfg_min_frames = _static_run_config()
    threshold = cfg_threshold if threshold is None else threshold
    min_frames = cfg_min_frames if min_frames is None else min_frames
    runs = []
    if len(hashes) < 2:
        return runs
    start = 0
    for i in range(1, len(hashes)):
        d = FrameComparer._popcount64(hashes[i - 1] ^ hashes[i])
        if d > threshold:
            if i - start >= min_frames:
                runs.append((base + start, base + i - 1))
            start = i
    if len(hashes) - start >= min_frames:
        runs.append((base + start, base + len(hashes) - 1))
    return runs


def _count_corroborates_offset_step(delta_frames, frame_ms, step_ms, quantum_ms):
    '''Point (i), Architect's ruling 2026-09-17: the locator's own
    audio-measured step across this bracket (`step_ms`, SIGNED, this
    file's own convention) and this stage's frame count are two
    INDEPENDENT instruments. They must agree within the pair's own audio
    quantum, or this stage declines -- a frame count with no
    corroboration is an unchecked second opinion, not evidence, and a
    disagreement (including a sign disagreement) is a decline, never a
    pick between the two.
    '''
    if step_ms is None or quantum_ms is None:
        return False
    return abs(delta_frames * frame_ms - step_ms) <= quantum_ms


def _static_run_length_delta(m_hashes, m_base, m_first, m_last,
                              c_hashes, c_base):
    '''THE COUNTING PRIMITIVE, isolated and independently testable: given
    the master's own stable run bounded inside `[m_first, m_last)` and
    the candidate's own stable run (anywhere in what was extracted), name
    the frame-count DIVERGENCE between them -- never a matched position.

    Fires ONLY when each side holds EXACTLY ONE such run (same "more than
    one is ambiguous" rule as stage 3): with two static spans on either
    side, WHICH one corresponds to which is a question this counting
    method cannot answer by itself.

    Returns `None` when the precondition does not hold (zero or multiple
    runs on either side) -- a decline for the CALLER to act on, not a
    guess. Otherwise returns
    `(m_run_start, m_run_end, m_run_len, c_run_len, delta_frames)`,
    `delta_frames = c_run_len - m_run_len` (RULE 8's own sign sense:
    positive means the CANDIDATE holds the span longer).

    `delta_frames == 0` on two genuinely equal-length runs is NOT rounded
    away or treated as noise -- it is the correct, exact answer, and the
    arm this function is most required to get right (Architect's ruling,
    2026-09-17: "a comparator that reports a plausible small divergence
    on identical material is worse than one that reports nothing").
    '''
    m_runs = [r for r in _detect_stable_runs(m_base, m_hashes)
             if m_first <= r[0] < m_last and r[1] < m_last]
    c_runs = _detect_stable_runs(c_base, c_hashes)
    if len(m_runs) != 1 or len(c_runs) != 1:
        return None
    m_run_start, m_run_end = m_runs[0]
    c_run_start, c_run_end = c_runs[0]
    m_run_len = m_run_end - m_run_start + 1
    c_run_len = c_run_end - c_run_start + 1
    delta_frames = c_run_len - m_run_len
    return m_run_start, m_run_end, m_run_len, c_run_len, delta_frames


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
        # STRINGS, NOT NUMBERS (Architect's ruling, 2026-09-16): the exact
        # contract is the frame index + `grid` above; this ms projection
        # is a ROUNDED, human-facing convenience, and a number here
        # invites arithmetic on a value that was never meant to carry it --
        # exactly what happened at this mission's combination site (lost
        # 0.0044ms consuming this field instead of `master_start_frame` x
        # the exact grid). A string makes that mistake a `TypeError` a
        # reviewer sees, not a silent precision loss nobody does.
        "derived_ms": {
            "master_start_ms": f"{round(start_frame * frame_ms, 2)}",
            "master_end_ms": f"{round(end_frame * frame_ms, 2)}",
        },
        "method": method,
        "evidence": evidence,
    }


def locate_bracket_boundary(master_path, candidate_path, fps_num, fps_den,
                            bracket_low_ms, bracket_high_ms,
                            offset_before_ms, offset_after_ms, pad_sec=3.0,
                            debug=False, step_ms=None, quantum_ms=None):
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
    frame_ms = 1000.0 * fps_den / fps_num

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
    # BOUNDED ON BOTH ENDS (dev-tiergate mission, 2026-09-17, live-fired
    # from this file's own gate: `method=uniform_run similarity=0.5385` in
    # production). The original filter bounded only the run's START
    # (`r[0]`); `_extract_brightness` reads a PADDED window
    # (`m_start_s - pad_sec` to `... + 2*pad_sec`), so a run that begins
    # inside the bracket can extend past `m_last` into the padding -- and
    # this stage would report `run_end+1` as `master_end_frame`, a frame
    # OUTSIDE the interval this stage ever searched. `locate_match_onset`
    # (this file, below) already self-enforces the equivalent contract
    # ("refuse to return anything that is not STRICTLY narrower than the
    # input") before ever returning `declined: False` -- this stage had no
    # matching check. Consuming code (`merge_video_chimeric.py`'s interior
    # clamp) happens to absorb the overrun today, which is exactly the
    # "masking a live unbounded return path" shape the Lead named when
    # this was first reported, not evidence the gap is harmless: any
    # future direct consumer of this function inherits an unrepresented
    # contract violation. A run whose END also lies past `m_last` is
    # EXCLUDED here, not clamped -- the run bled across the bracket
    # boundary into content the locator already established as matching,
    # which contradicts the premise this stage searches an undecided gap,
    # so declining (falling through to `structure_present_could_not_
    # narrow`) is the honest answer, matching this file's own "refuse
    # rather than fabricate" rule everywhere else.
    m_runs = [r for r in _detect_uniform_runs(m_bright_base, m_brightness)
             if m_first <= r[0] < m_last and r[1] < m_last]
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

    # --- STAGE 4: static-run length-differencing (owner's technique, verbatim
    # in the module header above; Architect's ruling, 2026-09-17, 5 points) -
    # stage 3 trusts the MASTER's own run boundaries and never asks whether
    # the two runs are the SAME LENGTH. When both sides hold a genuine
    # static span held a DIFFERENT number of frames, that is exactly what
    # stage 3 cannot answer -- pHash carries no matching signal inside
    # either run, so the divergence is COUNTED, not matched.
    #
    # SIGN CONVENTION, STATED EXPLICITLY BECAUSE A RELAYED VERSION OF THIS
    # RULING'S POINT (iii) READ AS THE OPPOSITE OF WHAT IT SAYS HERE, AND
    # I FLAGGED RATHER THAN GUESSED (open with the Architect, 2026-09-17):
    # `_static_run_length_delta`'s own `delta_frames = c_run_len - m_run_len`
    # is the SAME sign RULE 8 already uses in this file (negative =
    # candidate MISSING content the master has). `delta_frames > 0` means
    # the CANDIDATE holds the span longer; `delta_frames < 0` means the
    # MASTER holds it longer. If the Architect's intended `D` is the
    # opposite sign, only the two branches immediately below need
    # swapping -- isolated here on purpose so that correction costs one
    # read, not a rewrite.
    delta_result = _static_run_length_delta(m_hashes, m_base, m_first, m_last,
                                            c_hashes, c_base)
    if delta_result is not None:
        static_run_start, static_run_end, static_m_len, static_c_len, delta_frames = delta_result
        if delta_frames == 0:
            # Equal-length runs: NO divergence to report -- falls through
            # to the generic decline below, same as "no run at all", never
            # a fabricated zero-width interval standing in for a real one.
            pass
        elif step_ms is None or quantum_ms is None:
            # POINT (i), the corroboration this stage cannot skip: with no
            # audio step/quantum to check the count against, a frame count
            # is an unchecked second opinion, not evidence. Decline named
            # from the missing input, not the missing agreement -- the two
            # are different facts (this campaign's own standing rule).
            return {"declined": True, "reason": "step_or_quantum_unavailable",
                   "evidence": (f"counted delta={delta_frames} frames has no "
                              f"audio step/quantum to corroborate against "
                              f"(step_ms={step_ms} quantum_ms={quantum_ms})"),
                   "bracket_low_ms": bracket_low_ms, "bracket_high_ms": bracket_high_ms}
        elif not _count_corroborates_offset_step(
                delta_frames, frame_ms, step_ms, quantum_ms):
            # POINT (i): two independent instruments disagreeing is a
            # DECLINE, never a pick between them.
            return {"declined": True, "reason": "count_contradicts_offset_step",
                   "evidence": (f"counted delta={delta_frames} frames "
                              f"({delta_frames * frame_ms:.2f} ms) vs the "
                              f"locator's own audio step={step_ms} ms, "
                              f"quantum={quantum_ms} ms"),
                   "bracket_low_ms": bracket_low_ms, "bracket_high_ms": bracket_high_ms}
        elif abs(delta_frames) > static_m_len:
            # A convention slice wider than the run it is placed inside
            # would fall OUTSIDE the run, into content never shown to be
            # interchangeable -- decline rather than let the convention
            # silently stop being one.
            return {"declined": True, "reason": "count_exceeds_run_extent",
                   "evidence": (f"counted delta={delta_frames} frames exceeds "
                              f"the master run's own length ({static_m_len} "
                              f"frames) -- the exit-edge convention slice "
                              f"would not fit inside the run it is placed in"),
                   "bracket_low_ms": bracket_low_ms, "bracket_high_ms": bracket_high_ms}
        else:
            # POINT (ii): PLACEMENT IS A CONVENTION, NOT A MEASUREMENT.
            # Every frame inside a content-uniform run is interchangeable
            # by construction -- any splice point within it yields
            # identical output -- so there is no correct position to
            # DISCOVER here, only a deterministic one to DECLARE. THE EXIT
            # EDGE: the divergent, `abs(delta_frames)`-wide span sits
            # immediately before the master's own run end. The payload
            # says so explicitly (`placement`) so no reader ever mistakes
            # this for a located transition.
            width = abs(delta_frames)
            conv_start = static_run_end + 1 - width
            conv_end = static_run_end + 1
            # POINT (iv): VALIDATE THE RUN'S FULL BOUNDS, NEVER THE
            # CONVENTION SLICE. The convention interval sits INSIDE the
            # run -- probing its own flanks would test frames that are
            # themselves part of the same uniform content, trivially
            # green, proving nothing. The identity that must hold is
            # ACROSS the run's TRUE edges (`static_run_start`,
            # `static_run_end + 1`), outside it on both sides, where the
            # scene changes this stage anchored on actually are.
            similarity, margin = _validate_boundary(
                fps_num, fps_den, master_path, candidate_path,
                static_run_start, static_run_end + 1, before_shift, after_shift)
            payload = _f1_payload(
                fps_num, fps_den, conv_start, conv_end, after_shift,
                similarity, margin, method="static_run_length_diff",
                evidence=(f"master run [{static_run_start},{static_run_end}] "
                         f"({static_m_len} frames), candidate run "
                         f"({static_c_len} frames), counted delta="
                         f"{delta_frames} frames, corroborated by "
                         f"step_ms={step_ms} quantum_ms={quantum_ms}"))
            payload["placement"] = "convention_exit_edge"
            return payload

    # --- DECLINE: structure present, could not narrow --------------------
    return {"declined": True, "reason": "structure_present_could_not_narrow",
           "evidence": (f"vote non-monotone; landmarks before={before_count} "
                       f"(need {LANDMARK_MIN_CONFIRMATIONS}) "
                       f"after={after_count} (need {LANDMARK_MIN_CONFIRMATIONS}); "
                       f"uniform runs master={len(m_runs)} candidate={len(c_runs)} "
                       f"(need exactly 1 each) "
                       f"in bracket [{bracket_low_ms},{bracket_high_ms}] ms"),
           "bracket_low_ms": bracket_low_ms, "bracket_high_ms": bracket_high_ms}


def _hamming_series(comparer, master_path, candidate_path, start_s, end_s,
                    shift_frames, pad_sec):
    '''One master frame per index in `[frame_index(start_s), frame_index(end_s))`,
    each compared to its ONE candidate counterpart under `shift_frames` --
    a single distance series, not the two-hypothesis pairing
    `locate_bracket_boundary` builds. Returns (frames, distances);
    `distances[i]` is None where either side's hash could not be extracted
    (never fabricated as 0 or as "no match" -- absent, never zero, same
    rule as everywhere else in this module's siblings).'''
    m_base, m_hashes = _extract_hashes(comparer, master_path, start_s - pad_sec,
                                       (end_s - start_s) + 2 * pad_sec)
    c_base, c_hashes = _extract_hashes(comparer, candidate_path,
                                       start_s + shift_frames * comparer.fps_den
                                       / comparer.fps_num - pad_sec,
                                       (end_s - start_s) + 2 * pad_sec)
    m_first = comparer._frame_index(start_s)
    m_last = comparer._frame_index(end_s)
    frames, distances = [], []
    for m in range(m_first, m_last):
        mi = m - m_base
        ci = (m + shift_frames) - c_base
        mh = m_hashes[mi] if 0 <= mi < len(m_hashes) else None
        ch = c_hashes[ci] if 0 <= ci < len(c_hashes) else None
        distances.append(None if (mh is None or ch is None)
                         else FrameComparer._popcount64(mh ^ ch))
        frames.append(m)
    return frames, distances


def locate_match_onset(master_path, candidate_path, fps_num, fps_den,
                       bracket_low_ms, bracket_high_ms, offset_ms,
                       known_match_ms, known_absent_ms, edge,
                       pad_sec=3.0, debug=False):
    '''H-A3/H-TIER head/tail CONSTRUCTED-GAP primitive, additive
    (2026-09-16, Architect's ruling). `locate_bracket_boundary` is a
    TWO-HYPOTHESIS transition finder -- `_monotone_split` needs
    `offset_before_ms != offset_after_ms` to produce any signal at all.
    Head/tail's constructed-gap case has only ONE real offset (there is no
    "before" content to hypothesize about, by definition), and calling
    that function with `before==after` makes every frame pair identical
    (`db==da`) -- its own arithmetic then reports the FULL bracket back as
    a "clean" `declined: False` result, a false-positive success that
    narrows nothing while looking narrowed. Adapting that function to also
    answer this question would overload it with two contracts -- the same
    shape as `bracket_is_bound_only` collapsing two states into one bit,
    one level up. This is a SEPARATE function because the question is
    separate: "where does matching begin (edge="head") or end
    (edge="tail"), under the ONE offset that IS known?"

    RESOLUTION CONTROL FIRST, dual baseline -- forensic's own pattern
    (fire a KNOWN shift, read it back, THEN trust the real measurement),
    made structural here rather than left to the caller's discipline. Two
    reference points the CALLER already has strong reason to believe in --
    `known_match_ms` (comfortably inside the surviving plateau) and
    `known_absent_ms` (the far, unmeasured side of the bracket) -- are
    measured FIRST, under this SAME offset and SAME instrument. If their
    Hamming distances do not separate on THIS content, the instrument
    cannot see the difference here, and NOTHING it reports inside the
    bracket can be trusted -- decline `could_not_locate_onset` before
    scanning at all. Low-motion or already-similar content declines here
    honestly, matching the same shape id 12's bracket 2 already produces
    one layer up (`structure_present_could_not_narrow`) -- that is this
    design working, not failing.

    CONTRACT, enforced by the return shape, not by caller discipline:
    `declined: False` may ONLY accompany an interval STRICTLY NARROWER
    than `[bracket_low_ms, bracket_high_ms]`. The exact failure mode this
    function exists to prevent -- reporting the input bracket back as
    "located" -- is unrepresentable by this contract, not merely against
    the rules: every return path below either declines, or returns an
    interval provably inside the input bracket.

    Tokens: `located` (declined=False) / `could_not_locate_onset`, plus
    the grid/bracket declines `locate_bracket_boundary` already uses
    (`grid_unmeasured`, `empty_bracket`) -- SAME vocabulary, no new
    sibling invented for those two.
    '''
    fps_num = int(fps_num)
    fps_den = int(fps_den)
    if fps_num <= 0 or fps_den <= 0:
        return {"declined": True, "reason": "grid_unmeasured",
               "evidence": f"fps_num={fps_num} fps_den={fps_den}"}

    m_start_s = bracket_low_ms / 1000.0
    m_end_s = bracket_high_ms / 1000.0
    if m_end_s <= m_start_s:
        return {"declined": True, "reason": "empty_bracket",
               "evidence": f"[{bracket_low_ms},{bracket_high_ms}] ms"}
    if edge not in ("head", "tail"):
        raise ValueError(f"edge must be 'head' or 'tail', got {edge!r}")

    comparer = FrameComparer(master_path, candidate_path, m_start_s, m_end_s,
                             fps_num, fps_den, debug=debug)
    shift = _nominal_shift_frames(offset_ms, fps_num, fps_den)
    probe_frames, threshold = _boundary_validation_config()
    frame_s = fps_den / fps_num
    baseline_span_s = max(1.0, probe_frames * frame_s)

    def baseline_at(anchor_ms):
        anchor_s = anchor_ms / 1000.0
        _, distances = _hamming_series(comparer, master_path, candidate_path,
                                       max(0.0, anchor_s - baseline_span_s / 2.0),
                                       anchor_s + baseline_span_s / 2.0,
                                       shift, pad_sec)
        valid = [d for d in distances if d is not None]
        return (sum(valid) / len(valid)) if valid else None

    match_baseline = baseline_at(known_match_ms)
    if match_baseline is None:
        return {"declined": True, "reason": "could_not_locate_onset",
               "evidence": f"match baseline frames unextractable at "
                          f"known_match_ms={known_match_ms}"}
    absent_baseline = baseline_at(known_absent_ms)
    if absent_baseline is None:
        # A None here, UNLIKE on the match side just above, is not
        # automatically an instrument failure: the match-side extraction
        # already proved this instrument reads THESE two files fine, so a
        # failure only on the absent anchor is overwhelmingly a
        # CANDIDATE-SIDE SEEK the offset places outside the candidate's
        # own valid range (negative for `edge="head"`, past its end for
        # `edge="tail"`) -- which is not a measurement failure, it IS the
        # confirmation of absence this anchor exists to provide. Maximal
        # Hamming distance (64: total mismatch on a 64-bit hash), not a
        # fabricated "close" value -- absent, never zero, extended here to
        # "absent, never merely unextractable" for this one anchor.
        absent_baseline = 64.0
    # SEPARATION, NOT JUST A THRESHOLD CROSSING. A baseline pair that both
    # read "matching" or both read "absent" gives this instrument no way
    # to tell the two apart on THIS content -- exactly the id-12
    # bracket-2 shape, made explicit here instead of discovered downstream.
    if not (match_baseline <= threshold and absent_baseline - match_baseline >= threshold):
        return {"declined": True, "reason": "could_not_locate_onset",
               "evidence": f"baselines do not separate: match={match_baseline:.2f} "
                          f"absent={absent_baseline:.2f} threshold={threshold}"}

    frames, distances = _hamming_series(comparer, master_path, candidate_path,
                                        m_start_s, m_end_s, shift, pad_sec)
    valid_frames = [(f, d) for f, d in zip(frames, distances) if d is not None]
    if not valid_frames:
        return {"declined": True, "reason": "could_not_locate_onset",
               "evidence": f"no frame pair readable in "
                          f"[{bracket_low_ms},{bracket_high_ms}] ms"}

    split_threshold = (match_baseline + absent_baseline) / 2.0
    matches = [d <= split_threshold for _, d in valid_frames]
    onset_frame = None
    if edge == "head":
        # First frame where matching STARTS, and holds for the next frame
        # too -- the same two-consecutive discipline `_bracket_transition`
        # uses for its own onset, carried here for the same reason: one
        # matching frame can be a coincidental low-distance outlier.
        for i in range(len(matches) - 1):
            if matches[i] and matches[i + 1]:
                onset_frame = valid_frames[i][0]
                break
    else:
        for i in range(len(matches) - 1, 0, -1):
            if matches[i] and matches[i - 1]:
                onset_frame = valid_frames[i][0]
                break

    if onset_frame is None:
        return {"declined": True, "reason": "could_not_locate_onset",
               "evidence": f"baselines separated (match={match_baseline:.2f} "
                          f"absent={absent_baseline:.2f}) but no two-consecutive "
                          f"{'match' if edge=='head' else 'match-before-divergence'} "
                          f"found in the bracket"}

    onset_ms = round(onset_frame * frame_s * 1000.0, 2)
    if edge == "head":
        start_ms, end_ms = bracket_low_ms, onset_ms
    else:
        start_ms, end_ms = onset_ms, bracket_high_ms
    # THE CONTRACT: refuse to return anything that is not STRICTLY
    # narrower than the input -- the caller may trust `declined: False`
    # without re-checking width itself.
    if not (bracket_low_ms <= start_ms < end_ms <= bracket_high_ms
            and (end_ms - start_ms) < (bracket_high_ms - bracket_low_ms)):
        return {"declined": True, "reason": "could_not_locate_onset",
               "evidence": f"located interval [{start_ms},{end_ms}] does not "
                          f"strictly narrow the input bracket -- refusing to "
                          f"report a non-narrowing result as located"}
    return {"declined": False, "reason": "located",
           "grid": {"num": fps_num, "den": fps_den},
           "edge": edge, "onset_frame": onset_frame,
           "candidate_offset_frames": shift,
           # STRINGS, NOT NUMBERS -- same ruling as `_f1_payload` above;
           # the exact contract is `onset_frame` + `grid`, this is a
           # rounded display projection and must not be arithmetic-shaped.
           "derived_ms": {"master_start_ms": f"{start_ms}", "master_end_ms": f"{end_ms}"},
           "match_baseline": round(match_baseline, 2),
           "absent_baseline": round(absent_baseline, 2),
           "method": "single_hypothesis_onset",
           "evidence": f"onset frame {onset_frame} ({onset_ms}ms); baselines "
                      f"match={match_baseline:.2f} absent={absent_baseline:.2f}"}
