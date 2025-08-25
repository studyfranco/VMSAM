# -*- coding: utf-8 -*-
"""
get_cut_time.py — détection multi-plages, offset global, dérive (accéléré/ralenti),
raffinement par cadres (phash) ou audio, génération de chimères audio sécurisée.

Dépendances projet: tools, video, audioCorrelation.second_correlation, mergeVideo.get_delay_fidelity,
frame_compare.FrameComparer, ffmpeg/mkvmerge configurés via tools.software.
"""

import os
import math
import shutil
import tempfile
import traceback
from threading import Thread
from statistics import mean, stdev
from sys import stderr

import tools
import video

# Primitives déjà présentes dans le projet
from audioCorrelation import second_correlation  # mesure fine d’offset entre extraits audio
from mergeVideo import get_delay_fidelity        # fidélité/délais par segments et par paires audio
from frame_compare import FrameComparer          # recherche de rupture vidéo par phash (si FPS identiques)


# ---------- utilitaires locaux ----------

def _hhmmss_to_seconds(ts: str) -> float:
    if not ts:
        return 0.0
    if "." in ts:
        base, ms = ts.split(".", 1)
    else:
        base, ms = ts, "0"
    h, m, s = [int(x) for x in base.split(":")]
    return h * 3600 + m * 60 + s + (0.0 if ms in ("0", "") else float("0." + ms))


def _build_atempo_filters(tempo: float):
    """
    Construit une chaîne de filtres atempo pour couvrir des facteurs hors [0.5, 2.0].
    Retourne une liste de filtres 'atempo=x' à chaîner.
    """
    if tempo <= 0:
        return []
    chain = []
    t = tempo
    while t < 0.5 or t > 2.0:
        step = 2.0 if t > 2.0 else 0.5
        chain.append(f"atempo={step}")
        t /= step
    chain.append(f"atempo={t}")
    return chain


# ---------- classe principale ----------

class GetCutTime(Thread):
    """
    Détecte les zones de désynchronisation, l’offset global de début, la dérive de tempo,
    raffine les bornes de coupe et construit des chimères audio si et seulement si le contenu est compatible.
    """

    # Seuils et constantes
    FID_MIN_GOOD = 0.90           # fidélité jugée “bonne”
    FID_MIN_ACCEPT = 0.85         # fidélité minimale acceptée hors rupture
    DRIFT_SLOPE_ABS_MIN = 0.005   # 0.5% de dérive par seconde (relative sur la durée) déclenche un flag
    OFFSET_VAR_MAX = 0.12         # variance max (~0.12 s^2) des offsets locaux dans une plage pour la considérer stable
    LOCAL_WIN_MIN = 10.0          # taille min des fenêtres locales de corrélation (s)
    LOCAL_WIN_MAX = 30.0          # taille max des fenêtres locales de corrélation (s)
    GLOBAL_OFFSET_PROBE = 45.0    # durée sondée près du début pour offset global (s)
    DRIFT_SAMPLES = 5             # nombre d’échantillons pour la régression de dérive

    def __init__(
        self,
        main_video_obj,
        video_obj_to_cut,
        begin_in_second,
        audioParam,
        language,
        lenghtTime,
        lenghtTimePrepare,
        list_cut_begin_length,
        time_by_test_best_quality_converted,
        output_path,
        debug=False
    ):
        Thread.__init__(self)
        self.main = main_video_obj
        self.target = video_obj_to_cut
        self.begin_in_second = begin_in_second
        self.audioParam = audioParam
        self.language = language
        self.lenghtTime = lenghtTime
        self.lenghtTimePrepare = lenghtTimePrepare
        self.list_cut_begin_length = list_cut_begin_length
        self.time_by_test_best_quality_converted = time_by_test_best_quality_converted
        self.output_path = output_path
        self.debug = debug

        self.result = {
            "incompatible": False,
            "reason": "",
            "global_offset": 0.0,
            "tempo_factor": 1.0,
            "cuts": [],              # liste de dicts: type, start_time, end_time, start_frame, end_frame, method, fps
            "audio_chimeras": [],
            "subtitle_files": [],
            "mkv_path": None
        }

    # ---- pipeline principal ----

    def run(self):
        try:
            # 1) Mesure robuste d’un offset global de début
            global_off = self._detect_global_offset(self.GLOBAL_OFFSET_PROBE)
            self.result["global_offset"] = global_off

            # 2) Fidélité par segments (toutes paires audio compatibles)
            df = get_delay_fidelity(self.main, self.target, self.lenghtTime)
            if not df:
                self._fail("Aucune mesure de fidélité disponible (pistes audio incompatibles ou absentes)")
                return

            per_segment = self._summarize_segments(df)

            # 3) Détection de toutes les plages “mauvaises”
            bad_ranges = self._find_all_bad_ranges(per_segment)
            if not bad_ranges:
                # rien à corriger; considérer compatible sans coupe
                self.result["cuts"] = []
                subs = self._extract_subtitles_from_target()
                self.result["subtitle_files"] = subs
                self.result["mkv_path"] = self._build_final_mkv([], subs)
                return

            # 4) Cas “offsets tous différents” → raffinement de stabilité locale
            if self._every_segment_has_conflicting_offsets(per_segment):
                if not self._refine_conflicting_offsets(bad_ranges):
                    self._fail("Offsets contradictoires persistants par segments; contenus probablement différents")
                    return

            # 5) Détection de dérive (accéléré/ralenti) par régression offset(t)
            drift = self._detect_time_drift(self.DRIFT_SAMPLES)
            tempo_factor = 1.0
            if drift and abs(drift["slope"]) > self.DRIFT_SLOPE_ABS_MIN and drift["coherent"]:
                tempo_factor = 1.0 / (1.0 + drift["slope"])  # correction optionnelle si faible et régulière
            self.result["tempo_factor"] = tempo_factor

            # 6) Raffinement de chaque plage (cadres si FPS identiques, sinon audio)
            refined = []
            for rng in bad_ranges:
                cr = rng.copy()
                cr["start_sec"] += global_off
                cr["end_sec"] += global_off
                precise = self._refine_cut(cr)
                if precise is None:
                    self._fail("Impossible de raffiner une des plages de désynchronisation")
                    return
                refined.append(precise)

            # 7) Vérifications globales de compatibilité (fidélité hors coupures)
            if not self._passes_global_compatibility(per_segment):
                self._fail("Fidélité insuffisante hors coupures; contenus probablement différents")
                return

            # 8) Construction des chimères audio, avec éventuelle correction de tempo
            all_out = []
            for cut in refined:
                outs = self._build_chimeric_audios(cut, tempo_factor=tempo_factor)
                all_out.extend(outs)

            subs = self._extract_subtitles_from_target()
            mkv = self._build_final_mkv(all_out, subs)

            self.result["cuts"] = refined
            self.result["audio_chimeras"] = all_out
            self.result["subtitle_files"] = subs
            self.result["mkv_path"] = mkv

        except Exception as e:
            traceback.print_exc()
            self._fail(f"Erreur d’exécution: {e}")

    # ---- détection et synthèse ----

    def _detect_global_offset(self, probe_seconds: float) -> float:
        """
        Corrèle deux extraits près du début pour obtenir un offset initial robuste (en secondes).
        """
        ffmpeg = tools.software["ffmpeg"]
        tmp = tempfile.mkdtemp(prefix="global_off_")
        try:
            dur = float(max(10.0, min(probe_seconds, 90.0)))
            m_wav = os.path.join(tmp, "m.wav")
            t_wav = os.path.join(tmp, "t.wav")
            cmd_m = [ffmpeg, "-y", "-ss", "0", "-t", f"{dur}", "-i", self.main.filePath,
                     "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", m_wav]
            cmd_t = [ffmpeg, "-y", "-ss", "0", "-t", f"{dur}", "-i", self.target.filePath,
                     "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", t_wav]
            tools.launch_cmdExt(cmd_m)
            tools.launch_cmdExt(cmd_t)
            res = second_correlation(m_wav, t_wav)
            fps = self.target.get_fps() or self.main.get_fps() or 25.0
            try:
                # convention projet: res[1] ~ décalage en “unités d’échantillon/frame”
                off_sec = float(res[1]) / float(fps)
            except Exception:
                off_sec = 0.0
            return off_sec
        finally:
            if not self.debug:
                shutil.rmtree(tmp, ignore_errors=True)

    def _detect_time_drift(self, sample_count: int):
        """
        Échantillonne des fenêtres réparties et ajuste offset(t)=a·t+b; renvoie dict(slope, coherent, points).
        """
        dur = self.main.get_video_duration() or 0.0
        if dur <= 0.0:
            return {"slope": 0.0, "coherent": False, "points": []}
        ffmpeg = tools.software["ffmpeg"]
        tmp = tempfile.mkdtemp(prefix="drift_")
        points = []
        try:
            n = max(3, int(sample_count))
            win = max(self.LOCAL_WIN_MIN, min(self.LOCAL_WIN_MAX, dur / 10.0))
            for k in range(n):
                center = (k + 1) * dur / (n + 1)
                start = max(0.0, center - win / 2.0)
                m_wav = os.path.join(tmp, f"m_{k}.wav")
                t_wav = os.path.join(tmp, f"t_{k}.wav")
                cmd_m = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{win}", "-i", self.main.filePath,
                         "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", m_wav]
                cmd_t = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{win}", "-i", self.target.filePath,
                         "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", t_wav]
                tools.launch_cmdExt(cmd_m)
                tools.launch_cmdExt(cmd_t)
                res = second_correlation(m_wav, t_wav)
                fps = self.target.get_fps() or self.main.get_fps() or 25.0
                try:
                    off = float(res[1]) / float(fps)
                except Exception:
                    off = 0.0
                points.append((center, off))

            # régression linéaire simple
            n = len(points)
            sx = sum(p for p in points)
            sy = sum(p[1] for p in points)
            sxx = sum(p * p for p in points)
            sxy = sum(p * p[1] for p in points)
            denom = (n * sxx - sx * sx)
            slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom

            # cohérence: variance résiduelle faible par rapport au max(|offset|)
            residuals = []
            if n >= 2:
                intercept = (sy - slope * sx) / n
                residuals = [p[1] - (slope * p + intercept) for p in points]
            var_res = mean([r * r for r in residuals]) if residuals else 0.0
            coherent = var_res < self.OFFSET_VAR_MAX

            return {"slope": slope, "coherent": coherent, "points": points}
        finally:
            if not self.debug:
                shutil.rmtree(tmp, ignore_errors=True)

    def _summarize_segments(self, delay_fidelity_values):
        """
        Agrège par segment: moyenne des fidélités et ensemble des offsets observés (toutes paires).
        """
        keys = list(delay_fidelity_values.keys())
        if not keys:
            return []
        segs = len(delay_fidelity_values[keys])
        out = []
        for i in range(segs):
            fids = []
            offs = set()
            for key in keys:
                v = delay_fidelity_values[key][i]
                fids.append(v)
                offs.add(v[2])
            out.append({"index": i, "fid": mean(fids) if fids else 0.0, "offsets": offs})
        return out

    def _find_all_bad_ranges(self, per_segment):
        """
        Détecte toutes les plages contiguës où la fidélité chute ou les offsets divergent.
        """
        if not per_segment:
            return []
        vals = [p["fid"] for p in per_segment]
        mu = mean(vals) if vals else 1.0
        try:
            sigma = stdev(vals) if len(vals) >= 2 else 0.0
        except Exception:
            sigma = 0.0
        thr = max(0.85, min(0.95, mu - 0.75 * sigma))

        bad = []
        cur = None
        for p in per_segment:
            is_bad = (p["fid"] < thr) or (len(p["offsets"]) > 1)
            if is_bad:
                cur = [p["index"], p["index"]] if cur is None else [cur, p["index"]]
            else:
                if cur is not None:
                    bad.append(tuple(cur))
                    cur = None
        if cur is not None:
            bad.append(tuple(cur))

        # Conversion vers temps, avec marge d’un segment
        ranges = []
        for (a, b) in bad:
            seg0 = max(0, a - 1)
            seg1 = min(len(self.list_cut_begin_length) - 1, b + 1)
            t0 = self.list_cut_begin_length[seg0]
            t1 = self.list_cut_begin_length[seg1][1]
            start_sec = _hhmmss_to_seconds(t0)
            end_sec = _hhmmss_to_seconds(t1)
            # Heuristique sur le “type”
            typ = "delete"
            # si offsets majoritairement positifs dans la plage, considérer “add” côté cible
            # (la décision exacte sera affinée plus tard)
            ranges.append({"start_sec": start_sec, "end_sec": end_sec, "estimated_type": typ})
        return ranges

    def _every_segment_has_conflicting_offsets(self, per_segment):
        """
        Vrai si chaque segment présente plusieurs offsets simultanés → cas pathologique à raffiner.
        """
        if not per_segment:
            return False
        return all(len(p["offsets"]) > 1 for p in per_segment)

    def _refine_conflicting_offsets(self, ranges):
        """
        Pour chaque plage, mesurer offsets locaux sur sous-fenêtres; exiger stabilité (variance faible)
        et corrélation suffisante; sinon abandonner.
        """
        ffmpeg = tools.software["ffmpeg"]
        fps = self.target.get_fps() or self.main.get_fps() or 25.0
        tmp = tempfile.mkdtemp(prefix="conflict_")
        try:
            for r in ranges:
                length = max(0.5, r["end_sec"] - r["start_sec"])
                win = max(self.LOCAL_WIN_MIN, min(self.LOCAL_WIN_MAX, length * 0.6))
                centers = [r["start_sec"] + (k + 1) * length / 4.0 for k in range(3)]
                offsets = []
                scores = []
                for c in centers:
                    start = max(0.0, c - win / 2.0)
                    m_wav = os.path.join(tmp, f"m_{int(c*1000)}.wav")
                    t_wav = os.path.join(tmp, f"t_{int(c*1000)}.wav")
                    cmd_m = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{win}", "-i", self.main.filePath,
                             "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", m_wav]
                    cmd_t = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{win}", "-i", self.target.filePath,
                             "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", t_wav]
                    tools.launch_cmdExt(cmd_m)
                    tools.launch_cmdExt(cmd_t)
                    res = second_correlation(m_wav, t_wav)
                    try:
                        off = float(res[1]) / float(fps)
                        sc = float(res) if isinstance(res, (list, tuple)) else 0.0
                    except Exception:
                        off, sc = 0.0, 0.0
                    offsets.append(off)
                    scores.append(sc)

                var = mean([(o - mean(offsets)) ** 2 for o in offsets]) if offsets else 0.0
                if var > self.OFFSET_VAR_MAX:
                    return False
                # exiger des scores décents (pas nécessairement parfaits)
                if scores and mean(scores) < 0.5:
                    return False

            return True
        finally:
            if not self.debug:
                shutil.rmtree(tmp, ignore_errors=True)

    def _refine_cut(self, coarse):
        """
        Raffine une plage: si FPS identiques → phash cadres, sinon corrélation audio locale.
        """
        fps_main = self.main.get_fps()
        fps_tgt = self.target.get_fps()
        fps_use = fps_tgt if fps_tgt is not None else (fps_main if fps_main is not None else 25.0)

        length = max(0.5, coarse["end_sec"] - coarse["start_sec"])
        band = max(1, int(math.ceil(length * fps_use * 0.20)))
        max_search = max(8, int(math.ceil(length * fps_use * 0.30)))

        if fps_main is not None and fps_tgt is not None and abs(fps_main - fps_tgt) < 1e-6:
            # phash cadres
            try:
                fc = FrameComparer(
                    self.main.filePath,
                    self.target.filePath,
                    coarse["start_sec"],
                    coarse["end_sec"],
                    fps=int(round(fps_use)),
                    band_width=band,
                    max_search_frames=max_search,
                    debug=self.debug
                )
                fr = fc.find_scene_gap_requirements(before_common=2, after_common=3)
                if fr is not None:
                    return {
                        "type": coarse.get("estimated_type", "delete"),
                        "start_frame": fr["start_frame"],
                        "end_frame": fr["end_frame"],
                        "start_time": fr["start_time"],
                        "end_time": fr["end_time"],
                        "fps": fps_use,
                        "method": "phash_frames"
                    }
            except Exception as e:
                stderr.write(f"[refine_cut] phash error: {e}\n")

        # repli audio
        ffmpeg = tools.software["ffmpeg"]
        tmp = tempfile.mkdtemp(prefix="cut_refine_")
        try:
            pad = min(4.0, max(1.0, length * 0.1))
            start = max(0.0, coarse["start_sec"] - pad)
            dur = length + 2 * pad
            m_wav = os.path.join(tmp, "m.wav")
            t_wav = os.path.join(tmp, "t.wav")
            cmd_m = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", self.main.filePath,
                     "-vn", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", m_wav]
            cmd_t = [ffmpeg, "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", self.target.filePath,
                     "-vn", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", t_wav]
            tools.launch_cmdExt(cmd_m)
            tools.launch_cmdExt(cmd_t)
            res = second_correlation(m_wav, t_wav)
            try:
                off = float(res[1]) / float(fps_use)
            except Exception:
                off = 0.0
            st = coarse["start_sec"] + off
            et = coarse["end_sec"] + off
            return {
                "type": coarse.get("estimated_type", "delete"),
                "start_frame": int(round(st * fps_use)),
                "end_frame": int(round(et * fps_use)),
                "start_time": st,
                "end_time": et,
                "fps": fps_use,
                "method": "audio_second_correlation"
            }
        finally:
            if not self.debug:
                shutil.rmtree(tmp, ignore_errors=True)

    def _passes_global_compatibility(self, per_segment):
        """
        Exige une fidélité raisonnable hors coupures pour accepter la fusion.
        """
        if not per_segment:
            return False
        vals = [p["fid"] for p in per_segment]
        if not vals:
            return False
        return mean(vals) >= self.FID_MIN_ACCEPT

    # ---- génération MKV (audio chimères + sous-titres) ----

    def _build_chimeric_audios(self, precise_cut, tempo_factor=1.0):
        """
        Construit des pistes audio chimériques pour la cible autour de la coupe précise.
        """
        s_time = precise_cut["start_time"]
        e_time = precise_cut["end_time"]
        tmp_dir = tempfile.mkdtemp(prefix="chimeras_")
        outputs = []

        try:
            ffmpeg = tools.software["ffmpeg"]

            # constituer la liste des pistes à traiter (par langue, sinon toutes)
            target_audios = []
            if self.language in self.target.audios:
                target_audios = self.target.audios[self.language]
            else:
                for _, auds in self.target.audios.items():
                    target_audios.extend(auds)

            for aud in target_audios:
                ch = str(aud.get("Channels", 2))
                sr = str(aud.get("SamplingRate", 44100))
                codec = (aud.get("codec") or aud.get("Format") or "aac").lower()

                # extraire avant et après la zone (concat échantillons PCM)
                before = os.path.join(tmp_dir, f"b_{aud['StreamOrder']}.wav")
                after = os.path.join(tmp_dir, f"a_{aud['StreamOrder']}.wav")

                # marges faibles pour éviter les ruptures au sample près
                slop = 0.05
                if (s_time - slop) > 0.01:
                    cmd_b = [ffmpeg, "-y", "-ss", f"{max(0.0, s_time - 0.5)}", "-t", f"{(s_time - max(0.0, s_time - 0.5)) + slop}",
                             "-i", self.target.filePath, "-map", f"0:{aud['StreamOrder']}",
                             "-acodec", "pcm_s16le", "-ac", ch, "-ar", sr, before]
                    tools.launch_cmdExt(cmd_b)

                cmd_a = [ffmpeg, "-y", "-ss", f"{e_time + 0.01}", "-t", "5.0",
                         "-i", self.target.filePath, "-map", f"0:{aud['StreamOrder']}",
                         "-acodec", "pcm_s16le", "-ac", ch, "-ar", sr, after]
                tools.launch_cmdExt(cmd_a)

                # concat
                concat_list = os.path.join(tmp_dir, f"list_{aud['StreamOrder']}.txt")
                with open(concat_list, "w") as fh:
                    if os.path.exists(before):
                        fh.write(f"file '{before}'\n")
                    fh.write(f"file '{after}'\n")

                concat_wav = os.path.join(tmp_dir, f"c_{aud['StreamOrder']}.wav")
                cmd_concat = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", concat_wav]
                tools.launch_cmdExt(cmd_concat)

                # correction de tempo si applicable
                enc_input = concat_wav
                enc_args = []
                if abs(tempo_factor - 1.0) > 1e-3:
                    flt = _build_atempo_filters(tempo_factor)
                    enc_args = ["-filter:a", ",".join(flt)]

                # encodage final (préserver canaux et échantillonnage)
                out_path = os.path.join(tmp_dir, f"chimera_{aud['StreamOrder']}.mka")
                cmd_enc = [ffmpeg, "-y", "-i", enc_input] + enc_args + ["-ac", ch, "-ar", sr]
                if codec in ("aac", "aac_latm"):
                    cmd_enc += ["-c:a", "aac"]
                elif codec == "ac3":
                    cmd_enc += ["-c:a", "ac3"]
                elif codec in ("dts",):
                    cmd_enc += ["-c:a", "dts"]
                elif codec in ("mp3",):
                    cmd_enc += ["-c:a", "libmp3lame"]
                else:
                    cmd_enc += ["-c:a", "flac"]
                cmd_enc.append(out_path)
                tools.launch_cmdExt(cmd_enc)

                outputs.append({
                    "track_index": aud.get("StreamOrder"),
                    "file": out_path,
                    "language": aud.get("Language", self.language),
                    "codec": codec,
                    "original": aud
                })

            return outputs
        finally:
            # on conserve le dossier si debug
            if not self.debug:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _extract_subtitles_from_target(self):
        """
        Extrait tous les sous-titres de la cible dans un MKV “subs only”.
        """
        mkvmerge = tools.software.get("mkvmerge", "mkvmerge")
        tmp_dir = tempfile.mkdtemp(prefix="subs_")
        try:
            sub_only = os.path.join(tmp_dir, "subs_only.mkv")
            cmd = [mkvmerge, "-o", sub_only, "--no-audio", "--no-video", self.target.filePath]
            tools.launch_cmdExt(cmd)
            return [{"stream_index": None, "file": sub_only, "codec": "copy", "language": "und"}]
        except Exception as e:
            stderr.write(f"[subs] {e}\n")
            return []
        finally:
            # le fichier intermédiaire sera consommé par mkvmerge final; ne pas supprimer ici
            pass

    def _build_final_mkv(self, audio_outputs, subtitle_files):
        """
        Construit le MKV final sans vidéo: sous-titres cibles + pistes audio chimériques.
        """
        mkvmerge = tools.software.get("mkvmerge", "mkvmerge")
        args = [mkvmerge, "-o", self.output_path]
        # d’abord le conteneur de sous-titres uniquement (s’il existe)
        for s in subtitle_files:
            args.append(s["file"])
        # puis chaque piste audio
        for ao in audio_outputs:
            lang = ao.get("language", "und")
            args += ["--language", f"0:{lang}", ao["file"]]
        tools.launch_cmdExt(args)
        return self.output_path

    # ---- utilitaire d’échec ----

    def _fail(self, reason: str):
        self.result["incompatible"] = True
        self.result["reason"] = reason
        stderr.write(f"[get_cut_time] {reason}\n")
