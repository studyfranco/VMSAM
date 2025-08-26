# -*- coding: utf-8 -*-
"""
get_cut_time.py — détection multi-plages, offset global, dérive, raffinement (phash/scene/audio),
construction de chimères audio sécurisée. Compatible avec le style du projet.
"""

import os
import math
import shutil
import tempfile
import traceback
from threading import Thread
from statistics import mean
from sys import stderr

import tools
import video
from audioCorrelation import second_correlation  # offset audio précis
from mergeVideo import get_delay_fidelity       # (fidélité, offset, …) par segments
from frame_compare import FrameComparer         # phash DCT + repli scène

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

class GetCutTime(Thread):
    """
    Détecte les zones de désynchronisation, l’offset global, la dérive de tempo,
    raffine les limites par phash/scene/audio, et construit des chimères audio.
    """

    # Seuils/constantes
    FID_MIN_GOOD = 0.90
    FID_MIN_ACCEPT = 0.85
    DRIFT_SLOPE_ABS_MIN = 0.005
    OFFSET_VAR_MAX = 0.12
    LOCAL_WIN_MIN = 10.0
    LOCAL_WIN_MAX = 30.0
    GLOBAL_OFFSET_PROBE = 45.0
    DRIFT_SAMPLES = 5

    def __init__(self,
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
                 debug=False):
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
            "cuts": [],
            "audio_chimeras": [],
            "subtitle_files": [],
            "mkv_path": None
        }

    def run(self):
        try:
            # 1) Offset global robuste en début (audio)
            global_off = self._detect_global_offset(self.GLOBAL_OFFSET_PROBE)
            self.result["global_offset"] = global_off

            # 2) Fidélité/offsets par segments (toutes paires audio compatibles)
            df = get_delay_fidelity(self.main, self.target, self.lenghtTime)
            if not df:
                return self._fail("Aucune mesure de fidélité disponible")

            per_segment = self._summarize_segments(df)
            # 3) Détecter les plages “mauvaises”
            bad_ranges = self._find_all_bad_ranges(per_segment)
            if not bad_ranges:
                # compatible sans coupe
                subs = self._extract_subtitles_from_target()
                self.result["subtitle_files"] = subs
                self.result["mkv_path"] = self._build_final_mkv([], subs)
                return

            # 4) Conflits d’offsets → raffinement local (audio)
            if self._every_segment_has_conflicting_offsets(per_segment):
                if not self._refine_conflicting_offsets(bad_ranges):
                    return self._fail("Offsets contradictoires persistants; contenus probablement différents")

            # 5) Dérive temporelle (régression)
            drift = self._detect_time_drift(self.DRIFT_SAMPLES)
            tempo_factor = 1.0
            if drift and abs(drift["slope"]) > self.DRIFT_SLOPE_ABS_MIN and drift["coherent"]:
                tempo_factor = 1.0 / (1.0 + drift["slope"])
            self.result["tempo_factor"] = tempo_factor

            # 6) Raffinement de chaque plage (cadres si FPS identiques sinon scene/audio)
            refined = []
            for rng in bad_ranges:
                cr = rng.copy()
                cr["start_sec"] += global_off
                cr["end_sec"] += global_off
                precise = self._refine_cut(cr)
                if precise is None:
                    return self._fail("Impossible de raffiner une des plages")
                refined.append(precise)

            # 7) Vérification globale hors coupures
            if not self._passes_global_compatibility(per_segment):
                return self._fail("Fidélité insuffisante hors coupures")

            # 8) Chimères audio
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

    def _detect_global_offset(self, probe_seconds: float) -> float:
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
            # Convention projet: res[1] ~ “unité/échantillon” → approx. secondes via FPS
            fps = self.target.get_fps() or self.main.get_fps() or 25.0
            try:
                off_sec = float(res[1]) / float(fps)
            except Exception:
                off_sec = 0.0
            return off_sec
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _detect_time_drift(self, sample_count: int):
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

            # Régression linéaire correcte
            n = len(points)
            sx = sum(p for p in points)
            sy = sum(p[1] for p in points)
            sxx = sum(p * p for p in points)
            sxy = sum(p * p[1] for p in points)
            denom = (n * sxx - sx * sx)
            slope = 0.0 if abs(denom) < 1e-12 else (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n if n > 0 else 0.0

            # Variance résiduelle
            residuals = [p[1] - (slope * p + intercept) for p in points]
            var_res = mean([r * r for r in residuals]) if residuals else 0.0
            coherent = var_res < self.OFFSET_VAR_MAX
            return {"slope": slope, "coherent": coherent, "points": points}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _summarize_segments(self, delay_fidelity_values):
        """
        Agrège par segment: moyenne des fidélités et ensemble des offsets (toutes paires).
        delay_fidelity_values: dict clé piste -> liste par segment de tuples (fid, …, offset_ms)
        """
        keys = list(delay_fidelity_values.keys())
        if not keys:
            return []
        segs = len(delay_fidelity_values[keys])
        out = []
        for i in range(segs):
            fids = []
            offs = set()
            for k in keys:
                v = delay_fidelity_values[k][i]
                # v ~ fidélité, v[2] ~ offset (ms)
                try:
                    fids.append(float(v))
                except Exception:
                    pass
                try:
                    offs.add(int(v[2]))
                except Exception:
                    pass
            out.append({"index": i, "fid": (mean(fids) if fids else 0.0), "offsets": offs})
        return out

    def _find_all_bad_ranges(self, per_segment):
        """
        Plages contiguës où la fidélité chute ou offsets divergents.
        Retourne des dicts avec start_sec/end_sec/estimated_type.
        """
        if not per_segment:
            return []
        vals = [p["fid"] for p in per_segment]
        mu = mean(vals) if vals else 1.0
        # seuil adaptatif
        thr = max(0.85, min(0.95, mu - 0.75 * (max(vals) - min(vals) if len(vals) > 1 else 0.0)))

        bad = []
        cur_start = None
        for p in per_segment:
            is_bad = (p["fid"] < thr) or (len(p["offsets"]) > 1)
            if is_bad and cur_start is None:
                cur_start = p["index"]
            if (not is_bad) and (cur_start is not None):
                bad.append((cur_start, p["index"] - 1))
                cur_start = None
        if cur_start is not None:
            bad.append((cur_start, per_segment[-1]["index"]))

        ranges = []
        for (a, b) in bad:
            seg0 = max(0, a - 1)
            seg1 = min(len(self.list_cut_begin_length) - 1, b + 1)
            t0 = self.list_cut_begin_length[seg0]
            t1 = self.list_cut_begin_length[seg1]
            start_sec = _hhmmss_to_seconds(t0)
            end_sec = _hhmmss_to_seconds(t1)
            ranges.append({"start_sec": start_sec, "end_sec": end_sec, "estimated_type": "delete"})
        return ranges

    def _every_segment_has_conflicting_offsets(self, per_segment):
        return len(per_segment) > 0 and all(len(p["offsets"]) > 1 for p in per_segment)

    def _refine_conflicting_offsets(self, ranges):
        """
        Pour chaque plage, on échantillonne 3 fenêtres locales et on exige une variance faible des offsets audio.
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
                    except Exception:
                        off = 0.0
                    offsets.append(off)
                if len(offsets) >= 2:
                    m = mean(offsets)
                    var = mean([(o - m) ** 2 for o in offsets])
                    if var > self.OFFSET_VAR_MAX:
                        return False
            return True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _refine_cut(self, coarse):
        """
        Raffine une plage:
        - FPS identiques → FrameComparer (phash DCT + bande, repli scène si besoin)
        - Sinon → repli scène immédiatement, sinon audio locale second_correlation
        """
        fps_main = self.main.get_fps()
        fps_tgt = self.target.get_fps()
        fps_use = fps_tgt if fps_tgt is not None else (fps_main if fps_main is not None else 25.0)
        length = max(0.5, coarse["end_sec"] - coarse["start_sec"])
        band = max(1, int(math.ceil(length * fps_use * 0.20)))
        max_search = max(8, int(math.ceil(length * fps_use * 0.30)))

        # Si FPS identiques, tenter les cadres
        if fps_main is not None and fps_tgt is not None and abs(fps_main - fps_tgt) < 1e-6:
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
                if self.debug:
                    stderr.write(f"[refine_cut] phash error: {e}\n")

        # Repli: scène ffmpeg dans la fenêtre
        scene = self._scene_refine_window(coarse["start_sec"], coarse["end_sec"])
        if scene is not None:
            st, et = scene
            return {
                "type": coarse.get("estimated_type", "delete"),
                "start_frame": int(round(st * fps_use)),
                "end_frame": int(round(et * fps_use)),
                "start_time": st,
                "end_time": et,
                "fps": fps_use,
                "method": "ffmpeg_scene"
            }

        # Repli ultime: audio locale
        return self._refine_cut_audio(coarse, fps_use)

    def _scene_refine_window(self, start_sec, end_sec):
        ffmpeg = tools.software["ffmpeg"]
        dur = max(0.5, end_sec - start_sec)
        cmd = [
            ffmpeg, "-hide_banner", "-nostdin",
            "-ss", f"{start_sec}",
            "-t", f"{dur}",
            "-i", self.target.filePath,
            "-vf", "select='gt(scene,0.30)',showinfo",
            "-f", "null", "-"
        ]
        out, err, rc = tools.launch_cmdExt_no_test(cmd)
        import re
        text = err.decode("utf-8", errors="ignore")
        times = [float(m.group(1)) + start_sec for m in re.finditer(r"pts_time:([0-9]+\.[0-9]+)", text)]
        if not times:
            return None
        c = times[len(times) // 2]
        band = max(0.2, min(2.0, dur * 0.2))
        return (max(start_sec, c - band), min(end_sec, c + band))

    def _refine_cut_audio(self, coarse, fps_use):
        ffmpeg = tools.software["ffmpeg"]
        tmp = tempfile.mkdtemp(prefix="cut_refine_")
        try:
            length = max(0.5, coarse["end_sec"] - coarse["start_sec"])
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
            shutil.rmtree(tmp, ignore_errors=True)

    def _passes_global_compatibility(self, per_segment):
        if not per_segment:
            return False
        vals = [p["fid"] for p in per_segment]
        if not vals:
            return False
        return mean(vals) >= self.FID_MIN_ACCEPT

    def _build_chimeric_audios(self, precise_cut, tempo_factor=1.0):
        s_time = precise_cut["start_time"]
        e_time = precise_cut["end_time"]
        tmp_dir = tempfile.mkdtemp(prefix="chimeras_")
        outputs = []
        try:
            ffmpeg = tools.software["ffmpeg"]
            # sélectionner les pistes cibles
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

                before = os.path.join(tmp_dir, f"b_{aud['StreamOrder']}.wav")
                after = os.path.join(tmp_dir, f"a_{aud['StreamOrder']}.wav")

                slop = 0.05
                if (s_time - slop) > 0.005:
                    # “before” inclut un léger chevauchement de sécurité
                    b_ss = max(0.0, s_time - 0.5)
                    b_t = max(0.0, (s_time - b_ss) + slop)
                    cmd_b = [ffmpeg, "-y", "-ss", f"{b_ss}", "-t", f"{b_t}",
                             "-i", self.target.filePath, "-map", f"0:{aud['StreamOrder']}",
                             "-acodec", "pcm_s16le", "-ac", ch, "-ar", sr, before]
                    tools.launch_cmdExt(cmd_b)

                # “after” commence juste après la fenêtre coupée
                cmd_a = [ffmpeg, "-y", "-ss", f"{max(0.0, e_time + 0.01)}", "-t", "5.0",
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

                # atempo si applicable
                enc_input = concat_wav
                enc_args = []
                if abs(tempo_factor - 1.0) > 1e-3:
                    flt = _build_atempo_filters(tempo_factor)
                    enc_args = ["-filter:a", ",".join(flt)]

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
            if not self.debug:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _extract_subtitles_from_target(self):
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

    def _build_final_mkv(self, audio_outputs, subtitle_files):
        mkvmerge = tools.software.get("mkvmerge", "mkvmerge")
        args = [mkvmerge, "-o", self.output_path]
        for s in subtitle_files:
            args.append(s["file"])
        for ao in audio_outputs:
            lang = ao.get("language", "und")
            args += ["--language", f"0:{lang}", ao["file"]]
        tools.launch_cmdExt(args)
        return self.output_path

    def _fail(self, reason: str):
        self.result["incompatible"] = True
        self.result["reason"] = reason
        stderr.write(f"[get_cut_time] {reason}\n")