'''
Assemblage chimerique: reconstruire les pistes d'un candidat REFUSE sur la
timeline du maitre.

Ce module ne decide rien. Il recoit une partition de la timeline du maitre --
mesuree ailleurs -- et produit un fichier. La convention de signe est fixee une
fois, ici, sous forme d'equation, parce que c'est la seule chose qui ne se
rattrape pas plus loin:

    temps_candidat_ms = temps_maitre_ms + candidate_offset_ms

Donc pour remplir la position `m` du maitre on lit le candidat en
`m + candidate_offset_ms`. Un candidat auquel il MANQUE un generique que le
maitre possede donne un offset NEGATIF.

Les intervalles de la timeline du maitre que le plan ne couvre pas sont des
trous: le maitre a du contenu que le candidat n'a pas. On les remplit depuis le
maitre lui-meme quand il possede la meme langue, sinon par du silence. Les deux
sont honnetes; la troisieme option -- coller la piste d'une autre langue --
ne l'est pas.

Marquage: SPEC_ZONE_A.MD s4. La chaine est posee comme tag de piste Matroska
`VMSAM_FABRICATED` sur le fichier produit ici. Mesure 2026-09-03: ce tag survit
a la premiere passe ffmpeg (`-c copy -map_metadata 0`), a la seconde, au
`mkvmerge --no-global-tags` du split et au `mkvmerge` final.
'''

from decimal import Decimal
from os import path
import sys

import tools
import video

# Codecs audio qu'on sait reencoder sans changer de famille. Ne JAMAIS remonter
# un codec avec perte vers un codec sans perte: `keep_best_audio` compare codec,
# canaux, frequence et debit, et sa regle `fabricated` -- celle qui devait
# empecher une piste construite d'evincer une piste intacte -- est inerte
# (AGENT_NOTES.MD, 2026-09-03). Une chimere encodee en FLAC evincerait donc pour
# de vrai l'AAC d'origine que SPEC_ZONE_A.MD s4 voulait proteger.
audio_encoder_by_codec = {
    "flac": ("flac", "lossless"),
    "pcm_s16le": ("flac", "lossless"),
    "pcm_s24le": ("flac", "lossless"),
    "pcm_s32le": ("flac", "lossless"),
    "truehd": ("flac", "lossless"),
    "mlp": ("flac", "lossless"),
    "aac": ("aac", "lossy"),
    "ac3": ("ac3", "lossy"),
    "eac3": ("eac3", "lossy"),
    "mp3": ("libmp3lame", "lossy"),
    "opus": ("libopus", "lossy"),
    "vorbis": ("libvorbis", "lossy"),
}

# Sous-titres: docs/SUBTITLE_CODECS.MD. La liste blanche de la campagne 1 ne
# couvrait que 6 noms sur ~19 de la classe texte et appelait "bitmap" tout le
# reste -- une etiquette presentee comme un diagnostic. On accepte donc toute la
# classe texte, et on refuse le bitmap PAR SON NOM.
subtitle_text_codecs_to_ass = frozenset(["ass", "ssa"])
subtitle_text_codecs_to_srt = frozenset([
    "subrip", "srt", "text", "mov_text", "webvtt", "ttml", "microdvd", "sami",
    "jacosub", "mpl2", "pjs", "realtext", "stl", "subviewer", "subviewer1",
    "vplayer", "hdmv_text_subtitle",
])
subtitle_bitmap_codecs = frozenset([
    "dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle", "xsub",
])
subtitle_stream_codecs = frozenset([
    "arib_caption", "dvb_teletext", "eia_608", "ivtv_vbi",
])


class chimeric_error(Exception):
    '''Le plan ne peut pas etre execute. Refus explicite, jamais un fallback.'''
    pass


def normalize_segments(segments, master_duration_ms, candidate_duration_ms):
    '''Valide le plan et renvoie la liste des morceaux a coller, dans l'ordre.

    Chaque morceau est un dict: `source` ("candidate" | "master" | "silence"),
    `master_start_ms`, `master_end_ms`, et `source_start_ms` quand il y a une
    source. Les trous du plan deviennent des morceaux "master".

    Refuse plutot que de rattraper: un plan qui sort du candidat est un plan
    faux, et le clamper produirait un fichier plausible et silencieusement
    decale.
    '''
    if not len(segments):
        raise chimeric_error("empty plan: no segment to assemble")

    ordered = sorted(segments, key=lambda s: s["master_start_ms"])
    pieces = []
    cursor = Decimal("0")
    previous_candidate_end = None

    for segment in ordered:
        master_start = Decimal(str(segment["master_start_ms"]))
        master_end = Decimal(str(segment["master_end_ms"]))
        offset = Decimal(str(segment["candidate_offset_ms"]))

        if master_end <= master_start:
            raise chimeric_error(
                f"segment [{master_start},{master_end}) is empty or inverted")
        if master_start < cursor:
            raise chimeric_error(
                f"segment [{master_start},{master_end}) overlaps the previous one")
        if master_end > master_duration_ms:
            raise chimeric_error(
                f"segment ends at {master_end} ms, past the master's "
                f"{master_duration_ms} ms")

        candidate_start = master_start + offset
        candidate_end = master_end + offset
        if candidate_start < 0 or candidate_end > candidate_duration_ms:
            raise chimeric_error(
                f"segment reads the candidate at [{candidate_start},"
                f"{candidate_end}) ms, outside its {candidate_duration_ms} ms")
        # Monotonie cote candidat: exigee par la nature du probleme (les deux
        # timelines avancent), et exigee par l'implementation (le filtre concat
        # tire ses segments dans l'ordre; un retour en arriere obligerait
        # ffmpeg a bufferiser tout un episode en RAM).
        if previous_candidate_end != None and candidate_start < previous_candidate_end:
            raise chimeric_error(
                f"segment reads the candidate backwards at {candidate_start} ms, "
                f"after having read up to {previous_candidate_end} ms")
        previous_candidate_end = candidate_end

        if master_start > cursor:
            pieces.append({"source": "master", "master_start_ms": cursor,
                           "master_end_ms": master_start,
                           "source_start_ms": cursor})
        pieces.append({"source": "candidate", "master_start_ms": master_start,
                       "master_end_ms": master_end,
                       "source_start_ms": candidate_start})
        cursor = master_end

    if cursor < master_duration_ms:
        pieces.append({"source": "master", "master_start_ms": cursor,
                       "master_end_ms": master_duration_ms,
                       "source_start_ms": cursor})
    return pieces


def get_audio_stream_parameters(audio):
    '''Frequence, canaux et layout de la piste, tels que ffmpeg les nommera.'''
    ffprobe_data = audio.get("ffprobe", {})
    sample_rate = ffprobe_data.get("sample_rate") or audio.get("SamplingRate")
    channels = ffprobe_data.get("channels") or audio.get("Channels")
    layout = ffprobe_data.get("channel_layout")
    if sample_rate == None or channels == None:
        raise chimeric_error(
            f"track {audio.get('StreamOrder')} has no sampling rate or channel count")
    if layout == None or layout == "" or "unknown" in str(layout):
        layout = f"{int(channels)}c"
    return str(sample_rate), int(channels), str(layout)


def build_audio_filtergraph(pieces, candidate_stream_order, master_stream_order,
                            sample_rate, layout):
    '''Une seule commande ffmpeg par piste: pas de WAV intermediaire.

    Les morceaux sont tires dans l'ordre par le filtre concat, et chaque source
    est lue une seule fois en avancant -- d'ou la monotonie exigee plus haut.
    `aformat` sur chaque morceau parce que concat refuse des entrees de format
    different, et le remplissage vient du maitre, qui peut ne pas avoir la meme
    frequence.
    '''
    chains = []
    labels = []
    candidate_pieces = [p for p in pieces if p["source"] == "candidate"]
    master_pieces = [p for p in pieces if p["source"] == "master"]

    candidate_split = []
    if len(candidate_pieces) > 1:
        candidate_split = [f"cs{i}" for i in range(len(candidate_pieces))]
        chains.append(f"[0:{candidate_stream_order}]asplit={len(candidate_pieces)}"
                      + "".join(f"[{label}]" for label in candidate_split))
    master_split = []
    if master_stream_order != None and len(master_pieces) > 1:
        master_split = [f"ms{i}" for i in range(len(master_pieces))]
        chains.append(f"[1:{master_stream_order}]asplit={len(master_pieces)}"
                      + "".join(f"[{label}]" for label in master_split))

    candidate_index = 0
    master_index = 0
    for i, piece in enumerate(pieces):
        label = f"p{i}"
        duration = (piece["master_end_ms"] - piece["master_start_ms"]) / Decimal("1000")
        if piece["source"] == "candidate":
            start = piece["source_start_ms"] / Decimal("1000")
            end = start + duration
            if len(candidate_split):
                entry = f"[{candidate_split[candidate_index]}]"
            else:
                entry = f"[0:{candidate_stream_order}]"
            candidate_index += 1
            chains.append(f"{entry}atrim=start={start:.6f}:end={end:.6f},"
                          f"asetpts=PTS-STARTPTS,"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        elif piece["source"] == "master" and master_stream_order != None:
            start = piece["source_start_ms"] / Decimal("1000")
            end = start + duration
            if len(master_split):
                entry = f"[{master_split[master_index]}]"
            else:
                entry = f"[1:{master_stream_order}]"
            master_index += 1
            chains.append(f"{entry}atrim=start={start:.6f}:end={end:.6f},"
                          f"asetpts=PTS-STARTPTS,"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        else:
            chains.append(f"anullsrc=r={sample_rate}:cl={layout},"
                          f"atrim=start=0:end={duration:.6f},asetpts=PTS-STARTPTS,"
                          f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
                          f"[{label}]")
        labels.append(label)

    chains.append("".join(f"[{label}]" for label in labels)
                  + f"concat=n={len(labels)}:v=0:a=1[aout]")
    return ";".join(chains)


def get_encoder_arguments(audio, codec_name):
    '''Meme famille de codec que la source, jamais mieux.'''
    if codec_name not in audio_encoder_by_codec:
        raise chimeric_error(
            f"no encoder kept for codec {codec_name}: track declined by name")
    encoder, family = audio_encoder_by_codec[codec_name]
    arguments = ["-c:a", encoder]
    if family == "lossless":
        if encoder == "flac":
            arguments.extend(["-compression_level", "8"])
    else:
        bitrate = None
        try:
            bitrate = video.get_bitrate(audio)
        except Exception:
            bitrate = None
        if bitrate != None and str(bitrate).isdigit() and int(bitrate) > 0:
            arguments.extend(["-b:a", str(int(bitrate))])
    return arguments, family


def find_master_audio_for_language(master_obj, language):
    '''La piste du maitre qui remplira les trous, ou None.'''
    for holder in (master_obj.audios, master_obj.commentary, master_obj.audiodesc):
        if language in holder and len(holder[language]):
            return holder[language][0]
    return None


def build_one_audio_track(candidate_obj, master_obj, audio, language, pieces,
                          out_path, timeout):
    '''Produit une piste audio chimerique. Renvoie un dict de compte-rendu.'''
    codec_name = audio.get("ffprobe", {}).get("codec_name", "").lower()
    encoder_arguments, family = get_encoder_arguments(audio, codec_name)
    sample_rate, channels, layout = get_audio_stream_parameters(audio)

    master_audio = find_master_audio_for_language(master_obj, language)
    master_stream_order = None
    if master_audio != None:
        master_stream_order = int(master_audio["StreamOrder"])
    needs_master = any(piece["source"] == "master" for piece in pieces)
    fill = "none"
    if needs_master:
        fill = "master" if master_stream_order != None else "silence"

    filtergraph = build_audio_filtergraph(
        pieces, int(audio["StreamOrder"]), master_stream_order, sample_rate, layout)

    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath]
    if master_stream_order != None:
        command.extend(["-i", master_obj.filePath])
    else:
        # `concat` et les labels d'entree sont numerotes: on garde l'entree 1
        # meme inutilisee pour que le graphe ait toujours la meme forme.
        command.extend(["-i", master_obj.filePath])
    command.extend(["-filter_complex", filtergraph, "-map", "[aout]"])
    command.extend(encoder_arguments)
    command.extend(["-ar", sample_rate, "-vn", "-sn", "-dn",
                    "-max_muxing_queue_size", "16384", out_path])
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)

    return {"stream_order": int(audio["StreamOrder"]), "language": language,
            "codec": codec_name, "encoder": encoder_arguments[1],
            "family": family, "gap_fill": fill, "path": out_path,
            "title": audio.get("Title"), "kind": None}


def classify_subtitle(codec_name):
    if codec_name in subtitle_text_codecs_to_ass:
        return "ass"
    if codec_name in subtitle_text_codecs_to_srt:
        return "srt"
    if codec_name in subtitle_bitmap_codecs:
        return "bitmap"
    if codec_name in subtitle_stream_codecs:
        return "stream"
    return "unknown"


def retime_subtitle_file(subtitle_path, pieces):
    '''Reecrit les timestamps des repliques sur la timeline du maitre.

    Une replique dont le temps ne tombe dans aucun morceau du candidat designe
    du contenu que le maitre n'a pas: elle est SUPPRIMEE, pas deplacee. Renvoie
    (gardees, supprimees).
    '''
    import pysubs2
    subtitles = pysubs2.load(subtitle_path)
    candidate_pieces = [p for p in pieces if p["source"] == "candidate"]
    kept_events = []
    dropped = 0
    for event in subtitles.events:
        shift = None
        for piece in candidate_pieces:
            source_start = piece["source_start_ms"]
            source_end = source_start + (piece["master_end_ms"] - piece["master_start_ms"])
            if Decimal(str(event.start)) >= source_start and Decimal(str(event.start)) < source_end:
                shift = piece["master_start_ms"] - source_start
                break
        if shift == None:
            dropped += 1
            continue
        event.start = int(event.start + shift)
        event.end = int(event.end + shift)
        if event.end <= event.start:
            dropped += 1
            continue
        kept_events.append(event)
    subtitles.events = kept_events
    subtitles.save(subtitle_path)
    return len(kept_events), dropped


def build_one_subtitle_track(candidate_obj, subtitle, language, pieces, work_dir,
                             index, timeout):
    '''Extrait, re-cale, renvoie un dict de compte-rendu -- ou leve.'''
    codec_name = subtitle.get("ffprobe", {}).get("codec_name", "").lower()
    target = classify_subtitle(codec_name)
    if target == "bitmap":
        raise chimeric_error(
            f"codec {codec_name} is a bitmap subtitle: its timestamps live "
            f"inside binary segments, cue rewriting cannot reach them")
    if target == "stream":
        raise chimeric_error(
            f"codec {codec_name} carries its timing in a transport stream, not "
            f"in discrete cues")
    if target == "unknown":
        raise chimeric_error(f"codec {codec_name} is not a known subtitle codec here")

    out_path = path.join(work_dir, f"sub_{index}.{target}")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath,
               "-map", f"0:{int(subtitle['StreamOrder'])}",
               "-c:s", target, out_path]
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
    kept, dropped = retime_subtitle_file(out_path, pieces)
    return {"stream_order": int(subtitle["StreamOrder"]), "language": language,
            "codec": codec_name, "format": target, "path": out_path,
            "kept_cues": kept, "dropped_cues": dropped,
            "title": subtitle.get("Title")}


def mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout):
    '''Assemble les pistes produites et pose le tag VMSAM_FABRICATED.

    Le tag est pose ici, sur le fichier de la reparation, et non dans
    `generate_new_file`: cette fonction est hors zone taguee
    (`WRITE_ZONES.MD` s2), et la mesure du 2026-09-03 montre que le tag survit
    de toute facon aux deux passes ffmpeg et aux deux mkvmerge.
    '''
    command = [tools.software["ffmpeg"], "-y", "-nostdin"]
    for report in audio_reports:
        command.extend(["-i", report["path"]])
    for report in subtitle_reports:
        command.extend(["-i", report["path"]])

    for i in range(len(audio_reports) + len(subtitle_reports)):
        command.extend(["-map", f"{i}:0"])
    command.extend(["-c", "copy"])

    for i, report in enumerate(audio_reports):
        command.extend([f"-metadata:s:a:{i}", f"VMSAM_FABRICATED={marker_value}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:a:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:a:{i}", f"title={report['title']}"])
    for i, report in enumerate(subtitle_reports):
        command.extend([f"-metadata:s:s:{i}", f"VMSAM_FABRICATED={marker_value}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:s:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:s:{i}", f"title={report['title']}"])

    command.extend(["-max_muxing_queue_size", "16384", out_path])
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)


def iterate_candidate_audios(candidate_obj):
    for holder in (candidate_obj.audios, candidate_obj.commentary,
                   candidate_obj.audiodesc):
        for language, audios in holder.items():
            for audio in audios:
                yield language, audio


def get_master_timeline_length_ms(master_obj):
    '''La timeline de sortie EST celle du maitre, et sa longueur est celle que
    `generate_new_file` imposera: elle passe `-t duration_best_video` avec
    `duration_best_video = best_video.video['Duration']` (mergeVideo.py:1781).
    Construire plus long serait tronque, plus court laisserait un trou.
    '''
    return Decimal(str(master_obj.video["Duration"])) * Decimal("1000")


def get_candidate_audio_length_ms(candidate_obj):
    '''Ce qu'on peut reellement lire dans le candidat.

    La duree de sa piste video n'est pas la borne: on decoupe de l'audio, et
    une piste audio peut etre plus longue ou plus courte que l'image. On prend
    la plus longue piste audio, et on retombe sur la video si le fichier n'en a
    aucune.
    '''
    longest = None
    for language, audio in iterate_candidate_audios(candidate_obj):
        if "Duration" not in audio:
            continue
        duration = Decimal(str(audio["Duration"])) * Decimal("1000")
        if longest == None or duration > longest:
            longest = duration
    if longest == None:
        longest = Decimal(str(candidate_obj.video["Duration"])) * Decimal("1000")
    return longest


def assemble_on_master_timeline(candidate_obj, master_obj, segments, work_dir,
                                out_path, marker_value, timeout=3600):
    '''Point d'entree du module.

    Renvoie un compte-rendu: ce qui a ete construit, ce qui a ete REFUSE et
    pourquoi. Une piste refusee est comptee separement d'une piste en echec --
    docs/SUBTITLE_CODECS.MD: "count a declined codec separately from a failed
    extract".
    '''
    master_duration_ms = get_master_timeline_length_ms(master_obj)
    candidate_duration_ms = get_candidate_audio_length_ms(candidate_obj)
    pieces = normalize_segments(segments, master_duration_ms, candidate_duration_ms)

    tools.make_dirs(work_dir)
    audio_reports = []
    subtitle_reports = []
    declined = []
    failed = []

    index = 0
    for language, audio in iterate_candidate_audios(candidate_obj):
        track_path = path.join(work_dir, f"audio_{index}.mka")
        index += 1
        try:
            audio_reports.append(build_one_audio_track(
                candidate_obj, master_obj, audio, language, pieces, track_path,
                timeout))
        except chimeric_error as error:
            declined.append({"kind": "audio",
                             "stream_order": int(audio["StreamOrder"]),
                             "language": language, "reason": str(error)})
        except Exception as error:
            failed.append({"kind": "audio",
                           "stream_order": int(audio["StreamOrder"]),
                           "language": language, "reason": str(error)})
            tools.logs.append(f"chimeric: audio track {audio['StreamOrder']} failed: {error}\n")

    index = 0
    for language, subtitles in candidate_obj.subtitles.items():
        for subtitle in subtitles:
            try:
                subtitle_reports.append(build_one_subtitle_track(
                    candidate_obj, subtitle, language, pieces, work_dir, index,
                    timeout))
            except chimeric_error as error:
                declined.append({"kind": "subtitle",
                                 "stream_order": int(subtitle["StreamOrder"]),
                                 "language": language, "reason": str(error)})
            except Exception as error:
                failed.append({"kind": "subtitle",
                               "stream_order": int(subtitle["StreamOrder"]),
                               "language": language, "reason": str(error)})
                tools.logs.append(f"chimeric: subtitle track {subtitle['StreamOrder']} failed: {error}\n")
            index += 1

    if not len(audio_reports) and not len(subtitle_reports):
        raise chimeric_error(
            f"nothing could be rebuilt: {len(declined)} track(s) declined, "
            f"{len(failed)} failed")

    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout)

    return {"path": out_path, "pieces": pieces, "audios": audio_reports,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": marker_value}
