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
import re
import subprocess
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
# LA CLASSIFICATION DES SOUS-TITRES N'EST PLUS ICI. Elle vit dans `tools` et ce
# module la LIT -- decision du proprietaire, docs/SUBTITLE_CODECS.MD. Il y avait
# deux ensembles a tenir d'accord et ils avaient DEJA diverge dans les deux
# sens: `tools` connait s_hdmv/pgs, pgs, vobsub, s_vobsub que ce module
# ignorait; ce module connaissait dvb_subtitle et xsub que `tools` ignore. Deux
# copies d'une meme verite, maintenues separement.
#
# ON REFUSE PAR EXCLUSION, JAMAIS PAR LISTE BLANCHE: un codec absent d'une liste
# blanche est jete en silence et rapporte comme une image, alors qu'un codec
# absent d'une liste d'exclusion est TENTE et echoue bruyamment sur un vrai
# defaut.
#
# BESOIN BLOQUE, A REMONTER ET NON A CONTOURNER: `dvb_subtitle` et `xsub` sont
# des sous-titres bitmap et ne sont dans AUCUN ensemble de `tools`. Sous cette
# regle ils seront donc tentes comme du texte et echoueront. La ligne qui
# devrait les porter -- tools.py:207 -- est GELEE. On ne garde pas de
# supplement local: ce supplement est exactement la copie qui a diverge.


class chimeric_error(Exception):
    '''Le plan ne peut pas etre execute. Refus explicite, jamais un fallback.'''
    pass


def get_segment_offset(segment, stream_order=None):
    """Le decalage de CETTE tranche pour CETTE piste.

    `vmsam-dev-1` mesure et emet un decalage PAR FLUX --
    `candidate_offset_ms_by_stream`, indexe par StreamOrder -- parce que deux
    pistes de la meme langue dans un meme fichier ne sont pas forcement calees
    entre elles. Mesure sur l'erreur 266: 27.6 ms d'ecart entre les deux pistes
    jpn, confirme par trois instruments.

    Ne PAS le consommer laissait la seconde piste a 27 a 29 ms du maitre pendant
    que la premiere etait a 1 ms -- sous le cadre video, sous le pas de
    `adjust_delay_to_frame`, sous la tolerance du verificateur, donc livrable en
    silence. C'est mon propre verificateur qui l'a vu, piste par piste.

    Repli sur le decalage unique quand la mesure n'en donne pas par flux: un
    bloc plus ancien reste consommable.
    """
    by_stream = segment.get("candidate_offset_ms_by_stream")
    if by_stream and stream_order is not None:
        for key in (stream_order, str(stream_order), int(stream_order)):
            if key in by_stream:
                return Decimal(str(by_stream[key]))
    return Decimal(str(segment["candidate_offset_ms"]))


def normalize_segments(segments, master_duration_ms, candidate_duration_ms,
                       speed_ratio=None, stream_order=None):
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

    # UN DECALAGE CONSTANT SANS TROU N'EST PAS UNE REPARATION, et le reconstruire
    # serait pire que ne rien faire.
    #
    # Si toutes les tranches portent le meme decalage ET se touchent bout a bout
    # sur toute la timeline du maitre, alors le candidat n'a ni contenu en trop
    # ni contenu en moins: il est simplement decale. Ce cas se traite par un
    # delai de conteneur -- `--sync` ou `-itsoffset` -- que le merge sait deja
    # poser. Le reconstruire couterait une generation de codec sur chaque piste
    # audio, et docs/SUBTITLE_CODECS.MD dit la suite: "A constant delay needs no
    # adaptation at all, for any codec", alors que la reconstruction PERD les
    # sous-titres bitmap que le simple decalage aurait gardes.
    #
    # C'est la sixieme population que vmsam-forensic a mesuree: 24 fichiers
    # (7.6 %) au decalage constant, refuses quand meme, et pour 11 d'entre eux la
    # cause mesuree est dans le MAITRE et pas dans le candidat. Les reparer ici
    # reviendrait a reconstruire une piste pour corriger un defaut de la
    # reference.
    offsets = set(get_segment_offset(segment, stream_order) for segment in ordered)
    touching = all(Decimal(str(ordered[i]["master_end_ms"]))
                   == Decimal(str(ordered[i + 1]["master_start_ms"]))
                   for i in range(len(ordered) - 1))
    covers_all = (Decimal(str(ordered[0]["master_start_ms"])) <= 0
                  and Decimal(str(ordered[-1]["master_end_ms"])) >= master_duration_ms)
    # ...SAUF si une relation de vitesse est appliquee. Une tranche unique a
    # decalage nul par-dessus un reechantillonnage n'est pas un simple decalage:
    # c'est la forme normale d'un plan de VITESSE SEULE, ou le travail est fait
    # par asetrate et non par le decoupage. Sans cette exception le chemin de
    # l'objectif 3 serait refuse par une garde ecrite pour l'objectif 2 -- ce
    # qu'elle a fait sur le premier vrai run, et c'est pour cela qu'on tire les
    # gardes plutot que de les relire.
    if len(offsets) == 1 and touching and covers_all and speed_ratio == None:
        raise chimeric_error(
            f"constant offset of {offsets.pop()} ms with no gap: this pair needs a "
            f"container delay, not a rebuilt track. Rebuilding would cost a codec "
            f"generation on every audio track and would drop the bitmap subtitles "
            f"that a plain shift keeps")
    pieces = []
    cursor = Decimal("0")
    previous_candidate_end = None

    for segment in ordered:
        master_start = Decimal(str(segment["master_start_ms"]))
        master_end = Decimal(str(segment["master_end_ms"]))
        offset = get_segment_offset(segment, stream_order)

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


def get_stream_start_ms(audio):
    """Le `start_time` du conteneur pour ce flux, en ms.

    mediainfo l'expose aussi comme `Delay` avec `Delay_Source = Container`; on
    lit ffprobe, qui est la meme valeur et celle que les filtres voient.
    """
    if audio == None:
        return Decimal("0")
    value = (audio.get("ffprobe") or {}).get("start_time")
    if value == None:
        return Decimal("0")
    try:
        return Decimal(str(value)) * Decimal("1000")
    except Exception:
        return Decimal("0")


def build_audio_filtergraph(pieces, candidate_stream_order, master_stream_order,
                            sample_rate, layout, speed_chain=None,
                            candidate_start_ms=None, master_start_ms=None):
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

    # La relation de vitesse s'applique AVANT le decoupage: la pente est dessous,
    # l'escalier est dessus (SPEC_ZONE_A.MD s4, "chimeric+resampled" dans cet
    # ordre). Les temps des tranches sont donc lus sur le candidat DEJA
    # reechantillonne, ce qui est ce que dit INTERFACE_dev1_dev2.md s4.
    candidate_entry = f"[0:{candidate_stream_order}]"
    if speed_chain != None:
        chains.append(f"{candidate_entry}{speed_chain}[spd]")
        candidate_entry = "[spd]"

    candidate_split = []
    if len(candidate_pieces) > 1:
        candidate_split = [f"cs{i}" for i in range(len(candidate_pieces))]
        chains.append(f"{candidate_entry}asplit={len(candidate_pieces)}"
                      + "".join(f"[{label}]" for label in candidate_split))
    master_split = []
    if master_stream_order != None and len(master_pieces) > 1:
        master_split = [f"ms{i}" for i in range(len(master_pieces))]
        chains.append(f"[1:{master_stream_order}]asplit={len(master_pieces)}"
                      + "".join(f"[{label}]" for label in master_split))

    # LE MORCEAU DE TETE EST PLUS COURT QUE LA PLACE QU'IL OCCUPE quand la piste
    # commence apres zero. `atrim=start=0` sur un flux dont le `start_time` vaut
    # 1.103 s ne rend pas 1.103 s de contenu inexistant: il commence a 1.103.
    # `asetpts=PTS-STARTPTS` le remet a zero et `concat` colle bout a bout, DONC
    # TOUT CE QUI SUIT REMONTE DE 1.103 s. Mesure le 2026-09-03: un morceau
    # demande a [0, 120] rend 118.900 s, et l'erreur de la piste produite vaut
    # exactement le `start_time` DU MAITRE -- 1103.4 contre 1103.0 sur un
    # fichier, 887.6 contre 887.0 sur un autre.
    #
    # On rembourre donc la tete du morceau du silence qui manque. Ce n'est pas
    # une invention: il n'y a REELLEMENT pas de son avant `start_time`, et le
    # silence est ce que le lecteur entend deja la.
    pads = []

    def head_pad(source_start_ms, stream_start_ms, sink):
        if stream_start_ms == None:
            return ""
        missing = Decimal(str(stream_start_ms)) - Decimal(str(source_start_ms))
        if missing <= 0:
            return ""
        sink.append(missing)
        return f",adelay={int(missing)}:all=1"

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
                entry = candidate_entry
            candidate_index += 1
            chains.append(f"{entry}atrim=start={start:.6f}:end={end:.6f},"
                          f"asetpts=PTS-STARTPTS"
                          f"{head_pad(piece['source_start_ms'], candidate_start_ms, pads)},"
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
                          f"asetpts=PTS-STARTPTS"
                          f"{head_pad(piece['source_start_ms'], master_start_ms, pads)},"
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
    return ";".join(chains), sum(pads) if len(pads) else Decimal("0")


def resolve_source_bitrate(audio, source_path, timeout=120):
    '''Le debit de la piste source, et D'OU il vient.

    Renvoie (debit, origine). Leve si aucune des quatre voies ne repond: laisser
    l'encodeur choisir son defaut n'est PAS une issue neutre. Mesure 2026-09-03,
    et c'est ce qui a motive cette fonction: sur une piste AAC dont ni ffprobe
    ni mediainfo ne donne de debit, l'encodeur natif d'ffmpeg prend
    canaux x frequence x 1.5, soit 132300 bps pour du 2 canaux a 44100 -- plus
    haut que la source, et assez haut pour que `keep_best_audio` prefere la
    piste fabriquee a la piste intacte.

    Inventer une valeur plausible serait le defaut que SPEC_ZONE_A.MD s3 decrit
    a propos de `get_less_channel_number`: une valeur fabriquee indiscernable
    d'une vraie partout en aval. On mesure, ou on refuse.
    '''
    try:
        bitrate = video.get_bitrate(audio)
        if bitrate != None and str(bitrate).isdigit() and int(bitrate) > 0:
            return int(bitrate), "video.get_bitrate"
    except Exception:
        pass

    # `video.get_bitrate` couvre deja ffprobe.bit_rate, BitRate et
    # BitRate_Nominal (video.py:947-959). Il reste BitRate_Maximum, qu'il ne
    # regarde pas.
    for key in ("BitRate_Maximum",):
        value = audio.get(key)
        if value != None and str(value).isdigit() and int(value) > 0:
            return int(value), f"mediainfo.{key}"

    stream_size = audio.get("StreamSize")
    duration = audio.get("Duration")
    if stream_size != None and duration != None:
        try:
            computed = int(Decimal(str(stream_size)) * 8 / Decimal(str(duration)))
            if computed > 0:
                return computed, "StreamSize/Duration"
        except Exception:
            pass

    # Derniere voie, et la seule qui reponde toujours: une passe `-c copy` vers
    # null lit la taille reelle du flux sans le decoder. Mesure: 0.46 s pour
    # 592 s d'AAC.
    command = [tools.software["ffmpeg"], "-nostdin", "-hide_banner", "-i", source_path,
               "-map", f"0:{int(audio['StreamOrder'])}", "-c", "copy", "-f", "null", "-"]
    try:
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(
            command, 1, timeout)
    except Exception as error:
        raise chimeric_error(
            f"the source bitrate of track {audio.get('StreamOrder')} could not "
            f"be measured: {error}")
    text = stderror.decode("utf-8", errors="ignore")
    match = re.search(r"audio:\s*(\d+)\s*([KMG])iB", text)
    if match != None and duration != None:
        scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[match.group(2)]
        try:
            measured = int(Decimal(match.group(1)) * scale * 8 / Decimal(str(duration)))
            if measured > 0:
                return measured, "measured by a copy pass"
        except Exception:
            pass

    raise chimeric_error(
        f"the source bitrate of track {audio.get('StreamOrder')} could not be "
        f"determined by any of the four routes: track declined rather than "
        f"encoded at whatever default the encoder picks")


def get_encoder_arguments(audio, codec_name, source_path=None):
    '''Meme famille de codec que la source, jamais mieux, et jamais un debit
    plus haut que celui de la source.'''
    if codec_name not in audio_encoder_by_codec:
        raise chimeric_error(
            f"no encoder kept for codec {codec_name}: track declined by name")
    encoder, family = audio_encoder_by_codec[codec_name]
    arguments = ["-c:a", encoder]
    bitrate_origin = None
    if family == "lossless":
        if encoder == "flac":
            arguments.extend(["-compression_level", "8"])
    else:
        bitrate, bitrate_origin = resolve_source_bitrate(audio, source_path)
        arguments.extend(["-b:a", str(bitrate)])
    return arguments, family, bitrate_origin


def find_master_audio_for_language(master_obj, language, reference_stream=None):
    '''La piste du maitre qui remplira les trous, ou None.

    PAS UN COMMENTAIRE DU MAITRE: remplir le trou d'une piste principale avec
    un commentaire produirait un fichier qui passe tout controle structurel et
    qui est indefendable a l'ecoute. Meme decision que ci-dessus, cote
    remplissage.

    `holder[language][0]` est LE PREMIER flux de cette langue, pas le meilleur,
    et sur un maitre qui en porte deux c'est un choix arbitraire. `vmsam-forensic`
    a mesure 126 a 138 ms d'ecart entre deux pistes de meme langue d'un meme
    maitre, et a verifie que ce n'est PAS un artefact de `start_time` (dix
    maitres a 0.0 ms d'ecart, un a 16 ms contre un ecart de 126-138). Le cout
    n'est nul que si le remplissage et l'alignement retombent sur le MEME
    indice; c'est note comme une condition a verifier, pas comme un defaut
    mesure ici.
    '''
    for holder in (master_obj.audios, master_obj.audiodesc):
        tracks = holder.get(language)
        if tracks == None or not len(tracks):
            continue
        if len(tracks) == 1:
            return tracks[0]
        # THE ONE TRACK KNOWN TO BE ALIGNED WITH THE PLAN, and known by
        # MEASUREMENT rather than by inference: the measurement was taken
        # against it, and `change_point_locator` emits it as
        # `reference_stream`. Codec, channels and bitrate are not alignment --
        # ranking on them is principled on a dimension the repair does not care
        # about, and on a master whose same-language tracks disagree it trades
        # one wrong answer for another that is harder to predict. This prefers
        # the dimension that matters and falls back to quality only when the
        # measured stream is not a candidate for THIS language.
        if reference_stream != None:
            for track in tracks:
                if str(track.get("StreamOrder")) == str(reference_stream):
                    return track
        return pick_best_master_audio(tracks)
    return None


def find_fill_audio(master_obj, language, reference_stream=None,
                    comparison_language=None):
    """La piste du maitre qui remplira les trous de CETTE piste.

    Regle du proprietaire, `SPEC_ZONE_A.MD` s4c: la piste de MEME LANGUE du
    `best_video` comble le trou, et A DEFAUT LA MEILLEURE AUDIO DE LA LANGUE DE
    COMPARAISON. Le silence n'est plus le repli normal -- il ne reste que quand
    le maitre ne porte NI l'une NI l'autre.

    Mesure qui a motive la regle: sur 42 pistes remplies de silence, LES 42
    avaient une langue partagee disponible sur le maitre. Le code n'avait qu'une
    branche la ou la regle en a deux, et le remplissage inter-langue n'existait
    pas du tout.

    CECI EST UNE FONCTION SEPAREE ET PAS UN PARAMETRE DE PLUS SUR
    `find_master_audio_for_language`. Cette derniere sert AUSSI au verificateur,
    qui cherche la reference contre laquelle comparer une piste: y ajouter un
    repli inter-langue ferait comparer une piste francaise a du japonais et
    rendrait un "aligned" qui ne veut rien dire. Les deux usages ont l'air
    identiques et ne le sont pas.

    LE PROPRIETAIRE A STATUE EN SACHANT QUE C'EST AUDIBLE. Un trou francais
    comble en japonais s'entend comme un changement de langue; je l'ai signale
    avant d'implementer et la regle a ete confirmee. `fill_language` porte la
    langue REELLEMENT utilisee pour que le journal le dise.
    """
    same = find_master_audio_for_language(master_obj, language, reference_stream)
    if same != None:
        return same, language
    if comparison_language in (None, "", language):
        return None, None
    other = find_master_audio_for_language(master_obj, comparison_language,
                                           reference_stream)
    if other != None:
        return other, comparison_language
    return None, None


def pick_best_master_audio(tracks):
    """The BEST stream of that language, not the first.

    THE RANKING IS NOT MINE TO INVENT: `keep_best_audio` already defines "best"
    for this system, and a second definition would diverge -- we watched two
    subtitle classifications do exactly that today.

    BUT `keep_best_audio` IS A MUTATOR, NOT A SELECTOR. It sets keep=False on
    the losers IN PLACE, and the dicts passed here are the master's REAL audio
    dicts, which the merge reads afterwards. Calling it directly would make a
    repair mutate state the merge owns. So it runs on COPIES and the survivor is
    mapped back by StreamOrder.

    TWO PRECONDITIONS THIS MODULE DOES NOT ESTABLISH, checked rather than
    assumed: `tools.mergeRules` must be loaded and every dict must carry `keep`.
    Either missing -> fall back to the first track, which is today's behaviour.

    A TIE IS NOT A DECISION. If the ranking leaves zero or several survivors it
    has not chosen, and no tie-break is invented to make the function look
    decisive: it returns the first track, exactly the current behaviour.

    Why: `vmsam-forensic` measured that 22.6 % of masters carry more than one
    normal track of a language, up to four, and 126-138 ms between two of them
    on one master -- verified NOT to be a `start_time` artefact. Taking the
    first was an arbitrary choice that carried that gap into the repair.
    """
    import copy
    try:
        import mergeVideo
        rules = mergeVideo.decript_merge_rules(tools.mergeRules['audio'])
    except Exception:
        return tracks[0]
    candidates = []
    for track in tracks:
        clone = copy.deepcopy(track)
        clone["keep"] = True
        candidates.append(clone)
    try:
        mergeVideo.keep_best_audio(candidates, rules)
    except Exception:
        return tracks[0]
    survivors = [c for c in candidates if c.get("keep")]
    if len(survivors) != 1:
        return tracks[0]
    order = survivors[0].get("StreamOrder")
    for track in tracks:
        if track.get("StreamOrder") == order:
            return track
    return tracks[0]


def build_one_audio_track(candidate_obj, master_obj, audio, language, pieces,
                          out_path, timeout, speed_ratio=None,
                          reference_stream=None, comparison_language=None):
    '''Produit une piste audio chimerique. Renvoie un dict de compte-rendu.'''
    codec_name = audio.get("ffprobe", {}).get("codec_name", "").lower()
    encoder_arguments, family, bitrate_origin = get_encoder_arguments(
        audio, codec_name, candidate_obj.filePath)
    sample_rate, channels, layout = get_audio_stream_parameters(audio)

    master_audio, fill_language = find_fill_audio(
        master_obj, language, reference_stream, comparison_language)
    master_stream_order = None
    if master_audio != None:
        master_stream_order = int(master_audio["StreamOrder"])
    needs_master = any(piece["source"] == "master" for piece in pieces)
    fill = "none"
    if needs_master:
        fill = "master" if master_stream_order != None else "silence"
    # LA LANGUE REELLEMENT UTILISEE POUR REMPLIR, pas celle de la piste.
    # `SPEC_ZONE_A.MD` s4e l'exige nommement, et le champ est ajoute AVANT que le
    # remplissage inter-langue existe: aujourd'hui il vaut toujours la langue de
    # la piste, et le jour ou le remplissage ira chercher la langue de
    # comparaison, LA LIGNE DE JOURNAL SERA DEJA JUSTE. Ajouter le champ apres
    # le changement ferait decrire par ce champ quelque chose deja livre sans
    # journal, ce que l'exigence existe precisement pour empecher.
    # LE TITRE DU FLUX QUI REMPLIT, a cote de sa langue. `video.py` indexe le
    # dictionnaire sur le code ISO -- `data['Language'].split("-")[0]` -- donc
    # `es-ES` et `es-419` TOMBENT DANS LE MEME SEAU AVANT QUE CE MODULE NE VOIE
    # QUOI QUE CE SOIT, et le seul indice restant est le titre libre: sur un
    # fichier du corpus, quatre pistes `spa` titrees "European Spanish",
    # "Latinoamerican Spanish", "Spanish" et "European Spanish" a nouveau.
    #
    # RIEN ICI NE CHOISIT SUR LE TITRE et ce champ ne change aucun
    # comportement: il ECRIT CE QUI A ETE PRIS. Si un doublage est substitue a
    # un autre cette nuit, ce sera dans l'artefact au lieu d'etre invisible --
    # la meme raison qui a fait ajouter `fill_language` avant que le
    # remplissage inter-langue existe.
    fill_title = master_audio.get("Title") if fill == "master" and master_audio != None else None
    if fill != "master":
        fill_language = None

    speed_chain = None
    applied_ratio = None
    if speed_ratio != None:
        import merge_video_resample
        speed_chain, applied_ratio, _, _ = \
            merge_video_resample.build_speed_filter_chain(sample_rate, speed_ratio)
    # Le `start_time` du CANDIDAT suit la relation de vitesse: apres
    # reechantillonnage la piste commence `ratio` fois plus tard.
    candidate_start_ms = get_stream_start_ms(audio)
    if speed_ratio != None:
        candidate_start_ms = candidate_start_ms * Decimal(str(speed_ratio))
    filtergraph, head_pad_ms = build_audio_filtergraph(
        pieces, int(audio["StreamOrder"]), master_stream_order, sample_rate, layout,
        speed_chain, candidate_start_ms, get_stream_start_ms(master_audio))

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

    bitrate = None
    if "-b:a" in encoder_arguments:
        bitrate = encoder_arguments[encoder_arguments.index("-b:a") + 1]
    # Ce que le trou COUTE a cette piste-ci. Quand le maitre porte la langue, un
    # trou est du contenu de reference et ne coute rien; quand il ne la porte
    # pas, c'est du silence, et c'est une perte seche qu'il faut compter.
    # vmsam-dev-1 decide s'il construit un affineur video en fonction de la
    # taille reelle de cette population: le chiffre est donc un resultat, pas une
    # curiosite.
    filled = Decimal("0")
    for piece in pieces:
        if piece["source"] == "master":
            filled += piece["master_end_ms"] - piece["master_start_ms"]
    total = pieces[-1]["master_end_ms"] - pieces[0]["master_start_ms"]
    silence_ms = filled if fill == "silence" else Decimal("0")
    return {"stream_order": int(audio["StreamOrder"]), "language": language,
            "codec": codec_name, "encoder": encoder_arguments[1],
            "family": family, "gap_fill": fill, "fill_language": fill_language,
            "fill_title": fill_title, "path": out_path,
            "bitrate": bitrate, "bitrate_origin": bitrate_origin,
            "speed_ratio_requested": str(speed_ratio) if speed_ratio != None else None,
            "speed_ratio_applied": str(applied_ratio) if applied_ratio != None else None,
            "gap_filled_ms": str(filled),
            # Le silence ajoute EN TETE parce que la source ne commence pas a
            # zero. Se dit: c'est du contenu que la piste produite n'a pas, et
            # il ne doit pas se confondre avec le remplissage depuis le maitre.
            "head_pad_ms": str(head_pad_ms),
            "silence_filled_ms": str(silence_ms),
            "silence_fraction": str((silence_ms / total).quantize(Decimal("0.000001")))
                                if total > 0 else "0",
            "title": audio.get("Title"), "kind": None}


def classify_subtitle(codec_name):
    """Trois issues, par EXCLUSION, depuis `tools` et non depuis une copie."""
    name = (codec_name or "").lower()
    if name in tools.sub_type_not_encodable:
        return "bitmap"
    if name in tools.sub_type_near_srt:
        return "srt"
    return "ass"


def retime_subtitle_file(subtitle_path, pieces, speed_ratio=None):
    '''Reecrit les timestamps des repliques sur la timeline du maitre.

    Une replique dont le temps ne tombe dans aucun morceau du candidat designe
    du contenu que le maitre n'a pas: elle est SUPPRIMEE, pas deplacee. Renvoie
    (gardees, supprimees).
    '''
    import pysubs2
    subtitles = pysubs2.load(subtitle_path)
    if speed_ratio != None:
        # CAMPAIGN.MD et le brief sont explicites: on prend l'audio ET les
        # sous-titres. Le meme coefficient, et dans le meme ordre que l'audio --
        # la vitesse d'abord, le decoupage ensuite -- sinon les repliques et la
        # bande son derivent l'une par rapport a l'autre.
        import merge_video_resample
        merge_video_resample.retime_subtitle_events_by_ratio(subtitles, speed_ratio)
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
                             index, timeout, speed_ratio=None):
    '''Extrait, re-cale, renvoie un dict de compte-rendu -- ou leve.'''
    codec_name = subtitle.get("ffprobe", {}).get("codec_name", "").lower()
    target = classify_subtitle(codec_name)
    if target == "bitmap":
        raise chimeric_error(
            f"codec {codec_name} is a bitmap subtitle: its timestamps live "
            f"inside binary segments, cue rewriting cannot reach them")
    # Plus de branche "stream" ni "unknown": sous la regle d'exclusion, un codec
    # que `tools` ne nomme pas est TENTE. S'il n'est pas convertible, ffmpeg
    # echoue et le compte-rendu porte un ECHEC nomme -- un vrai defaut, visible.
    # L'ancienne branche "unknown" le declarait "pas un codec de sous-titre
    # connu ici", ce qui est une etiquette presentee comme un diagnostic.

    out_path = path.join(work_dir, f"sub_{index}.{target}")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath,
               "-map", f"0:{int(subtitle['StreamOrder'])}",
               "-c:s", target, out_path]
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
    kept, dropped = retime_subtitle_file(out_path, pieces, speed_ratio)
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
    """Les pistes que la reparation a le droit de reconstruire.

    PAS LES COMMENTAIRES -- decision du proprietaire, SPEC_ZONE_A.MD s4. La
    reparation repose sur "ce qui a ete fait au fichier a ete fait a tous ses
    flux": vrai d'une coupe, vrai d'un reechantillonnage, et c'est cette
    premisse qui autorise a appliquer a chaque flux un plan mesure sur une
    seule langue. UN COMMENTAIRE EST UN ENREGISTREMENT SEPARE SUR LA MEME
    IMAGE, pas une traduction de l'audio du programme: il n'herite ni de la
    premisse ni du plan, le maitre peut n'avoir aucun commentaire pour remplir
    un trou, et rien ne dit que ses points de montage soient ceux de la piste
    principale.

    L'AUDIO-DESCRIPTION EST DELIBEREMENT LAISSEE OUVERTE. Meme forme, et ce
    n'est PAS ce sur quoi le proprietaire a statue: "Raise it rather than
    extend this by analogy." Cela paraitra incoherent dans le code et c'est
    correct tant que la question n'est pas tranchee.
    """
    for holder in (candidate_obj.audios, candidate_obj.audiodesc):
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
                                out_path, marker_value, timeout=3600,
                                verify=True, verify_tolerance_ms=100,
                                verify_search_ms=30000, max_silence_fraction=None,
                                speed_ratio=None, reference_stream=None,
                                comparison_language=None):
    '''Point d'entree du module.

    Renvoie un compte-rendu: ce qui a ete construit, ce qui a ete REFUSE et
    pourquoi. Une piste refusee est comptee separement d'une piste en echec --
    docs/SUBTITLE_CODECS.MD: "count a declined codec separately from a failed
    extract".
    '''
    master_duration_ms = get_master_timeline_length_ms(master_obj)
    candidate_duration_ms = get_candidate_audio_length_ms(candidate_obj)
    if speed_ratio != None:
        # Apres reechantillonnage le candidat dure `ratio` fois plus longtemps,
        # et c'est cette timeline-la que les tranches decoupent. Garder l'ancienne
        # borne refuserait des plans corrects.
        candidate_duration_ms = candidate_duration_ms * Decimal(str(speed_ratio))
    pieces = normalize_segments(segments, master_duration_ms, candidate_duration_ms,
                                speed_ratio)

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
            # Les frontieres sur la timeline du MAITRE sont les memes pour toutes
            # les pistes; seul l'endroit ou l'on lit le candidat change. On
            # recalcule donc les morceaux avec le decalage de CE flux.
            track_pieces = normalize_segments(
                segments, master_duration_ms, candidate_duration_ms, speed_ratio,
                stream_order=int(audio["StreamOrder"]))
            audio_reports.append(build_one_audio_track(
                candidate_obj, master_obj, audio, language, track_pieces, track_path,
                timeout, speed_ratio, reference_stream, comparison_language))
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
                    timeout, speed_ratio))
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

    # Budget de silence: par defaut AUCUNE limite. Un seuil doit venir d'un ecart
    # mesure, pas d'une courbe (docs/AUDIO_SPEED_POLICY.MD), et personne n'a
    # encore mesure combien de silence rend une piste inutilisable. On expose
    # donc le chiffre et on laisse le proprietaire poser la limite le jour ou il
    # aura de quoi la justifier. Inventer 5000 ms ici serait exactement la valeur
    # plausible fabriquee que SPEC_ZONE_A.MD s3 decrit.
    if max_silence_fraction != None:
        kept = []
        for report in audio_reports:
            if Decimal(report["silence_fraction"]) > Decimal(str(max_silence_fraction)):
                declined.append({"kind": "audio", "stream_order": report["stream_order"],
                                 "language": report["language"],
                                 "reason": f"{report['silence_filled_ms']} ms of the track "
                                           f"would be silence "
                                           f"({report['silence_fraction']} of it), over the "
                                           f"configured budget {max_silence_fraction}"})
            else:
                kept.append(report)
        audio_reports = kept

    if not len(audio_reports) and not len(subtitle_reports):
        raise chimeric_error(
            f"nothing could be rebuilt: {len(declined)} track(s) declined, "
            f"{len(failed)} failed")

    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout)

    # L'ACCEPTATION PORTE SUR LE FICHIER ET ELLE PASSE AVANT L'ALIGNEMENT.
    # `SPEC_ZONE_A.MD` s4d. Verifier l'alignement d'une piste tronquee sonde des
    # positions qui existent encore et rend "aligned" sur un fichier ampute --
    # c'est exactement ce qui a rapporte "7 audio et 24 sous-titres
    # reconstruits, 0 refuse, 0 en echec" sur un fichier sans rien apres 21:21.
    output_check = verify_output_file(out_path, master_duration_ms, audio_reports,
                                      subtitle_reports,
                                      output_duration_tolerance_ms)

    verification = None
    if verify:
        verification = verify_on_master_timeline(
            out_path, master_obj, audio_reports, pieces, verify_tolerance_ms,
            verify_search_ms, reference_stream)

    return {"path": out_path, "pieces": pieces, "audios": audio_reports,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": marker_value,
            "output_check": output_check,
            "verification": verification}


# --------------------------------------------------------------------------
# Verification: la piste produite est-elle VRAIMENT sur la timeline du maitre?
# --------------------------------------------------------------------------

verify_window_seconds = 20
verify_probe_rate = 8000
verify_max_probes = 4
# Une fenetre sans signal n'est pas une fenetre. Mesure 2026-09-03 sur l'erreur
# 266: les premieres secondes du programme sont quasi muettes, RMS 1e-5 en pleine
# echelle, et la correlation y rend -170.69 ms avec r=0.82 -- du bruit contre du
# bruit, avec l'assurance d'une vraie mesure. Sur le meme fichier une fenetre avec
# contenu lit 2.4e-3 a 2.9e-2. Le seuil est pose entre les deux, un ordre de
# grandeur au-dessus du silence et un ordre en dessous du contenu le plus faible.
# Il vient de cet ecart mesure, pas d'une courbe.
verify_min_rms = 1e-4


def choose_probe_positions(pieces, window_seconds):
    """Positions de sondage, STRICTEMENT a l'interieur des morceaux venus du
    CANDIDAT, et DEUX PAR MORCEAU quand il est assez long.

    Sonder dans un trou serait un controle qui ne peut pas echouer: le trou est
    rempli depuis le maitre par construction, donc il correspond au maitre quel
    que soit le decalage applique au reste. `docs/AUDIO_SPEED_POLICY.MD` a une
    section entiere la-dessus -- "a control that cannot fail is not a control".

    POURQUOI DEUX. `vmsam-forensic`, note contamination_vs_resolution: l'erreur de
    contamination est bornee par l'ecart entre deux traits que l'on NE CONNAIT
    PAS, elle est de signe arbitraire, et echantillonner plus fin ne converge pas
    -- ca converge sur un pic deplace, stablement et avec confiance. Mes morceaux
    ne franchissent aucune frontiere DU PLAN; ils peuvent parfaitement en
    franchir une que le plan IGNORE -- c'est exactement l'erreur 108, dont le
    premier point de changement etait invisible a la mesure. Un seul sondage par
    morceau rendrait alors un decalage faux et confiant.

    Deux sondages ecartes dans le meme morceau ne peuvent pas etre deplaces de la
    meme facon par une frontiere situee entre eux: leur DESACCORD est le signal.
    On ne peut pas etablir qu'une fenetre ne franchit pas de frontiere, donc on
    mesure la chose qui le dirait.

    Renvoie une liste de (index_du_morceau, debut_ms).
    """
    window_ms = Decimal(str(window_seconds)) * Decimal("1000")
    # Une tranche est sondable des qu'elle contient UNE fenetre. L'ancienne
    # exigence (deux fenetres) ecartait une tranche de tete courte -- exactement
    # celle qu'un pas pres du debut produit, et exactement la region que la mesure
    # ne voit pas.
    indexed = [(i, p) for i, p in enumerate(pieces) if p["source"] == "candidate"
               and (p["master_end_ms"] - p["master_start_ms"]) >= window_ms]
    if not len(indexed):
        return []

    by_length = sorted(indexed, key=lambda x: x[1]["master_end_ms"] - x[1]["master_start_ms"],
                       reverse=True)
    by_position = sorted(indexed, key=lambda x: x[1]["master_start_ms"])
    # TOUTES les tranches, pas les quatre plus longues. Une tranche COURTE est
    # precisement celle dont le decalage n'a pas pu etre mesure proprement: la
    # mesure sonde sur une fenetre de 60 s, donc une tranche plus courte que
    # cela ne contient aucune sonde propre et son decalage vient d'un pic
    # DEPLACE. Mesure de vmsam-dev-1 sur l'erreur 266: une tranche de 29 s a
    # porte un decalage faux de 168 ms, emis sans drapeau.
    #
    # L'ancienne selection -- premiere, derniere, puis les plus longues jusqu'a
    # quatre -- pouvait sauter une tranche courte AU MILIEU, c'est-a-dire
    # exactement celle qui risque d'etre fausse. Sauter la plus suspecte pour
    # sonder deux fois la plus sure est un controle qui ne peut pas echouer.
    chosen = by_position

    # deux sondes par tranche quand elle est assez longue pour en porter deux --
    # c'est le desaccord ENTRE elles qui revele une frontiere non modelisee --
    # et une seule sinon, qui reste une borne plutot qu'une mesure.
    per_piece = 2
    positions = []
    for index, piece in chosen:
        span = piece["master_end_ms"] - piece["master_start_ms"] - window_ms
        if span < 0:
            span = Decimal("0")
        count = per_piece if span >= window_ms else 1
        divisor = Decimal(max(1, count - 1))
        for i in range(count):
            offset = span * Decimal(i) / divisor if count > 1 else span / 2
            positions.append((index, piece["master_start_ms"] + offset))
    return sorted(positions, key=lambda x: x[1])


def read_mono_samples(file_path, stream_specifier, start_ms, duration_ms, rate):
    """Seek de SORTIE, pas d'entree: un fichier sans index audio utilisable rend
    0 octet sur un seek d'entree, et une correlation sur du vide se lit comme
    "aucun accord" au lieu de "je n'ai rien lu". Mesure 2026-09-03 sur un
    fixture construit avec `-ss` en entree puis `-c:a copy`.
    """
    import numpy
    command = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
               "-i", file_path, "-map", stream_specifier,
               "-ss", f"{start_ms / Decimal('1000'):.3f}",
               "-t", f"{duration_ms / Decimal('1000'):.3f}",
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
               "-ar", str(rate), "-"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    samples = numpy.frombuffer(process.stdout, dtype=numpy.float32).astype(numpy.float64)
    if len(samples) < rate:
        raise chimeric_error(
            f"read only {len(samples)} samples from {file_path} {stream_specifier} "
            f"at {start_ms} ms: cannot verify against nothing")
    return samples - samples.mean()


def get_rms(samples):
    import numpy
    return float(numpy.sqrt((samples ** 2).mean())) if len(samples) else 0.0


def measure_lag_ms(reference, produced, rate, search_ms):
    import numpy
    size = 1 << int(numpy.ceil(numpy.log2(len(reference) + len(produced))))
    spectrum = numpy.fft.rfft(reference, size) * numpy.conj(numpy.fft.rfft(produced, size))
    correlation = numpy.fft.irfft(spectrum, size)
    correlation = numpy.concatenate((correlation[-(len(produced) - 1):],
                                     correlation[:len(reference)]))
    lags = numpy.arange(-(len(produced) - 1), len(reference))
    keep = numpy.abs(lags) <= int(search_ms * rate / 1000)
    correlation, lags = correlation[keep], lags[keep]
    peak = int(numpy.argmax(correlation))
    norm = numpy.linalg.norm(reference) * numpy.linalg.norm(produced)
    return (float(lags[peak]) * 1000.0 / rate,
            float(correlation[peak] / norm) if norm > 0 else 0.0)


# TOLERANCE DE DUREE DE SORTIE, POSEE APRES MESURE. 86 fichiers, 350 pistes
# audio, duree lue sur l'etiquette Matroska:
#
#     p50 +21.0 ms   p90 +22.0 ms   max +42.0 ms      <- le mode ordinaire
#     p10 -1968 ms   min -86400 ms                    <- l'autre mode
#     75 pistes sur 350 au-dela de +/-100 ms
#
# LA DISTRIBUTION EST BIMODALE et les deux modes sont separes de DEUX ORDRES DE
# GRANDEUR: +42 ms au pire du mode normal contre -1968 ms au dixieme centile de
# l'autre. N'IMPORTE QUELLE BORNE ENTRE 100 ms ET 1 s LES SEPARE. 500 ms est
# choisi parce que les donnees laissent un intervalle de deux decades, pas
# parce que le nombre est rond.
output_duration_tolerance_ms = Decimal("500")

# CE N'EST PAS UN DRAPEAU DE CAPACITE ET LA DISTINCTION COMPTE. Le proprietaire a
# statue qu'une reparation conditionnee a un parametre n'est pas une reparation,
# et la reparation reste inconditionnelle: ce qui est etage ici, c'est un REFUS
# NOUVEAU dont on mesure l'incidence avant de le rendre bloquant.
#
# A False le controle MESURE ET INSCRIT SON VERDICT, et ne refuse pas. Raison,
# et si le fait change la decision change: rien n'est livre cette nuit -- pas de
# boucle de production, pas de demon, et les sorties de la file sont supprimees
# apres verification. La valeur protectrice du refus est donc nulle cette nuit,
# et son cout en information est reel: refuser arrete les etapes suivantes sur
# un cinquieme du corpus, alors que MESURER donne le meme taux de refus PLUS
# tout ce qui se trouve en aval.
#
# Passe a True des que le balayage du corpus est termine, ou immediatement si
# quoi que ce soit doit etre produit pour de vrai avant cela.
output_check_enforcing = False


def probe_output_streams(file_path):
    """Ce que le FICHIER dit de lui-meme: par flux, type, langue et duree.

    On lit le fichier PRODUIT, pas le compte-rendu de ce qu'on croit avoir
    construit. `SPEC_ZONE_A.MD` s4d: un compte de pistes reconstruites est un
    enonce sur le TRAVAIL FAIT, pas sur un fichier.
    """
    import json as _json
    command = [tools.software["ffprobe"], "-v", "error",
               "-show_entries", "stream=index,codec_type:"
                                "stream_tags=language,DURATION:format=duration",
               "-of", "json", file_path]
    data = _json.loads(subprocess.run(command, check=True,
                                      stdout=subprocess.PIPE).stdout)
    ends = None
    streams = []
    for entry in data.get("streams", []):
        index = entry.get("index")
        tags = entry.get("tags") or {}
        # LA DUREE PAR FLUX EST UNE ETIQUETTE MATROSKA, PAS LE CHAMP `duration`.
        # `stream=duration` rend N/A sur toutes les pistes d'un mkv;
        # `stream_tags=DURATION` les porte. Mesure sur un fichier produit: sept
        # pistes, sept N/A d'un cote et sept horodatages de l'autre, DONT LA
        # PISTE COURTE. Trouve par `vmsam-ci`, qui avait ecrit sa sonde contre
        # un vrai fichier et change la REQUETE quand le champ est revenu vide,
        # la ou j'avais change d'INSTRUMENT.
        #
        # Le dernier paquet reste en repli pour un conteneur sans l'etiquette --
        # il SOUS-ESTIME la fin de 34 ms ici et de 42 a 85 ms chez ci, et cette
        # erreur VARIE d'une piste a l'autre, donc elle ne se soustrait pas.
        # C'est pourquoi elle est un repli et pas la mesure, ET POURQUOI CHAQUE
        # FLUX DIT LAQUELLE DES DEUX A SERVI.
        duration_ms, source = parse_duration_tag(tags.get("DURATION")), "matroska tag"
        if duration_ms == None:
            if ends == None:
                ends = last_audio_packet_ms(file_path)
            duration_ms = ends.get(index)
            source = "last packet (under-reads by one packet)" if duration_ms != None else None
        streams.append({"index": index,
                        "codec_type": entry.get("codec_type"),
                        "language": tags.get("language"),
                        "duration_ms": duration_ms,
                        "duration_source": source})
    container = (data.get("format") or {}).get("duration")
    return streams, (Decimal(str(container)) * 1000 if container not in (None, "N/A") else None)


def parse_duration_tag(value):
    """`00:23:51.993000000` -> ms. None quand l'etiquette est absente."""
    if value in (None, "", "N/A"):
        return None
    try:
        hours, minutes, seconds = str(value).split(":")
        return ((Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds))
                * 1000)
    except Exception:
        return None


def last_audio_packet_ms(file_path):
    """La fin REELLE de chaque piste audio, lue sur son dernier paquet.

    MATROSKA NE PORTE PAS DE DUREE PAR FLUX: `ffprobe -show_entries
    stream=duration` rend `N/A` sur toutes les pistes d'un mkv. Mesure sur un
    fichier produit -- sept pistes, sept `N/A`. Un controle de duree ecrit sur
    ce champ N'AURAIT MESURE AUCUN FICHIER: il aurait marque chaque piste
    `unmeasured` et refuse tout, ce qui est le miroir d'un controle qui ne peut
    pas echouer.

    On lit donc l'horodatage du DERNIER PAQUET de chaque flux, ce qui est la
    seule quantite par piste que le conteneur fournit reellement.
    `-read_intervals 99%` ne lit que la fin du fichier.

    La valeur rendue sous-estime la fin de la piste de la duree d'un paquet --
    13 a 17 ms mesures -- ce qui est bien en deca de toute tolerance
    defendable, et c'est un BIAIS CONNU DANS UNE SEULE DIRECTION plutot qu'une
    incertitude.
    """
    command = [tools.software["ffprobe"], "-v", "error", "-select_streams", "a",
               "-show_entries", "packet=stream_index,pts_time",
               "-of", "csv=p=0", "-read_intervals", "99%", file_path]
    try:
        output = subprocess.run(command, check=True,
                                stdout=subprocess.PIPE).stdout.decode()
    except Exception:
        return {}
    ends = {}
    for line in output.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or parts[1] in ("", "N/A"):
            continue
        try:
            index, value = int(parts[0]), Decimal(parts[1]) * 1000
        except Exception:
            continue
        if index not in ends or value > ends[index]:
            ends[index] = value
    return ends


def verify_output_file(out_path, master_duration_ms, audio_reports,
                       subtitle_reports, tolerance_ms):
    """L'ACCEPTATION PORTE SUR LE FICHIER, pas sur le compte de pistes.

    `SPEC_ZONE_A.MD` s4d, apres qu'une reparation a rapporte "7 audio et 24
    sous-titres reconstruits, 0 refuse, 0 en echec" ET LIVRE UN FICHIER
    TRONQUE. Le verificateur d'alignement ne pouvait pas le voir: il compare la
    piste qu'il a CONSTRUITE au maitre, sonde a quelques positions choisies, et
    ne regarde jamais la fin du fichier.

    LA REFERENCE EST LA DUREE VIDEO DU MAITRE, et le choix compte. C'est ce que
    le merge impose (`generate_new_file` passe `-t` sur cette valeur), c'est ce
    que l'assemblage vise, ET C'EST LE SEUL CANDIDAT QUE LA REPARATION N'A PAS
    CALCULE: verifier contre la fin de la derniere tranche du plan ferait
    approuver un plan lui-meme tronque -- le controle serait d'accord avec le
    defaut.

    LES SOUS-TITRES SONT VERIFIES PRESENTS ET PAS EN DUREE: la duree d'une
    piste de sous-titres est celle de sa DERNIERE REPLIQUE, qui finit
    legitimement avant le fichier.
    """
    streams, container_ms = probe_output_streams(out_path)
    audio = [s for s in streams if s["codec_type"] == "audio"]
    subtitle = [s for s in streams if s["codec_type"] == "subtitle"]
    problems = []
    if len(audio) != len(audio_reports):
        problems.append(f"{len(audio_reports)} audio track(s) were built and "
                        f"{len(audio)} are in the file")
    if len(subtitle) != len(subtitle_reports):
        problems.append(f"{len(subtitle_reports)} subtitle track(s) were built and "
                        f"{len(subtitle)} are in the file")
    short, unmeasured = [], []
    for stream in audio:
        duration = stream["duration_ms"]
        if duration == None:
            # ON NE SUBSTITUE PAS LA DUREE DU CONTENEUR. `vmsam-ci` a mesure que
            # `format=duration` d'ffprobe differe de la duree du FLUX video et
            # de mediainfo de 31 a 883 ms sur cinq fichiers -- flux video et
            # mediainfo identiques 5 fois sur 5, conteneur different des deux.
            # C'est une TROISIEME quantite, pas une approximation de celle-ci,
            # et la glisser ici ferait exactement ce que les trois vocabulaires
            # de noms de codec ont fait ce matin.
            #
            # Un flux sans duree declaree est donc NON MESURE, pas suppose
            # correct et pas suppose faux.
            unmeasured.append({"index": stream["index"],
                               "language": stream["language"],
                               "reason": "the stream declares no duration; the "
                                         "container's is a different quantity "
                                         "and is not substituted"})
            continue
        delta = duration - Decimal(str(master_duration_ms))
        if tolerance_ms != None and abs(delta) > Decimal(str(tolerance_ms)):
            short.append({"index": stream["index"], "language": stream["language"],
                          "duration_ms": str(duration), "delta_ms": str(delta)})
    if len(short):
        problems.append("track(s) not running to the master's duration: "
                        + "; ".join(f"stream {s['index']} ({s['language']}) "
                                    f"{s.get('delta_ms', s.get('reason'))}" for s in short))
    if len(unmeasured):
        problems.append("track(s) whose duration the file does not state: "
                        + "; ".join(f"stream {u['index']} ({u['language']})"
                                    for u in unmeasured))
    report = {"unmeasured": unmeasured,
              "expected_duration_ms": str(master_duration_ms),
              "expected_duration_source": "master video Duration (mediainfo)",
              "container_duration_ms": str(container_ms) if container_ms != None else None,
              "audio_built": len(audio_reports), "audio_in_file": len(audio),
              "subtitles_built": len(subtitle_reports), "subtitles_in_file": len(subtitle),
              "tolerance_ms": str(tolerance_ms) if tolerance_ms != None else None,
              "streams": [{"index": s["index"], "codec_type": s["codec_type"],
                           "language": s["language"],
                           "duration_ms": str(s["duration_ms"]) if s["duration_ms"] != None else None}
                          for s in streams],
              "problems": problems}
    # LE VERDICT EST INSCRIT, PAS SEULEMENT LA MESURE. `would_refuse` dit ce que
    # le controle FERAIT, pour que le comportement de demain se lise sur
    # l'artefact d'aujourd'hui sans le recalculer contre un seuil que quelqu'un
    # peut avoir change entre-temps. Un verdict recalcule derive.
    report["would_refuse"] = bool(len(problems))
    report["enforcing"] = bool(output_check_enforcing)
    # `measured` EST LA CONDITION DE LEVEE DU DRAPEAU, ET ELLE EST UNE SOMME SUR
    # LES LIGNES DU RUN PLUTOT QU'UN ACCORD A OBTENIR APRES COUP. Le drapeau
    # passe a vingt ARTEFACTS MESURES: des pistes audio presentes ET DES DUREES
    # LUES, pas une fusion qui a produit un fichier.
    #
    # POURQUOI CE N'EST PAS "un artefact existe": "0 tronque sur 0 artefact" est
    # INDEFINI et pas zero, et compter des fichiers plutot que des artefacts
    # ferait reapparaitre le meme trou un cran plus bas -- vingt fichiers que le
    # controle ne sait pas lire satisferaient un N pose sur un denominateur nul
    # dans un autre systeme de coordonnees.
    report["measured"] = bool(len(streams)) and not len(unmeasured) and bool(
        [x for x in streams if x["codec_type"] == "audio"])
    if len(problems) and output_check_enforcing:
        error = chimeric_error("the produced file does not match what was built: "
                               + "; ".join(problems))
        error.output_check = report
        raise error
    return report


def verify_on_master_timeline(out_path, master_obj, audio_reports, pieces,
                              tolerance_ms, search_ms, reference_stream=None):
    """Compare chaque piste reconstruite au maitre et REFUSE si elle n'y est pas.

    Ceci est la forme automatique de `SPEC_ZONE_A.MD` §2 point 3 -- produire la
    cible et la comparer. Ecrit apres qu'une erreur de SIGNE dans un plan ecrit a
    la main a produit une piste decalee de deux fois le decalage de base, avec
    l'escalier pourtant correctement retire: uniforme, donc invisible a tout
    controle par plateau, et visible seulement contre le maitre.

    Quantum-independant, donc compatible avec la contrainte de vmsam-dev-1 de ne
    jamais RE-MESURER un pas a une autre longueur de fenetre: on ne mesure pas un
    pas, on verifie un zero.

    Une piste dont le maitre n'a pas la langue n'est pas verifiable: c'est une
    troisieme issue, `skipped`, et surtout pas un succes.
    """
    probe_plan = choose_probe_positions(pieces, verify_window_seconds)
    positions = [start for _, start in probe_plan]
    if not len(positions):
        return [{"track": None, "outcome": "skipped",
                 "reason": "no candidate-sourced piece long enough to probe"}]

    window_ms = Decimal(str(verify_window_seconds)) * Decimal("1000")
    results = []
    produced_index = 0
    for report in audio_reports:
        language = report["language"]
        master_audio = find_master_audio_for_language(master_obj, language,
                                                     reference_stream)
        if master_audio == None:
            results.append({"track": report["stream_order"], "language": language,
                            "outcome": "skipped",
                            "reason": "the master has no track in this language"})
            produced_index += 1
            continue
        probes = []
        # LE VERIFICATEUR AVAIT LE MEME DEFAUT DE TETE QUE L'ASSEMBLEUR. Il lit
        # la reference par recherche PTS: demander la position 0 sur une piste
        # maitre qui commence a 1.103 s rend du contenu qui commence a 1.103,
        # donc la reference elle-meme est decalee et la piste produite parait
        # fausse d'exactement ce `start_time`. Mesure le 2026-09-03 sur un
        # fichier dont le corps venait de tomber a -5 ms apres la correction de
        # l'assemblage, pendant que la sonde de tete restait a -1103 ms.
        #
        # Une position anterieure au debut du flux maitre N'A PAS DE REFERENCE.
        # On l'avance jusqu'au debut du flux quand la fenetre tient encore dans
        # le meme morceau, sinon on le DIT -- quatrieme issue de sonde, et
        # surtout pas une mesure.
        master_start_ms = get_stream_start_ms(master_audio)
        for piece_index, start in probe_plan:
            probe_start = start
            if master_start_ms > start:
                piece_end = pieces[piece_index]["master_end_ms"]
                shifted = Decimal(str(master_start_ms))
                if shifted + window_ms <= Decimal(str(piece_end)):
                    probe_start = shifted
                else:
                    probes.append({"master_position_ms": str(start),
                                   "piece": piece_index, "outcome": "no_reference",
                                   "reason": f"the master's track starts at "
                                             f"{master_start_ms} ms, after this probe"})
                    continue
            start = probe_start
            reference = read_mono_samples(master_obj.filePath,
                                          f"0:{int(master_audio['StreamOrder'])}",
                                          start, window_ms, verify_probe_rate)
            produced = read_mono_samples(out_path, f"0:a:{produced_index}",
                                         start, window_ms, verify_probe_rate)
            reference_rms = get_rms(reference)
            produced_rms = get_rms(produced)
            if min(reference_rms, produced_rms) < verify_min_rms:
                # Troisieme issue au niveau de la SONDE, pas seulement de la
                # piste: cette fenetre ne porte pas de signal, donc elle ne dit
                # rien -- ni que la piste est calee, ni qu'elle ne l'est pas.
                probes.append({"master_position_ms": str(start), "piece": piece_index,
                               "outcome": "no_signal",
                               "reference_rms": reference_rms,
                               "produced_rms": produced_rms})
                continue
            lag, score = measure_lag_ms(reference, produced, verify_probe_rate, search_ms)
            probes.append({"master_position_ms": str(start), "piece": piece_index,
                           "lag_ms": lag, "correlation": score, "outcome": "measured",
                           "reference_rms": reference_rms,
                           "produced_rms": produced_rms})
        measured = [p for p in probes if p.get("outcome") == "measured"]
        if not len(measured):
            # Toutes les fenetres muettes: la piste est INVERIFIABLE ici. Ce n'est
            # pas un succes, et l'appeler `aligned` serait exactement l'erreur que
            # la campagne poursuit -- un controle qui ne peut pas echouer.
            results.append({"track": report["stream_order"], "language": language,
                            "outcome": "skipped",
                            "reason": "no probe window carried signal; the track "
                                      "is unverified, not verified",
                            "probes": probes})
            produced_index += 1
            continue
        worst = max(abs(probe["lag_ms"]) for probe in measured)
        # DESACCORD A L'INTERIEUR D'UN MEME MORCEAU: le plan dit qu'il n'y a pas
        # de frontiere la, et la mesure dit le contraire. C'est une TROISIEME
        # issue, distincte de "mal cale": la piste peut etre parfaitement calee
        # des deux cotes d'un point de changement que la mesure a manque, auquel
        # cas `worst` seul ne dirait rien.
        inconsistent = []
        for index in sorted({p["piece"] for p in measured}):
            same = [p["lag_ms"] for p in measured if p["piece"] == index]
            if len(same) > 1 and (max(same) - min(same)) > tolerance_ms:
                inconsistent.append({"piece": index, "spread_ms": max(same) - min(same),
                                     "lags_ms": same})
        if len(inconsistent):
            detail = "; ".join(f"piece {c['piece']} probes disagree by "
                               f"{c['spread_ms']:.1f} ms {c['lags_ms']}" for c in inconsistent)
            # LE REFUS EMPORTE SES MESURES. Sans cela, la seule chose qui
            # survit d'un declin est une phrase: les sondes qui l'expliquent --
            # morceau, position maitre, lag, correlation -- sont construites
            # puis jetees. `vmsam-dev-1` a demande ces quatre nombres pour
            # localiser un facteur 23 entre son plan et ma mesure, et il a fallu
            # rejouer le fichier pour les produire. Un refus qui ne peut pas
            # etre diagnostique coute plus cher que le refus lui-meme.
            error = chimeric_error(
                f"the plan says one alignment holds across a piece and the track "
                f"says otherwise: {detail}. That is a change point the measurement "
                f"missed, not a splice error -- the track may be correctly aligned "
                f"on both sides of a boundary nobody modelled")
            error.verification = results + [
                {"track": report["stream_order"], "language": language,
                 "outcome": "inconsistent", "inconsistent": inconsistent,
                 "probes": probes}]
            # Et CE QUI A ETE CONSTRUIT, pas seulement ce qui a ete mesure: un
            # declin sans ses comptes de remplissage rend la distribution de
            # silence mesurable UNIQUEMENT sur les fichiers qui ont reussi, donc
            # un maximum sur les survivants. `vmsam-ci`: les meilleurs cas d'un
            # defaut sont absents de tout echantillon collecte pendant que le
            # defaut agissait.
            error.audios = audio_reports
            raise error
        outcome = "aligned" if worst <= tolerance_ms else "misaligned"
        results.append({"track": report["stream_order"], "language": language,
                        "outcome": outcome, "worst_lag_ms": worst,
                        "probes_measured": len(measured),
                        "probes_without_signal": len(probes) - len(measured),
                        "probes": probes})
        produced_index += 1

    misaligned = [r for r in results if r["outcome"] == "misaligned"]
    if len(misaligned):
        detail = "; ".join(f"track {r['track']} ({r['language']}) off by "
                           f"{r['worst_lag_ms']:.1f} ms" for r in misaligned)
        error = chimeric_error(
            f"the rebuilt track is not on the master's timeline: {detail}. "
            f"Tolerance {tolerance_ms} ms. The plan is wrong, not the splice: a "
            f"uniform offset means the base offset carries the wrong sign, and a "
            f"residual that changes at a change point means a step was missed")
        error.verification = results
        error.audios = audio_reports
        raise error
    return results
