'''
Application d'une relation de vitesse mesuree.

`docs/AUDIO_SPEED_POLICY.MD` est une DECISION, pas une suggestion: `asetrate`,
ni `rubberband`, ni `atempo`, pour PAL comme pour NTSC. Ce module ne rejuge
rien; il applique, et il ecrit ce qu'il a applique.

CONVENTION, sous forme d'equation pour qu'elle ne se lise pas dans les deux
sens:

    speed_ratio = duree_maitre / duree_candidat

Donc PAL contre film lit 1.042709, le candidat va TROP VITE et doit etre
RALENTI: `asetrate = frequence / speed_ratio`. La faiblesse 3 du document est
exactement la: un balayage avait applique `* r` au lieu de `/ r`, atterri deux
fois plus loin, et `rubberband` avait gagne 33 fichiers sur 33 -- signe
constant, magnitudes plausibles, et rien a l'interieur de la campagne ne
pouvait le rattraper.

LE FACTEUR ECRIT DANS LE TAG EST CELUI REELLEMENT APPLIQUE. `asetrate` prend un
entier: 48000/1.042709 vaut 46033.55, donc la frequence posee est arrondie et le
facteur reellement obtenu n'est PAS celui demande. On passe par une frequence
intermediaire haute pour diviser cette erreur, on calcule le facteur effectif
exactement, et c'est lui qui part dans `VMSAM_FABRICATED`. Un tag
`resampled:1.042709` sur une piste etiree par 1.042731 est pire que pas de tag.
'''

from decimal import Decimal, getcontext
import sys

import tools

# `asetrate` ne prend qu'un entier. Poser la frequence cible directement a
# 48000/r arrondirait a 0.5 Hz pres, soit 1.1e-5 en relatif -- 15 ms de derive
# sur un episode de 1420 s. En remontant d'abord a 8x, la meme demi-unite pese
# huit fois moins. Le plafond existe parce qu'une frequence absurde coute du
# temps de calcul sans rien rendre.
speed_intermediate_factor = 8
speed_intermediate_rate_max = 768000


class resample_error(Exception):
    '''La relation de vitesse ne peut pas etre appliquee. Refus explicite.'''
    pass


def get_intermediate_rate(source_rate):
    return min(int(source_rate) * speed_intermediate_factor,
               speed_intermediate_rate_max)


def build_speed_filter_chain(source_rate, speed_ratio):
    '''Renvoie (chaine de filtres, facteur effectif, frequence intermediaire,
    frequence posee).

    Le facteur effectif est `intermediaire / round(intermediaire / ratio)`,
    exactement -- pas le ratio demande.
    '''
    ratio = Decimal(str(speed_ratio))
    if ratio <= 0:
        raise resample_error(f"speed ratio {ratio} is not positive")
    if ratio == 1:
        raise resample_error("speed ratio is exactly 1: nothing to apply")

    source_rate = int(source_rate)
    intermediate = get_intermediate_rate(source_rate)
    getcontext().prec = 28
    target = int((Decimal(intermediate) / ratio).to_integral_value(rounding="ROUND_HALF_EVEN"))
    if target < 1:
        raise resample_error(
            f"speed ratio {ratio} would set the sample rate to {target}")
    effective = Decimal(intermediate) / Decimal(target)
    chain = (f"aresample={intermediate},asetrate={target},"
             f"aresample={source_rate}")
    return chain, effective, intermediate, target


def get_drift_after_correction_ms(effective_ratio, requested_ratio, duration_ms):
    '''Ce que la quantisation d'`asetrate` laisse comme derive, en ms.

    Se dit avant le run, pas apres: c'est le residu que la correction NE corrige
    pas, et il doit rester petit devant un cadre video (41.7 ms a 23.976 fps).
    '''
    return abs(Decimal(str(effective_ratio)) - Decimal(str(requested_ratio))) \
        * Decimal(str(duration_ms))


def format_factor(effective_ratio, digits=6):
    '''La precision ecrite dans le tag. On n'ecrit pas plus de chiffres qu'on
    n'en tient: le facteur effectif est exact en tant que rapport d'entiers,
    mais le tag est une chaine et six decimales suffisent a le distinguer de
    tout autre facteur plausible du corpus.
    '''
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(str(effective_ratio)).quantize(quantum))


def retime_subtitle_events_by_ratio(subtitles, speed_ratio):
    '''Les sous-titres subissent le MEME coefficient que l'audio.

    `CAMPAIGN.MD` et le brief sont explicites: on prend les deux. Une piste
    audio recalee et des sous-titres laisses en place produisent un fichier qui
    a l'air correct et qui derive d'une seconde par demi-heure.
    '''
    ratio = Decimal(str(speed_ratio))
    for event in subtitles.events:
        event.start = int((Decimal(event.start) * ratio).to_integral_value())
        event.end = int((Decimal(event.end) * ratio).to_integral_value())
    return subtitles


def describe(source_rate, speed_ratio, duration_ms):
    '''Ce que l'application fera, AVANT de la faire. Sert au journal et au
    rapport: annoncer le resultat attendu avant le run est ce que
    `AGENT.MD` demande.
    '''
    chain, effective, intermediate, target = build_speed_filter_chain(
        source_rate, speed_ratio)
    return {"requested_ratio": str(Decimal(str(speed_ratio))),
            "effective_ratio": str(effective),
            "tag_factor": format_factor(effective),
            "intermediate_rate": intermediate,
            "asetrate_target": target,
            "filter": chain,
            "residual_drift_ms": str(get_drift_after_correction_ms(
                effective, speed_ratio, duration_ms))}

def iter_audio_dicts(video_obj):
    """Toutes les pistes audio de l'objet, dans l'ordre du conteneur."""
    audios = []
    for holder in (video_obj.audios, video_obj.commentary, video_obj.audiodesc):
        for language, tracks in holder.items():
            for audio in tracks:
                audios.append(audio)
    return sorted(audios, key=lambda a: int(a["StreamOrder"]))


def build_resampled_candidate(candidate_obj, speed_ratio, out_path, timeout=3600):
    '''Le candidat ENTIER avec son audio deja reechantillonne, ecrit comme fichier.

    POURQUOI CETTE FONCTION EXISTE. La mesure ne peut pas produire de plan sur
    la paire d'ORIGINE: a 4.27 % le correlateur s'effondre, ce qui est
    exactement pourquoi les pas de coupe du dossier 110 sont restes invisibles
    jusqu'au 2026-09-03. L'ordre pour un fichier a relation de vitesse est donc
    reechantillonner, PUIS localiser, PUIS assembler -- et localiser demande un
    FICHIER, pas un graphe de filtres interne. `vmsam-dev-1` prend celui-ci
    comme candidat et son module ne change pas.

    Video et sous-titres sont COPIES: seul l'audio subit la relation, et le
    reechantillonnage se fait en FLAC pour ne pas ajouter une generation de
    codec a une mesure. Les etiquettes de langue survivent par `-map 0`.

    Renvoie (chemin, facteur applique, liste des pistes vues).
    '''
    audios = iter_audio_dicts(candidate_obj)
    if not len(audios):
        raise resample_error("the candidate carries no audio track to resample")
    command = [tools.software["ffmpeg"], "-y", "-nostdin",
               "-analyzeduration", "1000M", "-probesize", "1000M",
               "-i", candidate_obj.filePath, "-map", "0",
               "-c", "copy", "-c:a", "flac"]
    applied = None
    seen = []
    for index, audio in enumerate(audios):
        rate = audio.get("SamplingRate")
        if rate == None:
            raise resample_error(
                f"stream {audio.get('StreamOrder')} has no sampling rate: "
                f"the applied factor could not be stated for it")
        chain, effective, _, _ = build_speed_filter_chain(rate, speed_ratio)
        # Le facteur EFFECTIF depend de la frequence source, donc deux pistes a
        # des frequences differentes ne subissent pas exactement le meme
        # coefficient. On le dit plutot que d'en écrire un seul.
        if applied == None:
            applied = effective
        elif effective != applied:
            applied = None if applied == "mixed" else "mixed"
        command.extend([f"-filter:a:{index}", chain])
        seen.append({"stream_order": int(audio["StreamOrder"]),
                     "sampling_rate": str(rate),
                     "applied_factor": format_factor(effective)})
    command.append(out_path)
    tools.launch_cmdExt_with_timeout_reload(command, 2, timeout)
    return out_path, applied, seen
