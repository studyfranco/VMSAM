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

WHERE THIS MODULE SITS IN THE CYCLE -- put here because a governing document a
seat has to be TOLD to read is weaker than a docstring it cannot avoid, and
because after a reset a successor has only the files. `PIPELINE.MD` is the
interim form of the same relationship; if the two ever disagree, MEASURE THE
CODE, and the code is this file.

    THERE IS ONE CYCLE AND THE REPAIR IS A DETOUR INSIDE IT. Most files
    correlate on a SINGLE CONSTANT DELAY, applied as a container offset -- no
    re-encode, and bitmap subtitles survive. THIS PATH EXISTS ONLY FOR FILES
    WHERE NO SINGLE DELAY HOLDS. The file LEAVES the normal flow at the refusal
    and RE-ENTERS AT THE POINT IT LEFT, hung on
    `best_video.sameAudioMD5UseForCalculation`. DOWNSTREAM -- track choice,
    subtitles, the merge command -- NOTHING DISTINGUISHES A REPAIRED FILE FROM
    A NORMAL ONE, and that is the point.

    THE RESAMPLE IS APPLIED BEFORE THE CUTTING. They are not two routes and not
    parallel boxes:

        [0:candidate] -> speed_chain -> [spd] -> asplit -> pieces -> concat

    THE SLOPE IS UNDERNEATH, THE STAIRCASE ON TOP. The candidate's `start_time`
    is multiplied by the ratio FIRST and every slice time is then read on the
    ALREADY-RESAMPLED candidate. `SPEC_ZONE_A.MD` s4 orders the marker
    `chimeric+resampled` for exactly this reason. Built at the two lines that
    prove it, not asserted: the speed chain is appended and `candidate_entry`
    is rebound to `[spd]` BEFORE the `asplit` that makes the pieces.

    A CONSTANT-OFFSET-NO-GAP PLAN IS REFUSED HERE, AND THAT REFUSAL MEANS
    "GOES BACK TO THE CHEAP PATH", NOT "CANNOT BE HELPED". Two sentences a
    successor must never merge. The reasoning is below, at the refusal itself.

    AND `normalize_segments` REFUSES RATHER THAN CLAMPS -- not by discipline:
    there is NO CLAMPING MACHINERY IN IT AT ALL. A gap in the plan becomes a
    MASTER piece, never emptiness.

Marquage: SPEC_ZONE_A.MD s4. La chaine est posee comme tag de piste Matroska
`VMSAM_FABRICATED` sur le fichier produit ici. Mesure 2026-09-03: ce tag survit
a la premiere passe ffmpeg (`-c copy -map_metadata 0`), a la seconde, au
`mkvmerge --no-global-tags` du split et au `mkvmerge` final.
'''

from decimal import Decimal
import os
from os import path, replace as replace_file
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


def delay_in_ms(track):
    """`Delay` en millisecondes, AVEC SON UNITE VERIFIEE CONTRE UN SECOND OUTIL.

    `mediainfo` rend `Delay` en SECONDES sur le binaire de cette image -- 0.083
    contre un `start_time` de 0.083000, mesure par vmsam-dev-3 sur 13 cas non nuls,
    zero correspondant a une lecture en millisecondes. Le `* 1000` d'ici est donc
    correct.

    MAIS C'EST UNE PROPRIETE DU BINAIRE, PAS DU SCHEMA JSON. Un build qui emettrait
    des millisecondes rendrait cette ligne fausse d'un FACTEUR 1000, EN SILENCE, et
    rien dans `src/` ne croisait une unite avec un second outil.

    Or `ffprobe.start_time` est deja attache au MEME dict, en secondes par
    definition, et dev-3 a mesure 291/291 accords sur les Delay de source
    `Container`. Le controle est donc GRATUIT: on ne devine pas l'unite, on la
    compare a une mesure qui n'en a qu'une.

    Trois issues, et la troisieme n'est pas un defaut:
        les deux concordent en secondes    -> on multiplie, unite confirmee
        elles concordent si Delay est en ms -> ON LEVE. Le binaire a change d'unite
                                               et tout ce qui suit serait faux de 1000.
        pas de `start_time`                 -> on multiplie et on ne peut pas verifier;
                                               c'est l'hypothese documentee, pas une mesure.
    """
    raw = track.get("Delay", 0) or 0
    try:
        seconds = Decimal(str(raw))
    except Exception:
        return Decimal("0")
    probe = (track.get("ffprobe") or {}).get("start_time")
    if probe not in (None, "", "N/A"):
        try:
            start = Decimal(str(probe))
        except Exception:
            return seconds * Decimal("1000")
        # On ne compare que si l'un des deux est non nul: a zero les deux lectures
        # sont identiques et le controle ne dit rien.
        if seconds != 0 or start != 0:
            as_seconds = abs(seconds - start)
            as_ms = abs(seconds / Decimal("1000") - start)
            if as_ms < as_seconds:
                raise chimeric_error(
                    f"mediainfo Delay {raw} matches ffprobe start_time {probe} only "
                    f"if it is in MILLISECONDS, not seconds. This module multiplies "
                    f"by 1000 and would be wrong by that factor. THIS IS A STATEMENT "
                    f"ABOUT THE mediainfo BUILD, not about the media")
    return seconds * Decimal("1000")


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


def offset_is_measured(segment, stream_order=None):
    """Ce flux a-t-il SON PROPRE decalage, ou emprunte-t-il celui d'une autre?

    La table par flux ne couvre QUE les flux de la langue mesuree -- le
    locator la construit sur `candidate_streams`, c'est-a-dire les flux de
    cette langue-la, et c'est correct dans SON contrat. Mais l'assembleur
    applique le meme plan a TOUTES les langues, donc chaque piste d'une autre
    langue retombe sur `candidate_offset_ms`, LE DECALAGE D'UNE AUTRE LANGUE.

    Mesure, id 47, meme paire, meme plan, langue de mesure changee:
        en mesure le flux 2 -> -983.54    fr mesure le flux 1 -> -959.48
        tables DISJOINTES; la piste que l'autre langue a mesuree emprunte,
        et porte 24.06 ms (segment 0) et 14.50 ms (segment 1) d'erreur.
        id 52: 31.11 et 32.35 ms.

    CORRECTION: ids 47 ET 52 SONT LE MEME DOSSIER -- meme maitre, meme dossier
    candidat. Je les ai cites comme DEUX fichiers, ici et dans un message de
    commit ("a second file gives..."), ce qui les fait lire comme une
    corroboration independante. C'EST UNE OBSERVATION, ECHANTILLONNEE DEUX FOIS.

    La mesure qui porte vraiment, elle, vient de quatre dossiers distincts, sur
    des sorties PRODUITES et non sur des plans (T67):
        2c611366b7  en 34.62 et 35.25 ms
        854984999c  fr 14.75 ms
        b39a89aa87  en 0.00 et fr 10.62 ms   <- MEME FICHIER, DEUX COUTS
        80e29f2b4d  fr 1.12 ms
    Le cout de l'emprunt est PAR PISTE et imprevisible: entre 0 et 35 ms, et
    rien dans une piste n'annonce dans quelle moitie elle se trouve.

    C'est SOUS la tolerance de 100 ms du verificateur, donc livrable en
    silence -- exactement la forme des 27.6 ms que `vmsam-dev-1` a trouves
    entre deux pistes jpn, un cran plus haut: regle DANS une langue, ouvert
    ENTRE les langues.

    On ne devine pas le bon decalage ici: on dit lequel des deux on a
    applique. Un repli tacite est une mesure que personne ne lit.
    """
    by_stream = segment.get("candidate_offset_ms_by_stream")
    if not by_stream or stream_order is None:
        return False
    return any(key in by_stream
               for key in (stream_order, str(stream_order), int(stream_order)))


def pairing_fidelity(stream_pairing, stream_order=None):
    """La fidelite du choix PAR FICHIER, celle a laquelle la barre a ete appliquee.

    DEUX NOMBRES, DEUX QUESTIONS, et `vmsam-dev-1` a vu la confusion avant moi.

      par fichier   `candidate_stream_pairing[flux]` -- UN partenaire maitre
                    choisi une fois pour le fichier, et la fidelite de ce choix.
                    C'est LE nombre auquel la barre s'applique, donc le nombre
                    qu'un REFUS doit citer.

      par tranche   `candidate_offset_fidelity_by_stream[flux]` dans chaque
                    segment -- la fidelite de CETTE sonde-la. Diagnostique.
                    C'est le nombre que la ligne de POSE doit citer, parce qu'il
                    decrit la mesure reellement utilisee pour poser ce
                    morceau-la.

    Les confondre ferait citer un nombre pour deux questions differentes. Ma
    premiere version le faisait: elle lisait le per-segment et s'en servait
    aussi dans le refus.

    Un flux ABSENT de la table par fichier n'a passe la barre avec AUCUN
    partenaire. Absent, jamais `fidelity: 0.0` -- des deux cotes du contrat.
    """
    if not stream_pairing or stream_order is None:
        return None
    for key in (stream_order, str(stream_order), int(stream_order)):
        if key in stream_pairing:
            entry = stream_pairing[key]
            if isinstance(entry, dict):
                return entry.get("fidelity")
            return entry
    return None


def offset_fidelity(segment, stream_order=None):
    """La fidelite de l'appariement de ce flux, quand la mesure la donne.

    `vmsam-dev-1` a mesure la barre sur 127 paires de 25 fichiers et a trouve un
    CHEVAUCHEMENT: la meilleure paire INTER-langue atteint 0.8477 et la plus
    basse paire MEME-etiquette qui a l'air vraie est a 0.8196. AUCUN SEUIL NE
    LES SEPARE PROPREMENT. 0.85 est donc un choix DANS un chevauchement, pas une
    frontiere.

    Consequence directe pour ce fichier-ci, et c'est leur formulation: A LA
    BARRE, UN REFUS N'EST PAS LA PREUVE QUE LA PISTE EST NON MESURABLE -- C'EST
    LA PREUVE QU'ON N'A PAS PU LA MESURER ASSEZ BIEN POUR EN ETRE SUR. Les deux
    ne sont pas la meme chose et le journal doit pouvoir dire laquelle.

    Renvoie None quand la mesure ne porte pas de fidelite -- ce qui est le cas
    de TOUS les plans aujourd'hui, la table par flux ne l'emettant pas encore.
    None se journalise alors comme absent, jamais comme zero: une fidelite
    inconnue n'est pas une fidelite nulle.
    """
    table = segment.get("candidate_offset_fidelity_by_stream")
    if not table or stream_order is None:
        return None
    for key in (stream_order, str(stream_order), int(stream_order)):
        if key in table:
            return table[key]
    return None


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
            # POURQUOI CE MORCEAU DE MAITRE EST LA, ET C'EST L'ARCHITECTE QUI A
            # MONTRE QUE PERSONNE NE POUVAIT LE DIRE.
            #
            # Un morceau maitre insere a TROIS causes possibles et le journal les
            # ecrivait toutes de la meme facon:
            #
            #   head_gap          rien avant le premier segment. Soit le candidat
            #                     n'a rien la (decalage negatif), soit un segment
            #                     a ete JETE comme inutilisable -- deux faits
            #                     opposes qui produisaient la meme ligne.
            #   interior_bracket  le trou EST l'incertitude du localisateur entre
            #                     deux plateaux; sa largeur est `bracket_high -
            #                     bracket_low`.
            #   tail_gap          rien apres le dernier segment.
            #
            # LA DISTINCTION EST LA QUESTION DU PROPRIETAIRE. Sa regle dit de
            # RETIRER l'exces de tete et de queue et de GARDER ce qui tombe dans
            # la portee du maitre. Un lecteur ne pouvait pas verifier laquelle des
            # trois s'etait produite, donc ne pouvait pas voir qu'une substitution
            # avait remplace du materiel candidat par du maitre.
            pieces.append({"source": "master", "master_start_ms": cursor,
                           "master_end_ms": master_start,
                           "source_start_ms": cursor,
                           "reason": "head_gap" if cursor == 0 else "interior_bracket"})
        pieces.append({"source": "candidate", "master_start_ms": master_start,
                       "master_end_ms": master_end,
                       "source_start_ms": candidate_start})
        cursor = master_end

    if cursor < master_duration_ms:
        pieces.append({"source": "master", "master_start_ms": cursor,
                       "master_end_ms": master_duration_ms,
                       "source_start_ms": cursor,
                       "reason": "tail_gap"})
    return pieces


def preferred_with_disagreement(name, preferred, other, preferred_tool, other_tool):
    """Rend la valeur PREFEREE et, si l'autre outil la contredit, LE DIT.

    `a or b` RESOUT UN DESACCORD EN SILENCE. Tant que les deux outils sont
    d'accord, la chaine de repli et la preference sont indistinguables; quand ils
    divergent, la chaine choisit sans que rien ne l'ecrive nulle part.

    LE MECANISME TIENT SANS AUCUNE INSTANCE. Un `or` ne peut pas distinguer un
    repli d'un choix, donc le jour ou les outils divergent la chaine tranche et
    rien ne l'ecrit. C'est vrai que le corpus contienne trois cas ou zero.

    LES INSTANCES, ELLES, SONT SOUS RESERVE ET NE SONT PAS ENCORE UNE MESURE.
    `vmsam-dev-3` rapporte sur 1 178 pistes de 500 fichiers -- SamplingRate
    44100/48000 une fois, Channels 1/2 deux fois -- PUIS A TROUVE UN TROU DANS
    SA PROPRE PREMISSE: sa comparaison fait correspondre le `StreamOrder` de
    mediainfo a l'index de flux d'ffprobe, et si cette correspondance se
    transpose sur un fichier a deux pistes audio de frequences differentes,
    IL COMPARE DEUX PISTES DIFFERENTES ET LES DEUX OUTILS ONT RAISON.
    44100 contre 48000 est exactement l'allure d'une paire echangee.

    Ce qu'il avait verifie -- 0 desaccord de TYPE sur 1 378 pistes -- ne couvre
    pas ce cas: une correspondance juste par type peut transposer deux pistes du
    MEME type. Son controle tourne (codec_name d'ffprobe contre Format de
    mediainfo sur les pistes en desaccord); s'ils divergent aussi, la
    correspondance est fausse et les instances disparaissent.

    ON N'ECRIT DONC PAS SES TROIS LIGNES COMME OBSERVEES. Le champ emis nomme un
    desaccord QUAND IL S'EN PRODUIT UN, et ne cite aucun taux.

    ET LES DEUX MOITIES DU PIPELINE NE LISENT PAS LE MEME OUTIL. `video.py`
    `get_less_sampling_rate` lit MEDIAINFO et alimente le clamp de
    `mergeVideo.py:567-569`, donc la grille de comparaison du chemin de decalage.
    Ce module-ci prefere FFPROBE et alimente le filtergraph. Sur cette piste-la
    le decalage se mesure sur une grille a 44100 pendant que le filtre tourne a
    48000 -- pas deux regles qui rendent le meme nombre par deux routes, MAIS
    DEUX OUTILS QUI RENDENT DES NOMBRES DIFFERENTS, chaque moitie lisant celui
    qui l'arrange.

    ON NE CHANGE PAS LA PREFERENCE. ffprobe est ce que ffmpeg lira lui-meme, donc
    c'est la bonne valeur POUR LE FILTRE. Ce qui manquait n'est pas le choix,
    c'est que le choix soit VISIBLE quand il en est un.

    ET LE JETON N'ARBITRE PAS, PARCE QUE LES DEUX OUTILS PEUVENT AVOIR RAISON.
    Ma premiere version ecrivait `CONTRADICTED_BY`, qui affirme que l'un des deux
    se trompe. `vmsam-dev-3` a localise la piste et la cause:

        ffprobe    profile HE-AAC, channels 2, layout stereo
        mediainfo  Format_AdditionalFeatures "LC SBR PS", Channels 1
        le DECODEUR rend exactement 2.00 canaux

    HE-AAC v2 en Parametric Stereo code un NOYAU MONO plus des parametres
    stereo. mediainfo rapporte le nombre de canaux CODES, ffprobe le nombre
    DECODES. LES DEUX SONT JUSTES. `Channels` nomme une quantite dans un outil et
    une autre dans le second, et elles coincident pour tout codec non
    parametrique -- c'est pourquoi 1 176 pistes sur 1 178 s'accordaient.

    LE DEFAUT N'EST DONC PAS DANS LES OUTILS, IL EST DANS L'HYPOTHESE QU'UN NOM
    DE CHAMP DESIGNE UNE SEULE QUANTITE. Un jeton qui accuse envoie un lecteur
    chercher un bogue qui n'existe pas. On ecrit donc les deux valeurs, l'outil
    de chacune, et CELLE QUI A SERVI -- et rien d'autre. Le lecteur arbitre s'il
    le veut; le journal ne le fait pas a sa place.

    LES TROIS INSTANCES DU CORPUS SONT LOCALISEES ET VERIFIEES PAR DECODAGE, ET
    AUCUNE N'EST UN DESACCORD:

        HE-AAC v2 / Parametric Stereo   mediainfo Channels 1, ffprobe 2,
                                        le decodeur rend 2.00 -> CODE contre DECODE
        OPUS                            mediainfo SamplingRate 44100, ffprobe 48000,
                                        le decodeur rend 48000 -> Opus decode
                                        TOUJOURS a 48 kHz et l'en-tete porte le
                                        taux d'ENTREE d'ORIGINE

    POURQUOI LE JETON NE NOMME PAS LES QUANTITES, ET C'EST UN DESACCORD ASSUME
    AVEC LA PROPOSITION DE dev-3. Il suggerait `48000(ffprobe,decoded)` contre
    `44100(coded/original)`. Ce serait AFFIRMER POUR TOUT CODEC une semantique
    verifiee sur DEUX. "ffprobe rend ce que le decodeur fera" est vrai ici et
    plausible en general, et la nature exacte de l'autre quantite CHANGE d'un cas
    a l'autre -- canaux CODES pour PS, taux d'entree D'ORIGINE pour Opus. Ce
    n'est pas une seule opposition, ce sont deux.

    Un jeton qui nommerait la quantite generiquement referait la faute qu'il
    signale: un seul nom pose sur des quantites differentes. On ecrit donc les
    deux valeurs, les deux outils et celle qui a servi; les mecanismes sont ici,
    ou ils sont attribues et bornes a ce qui a ete decode.

    ET dev-3 A CORRIGE SA PROPRE HYPOTHESE: il avait suppose que le cas de
    FREQUENCE venait du SBR qui double le taux de sortie. C'est bien la meme
    collision et le mecanisme est le decodage fixe a 48 kHz d'Opus. Une
    conclusion juste avec un mecanisme faux, evitee parce qu'il l'avait marquee
    non testee.

    AUCUN TAUX N'EST CITE.
    """
    if preferred == None:
        return other, None
    if other == None or str(preferred) == str(other):
        return preferred, None
    return preferred, (f"{name} {preferred_tool}={preferred} "
                       f"{other_tool}={other} used={preferred_tool}")


def encoding_feature_context(audio):
    """Les champs BRUTS qui expliquent un ecart, recopies sans interpretation.

    `vmsam-dev-3` a stratifie 1 178 pistes par (codec, feature d'encodage) et le
    discriminant separe parfaitement:

        HE-AAC | LC_SBR      ch=2   x87   d'accord
        HE-AAC | LC_SBR_PS   ch=1   x2    SEPARENT      2 sur 2
        OPUS   | sr=48000    ch=2   x21   d'accord
        OPUS   | sr=44100    ch=2   x1    SEPARE        1 sur 1

    LA COLLISION EST DETERMINISTE ETANT DONNE LA FEATURE D'ENCODAGE, PAS LE
    CODEC. HE-AAC v1 a un noyau STEREO et les deux outils disent 2; seule la v2
    en Parametric Stereo code un noyau mono qui decode en stereo. 93 pistes
    HE-AAC sur 95 s'accordent.

    ET LE DISCRIMINANT N'EST PAS DANS L'OUTIL QUE JE PREFERE. Le `profile`
    d'ffprobe rend `HE-AAC` pour la v1 comme pour la v2: il ne peut pas
    distinguer les 2 pistes qui separent des 93 qui ne separent pas. C'est
    `Format_AdditionalFeatures` de mediainfo, contenant `PS`, qui le fait.

    ON RECOPIE DONC LES DEUX CHAMPS TELS QUELS. Pas d'interpretation, pas de
    verdict, pas de "ceci est du Parametric Stereo" -- les octets des deux
    outils, pour qu'un lecteur du journal puisse distinguer ce que la valeur
    seule ne dit pas. C'est de la DONNEE, pas un arbitrage, et c'est ce qui la
    distingue du `CONTRADICTED_BY` que j'ai retire.

    Les espaces deviennent des `_`: la ligne se lit en `cle=valeur` separees par
    des espaces, et `LC SBR PS` casserait cette lecture chez `vmsam-dev-4`.
    """
    parts = []
    profile = audio.get("ffprobe", {}).get("profile")
    features = audio.get("Format_AdditionalFeatures")
    if profile not in (None, ""):
        parts.append(f"ffprobe_profile={str(profile).replace(' ', '_')}")
    if features not in (None, ""):
        parts.append(f"mediainfo_features={str(features).replace(' ', '_')}")
    return " ".join(parts)


def get_audio_stream_parameters(audio):
    '''Frequence, canaux et layout de la piste, tels que ffmpeg les nommera.

    Rend un quatrieme element: les DESACCORDS entre outils, vides quand il n'y en
    a pas. Vide et non `None`: "aucun desaccord" et "on n'a pas regarde" ne
    doivent pas imprimer le meme jeton.
    '''
    ffprobe_data = audio.get("ffprobe", {})
    sample_rate, rate_note = preferred_with_disagreement(
        "sample_rate", ffprobe_data.get("sample_rate"), audio.get("SamplingRate"),
        "ffprobe", "mediainfo")
    channels, channel_note = preferred_with_disagreement(
        "channels", ffprobe_data.get("channels"), audio.get("Channels"),
        "ffprobe", "mediainfo")
    layout = ffprobe_data.get("channel_layout")
    if sample_rate == None or channels == None:
        raise chimeric_error(
            f"track {audio.get('StreamOrder')} has no sampling rate or channel count")
    if layout == None or layout == "" or "unknown" in str(layout):
        layout = f"{int(channels)}c"
    # LE CONTEXTE N'EST ATTACHE QU'AUX NOTES, DONC SEULEMENT QUAND IL Y A UN
    # ECART. Sur 1 175 pistes qui s'accordent il n'apparait pas du tout.
    context = encoding_feature_context(audio)
    notes = [note for note in (rate_note, channel_note) if note != None]
    if context:
        notes = [f"{note} {context}" for note in notes]
    return str(sample_rate), int(channels), str(layout), notes


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
    # `head_pad_ms=0` COUVRAIT TROIS SITUATIONS DIFFERENTES ET LES ECRIVAIT AVEC
    # LE MEME CHIFFRE. vmsam-dev-3, sur mes octets: sur 39 pistes observees,
    # 11 portaient 0, et rien dans le journal ne dit laquelle des trois:
    #
    #   stream_start_ms == None   -> ON N'A PAS MESURE le debut du flux
    #   missing <= 0              -> le flux commence apres zero MAIS le plan lit
    #                                deja au-dela: un decalage EXISTE et ne coute
    #                                aucun rembourrage
    #   missing == 0 a l'origine  -> le flux commence vraiment a zero
    #
    # C'est ma propre regle -- une valeur et son absence ne doivent pas ecrire le
    # meme jeton -- dans un champ de mon module. Et c'est la raison pour laquelle
    # personne ne peut aujourd'hui confirmer NI refuter une hypothese de decalage
    # de conteneur a partir du journal.
    #
    # La decision se dit maintenant a cote du nombre. Le nombre ne change pas.
    head_decisions = []

    def head_pad(source_start_ms, stream_start_ms, sink):
        if stream_start_ms == None:
            head_decisions.append({"outcome": "unmeasured",
                                   "stream_start_ms": None, "missing_ms": None})
            return ""
        missing = Decimal(str(stream_start_ms)) - Decimal(str(source_start_ms))
        if missing <= 0:
            head_decisions.append({
                # LE DECALAGE EXISTE ET LE PLAN LE LIT AU-DELA. Ce n'est pas
                # "pas de decalage", et l'ecrire 0 le faisait lire ainsi.
                "outcome": "read_past" if Decimal(str(stream_start_ms)) > 0 else "none",
                "stream_start_ms": str(stream_start_ms), "missing_ms": str(missing)})
            return ""
        sink.append(missing)
        head_decisions.append({"outcome": "padded",
                               "stream_start_ms": str(stream_start_ms),
                               "missing_ms": str(missing)})
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
    return (";".join(chains), sum(pads) if len(pads) else Decimal("0"),
            head_decisions)


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


def same_language_principal_count(master_obj, language):
    """Combien de pistes PRINCIPALES le maitre porte-t-il dans cette langue?

    Plus d'une, et le choix du remplissage se fait entre des pistes que
    L'ETIQUETTE NE SEPARE PAS. `commentary` et `audiodesc` sont des holders
    distincts, donc ce compte ne voit que des pistes principales -- deux
    doublages, pas une piste et son commentaire.

    MESURE, ET CE N'EST PLUS UNE INQUIETUDE: sur le maitre de id 56, les deux
    pistes espagnoles principales correlent a 0.6321 l'une contre l'autre sur
    trois positions -- LE MEME REGIME QUE DEUX LANGUES DIFFERENTES, mediane
    0.6108 sur 87 paires inter-langues. Ce sont bien deux doublages distincts.
    Elles sont aussi DECALEES DE 21.3 ms L'UNE DE L'AUTRE, constant sur tout le
    fichier a 0.1 ms pres, soit exactement une trame AAC a 48 kHz.

    DONC REMPLIR DEPUIS LE MAUVAIS DOUBLAGE COUTE 21.3 ms. Sous la tolerance de
    100 ms du verificateur, sous une trame video, inaudible comme decalage et
    invisible comme defaut. Mesure par `vmsam-dev-1` sur un fichier, un couple
    de flux, trois positions.

    ET IL N'Y A AUCUN REPLI PAR ETIQUETTE: sur 290 pistes de 44 maitres, ZERO
    porte une etiquette de langue avec un tiret. Les deux doublages sont tagues
    'es' tous les deux, meme codec, meme nombre de canaux. Les seuls champs qui
    different sont le TITRE, texte libre, et le CONTENU, que seule une mesure
    atteint.

    On ne devine donc pas ici. On COMPTE, et le journal dit qu'il y avait un
    choix a faire.
    """
    tracks = (master_obj.audios or {}).get(language) or []
    return len(tracks)


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
                          reference_stream=None, comparison_language=None,
                          candidate_duration_ms=None):
    '''Produit une piste audio chimerique. Renvoie un dict de compte-rendu.'''
    codec_name = audio.get("ffprobe", {}).get("codec_name", "").lower()
    encoder_arguments, family, bitrate_origin = get_encoder_arguments(
        audio, codec_name, candidate_obj.filePath)
    sample_rate, channels, layout, tool_disagreements = get_audio_stream_parameters(audio)

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
    # COMBIEN DE PISTES PRINCIPALES LE MAITRE PORTAIT-IL DANS LA LANGUE DU
    # REMPLISSAGE? Plus d'une, et le choix s'est fait entre des pistes que
    # l'etiquette ne separe pas -- mesure: deux doublages espagnols du meme
    # programme correlent a 0.63 et sont a 21.3 ms l'un de l'autre.
    fill_choices = (same_language_principal_count(master_obj, fill_language)
                    if fill == "master" and fill_language else 0)
    # LA SOURCE DE REMPLISSAGE EST-ELLE ASSEZ LONGUE POUR LES TROUS QU'ON LUI
    # DEMANDE? Rien ne le verifiait. `find_master_audio_for_language` choisit par
    # LANGUE et deux pistes du meme maitre n'ont pas la meme duree.
    #
    # MESURE, sur un artefact reel: six pistes remplies depuis master/ja, une
    # depuis master/fr, comportement par langue CORRECT -- et la piste francaise
    # du maitre est 2008 ms PLUS COURTE que sa japonaise. La piste produite a
    # herite le manque, QUATRE FOIS la tolerance, et rien ne les a comparees.
    #
    # C'est le second danger de cette fonction. Le premier etait le TIMING --
    # deux pistes de la meme langue a 126-138 ms l'une de l'autre. Celui-ci est
    # la LONGUEUR, et il est quinze fois plus grand.
    #
    # ON MESURE ET ON DIT, ON NE REFUSE PAS: le proprietaire a tranche que
    # l'emprunt continue, et jeter une piste parce que sa source de remplissage
    # est courte serait la meme decision produit prise unilateralement.
    # SPEC_ZONE_A s4g -- LA TETE. "Prendre du maitre SI LE MAITRE L'A, sinon du
    # silence, pour que la sortie commence a 0 du maitre."
    #
    # "L'A" SE MESURE, IL NE SE DEDUIT PAS D'UNE DUREE. J'avais demande si une
    # piste plus COURTE que sa soeur devait servir la tete, et la question etait
    # mal posee: COURTE-A-LA-FIN ET ABSENTE-A-LA-TETE SONT DEUX FAITS
    # DIFFERENTS, et rien dans une duree ne dit lequel. Une piste de 2008 ms plus
    # courte peut manquer a la fin, a la tete, ou au milieu.
    #
    # Donc on SONDE la tete de la piste choisie. Si elle porte du signal, on la
    # prend -- langue d'abord, coherent avec la regle de remplissage. Sinon on
    # retombe: langue de comparaison, puis silence.
    head_piece = next((p for p in pieces
                       if p["source"] == "master" and p["master_start_ms"] == 0), None)
    head_source = None
    if head_piece != None and fill == "master" and master_audio != None:
        span = min(Decimal("20000"),
                   head_piece["master_end_ms"] - head_piece["master_start_ms"])
        try:
            samples = read_mono_samples(
                master_obj.filePath, f"0:{int(master_audio['StreamOrder'])}",
                Decimal("0"), span, verify_probe_rate)
            # LE REPLI N'EST PAS ENCORE IMPLEMENTE. Le graphe de filtres prend UN
            # flux maitre pour TOUS les morceaux maitre, donc servir la tete
            # depuis une autre piste demande une source PAR MORCEAU. Tant que ce
            # n'est pas fait, la tete vient de cette piste MEME SI ELLE EST
            # MUETTE, et `NO-HEAD` le dit sur la ligne au lieu de le taire.
            # LA CONDITION *ET* SON CONSEQUENT. `NO-HEAD` seul nomme le fait et
            # PAS le manque -- et un champ qui enregistre une condition sur
            # laquelle le code n'agit pas n'est honnete QUE si l'ecart est nomme.
            # C'est la moitie "pourquoi cela compte" de s4e, celle que j'ai
            # trouvee absente dans le journal du merge ce matin. Meme defaut,
            # chez moi.
            head_source = ("master/" + str(fill_language)
                           if get_rms(samples) >= verify_min_rms
                           else f"NO-HEAD(taken from master/{fill_language} "
                                f"anyway -- fall-through not implemented, "
                                f"this head is silent)")
        except Exception:
            # PAS de supposition: une tete illisible n'est pas une tete absente.
            head_source = "unprobed"
    elif head_piece != None:
        head_source = "silence"

    fill_source_ms = None
    fill_short_by_ms = None
    if fill == "master" and master_audio != None and "Duration" in master_audio:
        try:
            # UN POINT DE FIN MOINS UNE DUREE N'EST PAS UN MANQUE.
            #
            # `master_end_ms` est une POSITION sur la timeline du maitre, qui
            # commence a zero. `Duration` de mediainfo est une DUREE -- mesure de
            # `vmsam-forensic`: sur un flux demarrant a 1.103 s, Duration vaut
            # 1439.949 contre un span de 1439.926, soit la longueur de la
            # derniere trame, et PAS l'endpoint 1441.029. mediainfo expose le
            # decalage separement, en `Delay`.
            #
            # Soustraire l'une de l'autre ajoute donc le DECALAGE PROPRE de la
            # piste au manque rapporte. MESURE SUR UN MAITRE REEL:
            #     en  Delay 0.005  Duration 1441.984  -> finit a 1441.989
            #     ja  Delay 0.983  Duration 1439.997  -> finit a 1440.980
            #     de  Delay 1.007  Duration 1440.980  -> finit a 1441.987
            # CES PISTES FINISSENT ENSEMBLE ET COMMENCENT DECALEES. Comparer les
            # seules Duration declare ja court de 1.99 s alors qu'il finit 1.0 s
            # avant en. L'erreur vaut le Delay, soit ici une seconde -- le meme
            # ordre qu'un residu fantome qui a ete cru par trois agents a la fois.
            delay_ms = delay_in_ms(master_audio)
            fill_source_ms = (Decimal(str(master_audio["Duration"])) * Decimal("1000")
                              + delay_ms)
            furthest = max((p["master_end_ms"] for p in pieces
                            if p["source"] == "master"), default=None)
            if furthest != None and furthest > fill_source_ms:
                fill_short_by_ms = furthest - fill_source_ms
        except Exception:
            # PAS de zero par defaut: une duree illisible est une ABSENCE.
            fill_source_ms = None
    # LE CHOIX A-T-IL ETE TRANCHE PAR LA MESURE OU PAR L'ENCODAGE? Si la piste
    # retenue EST le flux de reference, le plan a ete mesure contre elle et le
    # choix est adosse a une mesure. Sinon `pick_best_master_audio` a tranche sur
    # le codec, les canaux et le debit -- des dimensions qui n'ont rien a voir
    # avec le doublage, donc un tirage.
    #
    # Sans cette distinction, la marque AMBIGUOUS crierait aussi sur les cas ou
    # le choix est fonde, et une alerte qui se declenche sur les cas sains
    # devient une alerte qu'on cesse de lire.
    fill_by_reference = bool(
        fill == "master" and master_audio != None and reference_stream != None
        and str(master_audio.get("StreamOrder")) == str(reference_stream))
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
    filtergraph, head_pad_ms, head_decisions = build_audio_filtergraph(
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
    # SPEC_ZONE_A s4e: CE QUI A ETE COUPE ET CE QUI A ETE AJOUTE, AVEC LES
    # TEMPS, ET D'OU VIENT CHAQUE REGION REMPLIE. Le compte-rendu ne portait que
    # des TOTAUX -- `gap_filled_ms=100000` ne dit pas OU, et la ligne de plan
    # donne la geometrie du FICHIER, identique pour toutes les pistes, alors que
    # la SOURCE du remplissage est par piste. Un lecteur ne pouvait donc pas dire
    # quelle region avait recu de l'audio maitre et laquelle du silence.
    #
    # "Les comptes ne s'auditent pas" -- le proprietaire, deux fois. Un total est
    # un compte.
    filled_regions = []
    cut_regions = []
    # CE QUE LA SORTIE PREND AU CANDIDAT, ET AVEC QUEL DECALAGE. C'est la
    # MAJORITE de chaque fichier et elle n'avait aucune ligne: `ADDED` couvre le
    # remplissage maitre, `CUT` couvre le candidat jete, et les regions
    # reellement UTILISEES n'etaient nommees nulle part.
    #
    # vmsam-dev-4 recuperait le decalage en chainant les bornes des lignes CUT a
    # travers les spans du plan -- 9 raccords a faire tomber juste, et surtout
    # RECUPERABLE SEULEMENT SI LE PLAN A COUPE QUELQUE CHOSE. Un plan sans coupe
    # n'emet aucune ligne CUT, donc le decalage devenait irrecuperable pour une
    # raison qui ne concerne pas la piste. 2 journaux sur 10 portaient des
    # lignes CUT. LA RECUPERABILITE ETAIT UNE PROPRIETE DU PLAN.
    #
    # Le decalage n'a de sens que sur un morceau venu du CANDIDAT: les morceaux
    # maitre portent `source_start_ms = cursor`, une valeur de la timeline du
    # MAITRE, et l'imprimer ailleurs afficherait un nombre qui ne veut rien dire.
    # C'est pourquoi c'est une troisieme liste et pas un champ de plus sur les
    # deux autres.
    used_regions = []
    previous_candidate_end = None
    for piece in pieces:
        if piece["source"] == "master":
            filled += piece["master_end_ms"] - piece["master_start_ms"]
            filled_regions.append({
                "master_start_ms": str(piece["master_start_ms"]),
                "master_end_ms": str(piece["master_end_ms"]),
                # LA SOURCE, POUR QUE LE DECALAGE SOIT CALCULABLE PLUTOT
                # QU'AFFIRME. La ligne `ADDED` imprimait le LITTERAL `offset_ms=0`
                # -- un invariant affirme par une constante ne peut pas detecter
                # sa propre violation, et c'est le champ que `vmsam-dev-3` a
                # demande precisement pour qu'un invariant casse SE VOIE.
                # Sans cette borne le calcul n'a pas d'operande.
                "source_start_ms": str(piece.get("source_start_ms")),
                # LE MOTIF, QUI N'ARRIVAIT JAMAIS. Je l'ai pose sur les PIECES et
                # la ligne `ADDED` lit `filled_regions` -- deux objets. Resultat:
                # `why=` a emis `unreported(assembly predates the field)` sur
                # 48 lignes sur 48 produites par le module COURANT, et les trois
                # vraies valeurs -- `head_gap`, `interior_bracket`, `tail_gap` --
                # n'ont JAMAIS atteint un artefact.
                #
                # ET LE REPLI MENTAIT SUR LA CAUSE. Il dit que l'ASSEMBLAGE
                # precede le champ; la verite est que le champ ne traversait pas
                # jusqu'a l'emetteur. Un message de repli qui NOMME une cause est
                # une affirmation, et celle-ci etait fausse a chaque occurrence.
                #
                # `vmsam-dev-4` avait predit exactement ce controle: "si les trois
                # valeurs n'apparaissent PAS dans les nouveaux artefacts, c'est
                # une trouvaille sur le chemin d'emission et je vous la
                # rapporterai plutot que de m'y adapter."
                "reason": piece.get("reason"),
                # La source REELLE de cette region: l'audio du maitre, ou du
                # silence quand le maitre ne porte ni la langue de la piste ni
                # celle de comparaison.
                "source": "silence" if fill == "silence" else "master",
                "language": None if fill == "silence" else fill_language})
            continue
        if piece["source"] == "candidate":
            source_start = piece["source_start_ms"]
            source_end = source_start + (piece["master_end_ms"]
                                         - piece["master_start_ms"])
            if previous_candidate_end == None and source_start > 0:
                # COUPE DE TETE: le plan commence a lire le candidat APRES son
                # debut. Le predicat d'origine ne testait que les sauts ENTRE
                # deux morceaux, donc une tete coupee ne produisait aucune
                # ligne.
                cut_regions.append({
                    "candidate_start_ms": "0",
                    "candidate_end_ms": str(source_start),
                    "dropped_ms": str(source_start), "where": "head"})
            if previous_candidate_end != None and source_start > previous_candidate_end:
                # DU CANDIDAT SAUTE: ce materiau existe dans la source et
                # n'apparait pas dans la sortie. C'est la coupe, et sans ces
                # bornes elle n'est visible nulle part.
                cut_regions.append({
                    "candidate_start_ms": str(previous_candidate_end),
                    "candidate_end_ms": str(source_start),
                    "dropped_ms": str(source_start - previous_candidate_end),
                    "where": "interior"})
            used_regions.append({
                "master_start_ms": str(piece["master_start_ms"]),
                "master_end_ms": str(piece["master_end_ms"]),
                "candidate_start_ms": str(source_start),
                "candidate_end_ms": str(source_end),
                # candidate_time = master_time + offset. Par region, donc une
                # piste dont le plan est piecewise_constant en montre plusieurs
                # au lieu d'un seul `offset=measured` qui les ecrase tous.
                "offset_ms": str(source_start - piece["master_start_ms"])})
            previous_candidate_end = source_end
    if previous_candidate_end != None and candidate_duration_ms != None:
        tail = Decimal(str(candidate_duration_ms)) - previous_candidate_end
        if tail > 0:
            # COUPE DE QUEUE: du candidat apres la derniere lecture. Rien ne
            # tournait apres la boucle, donc elle etait invisible.
            # MESUREE, PAS SUPPOSEE: 2516.95 ms sur le fichier meme qui a servi
            # a valider ces lignes -- j'ai regarde l'interieur et jamais
            # au-dela du dernier morceau.
            cut_regions.append({
                "candidate_start_ms": str(previous_candidate_end),
                "candidate_end_ms": str(candidate_duration_ms),
                "dropped_ms": str(tail), "where": "tail"})
    elif previous_candidate_end != None and candidate_duration_ms == None:
        # ON NE SUPPOSE PAS ZERO. Sans la duree du candidat la queue est
        # INCONNUE, et une absence de ligne se lirait comme "rien n'a ete
        # coupe". On le DIT.
        cut_regions.append({"candidate_start_ms": str(previous_candidate_end),
                            "candidate_end_ms": None, "dropped_ms": None,
                            "where": "tail", "unmeasured": True})
    total = pieces[-1]["master_end_ms"] - pieces[0]["master_start_ms"]
    silence_ms = filled if fill == "silence" else Decimal("0")
    return {"stream_order": int(audio["StreamOrder"]), "language": language,
            # QUAND MEDIAINFO ET FFPROBE NE DISENT PAS LA MEME CHOSE, LE RAPPORT
            # LE PORTE. Liste vide quand ils s'accordent -- pas `None`, qui se
            # lirait comme "on n'a pas regarde".
            "tool_disagreements": tool_disagreements,
            "codec": codec_name, "encoder": encoder_arguments[1],
            "family": family, "gap_fill": fill, "fill_language": fill_language,
            "fill_title": fill_title, "fill_choices": fill_choices,
            "head_source": head_source,
            # QUEL FLUX MAITRE A REELLEMENT SERVI, ET NON SEULEMENT SA LANGUE.
            #
            # `master_stream_order` etait une LOCALE et n'etait jamais renvoyee.
            # Le rapport portait `fill_language`, `fill_title`, `fill_choices` et
            # `fill_by_reference` -- tout sauf l'identite du flux.
            #
            # vmsam-dev-3, qui construit l'instrument s4c: quand le maitre porte
            # plusieurs pistes dans la langue de remplissage, il ne peut pas
            # savoir CONTRE QUOI comparer. Et le titre ne desambigue pas: video.py
            # regroupe sur le code ISO, donc es-ES et es-419 tombent dans le meme
            # seau avant que ce module ne voie quoi que ce soit, et un fichier du
            # corpus porte QUATRE pistes spa dont deux titrees "European Spanish".
            # `fill_choices` dit combien il y en avait, jamais laquelle.
            "fill_stream_order": master_stream_order,
            "fill_source_ms": str(fill_source_ms) if fill_source_ms != None else None,
            "fill_short_by_ms": str(fill_short_by_ms) if fill_short_by_ms != None else None,
            "fill_by_reference": fill_by_reference,
            "path": out_path,
            "bitrate": bitrate, "bitrate_origin": bitrate_origin,
            "speed_ratio_requested": str(speed_ratio) if speed_ratio != None else None,
            "speed_ratio_applied": str(applied_ratio) if applied_ratio != None else None,
            "gap_filled_ms": str(filled),
            "filled_regions": filled_regions, "cut_regions": cut_regions,
            "used_regions": used_regions,
            # Le silence ajoute EN TETE parce que la source ne commence pas a
            # zero. Se dit: c'est du contenu que la piste produite n'a pas, et
            # il ne doit pas se confondre avec le remplissage depuis le maitre.
            "head_pad_ms": str(head_pad_ms),
            # LA DECISION, PAR MORCEAU, A COTE DU TOTAL. Un total ne s'audite pas.
            "head_decisions": head_decisions,
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
    # LES DECALAGES REELLEMENT APPLIQUES, comptes par morceau. Une piste dont
    # toutes les repliques tombent dans UN morceau rend un seul decalage -- ce
    # qui est precisement pourquoi deux langues atterrissent sur la meme
    # constante, et ce n'est pas un defaut.
    applied = {}
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
        applied[str(shift)] = applied.get(str(shift), 0) + 1
        event.start = int(event.start + shift)
        event.end = int(event.end + shift)
        if event.end <= event.start:
            dropped += 1
            continue
        kept_events.append(event)
    subtitles.events = kept_events
    subtitles.save(subtitle_path)
    # QUEL DECALAGE A ETE APPLIQUE, ET DEPUIS QUEL MORCEAU. `vmsam-ci` ne pouvait
    # pas tester la prediction que je lui ai donnee: la ligne de sous-titre porte
    # des COMPTES DE REPLIQUES ET AUCUN DECALAGE, donc l'ecart audio/sous-titre
    # que j'annonce constant est INVISIBLE dans le seul enregistrement qui survit
    # a une recreation.
    #
    # TROISIEME FOIS CE SOIR QUE LA MEME FORME BLOQUE UN CONTROLE -- apres
    # `candidate_offset_points` calcule trois fois et jamais emis, et le quantum
    # publie sans sa fenetre. Une quantite calculee et non emise, quand le
    # controle qui en a besoin vit HORS du processus. Ma propre phrase: c'est un
    # saut manquant, et le saut est a moi.
    return len(kept_events), dropped, applied


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
    if not path.getsize(out_path):
        # LA PISTE ETAIT DEJA VIDE A LA SOURCE -- zero paquet dans le fichier
        # d'origine, donc ffmpeg extrait un .srt de zero octet. Aucune replique
        # n'a ete supprimee: IL N'Y EN A JAMAIS EU.
        #
        # Ce cas n'atteignait PAS le garde ci-dessous, et la condition n'y etait
        # pour rien: `pysubs2.load` leve FormatAutodetectionError sur un fichier
        # vide, AVANT que le garde ne soit lu. La piste partait alors en `failed`
        # avec "No suitable formats" -- une phrase qui decrit l'analyseur, pas le
        # fichier -- au lieu d'un refus nomme. L'ORDRE, PAS LA CONDITION.
        #
        # Recense par vmsam-ci: 5 pistes subrip a zero paquet a la source dans le
        # corpus, et 0 en ass. Population distincte de celle du garde suivant, et
        # nommee separement pour qu'un comptage ne les confonde pas.
        raise chimeric_error(
            "the source subtitle track carries no cue at all: nothing was "
            "dropped, there was never anything to drop")
    kept, dropped, shifts_applied = retime_subtitle_file(out_path, pieces, speed_ratio)
    if not kept:
        # TOUTES les repliques sont tombees hors des morceaux gardes du
        # candidat. `pysubs2` ecrit alors un .srt de ZERO OCTET -- un .ass garde
        # ses en-tetes, un .srt n'a rien a garder -- et ffmpeg refuse ce fichier
        # avec "Invalid data found when processing input" AU MOMENT DU MUX,
        # c'est-a-dire apres toutes les pistes audio, ce qui fait echouer LA
        # REPARATION ENTIERE pour un sous-titre qui ne portait aucun contenu.
        #
        # Observe sur deux fichiers du corpus. Le compte-rendu disait alors
        # `kept_cues: 0` et se presentait comme un SUCCES: le nombre etait juste
        # et la conclusion tiree du nombre etait absente.
        raise chimeric_error(
            f"every cue fell outside the pieces kept from the candidate "
            f"({dropped} dropped, 0 kept): the track has no content on the "
            f"master timeline")
    return {"stream_order": int(subtitle["StreamOrder"]), "language": language,
            "codec": codec_name, "format": target, "path": out_path,
            "kept_cues": kept, "dropped_cues": dropped,
            # CE QUI A ETE APPLIQUE, PAS SEULEMENT COMBIEN A SURVECU. `{shift: n}`
            # -- une piste dont toutes les repliques partagent un morceau rend UNE
            # entree, et c'est la reponse a "pourquoi deux langues ont-elles la
            # meme constante".
            "shifts_applied_ms": shifts_applied,
            "title": subtitle.get("Title")}


def log_prediction_outcome(predicted, would_refuse):
    """La prediction gelee, confrontee a ce que la porte a REELLEMENT fait.

    `held` / `BROKEN` et non un booleen nu: un desaccord est une TROUVAILLE SUR
    LA PORTE, pas un defaut du fichier, et il doit se lire comme tel dans un
    journal que quelqu'un parcourt.

    LES DEUX SENS COMPTENT. Une refusal predite qui n'arrive pas dit que la porte
    est plus permissive que son propre calcul de remplissage; un refus NON predit
    dit que la porte a trouve quelque chose que le plan ne connaissait pas -- ce
    qui est exactement ce qu'on lui demande.

    `unknown` quand la porte n'a rendu aucun verdict: une panne d'outil a pu
    s'echapper avant, et `absent` n'est pas `non refuse`.
    """
    if would_refuse == None:
        verdict = "unknown(no verdict reached)"
    elif bool(predicted) == bool(would_refuse):
        verdict = "held"
    elif predicted:
        verdict = "BROKEN(predicted a refusal, the gate passed it)"
    else:
        verdict = "BROKEN(no refusal predicted, the gate refused)"
    tools.logs.append(f"repair: prediction predicted={len(predicted)} "
                      f"would_refuse={would_refuse} agreement={verdict}\n")


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
                                comparison_language=None, stream_pairing=None):
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
            report = build_one_audio_track(
                candidate_obj, master_obj, audio, language, track_pieces, track_path,
                timeout, speed_ratio, reference_stream, comparison_language,
                candidate_duration_ms)
            # Cette piste a-t-elle son propre decalage, ou emprunte-t-elle celui
            # de la langue mesuree? La table par flux ne couvre que cette
            # langue-la, donc toute autre langue emprunte, silencieusement, avec
            # 14 a 32 ms d'erreur mesures -- SOUS la tolerance du verificateur.
            report["offset_measured"] = all(
                offset_is_measured(segment, int(audio["StreamOrder"]))
                for segment in segments)
            report["pairing_contract"] = stream_pairing != None
            # POURQUOI CETTE PISTE EMPRUNTE, SUR LA LIGNE QUI EST REELLEMENT
            # EMISE. La distinction entre "aucun partenaire n'existe" et "le
            # meilleur partenaire est sous la barre" vivait UNIQUEMENT dans le
            # message de refus -- et le proprietaire vient de trancher: ON
            # CONTINUE D'EMPRUNTER, le drapeau reste desarme. Le refus ne se
            # declenche donc jamais et la distinction ne serait jamais ecrite.
            #
            # Si l'emprunt est ce qui est LIVRE, dire quelles pistes ont emprunte
            # et pourquoi est la seule chose entre un fichier faux et un fichier
            # indecouvrable.
            if report["offset_measured"]:
                report["borrow_reason"] = None
            elif stream_pairing == None:
                report["borrow_reason"] = "no pairing table in this plan"
            elif pairing_fidelity(stream_pairing, int(audio["StreamOrder"])) != None:
                report["borrow_reason"] = (
                    f"best partner fidelity "
                    f"{pairing_fidelity(stream_pairing, int(audio['StreamOrder']))}"
                    f", below the bar")
            else:
                report["borrow_reason"] = (
                    f"the master carries no {language} stream; no probe was run")
            report["offset_fidelity"] = next(
                (offset_fidelity(segment, int(audio["StreamOrder"]))
                 for segment in segments
                 if offset_fidelity(segment, int(audio["StreamOrder"])) != None),
                None)
            # TROISIEME ETAT, QUE NI MOI NI L'AUTEUR DE LA MESURE N'AVIONS
            # ECRIT: le contrat a change DANS SON DEPOT ET PAS ENCORE EN
            # PRODUCTION. On avait deux sens pour "pas d'entree" -- "autre
            # langue" avant, "aucun partenaire n'a passe la barre" apres -- et
            # il en existe un troisieme: "CE LOCATOR NE CONNAIT PAS ENCORE LA
            # QUESTION".
            #
            # Armer le drapeau contre un ancien locator refuserait CHAQUE piste
            # hors langue mesuree, et contre un locator plus ancien encore
            # TOUTES les pistes, y compris celle du plan -- et le journal dirait
            # "no offset was measured", ce qui serait VRAI et ruineux.
            #
            # La presence de la table PAR FICHIER est donc la marque du contrat.
            # Absente, on emprunte et on le journalise, comme aujourd'hui.
            # Presente, une entree manquante veut enfin dire ce qu'elle doit.
            contract_present = stream_pairing != None
            if (not report["offset_measured"] and refuse_borrowed_offset
                    and contract_present):
                # La fidelite, QUAND ELLE EXISTE, distingue "aucun partenaire
                # n'a passe la barre, a 0.82" de "rien n'a ete mesure du tout".
                # LE REFUS CITE LE NOMBRE PAR FICHIER, pas celui d'une
                # tranche: la barre a ete appliquee au choix de partenaire, une
                # fois, pour tout le fichier.
                # DEUX FORMES, PARCE QUE CE SONT DEUX FAITS DIFFERENTS.
                #
                #   fidelite = UN NOMBRE sous la barre -> une sonde a tourne et a
                #       rendu une valeur. Depuis le defaut de sonde a cheval
                #       trouve ce matin, ce nombre peut decrire LE PLACEMENT DE
                #       LA SONDE et non la piste.
                #   fidelite = None -> AUCUNE SONDE N'A TOURNE. Le maitre ne
                #       porte aucun flux de cette langue, donc il n'existe aucun
                #       partenaire a scorer. Ce n'est pas un refus de mesurer:
                #       LA QUANTITE N'EXISTE PAS.
                #
                # Un lecteur voyant la premiere forme sur une piste de la seconde
                # chercherait une fidelite qui n'a jamais ete prise.
                seen = pairing_fidelity(stream_pairing, int(audio["StreamOrder"]))
                if seen != None:
                    raise chimeric_error(
                        f"no offset was measured for this stream (best partner "
                        f"fidelity {seen}, below the bar): it would carry another "
                        f"track's alignment, and a borrowed offset is not a "
                        f"measurement")
                raise chimeric_error(
                    f"no offset is measurable for this stream: the master "
                    f"carries no {language} stream, so no partner exists to "
                    f"measure against and no probe was run -- the quantity does "
                    f"not exist, by any method")
            audio_reports.append(report)
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

    # LA PREDICTION EST GELEE AVANT LE MUX, PUIS MESUREE.
    #
    # Discipline de `vmsam-ci`, celle qu'il applique a chaque deploiement, et sa
    # proposition plutot que la mienne: je proposais de DECLINER au moment du
    # plan quand la source de remplissage est trop courte. Son objection est
    # structurelle et decisive:
    #
    #   SI L'ASSEMBLEUR DECLINE D'ABORD, LA PORTE NE TOURNE JAMAIS SUR CE CAS --
    #   on retirerait le controle qui a TROUVE le defaut, au motif que le defaut
    #   existe.
    #
    # Et le cout d'un declin precoce est mesure plutot que suppose: le PREMIER
    # artefact refuse que cette campagne ait garde a produit deux defauts dans le
    # verificateur de ci, le diagnostic que la faute est en amont de ma porte, et
    # la premiere preuve que le magasin durable traverse le montage. UN DECLIN AU
    # PLAN AURAIT RENDU UNE LIGNE DE JOURNAL ET RIEN DE TOUT CELA.
    #
    # ON PREDIT DONC, ET ON CONSTRUIT QUAND MEME. Une ligne, et le refus devient
    # FALSIFIABLE A L'AVANCE au lieu d'etre explicable apres coup -- et si la
    # porte NE refuse PAS ce qui etait predit, ce desaccord est une trouvaille
    # SUR LA PORTE que rien d'autre ne produirait.
    predicted_refusals = []
    for report in audio_reports:
        short = report.get("fill_short_by_ms")
        if short in (None, "", "0"):
            continue
        if abs(Decimal(str(short))) <= output_duration_tolerance_ms:
            continue
        predicted_refusals.append(report)
        tools.logs.append(
            f"repair: PREDICTED_REFUSAL track={report.get('stream_order')} "
            f"fill_short_by_ms={short} "
            f"tolerance_ms={output_duration_tolerance_ms} "
            f"reason=the plan targets a duration the fill source cannot supply\n")

    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout)

    # L'ACCEPTATION PORTE SUR LE FICHIER ET ELLE PASSE AVANT L'ALIGNEMENT.
    # `SPEC_ZONE_A.MD` s4d. Verifier l'alignement d'une piste tronquee sonde des
    # positions qui existent encore et rend "aligned" sur un fichier ampute --
    # c'est exactement ce qui a rapporte "7 audio et 24 sous-titres
    # reconstruits, 0 refuse, 0 en echec" sur un fichier sans rien apres 21:21.
    # LE FICHIER EST DEJA ECRIT QUAND LE CONTROLE LE LIT, DONC UN REFUS LAISSE UN
    # ARTEFACT SUR LE DISQUE. Tant que le drapeau etait inerte, cet objet
    # n'existait pas; maintenant il y en a un par refus.
    #
    # "aucun fichier faux ne sort" TIENT A LA FRONTIERE DU MERGE -- l'objet n'est
    # accroche a `sameAudioMD5UseForCalculation` qu'apres le retour, donc
    # `mergeVideo` ne le voit jamais. IL NE TIENT PAS SUR LE SYSTEME DE FICHIERS:
    # le fichier porte le marqueur `VMSAM_FABRICATED` et AUCUN compte rendu ne le
    # revendique. C'est exactement la forme qui m'a coute une heure sur `out108`
    # -- un artefact trouve par son chemin et lu comme expedie.
    #
    # ON NE LE SUPPRIME PAS: c'est la seule trace de CE QUE le controle a refuse,
    # et le declin la nomme. ON LE RENOMME, ET LE SUFFIXE EST CELUI QUE
    # `vmsam-ci` A DEMANDE, POUR SES RAISONS ET PAS POUR LES MIENNES:
    #
    #   son preserveur balaye par EXTENSION (`-name '*.mkv'`), donc un refus
    #   qui finit en `.mkv` est lie en dur dans KEEP et compte comme produit par
    #   lui, par vmsam-forensic et par le registre.
    #
    # `<nom>.REFUSED.mkv` -- le marqueur AVANT l'extension: le fichier reste
    # ouvrable et diagnosticable, et il se repere par un motif de NOM et non par
    # une convention de CHEMIN. ci exclut `*.REFUSED.*` de son preserveur et de
    # `check_output`.
    #
    # LA VERIFICATION D'ALIGNEMENT EST DEDANS AUSSI. Elle leve apres le mux
    # exactement comme le controle de duree, et un artefact orphelin refuse pour
    # desalignement se compte de la meme facon qu'un refuse pour troncature.
    try:
        output_check = verify_output_file(out_path, master_duration_ms, audio_reports,
                                          subtitle_reports,
                                          output_duration_tolerance_ms)
        log_prediction_outcome(predicted_refusals, output_check.get("would_refuse"))

        verification = None
        if verify:
            verification = verify_on_master_timeline(
                out_path, master_obj, audio_reports, pieces, verify_tolerance_ms,
                verify_search_ms, reference_stream)
    except Exception as error:
        # LA PREDICTION EST MESUREE SUR LES DEUX BRANCHES. Ne la mesurer que sur
        # le chemin qui rend serait ne la mesurer que quand elle a echoue.
        log_prediction_outcome(predicted_refusals,
                               (getattr(error, "output_check", None) or {}).get(
                                   "would_refuse"))
        # UNE SEULE BRANCHE, ET C'EST UNE CORRECTION MESUREE SUR UN ARTEFACT.
        #
        # Il y en avait deux: `chimeric_error` attachait `partial_assembly`, la
        # branche generique ne l'attachait PAS. Consequence, lue sur le fichier
        # `FAILED` persiste pour vmsam-dev-4: DEUX lignes de journal au lieu de
        # vingt -- pas de `build`, pas de `sources`, pas de `plan`, aucune ligne
        # par piste. `log_assembly` est saute quand l'assemblage partiel manque.
        #
        # ET J'AVAIS ANNONCE LE CONTRAIRE. "Les deux chemins emettent maintenant
        # le bloc complet" etait faux pour le chemin `failed`, et c'est
        # l'artefact qui l'a dit, pas la relecture du code. C'est la TROISIEME
        # fois cette nuit qu'un trou ferme sur une sortie reste ouvert sur sa
        # jumelle -- d'ou une branche unique plutot que deux qui se ressemblent.
        marking = (OUTPUT_REFUSED if isinstance(error, chimeric_error)
                   else OUTPUT_NO_VERDICT)
        error.undelivered_state = marking[0]
        error.undelivered_path, error.undelivered_durable = mark_output(
            out_path, marking, stable_case_key(candidate_obj.filePath))
        error.undelivered_in_place = out_path
        # L'ASSEMBLAGE PARTIEL VOYAGE AVEC LE REFUS, ET C'EST LE DRAPEAU LEVE QUI
        # REND CETTE LIGNE NECESSAIRE.
        #
        # `log_assembly` est appele par `merge_video_repair` APRES cet appel-ci.
        # Une levee ici le saute, donc un fichier DECLINE par la porte n'emet
        # AUCUNE ligne `repair:` -- pas meme la ligne `plan`. `vmsam-dev-4` lit
        # ces journaux par STRUCTURE et rejette un bloc sans ligne `plan`: les
        # declins seraient invisibles dans son rapport exactement comme le sont
        # deja les pannes precoces. Il a pose la question avant le rendu plutot
        # que de decouvrir le trou dedans.
        #
        # ET LE COMMENTAIRE DE `merge_video_repair` DIT DEJA LE CONTRAIRE DE CE
        # QUI SE PASSAIT: "un journal ecrit seulement en cas de succes ne
        # documente jamais les cas qui en avaient besoin". Il etait ecrit avant
        # la RELECTURE du fichier, pas avant la PORTE.
        #
        # `verification` est None et non omis: le declin peut venir de la porte
        # de duree, auquel cas l'alignement n'a jamais ete mesure, et "pas
        # mesure" n'est pas "mesure et vide".
        error.partial_assembly = {
            "path": out_path, "pieces": pieces, "audios": audio_reports,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": marker_value, "verification": None}
        raise

    # LA CADENCE DU MAITRE, PUBLIEE. Elle n'est derivable d'AUCUNE ligne du
    # journal, et trois consommateurs en ont besoin: l'exclusion de vitesse de
    # vmsam-dev-3 ne peut pas etre calculee sans elle, la ligne GAP du rapport
    # doit NOMMER la cadence supposee, et `adjust_delay_to_frame` colle sur elle.
    #
    # vmsam-dev-1 a balaye 561 fichiers: 559 CFR, 2 VFR, et DEUX FICHIERS CFR A
    # CADENCE NON STANDARD -- 23.839 et 47.281 -- qui prennent quand meme la
    # branche de collage et collent donc sur une grille FABRIQUEE. Une cadence
    # supposee identique pour tout le corpus est fausse pour ces deux-la, et rien
    # dans le journal ne le disait.
    #
    # `unread` et non zero quand mediainfo ne la donne pas.
    #
    # DEUX CHAMPS, ET LEUR DESACCORD EST L'INFORMATION. vmsam-dev-3: sur les deux
    # seuls fichiers dont la cadence est inhabituelle, `FrameRate` et
    # `FrameRate_Original` DIFFERENT -- 23.839 contre 23.976, 47.281 contre
    # 29.970. Emettre un seul champ collapse precisement ce qui rend ces
    # fichiers interessants, et un lecteur qui calcule une periode d'image
    # obtient 41.948 ms la ou le nominal est 41.708.
    #
    # Et le MODE se dit aussi, parce que la branche de collage ne teste pas
    # "VFR": elle teste l'egalite avec la chaine exacte "CFR". Un mode absent ou
    # vide prend donc silencieusement le chemin non colle, et rien ne disait
    # lequel avait tourne.
    frame_rate = None
    frame_rate_mode = None
    frame_rate_original = None
    try:
        frame_rate = master_obj.video.get("FrameRate")
        frame_rate_mode = master_obj.video.get("FrameRate_Mode")
        original = master_obj.video.get("FrameRate_Original")
        if original != None and str(original) != str(frame_rate):
            frame_rate_original = original
    except Exception:
        pass
    return {"path": out_path, "pieces": pieces, "audios": audio_reports,
            "master_frame_rate": frame_rate,
            "master_frame_rate_mode": frame_rate_mode,
            "master_frame_rate_original": frame_rate_original,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": marker_value,
            "output_check": output_check,
            "verification": verification}


# --------------------------------------------------------------------------
# Verification: la piste produite est-elle VRAIMENT sur la timeline du maitre?
# --------------------------------------------------------------------------

# LA COUVERTURE DE CE VERIFICATEUR EST DE 5.5 A 11.3 POUR CENT DE LA DUREE.
# Mesuree sur douze fichiers reparés: deux sondes de 20 s par morceau candidat,
# soit 80 a 160 s sondes sur ~1430 s. MEDIANE 11.2 %.
#
# CONSEQUENCE, ET ELLE BORNE TOUT CE QUE CE MODULE AFFIRME: `aligned` VEUT DIRE
# "LES POSITIONS SONDEES SONT ALIGNEES", PAS "CE FICHIER N'A PAS DE PASSAGE
# DEPLACE". Un defaut TRANSITOIRE -- une region de vingt secondes portant de
# l'audio decale -- est invisible ICI par construction, quelle que soit la
# tolerance: LES SONDES NE SONT PAS LA.
#
# Ce n'est pas une hypothese. Un tel defaut existe et je l'ai manque: un balayage
# de laboratoire a 50 % de cycle utile a rate une region de 20 s a 1420-1440 s
# sur un fichier produit, et j'ai publie "aucune region nulle part". Le
# verificateur de production a un cycle utile QUATRE FOIS PLUS FAIBLE.
#
# `vmsam-forensic` a declare la meme limite sur son scanner de raccords (24 %) et
# sur six autres de ses instruments le meme jour. La regle commune: UN
# DISCRIMINANT A FAIBLE CYCLE UTILE DETECTE LE PERSISTANT ET RIEN D'AUTRE, parce
# qu'un pas qui continue jusqu'a la fin se lit de n'importe quel cote des sondes,
# et qu'une region bornee doit etre TOUCHEE pour etre vue.
#
# CE QUI N'EST PAS EN CAUSE: la conception. Deux sondes ecartees dans un morceau
# mesurent le DESACCORD, qui est le signal recherche, et le cycle utile est le
# prix de ne pas decoder chaque fichier en entier a chaque reparation.
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
    # L'OUTIL A-T-IL ECHOUE, OU LA PISTE EST-ELLE VIDE? CE SONT DEUX CHOSES.
    #
    # Cette fonction ne lisait NI `returncode` NI `stderr` -- zero occurrence de
    # l'un et de l'autre dans tout le module. Si ffmpeg echoue (specificateur de
    # flux faux, fichier illisible, recherche au-dela de la fin, codec qu'il ne
    # sait pas ouvrir), stdout est VIDE, `len(samples)` vaut 0, et le code levait
    # `the track carries no audio to compare there`.
    #
    # C'EST UNE AFFIRMATION SUR LE MEDIA ALORS QUE LA VERITE EST UNE AFFIRMATION
    # SUR L'OUTIL. Et ffmpeg avait ecrit la vraie raison sur stderr, qui etait
    # capturee puis JETEE.
    #
    # C'est exactement le defaut corrige ce soir sur `master_path`, une fonction
    # plus loin -- une RAISON FAUSSE dans une issue legitime, ou rien ne parait
    # anormal -- et il est pire ici: le commentaire ci-dessous dit que cette
    # phrase DEVIENT la raison d'un refus, et une raison se cite. Elle est faite
    # pour voyager, donc elle peut voyager fausse.
    #
    # STATUT: signale par le Lead comme LU ET NON OBSERVE -- aucune instance dans
    # le corpus. Ce qui est mesure, c'est que les deux canaux etaient ignores.
    if process.returncode != 0:
        raise chimeric_error(
            f"the reader FAILED on {stream_specifier} at {start_ms} ms: ffmpeg "
            f"exited {process.returncode}. THIS IS A STATEMENT ABOUT THE TOOL, "
            f"not about the media: "
            f"{(process.stderr or b'').decode('utf-8', 'replace').strip()[-300:]}")
    samples = numpy.frombuffer(process.stdout, dtype=numpy.float32).astype(numpy.float64)
    if len(samples) < rate:
        # Meme regle: le specificateur de flux et la position suffisent a
        # diagnostiquer, le chemin ne sert qu'a identifier le media. Cette
        # phrase devient la RAISON d'un refus, et une raison se cite.
        raise chimeric_error(
            f"read only {len(samples)} samples from {stream_specifier} at "
            f"{start_ms} ms: the track carries no audio to compare there")
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
# LA CONDITION DE LEVEE EST ECRITE UNE SEULE FOIS, PLUS BAS, AVEC `measured`:
# VINGT ARTEFACTS MESURES -- des pistes audio presentes ET des durees lues.
#
# Ce commentaire portait une SECONDE condition, "des que le balayage du corpus
# est termine", ecrite avant que la premiere ne soit affinee. LES DEUX NE
# CONCORDENT PAS: un balayage qui se termine et un compte qui atteint vingt sont
# des evenements differents et aucun n'implique l'autre. Un fichier qui porte
# deux conditions a celle que le prochain lecteur trouve en premier.
#
# ET LA CONDITION SUPPRIMEE EST DEVENUE INATTEIGNABLE: le balayage du corpus est
# SUSPENDU a 3 lignes sur 315, au profit d'une priorite du proprietaire. Rien de
# ce qui tourne aujourd'hui ne peut plus la satisfaire. C'est la quatrieme fois
# que la condition de ce drapeau echoue -- inatteignable, incomptable,
# satisfiable par la mauvaise chose, et maintenant DOUBLE -- et la seule qui
# ait tenu est celle qui compte une quantite emise par l'instrument lui-meme.
#
# CE QUI RESTE VRAI ET N'EST PAS UNE CONDITION: si quoi que ce soit doit etre
# PRODUIT POUR DE VRAI, le drapeau passe a True immediatement, quel que soit le
# compte. Ce n'est pas une seconde condition de levee, c'est un ARBITRAGE DU
# PROPRIETAIRE qui prime sur elle, et il n'appartient ni a ce fichier ni a moi
# de le renegocier. La question "quelque chose est-il prevu pour de vrai?" est
# ouverte aupres du proprietaire et sans reponse.
#
# ---- LA QUESTION A RECU SA REPONSE, ET C'EST L'ARBITRAGE, PAS LE COMPTE ----
#
# Le proprietaire a tranche, rapporte mot pour mot par `vmsam-ci`:
#
#     "aucun fichier faux ne sort. Aucun fichier reparable ne sort non plus
#      tant que le planificateur n'est pas corrige."
#
# LA SECONDE PHRASE EST LE COUT ET IL A ETE ACCEPTE EXPLICITEMENT: drapeau leve,
# les fichiers REPARABLES cessent de sortir eux aussi, jusqu'a correction du
# planificateur. C'est une decision de debit prise avec le chiffre sous les yeux.
#
# LE CHIFFRE, MESURE PAR `vmsam-ci` SUR LA POPULATION ENTIERE QUI PORTE LE CHAMP:
#
#     artefacts produits portant le champ    14
#     would_refuse = True                     6      <- ce qui s'arrete
#     enforcing = False                      14 / 14 <- avant aujourd'hui
#
# 14 est TOUTE la population portant le champ, pas 14 sur 315: les fichiers
# DECLINED n'atteignent jamais le controle. Ce n'est pas un taux de corpus.
#
# ET LA CONDITION POSEE PLUS HAUT N'A PAS ETE CONTOURNEE, ELLE A ETE SATISFAITE
# PAR L'EVENEMENT QU'ELLE NOMMAIT. Le drapeau a echoue quatre fois sur une
# condition CHIFFREE -- inatteignable, incomptable, satisfiable par la mauvaise
# chose, double. La cinquieme forme est la seule qui ait tenu, et elle n'est pas
# un seuil: c'est un arbitrage nomme d'avance, leve par la personne nommee.
#
# DEUX SEUILS MESURENT LA MEME PROPRIETE ET ILS NE SONT PAS D'ACCORD. La
# tolerance de piste courte de `vmsam-ci` est de 2 % -- 30 s sur un fichier de
# 1500 s -- et ce controle-ci refuse `id 5` a 1994 ms. Trois des six refuses
# portent son verdict FULL_LENGTH. LES DEUX SEUILS SONT RAPPORTES SUR CHAQUE
# LIGNE PLUTOT QUE RECONCILIES: deux seuils nommes qui divergent sont honnetes,
# un seul choisi en silence ne l'est pas. Lequel est LA norme appartient au
# proprietaire, pas a celui des deux qui imprime "validated".
output_check_enforcing = True

# UN DECALAGE EMPRUNTE SE REFUSE-T-IL? PAS ENCORE, ET LA RAISON N'EST PAS LA
# PRUDENCE: C'EST QUE "PAS D'ENTREE DANS LA TABLE" NE VEUT PAS ENCORE DIRE CE
# QU'IL VOUDRA DIRE.
#
#   AUJOURD'HUI  le locator ne mesure que les flux de la langue du plan, donc
#                une entree absente signifie "cette piste est d'une AUTRE
#                langue" -- et `vmsam-dev-1` vient de mesurer que ces pistes
#                sont parfaitement mesurables: 0.944 a 0.990 de fidelite contre
#                un flux maitre de LEUR langue. Refuser maintenant jetterait des
#                pistes dont le decalage est disponible, simplement pas demande.
#
#   APRES        quand la table sera construite par flux, chaque flux candidat
#                apparie a son MEILLEUR flux maitre et l'appariement filtre par
#                fidelite, une entree absente signifiera "aucun partenaire n'a
#                passe la barre", c'est-a-dire VRAIMENT non mesurable. Refuser
#                devient alors la seule reponse honnete.
#
# LA MEME CONDITION, DEUX SENS OPPOSES. Ce drapeau ne bascule donc PAS sur un
# comptage d'artefacts: il bascule quand le contrat du locator change, et pas
# avant. Le lier a un nombre serait mesurer la mauvaise chose avec assurance.
#
# En attendant, le repli reste, et il est VISIBLE: `offset=BORROWED` sur la
# ligne de chaque piste concernee. Mesure: 14 a 32 ms d'erreur sur deux
# fichiers, sous la tolerance de 100 ms du verificateur.
#
# Et le cas que ni dev-1 ni moi n'avions nomme, qui interdit la solution
# evidente: sur ces deux fichiers, DEUX pistes candidates portent l'etiquette
# eng et une seule est celle du maitre -- l'autre correle a 0.59 avec l'anglais
# du maitre et 0.57 avec son francais. ELLE NE CORRESPOND A RIEN. Une regle qui
# apparie PAR ETIQUETTE lui donnerait le decalage d'une piste avec laquelle elle
# ne partage aucun contenu, et cela aurait l'air juste dans tous les journaux.
# Une etiquette de langue est une AFFIRMATION sur une piste; la fidelite en est
# une MESURE.
refuse_borrowed_offset = False


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


# LES DEUX ETATS QUI NE SONT PAS "PRODUIT", ET ILS N'AFFIRMENT PAS LA MEME CHOSE.
#
#   REFUSED     LA PORTE A DECIDE contre le fichier
#   NOVERDICT   PERSONNE N'A DECIDE -- une panne d'outil s'est echappee avant
#               qu'un verdict existe
#
# Un seul jeton pour les deux recreerait exactement l'effondrement qu'on repare:
# un lecteur qui compte les refus absorberait en silence chaque panne d'ffprobe
# dans le cout de la porte. `vmsam-ci` a nomme le second `unadjudicated` dans son
# registre et il ne porte AUCUNE affirmation sur le media -- il dit ce qui est
# arrive au PROCESSUS, ce qui est ce qui s'est reellement passe.
#
# TROISIEME ETAT SANS JETON DE LA SOIREE, et le premier qui soit un FICHIER SUR
# LE DISQUE plutot qu'une valeur dans une ligne: `verdict != FULL_LENGTH` lu
# comme "piste courte", "pas de ligne DECLINED" lu comme "pas de declin", et ici
# un artefact NON VERIFIE portant le nom d'un fichier produit. Dans les trois cas
# le defaut par defaut etait le flatteur.
# LES OCTETS QUI ONT ETE COMPILES EN MEMOIRE, HACHES A L'IMPORT DE CE MODULE.
#
# CORRECTION APPORTEE PAR `vmsam-dev-4`, ET ELLE EST JUSTE: un condensat lu A
# L'APPEL est le condensat du FICHIER SUR LE DISQUE a ce moment-la, pas du CODE
# EN MEMOIRE. Un processus de longue duree garde son module charge; si le fichier
# change sous lui, l'empreinte bouge et le code execute ne bouge pas.
#
# CE N'EST PAS HYPOTHETIQUE, C'EST ARRIVE CE SOIR: mon balayage `t79` a charge ce
# module avant que je fusionne les deux clauses `except`, et il a continue a
# executer l'ancien code pendant que ses lignes portaient des condensats du
# nouveau fichier. "Le condensat identifie le code qui tourne" etait UNE CLAUSE
# PLUS FORTE que ce qu'il pouvait porter.
#
# A L'IMPORT, l'ecart se referme: le chargeur vient de lire ces octets pour en
# faire le code en memoire, et on les rehache immediatement. Reste une fenetre de
# quelques microsecondes entre les deux lectures, et un `importlib.reload`, qui
# ne se produit nulle part ici.
def _digest_of_loaded_source():
    import hashlib
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except Exception:
        # UNE EMPREINTE QU'ON N'A PAS PU LIRE SE DIT. Elle ne vaut pas zero et ne
        # s'omet pas: un champ absent se lirait comme un vieux build.
        return "unreadable"


LOADED_SOURCE_DIGEST = _digest_of_loaded_source()


OUTPUT_REFUSED = ("REFUSED", "the gate DECIDED against it")
OUTPUT_NO_VERDICT = ("NOVERDICT", "NOBODY decided -- a tool fault escaped before "
                                  "any verdict existed")


def mark_output(out_path, marking, case_key=None):
    """Renomme un artefact non livre en `<nom>.<JETON>.<ext>` et rend le chemin.

    Rend `None` si le fichier a disparu ou si le renommage echoue -- et le
    consigne. UN ECHEC DE RENOMMAGE NE DOIT PAS REMPLACER LA RAISON DU REFUS:
    l'appelant est deja en train de lever, et masquer un refus de troncature par
    une OSError de systeme de fichiers perdrait la seule information que le
    declin porte. On rapporte les deux plutot que d'en substituer une.

    LE JETON VA AVANT L'EXTENSION, forme demandee par `vmsam-ci` pour ses
    raisons: son preserveur balaye par EXTENSION et lierait un `.mkv` non livre
    dans KEEP, ou il compterait comme produit par lui, par `vmsam-forensic` et
    par le registre de `vmsam-dev-4`. Le fichier reste ouvrable et se repere par
    un motif de NOM et non par une convention de CHEMIN.
    """
    token, why = marking
    if not path.exists(out_path):
        return None, False
    base, extension = path.splitext(out_path)
    marked = f"{base}.{token}{extension}"
    try:
        replace_file(out_path, marked)
    except OSError as error:
        sys.stderr.write(f"repair: the undelivered artefact could NOT be renamed "
                         f"and is still at its produced name: {error}\n")
        tools.logs.append("repair: an undelivered artefact kept its produced name\n")
        return None, False
    sys.stderr.write(f"repair: the artefact was renamed to *.{token}{extension} "
                     f"-- {why} -- so it is inspectable and NOT counted as "
                     f"produced\n")
    # REND AUSSI *OU* IL A ATTERRI. `path=` seul ne distingue pas un artefact
    # sauve d'un artefact reste dans l'ephemere: les deux sont des chemins
    # plausibles, et seul l'un des deux survivra a la prochaine recreation.
    # `durable=` est un champ que le balayage de ci peut COMPTER.
    final = move_to_durable_store(marked, token, case_key)
    return final, final != marked


# OU VIT UN ARTEFACT NON LIVRE APRES LA FIN DU CONTENEUR.
#
# Le renommage etait correct et le FICHIER MOURAIT QUAND MEME: il reste sous
# `/tmp/gestionar_show_<demarrage-du-conteneur>/...`, donc CHAQUE RECREATION
# DETRUIT TOUS LES ARTEFACTS REFUSES DEPUIS LA PRECEDENTE. `vmsam-ci` recree sur
# les lots de promotion du Lead, donc LE RYTHME DE PERTE DE PREUVES EST FIXE PAR
# LA CADENCE D'EXPEDITION. Sa recreation de 21:18Z a emporte l'artefact de
# l'id 31 -- une reparation qui avait REFUSE SON PROPRE PLAN apres avoir mesure
# la piste produite contre lui.
#
# ci a fait sa moitie -- second balayage, second repertoire, `declined_by_gate`
# et `unadjudicated` comme deux jetons -- mais son balayage s'enracine sur le
# disque partage et CES FICHIERS N'Y SONT JAMAIS. SEUL LE PRODUCTEUR PEUT LES
# METTRE QUELQUE PART QUI SURVIT.
#
#   UN FICHIER REFUSE ET UN FICHIER JAMAIS PRODUIT SONT INDISCERNABLES UNE FOIS
#   L'ARTEFACT DISPARU.
#
# LE DEFAUT RESTE `None` = SUR PLACE, donc ce changement n'a AUCUN effet tant
# que personne ne pose la racine. La destination appartient a ci: c'est lui qui
# sait quelle racine son SECOND balayage regarde et laquelle son balayage
# PRINCIPAL ne doit pas voir -- il a mesure que remonter d'un cran y ferait
# entrer 509 fichiers d'autres agents. Un artefact refuse ne doit JAMAIS compter
# comme produit.
UNDELIVERED_STORE_ENV = "VMSAM_UNDELIVERED_ROOT"

# LA RACINE, NOMMEE PAR `vmsam-ci` APRES L'AVOIR TESTEE PLUTOT QUE LUE.
#
#   /config/output/undelivered/    le PRODUCTEUR ecrit ici
#   /config/output/DECLINED/       le SECOND balayage de ci lie en dur ICI
#
# ET PAS DANS `DECLINED/`: c'est la SORTIE de son balayage et il l'elague, donc
# ce que j'y ecrirais serait saute et n'aurait jamais de ligne de registre. La
# separation est le point: mon fichier reste supprimable, son lien survit.
#
# ET SURTOUT PAS SOUS `/config/output/srv`: c'est la racine de son balayage
# PRINCIPAL, dont la liste de motifs contient deja `*.REFUSED.*`. Rien ne serait
# MAL COMPTE -- il serait classe `declined_by_gate` et exclu des comptes de
# production -- mais il se trouverait dans le repertoire ou trois agents comptent
# des fichiers, et ci prefere qu'il n'y arrive jamais plutot que de dependre
# d'une colonne de classe pour rester honnete.
UNDELIVERED_STORE_DEFAULT = "/config/output/undelivered"

# QUI A OBSERVE LE REFUS, DANS LE NOM, TOUJOURS.
#
# `vmsam-ci` recevra un SECOND conteneur -- `showgestionar-test2` -- des la
# prochaine mise a jour. Il n'existe pas encore, et ci l'a ecrit dans son propre
# registre avec la date, parce que "j'ai deux conteneurs" est une capacite fausse
# tant que la mise a jour n'est pas la.
#
# LA COLLISION QUE CA CREE, ET ELLE VIENT DE CE QUE J'AI FAIT DE JUSTE:
# `stable_case_key` ne depend QUE du cas, ce qui empeche un meme refus de
# ressembler a un artefact neuf apres chaque recreation. Avec DEUX conteneurs, un
# meme cas refuse UNE FOIS et observe par les deux atterrit au meme `case_key` --
# et le suffixe anti-ecrasement `.1` le fait ressembler a DEUX refus.
#
#   C'EST `absent n'est jamais zero` A L'ENVERS: un compte GONFLE par un
#   OBSERVATEUR, la ou nous avons passe la nuit sur des comptes deflates par un
#   silence. Et c'est invisible dans un listing: `.1` est exactement ce a quoi
#   ressemble un vrai second refus.
#
# LE JETON DE L'OBSERVATEUR EST DONC DANS LE NOM ET NON DANS LE CHEMIN. Le chemin
# aurait fragmente le cas entre deux repertoires; le nom garde le `case_key`
# comme seul lieu du cas, laisse `*.REFUSED.*` apparier a n'importe quelle
# profondeur, et rend l'ordinal a son sens d'origine -- UN VRAI SECOND REFUS PAR
# LE MEME OBSERVATEUR.
#
# ET QUAND ON NE PEUT PAS LE DETERMINER, LE NOM LE DIT. `unknown-observer` plutot
# que rien: un jeton omis rendrait les deux mondes indiscernables, ce qui est la
# faute que ce champ existe pour empecher.
UNDELIVERED_OBSERVER_ENV = "VMSAM_CONTAINER_NAME"


def observing_container():
    """Le nom du conteneur qui a observe ce refus, ou `unknown-observer`.

    `HOSTNAME` en repli: sous Docker c'est l'identifiant du conteneur, donc deux
    conteneurs se distinguent meme sans configuration explicite. Un
    `unknown-observer` dans un nom de fichier est une question posee au lecteur;
    son absence n'en serait pas une.
    """
    name = (os.environ.get(UNDELIVERED_OBSERVER_ENV)
            or os.environ.get("HOSTNAME") or "").strip()
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name)
    return safe or "unknown-observer"


def stable_case_key(candidate_path):
    """L'identifiant d'un CAS, stable pour un episode donne, et pas du conteneur.

    EXIGENCE DE `vmsam-ci`, et c'est la seule chose qui pouvait mal tourner: son
    balayage SAUTE une destination qui existe deja, donc un chemin stable est
    balaye UNE FOIS. Un chemin indexe sur le demarrage du conteneur ferait
    ressembler le MEME refus a un artefact NEUF apres chaque recreation -- il
    preserverait des doublons et compterait une classe qui n'a pas grandi.

    C'est la meme derivation que la cle de travail de `merge_video_repair`, et
    elle vit ICI pour qu'elles ne puissent pas deriver l'une de l'autre: deux
    copies d'une meme derivation qui doivent s'accorder sont une divergence en
    attente.
    """
    import hashlib
    return hashlib.md5(candidate_path.encode()).hexdigest()[:16]


def move_to_durable_store(marked, token, case_key):
    """Deplace un artefact marque vers un stockage qui survit au conteneur.

    Rend le chemin final -- celui du magasin si le deplacement a reussi, sinon
    celui d'origine. NE LEVE JAMAIS: l'appelant est deja en train de lever un
    refus, et remplacer une raison de troncature par une OSError de disque
    detruirait la seule information que le declin porte. On rapporte les deux.

    ON N'ECRASE JAMAIS. Si le nom existe deja au magasin, on suffixe -- deux
    refus du meme candidat sont deux preuves et pas une correction. Ecraser
    detruirait exactement ce que ce magasin existe pour garder.

    `shutil.move` ET NON `os.replace`: le magasin est sur un AUTRE systeme de
    fichiers que le `/tmp` du conteneur, et `os.replace` rend `EXDEV` par-dessus
    une frontiere de montage. C'est precisement le cas normal ici.
    """
    root = os.environ.get(UNDELIVERED_STORE_ENV) or UNDELIVERED_STORE_DEFAULT
    try:
        import shutil
        # NICHE SOUS LA CLE DU CAS. ci: la classification doit vivre dans le NOM
        # -- il apparie `*.REFUSED.*` et `*.NOVERDICT.*` A N'IMPORTE QUELLE
        # PROFONDEUR -- et le chemin ne contribue que l'UNICITE. Nicher ne viole
        # donc pas la regle du nom, et cela resout la collision de deux refus du
        # meme nom de base sans que j'aie a la resoudre.
        directory = path.join(root, case_key) if case_key else root
        tools.make_dirs(directory)
        # `<nom>.<observateur>.<JETON>.<ext>`: le jeton reste AVANT l'extension,
        # comme ci l'a demande, et l'observateur se glisse devant lui.
        stem, extension = path.splitext(path.basename(marked))
        stem, token_part = path.splitext(stem)
        target = path.join(directory,
                           f"{stem}.{observing_container()}{token_part}{extension}")
        attempt = 0
        while path.exists(target):
            attempt += 1
            base, extension = path.splitext(path.basename(target))
            target = path.join(directory, f"{base}.{attempt}{extension}")
        shutil.move(marked, target)
        sys.stderr.write(f"repair: the {token} artefact was moved to durable "
                         f"storage at {target}\n")
        return target
    except Exception as error:
        # UN MAGASIN INDISPONIBLE DEGRADE VERS LE COMPORTEMENT D'AUJOURD'HUI ET
        # LE DIT. Silencieusement, ce serait un disque plein qui redevient une
        # perte de preuves sans que rien ne l'ecrive.
        sys.stderr.write(f"repair: the {token} artefact could NOT be moved to "
                         f"durable storage and stays in the container's "
                         f"ephemeral tree: {error}\n")
        tools.logs.append(f"repair: undelivered_store_failed token={token} "
                          f"reason={type(error).__name__}\n")
        return marked


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
    for position, stream in enumerate(audio):
        duration = stream["duration_ms"]
        if duration == None:
            # ON NE SUBSTITUE PAS LA DUREE DU CONTENEUR.
            #
            # PREMIERE MESURE, vmsam-ci: `format=duration` d'ffprobe differe de la
            # duree du FLUX video et de mediainfo de 31 a 883 ms sur cinq fichiers
            # -- flux video et mediainfo identiques 5 fois sur 5, conteneur
            # different des deux.
            #
            # CE CHIFFRE SOUS-ESTIME LE CAS D'UN FACTEUR 2470. vmsam-dev-3, audit
            # de 60 fichiers sources et 1378 pistes:
            #
            #   conteneur moins max(video, audio), 60 fichiers
            #     min 0.000    mediane 0.091    MAX 2179.937 SECONDES
            #
            #   un fichier declare 3600.000 s de conteneur pour 1420.063 s de
            #   contenu -- un conteneur 2.5 fois plus long que tout ce qu'il porte.
            #
            # ET LE MECANISME A UN NOM. `format=duration` est le MAXIMUM SUR TOUS
            # LES FLUX, et l'exces est un gabarit d'authoring de sous-titres:
            #
            #   fichiers ou le conteneur depasse max(video,audio) de >0.5 s      3
            #   dont le conteneur egale max(sous-titre) a 0.5 s pres          3 / 3
            #   les ecarts                     0.561 s, 59.901 s, 2179.937 s
            #
            # Sur un fichier a six pistes de sous-titres etiquetees
            # `01:00:00.000000000`, la "duree du conteneur" EST ce gabarit.
            #
            # C'est une TROISIEME quantite, pas une approximation de celle-ci, et
            # la glisser ici ferait exactement ce que les trois vocabulaires de
            # noms de codec ont fait ce matin.
            #
            # LA JUSTIFICATION IMPORTE AUTANT QUE LA CONCLUSION: a 31-883 ms un
            # lecteur peut raisonnablement conclure que la substitution est une
            # commodite a petite erreur et la retablir. A 2180 s elle ne l'est
            # sous aucune lecture. Le chiffre qui rend le garde inarguable est
            # celui de dev-3, pas le mien.
            #
            # BORNE DE dev-3 SUR SES PROPRES CHIFFRES: 48 des 60 fichiers sources
            # ne portent AUCUNE etiquette DURATION sur video ni audio, donc les
            # statistiques a >0.5 s reposent sur 12 fichiers, et les 48 forment un
            # bloc d'ids contigu -- un muxeur, pas 80 % au hasard.
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
            entry = {"index": stream["index"], "language": stream["language"],
                     "duration_ms": str(duration), "delta_ms": str(delta)}
            # LE REFUS DIT SI LE MANQUE EST DEJA EXPLIQUE PAR LA SOURCE.
            #
            # `vmsam-ci`, en inspectant le PREMIER artefact refuse que cette
            # campagne ait jamais garde: le `fr` produit est 1988 ms plus court
            # que la DUREE DU MAITRE et 20 ms PLUS LONG que l'audio `fr` DU
            # MAITRE -- parce que le maitre lui-meme porte un `fr` plus court que
            # sa propre image de 2008 ms. LA PISTE EST AUSSI COMPLETE QUE SA
            # SOURCE et le refus disait seulement "-1988.0".
            #
            # `fill_short_by_ms` EST CALCULE PAR L'ASSEMBLEUR ET LA PORTE NE LE
            # CONSULTAIT PAS. Sixieme fois ce soir qu'une quantite existe et
            # n'atteint pas le lecteur qui en a besoin -- et la premiere ou les
            # deux moities sont DANS LE MEME MODULE.
            #
            # ON N'ARRETE PAS DE REFUSER: c'est un arbitrage du proprietaire et
            # une piste plus courte que l'image reste un fait sur le fichier
            # livre. Le refus DIT desormais si le manque est deja explique, pour
            # que "le plan a vise une duree que la source ne pouvait pas fournir"
            # ne se lise pas comme "la reparation a perdu du contenu".
            #
            # LA JOINTURE EST POSITIONNELLE ET C'EST DIT: le i-eme flux audio du
            # fichier produit correspond au i-eme rapport de piste, dans l'ordre
            # du mux. Quand les comptes divergent, la porte le signale DEJA comme
            # un probleme distinct, donc l'hypothese n'est utilisee que la ou elle
            # est verifiee juste au-dessus.
            if position < len(audio_reports):
                fill_short = audio_reports[position].get("fill_short_by_ms")
                if fill_short not in (None, "", "0"):
                    entry["fill_short_by_ms"] = fill_short
            short.append(entry)
    if len(short):
        problems.append("track(s) not running to the master's duration: "
                        + "; ".join(f"stream {s['index']} ({s['language']}) "
                                    f"{s.get('delta_ms', s.get('reason'))}"
                                    + (f" [fill source itself short by "
                                       f"{s['fill_short_by_ms']} ms]"
                                       if s.get("fill_short_by_ms") else "")
                                    for s in short))
    if len(unmeasured):
        problems.append("track(s) whose duration the file does not state: "
                        + "; ".join(f"stream {u['index']} ({u['language']})"
                                    for u in unmeasured))
    report = {"unmeasured": unmeasured,
              "expected_duration_ms": str(master_duration_ms),
              "expected_duration_source": "master video Duration (mediainfo)",
              # CE CHAMP PEUT NE PAS ETRE UNE DUREE DE CONTENU. `format=duration`
              # est le maximum sur TOUS les flux, sous-titres compris, et un
              # gabarit d'authoring `01:00:00` le porte a 3600000 sur un fichier
              # de 1420 s. Le garde ci-dessus empeche l'usage DANGEREUX -- on ne
              # substitue jamais cette valeur a une duree de piste -- mais LE
              # CHAMP EST EMIS, et un lecteur qui voit `container_duration_ms
              # 3600000` a cote de `expected_duration_ms 1420002` n'a aucun champ
              # qui dise que l'ecart est une etiquette de sous-titre plutot qu'un
              # defaut du travail.
              #
              # On emet donc AUSSI le maximum sur les flux video et audio. Leur
              # ECART nomme la situation sans que personne ait a la deviner, et il
              # vaut zero sur un fichier ordinaire.
              "container_duration_ms": str(container_ms) if container_ms != None else None,
              "max_av_stream_duration_ms": (
                  str(max([s["duration_ms"] for s in streams
                           if s["duration_ms"] != None
                           and s["codec_type"] in ("video", "audio")] or [0]))
                  if streams else None),
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
                        "produced_index": produced_index,
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
                           "reference_rms": reference_rms, "produced_rms": produced_rms,
                           "reference_rms": reference_rms,
                           "produced_rms": produced_rms})
        measured = [p for p in probes if p.get("outcome") == "measured"]
        if not len(measured):
            # Toutes les fenetres muettes: la piste est INVERIFIABLE ici. Ce n'est
            # pas un succes, et l'appeler `aligned` serait exactement l'erreur que
            # la campagne poursuit -- un controle qui ne peut pas echouer.
            results.append({"track": report["stream_order"], "language": language,
                        "produced_index": produced_index,
                            "outcome": "skipped",
                            "reason": "no probe window carried signal; the track "
                                      "is unverified, not verified",
                            "probes": probes})
            produced_index += 1
            continue
        worst = max(abs(probe["lag_ms"]) for probe in measured)
        # LA CORRELATION LA PLUS FAIBLE PARMI LES SONDES. Elle etait ENREGISTREE
        # par sonde et n'entrait NULLE PART dans le verdict: `worst` ne regarde
        # que l'amplitude du decalage. Une piste calee sur du contenu SANS
        # RAPPORT rendait donc exactement le meme verdict qu'une piste calee sur
        # le bon programme, pourvu que le pic tombe pres de zero -- et sur de
        # l'audio sans rapport le pic tombe ou il veut.
        #
        # Le cas existe: `vmsam-dev-1` a trouve six fichiers ou AUCUNE paire de
        # flux n'atteint 0.70, c'est-a-dire ou le candidat n'est pas ce
        # programme, et CINQ D'ENTRE EUX NE SONT SUR AUCUNE LISTE cannot-help.
        # Le balayage complet les tentera.
        #
        # ON N'EN FAIT PAS UN SEUIL. Je n'ai aucune population derriere un
        # nombre pour CETTE mesure -- la barre de 0.85 de dev-1 porte sur un
        # appariement de flux en fenetres de 30 s, pas sur une piste PRODUITE en
        # fenetres de 20 s, et l'importer par ressemblance de nom serait la faute
        # que j'ai refusee ailleurs ce soir. On RAPPORTE, et un lecteur peut
        # enfin distinguer "aligne, r=0.98" de "aligne, r=0.31".
        weakest = min(probe["correlation"] for probe in measured)
        # LA BORNE DE SELECTION, A COTE DE LA STATISTIQUE QU'ELLE CONTAMINE.
        #
        # `verified=N/M` est un compte sur des sondes CHOISIES: une fenetre dont
        # le RMS tombe sous `verify_min_rms` est ecartee comme `no_signal`. Le
        # predicat d'appartenance mentionne donc une quantite du signal, et le
        # numerateur est un echantillon selectionne par une propriete du signal.
        #
        # vmsam-dev-3, apres avoir tue sa propre borne intra-plateau pour cette
        # raison exacte: QUAND LE PREDICAT D'APPARTENANCE D'UN ECHANTILLON
        # MENTIONNE LA QUANTITE MESUREE, IMPRIMER LA BORNE DU PREDICAT A COTE DE
        # LA STATISTIQUE. `n_distinct` fait ce travail pour la REPETITION; ceci
        # le fait pour la SELECTION, et rien ne repare la circularite -- on la
        # rend VISIBLE a qui tient le nombre.
        #
        # Un rapport proche de 1 dit que les sondes gardees frolaient le seuil et
        # que le compte est fortement censure. Un rapport tres grand dirait que le
        # seuil n'est jamais contraignant sur des donnees reelles -- ce qui serait
        # une decouverte a part entiere, et pas un repli: un seuil qui ne se
        # declenche jamais ne protege rien.
        quietest = min(min(probe.get("reference_rms", float("inf")),
                           probe.get("produced_rms", float("inf")))
                       for probe in probes
                       if probe.get("reference_rms") != None
                       or probe.get("produced_rms") != None)
        rms_over_floor = (quietest / verify_min_rms
                          if quietest not in (None, float("inf")) else None)
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
                                     "lags_ms": same,
                                     "correlations": [round(p["correlation"], 4)
                                                      for p in measured
                                                      if p["piece"] == index]})
        if len(inconsistent):
            # CE NOMBRE N'EST PAS LA TAILLE DE LA MARCHE MANQUEE, et il se lit
            # comme si c'en etait une.
            #
            # MESURE, id 160: ce champ a rapporte 2869.5 ms. Le profil dense du
            # FICHIER PRODUIT donne une marche de 66.75 ms entre 100 et 110 s, et
            # AUCUNE region au-dela de 500 ms nulle part. FACTEUR 43.
            # `vmsam-dev-1`, mesurant le candidat ORIGINAL, a trouve la meme
            # transition -- 66.8 ms dans [92.8, 139.1] s -- et a etabli que la
            # variation totale de la source sur 23 minutes est de 300.3 ms, donc
            # 2869.5 ms ne pouvait designer aucune structure du fichier.
            #
            # POURQUOI: une fenetre qui CHEVAUCHE une frontiere rend un pic
            # DEPLACE, pas un decalage. Le desaccord entre deux sondes est donc
            # un signal FIABLE qu'une frontiere existe entre elles, et son
            # amplitude n'est PAS une mesure de cette frontiere. La meme classe a
            # ete trouvee le meme jour dans la barre d'appariement du locator,
            # ou une sonde a cheval a fait refuser un fichier entier.
            # LA CORRELATION DE CHAQUE SONDE, DANS LE DESACCORD LUI-MEME.
            #
            # MECANISME REPRODUIT sur id 120, aux positions exactes du
            # verificateur:
            #     piece 6 @ 1340000 ms   lag    0.375   r 0.9521
            #     piece 6 @ 1411972 ms   lag -959.625   r 0.6228
            # La seconde sonde est LES VINGT DERNIERES SECONDES DU FICHIER, a
            # moitie moins de signal. Son pic est SPURIEUX, et le verdict --
            # max(abs(lag)) -- la traite exactement comme la sonde a 0.95.
            # Le profil dense du fichier produit ne trouve AUCUNE marche, sur
            # aucune piste mesurable: CE REFUS EST FAUX.
            #
            # Il y a un plancher de RMS et AUCUN plancher de CORRELATION. Je n'en
            # pose pas un: je n'ai aucune population pour cette mesure-ci, et
            # importer les 0.85 de l'appariement de flux serait la faute de
            # ressemblance de nom refusee trois fois aujourd'hui.
            #
            # On MONTRE donc les correlations dans le desaccord. Un lecteur voit
            # que l'ecart repose sur une sonde a 0.62.
            detail = "; ".join(
                f"piece {c['piece']} probes disagree by {c['spread_ms']:.1f} ms "
                f"{c['lags_ms']} r={c['correlations']} (SPREAD, NOT THE SIZE OF "
                f"THE MISSED STEP: a "
                f"window straddling a boundary returns a displaced peak, so this "
                f"establishes THAT a boundary lies between the probes and not "
                f"how large it is)" for c in inconsistent)
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
                 "produced_index": produced_index,
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
                        "produced_index": produced_index,
                        "outcome": outcome, "worst_lag_ms": worst,
                        "weakest_correlation": round(float(weakest), 4),
                        "probes_measured": len(measured),
                        "probes_without_signal": len(probes) - len(measured),
                        # LA BORNE DE SELECTION, PAS SEULEMENT LE COMPTE.
                        "quietest_probe_rms": quietest,
                        "rms_over_floor": (round(float(rms_over_floor), 2)
                                           if rms_over_floor != None else None),
                        "rms_floor": verify_min_rms,
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
