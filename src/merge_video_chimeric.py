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

    THE PIECES ARRIVE BUILT (stage 5, 2026-09-24). `repair_orchestrator.apply_plan`
    lays the frame-exact boundaries stage 4 resolved as candidate zones and master
    fills, one set per track at that track's own sub-frame offset; this module
    CHECKS that each set covers the master timeline contiguously and builds from
    it. It measures nothing (ADDENDUM 10 d): the frame tiers and edge walks that
    `normalize_segments` used to run here, and its rounding of offsets to the
    frame, were removed with it. A gap between zones is a MASTER piece, never
    emptiness.

Marquage: SPEC_ZONE_A.MD s4. La chaine est posee comme tag de piste Matroska
`VMSAM_FABRICATED` sur le fichier produit ici. Mesure 2026-09-03: ce tag survit
a la premiere passe ffmpeg (`-c copy -map_metadata 0`), a la seconde, au
`mkvmerge --no-global-tags` du split et au `mkvmerge` final.
'''

from decimal import Decimal
from fractions import Fraction
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

# COMBIEN D'ECHANTILLONS TIENNENT DANS UNE TRAME, PAR CODEC -- et `None` quand
# ce nombre N'EST PAS UNE PROPRIETE DU CODEC. Sert UNIQUEMENT a borner de
# combien la fin du dernier bloc d'une piste peut legitimement depasser la fin
# du contenu: un muxeur ne coupe pas une trame en deux, donc la derniere trame
# ecrite deborde d'au plus sa propre duree.
#
# `None` N'EST PAS ZERO ET N'EST PAS UNE VALEUR PAR DEFAUT. Opus (2.5 a 60 ms
# par paquet), Vorbis (blocs courts/longs alternes), FLAC (taille de bloc
# choisie par l'encodeur) et les familles DTS n'ont pas UNE taille de trame; en
# inventer une donnerait une tolerance qui a l'air mesuree. Ces codecs ne
# contribuent donc rien, et l'appelant DIT qu'il n'a pas pu les mesurer.
#
# AAC: 1024 echantillons par trame en LC -- le profil que ce module encode
# (`-c:a aac`). `ffprobe` rend `aac` aussi pour HE-AAC, dont la trame de sortie
# vaut 2048; une piste HE-AAC obtiendrait donc ici une tolerance DEUX FOIS TROP
# PETITE, ce qui rend la garde plus stricte et jamais plus permissive -- le
# sens sur lequel une erreur est acceptable.
audio_codec_frame_samples = {
    "aac": 1024,
    "ac3": 1536,
    "eac3": 1536,
    "mp3": 1152,
    "opus": None,
    "vorbis": None,
    "flac": None,
    "truehd": None,
    "mlp": None,
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
    '''Le plan ne peut pas etre execute. Refus explicite, jamais un fallback.

    Porte `cause`: UN JETON STABLE, pose AU SITE DE LEVEE et non chez
    l'appelant. `merge_video_repair.py:2215` attrape cette classe et doit
    ecrire `cause=<jeton>` dans son journal -- et il ne peut pas le DEDUIRE,
    parce que 23 sites levent cette exception et que la seule chose qui les
    distingue chez l'appelant est un message libre. RECONNAITRE UNE CHAINE
    SERAIT LE DEFAUT QUE `chimeric_bound_error` EXISTE DEJA POUR EVITER: un
    test de message se casse au premier reformulage, et c'est pour cela que la
    sous-classe porte `stream_order` et `bound_ms` en ATTRIBUTS. Meme forme
    ici, une raison de plus: le jeton alimente `excluded_with_stated_cause`,
    LA COLONNE SUR LAQUELLE LA CONDITION DE FIN DU PROPRIETAIRE EST NOTEE.

    `None` PAR DEFAUT, ET CE DEFAUT EST UN REFUS DE DEVINER. Les 23 sites
    passent leur message en positionnel, donc ajouter ce parametre ne change
    le sens d'AUCUN site existant. Deux l'ont pose sur autorisation explicite
    du Lead (R2) -- ceux que la production a reellement fait tourner (18 et 5
    occurrences sur 59 artefacts, mesure dev-cause 2026-09-15): deux sites
    achetent 23 declins sur 26, les 21 autres en achetent 3 pour un large
    diff dans un module porteur. Un TROISIEME est scope IN par l'Architect
    (ruling 2026-09-22, RULING_20260922_NO_BAND_ROUTING.MD, "RAISE SITE 1001
    SCOPED INTO THE TOKENED SET"): la regression cote candidat a :1001-1003,
    premiere occurrence de production 2026-09-22 (errid 25, wave table). Un
    QUATRIEME est scope IN par ce cas (CASE_errid12_untokened_5367.md, errid
    12, wave table pass 8): `delivery_timeline_misalignment` a :5360-5397 --
    la verification post-construction contre le maitre, premiere occurrence
    de production 2026-09-22/23. Un CINQUIEME est scope IN par ce cas
    (CASE_errid50_untokened_1279.md, errids 50 et 58, wave table pass 10):
    `candidate_admission_window_exceeded` a :1279-1283 -- le SEUL site qui leve
    `chimeric_bound_error`, la sous-classe borne, refusant un morceau dont la
    fenetre calculee [candidate_start,candidate_end) sort de la duree du
    candidat lui-meme (mesuree, jamais un reglage); deux occurrences de
    production 2026-09-24, meme fraction de borne inferieure negative sur les
    deux, un seul site.
    Cinq sites portent donc un jeton aujourd'hui; les 19 restants n'en ont
    toujours pas.

    LES AUTRES N'ONT DONC PAS DE JETON, ET C'EST DIT PLUTOT QUE COMBLE.
    L'appelant ecrit alors un SENTINELLE hors de la classe acceptee -- voir
    `merge_video_repair.py` -- et surtout PAS un jeton grossier du genre
    `assembly_refused`: remplir la colonne avec une valeur qui ne classe rien
    est PIRE qu'une colonne vide, parce que ca ressemble a une note.
    '''

    def __init__(self, message, cause=None):
        super().__init__(message)
        self.cause = cause


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
    tools.dev_log(f"chimeric: resolve_source_bitrate starting "
                  f"file={source_path} stream_order={audio.get('StreamOrder')}\n")
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


def split_master_fill_shortfall(pieces, fill_source_ms):
    '''How much `fill_source_ms` (the fill track's own measured extent) falls
    short, split by the REASON of the master piece asking for it.

    SPEC_ZONE_A.MD s4h, "Tail-gap boundary rule" (owner's order, 2026-09-21),
    Lead's ruling shape (a), 2026-09-21: a shortfall caused by the AUTOMATIC
    tail-gap fill reaching for `master_duration_ms` is not a refusal cause;
    a shortfall caused by any other reason (`head_gap`, `interior_bracket`,
    `interior_bracket_frame_narrowed`) still is, unchanged. The two cannot
    be told apart from a single pooled number, which is what this site
    computed before this landing (`max` over every `reason` together).

    SEPARABLE BY CONSTRUCTION, not by convention: `repair_orchestrator.
    track_pieces` merges adjacent master pieces, so at most ONE `tail_gap`
    piece exists, and it ends at `master_duration_ms` exactly -- the end
    `assemble_on_master_timeline` checks every plan reaches, and never
    exceeds.
    So the tail-gap piece can never be shadowed by, or hide, another
    piece's own shortfall: it is provably the largest `master_end_ms` among
    all master pieces whenever it exists, and every other reason's own
    shortfall is computed from the OTHER pieces alone.

    Takes THIS TRACK'S OWN `pieces` (each track gets its own set from
    `repair_orchestrator.apply_plan`) -- so the split is per-track by
    construction, not a file-wide default a later edit could drift into
    applying uniformly.

    Returns `(non_tail_shortfall_ms, tail_shortfall_ms)`, each `Decimal` or
    `None` when that class of master piece does not exist or does not
    exceed `fill_source_ms`.
    '''
    non_tail_ends = [p["master_end_ms"] for p in pieces
                     if p["source"] == "master" and p.get("reason") != "tail_gap"]
    tail_ends = [p["master_end_ms"] for p in pieces
                if p["source"] == "master" and p.get("reason") == "tail_gap"]
    furthest_non_tail = max(non_tail_ends, default=None)
    furthest_tail = max(tail_ends, default=None)
    non_tail_shortfall = (furthest_non_tail - fill_source_ms
                          if furthest_non_tail != None and furthest_non_tail > fill_source_ms
                          else None)
    tail_shortfall = (furthest_tail - fill_source_ms
                      if furthest_tail != None and furthest_tail > fill_source_ms
                      else None)
    return non_tail_shortfall, tail_shortfall


def build_one_audio_track(candidate_obj, master_obj, audio, language, pieces,
                          out_path, timeout, speed_ratio=None,
                          reference_stream=None, comparison_language=None,
                          track_bound_ms=None):
    '''Produit une piste audio chimerique. Renvoie un dict de compte-rendu.

    `track_bound_ms` est l'etendue de CETTE piste -- pas celle du fichier. Le
    parametre s'appelait `candidate_duration_ms` et recevait le maximum sur
    toutes les pistes audio du candidat: la coupe de queue plus bas s'en sert
    comme fin de la source, donc elle annoncait une queue de PLUSIEURS SECONDES
    trop longue des que le fichier portait une piste plus longue que celle-ci
    (mesure: 3005.7 ms et 1835.61 ms sur deux fichiers). C'est le meme nombre
    faux que la garde d'extraction, dans un second consommateur -- et celui-ci
    est un JOURNAL, ecrit pour etre cru.

    Renomme et pas seulement corrige: le nom precedent est ce qui rendait la
    confusion invisible a la relecture.
    '''
    tools.dev_log(f"chimeric: build_one_audio_track starting "
                  f"candidate={candidate_obj.filePath} "
                  f"stream_order={audio.get('StreamOrder')} language={language} "
                  f"out_path={out_path}\n")
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
    fill_short_by_ms_tail_exempt = None
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
            # TAIL-GAP EXEMPTION SPLIT (SPEC_ZONE_A.MD s4h, Lead's ruling
            # shape (a), 2026-09-21): see `split_master_fill_shortfall`'s
            # own docstring for the separability proof. `fill_short_by_ms`
            # now names ONLY the non-tail-gap shortfall -- the quantity
            # `output_check` (below, via `verify_output_file`) still refuses
            # on, unchanged. `fill_short_by_ms_tail_exempt` is new: the
            # amount attributable to THIS track's own `tail_gap` piece
            # alone, per-track because `pieces` here already is.
            fill_short_by_ms, fill_short_by_ms_tail_exempt = (
                split_master_fill_shortfall(pieces, fill_source_ms))
            if fill_short_by_ms_tail_exempt != None:
                # VISIBLE, NEVER SILENT (Lead's condition, 2026-09-21): an
                # exemption that stops a refusal from happening leaves no
                # trace by construction unless it is named here -- this is
                # the PLAN-STAGE prediction; `verify_output_file` logs its
                # own, independent confirmation from the PRODUCED file.
                tools.logs.append(
                    f"chimeric: fill_short_tail_exempt "
                    f"stream_order={audio['StreamOrder']} language={language} "
                    f"exempted_ms={fill_short_by_ms_tail_exempt} reason=tail_gap "
                    f"residual_non_exempt_ms="
                    f"{fill_short_by_ms if fill_short_by_ms != None else '0'}\n")
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
    # `-map_chapters -1`: ffmpeg copies by default the chapters of the first input that has
    # some -- the CANDIDATE's raw editions, or the master's when the candidate has none -- into
    # every file it writes. ADDENDUM 9 points 3 and 7: no raw candidate edition ever reaches
    # the product; the chapters are set once, re-timed, by `mux_chapters`.
    command.extend(["-filter_complex", filtergraph, "-map", "[aout]", "-map_chapters", "-1"])
    command.extend(encoder_arguments)
    command.extend(["-ar", sample_rate, "-vn", "-sn", "-dn",
                    "-max_muxing_queue_size", "16384", out_path])
    tools.dev_log(f"chimeric: build_one_audio_track ffmpeg build call "
                  f"candidate={candidate_obj.filePath} out_path={out_path}\n")
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
                "language": None if fill == "silence" else fill_language,
                # SPEC_ZONE_A.MD s4e, ses propres mots, littéralement: "where
                # each filled region came from -- same-language master,
                # comparison language, or silence -- and the language
                # actually used." Trois jetons, PAS INVENTES ICI -- ceux du
                # texte deja en vigueur, pour qu'un lecteur qui va relire s4e
                # retrouve exactement ces trois noms et aucun autre. Distingue
                # les deux cas MASTER (meme langue que la piste, ou langue de
                # comparaison) que "source"/"language" seuls laissaient a
                # deduire en comparant deux champs plutot que de le nommer.
                "fill_source_class": (
                    "silence" if fill == "silence"
                    else "same_language_master" if fill_language == language
                    else "comparison_language_master"),
                # LE VERDICT DU RAFFINEUR DE CADRES, PAS SEULEMENT SES BORNES.
                # Mesure du Lead (2026-09-16): `piece["frame_tier"]` n'est lu
                # NULLE PART ailleurs dans ce depot -- meme un narrowing
                # REUSSI etait invisible avant ce site, et un declin l'etait
                # doublement. `reason` PAS un booleen: `structure_present_
                # could_not_narrow` et `frames_unextractable` sont deux
                # DEFAUTS DIFFERENTS avec des PROPRIETAIRES differents, et
                # les distinguer est la raison d'etre de cette mission.
                # `None` couvre DEUX cas indiscernables ici a dessein (le
                # raffineur n'a jamais tourne; il a tourne et REUSSI) --
                # les deux se lisent deja par ailleurs sur cette region
                # (bornes narrowed vs. bornes brutes), donc seul le DECLIN
                # a besoin d'un champ pour exister du tout.
                "frame_tier_declined_reason": (
                    piece["frame_tier"]["reason"]
                    if piece.get("frame_tier") and piece["frame_tier"].get("declined")
                    else None),
                # UN SEUL JETON COUVRAIT QUATRE ETATS. `could_not_locate_onset`
                # est rendu par `frame_compare.locate_match_onset` sur QUATRE
                # sites (:1117 ligne de base illisible, :1139 lignes de base
                # qui ne separent pas, :1147 aucune paire lisible dans la
                # parenthese, :1170 calibre mais aucune paire consecutive) --
                # et le journal de production n'en portait que le NOM. Mesure,
                # 2026-09-21: 5 declins de ce jeton sur le corpus produit, dont
                # un remplissage maitre de ~102 s, ET AUCUNE CHAINE `evidence`
                # nulle part -- donc aucun moyen de savoir lequel des quatre a
                # tire. C'est la cinquieme regle de BRIEF_COMMON: *"je n'ai pas
                # pu mesurer" et "ce fichier est inverifiable" sont deux
                # reponses differentes*. La distinction EXISTE deja dans le
                # dict rendu; elle n'atteignait simplement jamais le journal.
                "frame_tier_declined_evidence": (
                    piece["frame_tier"].get("evidence")
                    if piece.get("frame_tier") and piece["frame_tier"].get("declined")
                    else None),
                # SIMILARITY/MARGIN -- REGION A's CARRIER INTO THE PLAN
                # (dev-tiergate mission, 2026-09-16). `frame_compare.py`
                # computes both on every interior result it does not
                # decline -- a validated narrowing and a validation
                # failure alike, since Region A's revert preserves them
                # via a shallow dict copy rather than dropping them.
                # Present but unreachable without digging into
                # `piece["frame_tier"]` is exactly the gap this mission
                # exists to close (measured: no consumer in this
                # repository's history has ever read either field).
                # `None` means no interior tier ran on this piece, or it
                # ran on a head/tail edge where `locate_match_onset`
                # computes neither field at all -- reported as a blocked
                # need, not this landing's to add.
                "frame_tier_similarity": (
                    piece["frame_tier"].get("similarity")
                    if piece.get("frame_tier") else None),
                "frame_tier_margin": (
                    piece["frame_tier"].get("margin")
                    if piece.get("frame_tier") else None)})
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
    if previous_candidate_end != None and track_bound_ms != None:
        tail = Decimal(str(track_bound_ms)) - previous_candidate_end
        if tail > 0:
            # COUPE DE QUEUE: du candidat apres la derniere lecture. Rien ne
            # tournait apres la boucle, donc elle etait invisible.
            # MESUREE, PAS SUPPOSEE: 2516.95 ms sur le fichier meme qui a servi
            # a valider ces lignes -- j'ai regarde l'interieur et jamais
            # au-dela du dernier morceau.
            cut_regions.append({
                "candidate_start_ms": str(previous_candidate_end),
                "candidate_end_ms": str(track_bound_ms),
                "dropped_ms": str(tail), "where": "tail"})
    elif previous_candidate_end != None and track_bound_ms == None:
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
            # THE TAIL-GAP-ATTRIBUTABLE PORTION, NAMED SEPARATELY -- never
            # folded back into `fill_short_by_ms` above, which is exactly
            # the pooling this field exists to undo. `None` on every track
            # with no `tail_gap` piece, or whose fill source reaches it.
            "fill_short_by_ms_tail_exempt": (
                str(fill_short_by_ms_tail_exempt)
                if fill_short_by_ms_tail_exempt != None else None),
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

    Une replique dont le temps ne tombe dans aucun morceau du candidat tombe
    dans un morceau MAITRE -- head_gap, interior_bracket ou tail_gap.

    H-A2 (2026-09-16). Deux versions de cette fonction, dans la meme nuit.
    La PREMIERE retimait ces repliques par le decalage du morceau candidat
    voisin le plus proche -- "la meilleure preuve disponible". Mesuree sur
    id 12 (forensic Table 4): cette preuve est confirmee FAUSSE loin de sa
    piece source (diff=58.07 a l'interieur d'un morceau maitre non modelise,
    contre diff=3.19 au vrai decalage), et confirmee VRAIE tout pres. La
    RULING de l'Architecte (2026-09-16, apres mesure du live locator sur le
    meme id) a tranche: **aucune proximite ne rend un emprunt defendable.**
    Sa raison, plus nette que la mienne: un morceau maitre etroit qu'un plan
    recent (frame-tier, `88290537`+`2b36d376`) laisse encore ouvert EST la
    divergence localisee elle-meme -- une replique qui y tombe designe un
    endroit ou aucun decalage candidat n'est, PAR CONSTRUCTION, defini. Et un
    morceau maitre LARGE est une dette du PLAN (H-A3), jamais un probleme du
    retimer: le corriger ici cacherait le vrai defaut sous un decalage
    invente.

    Donc: UNE SEULE piece candidate emet un decalage -- celle qui couvre
    reellement le temps source de la replique (`:1536`ish). Toute replique
    qui ne tombe dans AUCUNE piece candidate est un DROP NOMME, jamais un
    emprunt -- distingue par la largeur du morceau maitre concerne
    (`gap_ms`) pour qu'un lecteur (ou un futur routeur H-A3) separe la
    divergence etroite, correcte par construction, de la dette de plan
    large. Deux causes, jamais confondues: `dropped_master_filled_span`
    (aucune piece candidate ne couvre ce temps, mais le morceau maitre
    concerne est identifiable) et `dropped_no_offset_evidence` (la piste ne
    porte AUCUNE piece candidate du tout -- aucun morceau n'est meme
    identifiable). Une troisieme cause preexistante, `dropped_degenerate_duration`,
    reste inchangee: l'intervalle degenere (fin <= debut) apres un decalage
    par ailleurs valide.

    UNE REPLIQUE DONT LE DEBUT EST ADMIS PEUT AVOIR UNE FIN QUI DEPASSE LA
    TIMELINE, et rien ne la bornait. Le decalage s'applique aux DEUX bornes;
    seul le DEBUT est teste contre la piece. Une replique qui commence a
    l'interieur de la derniere piece et dure au-dela de la fin du maitre
    sortait donc avec une fin POSTERIEURE A TOUTE la timeline -- et Matroska
    prend la Segment Duration comme le MAXIMUM DES FINS DE BLOC SUR TOUTES LES
    PISTES, sous-titres compris. Mesure (Undead Unluck S01E12, 2026-09-22):
    UNE replique, la derniere du `ja` SDH chimerique, finissait 112 ms apres
    le dernier `master_end_ms` du plan, et le conteneur livre mesurait
    1428039 ms la ou le maitre en mesure 1427944 -- les 95 ms que la
    verification independante a refuses. La fin est donc RECADREE sur la fin
    de timeline (`clamped_to_timeline_end`), jamais silencieusement: le
    recadrage est une DECISION et il porte sa ligne comme une suppression.

    RECADRAGE SUR LA FIN DE TIMELINE SEULEMENT, jamais sur la fin de la piece
    qui a emis le decalage: une replique qui deborde a l'INTERIEUR du plan
    designe un morceau maitre que le plan n'a pas modelise, c'est-a-dire la
    meme dette de plan (H-A3) que ci-dessus, et la trancher ici la cacherait.
    Cette question-la est ouverte et tranchee ailleurs; celle-ci ne l'est pas:
    aucune lecture ne rend defendable un bloc apres la fin du fichier.

    Renvoie (gardees, supprimees, decalages_appliques, decisions).
    `decisions` porte une entree PAR SUPPRESSION OU PAR RECADRAGE, groupee par
    empan contigu de meme nature (jamais par correspondance candidate
    ordinaire -- le cas attendu n'a pas besoin d'etre nomme, sinon la ligne
    finit ignoree). Chaque entree: outcome, cue_count, source_start_ms,
    source_end_ms, shift_ms, piece_reason, gap_ms. `shift_ms` est None sur
    toute SUPPRESSION -- rien n'est plus emprunte -- et porte le decalage
    REELLEMENT applique sur un recadrage, qui est la seule issue ou une
    replique survit a une decision nommee. Sur un recadrage, `gap_ms` est le
    DEBORDEMENT retire (la fin decalee moins la fin de timeline) et non une
    largeur de morceau maitre: une seule replique est concernee a la fois,
    donc l'entree n'est jamais groupee.
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
    piece_index = {id(p): i for i, p in enumerate(pieces)}
    # LA FIN DE LA TIMELINE, LUE SUR LE PLAN LUI-MEME ET NON RECUE EN
    # PARAMETRE. `assemble_on_master_timeline` verifie deja que le dernier
    # `master_end_ms` de chaque plan EST `master_duration_ms` (il refuse tout
    # plan qui n'atteint pas exactement la fin de timeline). La calculer ici plutot que d'ajouter un
    # parametre garde la fonction testable avec des `pieces` litterales et
    # rend IMPOSSIBLE qu'un appelant passe une fin en desaccord avec le plan
    # qu'il passe dans la meme main -- deux valeurs a tenir d'accord est
    # exactement la forme qui a deja diverge deux fois dans ce module.
    # `max` et non `pieces[-1]`: l'ordre de la liste est une propriete de
    # `normalize_segments`, pas un contrat de cette fonction-ci.
    timeline_end_ms = max(Decimal(str(p["master_end_ms"])) for p in pieces)

    def bordering_master(piece, direction):
        '''Le morceau juste avant (direction=-1) ou apres (+1) `piece` dans
        `pieces`, si c'est un morceau MAITRE -- sinon None.'''
        i = piece_index[id(piece)] + direction
        if 0 <= i < len(pieces) and pieces[i]["source"] == "master":
            return pieces[i]
        return None

    def find_gap_piece(event_start):
        '''Le morceau MAITRE dans lequel tombe un temps source non couvert
        par aucune piece candidate -- SEULEMENT pour NOMMER le drop (raison,
        largeur). Ne calcule aucun decalage: l'Architecte a tranche qu'aucun
        n'est defendable ici. None si la piste ne porte aucune piece
        candidate du tout (population separee: `dropped_no_offset_evidence`).'''
        if not candidate_pieces:
            return None
        prev_piece, next_piece = None, None
        for p in candidate_pieces:
            source_start = p["source_start_ms"]
            source_end = source_start + (p["master_end_ms"] - p["master_start_ms"])
            if source_end <= event_start:
                prev_piece = p
            elif next_piece is None and source_start > event_start:
                next_piece = p
        if next_piece is not None:
            return bordering_master(next_piece, -1)
        if prev_piece is not None:
            return bordering_master(prev_piece, +1)
        return None

    kept_events = []
    kept_meta = {}
    dropped_master_filled_span = 0
    dropped_no_offset_evidence = 0
    dropped_degenerate_duration = 0
    # LES DECALAGES REELLEMENT APPLIQUES, comptes par morceau. Une piste dont
    # toutes les repliques tombent dans UN morceau rend un seul decalage -- ce
    # qui est precisement pourquoi deux langues atterrissent sur la meme
    # constante, et ce n'est pas un defaut.
    applied = {}
    decisions = []
    pending_span = None

    def flush_span():
        nonlocal pending_span
        if pending_span is not None:
            pending_span.pop("_key")
            decisions.append(pending_span)
            pending_span = None

    def record(outcome, original_start, original_end, shift, master_piece):
        nonlocal pending_span
        span_key = (outcome, str(shift), id(master_piece) if master_piece else None)
        if pending_span is not None and pending_span["_key"] == span_key:
            pending_span["cue_count"] += 1
            pending_span["source_end_ms"] = str(original_end)
        else:
            flush_span()
            pending_span = {
                "_key": span_key, "outcome": outcome, "cue_count": 1,
                "source_start_ms": str(original_start),
                "source_end_ms": str(original_end),
                "shift_ms": str(shift) if shift is not None else None,
                "piece_reason": master_piece.get("reason") if master_piece else None,
                "gap_ms": (str(master_piece["master_end_ms"] - master_piece["master_start_ms"])
                          if master_piece else None)}

    for event in subtitles.events:
        original_start, original_end = event.start, event.end
        shift = None
        # LA PIECE EST CAPTUREE, PAS LAISSEE A LA VARIABLE DE BOUCLE. `piece`
        # survit au `for` en Python et vaut la DERNIERE piece essayee quand
        # aucune ne correspond -- la lire apres coup nommerait une piece qui
        # n'a rien decide. Le recadrage ci-dessous doit citer la piece qui a
        # reellement emis le decalage, donc elle est nommee ici.
        matched_piece = None
        for piece in candidate_pieces:
            source_start = piece["source_start_ms"]
            source_end = source_start + (piece["master_end_ms"] - piece["master_start_ms"])
            if Decimal(str(event.start)) >= source_start and Decimal(str(event.start)) < source_end:
                shift = piece["master_start_ms"] - source_start
                matched_piece = piece
                break
        if shift == None:
            gap_piece = find_gap_piece(Decimal(str(event.start)))
            if gap_piece is not None:
                dropped_master_filled_span += 1
                record("dropped_master_filled_span", original_start, original_end, None, gap_piece)
            else:
                dropped_no_offset_evidence += 1
                record("dropped_no_offset_evidence", original_start, original_end, None, None)
            continue
        applied[str(shift)] = applied.get(str(shift), 0) + 1
        event.start = int(event.start + shift)
        event.end = int(event.end + shift)
        if Decimal(str(event.end)) > timeline_end_ms:
            # LE BLOC NE PEUT PAS FINIR APRES LE FICHIER. La Segment Duration
            # de Matroska est le maximum des fins de bloc sur TOUTES les
            # pistes: une fin de replique posee au-dela de la timeline
            # ALLONGE le conteneur livre, et le controle de duree ne regardait
            # que l'audio piste par piste et le COMPTE des sous-titres.
            #
            # AVANT LA GARDE DE DEGENERESCENCE, et l'ordre est le fond: une
            # replique dont le debut decale atteint deja la fin de timeline
            # recadre vers `fin <= debut` et doit etre SUPPRIMEE, pas gardee a
            # duree nulle. La garde existante le fait, le nomme
            # (`dropped_degenerate_duration`) et n'a pas besoin d'etre
            # repetee ici. Cette replique-la porte alors DEUX entrees -- le
            # recadrage, puis la suppression qu'il a causee -- et c'est le
            # compte rendu exact: deux choses lui sont arrivees, dans cet
            # ordre. Les fusionner dirait qu'une seule a eu lieu.
            #
            # `flush_span()` d'abord: un empan de suppressions en cours ne
            # doit pas se poursuivre PAR-DESSUS cette entree-ci, sinon les
            # decisions ne se lisent plus dans l'ordre des repliques.
            overhang_ms = Decimal(str(event.end)) - timeline_end_ms
            flush_span()
            decisions.append({
                "outcome": "clamped_to_timeline_end", "cue_count": 1,
                "source_start_ms": str(original_start),
                "source_end_ms": str(original_end),
                "shift_ms": str(shift),
                "piece_reason": matched_piece.get("reason"),
                "gap_ms": str(overhang_ms)})
            event.end = int(timeline_end_ms)
        if event.end <= event.start:
            dropped_degenerate_duration += 1
            record("dropped_degenerate_duration", original_start, original_end, shift, None)
            continue
        flush_span()  # correspondance candidate ordinaire: attendue, non nommee, mais ne fusionne pas a travers elle
        kept_events.append(event)
        kept_meta[id(event)] = (matched_piece, original_start, original_end, shift)
    flush_span()
    # LA SEULE INTELLIGENCE DE L'APPLICATION DU PLAN (ADDENDUM 10 d): aux
    # raccords, jamais deux fois la meme replique, et une replique s'arrete
    # avant la suivante au lieu de la chevaucher.
    kept_events, hygiene = splice_cue_hygiene(kept_events, kept_meta)
    decisions.extend(hygiene)
    dropped_at_splice = sum(1 for d in hygiene if d["outcome"].startswith("dropped"))
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
    return (len(kept_events),
            dropped_master_filled_span + dropped_no_offset_evidence + dropped_degenerate_duration
            + dropped_at_splice,
            applied, decisions)


# COMBIEN DE REPLIQUES DE PART ET D'AUTRE D'UN RACCORD SONT COMPAREES pour y
# trouver un doublon (ADDENDUM 9 point 8: "les N cues de part et d'autre"). Un
# raccord deplace au plus quelques repliques l'une contre l'autre: la meme
# phrase lue deux fois tombe dans les deux ou trois voisines, jamais plus loin,
# parce que les deux lectures sont a la MEME position maitre a un decalage de
# plan pres. Trois couvre une replique intercalee de chaque cote.
SPLICE_DEDUP_NEIGHBOURS = 3


def normalized_cue_text(event):
    """Le texte d'une replique tel qu'on le compare pour un doublon: sans balises
    (`plaintext` de pysubs2 retire les `{\\...}` et rend les `\\N` en fin de
    ligne), sans casse, sans ponctuation, espaces reduits (ADDENDUM 9 point 8)."""
    import unicodedata
    text = event.plaintext.lower()
    kept = "".join(ch if not unicodedata.category(ch).startswith("P") else " "
                   for ch in text)
    return " ".join(kept.split())


def splice_cue_hygiene(events, meta):
    """Les deux regles de sous-titres de l'application du plan, et SEULEMENT aux
    raccords -- entre deux repliques venues de DEUX MORCEAUX differents du plan.

    1. PAS DE DOUBLON: une replique dont le texte normalise egale celui d'une des
       `SPLICE_DEDUP_NEIGHBOURS` suivantes d'un autre morceau, et qui la
       chevauche ou la touche dans le temps, est la meme phrase lue deux fois
       par le raccord: la seconde est retiree (`dropped_duplicate_at_splice`).
    2. PAS DE CHEVAUCHEMENT: une replique qui depasse le debut de la prochaine
       replique d'un AUTRE morceau est ARRETEE a ce debut
       (`truncated_before_next_cue`); si elle n'a plus de duree, elle est
       retiree (`dropped_degenerate_after_truncation`).

    POURQUOI PAS ENTRE DEUX REPLIQUES DU MEME MORCEAU: leur relation est celle
    de la SOURCE (un panneau ASS pose pendant un dialogue se chevauche par
    construction), et le plan ne l'a pas creee; y toucher serait modifier un
    fichier au-dela de ce que le plan applique.

    `meta[id(event)] = (morceau, debut_source, fin_source, decalage)`. Renvoie
    (repliques gardees dans leur ordre d'origine, decisions nommees)."""
    order = sorted(range(len(events)), key=lambda i: (events[i].start, events[i].end))
    dropped = set()
    decisions = []

    def decide(outcome, index, gap_ms):
        piece, source_start, source_end, shift = meta[id(events[index])]
        decisions.append({
            "outcome": outcome, "cue_count": 1,
            "source_start_ms": str(source_start), "source_end_ms": str(source_end),
            "shift_ms": str(shift), "piece_reason": (piece or {}).get("reason"),
            "gap_ms": str(gap_ms)})

    def piece_of(index):
        return id(meta[id(events[index])][0])

    for rank, i in enumerate(order):
        if i in dropped:
            continue
        text = normalized_cue_text(events[i])
        if not text:
            continue
        seen = 0
        for j in order[rank + 1:]:
            if j in dropped:
                continue
            seen += 1
            if seen > SPLICE_DEDUP_NEIGHBOURS:
                break
            if piece_of(j) == piece_of(i):
                continue
            if (events[j].start <= events[i].end
                    and normalized_cue_text(events[j]) == text):
                dropped.add(j)
                decide("dropped_duplicate_at_splice", j, events[i].end - events[j].start)

    for rank, i in enumerate(order):
        if i in dropped:
            continue
        following = next((j for j in order[rank + 1:]
                          if j not in dropped and piece_of(j) != piece_of(i)
                          and events[j].start >= events[i].start), None)
        if following is None or events[i].end <= events[following].start:
            continue
        overlap = events[i].end - events[following].start
        events[i].end = events[following].start
        if events[i].end <= events[i].start:
            dropped.add(i)
            decide("dropped_degenerate_after_truncation", i, overlap)
        else:
            decide("truncated_before_next_cue", i, overlap)
    return [event for index, event in enumerate(events) if index not in dropped], decisions


def build_one_subtitle_track(candidate_obj, subtitle, language, pieces, work_dir,
                             index, timeout, speed_ratio=None):
    '''Extrait, re-cale, renvoie un dict de compte-rendu -- ou leve.'''
    tools.dev_log(f"chimeric: build_one_subtitle_track starting "
                  f"candidate={candidate_obj.filePath} "
                  f"stream_order={subtitle.get('StreamOrder')} language={language} "
                  f"work_dir={work_dir}\n")
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
               "-map", f"0:{int(subtitle['StreamOrder'])}", "-map_chapters", "-1",
               "-c:s", target, out_path]
    tools.dev_log(f"chimeric: build_one_subtitle_track ffmpeg extract call "
                  f"candidate={candidate_obj.filePath} out_path={out_path}\n")
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
    kept, dropped, shifts_applied, decisions = retime_subtitle_file(out_path, pieces, speed_ratio)
    # UNE SUPPRESSION EST UNE DECISION ET RECOIT UNE LIGNE --
    # H-A2 (2026-09-16), meme forme que ea233239 (mergeVideo.py:1746-1747):
    # le drapeau que le code portait deja implicitement se dit maintenant a
    # cote du nombre. NI chemin NI titre NI nom de fichier ici (PRIVACY,
    # BRIEF_COMMON.md): stream_order, langue, l'issue, les bornes SOURCE (le
    # temps du candidat extrait, pas un chemin), le compte et la cause.
    # Verifiee des DEUX cotes (tools/verify_keep_false_line.py-style): une
    # piste sans decision n'ecrit RIEN ici, une piste avec en ecrit une par
    # empan -- et ceci tourne que `tools.dev` soit vrai ou faux, ce n'est pas
    # un aide au debogage.
    for decision in decisions:
        tools.logs.append(
            f"chimeric: subtitle stream_order={subtitle['StreamOrder']} "
            f"language={language} decision={decision['outcome']} "
            f"cue_count={decision['cue_count']} "
            f"source_span_ms=[{decision['source_start_ms']},{decision['source_end_ms']}) "
            f"shift_ms={decision['shift_ms']} piece_reason={decision['piece_reason']} "
            f"gap_ms={decision['gap_ms']}\n")
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
            # PAR SUPPRESSION OU PAR RECADRAGE, jamais par correspondance
            # ordinaire (le cas attendu n'a pas besoin d'une ligne) -- la
            # famille de `head_decisions` / `cut_regions` (audio), etendue au
            # sous-titre pour fermer l'asymetrie que H-A2 a mesuree: le
            # compte-rendu audio portait des DECISIONS avec bornes, celui des
            # sous-titres ne portait que des comptes.
            "subtitle_decisions": decisions,
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
                      timeout, job_start_utc, chapters_path=None):
    '''Assemble les pistes produites et pose les tags VMSAM_FABRICATED et VMSAM_ERA.

    Le tag est pose ici, sur le fichier de la reparation, et non dans
    `generate_new_file`: cette fonction est hors zone taguee
    (`WRITE_ZONES.MD` s2), et la mesure du 2026-09-03 montre que le tag survit
    de toute facon aux deux passes ffmpeg et aux deux mkvmerge.

    VMSAM_ERA (Architect's ruling, 2026-09-16): l'ERE d'un artefact -- quelle
    revision de code, quel job -- doit se lire DANS l'artefact, jamais se
    deviner. C'est le SEUL site de mux du chemin de reparation: `REFUSED.mkv`,
    le marquage `NOVERDICT` et la copie dans le magasin durable renomment ou
    deplacent CE MEME fichier apres coup, jamais un second mux -- donc poser le
    tag ici suffit pour les quatre. `job_start_utc` est fourni par l'appelant
    (capture au sommet de la boucle par-candidat, `merge_video_repair.py`) et
    non recalcule ici: cette fonction n'a aucune idee de quand le job a
    commence, seulement de quand elle-meme tourne. TESTER LA PRESENCE, jamais
    l'egalite du champ entier -- meme discipline que VMSAM_FABRICATED ("test
    truthiness, never equality"): un champ ajoute plus tard ne doit rien casser
    chez qui lit celui-ci aujourd'hui.
    '''
    tools.dev_log(f"chimeric: mux_repaired_file starting out_path={out_path} "
                  f"n_audio={len(audio_reports)} n_subtitle={len(subtitle_reports)} "
                  f"chapters={chapters_path}\n")
    command = [tools.software["ffmpeg"], "-y", "-nostdin"]
    for report in audio_reports:
        command.extend(["-i", report["path"]])
    for report in subtitle_reports:
        command.extend(["-i", report["path"]])

    for i in range(len(audio_reports) + len(subtitle_reports)):
        command.extend(["-map", f"{i}:0"])
    # NO CHAPTERS FROM THE INPUTS (see `build_one_audio_track`): the only chapters this file
    # carries are the ones `mux_chapters` sets from the re-timed XML.
    command.extend(["-map_chapters", "-1", "-c", "copy"])

    era_value = f"git_commit={tools.get_git_commit()} job_start_utc={job_start_utc}"
    # LE MARQUEUR DE CHAQUE PISTE EST LE SIEN (`report["marker"]`, pose par
    # l'assemblage au facteur que CETTE piste a recu); `marker_value` reste le
    # repli d'un compte-rendu qui n'en porterait pas.
    for i, report in enumerate(audio_reports):
        command.extend([f"-metadata:s:a:{i}",
                        f"VMSAM_FABRICATED={report.get('marker', marker_value)}"])
        command.extend([f"-metadata:s:a:{i}", f"VMSAM_ERA={era_value}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:a:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:a:{i}", f"title={report['title']}"])
    for i, report in enumerate(subtitle_reports):
        command.extend([f"-metadata:s:s:{i}",
                        f"VMSAM_FABRICATED={report.get('marker', marker_value)}"])
        command.extend([f"-metadata:s:s:{i}", f"VMSAM_ERA={era_value}"])
        if report["language"] != None and report["language"] != "und":
            command.extend([f"-metadata:s:s:{i}", f"language={report['language']}"])
        if report["title"] != None:
            command.extend([f"-metadata:s:s:{i}", f"title={report['title']}"])

    command.extend(["-max_muxing_queue_size", "16384", out_path])
    tools.dev_log(f"chimeric: mux_repaired_file ffmpeg mux call "
                  f"out_path={out_path}\n")
    tools.launch_cmdExt_with_timeout_reload(command, 1, timeout)
    if chapters_path is not None:
        mux_chapters(out_path, chapters_path, timeout)


def mux_chapters(out_path, chapters_path, timeout):
    '''Pose le XML de chapitres (deja re-cale, `build_delivered_chapters`) sur le
    fichier produit: `mkvmerge --chapters` (ADDENDUM 9 point 7), en remux par
    copie -- les pistes, leurs octets et leurs tags de piste (VMSAM_FABRICATED,
    VMSAM_ERA) traversent inchanges, mesure a la premiere production de ce
    site. mkvmerge rend 1 sur un simple avertissement: 0 et 1 sont des succes,
    2 est une erreur, et une erreur ici LEVE -- un fichier dont les chapitres
    n'ont pas pu etre poses n'est pas le fichier que le plan decrit.'''
    remuxed = out_path + ".chapters.mkv"
    command = [tools.software["mkvmerge"], "-q", "-o", remuxed,
               "--chapters", chapters_path, out_path]
    tools.dev_log(f"chimeric: mux_chapters mkvmerge call out_path={out_path} "
                  f"chapters={chapters_path}\n")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode not in (0, 1) or not path.exists(remuxed):
        raise chimeric_error(
            f"mkvmerge could not set the chapters on the produced file (exit "
            f"{completed.returncode}): {(completed.stdout + completed.stderr).strip()[-300:]}")
    replace_file(remuxed, out_path)


def extract_chapters_xml(file_path, out_path, timeout=120):
    """`mkvextract <fichier> chapters <xml>` -> `(racine ElementTree | None, raison)`.

    `raison` vaut `extracted`, ou dit POURQUOI il n'y a rien: `no_chapters` (le
    fichier n'en porte pas -- un fait, pas une panne), `mkvextract_not_configured`,
    `mkvextract_exit_<n>`, `timeout`, `unparseable(<type>)`. "Pas de chapitres" et
    "je n'ai pas pu les lire" sont deux reponses differentes."""
    import xml.etree.ElementTree as ElementTree
    tool = tools.software.get("mkvextract")
    if not tool:
        return None, "mkvextract_not_configured"
    command = [tool, file_path, "chapters", out_path]
    tools.dev_log(f"chimeric: extract_chapters_xml mkvextract call file={file_path} "
                  f"out={out_path}\n")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if completed.returncode not in (0, 1):
        return None, f"mkvextract_exit_{completed.returncode}"
    if not path.exists(out_path) or not path.getsize(out_path):
        return None, "no_chapters"
    try:
        root = ElementTree.parse(out_path).getroot()
    except Exception as error:
        return None, f"unparseable({type(error).__name__})"
    if root.find("EditionEntry") is None:
        return None, "no_chapters"
    return root, "extracted"


CHAPTER_TIME_PATTERN = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$")


def parse_chapter_time_ms(text):
    """`HH:MM:SS.nnnnnnnnn` -> millisecondes EXACTES (Decimal), ou None."""
    match = CHAPTER_TIME_PATTERN.match(text or "")
    if match is None:
        return None
    hours, minutes, seconds, fraction = match.groups()
    nanoseconds = int((fraction or "0").ljust(9, "0"))
    return (Decimal(int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * Decimal(1000)
            + Decimal(nanoseconds) / Decimal(1000000))


def format_chapter_time(ms):
    """Millisecondes -> `HH:MM:SS.nnnnnnnnn`, a la nanoseconde (Matroska)."""
    nanoseconds = int((Decimal(str(ms)) * Decimal(1000000)).to_integral_value())
    seconds, nanoseconds = divmod(max(0, nanoseconds), 1000000000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{nanoseconds:09d}"


def map_candidate_time_to_master(equivalent_ms, candidate_pieces):
    """Le temps candidat (timeline corrigee sur une paire a taux) -> la timeline du
    MAITRE, par la fonction par morceaux du plan. `(ms_maitre | None, comment)`:
      mapped                   le temps tombe dans un morceau candidat: temps
                               moins le decalage de CE morceau
      snapped_to_next_piece    il tombe dans du contenu candidat que le plan a
                               COUPE: sa place sur le maitre est le debut du
                               morceau suivant, la ou la coupe se referme
      past_last_piece          apres le dernier contenu candidat lu: aucune place
    """
    following = None
    for piece in candidate_pieces:
        start = Decimal(str(piece["source_start_ms"]))
        length = Decimal(str(piece["master_end_ms"])) - Decimal(str(piece["master_start_ms"]))
        if start <= equivalent_ms < start + length:
            return (Decimal(str(piece["master_start_ms"])) + (equivalent_ms - start),
                    "mapped")
        if start > equivalent_ms and (following is None
                                      or start < Decimal(str(following["source_start_ms"]))):
            following = piece
    if following is not None:
        return Decimal(str(following["master_start_ms"])), "snapped_to_next_piece"
    return None, "past_last_piece"


def build_delivered_chapters(master_path, candidate_path, reference_pieces, time_scale,
                             timeline_ms, work_dir):
    """LES CHAPITRES DU FICHIER PRODUIT (ADDENDUM 9 points 3 et 7), en XML.

    Les editions du MAITRE sont conservees TELLES QUELLES: elles sont deja sur la
    timeline du maitre, a decalage nul. Une edition du CANDIDAT n'est livree que
    RE-CALEE, jamais brute: chaque `ChapterTimeStart`/`ChapterTimeEnd` passe par
    le facteur de vitesse (paire a taux) puis par la fonction par morceaux du
    plan (`map_candidate_time_to_master`), et est recadre sur la fin de
    timeline. Ses UID sont retires pour que mkvmerge en pose de neufs -- une
    edition candidate copiee du meme disque que le maitre porterait sinon les
    MEMES UID. Une edition ORDONNEE (segment linking) n'est pas re-calable par
    une fonction de temps: elle n'est pas livree, et c'est dit.

    TOUTE DECISION CHAPITRE EST JOURNALISEE (point 7), une ligne par atome
    candidat et une par edition maitre. Renvoie `(chemin_xml | None, decisions)`
    -- None quand aucun des deux fichiers ne porte de chapitres."""
    import xml.etree.ElementTree as ElementTree
    decisions = []
    master_root, master_reason = extract_chapters_xml(
        master_path, path.join(work_dir, "chapters_master.xml"))
    candidate_root, candidate_reason = extract_chapters_xml(
        candidate_path, path.join(work_dir, "chapters_candidate.xml"))
    decisions.append({"source": "master", "extraction": master_reason})
    decisions.append({"source": "candidate", "extraction": candidate_reason})
    out_root = ElementTree.Element("Chapters")
    master_editions = list(master_root.findall("EditionEntry")) if master_root is not None else []
    for number, edition in enumerate(master_editions):
        out_root.append(edition)
        decisions.append({"source": "master", "edition": number, "decision": "kept_as_is",
                          "atoms": len(list(edition.iter("ChapterAtom")))})
    candidate_pieces = [piece for piece in reference_pieces if piece["source"] == "candidate"]
    scale = Decimal(str(time_scale)) if time_scale is not None else Decimal(1)
    for number, edition in enumerate(candidate_root.findall("EditionEntry")
                                     if candidate_root is not None else []):
        if (edition.findtext("EditionFlagOrdered") or "0").strip() == "1":
            decisions.append({"source": "candidate", "edition": number,
                              "decision": "not_delivered_ordered_edition"})
            continue
        for tag in ("EditionUID",):
            for element in edition.findall(tag):
                edition.remove(element)
        if len(master_editions):
            flag = edition.find("EditionFlagDefault")
            if flag is None:
                flag = ElementTree.SubElement(edition, "EditionFlagDefault")
            flag.text = "0"
        dropped = []
        for parent in [edition] + list(edition.iter("ChapterAtom")):
            for atom in list(parent.findall("ChapterAtom")):
                for element in atom.findall("ChapterUID"):
                    atom.remove(element)
                title = atom.findtext("ChapterDisplay/ChapterString")
                start_in = parse_chapter_time_ms(atom.findtext("ChapterTimeStart"))
                if start_in is None:
                    dropped.append((parent, atom))
                    decisions.append({"source": "candidate", "edition": number,
                                      "atom": title, "decision": "dropped_unreadable_start"})
                    continue
                start_out, start_how = map_candidate_time_to_master(start_in * scale,
                                                                    candidate_pieces)
                if start_out is None or start_out >= timeline_ms:
                    dropped.append((parent, atom))
                    decisions.append({"source": "candidate", "edition": number,
                                      "atom": title, "start_in_ms": str(start_in),
                                      "decision": f"dropped_{start_how}"
                                      if start_out is None else "dropped_past_timeline"})
                    continue
                atom.find("ChapterTimeStart").text = format_chapter_time(start_out)
                end_element = atom.find("ChapterTimeEnd")
                end_in = end_out = None
                end_how = "absent"
                if end_element is not None:
                    end_in = parse_chapter_time_ms(end_element.text)
                    if end_in is None:
                        atom.remove(end_element)
                        end_how = "removed_unreadable"
                    else:
                        end_out, end_how = map_candidate_time_to_master(end_in * scale,
                                                                        candidate_pieces)
                        if end_out is None or end_out > timeline_ms:
                            end_out, end_how = timeline_ms, f"clamped_to_timeline_end({end_how})"
                        if end_out < start_out:
                            end_out, end_how = start_out, f"raised_to_start({end_how})"
                        end_element.text = format_chapter_time(end_out)
                decisions.append({"source": "candidate", "edition": number, "atom": title,
                                  "start_in_ms": str(start_in), "start_out_ms": str(start_out),
                                  "start": start_how,
                                  "end_in_ms": None if end_in is None else str(end_in),
                                  "end_out_ms": None if end_out is None else str(end_out),
                                  "end": end_how})
        for parent, atom in dropped:
            parent.remove(atom)
        if edition.find("ChapterAtom") is None:
            decisions.append({"source": "candidate", "edition": number,
                              "decision": "not_delivered_no_atom_left"})
            continue
        out_root.append(edition)
        decisions.append({"source": "candidate", "edition": number,
                          "decision": "delivered_retimed"})
    if out_root.find("EditionEntry") is None:
        return None, decisions
    out_path = path.join(work_dir, "chapters_delivered.xml")
    ElementTree.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path, decisions


def probe_delivered_durations(file_path, timeout=300):
    """DELIVERED_DURATIONS (ADDENDUM 10 b): ce que le FICHIER CHIMERIQUE mesure,
    par ffprobe, juste apres l'application du plan -- le mux final est une copie
    de flux, ces durees ne bougent plus. Conteneur, chaque flux (par son type et
    son index), et la fin de la derniere replique par piste de sous-titres
    (max pts+duree des paquets). C'est ce que le verificateur forensique
    compare. Renvoie un dict; une valeur illisible vaut None, jamais 0."""
    import json as _json
    result = {"container_ms": None, "streams": [], "max_cue_end_ms": None}
    command = [tools.software["ffprobe"], "-v", "error", "-show_entries",
               "format=duration:stream=index,codec_type,duration:stream_tags=DURATION",
               "-of", "json", file_path]
    tools.dev_log(f"chimeric: probe_delivered_durations ffprobe call file={file_path}\n")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    data = _json.loads(completed.stdout or "{}")
    try:
        result["container_ms"] = str(Decimal(str(data["format"]["duration"])) * 1000)
    except Exception:
        pass
    subtitle_indices = []
    for stream in data.get("streams") or []:
        duration = stream.get("duration") or (stream.get("tags") or {}).get("DURATION")
        value = None
        if duration not in (None, "N/A"):
            try:
                value = (str(Decimal(str(duration)) * 1000) if ":" not in str(duration)
                         else str(parse_chapter_time_ms(str(duration))))
            except Exception:
                value = None
        result["streams"].append({"index": stream.get("index"),
                                  "type": stream.get("codec_type"), "duration_ms": value})
        if stream.get("codec_type") == "subtitle":
            subtitle_indices.append(stream.get("index"))
    cue_ends = []
    for index in subtitle_indices:
        command = [tools.software["ffprobe"], "-v", "error", "-select_streams", str(index),
                   "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0",
                   file_path]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        last = None
        for line in completed.stdout.splitlines():
            parts = line.split(",")
            try:
                end = (Decimal(parts[0]) + Decimal(parts[1])) * 1000
            except Exception:
                continue
            last = end if last is None or end > last else last
        if last is not None:
            cue_ends.append(last)
    if cue_ends:
        result["max_cue_end_ms"] = str(max(cue_ends))
    return result


def iterate_candidate_audios(candidate_obj):
    """Les pistes que la reparation a le droit de reconstruire.

    COMMENTAIRES: exclusion LEVEE (owner, 2026-09-16) -- "on repare
    toujours... la perte est accepte. vraiment le script doit juste reparer
    avec le plan. le reste j'en fais mon affaire." Un commentaire est
    reconstruit avec le meme plan que n'importe quelle autre classe audio et
    marque `fabricated` comme elle (`mux_repaired_file` tague chaque piste
    produite sans distinguer son holder d'origine); qu'une piste de
    commentaire intacte batte ensuite la reconstruite est un resultat ACCEPTE,
    pas un defaut -- et aucune logique de keep-best n'est inventee ici pour
    eux: rien dans `mergeVideo.py` ne fait jamais passer `.commentary` par
    `keep_best_audio` (grep: seul `tools.special_params["remove_commentary"]`
    decide de leur `keep`), donc cette levee ne change que ce que CE module a
    le droit de reconstruire, pas comment le reste du pipeline les traite.

    L'AUDIO-DESCRIPTION EST DELIBEREMENT LAISSEE OUVERTE. Meme forme, et ce
    n'est PAS ce sur quoi le proprietaire a statue: "Raise it rather than
    extend this by analogy." Cela paraitra incoherent dans le code et c'est
    correct tant que la question n'est pas tranchee.
    """
    for holder in (candidate_obj.audios, candidate_obj.audiodesc, candidate_obj.commentary):
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


def get_master_container_length_ms(master_obj):
    '''La duree de CONTENEUR du maitre -- une TROISIEME quantite, distincte de
    `get_master_timeline_length_ms`, et `None` quand le maitre ne la declare
    pas.

    Ce n'est PAS la duree de la piste video, et l'ecart n'est pas theorique:
    sur le maitre d'Undead Unluck S01E12, video `Duration` 1427927 ms et
    General `Duration` 1427944 ms -- 17 ms, mesures le 2026-09-22. La
    difference est celle que Matroska cree par construction: la Segment
    Duration est le MAXIMUM DES FINS DE BLOC SUR TOUTES LES PISTES, donc une
    piste quelconque qui finit apres l'image la porte.

    C'est donc la SEULE reference contre laquelle la duree de conteneur du
    fichier produit se compare. La comparer a la duree VIDEO ferait refuser
    tout fichier dont le maitre porte deja ce depassement -- le controle
    accuserait la reparation d'un fait de la reference, exactement l'erreur
    que `fill_short_by_ms` existe pour ne plus commettre du cote audio.

    Lue sur `mediadata` deja en memoire et non sondee: mesure comparative sur
    ce meme maitre, `mediainfo` General `Duration` = 1427.944 s et `ffprobe
    format=duration` = 1427.944000 s -- la MEME quantite, une lecture
    gratuite contre un processus. `None` plutot qu'un repli sur la duree
    video: un maitre qui ne declare pas sa duree de conteneur est NON MESURE,
    et un controle qui ne peut pas mesurer ne doit pas conclure.
    '''
    try:
        for track in master_obj.mediadata["media"]["track"]:
            if track.get("@type") == "General" and "Duration" in track:
                return Decimal(str(track["Duration"])) * Decimal("1000")
    except Exception:
        pass
    return None


def get_track_audio_length_ms(audio):
    '''L'etendue de CETTE piste-la, ou `None` quand elle n'est pas mesurable.

    C'est la quantite que la garde d'extraction aurait toujours du comparer.
    `get_candidate_audio_length_ms` repond a une AUTRE question -- "quelle est
    la plus longue piste audio de ce fichier" -- et sa reponse est correcte
    pour cette question-la; la confondre avec celle-ci est le defaut, et il
    etait invisible parce que la garde PASSAIT.

    Mesure, deux fichiers, `mediainfo --Output=JSON`, hors du pipeline:
        E04  flux 1 en/AAC 1428.646 s   vs  max du fichier 1435.008 s (flux 2, ja)
        E06  flux 1 en/AAC 1429.520 s   vs  max du fichier 1434.976 s (flux 2, en)
    E06 porte DEUX pistes `en`, flux 1 et 2. LA LANGUE NE LES SEPARE PAS, le
    numero de flux si -- c'est pourquoi la borne est une propriete de la PISTE
    et jamais d'une langue, et pourquoi la ligne de journal cite le flux.

    `None` et non un repli silencieux: une piste sans `Duration` est une piste
    NON MESUREE, pas une piste de longueur connue. L'appelant decide quoi en
    faire et le DIT.
    '''
    if "Duration" not in audio:
        return None
    return Decimal(str(audio["Duration"])) * Decimal("1000")


def measure_track_extent_ms(file_path, stream_order, timeout=300):
    '''L'etendue REELLE de cette piste lue dans les paquets, ET POURQUOI quand
    elle manque. Renvoie `(extent_ms | None, reason)`.

    POURQUOI ELLE EXISTE, et c'est une mesure et pas une precaution: `Duration`
    est un champ de CONTENEUR et il SOUS-ESTIME parfois son propre flux. Mesure
    sur les plans de production disponibles -- 59 couples (fichier, piste),
    `tools/blast_radius.py`: **16 pistes sur 59 declarent une `Duration`
    INFERIEURE a leur etendue reelle**, jusqu'a **120 ms**. Refuser une piste sur
    ce nombre-la seul reviendrait a refuser, un jour, une paire qui fonctionne.

    On ne paye JAMAIS ce sondage sur le chemin qui passe: il ne tourne que
    lorsque la borne declaree a deja prononce un refus. Cout mesure: 310 ms pour
    un episode de 24 minutes, contre 0 sur les 57 couples qui passent.

    POURQUOI UN COUPLE ET PAS UN SEUL `None` (revue du Lead, 2026-09-15). Un
    `None` nu voulait dire TROIS choses a la fois -- ffprobe pas enregistre,
    ffprobe a tourne et n'a rien rendu, ffprobe a expire -- et depuis le journal
    UN INSTRUMENT EN PANNE ETAIT INDISCERNABLE D'UNE PISTE NON MESURABLE. C'est
    la cinquieme regle de `BRIEF_COMMON.md`: *"je n'ai pas pu mesurer" et "ce
    fichier est inverifiable" sont deux reponses differentes, et l'issue
    manquante est toujours L'INSTRUMENT N'A PAS TOURNE*.

    Le risque concret est petit et c'est exactement la pathologie de la
    campagne: si `ffprobe` se retrouve non enregistre dans un enfant forkserver
    (`AGENT.MD`, Python 3.14), CHAQUE escalade renvoie `None`, chaque refus sur
    borne declaree tient sans elargissement, les refus montent -- et le journal
    dit `unmeasured` partout sans rien qui designe l'instrument.

    `reason` vaut `measured` en cas de succes, sinon: `binary-absent`,
    `probe-exit-<code>`, `timeout`, `probe-failed(<type>)`, `no-packets`.
    L'etendue n'est JAMAIS inventee: absente, elle est `None`, et l'appelant
    laisse tenir le refus d'origine.
    '''
    # LE BINAIRE EST CHERCHE DANS LE `try`, ET CE N'EST PAS UN DETAIL DE STYLE.
    # `tools.software["ffprobe"]` leve `KeyError` quand l'outil n'est pas
    # enregistre. Cette exception sortait d'ici comme une exception quelconque et
    # faisait tomber la piste dans `failed` AU LIEU DE `declined`: une panne
    # d'INSTRUMENT deguisee en un autre verdict. Trouve en tirant la garde sans
    # ffprobe enregistre, pas en relisant le code.
    try:
        probe = tools.software["ffprobe"]
    except Exception:
        return None, "binary-absent"
    command = [probe, "-v", "error", "-select_streams", str(stream_order),
               "-show_entries", "packet=pts_time,duration_time",
               "-of", "csv=p=0", file_path]
    tools.dev_log(f"chimeric: measure_track_extent_ms starting "
                  f"file={file_path} stream_order={stream_order}\n")
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as error:
        return None, f"probe-failed({type(error).__name__})"
    if result.returncode != 0:
        return None, f"probe-exit-{result.returncode}"
    last = None
    for line in result.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            continue
        try:
            last = (Decimal(parts[0].strip()) + Decimal(parts[1].strip())) \
                * Decimal("1000")
        except Exception:
            continue
    # RIEN DE LISIBLE N'EST UN RESULTAT, ET IL A SON PROPRE NOM. Un flux qui
    # n'existe pas et un flux sans paquet exploitable tombent tous deux ici; ce
    # qui compte est que ce n'est PAS la meme chose qu'un instrument absent.
    if last == None:
        return None, "no-packets"
    return last, "measured"


def describe_candidate_track(audio, language):
    '''De quoi la garde parle, en toutes lettres: flux + langue + codec.

    Exigence de conception ratifiee par l'Architecte, et elle a une raison
    mesuree et pas une raison d'elegance: sur E06 une ligne disant "la piste
    en" ne distingue PAS la bonne reponse de la mauvaise -- il y a deux pistes
    `en` dans ce fichier. Une ligne portant le numero de flux, si.

    Une garde qui ne peut pas dire ce qu'elle a verifie ne peut pas etre
    auditee contre ce qu'elle AURAIT DU verifier.
    '''
    codec = (audio.get("ffprobe", {}).get("codec_name")
             or audio.get("Format") or "unknown")
    return (f"stream_order={audio.get('StreamOrder')} "
            f"language={language or 'unknown'} codec={codec}")


def parse_positive_rate(value):
    '''The ONE normaliser, used everywhere a rate enters this module: parse
    to an exact positive rational, or `None`. `None` in, `None` out; a blank
    or whitespace-only string, a non-positive value, and anything
    `Fraction()` cannot parse (including `"inf"`/`"nan"`/a non-finite float,
    a malformed `"num/den"`, a wrong type) all collapse to the SAME `None` --
    "I could not measure" is a property of the value, not of its absence
    (Architect's ruling, frame-indexed contract rule 6, `1ea300f1`).

    Returning `Fraction | None` rather than a bool is the point, not a style
    choice: it makes an infinite or non-finite rate UNREPRESENTABLE rather
    than merely rejected by a comparison. `Fraction(float("inf"))` raises
    before any `> 0` check runs -- caught here, never reaches one (the Lead
    measured a float-based predicate accepting `"inf"` for exactly this
    reason, 2026-09-15). Every downstream call reduces to `grid is None`.
    '''
    if value is None:
        return None
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if rate > 0 else None


def resolve_master_grid(frame_rate_mode, frame_rate, frame_rate_original):
    '''The exact rational the master's frame numbers are actually expressed
    on, or `None` when none is measured -- ARCH_FRAME_ACCURATE.MD defect 4.

    Follows the codebase's own CFR convention (`mergeVideo.py:679`): tested
    against the exact string "CFR", never a looser truthy check, so an absent
    or empty mode is NOT CFR -- silently trusting it would be the same
    "invented default" shape as ARCH_FRAME_ACCURATE.MD defect 3's 25.0 fps.
    CFR trusts the nominal `FrameRate` (that is what CFR means, and nothing
    downstream of this module does arithmetic with it that would need the
    more precise `FrameRate_Original` preferred instead -- that preference
    belongs to `adjust_delay_to_frame`, `mergeVideo.py`, frozen, not this
    module). Any other mode resolves only through a genuinely parseable
    `FrameRate_Original` -- it is the rate the frames were actually authored
    on, per the field's own MediaInfo semantics.
    '''
    if frame_rate_mode == "CFR":
        return parse_positive_rate(frame_rate)
    return parse_positive_rate(frame_rate_original)


def compose_marker(base_marker, factor):
    '''`chimeric+resampled:<facteur>` DANS CET ORDRE (SPEC_ZONE_A.MD s4), par
    PISTE. `base_marker` porte la decision de l'ADDENDUM 5 (chimerique ou non,
    decidee par l'orchestrateur sur le plan); `factor` est le facteur REELLEMENT
    applique a cette piste-ci (`build_speed_filter_chain` le quantifie sur SA
    frequence, donc deux pistes a deux frequences recoivent deux facteurs
    effectifs mesurablement differents) -- ou None quand aucune vitesse n'est
    appliquee (ADDENDUM 6: pas de filtre a 1, donc pas de marqueur).
    ADDENDUM 5 clause (c): le marqueur `resampled` est INDEPENDANT du seuil des
    15 s, une piste resamplee le porte meme sans splice.'''
    parts = [base_marker] if base_marker else []
    if factor is not None:
        import merge_video_resample
        parts.append(f"resampled:{merge_video_resample.format_factor(factor)}")
    return "+".join(parts)


def assemble_on_master_timeline(candidate_obj, master_obj, track_plans, reference_pieces,
                                work_dir, out_path, marker_value, job_start_utc,
                                timeout=3600, verify=True, verify_tolerance_ms=100,
                                verify_search_ms=30000, max_silence_fraction=None,
                                speed_ratio=None, reference_stream=None,
                                comparison_language=None, chapters_path=None):
    '''Point d'entree du module: CONSTRUIT le fichier, il ne MESURE rien.

    STAGE 5 (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD ADDENDUM 10 d): "LA
    FONCTION QUI TRAITE LE PLAN NE FAIT QUE TRAITER LE PLAN". Les morceaux
    arrivent CONSTRUITS par `repair_orchestrator.apply_plan`, a partir des
    frontieres a la frame exacte que l'etage 4 a resolues et des decalages
    sous-frame mesures par piste. Il n'y a plus ici de tier de frames, de
    marche de bord ni de reconciliation sous-frame: ces mesures ont eu lieu une
    fois, a l'etage 4, et les refaire ici serait re-mesurer ce que le plan a
    decide (l'ancienne `normalize_segments` le faisait, et l'arrondi de ses
    decalages a la frame est exactement le saut d'une frame que l'ADDENDUM 9
    point 2 interdit).

    `track_plans`: {StreamOrder (int): {"pieces", "extent_ms", "extent_source",
    "offset_measured", "borrow_reason", "offset_sources"}} -- CHAQUE piste
    porte ses PROPRES morceaux, parce que chaque piste porte son propre decalage
    mesure (ADDENDUM 9 point 14). Les frontieres sur la timeline du MAITRE sont
    les memes pour toutes; seul l'endroit ou l'on lit le candidat change.

    `reference_pieces`: les morceaux de la piste de comparaison -- ceux qui
    re-calent les SOUS-TITRES (le plan de la langue sur laquelle la mesure a ete
    prise) et que le verificateur sonde.

    `chapters_path`: le XML de chapitres deja re-cale (`build_delivered_
    chapters`), pose au mux; None = aucun chapitre a livrer.

    `job_start_utc`: EXIGE, SANS DEFAUT (Architect's ruling, VMSAM_ERA,
    2026-09-16), transmis tel quel jusqu'a `mux_repaired_file`.

    Renvoie un compte-rendu: ce qui a ete construit, ce qui a ete REFUSE et
    pourquoi. Une piste refusee est comptee separement d'une piste en echec --
    docs/SUBTITLE_CODECS.MD: "count a declined codec separately from a failed
    extract".
    '''
    tools.dev_log(f"chimeric: assemble_on_master_timeline starting "
                  f"candidate={candidate_obj.filePath} "
                  f"master={master_obj.filePath} work_dir={work_dir} "
                  f"out_path={out_path}\n")
    # LA CADENCE DU MAITRE, SONDEE A L'ADMISSION. Architect's ruling,
    # 2026-09-15: "a precondition is probed at admission, before any work
    # begins; a refusal must cost a probe, not a mux." Les deux champs, et leur
    # desaccord, sont publies dans le compte-rendu (vmsam-dev-3: 23.839 contre
    # 23.976 sur les deux seuls fichiers a cadence inhabituelle).
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
    # ARCH_FRAME_ACCURATE.MD defect 4, refused AT ADMISSION: the pieces carry
    # boundaries that are master FRAMES; if nothing anchors the master's grid,
    # those boundaries are not a coordinate.
    if resolve_master_grid(frame_rate_mode, frame_rate, frame_rate_original) is None:
        raise chimeric_error(
            f"the master's frame grid is not measurable -- FrameRate_Mode="
            f"{frame_rate_mode!r} FrameRate={frame_rate!r} "
            f"FrameRate_Original={frame_rate_original!r}: this candidate's "
            f"segment boundaries are not expressed on a measured grid")

    master_duration_ms = get_master_timeline_length_ms(master_obj)
    # LE PLAN COUVRE LA TIMELINE DU MAITRE, DE 0 A SA FIN, SANS TROU NI
    # CHEVAUCHEMENT -- verifie ici parce que c'est la propriete dont tout le
    # reste depend (`split_master_fill_shortfall`, le recadrage des repliques
    # sur la fin de timeline, le controle de duree). Un plan qui ne la tient pas
    # est un defaut de l'appelant, refuse par son nom, jamais complete ici.
    for label, pieces_to_check in [("reference", reference_pieces)] + [
            (f"stream {order}", plan["pieces"]) for order, plan in track_plans.items()]:
        cursor = Decimal("0")
        for piece in pieces_to_check:
            if Decimal(str(piece["master_start_ms"])) != cursor:
                raise chimeric_error(
                    f"the {label} plan is not contiguous on the master timeline: a "
                    f"piece starts at {piece['master_start_ms']} ms where the previous "
                    f"one ended at {cursor} ms")
            cursor = Decimal(str(piece["master_end_ms"]))
            if cursor <= Decimal(str(piece["master_start_ms"])):
                raise chimeric_error(
                    f"the {label} plan carries an empty or inverted piece "
                    f"[{piece['master_start_ms']},{piece['master_end_ms']})")
        if cursor != master_duration_ms:
            raise chimeric_error(
                f"the {label} plan ends at {cursor} ms, not at the master's "
                f"timeline end {master_duration_ms} ms")

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
            stream_order = int(audio["StreamOrder"])
            track_plan = track_plans.get(stream_order)
            track_label = describe_candidate_track(audio, language)
            if track_plan is None:
                raise chimeric_error(
                    f"the plan carries no pieces for {track_label}: no offset was "
                    f"established for this stream, and none is borrowed silently")
            track_bound_ms = track_plan["extent_ms"]
            # UNE GARDE QUI N'A JAMAIS TIRE EST INDISCERNABLE D'UNE GARDE QUI
            # MARCHE: la borne de CETTE piste, et sa provenance, a chaque piste.
            tools.logs.append(
                f"chimeric: extraction bound {track_label} "
                f"bound_ms={track_bound_ms} source={track_plan['extent_source']}\n")
            report = build_one_audio_track(
                candidate_obj, master_obj, audio, language, track_plan["pieces"],
                track_path, timeout, speed_ratio, reference_stream, comparison_language,
                track_bound_ms)
            report["extraction_bound_ms"] = str(track_bound_ms)
            report["extraction_bound_source"] = track_plan["extent_source"]
            report["extraction_bound_track"] = track_label
            # SON PROPRE DECALAGE, OU UN HERITAGE NOMME (ADDENDUM 9 point 14):
            # jamais l'offset d'une autre langue en silence.
            report["offset_measured"] = track_plan["offset_measured"]
            report["borrow_reason"] = track_plan.get("borrow_reason")
            report["offset_sources"] = track_plan.get("offset_sources")
            report["offset_fidelity"] = None
            report["marker"] = compose_marker(
                marker_value,
                Decimal(report["speed_ratio_applied"])
                if report.get("speed_ratio_applied") is not None else None)
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
                report = build_one_subtitle_track(
                    candidate_obj, subtitle, language, reference_pieces, work_dir, index,
                    timeout, speed_ratio)
                # LES REPLIQUES SUBISSENT LE RATIO DEMANDE, EXACT (pas un facteur
                # quantifie par une frequence d'echantillonnage): c'est lui que
                # porte leur marqueur.
                report["marker"] = compose_marker(
                    marker_value, Decimal(str(speed_ratio)) if speed_ratio is not None else None)
                subtitle_reports.append(report)
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
        # LE JETON EST STABLE; LA PROSE N'ENTRE PAS DANS UN CHAMP `key=`.
        # Ce site emettait UNE PHRASE ANGLAISE COMPLETE dans le champ `reason=`,
        # que les consommateurs tokenisent. NOMMER LA FORME, NE PAS LA CITER:
        # ecrire l'ancienne valeur ici la ferait retrouver par tout recensement
        # cherchant le defaut, et le compte-rendu d'un defaut ne doit pas
        # correspondre au detecteur du defaut. Mesure de `vmsam-dev-4` sous
        # /config/output: le champ se lisait comme un jeton d'un seul mot,
        # 6 occurrences, un fantome qui pollue tout recensement de causes. Le site :2605 du meme module fait deja la chose
        # correcte -- `reason={type(error).__name__}` -- donc le module portait
        # la regle a un endroit et la brisait a un autre.
        # Le parseur de dev-4 lit POSITIONNELLEMENT et attend [A-Za-z0-9_]+:
        # un jeton sans espace est la seule forme qu'il peut extraire.
        # Les deux nombres au-dessus portent deja le detail; le jeton porte la CLASSE.
        tools.logs.append(
            f"repair: PREDICTED_REFUSAL track={report.get('stream_order')} "
            f"fill_short_by_ms={short} "
            f"tolerance_ms={output_duration_tolerance_ms} "
            f"reason=fill_source_too_short\n")

    # LE MARQUEUR DE FICHIER, pour les lecteurs qui n'en lisent qu'un
    # (`mark_audio_dicts` en repli): la decision de l'ADDENDUM 5 et le ratio
    # DEMANDE. Chaque piste porte en plus le sien, au facteur qu'ELLE a recu.
    file_marker = compose_marker(
        marker_value, Decimal(str(speed_ratio)) if speed_ratio is not None else None)
    mux_repaired_file(audio_reports, subtitle_reports, out_path, marker_value,
                      timeout, job_start_utc, chapters_path=chapters_path)

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
        # LA DUREE DE CONTENEUR DU MAITRE EST PASSEE A COTE DE SA DUREE VIDEO,
        # et les deux ne sont pas interchangeables: 1427944 contre 1427927 ms
        # sur le maitre d'Undead Unluck S01E12. La porte compare conteneur a
        # conteneur; `master_duration_ms` reste ce que l'assemblage VISE.
        output_check = verify_output_file(out_path, master_duration_ms, audio_reports,
                                          subtitle_reports,
                                          output_duration_tolerance_ms,
                                          get_master_container_length_ms(master_obj))
        log_prediction_outcome(predicted_refusals, output_check.get("would_refuse"))

        verification = None
        if verify:
            verification = verify_on_master_timeline(
                out_path, master_obj, audio_reports, reference_pieces, verify_tolerance_ms,
                verify_search_ms, reference_stream)

        # VERIFY-THE-FILL (Architect's ruling, verification half, 2026-09-16),
        # distinct from the VMSAM_ERA tag elsewhere in this file. UNCONDITIONNEL --
        # ne depend PAS de `verify`, qui gouverne une question DIFFERENTE
        # (l'alignement AV) et n'est deja plus jamais False en production.
        # ENTOUREE DE SON PROPRE try/except: `verify_fill_content` ne leve
        # rien PAR CONSTRUCTION, mais cette ligne ne doit jamais pouvoir
        # transformer un bug futur dans cette fonction en refus de fichier --
        # le Lead a trace la limite: cette capacite ENREGISTRE, elle ne
        # REFUSE JAMAIS, et cette garde tient cette limite meme si le corps
        # de la fonction se trompe un jour.
        fill_content = []
        try:
            fill_content = verify_fill_content(
                out_path, master_obj, audio_reports, master_duration_ms)
        except Exception as error:
            tools.logs.append(
                f"chimeric: fill_content_verification_failed "
                f"{type(error).__name__}: {error}\n")
        if fill_content:
            # `verify=False` NE DOIT PAS FAIRE DISPARAITRE CE CONTROLE --
            # c'est le drapeau EXISTANT et sans rapport que ce controle
            # refuse justement de partager. `verification` peut donc naitre
            # ICI, pour la piste qui en a besoin, meme quand l'alignement AV
            # n'a jamais tourne.
            if verification is None:
                verification = []
            by_track = {entry.get("track"): entry for entry in verification}
            for fc in fill_content:
                entry = by_track.get(fc["track"])
                if entry is None:
                    # PAS DE SUPPOSITION SUR LA CAUSE. `verify=False`, ou
                    # `verify_on_master_timeline` qui n'a produit aucune
                    # entree POUR CETTE PISTE (ex: aucun morceau candidat
                    # assez long a sonder sur AUCUNE piste, retour global
                    # `{"track": None, ...}`) sont deux causes distinctes
                    # du meme fait observable -- ne pas en affirmer une.
                    entry = {"track": fc["track"], "produced_index": fc["produced_index"],
                            "outcome": "skipped",
                            "reason": "no AV-alignment result recorded for this track"}
                    verification.append(entry)
                    by_track[fc["track"]] = entry
                entry["fill_content"] = fc["regions"]
                # PROMEUT `skipped` (le maitre n'a pas cette langue -- rien a
                # aligner) EN UN VRAI VERDICT DE CONTENU quand ce controle EN A
                # UN. NE TOUCHE JAMAIS un verdict de synchronisation deja
                # significatif (`aligned`/`misaligned`/`inconsistent`): ce
                # controle COMBLE une case vide, il n'en ECRASE aucune. C'est
                # exactement le mecanisme `verified_count` (merge_video_repair.py)
                # qui ferme `verified=1/5` -- par le compte qui existe deja,
                # pas par un nouveau.
                if entry.get("outcome") == "skipped":
                    outcomes = [r["outcome"] for r in fc["regions"]]
                    if any(o == "content_mismatch" for o in outcomes):
                        entry["outcome"] = "content_mismatch"
                    elif any(o == "content_indiscriminate" for o in outcomes):
                        entry["outcome"] = "content_indiscriminate"
                    elif any(o == "content_verified" for o in outcomes):
                        entry["outcome"] = "content_verified"
                    # Sinon: toutes les regions sont skipped_silent /
                    # skipped_unmeasurable -- `outcome` reste `skipped`,
                    # honnetement, et `fill_content` dit pourquoi.
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
        error.undelivered_path = mark_output(out_path, marking)
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
            "path": out_path, "pieces": reference_pieces, "audios": audio_reports,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": file_marker, "base_marker": marker_value,
            "verification": None}
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
    # `unread` et non zero quand mediainfo ne la donne pas. `frame_rate` /
    # `frame_rate_mode` / `frame_rate_original` are captured once, at
    # admission (top of this function, alongside the grid refusal) -- reused
    # here rather than re-read, so there is exactly one place this module
    # asks the master what its grid is.
    return {"path": out_path, "pieces": reference_pieces, "audios": audio_reports,
            "master_frame_rate": frame_rate,
            "master_frame_rate_mode": frame_rate_mode,
            "master_frame_rate_original": frame_rate_original,
            "subtitles": subtitle_reports, "declined": declined,
            "failed": failed, "marker": file_marker, "base_marker": marker_value,
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
    # PRE-CALL LOG, NOT POST-CALL (owner's decision, 2026-09-22, on a real
    # 7-hour hang: this `subprocess.run` carries no `timeout=`, and every
    # log line this module emits otherwise fires on the way OUT of a call --
    # which a hang never reaches. A line naming `file_path` immediately
    # BEFORE the blocking call is the only kind observable while it is
    # stuck, so it is placed here rather than after. Not fixing the missing
    # timeout tonight -- this ffmpeg extraction and an ffprobe show_entries
    # call have no shared legitimate duration, and a bound picked without
    # measurement is a guess dressed as a fix (Lead's ruling, 2026-09-22).
    tools.dev_log(f"chimeric: read_mono_samples starting file={file_path} "
                  f"stream={stream_specifier} start_ms={start_ms} "
                  f"duration_ms={duration_ms}\n")
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


def read_track_samples(file_path, stream_order, rate, audio_filter=None, timeout=900):
    """UNE piste ENTIERE, decodee une fois, mono, a `rate` Hz, en float32 --
    pour la mesure sous-frame des decalages de l'application du plan
    (ADDENDUM 9 point 2), ou chaque zone de chaque piste est sondee plusieurs
    fois: relire le fichier par recherche de SORTIE a chaque fenetre (ce que
    `read_mono_samples` fait, a raison, pour quelques sondes) redecoderait
    depuis le debut a chaque fois.

    L'echantillon `i` est au temps `start_time + i / rate` de la piste -- la
    meme convention que `atrim` dans `build_audio_filtergraph` et que la
    recherche du verificateur, qui lisent les deux le temps du flux.
    `audio_filter` (la chaine de vitesse, sur une paire a taux) est applique
    AVANT la conversion: les temps sont alors ceux de la timeline corrigee,
    `start_time * ratio + i / rate`, exactement comme l'assemblage les lit.

    BORNE (`timeout`) et journalisee AVANT l'appel: un decodage entier qui
    bloquerait laisserait sinon un journal muet. Leve `chimeric_error` sur un
    echec d'outil -- une affirmation sur l'OUTIL, jamais sur le media."""
    import numpy
    command = [tools.software["ffmpeg"], "-v", "error", "-nostdin",
               "-i", file_path, "-map", f"0:{int(stream_order)}", "-vn", "-sn", "-dn"]
    if audio_filter:
        command.extend(["-af", audio_filter])
    command.extend(["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
                    "-ar", str(rate), "-"])
    tools.dev_log(f"chimeric: read_track_samples starting file={file_path} "
                  f"stream_order={stream_order} rate={rate} filter={audio_filter}\n")
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=timeout)
    except subprocess.TimeoutExpired:
        raise chimeric_error(
            f"the whole-track read of stream {stream_order} did not finish in "
            f"{timeout} s: a statement about the tool, not about the media")
    if process.returncode != 0:
        raise chimeric_error(
            f"the whole-track read of stream {stream_order} FAILED: ffmpeg exited "
            f"{process.returncode}: "
            f"{(process.stderr or b'').decode('utf-8', 'replace').strip()[-300:]}")
    return numpy.frombuffer(process.stdout, dtype=numpy.float32)


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


def probe_output_streams(file_path):
    """Ce que le FICHIER dit de lui-meme: par flux, type, langue et duree.

    On lit le fichier PRODUIT, pas le compte-rendu de ce qu'on croit avoir
    construit. `SPEC_ZONE_A.MD` s4d: un compte de pistes reconstruites est un
    enonce sur le TRAVAIL FAIT, pas sur un fichier.
    """
    import json as _json
    # `codec_name`, `sample_rate` ET `r_frame_rate` SONT LUS DANS LE MEME
    # APPEL, pas dans un second. Ils servent la tolerance de duree de
    # conteneur (`container_grid_tolerance_ms`), qui a besoin de la periode
    # d'une image et de celle d'une trame audio SUR CE FICHIER-CI. Les
    # demander ici ne coute rien -- `ffprobe` lit deja ces champs pour
    # repondre aux autres -- et evite une seconde sonde dont le resultat
    # pourrait decrire un autre fichier.
    command = [tools.software["ffprobe"], "-v", "error",
               "-show_entries", "stream=index,codec_type,codec_name,"
                                "sample_rate,r_frame_rate:"
                                "stream_tags=language,DURATION:format=duration",
               "-of", "json", file_path]
    # PRE-CALL LOG -- same reasoning as `read_mono_samples` above: this
    # `subprocess.run` carries no `timeout=` either, and only a line emitted
    # BEFORE the call can be observed if it hangs.
    tools.dev_log(f"chimeric: probe_output_streams starting file={file_path}\n")
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
                        "codec_name": entry.get("codec_name"),
                        "sample_rate": entry.get("sample_rate"),
                        "frame_rate": entry.get("r_frame_rate"),
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
    # PRE-CALL LOG -- same reasoning as the other two sites in this module.
    tools.dev_log(f"chimeric: last_audio_packet_ms starting file={file_path}\n")
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


def mark_output(out_path, marking):
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

    L'ARTEFACT RESTE SUR PLACE, dans l'arborescence ephemere du conteneur --
    plus de deplacement vers un magasin durable (owner, 2026-09-21: "je ne
    veux pas que l'app serve de tests").
    """
    token, why = marking
    if not path.exists(out_path):
        return None
    base, extension = path.splitext(out_path)
    marked = f"{base}.{token}{extension}"
    try:
        replace_file(out_path, marked)
    except OSError as error:
        sys.stderr.write(f"repair: the undelivered artefact could NOT be renamed "
                         f"and is still at its produced name: {error}\n")
        tools.logs.append("repair: an undelivered artefact kept its produced name\n")
        return None
    sys.stderr.write(f"repair: the artefact was renamed to *.{token}{extension} "
                     f"-- {why} -- so it is inspectable and NOT counted as "
                     f"produced\n")
    return marked


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


def apply_tail_exemption(delta_ms, exempted_ms):
    '''Deduct the tail-gap-attributable portion from a PRODUCED stream's
    measured `delta_ms` (duration minus `master_duration_ms`; negative means
    the stream is short), capped so a PLAN-STAGE prediction can never
    manufacture headroom the PRODUCED FILE does not actually have.

    SPEC_ZONE_A.MD s4h, Lead's ruling shape (a) and its draft condition,
    2026-09-21: the refusal must name the NON-EXEMPT amount, never the
    total, "otherwise the number in the error stops meaning what it says."
    `min(exempted_ms, abs(delta_ms))` is that cap in one call: a prediction
    bounded by a measurement is defensible, a prediction replacing a
    measurement is not, and the difference between the two is this `min`.

    Only a SHORTFALL (`delta_ms < 0`) can be tail-gap-caused: the assembler
    never fills past `master_duration_ms`, so a stream running LONG is never
    this exemption's business and is returned unchanged.

    `exempted_ms` is `None` for any stream whose own report carries no
    `tail_gap` piece (see `split_master_fill_shortfall`) -- the exemption is
    per-stream by construction, not a default this function could widen.

    Returns `(residual_delta_ms, deduction_ms)`; `deduction_ms` is `None`
    when nothing was deducted (positive delta, or no exemption for this
    stream) -- distinct from `Decimal(0)`, which would mean "deducted zero".
    '''
    if exempted_ms is None or delta_ms >= 0:
        return delta_ms, None
    deduction = min(exempted_ms, abs(delta_ms))
    return delta_ms + deduction, deduction


def container_grid_tolerance_ms(streams):
    '''De combien la duree de conteneur du fichier produit peut depasser
    celle du maitre SANS QUE CE SOIT DU CONTENU EN TROP. Renvoie
    `(tolerance_ms, detail)`; `tolerance_ms` est `None` quand rien n'est
    mesurable.

    UN MUXEUR N'ECRIT PAS UNE DEMI-TRAME. La derniere unite ecrite sur une
    piste deborde donc de la fin du contenu d'au plus sa propre duree, et
    c'est tout ce que le remultiplexage d'un contenu identique peut ajouter.
    La tolerance est le MAXIMUM des deux periodes, pas leur somme: un seul
    bloc porte la fin du Segment, et additionner deux quantites dont une
    seule s'applique ferait une tolerance que rien ne mesure.

    LA CADENCE EST LUE SUR LE FICHIER PRODUIT, COMME UN RATIONNEL EXACT.
    `r_frame_rate` d'`ffprobe` rend `24000/1001`; le `FrameRate` de mediainfo
    rend la decimale `23.976`, qui est un ARRONDI D'AFFICHAGE de ce
    rationnel-la (41.708375 ms contre 41.708333 ms par image). La video du
    fichier produit est une COPIE DE FLUX de celle du maitre (`mux_repaired_file`
    passe `-c copy`), donc sa grille EST celle du maitre, lue sans arrondi et
    sans second parametre a tenir d'accord.

    LA TRAME AUDIO VIENT DU CODEC REELLEMENT LIVRE et de SON taux
    d'echantillonnage (`audio_codec_frame_samples`). Un codec dont la taille
    de trame n'est pas une propriete du codec ne contribue RIEN et le `detail`
    le dit -- la tolerance retombe alors sur la seule periode image, qui reste
    une vraie mesure.
    '''
    video_frame_ms, audio_frame_ms = None, None
    audio_from = None
    for stream in streams:
        if stream.get("codec_type") == "video" and video_frame_ms is None:
            grid = parse_positive_rate(stream.get("frame_rate"))
            if grid is not None:
                video_frame_ms = Decimal(1000 * grid.denominator) / Decimal(grid.numerator)
        elif stream.get("codec_type") == "audio":
            samples = audio_codec_frame_samples.get(
                (stream.get("codec_name") or "").lower())
            rate = parse_positive_rate(stream.get("sample_rate"))
            if samples is None or rate is None:
                continue
            # LE MAXIMUM SUR LES PISTES AUDIO: n'importe laquelle peut porter
            # le dernier bloc du fichier, donc la borne doit couvrir la plus
            # grossiere d'entre elles.
            frame_ms = Decimal(1000 * samples * rate.denominator) / Decimal(rate.numerator)
            if audio_frame_ms is None or frame_ms > audio_frame_ms:
                audio_frame_ms, audio_from = frame_ms, stream.get("codec_name")
    measured = [value for value in (video_frame_ms, audio_frame_ms) if value is not None]
    # LES TROIS CHAMPS SONT DES JETONS `cle=valeur` SANS ESPACE dans la
    # valeur: la ligne de journal qui les porte est lue POSITIONNELLEMENT par
    # le recensement de dev-4, et une prose dans un champ `cle=` est
    # exactement le defaut corrige a `:3364`.
    detail = (f"video_frame_ms={video_frame_ms} "
              f"audio_frame_ms={audio_frame_ms} "
              f"audio_codec={audio_from or 'none_with_codec_fixed_frame_size'}")
    return (max(measured) if measured else None), detail


def verify_output_file(out_path, master_duration_ms, audio_reports,
                       subtitle_reports, tolerance_ms, master_container_ms):
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

    LES SOUS-TITRES SONT VERIFIES PRESENTS ET PAS EN DUREE, ET CE N'EST PAS
    PARCE QU'ILS FINISSENT AVANT LE FICHIER. La phrase que cette docstring
    portait -- "la duree d'une piste de sous-titres est celle de sa DERNIERE
    REPLIQUE, qui finit legitimement avant le fichier" -- etait FAUSSE dans sa
    seconde moitie, et c'est elle qui a laisse passer le defaut d'Undead
    Unluck S01E12: la Segment Duration de Matroska est le MAXIMUM DES FINS DE
    BLOC SUR TOUTES LES PISTES, sous-titres COMPRIS. Une replique dont la fin
    decalee depassait la timeline a donc allonge le conteneur de 95 ms, sans
    qu'aucune piste audio ne bouge et sans qu'aucun COMPTE ne change -- les
    deux seules choses que cette fonction regardait. Une duree PAR PISTE de
    sous-titres reste hors de portee ici (le conteneur n'en publie pas une
    fiable), mais la consequence qui compte, elle, se mesure: LA DUREE DU
    CONTENEUR PRODUIT, comparee a celle du MAITRE.

    `master_container_ms` EST DONC UNE QUATRIEME QUANTITE, exigee et sans
    defaut. Ni `master_duration_ms` (la duree VIDEO, 17 ms plus courte sur le
    maitre du cas), ni la duree d'une piste, ni `container_duration_ms` du
    produit. Comparer le conteneur produit a la duree VIDEO du maitre ferait
    refuser tout maitre qui porte deja ce depassement -- le controle
    accuserait la reparation d'un fait de la reference. Sans defaut parce
    qu'un fil oublie doit lever un `TypeError` a l'appel, et non redevenir
    silencieusement la garde qui n'a rien vu.

    LA TOLERANCE EST UNE MESURE, PAS LE `tolerance_ms` DE 500 MS. Celle-ci
    borne une piste audio contre la duree visee et repond a une autre
    question; le proprietaire a fixe la barre du conteneur a UNE IMAGE. La
    valeur exacte vient de `container_grid_tolerance_ms` -- la plus longue des
    deux unites indivisibles que le muxeur a pu ecrire en dernier.
    """
    tools.dev_log(f"chimeric: verify_output_file starting out_path={out_path}\n")
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
    short, unmeasured, tail_exempted = [], [], []
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
        # TAIL-GAP EXEMPTION, AT THE SITE THAT ACTUALLY GATES THE RAISE
        # (SPEC_ZONE_A.MD s4h, Lead's ruling shape (a), 2026-09-21 --
        # corrected after the first draft named `fill_short_by_ms` alone,
        # which this function never reads to DECIDE, only to ANNOTATE: the
        # decision below, on `residual_delta`, is the one that matters).
        # PER-STREAM, STRUCTURALLY: `exempted_ms` comes only from THIS
        # position's own report, which is `None` unless that track's own
        # `pieces` carried a `tail_gap` piece (`split_master_fill_shortfall`)
        # -- a stream with no tail-gap fill gets no exemption at any
        # magnitude, by construction, not by a check added here.
        exempted_ms = None
        if position < len(audio_reports):
            _tail_exempt = audio_reports[position].get("fill_short_by_ms_tail_exempt")
            if _tail_exempt not in (None, "", "0"):
                exempted_ms = Decimal(str(_tail_exempt))
        residual_delta, deduction_ms = apply_tail_exemption(delta, exempted_ms)
        if tolerance_ms != None and abs(residual_delta) > Decimal(str(tolerance_ms)):
            entry = {"index": stream["index"], "language": stream["language"],
                     "duration_ms": str(duration), "delta_ms": str(delta)}
            if deduction_ms != None:
                # delta_ms ABOVE STAYS THE REAL MEASUREMENT, NEVER OVERWRITTEN
                # (Lead's condition, 2026-09-21): two fields, two meanings.
                # `residual_delta_ms` is what actually gated this refusal.
                entry["residual_delta_ms"] = str(residual_delta)
                entry["tail_exempt_ms"] = str(deduction_ms)
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
        elif deduction_ms != None:
            # VISIBLE, NEVER SILENT (Lead's condition, 2026-09-21): a
            # refusal that stops happening leaves no trace by construction
            # unless it is named here too. This confirms the exemption from
            # the PRODUCED file's own measurement, distinct from the
            # plan-stage prediction logged in `build_one_audio_track`.
            tail_exempted.append({
                "index": stream["index"], "language": stream["language"],
                "delta_ms": str(delta), "exempted_ms": str(deduction_ms),
                "residual_delta_ms": str(residual_delta)})
            tools.logs.append(
                f"chimeric: output_check_tail_exempt stream={stream['index']} "
                f"language={stream['language']} delta_ms={delta} "
                f"exempted_ms={deduction_ms} residual_delta_ms={residual_delta}\n")
    if len(short):
        problems.append("track(s) not running to the master's duration: "
                        + "; ".join(f"stream {s['index']} ({s['language']}) "
                                    f"{s.get('residual_delta_ms', s.get('delta_ms', s.get('reason')))}"
                                    + (f" [of measured {s['delta_ms']}, "
                                       f"{s['tail_exempt_ms']} ms tail-gap-exempted]"
                                       if s.get("tail_exempt_ms") else "")
                                    + (f" [fill source itself short by "
                                       f"{s['fill_short_by_ms']} ms]"
                                       if s.get("fill_short_by_ms") else "")
                                    for s in short))
    if len(unmeasured):
        problems.append("track(s) whose duration the file does not state: "
                        + "; ".join(f"stream {u['index']} ({u['language']})"
                                    for u in unmeasured))
    # LE FICHIER NE DOIT PAS ETRE PLUS LONG QUE LE MAITRE -- LA MESURE QUI
    # EXISTAIT ET QUE PERSONNE NE LISAIT. `container_duration_ms` et
    # `max_av_stream_duration_ms` etaient deja calcules et EMIS juste en
    # dessous, et aucun `problems.append` ne les consommait: la porte publiait
    # le chiffre qui la contredisait. Sur Undead Unluck S01E12 elle a ecrit
    # `container_ms=1428039 expected_ms=1427927` et rendu `would_refuse=False`.
    #
    # DANS UN SEUL SENS. Un conteneur plus COURT que celui du maitre est deja
    # la question des gardes par piste ci-dessus, qui la posent avec les bons
    # moyens (par piste, avec l'exemption de queue quand la source elle-meme
    # est courte). Refuser ici les deux sens ferait un second juge sur une
    # question deja jugee, avec moins d'information.
    #
    # NON MESURE N'EST PAS CONFORME: quand l'une des deux durees manque, la
    # ligne le dit et la porte ne conclut pas -- on ne substitue rien.
    container_overshoot_ms, container_tolerance_ms = None, None
    container_refused = False
    tolerance_detail = "video_frame_ms=None audio_frame_ms=None audio_codec=not_reached"
    if container_ms != None and master_container_ms != None:
        container_tolerance_ms, tolerance_detail = container_grid_tolerance_ms(streams)
        container_overshoot_ms = container_ms - Decimal(str(master_container_ms))
        if container_tolerance_ms != None and container_overshoot_ms > container_tolerance_ms:
            container_refused = True
            problems.append(
                f"the produced container runs {container_overshoot_ms} ms past "
                f"the master's own container ({container_ms} vs "
                f"{master_container_ms}), more than the {container_tolerance_ms} ms "
                f"a last indivisible block can explain ({tolerance_detail})")
    tools.logs.append(
        f"chimeric: output_container_check container_ms={container_ms} "
        f"master_container_ms={master_container_ms} "
        f"overshoot_ms={container_overshoot_ms} "
        f"tolerance_ms={container_tolerance_ms} {tolerance_detail}\n")
    report = {"unmeasured": unmeasured,
              # STREAMS THE TAIL-GAP EXEMPTION KEPT OUT OF `short` -- present
              # here so a refusal that stopped happening is still on the
              # record (SPEC_ZONE_A.MD s4h; Lead's ruling, 2026-09-21).
              "tail_exempted": tail_exempted,
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
              # CE QUE LA GARDE DE CONTENEUR A REELLEMENT COMPARE, et contre
              # quelle borne. Un verdict qu'on ne peut pas recalculer depuis
              # l'artefact est un verdict a croire sur parole.
              "master_container_duration_ms": (str(master_container_ms)
                                               if master_container_ms != None else None),
              "container_overshoot_ms": (str(container_overshoot_ms)
                                         if container_overshoot_ms != None else None),
              "container_tolerance_ms": (str(container_tolerance_ms)
                                         if container_tolerance_ms != None else None),
              "container_tolerance_detail": tolerance_detail,
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
        # LE JETON EST POSE ICI, A LA DECISION, ET NON CHEZ L'APPELANT.
        # C'est le site DOMINANT en production: 18 des 26 declins mesures sur
        # les 59 artefacts de `/config/output` (dev-cause, 2026-09-15). Le
        # jeton nomme CE controle-la -- le fichier produit relu et compare a ce
        # qui a ete construit -- et non "l'assemblage a refuse", qui couvrirait
        # 24 decisions distinctes et n'en classerait aucune.
        #
        # LA PARTIE QUI VARIE RESTE DANS LA PROSE. `problems` change d'un
        # fichier a l'autre; le jeton non. Un jeton qui varie n'est pas un
        # jeton -- il ne s'agrege pas, donc il ne compte rien.
        #
        # MASTER-NAMED VERDICT, REVISED (Architect, 2026-09-21, second pass).
        # First pass named one master-side token for every track short of
        # `expected_duration_ms` because its own fill source did not reach
        # the master's video. Forensic then packet-verified all 40 flagged
        # masters at L3 and split them 100/0 by a property this function can
        # already see: whether the master's OWN per-track measurements agree
        # with each other. That split is two classes IN KIND, not one class
        # in detail, so it earns two tokens -- and the tokens name the SHAPE
        # of the disagreement only, never a mechanism word (`truncation`,
        # `lie`): the shape is what is measured here; which mechanism
        # produces it is the L3 packet decode's finding, not this function's,
        # and a token naming the mechanism would need retracting the day a
        # counterexample shape turns up with the other mechanism behind it.
        #
        # THE DISCRIMINATOR, from data already in `audio_reports` -- no new
        # probe, no filename, no language tag, no track count: which reports
        # actually got a master-duration MEASUREMENT (`fill_source_ms` set,
        # i.e. `fill == "master"` and the master's own `Duration` was
        # readable), and which of those came up short beyond `tolerance_ms`
        # (`fill_short_by_ms`, set at the report-building site, read here,
        # never recomputed). A report with no such measurement at all says
        # nothing about the master and is excluded from both classes.
        #
        #   every measured report short, ALL BY THE SAME VALUE
        #       -> the whole complement moves together: master_audio_complement_short
        #   at least one measured report short and at least one NOT,
        #   or the short ones disagree on the amount
        #       -> the master's own tracks contradict each other about
        #          reaching its own declared video: master_duration_sources_disagree
        #   no measured report short at all
        #       -> nothing here traces to the master; untouched path below
        #
        # Same guard as before on top of either branch: nothing else about
        # the file may be wrong (no track-count mismatch, no unmeasured
        # stream) or the refusal is not wholly accounted for by the master
        # and the candidate-side token applies, byte-identical.
        def _short_beyond_tolerance(fill_report):
            value = fill_report.get("fill_short_by_ms")
            if value in (None, "", "0"):
                return False
            if tolerance_ms == None:
                return True
            return abs(Decimal(str(value))) > Decimal(str(tolerance_ms))

        measured_fill_reports = [r for r in audio_reports
                                 if r.get("fill_source_ms") not in (None, "")]
        short_fill_reports = [r for r in measured_fill_reports
                              if _short_beyond_tolerance(r)]
        agreeing_fill_reports = [r for r in measured_fill_reports
                                 if not _short_beyond_tolerance(r)]
        nothing_else_wrong = (
            not len(unmeasured) and len(audio) == len(audio_reports)
            and len(subtitle) == len(subtitle_reports)
            # UN DEPASSEMENT DE CONTENEUR EST "AUTRE CHOSE QUI CLOCHE", et la
            # regle enoncee juste au-dessus s'y applique telle quelle: un
            # jeton cote MAITRE affirme que le refus est ENTIEREMENT explique
            # par le maitre. Un fichier qui deborde sa reference porte un
            # defaut que ni `master_audio_complement_short` ni
            # `master_duration_sources_disagree` ne decrit, et le nommer
            # ainsi ferait compter ce cas-ci dans une population mesuree pour
            # autre chose. Il retombe donc sur `output_check_mismatch`.
            and not container_refused)

        cause = "output_check_mismatch"
        if nothing_else_wrong and short_fill_reports:
            # EXACT TIE, NO TOLERANCE BAND -- DELIBERATE, NOT AN OVERSIGHT.
            # The measured shape-B corpus example ties three languages at
            # 1435.491 s, byte-identical -- exact equality is what the
            # evidence actually shows, and nothing in the ruling says a
            # jitter band was intended, so none is invented here. The token
            # claims the SHAPE, "short by the SAME value": a master whose
            # tracks are short by only approximately the same amount has not
            # shown that shape, and correctly falls to
            # `master_duration_sources_disagree` by definition, not by
            # accident of a missing tolerance. If forensic's tie-exactness
            # distribution across the 21 REAL_TRUNCATION masters comes back
            # with jitter, that is new data to revisit this line with, not a
            # sign this line was wrong when written.
            tied_values = {Decimal(str(r["fill_short_by_ms"]))
                           for r in short_fill_reports}
            # A COMPLEMENT OF ONE IS STILL A COMPLEMENT. With exactly one
            # measured report, "every measured report short, all by the same
            # value" is TRUE, not vacuous -- a single track short is, as a
            # whole, short by that one value, and the shape claim holds. The
            # awkwardness is in the ENGLISH ("complement" reads as implying
            # more than one member), not in the measurement, so no third
            # class is carved out for n=1: `sources_disagree` would be false
            # here (nothing disagrees with anything), and falling through to
            # the candidate-side token would misattribute a genuine
            # master-traced shortfall to the candidate. Deliberate, and
            # untested against real material -- neither measured shape
            # (forensic's 21 REAL_TRUNCATION, 19 CONTAINER_DURATION_LIE
            # masters) was single-track.
            if not agreeing_fill_reports and len(tied_values) == 1:
                cause = "master_audio_complement_short"
            else:
                cause = "master_duration_sources_disagree"
        error = chimeric_error("the produced file does not match what was built: "
                               + "; ".join(problems),
                               cause=cause)
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
    tools.dev_log(f"chimeric: verify_on_master_timeline starting "
                  f"out_path={out_path} master={master_obj.filePath}\n")
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
            # SECOND SITE AUTORISE (Lead, R2): 5 des 26 declins mesures.
            # Le jeton dit CE QUE LA MESURE A CONTREDIT -- le plan affirme un
            # alignement unique sur une piece, la piste dit le contraire --
            # parce que c'est la chose sur laquelle un correctif futur agit:
            # un point de changement que la mesure a MANQUE. Distinct de
            # `output_check_mismatch`, qui dit que le FICHIER PRODUIT est faux;
            # ici le fichier produit est fidele au plan et c'est LE PLAN qui
            # est incomplet. Deux corrections differentes, donc deux jetons.
            error = chimeric_error(
                f"the plan says one alignment holds across a piece and the track "
                f"says otherwise: {detail}. That is a change point the measurement "
                f"missed, not a splice error -- the track may be correctly aligned "
                f"on both sides of a boundary nobody modelled",
                cause="alignment_contradicts_plan")
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
        # FOURTH SITE AUTORISE (this case, errid 12, CASE_errid12_untokened_5367.md):
        # first production occurrence 2026-09-22/23 (errid 12, Xian Wang / GST E02,
        # wave table pass 8). Le jeton dit CE QUE LA MESURE A VU: la piste livree ne
        # tombe pas sur la timeline du maitre au-dela de `verify_tolerance_ms`
        # (100 ms) -- et cette tolerance-la est deja une MESURE, pas un reglage
        # libre (voir `merge_video_repair.py:58-64`: plan correct atterrit a
        # 0.5-2.8 ms, plan faux a 503+ ms, 100 ms est deux ordres au-dessus du
        # premier et un ordre en dessous du second). Distinct des jetons voisins:
        # `alignment_contradicts_plan` dit qu'un alignement declare ne tient pas
        # ENTRE LES SONDES D'UNE MEME PIECE, au moment de l'ASSEMBLAGE (avant le
        # mux); `output_check_mismatch` dit que le FICHIER PRODUIT deborde son
        # maitre en DUREE DE CONTENEUR. Ici c'est cette VERIFICATION
        # POST-CONSTRUCTION (apres le mux, comparaison directe contre le maitre)
        # qui trouve la piste hors fenetre -- une quatrieme facon de rater, donc
        # un quatrieme jeton (regle de granularite R1). La question plus
        # profonde reste OUVERTE et ce jeton ne la tranche pas: sur errid 12,
        # une seule sonde sur quatre depasse la tolerance (100.5 ms, a peine
        # 0.5 ms au-dessus), et c'est la sonde a la toute tete du fichier
        # (541.67 ms dans le maitre, au bord de la piece de tete) -- les trois
        # autres sondes de la meme piste sont a 43.875/13.875/15.875 ms, bien en
        # dessous. Locator-precision-artefact-a-la-frontiere-de-piece contre
        # vrai decalage de plan (signe inverse ou point de changement manque):
        # NON TRANCHE ici, faute de plan de segments retenu pour cet artefact --
        # meme lacune de preuve que celle qui a laisse errid 25 ouvert.
        error = chimeric_error(
            f"the rebuilt track is not on the master's timeline: {detail}. "
            f"Tolerance {tolerance_ms} ms. The plan is wrong, not the splice: a "
            f"uniform offset means the base offset carries the wrong sign, and a "
            f"residual that changes at a change point means a step was missed",
            cause="delivery_timeline_misalignment")
        error.verification = results
        error.audios = audio_reports
        raise error
    return results


def _log_mel_z(samples, rate, n_fft=2048, hop=441, n_mels=40):
    '''Log-mel, z-scored per band -- PORTED from `VMSAM_HELP_AI/tools/tool_logmel_ncc.py`
    (forensic's reference instrument, `REPORT_owner_flagged_borrowed_fill_
    correlation.md`) and its sibling `VMSAM_HELP_AI/tools/verify_fill_provenance.py`.
    Same math, credited at the port site per the mission's own instruction.
    '''
    import numpy
    import scipy.signal
    _, _, spectrum = scipy.signal.stft(samples, fs=rate, nperseg=n_fft,
                                       noverlap=n_fft - hop, boundary=None)
    magnitude = numpy.abs(spectrum)

    def hz_to_mel(hz):
        return 2595 * numpy.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_points = numpy.linspace(hz_to_mel(50), hz_to_mel(rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = numpy.floor((n_fft + 1) * hz_points / rate).astype(int)
    bank = numpy.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, centre):
            if centre > left:
                bank[i - 1, k] = (k - left) / (centre - left)
        for k in range(centre, right):
            if right > centre:
                bank[i - 1, k] = (right - k) / (right - centre)
    log_mel = numpy.log1p(bank @ magnitude)
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-8
    return (log_mel - mean) / std


def _ncc_at_zero_offset(a, b):
    '''NCC at the KNOWN offset (zero), never searched -- unlike
    `tool_logmel_ncc.py`'s sliding `ncc_search`: a filled region's offset is
    ASSERTED by construction (`normalize_segments` sets `source_start_ms =
    cursor` on both master-piece branches, so `[a,b]` of the master always
    fills at master `[a,b]`, no shift) -- a search would answer a question
    this site does not have. Stated deviation from the ported tool, per the
    mission's instruction to record one.
    '''
    import numpy
    length = min(a.shape[1], b.shape[1])
    if length < 2:
        return None
    flat_a, flat_b = a[:, :length].flatten(), b[:, :length].flatten()
    norm_a, norm_b = numpy.linalg.norm(flat_a), numpy.linalg.norm(flat_b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    return float(numpy.dot(flat_a, flat_b) / (norm_a * norm_b))


fill_content_ncc_floor = 0.85  # forensic's controls; a constant, never re-derived per run.
fill_content_control_min_offset_ms = Decimal("60000")  # forensic's own separation: "130 s away".


def _fill_control_window_ms(start_ms, span_ms, master_duration_ms):
    '''A DELIBERATELY MISMATCHED window of the SAME claimed master source --
    a known-WRONG pairing, which needs no ground truth to build, only
    distance (Architect's ruling on the control, 2026-09-16). Same logic as
    `verify_fill_provenance.py`'s `_find_master_window`, ported for the same
    reason as the NCC math above.
    '''
    offset = max(fill_content_control_min_offset_ms, span_ms * 4)
    forward = start_ms + offset
    if forward + span_ms <= master_duration_ms:
        return forward
    backward = start_ms - offset
    if backward >= 0:
        return backward
    if start_ms > (master_duration_ms - (start_ms + span_ms)):
        return Decimal("0")
    return max(Decimal("0"), master_duration_ms - span_ms)


def verify_fill_content(out_path, master_obj, audio_reports, master_duration_ms):
    '''Does a shipped master-fill span carry the content it claims to?

    RECORDS, NEVER REFUSES (Lead's ruling, 2026-09-16, scope boundary on this
    function specifically): this function raises NOTHING. What ships on a
    cross-language fill is `SPEC_ZONE_A.MD` s4c's mandatory territory, not
    this function's; a `content_mismatch` is reported on the plan, exactly
    like `content_verified` or `content_indiscriminate`, and changes nothing
    about the file already on disk. Every extraction/probe failure degrades
    to `"skipped_unmeasurable"`, never an exception -- "I could not measure"
    is a different answer from "it does not match", and this function must
    be able to say the first without ever risking the second by accident.

    UNCONDITIONAL: no parameter here can turn this check off. Called for
    every produced file, independent of the (pre-existing, unrelated)
    `verify` flag that gates the AV-alignment probe in
    `verify_on_master_timeline` -- that flag exists for a different question
    and piggybacking this one on it would be exactly the kind of
    sub-option-on-an-unconditional-capability `WRITE_ZONES.MD` s4 names as
    the same defect one level down.

    KNOWN OFFSET, NEGATIVE CONTROL, THREE-STATE decision -- same design as
    the sweep-side `tools/verify_fill_provenance.py`, fired on synthetic
    material with all three outcomes plus the "nothing to check" case before
    this production site was written. `fill_content_ncc_floor` (0.85) is
    forensic's controls, applied as a CONSTANT to both readings: a positive
    (identity) control is impossible in production (needs a known answer a
    live job does not have), but a negative control needs only a
    deliberately-wrong pairing, which is always constructible.
    '''
    tools.dev_log(f"chimeric: verify_fill_content starting out_path={out_path} "
                  f"master={master_obj.filePath}\n")
    results = []
    for produced_index, report in enumerate(audio_reports):
        if report.get("gap_fill") != "master":
            continue
        fill_stream_order = report.get("fill_stream_order")
        if fill_stream_order is None:
            continue
        regions = []
        for region in report.get("filled_regions") or []:
            if region.get("source") != "master":
                continue
            try:
                start_ms = Decimal(str(region["master_start_ms"]))
                end_ms = Decimal(str(region["master_end_ms"]))
            except Exception:
                regions.append({"master_start_ms": region.get("master_start_ms"),
                               "master_end_ms": region.get("master_end_ms"),
                               "outcome": "skipped_unmeasurable",
                               "reason": "region bounds unreadable"})
                continue
            span_ms = end_ms - start_ms
            if span_ms <= 0:
                regions.append({"master_start_ms": str(start_ms), "master_end_ms": str(end_ms),
                               "outcome": "skipped_unmeasurable", "reason": "degenerate span"})
                continue
            control_start_ms = _fill_control_window_ms(start_ms, span_ms, master_duration_ms)
            entry = {"master_start_ms": str(start_ms), "master_end_ms": str(end_ms),
                    "fill_source_class": region.get("fill_source_class")}
            try:
                produced_samples = read_mono_samples(
                    out_path, f"0:a:{produced_index}", start_ms, span_ms, verify_probe_rate)
                reading_samples = read_mono_samples(
                    master_obj.filePath, f"0:{fill_stream_order}", start_ms, span_ms,
                    verify_probe_rate)
                control_samples = read_mono_samples(
                    master_obj.filePath, f"0:{fill_stream_order}", control_start_ms, span_ms,
                    verify_probe_rate)
            except Exception as error:
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": f"extraction failed: {type(error).__name__}"})
                regions.append(entry)
                continue
            if (get_rms(produced_samples) < verify_min_rms
                   or get_rms(reading_samples) < verify_min_rms):
                entry["outcome"] = "skipped_silent"
                regions.append(entry)
                continue
            try:
                reading_ncc = _ncc_at_zero_offset(
                    _log_mel_z(produced_samples, verify_probe_rate),
                    _log_mel_z(reading_samples, verify_probe_rate))
                control_ncc = _ncc_at_zero_offset(
                    _log_mel_z(produced_samples, verify_probe_rate),
                    _log_mel_z(control_samples, verify_probe_rate))
            except Exception as error:
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": f"NCC computation failed: {type(error).__name__}"})
                regions.append(entry)
                continue
            if reading_ncc is None or control_ncc is None:
                entry.update({"outcome": "skipped_unmeasurable",
                             "reason": "degenerate window for NCC"})
                regions.append(entry)
                continue
            if control_ncc >= fill_content_ncc_floor:
                outcome = "content_indiscriminate"
            elif reading_ncc >= fill_content_ncc_floor:
                outcome = "content_verified"
            else:
                outcome = "content_mismatch"
            entry.update({"outcome": outcome, "reading_ncc": round(reading_ncc, 4),
                         "control_ncc": round(control_ncc, 4),
                         "separation": round(reading_ncc - control_ncc, 4)})
            regions.append(entry)
        if regions:
            results.append({"track": report["stream_order"], "produced_index": produced_index,
                           "regions": regions})
    return results
