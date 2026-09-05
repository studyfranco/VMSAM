'''
Reparation d'un fichier REFUSE, au moment du refus.

Appele depuis la zone A (`mergeVideo.remove_not_compatible_video`,
`SPEC_ZONE_A.MD` s1). Ce module est l'orchestrateur: il decide s'il y a quelque
chose a tenter, va chercher la mesure, fait construire le fichier par le module
d'assemblage, puis raccroche l'objet reparé au merge.

TROIS EXIGENCES VERIFIEES CONTRE LE CODE (SPEC_ZONE_A.MD s1), toutes tenues ici:

1. `delay_same_md5_audio` doit etre un `Decimal`. Il vaut None par defaut
   (`video.py:55`) et `mergeVideo.py:1446` calcule
   `delay_same_md5_audio + delay_to_put`; `None + Decimal` leve. On pose
   `Decimal('0')` parce que la reparation a DEJA cale le fichier sur la
   timeline du maitre: il ne reste aucun retard a appliquer.
   (SPEC_ZONE_A.MD s1 cite la ligne 1426; c'est 1446 dans l'arbre. Verifie.)
2. L'objet doit avoir subi `get_mediadata()`: `generate_new_file` lit `audios`
   et `video['Duration']`.
3. Le fichier temporaire doit survivre jusqu'a `mkvmerge`. Il est ecrit sous
   `tools.tmpFolder/repair/`, cree avant le merge et supprime seulement apres
   (main.py:73, main_gestionar_show.py:191, fusion.py:323). Rien ne le balaie
   entre le refus et la fusion: `remove_tmp_files` ne supprime que les fichiers
   inscrits dans `self.tmpFiles`, et l'objet repare n'en a aucun.

QUATRE ISSUES. BRIEF_COMMON regle 5: "je n'ai pas pu mesurer" et "ce fichier est
irreparable" sont des reponses differentes, et l'issue qui manque toujours est
*l'instrument n'a pas tourne*. D'ou:

    no_plan       la mesure n'a rien pu etablir -- le refus reste intact
    declined      une mesure existait, la reparation l'a REFUSEE, avec sa raison
    repaired      un fichier a ete produit et raccroche
    failed        la reparation a tourne et a casse

IL N'Y A PLUS DE DRAPEAU, et c'est une decision du proprietaire inscrite dans
`WRITE_ZONES.MD` §4: *"une reparation conditionnee a un parametre n'est pas une
reparation"*. VMSAM tourne sans personne pour regarder -- c'est la premisse sur
laquelle tout le reste repose -- donc une capacite qui attend qu'un humain
l'active ne tournera jamais. Il y avait une cinquieme issue, `disabled`; elle
existait parce que le drapeau existait.

LA PORTE EST LA MESURE. `change_point_locator` rend `None` quand il ne peut rien
etablir, et on decline. L'erreur 237 a ete refusee sur une fidelite mediane de
0.576 avec 18 changements de signe, pas sur un reglage: c'est une procedure de
decision, pas un renvoi vers quelqu'un qui n'est pas la.
'''

from decimal import Decimal
from os import environ, path
import hashlib
import json
import sys

import tools
import video

# Tolerance d'alignement du verificateur, en millisecondes. CONSTANTE et non
# reglage: un seuil doit venir d'un ecart mesure. Mesure sur de vrais fichiers --
# un plan correct atterrit entre 0.5 et 2.8 ms (erreurs 266 et 108, six sondes
# chacune); un plan faux atterrit a 503 ms (point de changement manque) ou
# 16146 ms (signe inverse). 100 ms est deux ordres au-dessus du premier et un
# ordre en dessous du second.
verify_tolerance_ms = 100

last_repair_report = []


def parse_segments(raw_segments):
    '''JSON -> Decimal. Les nombres arrivent en chaines pour ne rien perdre.

    On COPIE la tranche et on convertit, au lieu de reconstruire un dict avec
    trois cles choisies. La version precedente enumerait les champs qu'elle
    connaissait et jetait tout le reste en silence -- dont
    `candidate_offset_ms_by_stream`, que la mesure emettait et que l'assemblage
    savait deja lire. Le consommateur et l'emetteur etaient tous les deux
    corrects; le transport entre les deux perdait la charge utile, et le test
    unitaire ne l'a pas vu parce qu'il passait un dict a la main sans traverser
    ce transport.

    Une liste blanche de champs dans un transport est un defaut par
    construction: elle rend muette toute extension du format, et le seul signe
    est que rien ne change.
    '''
    segments = []
    for raw in raw_segments:
        segment = dict(raw)
        segment["master_start_ms"] = Decimal(str(raw["master_start_ms"]))
        segment["master_end_ms"] = Decimal(str(raw["master_end_ms"]))
        segment["candidate_offset_ms"] = Decimal(str(raw["candidate_offset_ms"]))
        segments.append(segment)
    return segments


def get_speed_ratio(plan):
    """Le coefficient a appliquer, ou None. Le VERDICT decide, pas le nombre.

    `docs/AUDIO_SPEED_POLICY.MD` exige trois issues et un refus, et mesure que
    l'erreur destructrice -- reechantillonner un fichier qui n'avait besoin de
    rien -- etait DEUX FOIS plus frequente que le cas inversant que le detecteur
    existe pour trouver. On n'applique donc rien sans verdict explicite.
    """
    if plan.get("kind") != "speed" and plan.get("speed_ratio") == None:
        return None, None
    verdict = plan.get("verdict")
    if verdict == None:
        return None, ("the measurement carries a speed ratio but no verdict; "
                      "AUDIO_SPEED_POLICY.MD requires three outcomes and a decline, "
                      "and applying asetrate on a bare coefficient would let an "
                      "inverting case through undetected")
    if verdict == "leave_alone":
        return None, ("the measurement says LEAVE IT ALONE: the pair already "
                      "matches and a correction would take it apart")
    if verdict == "decline":
        return None, "the measurement declined to name a transformation"
    if verdict == "indeterminate":
        # SPEC_ZONE_A s4f: DEUX HYPOTHESES AU-DESSUS DE LA BARRE ET TROP PROCHES
        # POUR ETRE SEPAREES EST *INDETERMINE*, PAS PAL. Le proprietaire l'a
        # nomme parce que la faute inverse a deja ete commise: un balayage dont
        # les trois meilleures positions tenaient dans 0.028 a ete lu comme une
        # localisation et a produit deux cartes fausses en une nuit.
        #
        # Appliquer la meilleure des deux ici serait choisir par la marge la plus
        # mince disponible, c'est-a-dire par le bruit.
        return None, ("the measurement could not separate two rate hypotheses "
                      "above the bar: INDETERMINATE, not a rate. "
                      "SPEC_ZONE_A.MD s4f requires escalation to scene detection "
                      "-- a different modality -- and a tie-break computed from "
                      "the same correlations is not a third opinion")
    # LA CONVENTION DU RAPPORT, VERIFIEE CONTRE LES DUREES ET NON CONTRE UN NOM.
    #
    # vmsam-dev-1 a emis `speed_ratio` dans SA convention -- candidat/maitre --
    # la ou `TASKS/009` definit maitre/candidat. RECIPROQUES. Sur l'id 70 cela
    # aurait etire la piste de 0.9590 la ou il faut 1.0425: 8.7 % dans le MAUVAIS
    # SENS. `AUDIO_SPEED_POLICY.MD` faiblesse 3 enregistre exactement ce defaut,
    # et note que RIEN A L'INTERIEUR DU BALAYAGE NE POUVAIT L'ATTRAPER.
    #
    # Mes deux gardes -- bornes et verificateur -- l'attrapent quand le rapport
    # est loin de 1. Elles NE L'ATTRAPENT PAS pres de l'unite: 0.999001 contre
    # 1.000999 passe toute borne et toute tolerance, et c'est le cas DESTRUCTEUR.
    #
    # UN NOM DE CHAMP NE PORTE PAS SA CONVENTION. Les DUREES si. Quand le plan
    # les porte, on demande laquelle de `r` ou `1/r` est proche du rapport des
    # durees -- et on ne tranche que lorsque la reponse est nette.
    #
    # TROIS ETATS, parce que la verification n'est pas toujours possible: sur
    # l'id 33 les durees sont dans un rapport de 1.0687 pour une relation de
    # cadence de 1.001 -- LE CANDIDAT EST PLUS LONG PARCE QU'IL PORTE DU CONTENU
    # DIFFERENT. Un rapport de durees N'EST PAS UNE CADENCE, et un controle qui
    # l'oublierait refuserait l'id 33 a tort.
    convention = check_ratio_convention(plan)
    if convention != None:
        return None, convention
    labelled = check_ratio_labelled(plan)
    if labelled != None:
        return None, labelled
    if verdict == "rubberband":
        return None, ("the measurement says rubberband -- the inverting case, a "
                      "source already pitch-corrected at origin. Not implemented: "
                      "applying asetrate here would drag the pitch 72.4 cents flat")
    if verdict != "asetrate":
        return None, f"unknown speed verdict {verdict!r}"
    ratio = plan.get("speed_ratio")
    if ratio == None:
        return None, "verdict asetrate with no speed_ratio"
    return Decimal(str(ratio)), None


def get_speed_margin(plan):
    """DE COMBIEN LA MEILLEURE HYPOTHESE A GAGNE, pas seulement qu'elle a gagne.

    `SPEC_ZONE_A.MD` s4f: "prendre la meilleure transformation au-dessus de 90 %
    ET RAPPORTER DE COMBIEN ELLE L'EMPORTE". La marge est la quantite qui
    distingue un verdict d'un tirage, et une marge qui n'est pas EMISE est une
    marge que personne ne verifie -- meme regle que le denominateur dans la
    ligne et que la ligne rouge qui doit exister comme champ.

    Renvoie None quand la mesure n'en porte pas: absente, jamais zero. Une marge
    de zero serait DEUX HYPOTHESES A EGALITE, ce qui est le cas `indeterminate`
    et surtout pas "pas de marge rapportee".
    """
    margin = plan.get("speed_margin")
    return None if margin == None else str(margin)


def get_marker_value(plan):
    '''SPEC_ZONE_A.MD s4. `chimeric+resampled:<factor>` DANS CET ORDRE.

    Le facteur ecrit est celui reellement applique, a la precision reellement
    appliquee: un tag `resampled:1.042709` sur une piste etiree autrement est
    pire que pas de tag du tout.
    '''
    parts = []
    # `chimeric` veut dire ASSEMBLE DE PLUSIEURS SOURCES. Une relation de vitesse
    # seule, sur une tranche unique qui couvre tout, ne l'est pas: la marquer
    # ainsi mentirait sur ce qu'a subi la piste.
    segments = plan.get("segments") or []
    if len(segments) > 1 or (len(segments) == 1 and plan.get("kind") == "piecewise_constant"):
        parts.append("chimeric")
    applied = plan.get("applied_speed_factor")
    if applied != None:
        parts.append(f"resampled:{applied}")
    return "+".join(parts)


def get_master_timeline_ms(master_obj):
    return Decimal(str(master_obj.video["Duration"])) * Decimal("1000")


def get_marker_value_for(plan, speed_ratio, candidate_obj, master_obj):
    """`SPEC_ZONE_A.MD` s4, avec le facteur REELLEMENT applique.

    Le facteur ecrit n'est pas celui demande: `asetrate` prend un entier, donc la
    transformation obtenue est `intermediaire / round(intermediaire / ratio)`. On
    le calcule ici avec la meme frequence que l'assemblage utilisera, sinon le
    tag decrirait une transformation que le fichier n'a pas subie.
    """
    applied_factor = None
    if speed_ratio != None:
        import merge_video_resample
        rate = None
        for language, audios in candidate_obj.audios.items():
            for audio in audios:
                rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
                if rate != None:
                    break
            if rate != None:
                break
        if rate == None:
            raise Exception("no sampling rate on the candidate: cannot state the applied factor")
        _, applied, _, _ = merge_video_resample.build_speed_filter_chain(rate, speed_ratio)
        applied_factor = merge_video_resample.format_factor(applied)
    return get_marker_value(dict(plan, applied_speed_factor=applied_factor))


def get_delay_language(best_video, candidate_obj):
    """La langue sur laquelle le merge a mesure le delai.

    `remove_not_compatible_video` ne la recoit pas -- elle est choisie dans
    `sync_merge_video` et jamais passee plus bas. On la LIT plutot que de la
    redeviner: `prepare_get_delay` pose `videoObj.delays[language] = 0` sur chaque
    objet (mergeVideo.py:768), donc la cle est deja sur `best_video` au moment du
    refus. Redupliquer ici la logique de choix de `sync_merge_video` reviendrait a
    tenir une copie d'une regle qu'on ne controle pas.

    Constat pour le proprietaire: la langue de mesure n'est pas transmise a la
    zone A, et la relire depuis `delays` marche mais tient a un effet de bord.
    """
    keys = [language for language in best_video.delays.keys() if language != "und"]
    preferred = tools.special_params.get("original_language", "")
    if preferred in keys:
        return preferred, "preferred language is among the measured ones"
    if len(keys) == 1:
        return keys[0], "only one language was measured"
    # Repli: la seule langue audio commune aux deux fichiers.
    common = set(best_video.audios.keys()) & set(candidate_obj.audios.keys())
    common.discard("und")
    if len(common) == 1:
        return common.pop(), "only one language is common to both files"
    if preferred in common:
        return preferred, "preferred language is common to both files"
    if not len(keys):
        return None, "no language to choose from"
    # QUEUE ARBITRAIRE, ET ELLE DECIDE DE QUELQUE CHOSE. `keys` vient d'un dict,
    # donc keys[0] est l'ORDRE D'INSERTION, pas un choix. Mesure: cette queue
    # tranche sur 6 paires sur 29, toujours entre {en, fr}, toujours vers 'en'.
    #
    # Et ce n'est pas gratuit: le locator ne mesure les decalages par flux QUE
    # pour la langue du plan, donc cette queue decide QUELLE LANGUE EST CALEE
    # CORRECTEMENT et laquelle emprunte -- 14 a 32 ms mesures, sous la tolerance
    # du verificateur, donc livrable en silence.
    #
    # On ne l'ameliore pas ici: il n'existe pas de regle meilleure a partir de ce
    # que cette fonction voit, et en inventer une donnerait a un tirage l'allure
    # d'une decision. On DIT que c'en est un. Devient sans consequence quand la
    # table par flux couvrira tous les flux (contrat de `vmsam-dev-1`).
    return keys[0], f"ARBITRARY: insertion order among {sorted(keys)}"


def get_plan_from_locator(best_video, candidate_obj, language):
    """La mesure de `vmsam-dev-1`, appelee ici et nulle part ailleurs.

    Import tardif et tolerant: le module peut ne pas etre deploye, et une
    capacite fermee par defaut n'a pas le droit de casser un merge parce qu'une
    dependance manque.

    `None` veut dire *je n'ai pas pu mesurer*, jamais *les fichiers sont
    compatibles*. On laisse alors le refus tel quel.
    """
    try:
        import change_point_locator
    except Exception as error:
        if tools.dev:
            tools.logs.append(f"repair: no change_point_locator module: {error}\n")
        # THE MODULE IS NOT DEPLOYED. This is MY OWN process state and I am
        # entitled to state it: no measurement was attempted, because there was
        # nothing to attempt it with. Distinct from "the locator ran and refused".
        # LE JETON EST STABLE; la classe d'exception va dans la PROSE. dev-4
        # classe sur le jeton, donc un jeton qui varie n'est pas un jeton.
        return None, "locator_module_absent"
    plan = change_point_locator.locate_change_points(best_video, candidate_obj,
                                                     language)
    if plan != None:
        return plan, None
    # THE LOCATOR RAN AND RETURNED None, AND SAID NOTHING ABOUT WHY.
    #
    # I MAY NOT NAME THE CAUSE. The producer half that would emit one is
    # dev-1's and is unlanded, held by ITS user -- so any cause I wrote here
    # would be INVENTED, NOT READ, which is the acceptance test's own R2 and
    # the defect the whole campaign is convened against.
    #
    # `cause_unavailable` IS A TRUE STATEMENT AND THE OLD STRING WAS NOT.
    # "no measurement available" claims a property of the WORLD -- that no
    # measurement exists. For the fidelity-floor path that is FALSE: the probes
    # RAN, they SUCCEEDED, and they returned a CONCLUSIVE NEGATIVE. A
    # conclusive negative filed as an absence of evidence is exactly the
    # substitution change_point_locator warns about in its own words:
    # "None means I could not measure -- never the files are compatible."
    #
    # What I can honestly say is a property of THIS CONSUMER: the producer told
    # me nothing. WHEN THE PRODUCER HALF LANDS, READ ITS CAUSE HERE AND PASS IT
    # THROUGH UNCHANGED -- one site, this one. Do not translate it, do not
    # normalise it, and do not add a cause of your own beside it.
    return None, "cause_unavailable"


def drop_unverified_segments(segments):
    """Une tranche dont le decalage n'a pas pu etre mesure proprement devient un
    TROU, et le trou est rempli depuis le maitre.

    `vmsam-dev-1` marque `offset_unverified` quand la tranche est plus courte que
    sa fenetre de sonde: aucune sonde propre n'y tient, toute fenetre qui la
    recouvre franchit la transition, et un correlateur a pic sur une fenetre a
    cheval rend un pic DEPLACE -- de signe arbitraire et non borne par la grille.
    Sur l'erreur 266 cela valait 168 ms, mais 168 n'est pas un plafond: il n'y a
    rien a comparer a ma tolerance de 100 ms, donc la verification ne peut pas
    rattraper le cas.

    Coller du contenu candidat a un decalage non borne et non verifie est
    exactement le cas de DEGAT. Le contenu du maitre dans un trou de la timeline
    du maitre est correct par definition, et le cout est la duree de la tranche
    elle-meme -- 29 s sur 1428 pour 266, environ 2 %. On paie ce cout et on le
    DECLARE dans la colonne de remplissage.

    On ne refuse PAS la paire: jeter trois points de changement confirmes pour en
    proteger un seul mauvais est le mauvais echange, et `vmsam-dev-1` a eu raison
    de ne pas le faire dans son module.
    """
    kept, dropped_ms, dropped = [], Decimal("0"), []
    for segment in segments:
        if segment.get("offset_unverified"):
            span = Decimal(str(segment["master_end_ms"])) - Decimal(str(segment["master_start_ms"]))
            dropped_ms += span
            # LES BORNES, PAS SEULEMENT LE TOTAL. Un segment jete ici devient une
            # region remplie DEPUIS LE MAITRE plus bas, et sur la ligne de
            # journal elle est indistinguable d'un trou ordinaire du plan. Le
            # lecteur ne peut donc pas separer "le plan n'avait pas de candidat
            # ici" de "le plan en avait un ET ON L'A JETE".
            dropped.append({"master_start_ms": str(segment["master_start_ms"]),
                            "master_end_ms": str(segment["master_end_ms"]),
                            "dropped_ms": str(span)})
            continue
        kept.append(segment)
    return kept, dropped_ms, dropped


def clamp_segments_to_master(segments, master_obj):
    """Coupe le plan a la duree VIDEO du maitre, et jette ce qui tombe apres.

    `generate_new_file` passe `-t best_video.video['Duration']`
    (mergeVideo.py:1781), donc tout ce qui depasse est tronque par le merge de
    toute facon. `vmsam-dev-1` s'arrete a la duree AUDIO la plus courte et m'a
    demande de serrer ici, ou la valeur est disponible, plutot que de deviner
    l'attribut de son cote. Sur les fichiers d'exemple l'ecart est de l'ordre de
    la seconde.
    """
    limit = Decimal(str(master_obj.video["Duration"])) * Decimal("1000")
    clamped = []
    for segment in segments:
        start = Decimal(str(segment["master_start_ms"]))
        end = Decimal(str(segment["master_end_ms"]))
        if start >= limit:
            continue
        if end > limit:
            segment = dict(segment)
            segment["master_end_ms"] = limit
        clamped.append(segment)
    return clamped


# Un decalage est mesure a un QUANTUM pres -- 124 a 142 ms selon l'appel chez
# `vmsam-dev-1`. Une tranche dont le debut cote candidat tombe juste avant zero
# est donc du bruit de mesure, pas un plan qui lit hors du fichier. On rogne
# jusqu'a un quantum (borne haute de la plage mesuree); au-dela on REFUSE,
# parce qu'un debut negatif d'une seconde n'est plus une precision, c'est un
# plan faux.
head_clamp_max_ms = Decimal("150")


def clamp_segments_to_candidate_head(segments, stream_order=None):
    """Rogne une tranche qui commence juste AVANT le debut du candidat.

    Symetrique de `clamp_segments_to_master`, qui coupe a l'autre bout. Trouve
    le 2026-09-03 sur le premier plan chimeric+resampled reel: la premiere
    tranche partait de master 1876 ms avec un decalage de -1876.36 ms, soit un
    debut candidat de -0.36 MS, et l'assemblage refusait le fichier entier pour
    trois dixiemes de milliseconde.

    On avance le DEBUT MAITRE du depassement plutot que de bricoler le
    decalage: on perd le fragment qui n'existe pas dans le candidat, et le
    maitre le remplit -- ce que l'assemblage fait deja pour tout trou. Aucune
    seconde n'est inventee.
    """
    import merge_video_chimeric
    clamped = []
    for segment in segments:
        # LE MINIMUM SUR TOUS LES FLUX, pas le repli. Le rognage vaut pour
        # TOUTES les pistes -- une seule borne de morceau -- donc il doit
        # proteger la piste la plus negative. `vmsam-dev-1` a demande que la
        # garantie vive ici: son emetteur ne promet PAS que le decalage de repli
        # soit le plus negatif, c'est simplement la langue sur laquelle l'appel
        # a ete fait. Dependre de cette propriete serait dependre de quelque
        # chose que personne n'a promis.
        offsets = [merge_video_chimeric.get_segment_offset(segment, stream_order)]
        by_stream = segment.get("candidate_offset_ms_by_stream") or {}
        offsets.extend(Decimal(str(v)) for v in by_stream.values())
        offset = min(offsets)
        start = Decimal(str(segment["master_start_ms"]))
        candidate_start = start + Decimal(str(offset))
        if candidate_start < 0 and -candidate_start <= head_clamp_max_ms:
            segment = dict(segment)
            segment["master_start_ms"] = start - candidate_start
        clamped.append(segment)
    return clamped


def assemble_or_log_the_decline(logged_candidate, plan, unverified_ms, *args, **kwargs):
    """Assemble, et si l'assemblage REFUSE, ECRIT QUAND MEME LE JOURNAL DE PISTES.

    UNE ENVELOPPE ET NON UNE GARDE. Elle ne rattrape rien: la levee repart
    telle quelle, avec ses attributs. Ce qu'elle ajoute est que les lignes
    `repair:` existent pour un fichier DECLINE.

    Avant le passage de `output_check_enforcing` a True, la porte de duree
    n'avait jamais fait lever cet appel sur un fichier reel; un declin venait
    d'ailleurs et plus tot. Maintenant qu'elle leve, un fichier decline
    n'emettait plus AUCUNE ligne -- ni `build`, ni `plan`, ni `ADDED`, ni `CUT`
    -- et le lecteur de `vmsam-dev-4` rejette par structure un bloc sans ligne
    `plan`. LE FICHIER DISPARAISSAIT DU RAPPORT AU LIEU D'Y APPARAITRE COMME
    REFUSE, ce qui est la forme exacte du defaut que ce journal existe pour
    empecher.

    On ecrit ce qui a ete FAIT, pas ce qui a ete obtenu: `partial_assembly`
    porte les pieces posees et les pistes construites au moment du refus.
    """
    # IMPORT LOCAL, comme partout ailleurs dans ce module: `merge_video_chimeric`
    # n'est pas lie au niveau du module ici. `t58_unbound_names` l'a dit avant la
    # premiere execution -- troisieme fois ce soir qu'il attrape un nom que je
    # venais d'ecrire.
    import merge_video_chimeric
    try:
        return merge_video_chimeric.assemble_on_master_timeline(*args, **kwargs)
    except Exception as error:
        # ON ATTRAPE `Exception` ET NON `chimeric_error`, ET C'EST LA QUESTION DE
        # `vmsam-dev-4` QUI L'A OUVERT. Son lecteur consomme des LIGNES DE
        # JOURNAL par prefixe et rien d'autre; il a demande si l'etat non livre
        # atteint une ligne. Il n'y arrivait pas -- et pire, un `failed`
        # n'emettait AUCUNE ligne, exactement le trou que ce bloc venait de
        # boucher pour les declins. La panne d'outil sortait par une porte que
        # la reparation ne venait pas de reparer.
        partial = getattr(error, "partial_assembly", None)
        if partial != None:
            partial["unverified_segment_ms"] = unverified_ms
            try:
                log_assembly(logged_candidate.filePath, partial, plan)
            except Exception as logging_error:
                tools.logs.append("repair: could not write the per-track log for "
                                  f"an UNDELIVERED file: {logging_error}\n")
        # LA LIGNE TERMINALE EST INCONDITIONNELLE, meme sans assemblage partiel:
        # sans elle, "pas de ligne DECLINED" se lirait comme "pas de declin", la
        # lecture par omission que s4e interdit ailleurs.
        #
        # DEUX PREFIXES ET PAS UN, parce que les deux issues ne disent pas la
        # meme chose et que le pilote les classe differemment: `chimeric_error`
        # -> `declined` (le module a regarde et a dit non), tout le reste ->
        # `failed` (une panne d'outil ou un defaut a nous). Un prefixe unique
        # ferait absorber chaque echec d'ffprobe dans le cout de la porte.
        if isinstance(error, merge_video_chimeric.chimeric_error):
            tools.logs.append(f"repair: DECLINED {error}\n")
        else:
            tools.logs.append(f"repair: FAILED {type(error).__name__}: {error}\n")
        # ET L'ETAT DE L'ARTEFACT ATTEINT UNE LIGNE, PAR CLE ET NON PAR PROSE.
        # dev-4 lit par nom; `state=` et `path=` se lisent, "the file was
        # renamed" ne se lit pas. Emise seulement quand un fichier a REELLEMENT
        # ete marque: son ABSENCE dit "aucun artefact n'existait a marquer",
        # ce qui est un troisieme fait et pas un defaut de journal.
        marked = getattr(error, "undelivered_path", None)
        if marked != None:
            tools.logs.append(
                f"repair: undelivered state={getattr(error, 'undelivered_state', 'unnamed')} "
                # DURABLE OU EPHEMERE. `path=` seul ne les distingue pas -- les
                # deux sont des chemins plausibles et un seul survit a la
                # prochaine recreation du conteneur. `false` veut dire soit
                # qu'aucun magasin n'est configure, soit que le deplacement a
                # echoue, et la seconde cause est ecrite en clair par
                # `move_to_durable_store` plutot que devinee ici.
                # `durable=` EST EMIS SUR CHAQUE REFUS, Y COMPRIS LES REUSSITES.
                # Exigence de ci et sa raison est la notre: si le champ
                # n'apparaissait qu'en cas d'echec, son ABSENCE voudrait dire
                # soit que l'ecriture a reussi, soit que la ligne precede ce
                # changement, soit que le journal a ete tronque -- et ce serait
                # encore compter l'absence d'un champ comme une valeur, apres
                # l'avoir fait a un champ, une ligne, un jeton de nom de fichier,
                # un ensemble d'artefacts preserves, un recu et un remote git.
                #
                # SUR `true` LE CHEMIN EST CELUI DE LA DESTINATION, ce qui permet
                # a ci de joindre cette ligne a sa ligne de registre sans
                # deviner. Sur `false` c'est le chemin EPHEMERE, parce que la
                # c'est le seul qui existe -- et `in_place=` le repete pour que
                # les deux branches se lisent sans savoir laquelle on regarde.
                f"durable={bool(getattr(error, 'undelivered_durable', False))} "
                f"path={marked} "
                f"{'' if getattr(error, 'undelivered_durable', False) else 'in_place=' + str(getattr(error, 'undelivered_in_place', 'unreported')) + ' '}"
                .rstrip() + "\n")
        raise


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root):
    '''Construit le fichier repare et l'objet video qui va avec.

    Renvoie (objet, compte-rendu de l'assemblage).
    '''
    import merge_video_chimeric

    # UNE SEULE DERIVATION, PARTAGEE. La meme cle sert de repertoire de travail
    # ici et de repertoire de cas dans le magasin durable de `merge_video_chimeric`;
    # deux copies qui doivent s'accorder sont une divergence en attente.
    key = merge_video_chimeric.stable_case_key(candidate_obj.filePath)
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    speed_ratio, _ = get_speed_ratio(plan)
    segments = plan.get("segments")
    if not segments:
        # Un plan de VITESSE SEULE n'a pas de tranche: la relation couvre tout le
        # fichier. On en fabrique une qui couvre la timeline du maitre, plutot que
        # d'exiger de la mesure une structure qu'elle n'a pas a inventer.
        # `.get("base_offset_ms", 0)` FABRIQUE UNE VALEUR A PARTIR D'UNE ABSENCE.
        #
        # Regle de `vmsam-ci`: un defaut zero sur une quantite qui a un plancher
        # est un defaut DETECTABLE -- tout ce qui passe sous le plancher est une
        # valeur manufacturee. Un decalage n'a PAS de plancher, donc rien dans la
        # donnee ne trahit le cas ici: `absent` et `zero` sortent identiques.
        #
        # ET LES DEUX SONT DES FAITS DIFFERENTS. "ce plan est une vitesse pure,
        # sans decalage" et "aucun decalage n'a ete mesure" produisent tous deux
        # `0`, et le second place le candidat au zero du maitre sur la foi d'une
        # cle manquante.
        #
        # ON NE CHANGE PAS LE COMPORTEMENT -- je ne peux pas justifier de refuser
        # un plan de vitesse pure qui n'a legitimement pas de decalage. ON REND
        # LE CHOIX VISIBLE: le segment fabrique porte de quel cas il vient, et la
        # ligne `repair: segment` le dit.
        stated_offset = plan.get("base_offset_ms")
        segments = [{"master_start_ms": Decimal("0"),
                     "master_end_ms": get_master_timeline_ms(master_obj),
                     "candidate_offset_ms": Decimal(str(
                         stated_offset if stated_offset != None else 0)),
                     # PAS D'ESPACE DANS LE JETON. `vmsam-ci` a teste le
                     # marqueur contre ses deux lecteurs AVANT qu'il ne se
                     # deploie: l'un capturait `0` et JETAIT le marqueur -- une
                     # valeur fabriquee lue comme une mesure, exactement ce que
                     # le marqueur existe pour empecher -- et l'autre ne
                     # correspondait PLUS DU TOUT, parce que le texte s'intercale
                     # avant ` by_stream=`, donc le segment disparaissait en
                     # silence.
                     #
                     # LE SECOND EST LE PIRE: le lecteur perd EXACTEMENT les
                     # lignes qu'on lui demande de surveiller, et rapporte un
                     # denominateur plus petit, plus propre et entierement faux.
                     # 21 segments devenus 18 se lit comme trois fichiers qui
                     # n'ont pas emis.
                     #
                     # Meme correction que `language_route` il y a une heure: un
                     # jeton `cle=valeur` separe par des espaces ne peut pas
                     # CONTENIR d'espace.
                     "offset_origin": ("stated" if stated_offset != None
                                       else "DEFAULTED_plan_carries_no_"
                                            "base_offset_ms")}]
    else:
        segments = parse_segments(segments)
    segments, unverified_ms, dropped_segments = drop_unverified_segments(segments)
    if not len(segments):
        raise merge_video_chimeric.chimeric_error(
            "every segment's offset is unverified (each shorter than the "
            "measurement's probe window); nothing can be spliced at a bounded offset")
    marker = get_marker_value_for(plan, speed_ratio, candidate_obj, master_obj)
    assembly = assemble_or_log_the_decline(
        candidate_obj, plan, unverified_ms,
        candidate_obj, master_obj,
        clamp_segments_to_candidate_head(
            clamp_segments_to_master(segments, master_obj)),
        work_dir, out_path, marker,
        speed_ratio=speed_ratio,
        # LE FLUX MAITRE SUR LEQUEL LA MESURE A ETE PRISE. C'est la seule piste
        # dont on SAIT qu'elle est calee sur le plan, et on le sait par mesure
        # et non par deduction: le plan a ete produit contre elle.
        reference_stream=plan.get("reference_stream"),
        # LA LANGUE DE COMPARAISON: celle sur laquelle la mesure a ete prise, et
        # le repli de remplissage quand le maitre ne porte pas la langue de la
        # piste (SPEC_ZONE_A.MD s4c, decision du proprietaire).
        comparison_language=plan.get("language"),
        # LE CHOIX DE PARTENAIRE PAR FICHIER, auquel la barre de fidelite a ete
        # appliquee. Distinct de la fidelite par tranche, qui vit dans chaque
        # segment: le refus cite celui-ci, la ligne de pose cite celui-la.
        stream_pairing=plan.get("candidate_stream_pairing"),
        verify=True, verify_tolerance_ms=verify_tolerance_ms)

    # Le compte-rendu porte la mesure jetee: `repair_not_compatible_videos` la
    # cite dans son entree "repaired", et elle etait jusqu'ici une locale d'ici,
    # donc invisible la-bas -- toute reparation REUSSIE levait un NameError,
    # apres avoir deja accroche l'objet a best_video. Trouve le 2026-09-03 en
    # branchant le balayage sur cette fonction plutot que sur l'assembleur:
    # aucun test ne parcourait la branche de succes de l'orchestrateur.
    assembly["unverified_segment_ms"] = unverified_ms
    # LE JOURNAL EST ECRIT ICI, avant que l'objet video soit construit: si la
    # relecture du fichier produit echoue, on veut quand meme savoir ce qui a
    # ete fait a chaque piste. Un journal ecrit seulement en cas de succes ne
    # documente jamais les cas qui en avaient besoin.
    try:
        log_assembly(candidate_obj.filePath, assembly, plan)
    except Exception as error:
        tools.logs.append(f"repair: could not write the per-track log: {error}\n")

    repaired_obj = video.video(path.dirname(out_path), path.basename(out_path))
    # `generate_new_file` ne verifie pas qu'il reste une piste audio: le
    # candidat peut n'apporter que des sous-titres, et c'est un resultat valide.
    repaired_obj.need_one_audio_track = False
    repaired_obj.get_mediadata()
    # Exigence 1. Zero, et non None: la reparation a deja pose le contenu sur la
    # timeline du maitre.
    repaired_obj.delay_same_md5_audio = Decimal('0')

    # Exigence de SPEC_ZONE_A.MD s4 cote memoire. A savoir, et a dire: cette cle
    # n'est lue par personne aujourd'hui. Les deux appels de `keep_best_audio`
    # (mergeVideo.py:1853, et :998 via :1861) parcourent les dicts de
    # `out_video_metadata`, un objet neuf construit sur le fichier fusionne
    # (:1822) et rempli par `get_mediadata` (:1823) -- qui ne pose jamais
    # `fabricated`. Le marqueur revient du fichier sous
    # `extra['VMSAM_FABRICATED']`. Mesure le 2026-09-03. On pose quand meme la
    # cle: c'est le contrat, et le jour ou le consommateur sera corrige elle
    # sera la.
    mark_audio_dicts(repaired_obj, assembly["marker"])
    # LES SEGMENTS JETES VOYAGENT AVEC L'ASSEMBLAGE, pour que le journal puisse
    # les nommer. Ils etaient comptes (`unverified_segment_ms`) et jamais dits.
    assembly["dropped_segments"] = dropped_segments

    return repaired_obj, assembly


def mark_audio_dicts(repaired_obj, marker):
    if not len(marker):
        return
    # PAS LES COMMENTAIRES: l'assemblage n'en construit plus, donc en marquer un
    # serait ecrire "fabrique" sur une piste copiee -- un enregistrement de
    # provenance FAUX, ce qui est pire qu'aucun. L'audio-description reste
    # marquee tant que le proprietaire n'a pas tranche; l'incoherence est
    # voulue et documentee.
    for holder in (repaired_obj.audios, repaired_obj.audiodesc):
        for language, audios in holder.items():
            for audio in audios:
                audio["fabricated"] = marker


def quanta(value_ms, quantum_ms):
    """Un ecart exprime en FENETRES, pas en millisecondes.

    `SPEC_ZONE_A.MD` s5: un seuil en fenetres est le meme seuil sur tous les
    fichiers; en millisecondes c'en est un different sur chacun, parce que le
    quantum de la mesure varie d'un appel a l'autre -- 124 a 142 ms mesures.
    """
    if value_ms == None or quantum_ms in (None, 0):
        return None
    try:
        return round(float(Decimal(str(value_ms)) / Decimal(str(quantum_ms))), 2)
    except Exception:
        return None


def _track_shortfall_ms(assembly, report):
    """De combien CETTE piste produite est-elle plus courte que le maitre?

    Le controle de sortie a deja lu les durees par flux; elles etaient dans
    l'artefact et personne ne les rapprochait de la piste. Renvoie None quand la
    duree n'a pas ete lue -- JAMAIS zero: une duree non mesuree n'est pas une
    piste de longueur juste.
    """
    check = assembly.get("output_check") or {}
    expected = check.get("expected_duration_ms")
    if expected == None:
        return None
    for stream in check.get("streams") or []:
        if stream.get("codec_type") != "audio":
            continue
        if str(stream.get("language")) != str(report.get("language")):
            continue
        if stream.get("duration_ms") == None:
            return None
        return Decimal(str(expected)) - Decimal(str(stream["duration_ms"]))
    return None


def check_ratio_convention(plan):
    """`speed_ratio` est-il dans MA convention? Renvoie un refus, ou None.

    Ma convention, `TASKS/009`: r = duree_maitre / duree_candidat, donc r > 1
    veut dire que le candidat court VITE et doit etre RALENTI.

    On ne decide que si la reponse est NETTE: le bon sens a moins de 1 % du
    rapport des durees ET le sens inverse au-dela. Sinon on ne dit rien, parce
    qu'un rapport de durees n'est une cadence que si les deux fichiers portent
    la meme quantite de contenu -- ce qui est faux des qu'il y a une coupe.
    """
    ratio = plan.get("speed_ratio")
    master_s = plan.get("duration_master_s")
    candidate_s = plan.get("duration_candidate_s")
    if ratio == None or master_s in (None, 0) or candidate_s in (None, 0):
        return None
    try:
        r = Decimal(str(ratio))
        expected = Decimal(str(master_s)) / Decimal(str(candidate_s))
    except Exception:
        return None
    if r == 0:
        return None
    direct = abs(r - expected)
    inverse = abs((Decimal(1) / r) - expected)
    near = expected * Decimal("0.01")
    if inverse <= near and direct > near:
        return ("the speed_ratio looks like the RECIPROCAL of this module's "
                "convention: TASKS/009 defines r = master_span / candidate_span, "
                "and the value shipped matches candidate_span / master_span "
                "against the durations in the same plan. Applying it would "
                "resample in the WRONG DIRECTION")
    return None


# LA OU LES DUREES NE PEUVENT PAS TRANCHER, ET C'EST LE CAS DESTRUCTEUR.
#
# `check_ratio_convention` n'attrape l'inversion que LOIN de l'unite -- c'est-a-
# dire exactement la ou mes bornes et mon verificateur l'attrapaient deja. Pres
# de l'unite il ne dit rien, parce qu'un rapport de durees n'est pas une cadence
# des qu'il y a une coupe: sur l'id 33 les durees sont dans un rapport de 1.0687
# pour une cadence de 1.001.
#
# Or 0.999001 contre 1.000999 passe TOUTE borne et TOUTE tolerance. C'est 0.2 %
# dans le mauvais sens, livrable en silence, et vmsam-dev-1 l'a nomme comme le
# cas destructeur.
#
# DONC: pres de l'unite, un rapport SANS CONVENTION DECLAREE ne s'applique pas.
# Ce n'est pas de la prudence, c'est la seule position defendable: aucun controle
# de ce module ne peut distinguer les deux sens la, donc appliquer revient a
# parier sur l'identite de l'agent qui a ecrit le champ.
RATIO_CONVENTION = "master_span / candidate_span"
CONVENTION_FREE_MARGIN = Decimal("0.01")


def normalise_convention(stated):
    """`mine` / `inverse` / `unknown`, en IGNORANT la forme.

    PREMIERE VERSION: egalite de chaines apres suppression des espaces. Elle a
    REFUSE le premier plan correctement etiquete que vmsam-dev-1 m'ait envoye,
    parce que l'etiquette portait une glose:

        "master_span / candidate_span  (dev-2's definition, TASKS/009)"

    La convention est JUSTE. Seule la FORME differait. C'est la regle que j'ai
    adoptee ce matin -- un controle qui echoue pour une raison de forme est un
    controle qu'on eteint -- et je l'ai enfreinte quelques heures plus tard, dans
    un controle ecrit pour empecher une inversion.

    On lit donc l'ORDRE DES DEUX TERMES et rien d'autre: la glose, la casse, les
    espaces et la ponctuation ne portent aucun sens ici.
    """
    text = str(stated).lower()
    # TROISIEME FORME, ET LA TRONCATURE ETAIT ELLE-MEME LA FAUTE.
    #
    # La version precedente coupait au premier `(` pour jeter une glose EN
    # SUFFIXE. Sur une glose ENVELOPPANTE -- `"ratio (master_span /
    # candidate_span)"` -- la coupe tombe AVANT les deux termes, il ne reste que
    # `"ratio "`, et un plan CORRECTEMENT ETIQUETE est refuse.
    #
    # Trouve par `vmsam-auditor`. Cette fonction DOCUMENTE la regle -- un
    # controle qui echoue pour une raison de forme est un controle qu'on eteint
    # -- DOCUMENTE l'avoir enfreinte une fois, a ete reecrite pour cesser de
    # l'enfreindre, ET L'ENFREINT UNE FORME PLUS LOIN. Lire une regle installe la
    # RECONNAISSANCE, pas l'EVITEMENT.
    #
    # ON NE COUPE DONC PLUS RIEN. On cherche les deux termes dans TOUTE la
    # chaine: une glose en suffixe laisse deja les termes dans le bon ordre avant
    # elle, et une glose enveloppante les laisse dans le bon ordre dedans. La
    # troncature ne protegeait contre rien et coutait une forme entiere.
    master = text.find("master")
    candidate = text.find("candidate")
    if master == -1 or candidate == -1:
        return "unknown"
    return "mine" if master < candidate else "inverse"


def check_ratio_labelled(plan):
    """La convention est-elle DECLAREE, et est-ce la mienne? Refus, ou None."""
    ratio = plan.get("speed_ratio")
    if ratio == None:
        return None
    stated = plan.get("speed_ratio_convention")
    if stated != None:
        named = normalise_convention(stated)
        if named == "mine":
            return None
        if named == "inverse":
            return (f"the plan states its ratio convention as {stated!r}, which "
                    f"is the RECIPROCAL of {RATIO_CONVENTION!r}; this module will "
                    f"not reinterpret a coefficient whose meaning it did not define")
        return (f"the plan states a ratio convention this module does not "
                f"recognise ({stated!r}); it applies {RATIO_CONVENTION!r} and a "
                f"convention it cannot read is not a convention it can trust")
    try:
        distance = abs(Decimal(str(ratio)) - Decimal(1))
    except Exception:
        return None
    if distance < CONVENTION_FREE_MARGIN:
        return ("the plan carries no speed_ratio_convention and the ratio is "
                "within 1% of unity, where NEITHER the bounds check NOR the "
                "verifier can tell the two directions apart. An unlabelled "
                "near-unity coefficient is not applied")
    return None


def _margin_fields(plan):
    """`speed_margin=` when it exists, the REASON when it does not.

    Four fields, each with its own emitter, because they answer four questions
    and a single conditional emitter makes them disappear together:

        speed_margin                 the plateau margin, in ms
        speed_margin_absent_reason   why it does not exist
        fidelity_margin              another quantity, unitless
        decided_by                   which criterion decided

    `NOT EMITTED` at the consumer must fire only when the margin is absent AND
    no reason accompanies it -- that is, when the producer said nothing. Until
    today it fired on the INVERSE case.
    """
    if not plan:
        return ""
    # THREE STATES, THREE TOKENS. There were TWO, and the third emitted NOTHING
    # -- so a line without `speed_margin` did not distinguish "the plan did not
    # carry it" from "this line predates the field".
    #
    # I FIRST REFUSED TO ADD IT, arguing that `repair: build` dates the line
    # without a per-field marker. `vmsam-dev-4` MEASURED it and the argument
    # falls:
    #
    #     job logs                          28
    #     carrying a `repair: build` line   12
    #     NOT CARRYING ONE                  16   <- the OLD ones
    #
    # THE MARKER IS MISSING PRECISELY ON THE ARTEFACTS WHOSE AGE IS THE
    # QUESTION. A provenance line that POSTDATES the artefacts it would date
    # cannot date them -- it is itself a field that arrived at some moment.
    #
    # AND WHERE IT IS PRESENT, IT DOES NOT ORDER: two content digests, no
    # timestamp, no sequence. A reader sees that two artefacts come from
    # different builds and nothing says which is first.
    # dev-4: A DIGEST IS NOT A DATE -- a sentence it had written elsewhere in
    # its own module and that my proposal made it rediscover.
    #
    # No space inside the token, like `language_route` and the DEFAULTED marker.
    #
    # NO PRODUCER WRITES THESE FOUR KEYS. VERIFIED, 2026-09-05.
    #
    #     grep -c "speed_margin\|fidelity_margin\|decided_by" \
    #          src/change_point_locator.py
    #       my tree             0
    #       the authority 24bf25f  0   (= dev-1's tree, 25 keys)
    #     control on the finder -- `quantum_ms`, a name that IS present: 11 and 12.
    #     THE GREP FIRES, so the zero is a measurement and not a mute finder.
    #
    # The only producer feeding this emitter is
    # `change_point_locator.locate_change_points` (see 1893 then 294), and its
    # dictionary carries none of the four. `merge_plan_report` (dev-4) FORWARDS
    # them, this module READS them, no module ORIGINATES them.
    #
    # I FIRST WROTE THE ABSENCE AS AN ENUMERATION -- "21 keys at line 1209" --
    # and it was ALREADY FALSE in dev-1's tree and at the authority, which I had
    # fetched half an hour earlier. AN ABSENCE PROVED BY A GREP DOES NOT DEPEND
    # ON WHICH REVISION YOU HOLD; AN ABSENCE PROVED BY AN ENUMERATION DOES.
    # This is the portable form.
    #
    # SO WHAT THESE THREE TOKENS SAY TODAY:
    #   `absent(not_in_plan)` is TRUE at every emission and cannot vary.
    #   It does NOT say "this plan had no margin" -- it says "four keys were
    #   designed, consumed, and never originated". dev-4 had filed the defect
    #   AGAINST A PRODUCER THAT DOES NOT EXIST (a named seam, a described
    #   mechanism) and corrected it to NO WRITER EXISTS.
    #
    # DO NOT REMOVE THESE THREE FIELDS WHILE (a) IS OPEN -- AND REMOVE THEM OR
    # FILL THEM AS SOON AS IT IS DECIDED. Lead's ruling, 2026-09-05, AMENDED THE
    # SAME DAY AFTER dev-4's OBJECTION.
    #
    # The initial ruling said KEEP with no end, and dev-4 objected: A FIELD KEPT
    # ALIVE SO THAT A DOWNSTREAM ROW STAYS STABLE IS A CONSTANT THAT CANNOT
    # VARY, AND A VALUE THAT CANNOT VARY IS NOT EVIDENCE -- the very rule the
    # whole team applies to other people's denominators. A permanent KEEP would
    # have institutionalised the defect catalogued all night. Objection upheld.
    #
    # WHAT SURVIVES OF THE RULING, AND IT BEARS ON THE *WHEN* AND NOT ON THE
    # *ALWAYS*: after a cut, the fact lives only in dev-4's register and in this
    # comment -- IN THE OBSERVERS, NOT IN THE PRODUCED RECORD. An artefact
    # travels; a register and a comment do not. A reader six months from now
    # holding only a log would see nothing, and silence is what nobody
    # investigates.
    #
    # SO THIS TOKEN IS NOT A PERMANENT RECORD: IT IS THE MARKER OF AN OPEN
    # DECISION, and it has an OBSERVABLE END.
    #
    #   (a) open                 keep -- the artefact carries the evidence that
    #                            it is open
    #   (a) says ORIGINATE       the fields FILL. Nothing to cut.
    #   (a) says ABANDON         the fields COME OUT, in the order dev-4 set:
    #                            its register changes FIRST, my bytes AFTER, so
    #                            that no artefact renders a state its reader
    #                            cannot explain.
    #
    # So it is not a constant that cannot vary: it is a constant whose variation
    # is the owner's decision, and that decision is PENDING and not ABSENT. If
    # you are reading this and (a) has been decided, THIS BLOCK IS STALE -- act,
    # do not copy it forward.
    #
    # AWAITING AN OWNER DECISION: does the locator originate the four keys?
    # dev-1 REFUSED to add them, not on the merits but under a standing
    # instruction from ITS OWN user (report defects, fix none). `WRITE_ZONES`
    # grants it the SCOPE; its user constrains the ACTION, and the narrower
    # constraint governs.
    parts = []
    margin = get_speed_margin(plan)
    if margin != None:
        parts.append(f"speed_margin={margin}")
    else:
        reason = plan.get("speed_margin_absent_reason")
        parts.append(f"speed_margin=absent({reason})" if reason != None
                     else "speed_margin=absent(not_in_plan)")
    fidelity = plan.get("fidelity_margin")
    parts.append(f"fidelity_margin={fidelity}" if fidelity != None
                 else "fidelity_margin=absent(not_in_plan)")
    decided = plan.get("decided_by")
    parts.append(f"decided_by={decided}" if decided != None
                 else "decided_by=absent(not_in_plan)")
    return (" ".join(parts) + " ") if len(parts) else ""


def compare_plan_master(plan, best_video):
    """Le plan a-t-il ete mesure contre CE maitre? Renvoie une raison, ou None.

    TROIS ETATS ET NON DEUX, et le troisieme a ete trouve en faisant tourner ce
    lecteur sur les VRAIS octets de vmsam-dev-1 plutot que sur le contrat:

        absent          rien a comparer -- le plan ne nomme pas de maitre
        egal            meme maitre
        different       maitres differents  -> DECLIN, et c'est le controle
        INCOMPARABLE    la valeur n'est pas un chemin: `WRITE_ZONES.MD` s8 dit
                        de RETENIR plutot que d'assainir, et dev-1 emet donc un
                        jeton opaque. `'opaque:...' != '/srv/...'` est VRAI, donc
                        l'ancienne ligne declinait TOUT plan portant un jeton --
                        en disant `mesure contre un autre maitre`, ce qui est
                        FAUX. Une raison fausse est pire qu'un refus: elle envoie
                        le lecteur chercher un desaccord de maitre qui n'existe
                        pas.

    ON DECLINE QUAND MEME dans le cas incomparable -- ne pas pouvoir verifier
    l'identite du maitre n'autorise pas a l'assumer -- mais la raison DIT
    laquelle des deux choses s'est produite. `AGENT.MD`: je n'ai pas pu mesurer
    n'est pas un verdict sur le fichier.
    """
    # LE DIGEST D'ABORD QUAND IL EXISTE: c'est la seule forme comparable qui ne
    # fait voyager aucun texte libre. `WRITE_ZONES.MD` s8.
    #
    # ET SA LIMITE SE DIT, parce que vmsam-dev-1 l'a nommee avant moi: un digest
    # de CHEMIN prouve que deux agents ont recu la meme CHAINE, pas le meme
    # FICHIER. Un lien symbolique, une barre finale, un prefixe de montage ou une
    # normalisation unicode differente donnent un digest different pour les memes
    # octets sur le disque. Un desaccord de digest n'est donc PAS une preuve de
    # maitre different: c'est le meme etat `non verifie`, un cran plus bas.
    digest = plan.get("master_path_digest")
    if digest != None:
        import hashlib
        mine = hashlib.sha256(best_video.filePath.encode()).hexdigest()
        if mine == digest:
            return None
        return ("the plan's master path digest does not match this master's. "
                "NOTE: a path digest proves two agents were handed the same "
                "STRING, not the same FILE -- a symlink, a mount prefix or a "
                "different unicode normalisation differs here too, so this is "
                "UNVERIFIED rather than proof of a different master")
    stated = plan.get("master_path")
    if stated == None or stated == best_video.filePath:
        return None
    if not str(stated).startswith("/"):
        return ("the plan names its master with a token this reader cannot "
                "compare to a filesystem path, so the master's identity is "
                "UNVERIFIED -- this is not evidence of a different master")
    return ("the plan was measured against a different master than the "
            "one selected here")


def _head_pad_summary(report):
    """De quoi le total de rembourrage de tete est-il fait.

        unmeasured   le debut du flux n'a pas ete lu -- on ne sait pas
        read_past    le flux commence apres zero et le plan lit deja au-dela:
                     UN DECALAGE EXISTE et ne coute aucun rembourrage
        none         le flux commence vraiment a zero
        padded       du silence a ete ajoute, et combien

    `head_pad_ms=0` ecrivait les trois premieres avec le meme chiffre. Une valeur
    et son absence ne doivent pas imprimer le meme jeton -- et c'est pourquoi
    aucune hypothese de decalage de conteneur n'etait ni confirmable ni
    refutable depuis le journal.
    """
    decisions = report.get("head_decisions")
    if decisions == None:
        # NI ZERO NI VIDE: ce rapport vient d'un assembleur qui ne produisait pas
        # encore le champ. Le dire evite qu'un lecteur compte une absence de
        # format comme une absence de decision.
        # LA CAUSE EST VERIFIEE, PAS SUPPOSEE -- ET LE JETON DIT LEQUEL DES DEUX.
        #
        # Balayage de ma propre classe apres celui de `vmsam-dev-4`: cinq replis
        # qui NOMMENT une cause dans mes deux modules. Celui-ci a survecu au
        # controle -- `head_decisions` atteint bien le rapport, verifie sur deux
        # runs reels qui rendent `none=2,padded=1,read_past=2` -- la ou `why=` ne
        # traversait jamais et affirmait une cause fausse 48 fois sur 48.
        #
        # MAIS MA VERIFICATION EST "DEUX FICHIERS CE SOIR", pas une preuve
        # qu'aucun autre chemin ne peut rendre None. Le jeton dit donc ce qui est
        # OBSERVABLE puis ce qui est ATTENDU, et un lecteur qui trouve ce jeton
        # sur un assemblage recent tient une trouvaille plutot qu'une explication.
        return ("unreported(no head_decisions on this report; expected only for "
                "assemblies predating the field)")
    if not len(decisions):
        return "no-candidate-piece"
    counts = {}
    for decision in decisions:
        counts[decision["outcome"]] = counts.get(decision["outcome"], 0) + 1
    return ",".join(f"{name}={counts[name]}" for name in sorted(counts))


def _shortfall_annotation(assembly, report):
    """Ce que la source explique, ce que la piste a perdu, et le RESTE.

    Etait une seule expression conditionnelle avec deux operateurs morse dedans.
    Elle etait juste et illisible, et une ligne qu'on ne relit pas est une ligne
    ou un signe se cache.

    ET UN SIGNE S'Y CACHAIT. `UNEXPLAINED` etait emis SIGNE, donc un artefact
    reel a 7 pistes portait `UNEXPLAINED -21.0 ms` sur chacune. Le calcul est
    juste -- la piste a perdu 21 ms de MOINS que la source n'etait courte -- mais
    le mot dit une PERTE, et une perte negative n'a pas de sens pour un lecteur.
    Meme classe que `verify=skipped` sans cause: un champ exact et illisible.

    `UNEXPLAINED` ne descend donc plus sous zero, et le sur-compte se DIT au lieu
    d'etre encode dans un signe que personne n'attendait. La valeur n'est pas
    perdue, elle est nommee.
    """
    lost = _track_shortfall_ms(assembly, report)
    short = report.get("fill_short_by_ms")
    if short:
        if lost == None:
            return "[FILL SOURCE SHORT BY " + str(short) + " ms; TRACK LOSS UNMEASURED]"
        residual = Decimal(str(lost)) - Decimal(str(short))
        if residual > 0:
            tail = "UNEXPLAINED " + str(residual) + " ms"
        else:
            tail = ("UNEXPLAINED 0 ms (the fill shortfall over-accounts by "
                    + str(-residual) + " ms)")
        return ("[FILL SOURCE SHORT BY " + str(short) + " ms; TRACK LOST "
                + str(lost) + " ms; " + tail + "]")
    if lost != None and lost > 0:
        return "[TRACK LOST " + str(lost) + " ms, NO SHORT FILL SOURCE -- UNEXPLAINED]"
    return ""


# LES OCTETS DE CE MODULE, HACHES A SON PROPRE IMPORT. Meme raison que dans
# `merge_video_chimeric`: a l'appel on hache le FICHIER, a l'import on hache ce
# qui vient d'etre compile en memoire. Chaque module hache LES SIENS, parce que
# les deux ne sont pas importes au meme instant et qu'un condensat pris ailleurs
# redeviendrait un condensat de fichier.
def _digest_of_loaded_source():
    import hashlib
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except Exception:
        return "unreadable"


LOADED_SOURCE_DIGEST = _digest_of_loaded_source()


# LE CONDENSAT DES SOURCES QUI TOURNENT, CALCULE UNE FOIS PAR PROCESSUS.
_sources_digest_cache = None

# LA PORTEE EST CELLE DE L'IMAGE, PAS CELLE DU DEPOT, ET LES DEUX DIFFERENT.
#
#   Dockerfile:142  COPY src/*.ini src/*.py ...  -> /home/vmsam/
#   Dockerfile:143  COPY src/gestionar_show      -> /home/vmsam/gestionar_show/
#   Dockerfile:144  COPY src/gestionar_movie     -> /home/vmsam/gestionar_movie/
#
# `COPY src/*.py` N'EST PAS RECURSIF. Mesure de vmsam-ci: 17 fichiers a plat,
# 7 dans gestionar_show, 3 dans gestionar_movie -- 27 EXPEDIES -- contre 28 pour
# un `src/**.py` recursif. UN CONDENSAT A PLAT MANQUE DIX FICHIERS QUI PARTENT;
# UN CONDENSAT RECURSIF EN INCLUT UN QUI NE PART PAS.
#
# Le vingt-huitieme est `src/tools/database.py`, 52 octets, une docstring sans
# code, sans importateur, dans un repertoire sans `__init__.py`. Inoffensif --
# et il ferait diverger un condensat recursif de l'image EN PERMANENCE, pour une
# raison qu'aucun lecteur ne devinerait. C'est ce genre d'ecart inexplique qui
# fait desactiver un bon controle.
#
# ON ENUMERE DONC LES TROIS CIBLES `COPY` et pas un motif recursif, ET ON PART DE
# `__file__`: dans l'image la racine est `/home/vmsam`, dans une copie du depot
# c'est `src/`. Enumerer les memes cibles depuis la racine du module fait que les
# deux DOIVENT concorder -- et un desaccord devient une mesure au lieu d'un
# artefact de chemin.
SOURCE_SCOPE = ("*.py", "gestionar_show/**/*.py", "gestionar_movie/**/*.py")


def sources_digest():
    """Condensat du CODE DEPLOYE, lu sur le disque a l'execution.

    CE QU'IL REPOND, ET QUE `org.opencontainers.image.revision` NE REPOND PAS:
    `vmsam-ci` attend qu'une image annonce la revision visee et NE REGARDE JAMAIS
    LES OCTETS. Or `Dockerfile:137 ARG VMSAM_GIT_COMMIT` et `Dockerfile:142 COPY`
    sont poses INDEPENDAMMENT: une image construite depuis un arbre sale ou en
    avance porte l'etiquette qu'on lui a passee et le controle passe. Ce soir
    l'arbre du relais porte `validate_job` et la reference forgejo ne l'a pas --
    DEUX IMAGES, MEME ETIQUETTE DE REVISION, CODE DIFFERENT.

    A L'EXECUTION ET NON A LA CONSTRUCTION, et c'est la moitie qui compte: un
    condensat calcule a la construction resume le CONTEXTE DE CONSTRUCTION et se
    transmet exactement comme `VMSAM_GIT_COMMIT`. Celui-ci lit ce qui est
    reellement dans l'image.

    CE QU'IL N'IDENTIFIE PAS, ET LE CHAMP LE DIT: 27 fichiers `.py`. Pas
    l'interprete, pas ffmpeg, pas mkvtoolnix -- installes NON EPINGLES depuis
    Debian testing, ce que ci signale depuis le debut comme la moitie que
    `image_git_commit` n'a jamais identifiee. UN CONDENSAT DONT LA COUVERTURE
    N'EST PAS DITE REDEVIENT UNE ETIQUETTE.

    `files=` accompagne le condensat parce qu'un condensat sur un ENSEMBLE ne
    veut rien dire sans la taille de l'ensemble: un deploiement qui PERD un
    fichier change le sha, et sans le compte on ne le distingue pas d'une
    modification.
    """
    global _sources_digest_cache
    if _sources_digest_cache != None:
        return _sources_digest_cache
    import glob, hashlib
    root = path.dirname(path.abspath(__file__))
    found = {}
    for pattern in SOURCE_SCOPE:
        for name in glob.glob(path.join(root, pattern), recursive=True):
            if path.isfile(name):
                found[path.relpath(name, root)] = name
    per_file, rolled = [], hashlib.sha256()
    # TRI PAR CHEMIN AVANT DE CONDENSER: contenu-et-nom, pas ordre de repertoire.
    # `glob` ne garantit pas d'ordre, donc sans ceci le meme code rendrait des
    # condensats differents selon le systeme de fichiers.
    for relative in sorted(found):
        try:
            with open(found[relative], "rb") as handle:
                payload = handle.read()
        except OSError as error:
            # UN FICHIER ILLISIBLE EST NOMME, PAS SAUTE. Le sauter rendrait le
            # meme condensat qu'un deploiement ou il est absent, et les deux
            # situations demandent des actions differentes.
            digest = f"unreadable({type(error).__name__})"
            rolled.update(relative.encode("utf-8") + b"\x00" + digest.encode("utf-8") + b"\n")
            per_file.append({"path": relative, "sha12": digest})
            continue
        one = hashlib.sha256(payload).hexdigest()
        rolled.update(relative.encode("utf-8") + b"\x00" + one.encode("utf-8") + b"\n")
        per_file.append({"path": relative, "sha12": one[:12], "bytes": len(payload)})
    _sources_digest_cache = {"sha12": rolled.hexdigest()[:12],
                             "files": len(per_file),
                             "scope": " + ".join(SOURCE_SCOPE),
                             "root": root,
                             "per_file": per_file}
    return _sources_digest_cache


def write_sources_manifest():
    """Ecrit le detail par fichier UNE FOIS, et rend son chemin ou None.

    LA LIGNE DE JOURNAL PORTE LE ROULE, LE MANIFESTE PORTE LE DETAIL. `vmsam-ci`
    veut les deux et pas au meme endroit: 27 condensats sur chaque travail sont
    un journal qu'il faudrait contourner, et le roule seul ne dit que "quelque
    chose a bouge" la ou il faut "CES deux fichiers ont bouge".

    LE MANIFESTE EST DESIGNE PAR LA LIGNE PLUTOT QUE RECOPIE DEDANS -- forme de
    POINTEUR, adoptee par le Lead ce soir apres qu'une COPIE d'un compte rendu
    et son original ont diverge de trois sections. Un pointeur ne peut pas etre
    en desaccord avec ce qu'il designe; une copie l'a ete.
    """
    digest = sources_digest()
    try:
        # LE MANIFESTE EST ADRESSE PAR SON CONTENU, ET C'EST UNE CORRECTION.
        #
        # Premiere version: un nom FIXE, ecrit seulement s'il n'existait pas
        # deja. Un fichier source change, le roule change, ET LE MANIFESTE
        # RESTAIT CELUI D'AVANT -- la ligne pointait sur un detail qui ne
        # correspondait plus a son propre condensat. C'est EXACTEMENT la
        # divergence copie-contre-original que la forme pointeur existe pour
        # empecher, reconstruite a l'interieur de la forme pointeur.
        #
        # Attrape par `t80`, par le controle qui verifie que le manifeste est
        # D'ACCORD avec la ligne. Un test qui aurait seulement verifie que le
        # fichier existe serait passe.
        #
        # Le nom porte donc le condensat: un manifeste perime est IMPOSSIBLE
        # plutot qu'evite, et deux deploiements coexistent au lieu de s'ecraser.
        target = path.join(tools.tmpFolder,
                           f"vmsam_sources_{digest['sha12']}.json")
        if not path.exists(target):
            import json as _json
            with open(target, "w") as handle:
                _json.dump({"sha12": digest["sha12"], "files": digest["files"],
                            "scope": digest["scope"], "root": digest["root"],
                            "per_file": digest["per_file"]}, handle, indent=1)
        return target
    except Exception as error:
        # UN MANIFESTE QU'ON NE PEUT PAS ECRIRE NE DOIT PAS EMPECHER LA LIGNE.
        # Le roule est la donnee; le detail est un confort.
        tools.logs.append(f"repair: the sources manifest could not be written: {error}\n")
        return None


def module_fingerprint():
    """L'IDENTITE DU CODE QUI TOURNE, EMISE INCONDITIONNELLEMENT.

    `build_identity` est une absence que vmsam-dev-4 a signalee le premier jour et
    qui vient de couter une colonne a vmsam-forensic: aucun artefact ne dit par
    quelle version il a ete produit, donc sa colonne `image` nomme une pointe de
    branche que le conteneur ne fait peut-etre pas tourner. Mesure: mon `027feab`
    est PROMU a 16:11 UTC et un artefact de 17:25 UTC ne porte pas son champ.
    PROMU N'EST PAS EN COURS D'EXECUTION.

    ET SA TENTATIVE DE REPARATION A ECHOUE POUR UNE RAISON QUI EST LA REGLE DU
    JOUR: il a voulu identifier le build a partir des CHAMPS presents dans un
    journal. `FILL SOURCE SHORT BY` n'apparait que sur un fichier qui a un manque,
    donc son absence ne distingue pas `le build n'a pas le champ` de `le fichier ne
    l'a pas declenche`. UNE PRESENCE DE CHAMP EST CONFONDUE AVEC LE CONTENU DU
    FICHIER et ne peut pas servir d'empreinte.

    Un condensat de la SOURCE ne l'est pas. Il est emis sur chaque reparation,
    quel que soit le fichier, il change exactement quand le code change, et il ne
    demande a personne de penser a l'incrementer -- un numero de version a la main
    est un second exemplaire de la verite et il derive.

    CE QU'IL IDENTIFIE ET CE QU'IL N'IDENTIFIE PAS: les deux modules de
    reparation, et rien d'autre. Pas l'image, pas l'interprete, pas `mergeVideo.py`
    ni `video.py`, pas les binaires. Un lecteur qui voit deux artefacts avec le
    meme condensat sait que CE code etait identique; il ne sait pas que le reste
    l'etait.
    """
    # ON NE RELIT PLUS LES FICHIERS ICI. Chaque module a hache SES PROPRES octets
    # AU MOMENT DE SON IMPORT; on assemble ces constantes. Relire a l'appel
    # rendait le condensat du FICHIER et non du CODE CHARGE -- et un processus de
    # longue duree fait diverger les deux, ce qui est arrive ce soir sur mon
    # propre balayage.
    #
    # IMPORT TARDIF, comme partout ailleurs dans ce module: la tete de
    # `mergeVideo.py` est hors zone taguee et un deploiement partiel ne doit pas
    # pouvoir empecher le demarrage.
    parts = [f"{path.basename(__file__)}:{LOADED_SOURCE_DIGEST}"]
    try:
        import merge_video_chimeric as _chi
        parts.append(f"{path.basename(_chi.__file__)}:"
                     f"{getattr(_chi, 'LOADED_SOURCE_DIGEST', 'unreported')}")
    except Exception:
        parts.append("merge_video_chimeric.py:unimportable")
    return " ".join(parts)


def master_fill_offset(region):
    """Le decalage REEL d'un morceau maitre, calcule et non affirme.

    Il etait le LITTERAL `0`. Un invariant affirme par une constante ne peut pas
    detecter sa propre violation: si `normalize_segments` posait un jour un
    morceau maitre a un decalage, la ligne imprimerait ENCORE 0. C'est la forme
    `check(..., True)` dans un champ emis, et dans le champ demande precisement
    pour que l'invariant casse SE VOIE -- `vmsam-dev-3` l'a demande parce que
    SANS LUI un invariant casse et une erreur de placement sont la meme
    observation. Avec un litteral, ils le restent.
    """
    try:
        return (Decimal(str(region.get("source_start_ms")))
                - Decimal(str(region.get("master_start_ms"))))
    except Exception:
        # `unreported` et non `0`: un morceau sans bornes lisibles n'a pas un
        # decalage nul, il n'en a pas de connu.
        return "unreported(bounds unreadable)"


def log_assembly(candidate_path, assembly, plan):
    """CE QUI A ETE FAIT AU FICHIER, PISTE PAR PISTE, AVEC LES TIMINGS.

    `SPEC_ZONE_A.MD` s4e. Un compte de pistes reconstruites est un enonce sur le
    travail fait et pas sur un fichier: une reparation a rapporte "7 audio et 24
    sous-titres reconstruits, 0 refuse, 0 en echec" ET LIVRE UN FICHIER TRONQUE.
    Ce que ces lignes doivent permettre, et que des comptes ne permettent pas:
    dire QUELS fichiers sont concernes en lisant le journal, sans reconstruire.

    UN SAUT EST UNE DECISION ET A SA LIGNE. Une piste refusee ou en echec est
    ecrite avec sa raison, pas omise: une omission se lit comme "il n'y en avait
    pas".
    """
    quantum_ms = plan.get("quantum_ms") if plan else None
    pieces = assembly.get("pieces") or []
    spans = []
    for piece in pieces:
        start = Decimal(str(piece["master_start_ms"]))
        end = Decimal(str(piece["master_end_ms"]))
        # LA MEME BORNE, LE MEME NOMBRE. `int()` TRONQUAIT.
        #
        # `vmsam-dev-4` a mesure la consequence: la ligne `plan` disait `m0-983`
        # et la ligne `ADDED` disait `master 0-983.54` -- UNE borne, DEUX
        # representations, DU MEME PRODUCTEUR, sur deux lignes. Sa cle de
        # jointure exacte ne les appariait pas, et 11 regions sur 5 artefacts
        # rendaient `source: absent` pendant que le `from=master/<lang>` etait
        # trois lignes plus bas dans le meme journal.
        #
        # UN RENDU AVEC PERTE D'UNE VALEUR EMISE AILLEURS SANS PERTE EST UNE
        # SECONDE REPRESENTATION QUI PEUT DIVERGER -- et celle-ci divergeait sur
        # toute borne fractionnaire, ce que le decalage par flux produit
        # normalement.
        #
        # dev-4 a construit une jointure par TRONCATURE qui recupere les
        # anciennes lignes et l'annonce (`matched_by=[joined by TRUNCATION...]`),
        # donc les deux formes restent lisibles chez lui. Ce changement fait que
        # les nouvelles s'apparient EXACTEMENT et que sa voie de secours cesse
        # d'etre le cas normal.
        spans.append(f"{piece['source'][0]}{start}-{end}")
    # LA LANGUE DE MESURE SUR LA LIGNE DU PLAN. C'est elle qui decide quelles
    # pistes ont leur propre decalage et lesquelles empruntent, et elle
    # n'apparaissait nulle part dans le journal -- on pouvait lire `BORROWED`
    # sans pouvoir dire emprunte A QUOI.
    # LE MAITRE EST NOMME. Le journal nommait le candidat et la sortie et jamais
    # le maitre -- "merged <candidate> with the master into <output>", UN ARTICLE
    # DEFINI LA OU IL FAUT UN IDENTIFIANT.
    #
    # Consequence, rapportee par le validateur: il peut verifier un fichier
    # produit contre ses propres affirmations et contre son candidat, ET PAS
    # CONTRE CE A PARTIR DE QUOI IL A ETE CONSTRUIT. C'est exactement la question
    # que la colonne VALIDATED existe pour poser, et elle etait sans reponse.
    # Il a cherche le maitre dans les racines voisines, le repertoire de sortie
    # et le code d'episode, sans le retrouver pour aucun des trois fichiers.
    # PREMIERE LIGNE DE CHAQUE REPARATION: QUEL CODE A TOURNE. Inconditionnelle,
    # donc utilisable comme empreinte -- contrairement a la presence d'un champ,
    # qui depend du fichier.
    tools.logs.append(f"repair: build {module_fingerprint()}\n")
    # LE CODE DEPLOYE, PAR ARTEFACT. `vmsam-ci` a demande cette forme plutot que
    # `/health`: un point d'ancrage PAR ARTEFACT survit a un redeploiement en
    # cours de run, ce qu'un condensat par conteneur ne sait pas exprimer.
    _sources = sources_digest()
    _manifest = write_sources_manifest()
    tools.logs.append(f"repair: sources {_sources['sha12']} "
                      f"files={_sources['files']} scope={_sources['scope']} "
                      f"manifest={_manifest or 'unwritten'}\n")
    if plan and plan.get("master_path"):
        tools.logs.append(f"repair: master {plan['master_path']}\n")
    # L'IDENTITE DU CANDIDAT, SANS SON CHEMIN.
    #
    # `repair: master` nomme le maitre; RIEN ne nommait le candidat, parce que
    # son chemin est exactement ce que `WRITE_ZONES.MD` s8 dit de ne pas emettre.
    # Consequence trouvee par vmsam-dev-4 en comptant SON corpus: son unite est
    # la PAIRE (maitre, candidat), et sans le second terme deux candidats
    # fusionnes vers un meme maitre se replient en un seul cas. Son compte a la
    # main donnait 15, la mesure en donne 16.
    #
    # LE RETRAIT D'UN CHAMP A RENDU UN CONSOMMATEUR INCAPABLE DE COMPTER, et il
    # a fallu son probleme de regroupement pour le voir.
    #
    # Un digest satisfait les deux: il ne porte aucun texte libre et il resout
    # l'ambiguite au lieu de la documenter. Meme construction que
    # `master_path_digest`, convenue avec vmsam-dev-1: sha256 des octets du
    # chemin. Sa limite est la meme et se dit ailleurs -- un digest de CHEMIN
    # prouve que deux agents ont recu la meme CHAINE, pas le meme FICHIER.
    if candidate_path:
        import hashlib
        tools.logs.append(
            f"repair: candidate_digest "
            f"{hashlib.sha256(str(candidate_path).encode()).hexdigest()}\n")
    # LES SEGMENTS JETES PAR LE LOCALISATEUR, SUR LA LIGNE DU PLAN.
    #
    # `segments_dropped_unusable` arrive dans le dict du plan et MOURAIT ICI --
    # troisieme quantite ce soir a atteindre le plan et a ne jamais etre emise,
    # apres `step_floor_ms` et `plateau_tolerance_ms`. Et sa trace cote
    # localisateur est `_log(...)`, qui est GARDEE PAR `tools.dev`: en production
    # elle n'est jamais ecrite. Donc un plateau jete ne laissait AUCUNE trace
    # nulle part.
    #
    # C'EST LA QUESTION QUE L'ARCHITECTE NE POUVAIT PAS TRANCHER: un trou de tete
    # parce que le candidat n'a rien a donner, ou parce qu'un plateau a ete jete.
    # Mesure sur mon propre bras conserve: 2 fichiers sur 10 ont jete un segment.
    # Ce n'est pas rare.
    # LES BRACKETS, SUR LEURS PROPRES LIGNES. UNE LIGNE, ET TROIS AGENTS ETAIENT
    # BLOQUES DERRIERE.
    #
    # `bracket_is_bound_only` avait UNE SEULE occurrence dans `src/`: sa propre
    # affectation. LE LOCALISATEUR SAIT QUAND SA REPONSE EST GROSSIERE, IL
    # L'ECRIT, ET RIEN NE LE PORTAIT NULLE PART. La ligne `plan` n'avait ni
    # largeur de bracket, ni drapeau, ni `step_ms`.
    #
    # CE QUE PERSONNE NE POUVAIT DISTINGUER SANS CA -- et ce sont une reparation
    # correcte et un defaut, rendus a l'identique:
    #
    #     "rempli PARCE QU'IL Y AVAIT UN TROU"
    #     "rempli PARCE QUE LE LOCALISATEUR N'ETAIT PAS SUR"
    #
    # `vmsam-dev-4` a nomme le joint avant qu'on me le confie: "rendu par le
    # producteur n'est pas emis dans le journal, et le journal est ma seule
    # entree -- l'etat n'est pas `personne ne le produit` mais `le producteur le
    # fait et ca ne traverse jamais jusqu'a log_assembly`."
    #
    # DES LIGNES A PART ET NON DES CHAMPS SUR LA LIGNE `plan`: leur nombre varie
    # avec le fichier, et un lecteur qui resout PAR NOM ne paie pas les lignes
    # supplementaires. Une ligne `plan` a longueur variable serait a tronquer.
    # LE DECALAGE DE BASE PAR SEGMENT, ET LA TABLE PAR FLUX A COTE.
    #
    # `vmsam-ci` ne peut PAS tester la prediction que je lui ai donnee: ni le
    # decalage de base ni le decalage applique aux sous-titres n'atteignaient le
    # journal. `pieces=` porte des bornes, `USED` porte le decalage PAR FLUX, et
    # la ligne de sous-titre ne portait aucun decalage du tout.
    #
    # LES DEUX SONT SUR LA MEME LIGNE PARCE QUE C'EST LEUR ECART QUI EST LA
    # MESURE: les pistes audio consomment `by_stream`, le chemin sous-titre
    # consomme la base, et personne ne pouvait voir les deux.
    for index, segment in enumerate(plan.get("segments") or []):
        by_stream = segment.get("candidate_offset_ms_by_stream")
        tools.logs.append(
            f"repair: segment {index} "
            f"master={segment.get('master_start_ms')}-{segment.get('master_end_ms')} "
            f"base_offset_ms={segment.get('candidate_offset_ms')}"
            # D'OU VIENT CE DECALAGE. Present seulement sur un segment FABRIQUE:
            # un plan qui porte ses propres tranches n'a pas cette question.
            f"{'(' + str(segment['offset_origin']) + ')' if segment.get('offset_origin') else ''} "
            f"by_stream={by_stream if by_stream else 'none'}\n")

    for index, change in enumerate(plan.get("change_points") or []):
        low = change.get("bracket_low_ms")
        high = change.get("bracket_high_ms")
        width = (Decimal(str(high)) - Decimal(str(low))
                 if low != None and high != None else None)
        tools.logs.append(
            f"repair: bracket {index} low_ms={low} high_ms={high} "
            f"width_ms={width if width != None else 'unreported'} "
            # LE DRAPEAU QUI N'AVAIT JAMAIS QUITTE SON AFFECTATION. `true` veut
            # dire que les deux longueurs de fenetre ont diverge et que la mesure
            # est retombee sur un intervalle d'une inter-fenetre entiere: la
            # position N'A PAS ETE RESSERREE, elle est BORNEE.
            f"bound_only={change.get('bracket_is_bound_only')} "
            # `clamped_to_next` n'apparait que s'il a eu lieu: le localisateur ne
            # pose la cle que dans ce cas, et un `false` partout serait du bruit.
            f"{'clamped_to_next=true ' if change.get('bracket_clamped_to_next') else ''}"
            f"step_ms={change.get('step_ms')} "
            f"step_points={change.get('step_points')}\n")

    dropped_note = plan.get("segments_dropped_unusable") if plan else None
    tools.logs.append(f"repair: plan {plan.get('kind') if plan else 'none'} "
                      f"{'language_route=' + str(plan['language_route']).replace(' ', '_') + ' ' if plan and plan.get('language_route') else ''}"
                      # TRUTHINESS: `0` ET `None` SE TAISAIENT TOUS LES DEUX.
                      #
                      # `if dropped_note` supprimait le champ pour "le localisateur
                      # a jete ZERO segment" ET pour "le localisateur ne l'a pas
                      # dit" -- deux faits, un silence. Meme forme que le
                      # `speed_margin` que `vmsam-dev-4` vient de me faire
                      # corriger, laissee sur le champ d'a cote: j'ai reparé la
                      # classe sur trois champs et garde le quatrieme.
                      #
                      # ON N'EMET PAS `0` PARTOUT: zero est le cas courant et
                      # trois cents lignes de `dropped_segments=0` cacheraient
                      # celle qui compte. On emet quand il y a QUELQUE CHOSE A
                      # DIRE -- un compte non nul, ou le fait que le localisateur
                      # n'a rien dit -- ce qui laisse a l'absence UNE seule
                      # signification: zero segment jete, rapporte.
                      f"{'dropped_segments=' + str(dropped_note) + ' ' if dropped_note else ''}"
                      f"{'dropped_segments=unreported(locator did not report it) ' if plan and 'segments_dropped_unusable' not in plan else ''}"
                      f"language={plan.get('language') if plan else None} "
                      # DE COMBIEN LA TRANSFORMATION DE RYTHME L'A EMPORTE.
                      # Absente quand la mesure n'en porte pas -- JAMAIS zero:
                      # une marge nulle serait deux hypotheses a egalite, qui est
                      # le cas `indeterminate` et non "pas de marge rapportee".
                      # LA MARGE, ET SON ABSENCE, ET LA RAISON DE SON ABSENCE.
                      #
                      # L'emetteur etait conditionne a la VERACITE de la valeur.
                      # Une marge INDEFINIE vaut None, donc rien du tout n'etait
                      # ecrit -- precisement sur les fichiers ou la barre de
                      # fidelite a decide et ou aucune marge de platitude
                      # n'existe. vmsam-dev-4 rendait alors `marge de victoire:
                      # NON EMISE` par-dessus une decision prise avec 0.3637 de
                      # separation.
                      #
                      # C'est `head_pad_ms=0` a nouveau, en pire: la ou ce champ
                      # confondait trois etats sous un chiffre, celui-ci
                      # confondait `pas de marge` et `producteur muet` sous une
                      # LIGNE NON ECRITE. Un emetteur conditionne a la veracite
                      # de ce qu'il emet ne peut jamais dire `absent`.
                      f"{_margin_fields(plan)}"
                      f"quantum={quantum_ms}"
                      # LA FENETRE A COTE DU QUANTUM, PARCE QU'UN QUANTUM SANS SA
                      # FENETRE N'EST PAS COMPARABLE.
                      #
                      # `vmsam-ci`: `quantum=129` est invariant sur les seize
                      # plans du disque, et le quantum du PIPELINE pour les memes
                      # fichiers vaut 124 ou 125. Ce n'est pas un defaut -- celui
                      # du localisateur vient de ses sondes fixes de 60 s, celui
                      # du pipeline de `int(lengthFile/n_items*1000)` sur le
                      # fichier entier. DEUX FENETRES, DEUX QUANTA.
                      #
                      # Mais `quantum=` est le SEUL quantum que ce journal
                      # publie, dans une ligne qu'un lecteur comparera a des
                      # chiffres du pipeline. Et l'avertissement est dans la
                      # docstring de dev-1: UN MEME PAS PHYSIQUE A MESURE 500,
                      # 540 ET 600 ms A TROIS LONGUEURS DE FENETRE. Le quantum
                      # n'est interpretable qu'avec la fenetre qui l'a produit.
                      f"{'@window_s=' + str(plan['probe_window_seconds']) if plan and plan.get('probe_window_seconds') != None else ''} "
                      f"pieces={' '.join(spans)}\n")

    verification = {}
    for entry in assembly.get("verification") or []:
        verification[entry.get("track")] = entry

    verified_count = sum(1 for v in verification.values()
                         if v.get("outcome") not in (None, "skipped"))
    for report in assembly.get("audios") or []:
        checked = verification.get(report["stream_order"], {})
        worst = checked.get("worst_lag_ms")
        line = (f"repair: audio track {report['stream_order']} "
                f"lang={report['language']} "
                f"fill={report['gap_fill']}"
                f"{'/' + str(report['fill_language']) if report.get('fill_language') else ''}"
                f"{'[' + str(report['fill_title']) + ']' if report.get('fill_title') else ''}"
                # AMBIGU = le maitre portait PLUSIEURS pistes principales dans
                # cette langue et l'etiquette ne les separe pas. Cout mesure du
                # mauvais choix: 21.3 ms, sous la tolerance, silencieux.
                # AMBIGUOUS seulement quand le choix N'A PAS ete tranche par
                # la mesure. Quand il l'a ete, on le dit aussi -- `among N by
                # measurement` -- pour qu'on voie qu'il y avait un choix ET
                # qu'il etait fonde.
                f"{('(among ' + str(report['fill_choices']) + ' by measurement)' if report.get('fill_by_reference') else '(AMBIGUOUS among ' + str(report['fill_choices']) + ')') if (report.get('fill_choices') or 0) > 1 else ''}"
                # LA SOURCE DE REMPLISSAGE EST-ELLE TROP COURTE POUR LES TROUS
                # QU'ON LUI DEMANDE? Mesure sur un artefact reel: la piste fr du
                # maitre 2008 ms plus courte que sa ja, le manque HERITE par la
                # sortie, quatre fois la tolerance, et rien ne les comparait.
                # CE QUE LA SOURCE EXPLIQUE, ET CE QUE LA PISTE A REELLEMENT
                # PERDU. Le validateur a mesure un fichier ou le maitre etait
                # court de 907 ms et la piste produite courte de 1988: MON
                # ANNOTATION AURAIT DIT 907 ET SOUS-DECLARE DE MOITIE. Un lecteur
                # a qui l'on donne 907 croit le manque explique.
                #
                # LE RESIDU EST LE DEFAUT; la part expliquee est celle qui n'en
                # est pas un. On emet donc les deux et leur difference.
                f"{_shortfall_annotation(assembly, report)} "
                f"filled_ms={report['gap_filled_ms']} "
                f"silence_ms={report['silence_filled_ms']} "
                f"head_pad_ms={report['head_pad_ms']} "
                # POURQUOI CE NOMBRE, ET SURTOUT POURQUOI ZERO. `head_pad_ms=0`
                # couvrait trois situations -- non mesure, decalage lu au-dela,
                # et pas de decalage -- avec le meme chiffre. On compte les
                # decisions par issue plutot que d'en imprimer une par morceau:
                # la ligne de piste est deja longue, et ce qu'un lecteur doit
                # pouvoir dire est "de quoi ce zero est-il fait".
                f"head_pad={_head_pad_summary(report)} "
                # LE REMPLISSAGE INTERLINGUE S'ANNONCE AU LIEU D'ETRE DEDUIT.
                #
                # `fill=master/ja` sur une piste `en` etait DIVULGUE et pas
                # SIGNALE: un lecteur devait comparer deux champs pour voir que
                # la langue de remplissage n'est pas celle de la piste. Le
                # proprietaire a entendu la consequence sur un fichier reel --
                # l'anglais s'arrete a 21:20 et le japonais prend la suite -- et
                # c'est s4c FONCTIONNANT COMME IL A ETE ARBITRE: le maitre ne
                # porte que `ja` et `fr`, il n'y a pas d'anglais avec quoi
                # remplir. Signale, ce n'est plus une ligne a decoder.
                #
                # Emis seulement quand les deux langues DIFFERENT: `false` sur
                # chaque piste serait du bruit, la divergence est l'evenement.
                + ("cross_language_fill=true "
                   if (report.get("fill_language")
                       and report.get("fill_language") != report.get("language"))
                   else "")
                # LE DESACCORD ENTRE OUTILS SUR LA LIGNE, ET SEULEMENT QUAND IL
                # Y EN A UN. `tools=agree` sur chaque piste serait du bruit sur
                # 1 175 lignes pour en signaler 3; l'ABSENCE du champ est le cas
                # normal et sa PRESENCE est l'evenement. Mesure de vmsam-dev-3:
                # mediainfo 44100 contre ffprobe 48000 une fois, mediainfo mono
                # contre ffprobe stereo deux fois, sur 1 178 pistes.
                + (f"tool_split={';'.join(report['tool_disagreements'])} "
                   if report.get("tool_disagreements") else "")
                + 
                # SPEC_ZONE_A s4g: QUELLE BRANCHE A SERVI LA TETE.
                #   master/<lang>  la piste de cette langue porte la tete, mesuree
                #   NO-HEAD        elle NE la porte pas -- et le repli n'est PAS
                #                  encore implemente, donc la tete vient QUAND MEME
                #                  de cette piste et elle est muette
                #   unprobed       tete illisible: pas une tete absente
                #   silence        aucun remplissage maitre pour ce fichier
                f"{'head=' + str(report['head_source']) + ' ' if report.get('head_source') else ''}"
                # `speed=none` DISAIT DEUX CHOSES: "la mesure n'a propose aucun
                # changement de rythme" et "il n'y a pas de probleme de rythme".
                # Ce sont mesure-zero contre n'ai-pas-pu-mesurer, sur le rythme.
                #
                # Cela compte parce qu'un `r_min` faible a deux causes: LE MAUVAIS
                # PROGRAMME, ou LE BON PROGRAMME A UN RYTHME NON COMPENSE. Quatre
                # fichiers du corpus sont a 4.27 % lents (PAL) et SONT le bon
                # programme -- forensic l'a etabli contre un controle negatif a
                # 0.0052. Sans ce champ, leurs lignes seraient indistinguables
                # d'un vrai desappariement.
                f"speed={report.get('speed_ratio_applied') if report.get('speed_ratio_applied') != None else 'none(no rate proposed by the measurement)'} "
                # BORROWED = cette piste porte le decalage d'une AUTRE langue.
                # BORROWED PORTE SA RAISON. Le proprietaire a tranche que
                # l'emprunt continue, donc cette ligne est ce qui est livre avec
                # le fichier, et "BORROWED" seul ne dit pas si un partenaire
                # existait.
                f"offset={'measured' if report.get('offset_measured') else 'BORROWED'}"
                f"{'[' + str(report['borrow_reason']) + ']' if report.get('borrow_reason') else ''}"
                # `fid` absent = la mesure n'en donne pas. JAMAIS 0.0: une
                # fidelite inconnue n'est pas une fidelite nulle.
                f"{'(fid ' + str(report['offset_fidelity']) + ')' if report.get('offset_fidelity') != None else ''} "
                # UN `skipped` NU FAIT RECONSTRUIRE SA CAUSE. Deux agents l'ont
                # deduite de `fill=` sur la meme ligne et l'ont eu juste, CE QUI
                # N'EST PAS UNE PREUVE QUE LE PROCHAIN LECTEUR Y ARRIVERA.
                #
                # Et la cause compte plus qu'un detail de forme: le meme predicat
                # -- LE MAITRE PORTE-T-IL CETTE LANGUE? -- decide A LA FOIS que
                # le remplissage tombe sur la langue de comparaison ET que la
                # verification n'a aucune reference. `skipped_iff_foreign` n'est
                # donc pas une correlation observee, c'est une IDENTITE: LA
                # VERIFICATION EST INDISPONIBLE EXACTEMENT LA OU LE RISQUE SE
                # CONCENTRE, et elle ne peut pas en etre autrement.
                f"verify={checked.get('outcome')}"
                f"{'(' + str(checked['reason']) + ')' if checked.get('outcome') == 'skipped' and checked.get('reason') else ''} "
                # LES UNITES VOYAGENT AVEC LES NOMBRES. L'ancienne forme
                # `residual=(4,0.08,129)` mettait un COMPTE, un RAPPORT et une
                # DUREE dans une seule parenthese sans nom ni unite, et le
                # premier champ qu'un lecteur rencontre est un entier qui
                # ressemble a des millisecondes. Un agent l'a lu comme un
                # decalage de 4.0 ms -- c'etait QUATRE SONDES -- et allait le
                # rapporter comme un desaccord entre le conteneur et le
                # laboratoire. Le format etait le defaut, pas la lecture.
                f"residual=probes={checked.get('probes_measured')} "
                f"worst={quanta(worst, quantum_ms)}q "
                f"quantum={quantum_ms}ms "
                # La COUVERTURE voyage avec le pire ecart: "worst 9.88" ne porte
                # aucune trace de "sur 2 pistes verifiees parmi 7", et se cite
                # donc comme s'il decrivait le fichier.
                # LA CORRELATION LA PLUS FAIBLE, a cote du verdict. "aligned"
                # ne distingue pas une piste calee sur le bon programme d'une
                # piste calee sur du contenu sans rapport; r le fait.
                f"{'r_min=' + str(checked['weakest_correlation']) + ' ' if checked.get('weakest_correlation') != None else ''}"
                # LA BORNE DE SELECTION, A COTE DU COMPTE QU'ELLE CONTAMINE.
                # `probes=` est un compte sur des sondes CHOISIES: une fenetre
                # sous `verify_min_rms` est ecartee. Le predicat d'appartenance
                # mentionne donc une quantite du signal. Un rapport proche de 1
                # dit que les sondes gardees frolaient le seuil et que le compte
                # est fortement censure; un rapport tres grand dirait que le
                # seuil n'est jamais contraignant, ce qui serait une decouverte
                # et pas un repli. Regle de vmsam-dev-3, tiree du fait qu'il a
                # tue sa propre borne pour cette raison exacte.
                f"{'rms_over_floor=' + str(checked['rms_over_floor']) + 'x ' if checked.get('rms_over_floor') != None else ''}"
                # LA CARTE VERS LE FICHIER PRODUIT. `stream_order` est l'index
                # dans le CANDIDAT; l'index audio de la sortie est un compteur de
                # boucle du verificateur que rien ne renvoyait. Sans lui, un
                # consommateur qui veut comparer une piste du journal a un flux
                # de l'artefact doit DEDUIRE l'ordre depuis la position -- le
                # defaut qui a lu la colonne 2 comme un statut, et celui qui
                # aurait fausse la jointure USED/CUT si dev-4 avait apparie par
                # index plutot que par nom.
                f"produced_index={checked.get('produced_index') if checked.get('produced_index') != None else 'unknown'} "
                f"verified={verified_count}/{len(assembly.get('audios') or [])}\n")
        tools.logs.append(line)
        # SPEC_ZONE_A s4e, UNE LIGNE PAR REGION: ce qui a ete AJOUTE, ou, et
        # d'ou -- avec LA LANGUE REELLEMENT UTILISEE et non celle demandee.
        # La ligne de piste ci-dessus porte des TOTAUX, et un total ne dit pas
        # quelle region a recu de l'audio maitre et laquelle du silence.
        #
        # Prefixe ADDED, distinct de `repair: audio track`, pour qu'un grep qui
        # compte les pistes construites n'y compte pas les regions.
        # CE QUE LA SORTIE PREND AU CANDIDAT. La majorite de chaque fichier
        # n'avait aucune ligne de provenance: seuls le remplissage (ADDED) et le
        # rejet (CUT) en avaient une. Prefixe DISTINCT de `audio track` pour la
        # meme raison qu'ADDED -- un grep qui compte les pistes reconstruites ne
        # doit pas compter les regions.
        #
        # `offset_ms` par region rend le decalage INCONDITIONNEL: il ne depend
        # plus de l'existence d'une coupe, et une piste piecewise_constant montre
        # ses decalages successifs au lieu du seul mot `measured` sur la ligne de
        # piste, qui les ecrase.
        for region in report.get("used_regions") or []:
            tools.logs.append(
                f"repair: USED audio track {report['stream_order']} "
                f"master {region['master_start_ms']}-{region['master_end_ms']} "
                f"candidate {region['candidate_start_ms']}-{region['candidate_end_ms']} "
                f"offset_ms={region['offset_ms']}\n")
        for region in report.get("filled_regions") or []:
            tools.logs.append(
                f"repair: ADDED audio track {report['stream_order']} "
                f"master {region['master_start_ms']}-{region['master_end_ms']} "
                # POURQUOI, ET PAS SEULEMENT OU. `head_gap`, `interior_bracket`
                # et `tail_gap` sont trois causes differentes qui produisaient une
                # ligne identique -- et la regle du proprietaire porte exactement
                # sur cette distinction: retirer l'exces de TETE et de QUEUE,
                # garder ce qui tombe dans la portee du maitre. Sans le motif, une
                # SUBSTITUTION et une incertitude de localisateur se lisent pareil.
                # LE REPLI NE NOMME PLUS DE CAUSE, PARCE QU'IL EN A NOMME UNE
                # FAUSSE 48 FOIS SUR 48.
                #
                # Il disait `unreported(assembly predates the field)`. La cause
                # reelle etait que le motif ne traversait pas jusqu'a l'emetteur.
                # Ma propre phrase, ecrite une heure avant que je m'y reprenne:
                # UN REPLI QUI NOMME UNE CAUSE EST UNE AFFIRMATION -- et la forme
                # aigue est de `vmsam-dev-4`: un blanc envoie un enqueteur
                # CHERCHER; une cause nommee l'envoie chercher A UN SEUL ENDROIT,
                # LE MAUVAIS. Ici: les dates de deploiement, qui auraient eu
                # l'air correctes, et il aurait conclu que le champ marchait.
                #
                # Le nouveau repli dit ce qui est OBSERVABLE -- la region ne porte
                # pas de motif -- et REFUSE EXPLICITEMENT la question causale, que
                # je ne peux pas trancher depuis ici: un assemblage anterieur au
                # champ et une region qu'un chemin nouveau n'a pas annotee
                # produisent la meme absence.
                f"why={region.get('reason') or 'absent(region carries no reason; cause of the absence NOT established)'} "
                f"from={region['source']}"
                f"{'/' + str(region['language']) if region.get('language') else ''}"
                # QUEL FLUX, pas seulement quelle langue. Un maitre peut porter
                # quatre pistes `spa` dont deux au meme titre; sans le
                # StreamOrder un consommateur doit DEVINER contre quoi comparer,
                # exactement la ou le commentaire de `find_fill_audio` dit que
                # deviner est faux. `unknown` et jamais un defaut silencieux.
                f" stream={report.get('fill_stream_order') if report.get('fill_stream_order') != None else 'unknown'}"
                # L'INVARIANT, EMIS PLUTOT QUE SUPPOSE. `normalize_segments`
                # pose `source_start_ms = cursor` sur les deux branches de
                # morceau maitre, donc une region remplie [a,b] prend l'audio
                # maitre [a,b] SANS decalage. vmsam-dev-3 l'a verifie dans la
                # source et demande quand meme le champ, pour la bonne raison:
                # SANS LUI, UN INVARIANT CASSE ET UNE ERREUR DE PLACEMENT SONT LA
                # MEME OBSERVATION, et il classerait le premier comme le second,
                # contre mon assembleur.
                # ET IL EST CALCULE, PAS ECRIT. Il etait le LITTERAL `0`.
                #
                # Un invariant affirme par une constante ne peut pas detecter sa
                # propre violation: si `normalize_segments` posait un jour un
                # morceau maitre a un decalage, CETTE LIGNE IMPRIMERAIT ENCORE 0.
                # C'est la forme `check(..., True)` dans un champ emis -- une
                # affirmation qui ne peut pas echouer -- dans le champ demande
                # precisement pour que l'invariant casse SE VOIE.
                #
                # `vmsam-ci` l'a mesure du cote lecteur: sur ses artefacts, 159
                # de ces zeros structurels contre 8 vraies mesures portant le
                # meme nom -- donc une moyenne sur cette colonne serait a 95 %
                # composee d'un jeton qui existe pour que son ABSENCE ne soit pas
                # mal lue. LA DEFENSE CONTRE `absent n'est pas zero` DEVIENT LE
                # VECTEUR D'UN FAUX ZERO des que quelqu'un agrege.
                f" offset_ms={master_fill_offset(region)}\n")
        # ET CE QUI A ETE COUPE: du materiau du candidat qui existe et
        # n'apparait pas dans la sortie. Sans ces bornes la coupe n'est visible
        # nulle part -- ni dans le plan, qui donne la timeline du MAITRE, ni
        # dans les totaux.
        for region in report.get("cut_regions") or []:
            # `where` distingue tete, interieur et queue: une coupe de tete et
            # une coupe de queue ne se diagnostiquent pas comme un saut entre
            # deux morceaux. Et une queue NON MESUREE se dit, au lieu de ne
            # produire aucune ligne -- l'absence de ligne se lirait "rien n'a
            # ete coupe".
            if region.get("unmeasured"):
                tools.logs.append(
                    f"repair: CUT audio track {report['stream_order']} "
                    f"candidate {region['candidate_start_ms']}-? "
                    f"where={region.get('where')} dropped_ms=UNMEASURED "
                    f"(the candidate duration was not available)\n")
                continue
            tools.logs.append(
                f"repair: CUT audio track {report['stream_order']} "
                f"candidate {region['candidate_start_ms']}-"
                f"{region['candidate_end_ms']} dropped_ms={region['dropped_ms']} "
                f"where={region.get('where')}\n")

    for report in assembly.get("subtitles") or []:
        tools.logs.append(f"repair: subtitle track {report['stream_order']} "
                          f"lang={report['language']} format={report.get('format')} "
                          f"kept_cues={report.get('kept_cues')} "
                          # LE DECALAGE APPLIQUE, ET COMBIEN DE MORCEAUX
                          # DISTINCTS L'ONT FOURNI. Une seule entree veut dire
                          # que toutes les repliques partagent un morceau -- ce
                          # qui rend une constante ATTENDUE et non suspecte, et
                          # c'est exactement la lecture qui manquait a ci sur
                          # id 47. La ligne portait des comptes de repliques et
                          # aucun decalage.
                          f"shifts_ms={report.get('shifts_applied_ms') or 'none'} "
                          f"dropped_cues={report.get('dropped_cues')}\n")

    # UN SAUT EST UNE DECISION.
    # LE PREFIXE DISTINGUE UN SAUT D'UNE PISTE CONSTRUITE. Ecrites comme
    # "repair: audio track N ...", les deux se comptent ensemble: un lecteur ou
    # un grep qui compte les pistes construites compterait aussi les sautees.
    # C'est la meme forme que le compte de pistes "reconstruites" qui a decrit
    # un fichier tronque -- une phrase vraie dont une moitie dit autre chose que
    # ce qu'on en lit. Trouve en ecrivant le controle, pas apres.
    # SPEC_ZONE_A s4e: UN ELEMENT ECARTE EST UNE DECISION ET SE DIT, avec ce
    # qui a ete ecarte ET POURQUOI CELA COMPTE. Ces segments etaient comptes en
    # millisecondes dans le compte-rendu et n'apparaissaient sur AUCUNE ligne.
    #
    # Pourquoi cela compte: la region devient un remplissage DEPUIS LE MAITRE, et
    # sur la ligne ADDED elle est indistinguable d'un trou ordinaire du plan. Le
    # lecteur ne pouvait pas separer "le plan n'avait pas de candidat ici" de
    # "le plan en avait un et on l'a jete parce que son decalage etait invalide".
    for entry in assembly.get("dropped_segments") or []:
        tools.logs.append(
            f"repair: SKIPPED segment master {entry['master_start_ms']}-"
            f"{entry['master_end_ms']} dropped_ms={entry['dropped_ms']} "
            f"DECLINED: offset unverified (segment shorter than the "
            f"measurement's probe window); this span is filled from the master "
            f"instead of the candidate\n")

    for entry in assembly.get("declined") or []:
        tools.logs.append(f"repair: SKIPPED {entry.get('kind')} track "
                          f"{entry.get('stream_order')} DECLINED: "
                          f"{entry.get('reason')}\n")
    for entry in assembly.get("failed") or []:
        tools.logs.append(f"repair: SKIPPED {entry.get('kind')} track "
                          f"{entry.get('stream_order')} FAILED: "
                          f"{entry.get('reason')}\n")

    check = assembly.get("output_check")
    if check:
        tools.logs.append(f"repair: output file audio {check['audio_in_file']}/"
                          f"{check['audio_built']} subtitles "
                          f"{check['subtitles_in_file']}/{check['subtitles_built']} "
                          f"expected_ms={check['expected_duration_ms']} "
                          f"source={check['expected_duration_source']} "
                          # LA CADENCE DU MAITRE. Elle n'est derivable d'aucune
                          # autre ligne, et sans elle personne ne peut calculer
                          # une exclusion de vitesse ni dire sur quelle grille
                          # `adjust_delay_to_frame` a colle. `unread` et pas
                          # zero quand mediainfo ne la donne pas.
                          f"frame_rate={assembly.get('master_frame_rate') or 'unread'}"
                          f"({assembly.get('master_frame_rate_mode') or 'mode unread'}"
                          f"{',used' if assembly.get('master_frame_rate_original') else ''}) "
                          # LE SECOND CHAMP N'APPARAIT QUE S'IL DIFFERE. Deux
                          # champs identiques sur chaque ligne seraient du bruit;
                          # leur DESACCORD est l'information, et il est rare.
                          f"{'frame_rate_original=' + str(assembly['master_frame_rate_original']) + ' ' if assembly.get('master_frame_rate_original') else ''}"
                          f"tolerance_ms={check['tolerance_ms']} "
                          f"measured={check.get('measured')} "
                          f"would_refuse={check.get('would_refuse')} "
            # LE RESUME NE CONTREDIT PLUS LE DETAIL. `enforcing=False` etait une
            # decision sur REFUSER OU NON -- jamais une decision de calculer le
            # resume comme si le controle n'avait pas tire. Un validateur a lu un
            # resume disant que tout allait bien a cote d'une ligne de detail qui
            # disait le contraire.
            #
            # Gratuit, aucun changement de comportement, et cela ferme la forme
            # s4d au seul endroit disponible tant que le gate est desarme.
            f"{'-- 1 WOULD HAVE BEEN DECLINED (gate inert) ' if check.get('would_refuse') and not check.get('enforcing') else ''}"
                          f"enforcing={check.get('enforcing')}\n")
        # LA CAUSE, SUR SA PROPRE LIGNE, ET C'EST LA CORRECTION LA PLUS CHERE DE
        # LA SOIREE PARCE QUE LE PROPRIETAIRE L'A TROUVEE EN ECOUTANT.
        #
        # `would_refuse=True` etait publie SANS SA RAISON. Le controle avait
        # calcule `problems` -- LEQUEL des quatre a tire, et de COMBIEN chaque
        # piste est courte -- et la ligne n'en emettait rien. Les comptes
        # concordaient (7/7, 19/19), donc un lecteur devait faire l'elimination
        # de tete pour arriver a "court ou non mesure", et seulement parce que le
        # champ qui le nommait avait ete calcule puis jete.
        #
        # UN VERDICT DONT LA CAUSE N'EST PAS A COTE NE PEUT PAS ETRE ACTIONNE; il
        # peut seulement etre cru ou ignore. C'est la classe de cette campagne --
        # la garde annoncee comme un succes avant d'avoir tire -- arrivee dans la
        # SEULE ligne qui porte un verdict.
        #
        # UNE LIGNE A PART PLUTOT QUE TRONQUEE. `problems` peut etre long; une
        # raison tronquee est pire que pas de raison, et le lecteur de
        # `vmsam-dev-4` resout PAR NOM, donc une ligne de plus ne coute rien.
        # Emise seulement quand il y a quelque chose a dire: son ABSENCE dit
        # "aucun probleme", ce qui est un fait et non un champ manquant.
        for problem in (check.get("problems") or []):
            tools.logs.append(f"repair: output problem {problem}\n")
        # LES DEUX DUREES QUE J'AI CONSTRUITES POUR CE LECTEUR-CI ET QUE JE NE LUI
        # DONNAIS PAS. Mon propre commentaire, deux fonctions plus haut: "un
        # lecteur qui voit `container_duration_ms 3600000` a cote de
        # `expected_duration_ms 1420002` n'a AUCUN CHAMP qui dise que l'ecart est
        # une etiquette de sous-titre plutot qu'un defaut du travail." Les deux
        # champs existent depuis des heures et n'atteignaient pas la ligne.
        if check.get("container_duration_ms") != None or check.get("max_av_stream_duration_ms") != None:
            tools.logs.append(
                f"repair: output durations container_ms={check.get('container_duration_ms')} "
                f"max_av_stream_ms={check.get('max_av_stream_duration_ms')} "
                f"expected_ms={check.get('expected_duration_ms')}\n")


def decline_detail(error):
    """Ce qu'un DECLIN emporte, extrait pour etre testable sans rejouer un fichier.

    Une fonction et non un dictionnaire en ligne parce que le contenu d'un declin
    est devenu une reponse a une question posee par d'autres agents -- le
    registre de `vmsam-dev-4` et les comptes de `vmsam-ci` le lisent -- et un
    dictionnaire construit en ligne dans une branche `except` ne se verifie qu'en
    faisant lever un vrai fichier.

    `output_check` EST ICI PARCE QUE LE DRAPEAU EST LEVE. Tant que le controle
    de duree etait inerte, ce rapport n'apparaissait que sur des artefacts
    PRODUITS; il est maintenant la RAISON d'un declin, et sans lui le declin dit
    "le fichier produit ne correspond pas a ce qui a ete construit" sans jamais
    dire QUELLE piste ni de combien.

    `undelivered_path` EST LE CHEMIN DE L'ARTEFACT RENOMME, ET `undelivered_state`
    DIT LEQUEL DES DEUX ETATS NON LIVRES IL PORTE -- `REFUSED` (la porte a decide
    contre) ou `NOVERDICT` (personne n'a decide; une panne d'outil s'est echappee
    avant qu'un verdict existe). UN SEUL NOM POUR LES DEUX ferait absorber en
    silence chaque panne d'ffprobe dans le cout de la porte.

    Le champ s'appelait `refused_path` pendant une heure, avant que le second
    etat existe. `vmsam-dev-4` et `vmsam-ci` en ont ete prevenus avant que quoi
    que ce soit soit expedie: rien de tout ceci n'est encore dans l'image.

    `None` a DEUX causes distinctes -- le declin est arrive avant que le fichier
    existe, ou le renommage a echoue -- et la seconde est ecrite sur stderr par
    `mark_output` plutot que devinee ici.
    """
    return {"verification": getattr(error, "verification", None),
            "audios": getattr(error, "audios", None),
            "output_check": getattr(error, "output_check", None),
            "undelivered_state": getattr(error, "undelivered_state", None),
            "undelivered_path": getattr(error, "undelivered_path", None),
            "undelivered_durable": getattr(error, "undelivered_durable", None)}


def record(candidate_path, outcome, reason, detail=None):
    entry = {"candidate": candidate_path, "outcome": outcome, "reason": reason,
             "detail": detail}
    last_repair_report.append(entry)
    # UNCONDITIONAL, AND IT USED TO BE UNCONDITIONAL ONLY BY COINCIDENCE.
    # Found by `vmsam-dev-1`, verified here before changing anything.
    #
    # The guard was `if tools.dev or outcome in ("repaired", "failed",
    # "declined", "no_plan")`. MEASURED, by AST over every call site:
    #
    #     outcomes actually passed   declined x6 . no_plan x2 . failed x1
    #                                . repaired x1
    #     the whitelist              the SAME FOUR STRINGS
    #     outcomes not whitelisted   NONE
    #
    # So the `tools.dev or` left operand COULD NEVER CHANGE THE RESULT: a dead
    # clause that made a reader think the line was debug-gated when it was not.
    # And the line fired in production only because the used-outcome set HAPPENED
    # to equal the whitelist -- ONE NEW OUTCOME STRING would have silently
    # re-gated it to `tools.dev` only.
    #
    # THAT IS F26 EXACTLY: A REFUSAL THAT EMITS NOTHING IN PRODUCTION BECAUSE ITS
    # EMISSION IS DEV-GATED -- reintroduced, one word away, inside the module
    # that exists to consume F26's victims.
    #
    # Removing the condition is BEHAVIOUR-IDENTICAL TODAY (all ten outcomes were
    # whitelisted, so the guard was always true) and makes the property TRUE BY
    # CONSTRUCTION instead of true by coincidence. A `record()` call is by
    # definition an outcome worth recording; there is no outcome this function
    # should swallow.
    tools.logs.append(f"repair: {outcome} for {candidate_path}: {reason}\n")
    return entry


def repair_not_compatible_videos(list_not_compatible_video, dict_file_path_obj,
                                 best_video):
    '''Point d'entree appele depuis la zone A.

    Renvoie la liste des chemins effectivement repares et raccroches. Les
    fichiers restent retires de `dict_file_path_obj` par la zone A dans tous les
    cas: l'objet repare rejoint le merge par
    `best_video.sameAudioMD5UseForCalculation`, consomme par
    `generate_merge_command_common_md5`, qui ne passe jamais par la machinerie
    de delai.
    '''
    del last_repair_report[:]
    work_root = path.join(tools.tmpFolder, "repair")
    tools.make_dirs(work_root)
    repaired = []

    for candidate_path in list_not_compatible_video:
        candidate_obj = dict_file_path_obj.get(candidate_path)
        if candidate_obj == None:
            record(candidate_path, "declined",
                   "the rejected path has no video object in dict_file_path_obj")
            continue
        language, language_route = get_delay_language(best_video, candidate_obj)
        if language == None:
            # LA RAISON VOYAGE AVEC LE REFUS. `language_route` etait calcule,
            # rendu, DEPAQUETE ET JAMAIS UTILISE -- une occurrence dans tout le
            # fichier. Trouve par `vmsam-auditor`.
            # FORME `cause=<jeton>: <prose>` -- convenue avec dev-4 (EMISSION)
            # AVANT ecriture, des deux cotes. dev-4 decoupe sur `cause=`, classe
            # le JETON et rend la prose VERBATIM sans la relire dans sa voix.
            # Ce site nommait deja sa cause; seule la FORME change, pour que le
            # lecteur de dev-4 ne voie pas une population MIXTE ou un jeton reel
            # et une constante survivante sont indiscernables.
            record(candidate_path, "no_plan",
                   f"cause=language_undetermined: could not tell which language "
                   f"the merge measured on ({language_route})")
            continue
        plan, plan_refusal_cause = get_plan_from_locator(
            best_video, candidate_obj, language)
        plan_source = "change_point_locator"
        if plan != None:
            # COMMENT LA LANGUE A ETE CHOISIE, JUSQU'AU JOURNAL.
            #
            # `get_delay_language` peut rendre `"ARBITRARY: insertion order
            # among [...]"` -- LE MODULE DECLARE QUE SON DEPARTAGE EST UN TIRAGE
            # AU SORT -- et cette declaration n'atteignait aucun artefact. Un
            # verdict de reparation ne pouvait donc pas etre recalcule depuis sa
            # propre ligne de journal.
            #
            # C'est la meme classe que mon ecart sous-titre/audio de 119.55 ms:
            # une valeur que le code connait et que le compte rendu ne porte pas.
            # Et elle touche la question OWNER-PENDING -- quelle piste audio un
            # sous-titre suit quand sa langue n'a pas d'audio -- parce qu'avec ce
            # champ la question aurait des preuves sur CHAQUE fichier livre au
            # lieu d'un cas mesure.
            #
            # Attache au plan plutot que passe en parametre: `log_assembly` recoit
            # le plan et pas cette variable, et une annotation locale sur un dict
            # que je viens de recevoir coute moins qu'une signature de plus.
            plan["language_route"] = language_route
        if plan == None:
            # None de la mesure = "je n'ai pas pu mesurer", et surtout pas
            # "les fichiers vont ensemble". Le refus reste, intact.
            # LA CAUSE, PAS UNE CONSTANTE. Voir ACCEPTANCE_T12A: la chaine
            # precedente affirmait qu'AUCUNE mesure n'existait, ce qui est FAUX
            # sur le chemin du plancher de fidelite -- les sondes ont TOURNE et
            # ont rendu un NEGATIF CONCLUANT.
            record(candidate_path, "no_plan",
                   f"cause={plan_refusal_cause}: no plan from {plan_source}")
            continue
        if plan.get("kind") == "constant":
            # Troisieme issue de la mesure, et elle n'est pas la notre. Un
            # decalage constant se corrige par un delai de conteneur; le
            # reconstruire couterait une generation de codec par piste et
            # perdrait les sous-titres bitmap qu'un simple decalage garde
            # (docs/SUBTITLE_CODECS.MD). C'est aussi la sixieme population que
            # vmsam-forensic a mesuree, dont 11 fichiers sont refuses pour un
            # defaut du MAITRE et pas du candidat.
            # On DECLINE, et on ne recommande rien. La formulation precedente
            # disait "cette paire a besoin d'un delai de conteneur", ce qui est
            # un conseil -- et vmsam-dev-1 a mesure le 2026-09-03 qu'il peut etre
            # faux: son balayage a dix fenetres ne voit structurellement pas les
            # ~227 premieres secondes d'un episode, donc "constant" veut dire
            # "aucun pas visible" et non "le decalage est constant". Sur
            # l'erreur 108 un delai de conteneur serait faux de 500 ms pendant
            # les 146 premieres secondes et juste ensuite: un fichier plausible,
            # silencieusement faux en tete. Un message de refus est ce sur quoi
            # un lecteur agit; il ne doit pas porter une recommandation que la
            # mesure ne soutient pas.
            record(candidate_path, "declined",
                   f"the measurement reports no change point, so there is nothing "
                   f"to splice and a rebuild would cost a codec generation and "
                   f"drop bitmap subtitles. NOT a warrant for a container delay: "
                   f"'constant' means no step was VISIBLE, and the measurement is "
                   f"blind to the head of the file ({plan_source})",
                   {"plan_kind": "constant", "plan": plan})
            continue
        speed_ratio, speed_refusal = get_speed_ratio(plan)
        if speed_refusal != None:
            record(candidate_path, "declined", f"{speed_refusal} ({plan_source})",
                   {"plan_kind": plan.get("kind"), "verdict": plan.get("verdict")})
            continue
        if speed_ratio == None and not len(plan.get("segments") or []):
            record(candidate_path, "declined",
                   f"the measurement returned neither a segment nor a speed "
                   f"relation ({plan_source})")
            continue
        master_check = compare_plan_master(plan, best_video)
        if master_check != None:
            # LA RAISON NE PORTE PAS LE CHEMIN. `record` ecrit deja le fichier
            # sur sa propre ligne; une RAISON, elle, se cite -- dans un rapport,
            # dans un message a un autre agent, dans un resume -- et une raison
            # qui contient un chemin media voyage avec lui. On redige avant que
            # l'extrait ne parte, pas apres.
            record(candidate_path, "declined", master_check)
            continue
        try:
            repaired_obj, assembly = build_repaired_video_object(
                candidate_obj, best_video, plan, work_root)
        except Exception as error:
            # Un refus de l'assemblage est un DECLIN, pas une panne: le module a
            # tourne, a regarde le plan ou le fichier produit, et a dit non. Les
            # confondre reduirait les cinq issues a quatre, et l'issue perdue
            # serait justement celle qui porte une raison.
            import merge_video_chimeric
            if isinstance(error, merge_video_chimeric.chimeric_error):
                # Le declin porte ses sondes quand il en a: c'est ce qui permet
                # a la mesure de diagnostiquer son propre plan sans rejouer le
                # fichier.
                # `output_check` VOYAGE AVEC LE DECLIN, ET C'EST LE DRAPEAU
                # LEVE QUI REND CETTE LIGNE NECESSAIRE. Tant que le controle
                # etait inerte, ce rapport n'apparaissait QUE sur des artefacts
                # PRODUITS; maintenant il est la RAISON d'un declin, et sans lui
                # le declin dit "le fichier produit ne correspond pas" sans
                # jamais dire QUELLE piste ni de combien. La levee le porte
                # (`error.output_check = report`) et le pilote le jetait.
                record(candidate_path, "declined", str(error),
                       decline_detail(error))
                sys.stderr.write(f"repair: declined {candidate_path}: {error}\n")
            else:
                # LE MEME DETAIL SUR `failed` QUE SUR `declined`, ET C'EST CE
                # CHEMIN-CI QUI PRODUIT `NOVERDICT`: une panne d'outil apres le
                # mux laisse un artefact renomme, et le seul enregistrement qui
                # peut le nommer est celui-ci. L'omettre remettrait le fichier
                # hors de tout compte rendu, ce que le renommage existe pour
                # empecher.
                record(candidate_path, "failed", str(error), decline_detail(error))
                sys.stderr.write(f"repair: failed for {candidate_path}: {error}\n")
            continue

        # ORDRE, ET NON GARDE. Tout ce qui suit la reparation peut lever, et la
        # mutation etait EN TETE: `best_video.sameAudioMD5UseForCalculation` etait
        # deja accroche quand la levee partait. La zone A attrape
        # (mergeVideo.py:808) et journalise "The repair raised and was
        # abandoned", et `repaired_videos` reste vide parce que le `return` n'a
        # jamais eu lieu. Le journal dit donc QU'IL NE S'EST RIEN PASSE pendant
        # que l'objet repare est accroche -- et a trois fichiers ou plus, le
        # garde `len(dict_file_path_obj) < 2` ne se declenche pas, la fusion
        # continue, et mergeVideo.py:1815 parcourt cette liste SANS consulter
        # `repaired_videos`. Le fichier produit porterait une piste fabriquee
        # que le compte rendu declare inexistante -- la panne de provenance que
        # SPEC_ZONE_A.MD s4 existe pour empecher, arrivee par le chemin d'erreur.
        #
        # On calcule donc TOUT ce qui peut lever d'abord, on raconte, et on
        # mute en dernier. `plan.get("change_points", [])` rend None quand la
        # cle existe a None -- ce que la mesure produit -- et `for c in None`
        # est une TypeError: c'etait une levee REELLE juste apres la mutation.
        # Trouve par l'architecte en lisant le correctif precedent plutot que
        # le code corrige.
        coarse = [c for c in (plan.get("change_points") or [])
                  if c.get("narrowed") is False]
        detail = {"plan_source": plan_source,
                  "unverified_segment_ms": str(assembly["unverified_segment_ms"]),
                  "coarse_brackets": coarse,
                  "marker": assembly["marker"], "path": assembly["path"],
                  "audios": assembly["audios"], "subtitles": assembly["subtitles"],
                  "declined": assembly["declined"], "failed": assembly["failed"],
                  "verification": assembly["verification"]}
        reason = (f"{len(assembly['audios'])} audio and "
                  f"{len(assembly['subtitles'])} subtitle track(s) rebuilt, "
                  f"{len(assembly['declined'])} declined, "
                  f"{len(assembly['failed'])} failed")
        if len(coarse):
            # `narrowed: false` = les deux longueurs de fenetre ont diverge et la
            # mesure est retombee sur un intervalle d'une inter-fenetre, ~108 s.
            # Le trou correspondant est cher; il faut que ca se voie dans le
            # journal plutot que dans un champ que personne ne lit.
            sys.stderr.write(f"repair: {len(coarse)} change point(s) of "
                             f"{candidate_path} have a COARSE bracket; the gap "
                             f"substituted from the master is correspondingly wide\n")
            tools.logs.append(f"repair: {len(coarse)} coarse bracket(s) for {candidate_path}\n")
        sys.stdout.write(f"\tRepaired {candidate_path} as {assembly['path']} "
                         f"({assembly['marker']})\n")
        # DERNIER, ET ADJACENT. Plus rien entre le compte rendu et la mutation:
        # soit les deux ont lieu, soit aucun des deux, et le journal ne peut plus
        # etre en desaccord avec l'etat partage. L'accrochage a `best_video` est
        # la toute derniere instruction parce que c'est la seule que l'appelant
        # peut voir apres une levee.
        record(candidate_path, "repaired", reason, detail)
        repaired.append(candidate_path)
        best_video.sameAudioMD5UseForCalculation.append(repaired_obj)
    return repaired
