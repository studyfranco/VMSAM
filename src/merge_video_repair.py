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
        return None
    return change_point_locator.locate_change_points(best_video, candidate_obj, language)


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


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root):
    '''Construit le fichier repare et l'objet video qui va avec.

    Renvoie (objet, compte-rendu de l'assemblage).
    '''
    import merge_video_chimeric

    key = hashlib.md5(candidate_obj.filePath.encode()).hexdigest()[:16]
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    speed_ratio, _ = get_speed_ratio(plan)
    segments = plan.get("segments")
    if not segments:
        # Un plan de VITESSE SEULE n'a pas de tranche: la relation couvre tout le
        # fichier. On en fabrique une qui couvre la timeline du maitre, plutot que
        # d'exiger de la mesure une structure qu'elle n'a pas a inventer.
        segments = [{"master_start_ms": Decimal("0"),
                     "master_end_ms": get_master_timeline_ms(master_obj),
                     "candidate_offset_ms": Decimal(str(plan.get("base_offset_ms", 0)))}]
    else:
        segments = parse_segments(segments)
    segments, unverified_ms, dropped_segments = drop_unverified_segments(segments)
    if not len(segments):
        raise merge_video_chimeric.chimeric_error(
            "every segment's offset is unverified (each shorter than the "
            "measurement's probe window); nothing can be spliced at a bounded offset")
    marker = get_marker_value_for(plan, speed_ratio, candidate_obj, master_obj)
    assembly = merge_video_chimeric.assemble_on_master_timeline(
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
    cut = text.find("(")
    if cut != -1:
        text = text[:cut]
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
    """`speed_margin=` quand elle existe, la RAISON quand elle n'existe pas.

    Quatre champs, chacun avec son propre emetteur, parce qu'ils repondent a
    quatre questions et qu'un seul emetteur conditionnel les fait disparaitre
    ensemble:

        speed_margin                 la marge de platitude, en ms
        speed_margin_absent_reason   pourquoi elle n'existe pas
        fidelity_margin              une autre quantite, sans unite
        decided_by                   quel critere a tranche

    `NON EMISE` chez le consommateur ne doit se declencher que quand la marge est
    absente ET qu'aucune raison ne l'accompagne -- c'est-a-dire quand le
    producteur n'a rien dit. Aujourd'hui il se declenchait sur le cas INVERSE.
    """
    if not plan:
        return ""
    parts = []
    margin = get_speed_margin(plan)
    if margin != None:
        parts.append(f"speed_margin={margin}")
    else:
        reason = plan.get("speed_margin_absent_reason")
        if reason != None:
            parts.append(f"speed_margin=absent({reason})")
    fidelity = plan.get("fidelity_margin")
    if fidelity != None:
        parts.append(f"fidelity_margin={fidelity}")
    decided = plan.get("decided_by")
    if decided != None:
        parts.append(f"decided_by={decided}")
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
        return "unreported(this assembly predates the field)"
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
    import hashlib
    # IMPORT TARDIF, comme partout ailleurs dans ce module: la tete de
    # `mergeVideo.py` est hors zone taguee et un deploiement partiel ne doit pas
    # pouvoir empecher le demarrage.
    try:
        import merge_video_chimeric as _chi
        chimeric_file = getattr(_chi, "__file__", None)
    except Exception:
        chimeric_file = None
    parts = []
    for module in (__file__, chimeric_file):
        if not module:
            continue
        try:
            with open(module, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()[:12]
        except Exception:
            # UNE EMPREINTE QU'ON N'A PAS PU LIRE SE DIT. Elle ne vaut pas zero et
            # elle ne s'omet pas: un champ absent se lirait comme un vieux build.
            digest = "unreadable"
        parts.append(f"{path.basename(module)}:{digest}")
    return " ".join(parts) if parts else "unreadable"


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
        spans.append(f"{piece['source'][0]}{int(start)}-{int(end)}")
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
    tools.logs.append(f"repair: plan {plan.get('kind') if plan else 'none'} "
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
                      f"quantum={quantum_ms} pieces={' '.join(spans)}\n")

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
                f" offset_ms=0\n")
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


def record(candidate_path, outcome, reason, detail=None):
    entry = {"candidate": candidate_path, "outcome": outcome, "reason": reason,
             "detail": detail}
    last_repair_report.append(entry)
    if tools.dev or outcome in ("repaired", "failed", "declined", "no_plan"):
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
            record(candidate_path, "no_plan",
                   "could not tell which language the merge measured on")
            continue
        plan = get_plan_from_locator(best_video, candidate_obj, language)
        plan_source = "change_point_locator"
        if plan == None:
            # None de la mesure = "je n'ai pas pu mesurer", et surtout pas
            # "les fichiers vont ensemble". Le refus reste, intact.
            record(candidate_path, "no_plan",
                   f"no measurement available for this pair ({plan_source})")
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
                record(candidate_path, "declined", str(error),
                       {"verification": getattr(error, "verification", None),
                        "audios": getattr(error, "audios", None)})
                sys.stderr.write(f"repair: declined {candidate_path}: {error}\n")
            else:
                record(candidate_path, "failed", str(error))
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
