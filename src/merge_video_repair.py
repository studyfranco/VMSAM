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
        return preferred
    if len(keys) == 1:
        return keys[0]
    # Repli: la seule langue audio commune aux deux fichiers.
    common = set(best_video.audios.keys()) & set(candidate_obj.audios.keys())
    common.discard("und")
    if len(common) == 1:
        return common.pop()
    if preferred in common:
        return preferred
    return keys[0] if len(keys) else None


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
    kept, dropped_ms = [], Decimal("0")
    for segment in segments:
        if segment.get("offset_unverified"):
            span = Decimal(str(segment["master_end_ms"])) - Decimal(str(segment["master_start_ms"]))
            dropped_ms += span
            continue
        kept.append(segment)
    return kept, dropped_ms


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
    segments, unverified_ms = drop_unverified_segments(segments)
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
        verify=True, verify_tolerance_ms=verify_tolerance_ms)

    # Le compte-rendu porte la mesure jetee: `repair_not_compatible_videos` la
    # cite dans son entree "repaired", et elle etait jusqu'ici une locale d'ici,
    # donc invisible la-bas -- toute reparation REUSSIE levait un NameError,
    # apres avoir deja accroche l'objet a best_video. Trouve le 2026-09-03 en
    # branchant le balayage sur cette fonction plutot que sur l'assembleur:
    # aucun test ne parcourait la branche de succes de l'orchestrateur.
    assembly["unverified_segment_ms"] = unverified_ms

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
    return repaired_obj, assembly


def mark_audio_dicts(repaired_obj, marker):
    if not len(marker):
        return
    for holder in (repaired_obj.audios, repaired_obj.commentary,
                   repaired_obj.audiodesc):
        for language, audios in holder.items():
            for audio in audios:
                audio["fabricated"] = marker


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
        language = get_delay_language(best_video, candidate_obj)
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
        if plan.get("master_path") not in (None, best_video.filePath):
            record(candidate_path, "declined",
                   f"the plan was measured against another master than "
                   f"{best_video.filePath}")
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
                       {"verification": getattr(error, "verification", None)})
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
