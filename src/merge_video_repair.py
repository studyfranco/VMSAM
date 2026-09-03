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

CINQ ISSUES, PAS DEUX. BRIEF_COMMON regle 5: "je n'ai pas pu mesurer" et "ce
fichier est irreparable" sont des reponses differentes, et l'issue qui manque
toujours est *l'instrument n'a pas tourne*. D'ou:

    disabled      le drapeau est ferme -- rien n'a tourne
    no_plan       aucune mesure pour cette paire -- rien n'a tourne
    declined      une mesure existait, la reparation l'a REFUSEE, avec sa raison
    repaired      un fichier a ete produit et raccroche
    failed        la reparation a tourne et a casse

Le drapeau vit dans `config.ini` [features], ferme par defaut, et PAS dans
l'environnement: `run.sh` ne passe aucun `--config` et le `Dockerfile` est gele,
donc un nouveau nom `VMSAM_*` est absent au runtime et un drapeau
d'environnement ferme par defaut est ferme pour toujours -- une suppression, pas
une porte. La campagne 1 l'a mesure: le code lisait 18 noms `VMSAM_`, l'image en
posait deux.
'''

from decimal import Decimal
from os import environ, path
import hashlib
import json
import sys

import tools
import video

feature_section = "features"
feature_chimeric = "repair_chimeric"
feature_resample = "repair_resample"
feature_plan_file = "repair_plan_file"
feature_verify = "repair_verify"
feature_verify_tolerance = "repair_verify_tolerance_ms"
feature_max_silence = "repair_max_silence_fraction"
feature_use_locator = "repair_use_locator"
feature_resample = "repair_resample"

# Compte-rendu de la derniere execution, remis a zero a chaque appel. Existe
# pour qu'un test puisse affirmer LAQUELLE des cinq issues s'est produite: une
# reparation qui decline silencieusement est indiscernable d'une reparation qui
# n'a jamais tourne, et c'est exactement le defaut que la campagne 1 a paye.
last_repair_report = []


def get_config_path():
    '''Le meme chemin que `tools.load_merge_runtime_from_env` resout.'''
    from_environment = environ.get("VMSAM_CONFIG", "").strip()
    if len(from_environment):
        return from_environment
    return tools.config_file


def get_features():
    '''[features] du config.ini, ou vide.

    Defensif a dessein: `tools.config_loader` LEVE quand la section manque, et
    un binaire deploye avec un config.ini plus ancien que ce module ne doit pas
    faire tomber un merge qui marchait. Section absente == tout ferme.
    '''
    try:
        return tools.config_loader(get_config_path(), feature_section)
    except Exception:
        return {}


def feature_enabled(features, key):
    value = features.get(key, "")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_plans(features):
    '''Charge le fichier de plans, ou {}.

    Sert deux usages: la mesure de dev-1 quand elle arrivera par disque, et --
    aujourd'hui -- le point de changement que je fournis A LA MAIN pour prouver
    le chemin. Cle = chemin absolu du candidat.
    '''
    plan_path = features.get(feature_plan_file, "").strip()
    if not len(plan_path):
        return {}
    if not tools.file_exists(plan_path):
        sys.stderr.write(f"repair: plan file {plan_path} does not exist\n")
        tools.logs.append(f"repair: plan file {plan_path} does not exist\n")
        return {}
    with open(plan_path) as plan_file:
        return json.load(plan_file)


def parse_segments(raw_segments):
    '''JSON -> Decimal. Les nombres arrivent en chaines pour ne rien perdre.'''
    segments = []
    for raw in raw_segments:
        segments.append({
            "master_start_ms": Decimal(str(raw["master_start_ms"])),
            "master_end_ms": Decimal(str(raw["master_end_ms"])),
            "candidate_offset_ms": Decimal(str(raw["candidate_offset_ms"])),
        })
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


def feature_enabled_by_default(features, key):
    '''Comme `feature_enabled`, mais OUVERT quand la cle est absente.

    Reserve a une sous-option d'une capacite deja fermee par defaut. La regle de
    `WRITE_ZONES.MD` §4 est qu'une capacite nouvelle ne doit pas pouvoir tourner
    sans qu'on l'ouvre; elle est tenue par `repair_chimeric`, qui est ferme. La
    verification n'est pas une capacite de plus: c'est ce qui empeche la
    reparation de livrer une piste silencieusement fausse, et une securite qu'il
    faut penser a armer n'en est pas une.
    '''
    value = features.get(key)
    if value == None or not len(str(value).strip()):
        return True
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def get_max_silence_fraction(features):
    '''Aucune limite par defaut, et c'est deliberé.

    Un seuil doit venir d'un ecart mesure. Personne n'a mesure combien de silence
    rend une piste inutilisable, donc en poser un ici serait une valeur plausible
    inventee -- le defaut que `SPEC_ZONE_A.MD` s3 raconte a propos de
    `get_less_channel_number`. Le chiffre est expose par piste dans le rapport;
    le proprietaire posera la limite quand il aura de quoi la justifier.
    '''
    value = features.get(feature_max_silence, "").strip()
    if not len(value):
        return None
    try:
        return Decimal(value)
    except Exception:
        sys.stderr.write(f"repair: {feature_max_silence}={value!r} is not a number, ignored\n")
        tools.logs.append(f"repair: {feature_max_silence}={value!r} is not a number, ignored\n")
        return None


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


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root):
    '''Construit le fichier repare et l'objet video qui va avec.

    Renvoie (objet, compte-rendu de l'assemblage).
    '''
    import merge_video_chimeric

    key = hashlib.md5(candidate_obj.filePath.encode()).hexdigest()[:16]
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    features = get_features()
    tolerance = features.get(feature_verify_tolerance, "").strip()
    tolerance = int(tolerance) if tolerance.isdigit() else 100
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
    marker = get_marker_value_for(plan, speed_ratio, candidate_obj, master_obj)
    assembly = merge_video_chimeric.assemble_on_master_timeline(
        candidate_obj, master_obj,
        clamp_segments_to_master(segments, master_obj),
        work_dir, out_path, marker,
        speed_ratio=speed_ratio,
        verify=feature_enabled_by_default(features, feature_verify),
        verify_tolerance_ms=tolerance,
        max_silence_fraction=get_max_silence_fraction(features))

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
    if tools.dev or outcome in ("repaired", "failed", "declined"):
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
    features = get_features()
    if not feature_enabled(features, feature_chimeric):
        for candidate_path in list_not_compatible_video:
            record(candidate_path, "disabled",
                   f"config.ini [{feature_section}] {feature_chimeric} is off")
        return []

    plans = load_plans(features)
    work_root = path.join(tools.tmpFolder, "repair")
    tools.make_dirs(work_root)
    repaired = []

    for candidate_path in list_not_compatible_video:
        candidate_obj = dict_file_path_obj.get(candidate_path)
        if candidate_obj == None:
            record(candidate_path, "declined",
                   "the rejected path has no video object in dict_file_path_obj")
            continue
        plan = plans.get(candidate_path)
        plan_source = "plan file"
        if plan == None and feature_enabled_by_default(features, feature_use_locator):
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
        if speed_ratio != None and not feature_enabled(features, feature_resample):
            record(candidate_path, "disabled",
                   f"the measurement asks for a speed correction and "
                   f"config.ini [{feature_section}] {feature_resample} is off")
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
                record(candidate_path, "declined", str(error))
                sys.stderr.write(f"repair: declined {candidate_path}: {error}\n")
            else:
                record(candidate_path, "failed", str(error))
                sys.stderr.write(f"repair: failed for {candidate_path}: {error}\n")
            continue

        best_video.sameAudioMD5UseForCalculation.append(repaired_obj)
        repaired.append(candidate_path)
        record(candidate_path, "repaired",
               f"{len(assembly['audios'])} audio and "
               f"{len(assembly['subtitles'])} subtitle track(s) rebuilt, "
               f"{len(assembly['declined'])} declined, "
               f"{len(assembly['failed'])} failed",
               {"plan_source": plan_source,
                "coarse_brackets": [c for c in plan.get("change_points", [])
                                    if c.get("narrowed") is False],
                "marker": assembly["marker"], "path": assembly["path"],
                "audios": assembly["audios"], "subtitles": assembly["subtitles"],
                "declined": assembly["declined"], "failed": assembly["failed"],
                "verification": assembly["verification"]})
        coarse = [c for c in plan.get("change_points", []) if c.get("narrowed") is False]
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
    return repaired
