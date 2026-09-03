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


def get_marker_value(plan):
    '''SPEC_ZONE_A.MD s4. `chimeric+resampled:<factor>` DANS CET ORDRE.

    Le facteur ecrit est celui reellement applique, a la precision reellement
    appliquee: un tag `resampled:1.042709` sur une piste etiree autrement est
    pire que pas de tag du tout.
    '''
    parts = []
    if plan.get("kind") == "piecewise_constant" or len(plan.get("segments", [])):
        parts.append("chimeric")
    applied = plan.get("applied_speed_factor")
    if applied != None:
        parts.append(f"resampled:{applied}")
    return "+".join(parts)


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root):
    '''Construit le fichier repare et l'objet video qui va avec.

    Renvoie (objet, compte-rendu de l'assemblage).
    '''
    import merge_video_chimeric

    key = hashlib.md5(candidate_obj.filePath.encode()).hexdigest()[:16]
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    marker = get_marker_value(plan)
    assembly = merge_video_chimeric.assemble_on_master_timeline(
        candidate_obj, master_obj, parse_segments(plan["segments"]),
        work_dir, out_path, marker)

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
        if plan == None:
            record(candidate_path, "no_plan",
                   "no measurement available for this pair")
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
               {"marker": assembly["marker"], "path": assembly["path"],
                "audios": assembly["audios"], "subtitles": assembly["subtitles"],
                "declined": assembly["declined"], "failed": assembly["failed"]})
        sys.stdout.write(f"\tRepaired {candidate_path} as {assembly['path']} "
                         f"({assembly['marker']})\n")
    return repaired
