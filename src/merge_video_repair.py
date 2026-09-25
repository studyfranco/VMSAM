'''
Reparation d'un fichier REFUSE, au moment du refus.

Appele depuis la zone A (`mergeVideo.remove_not_compatible_video`,
`SPEC_ZONE_A.MD` s1). Ce module est l'ENTREE de la reparation: il ecarte ce que
l'orchestrateur ne peut pas voir (pas d'objet, pas de langue), passe chaque
candidat a `repair_orchestrator.repair()` -- LA chaine depuis la bascule du
2026-09-24 (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD ADDENDUM 8) -- puis
raccroche l'objet repare au merge. Il porte aussi les briques que
l'orchestrateur et l'application du plan reutilisent: `record`,
`master_intertrack_verdict`, `build_repaired_video_object`,
`gate_fabricated_delivery`, et la retraite des pools ffmpeg (CASE id 6).

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

LA PORTE EST LA MESURE. L'orchestrateur rend False quand il ne peut rien
etablir, avec un jeton et sa classe de mesure, et on decline. L'erreur 237 a ete refusee sur une fidelite mediane de
0.576 avec 18 changements de signe, pas sur un reglage: c'est une procedure de
decision, pas un renvoi vers quelqu'un qui n'est pas la.
'''

from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from os import path
import hashlib
import json
import sys

import tools
import repair_log
import video

# Tolerance d'alignement du verificateur, en millisecondes. CONSTANTE et non
# reglage: un seuil doit venir d'un ecart mesure. Mesure sur de vrais fichiers --
# un plan correct atterrit entre 0.5 et 2.8 ms (erreurs 266 et 108, six sondes
# chacune); un plan faux atterrit a 503 ms (point de changement manque) ou
# 16146 ms (signe inverse). 100 ms est deux ordres au-dessus du premier et un
# ordre en dessous du second.
# ADDENDUM 25.3 (owner, 2026-09-25): 100 -> 15 ms, SUR MESURE. La certification des etages 4+5
# a mesure les produits propres a <= 10,5 ms (13 produits) et les vraies erreurs de plan a
# 26-67 ms (trois sauts sous-quantum livres parce qu'ils passaient sous 100 ms). 15 ms separe
# les deux populations avec de la marge des deux cotes. Au-dela: `delivery_offset_exceeds_
# tolerance`, avec la mesure -- jamais une livraison.
verify_tolerance_ms = 15

last_repair_report = []


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
    tools.dev_log(f"repair: assemble_or_log_the_decline starting "
                  f"candidate={logged_candidate.filePath}\n")
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
        # ROUTED THROUGH `tools.log_always` (owner's order via the Lead,
        # 2026-09-22, wave 3b), NOT `tools.dev_log` -- this line is
        # deliberately unconditional (see the comment above: "sans elle,
        # 'pas de ligne DECLINED' se lirait comme 'pas de declin'"), and
        # `dev_log` would put it BEHIND a gate that was never there. This is
        # the terminal verdict for the assembly attempt -- exactly the line
        # a hung-then-recovered or completed job needs in stderr, and until
        # now it only ever reached `tools.logs`.
        if isinstance(error, merge_video_chimeric.chimeric_error):
            tools.log_always(f"repair: DECLINED {error}\n")
        else:
            tools.log_always(f"repair: FAILED {type(error).__name__}: {error}\n")
        # ET L'ETAT DE L'ARTEFACT ATTEINT UNE LIGNE, PAR CLE ET NON PAR PROSE.
        # dev-4 lit par nom; `state=` et `path=` se lisent, "the file was
        # renamed" ne se lit pas. Emise seulement quand un fichier a REELLEMENT
        # ete marque: son ABSENCE dit "aucun artefact n'existait a marquer",
        # ce qui est un troisieme fait et pas un defaut de journal.
        marked = getattr(error, "undelivered_path", None)
        if marked != None:
            # `durable=` RETIRE (vmsam-lead, 2026-09-21): depuis que le
            # magasin durable n'existe plus, ce champ n'aurait plus jamais pu
            # rendre qu'une seule valeur -- un champ a une seule valeur
            # possible est le meme defaut de champ inerte qu'un champ jamais
            # rempli, sous un autre angle. `undelivered_durable` n'existe plus
            # du tout (ni sur `error`, ni ici) -- ce n'etait plus qu'une
            # constante que rien ne lisait. `state=`/`path=`/`in_place=`
            # restent: ils portent la DECISION du refus et ou le produit
            # refuse se trouve, ce que la ruling de l'Architect preserve
            # explicitement.
            tools.logs.append(
                f"repair: undelivered state={getattr(error, 'undelivered_state', 'unnamed')} "
                f"path={marked} "
                f"in_place={getattr(error, 'undelivered_in_place', 'unreported')}"
                .rstrip() + "\n")
        raise


SPEED_EVIDENCE_INSTRUMENTS = frozenset({"rate_arm"})
# SINCE ADDENDUM 30.5 THE ONLY PRODUCER IS THE RATE ARM (`repair_orchestrator.rate_arm`): its
# evidence is the winner's re-fingerprinted alignment, not a sweep's window median -- `rate_sweep`
# left this vocabulary with its producer.
# THE DECIDING INSTRUMENTS THIS GUARD RECOGNISES, as a closed vocabulary --
# the same shape as `repair_orchestrator.DECLINE_CAUSES`, and for the same
# reason: a vocabulary kept next to its only consumer cannot drift from it,
# and a new instrument is not admissible without being enumerated in the same
# edit that starts producing it. An unenumerated `rate_source` is REFUSED
# here, not tolerated: this gate stands in front of a destructive transform,
# so the safe direction of an unknown value is "no".

# THE APPLIED-VS-EVIDENCED TOLERANCE, RELATIVE. Moved here 2026-09-24 from
# `pal_speed_discriminator.SNAP_RELATIVE_TOLERANCE` when the switch to the
# orchestrator left this guard its only reader and the module was removed.
# ARITHMETIC, NOT A TUNED MARGIN: the two closest named broadcast rates,
# 1001/960 = 1.04270833 and 25/24 = 1.04166667, sit a relative 1.0e-3 apart,
# and 5e-4 is half of that, so no measurement can be within tolerance of two
# of them at once. `speed_plan_evidence` restates why it still holds on the
# sweep's larger vocabulary.
SPEED_EVIDENCE_RELATIVE_TOLERANCE = Decimal("0.0005")


def speed_plan_evidence(plan, speed_ratio):
    '''Does this plan carry the VALIDATION EVIDENCE that makes its
    `speed_ratio` admissible? Returns `(admissible, token, prose)`.

    RULING_20260922_NO_BAND_ROUTING.MD, ADDENDUM: the
    `speed_transform_not_validated` guard is COMPLETED, NOT LIFTED. The three
    things the addendum names must ALL be present, and this function is where
    "present" is defined:

      1. THE WINNING EXACT RATIONAL -- `speed_ratio_exact` parses as a
         Fraction AND is a member of the sweep's own vocabulary
         (`merge_video_resample.build_rate_ratio_vocabulary`), AND the
         `speed_ratio` actually being applied is that rational. A plan may
         not carry evidence about one number and apply another; that is the
         whole failure mode a guard in front of a destructive transform
         exists to catch.

         CHECKED AGAINST THE SWEEP'S VOCABULARY, NOT THE DISCRIMINATOR'S SIX
         (ADDENDUM 2 -- the sweep decides now, so the sweep's set is the
         authority on what a winner may be). The discriminator's
         `NAMED_RATE_RATIONALS` is a SUBSET of those sixteen, so nothing that
         was admissible before this change stopped being admissible.
      2. THE RATE ARM'S WINNER (ADDENDUM 30.5) -- `resample_gate["verdict"] ==
         "confirmed"`, a REAL `span_coverage` (the re-fingerprinted alignment's
         share of the master timeline) at or above
         `rate_direction.RATE_ARM_MIN_SPAN_COVERAGE`, read as a number, never as
         the presence of a key; and the plan's `speed_engine` is the engine
         that aligned.
      3. THE DECIDING INSTRUMENT -- `rate_source` names one of
         `SPEED_EVIDENCE_INSTRUMENTS`.

    EACH FAILURE HAS ITS OWN TOKEN, because a future correction acts
    differently on each (the Lead's granularity rule R1): a plan with no
    evidence at all is a producer that never ran this route; a plan whose
    rational is not named is a producer inventing factors; a plan whose span
    sits below the floor is a producer shipping a measured negative as a
    confirmation. The token travels in the refusal PROSE -- the raised
    `cause` stays the stable `speed_transform_not_validated`, because
    `chimeric_cause`'s own docstring bounds the tokened `chimeric_error`
    sites and this completion does not add a fourth.

    THIS FUNCTION ONLY EVER SAYS YES TO A NUMBER THAT THREE INDEPENDENT
    THINGS AGREE ON. It cannot say yes to a plan that merely looks confident.
    '''
    import merge_video_resample

    vocabulary = merge_video_resample.build_rate_ratio_vocabulary()
    exact = plan.get("speed_ratio_exact")
    if exact is None:
        return False, "speed_evidence_absent", (
            "the plan carries no speed_ratio_exact: no winning exact rational "
            "travels with this coefficient, so nothing says WHICH exact ratio "
            "was recognised or by what")
    try:
        named = Fraction(str(exact))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False, "speed_evidence_rational_unreadable", (
            f"speed_ratio_exact={exact!r} does not parse as an exact rational")
    if named not in vocabulary:
        return False, "speed_evidence_rational_not_named", (
            f"speed_ratio_exact={named} is not a member of the rate sweep's "
            f"vocabulary {[str(f) for f in vocabulary]}: an exact-looking "
            f"fraction is not a recognised broadcast rate combination")

    # THE APPLIED NUMBER MUST BE THE EVIDENCED NUMBER. Compared at the same
    # RELATIVE tolerance the snap uses, and that constant survives the move to
    # the larger set by ARITHMETIC, not by luck: the closest pair anywhere in
    # the sixteen is 800/1001 = 0.7992008 against 4/5 = 0.8000000, a relative
    # gap of 9.99e-4, so 5e-4 is still just under half of it and still cannot
    # confuse two vocabulary members. An exact equality test would fail on
    # representation alone -- `speed_ratio` reaches here through `str()`.
    nominal = Decimal(named.numerator) / Decimal(named.denominator)
    drift = abs(Decimal(str(speed_ratio)) - nominal) / nominal
    if drift > SPEED_EVIDENCE_RELATIVE_TOLERANCE:
        return False, "speed_evidence_ratio_is_not_the_snapped_rational", (
            f"the plan would apply speed_ratio={speed_ratio} while its "
            f"evidence is for {named} ({nominal}): relative drift {drift} "
            f"exceeds {SPEED_EVIDENCE_RELATIVE_TOLERANCE}. "
            f"Evidence about one coefficient does not license another")

    instrument = plan.get("rate_source")
    if instrument not in SPEED_EVIDENCE_INSTRUMENTS:
        return False, "speed_evidence_instrument_unrecognised", (
            f"rate_source={instrument!r} is not one of "
            f"{sorted(SPEED_EVIDENCE_INSTRUMENTS)}: the deciding instrument "
            f"must be named, and an unenumerated one is refused rather than "
            f"trusted")

    gate = plan.get("resample_gate")
    if not isinstance(gate, dict):
        return False, "speed_evidence_no_gate", (
            "the plan names an instrument but carries no resample_gate: the rate arm's own "
            "result is missing, so the coefficient was never validated")
    if gate.get("verdict") != "confirmed":
        return False, "speed_evidence_gate_not_confirmed", (
            f"resample_gate verdict={gate.get('verdict')!r} cause={gate.get('cause')!r}: the "
            f"rate arm did not confirm")
    # THE ARM'S EVIDENCE (ADDENDUM 30.5): the winner's span coverage at or above the arm's own
    # floor, measured as a number (a missing number is not a passing number), and the engine
    # the plan will apply is the engine that aligned.
    import rate_direction
    span = gate.get("span_coverage")
    if not isinstance(span, (int, float)) or isinstance(span, bool):
        return False, "speed_evidence_span_absent", (
            f"resample_gate says confirmed but span_coverage={span!r} is not a measured number")
    if span < rate_direction.RATE_ARM_MIN_SPAN_COVERAGE:
        return False, "speed_evidence_span_below_floor", (
            f"the winner's span coverage {span} is below the arm's floor "
            f"{rate_direction.RATE_ARM_MIN_SPAN_COVERAGE}")
    engine = plan.get("speed_engine")
    if engine not in merge_video_resample.SPEED_ENGINES or engine != gate.get("engine"):
        return False, "speed_evidence_engine_mismatch", (
            f"the plan would apply engine {engine!r} while the rate arm's winner aligned with "
            f"{gate.get('engine')!r}")

    return True, "speed_evidence_complete", (
        f"snapped named rational {named} (applied as {speed_ratio}, engine {engine}), winner "
        f"span coverage {span} >= {rate_direction.RATE_ARM_MIN_SPAN_COVERAGE}, deciding "
        f"instrument {instrument}")


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root, job_start_utc):
    '''Construit le fichier repare et l'objet video qui va avec, A PARTIR DU PLAN
    DE L'ORCHESTRATEUR (`repair_orchestrator.apply_plan`, etage 5).

    `plan` porte des MORCEAUX DEJA CONSTRUITS, un jeu par piste (`track_plans`)
    et celui de la piste de comparaison (`reference_pieces`, qui re-cale les
    sous-titres et que le verificateur sonde), le marqueur decide par l'ADDENDUM
    5, et -- sur une paire a taux -- le ratio EXACT avec ses preuves. Cette
    fonction ne mesure rien et ne re-decoupe rien (ADDENDUM 10 d).

    `job_start_utc`: EXIGE, SANS DEFAUT -- voir VMSAM_ERA a l'appelant.
    Traverse cette fonction sans etre lu: seul `assemble_on_master_timeline`
    en a besoin, pour le tag pose au mux.

    Renvoie (objet, compte-rendu de l'assemblage).
    '''
    import merge_video_chimeric

    # UNE SEULE DERIVATION, PARTAGEE: la meme cle sert de repertoire de travail
    # et nomme le fichier produit.
    key = merge_video_chimeric.stable_case_key(candidate_obj.filePath)
    work_dir = path.join(work_root, key)
    tools.make_dirs(work_dir)
    out_path = path.join(work_root, f"{key}_repaired.mkv")

    # WHERE IT PLANTS (owner's decision, 2026-09-22): if this candidate's repair
    # wedges anywhere downstream, this is the last line that says where on disk
    # its intermediate and final artefacts were headed.
    tools.dev_log(f"repair: build_repaired_video_object starting "
                  f"candidate={candidate_obj.filePath} work_dir={work_dir} "
                  f"out_path={out_path}\n")

    speed_ratio = plan.get("speed_ratio")
    if speed_ratio is not None:
        # LA GARDE DEVANT LA TRANSFORMATION DESTRUCTIVE, COMPLETEE ET NON LEVEE
        # (RULING_20260922_NO_BAND_ROUTING.MD, ADDENDUM), ET MAINTENANT
        # ATTEIGNABLE: l'ADDENDUM 8 autorise la livraison d'un audio resample,
        # et un coefficient n'est admissible qu'avec la preuve qui l'a valide --
        # un rationnel nomme du vocabulaire du balayage, une mediane d'echelle
        # au-dessus du plancher inchange, un instrument nomme, les trois sur le
        # MEME nombre. `speed_plan_evidence` est cet enonce; ce qu'il ne peut
        # pas garantir leve, avec la cause stable qu'il a toujours eue.
        admissible, evidence_token, evidence_prose = speed_plan_evidence(
            plan, speed_ratio)
        tools.dev_log(
            f"repair: speed evidence gate for {candidate_obj.filePath}: "
            f"speed_ratio={speed_ratio} admissible={admissible} "
            f"token={evidence_token} detail={evidence_prose}\n")
        if not admissible:
            raise merge_video_chimeric.chimeric_error(
                f"speed transform not validated for production application: "
                f"speed_ratio={speed_ratio} reached build_repaired_video_object "
                f"without the validation evidence that makes it admissible "
                f"({evidence_token}: {evidence_prose}) -- refusing rather "
                f"than applying an unevidenced transform",
                cause="speed_transform_not_validated")

    assembly = assemble_or_log_the_decline(
        candidate_obj, plan, Decimal("0"),
        candidate_obj, master_obj, plan["track_plans"], plan["reference_pieces"],
        work_dir, out_path, plan["marker"],
        job_start_utc=job_start_utc,
        speed_ratio=speed_ratio,
        # LE FLUX MAITRE SUR LEQUEL LA MESURE A ETE PRISE: la seule piste dont
        # on SAIT, par mesure, qu'elle est calee sur le plan.
        reference_stream=plan.get("reference_stream"),
        # LA LANGUE DE COMPARAISON, repli de remplissage quand le maitre ne
        # porte pas la langue de la piste (SPEC_ZONE_A.MD s4c).
        comparison_language=plan.get("language"),
        chapters_path=plan.get("chapters_path"),
        verify=True, verify_tolerance_ms=verify_tolerance_ms,
        # ADDENDUM 26 (report commit): the verifier's time counts against the repair's budget.
        deadline=plan.get("repair_deadline"),
        # ADDENDUM 30.5: the engine the rate arm measured.
        speed_engine=plan.get("speed_engine") or "asetrate")

    assembly["unverified_segment_ms"] = Decimal("0")
    # LE JOURNAL EST ECRIT ICI, avant que l'objet video soit construit: si la
    # relecture du fichier produit echoue, on veut quand meme savoir ce qui a
    # ete fait a chaque piste.
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
    mark_audio_dicts(repaired_obj, assembly["marker"])
    # LA PORTE DE LIVRAISON DES PISTES FABRIQUEES (CASE_wakeup20260924, defaut A):
    # le seul point ouvert qui voit TOUTES les pistes construites. `keep=False`
    # pose ici est lu par `generate_new_file_audio_config`.
    assembly["fabricated_dropped"] = gate_fabricated_delivery(
        repaired_obj, master_obj, work_dir=work_dir)
    return repaired_obj, assembly


def mark_audio_dicts(repaired_obj, marker):
    # LE MARQUEUR DE CHAQUE PISTE EST CELUI QU'ELLE PORTE DANS LE FICHIER
    # (`extra.VMSAM_FABRICATED`, pose par piste au mux: deux pistes a deux
    # frequences recoivent deux facteurs `resampled:` differents); le marqueur
    # de fichier n'est que le repli d'une piste relue sans son tag.
    # PAS LES COMMENTAIRES: les marquer ici changerait le holder par lequel
    # `gate_fabricated_delivery` les reconnait -- ils y sont traites a part.
    for holder in (repaired_obj.audios, repaired_obj.audiodesc):
        for language, audios in holder.items():
            for audio in audios:
                own = fabricated_marker_of(audio)
                if own or len(marker):
                    audio["fabricated"] = own or marker


# Les trois porteurs d'audio d'un objet video. Les commentaires y sont: c'est
# par eux que Rick and Morty S01E01/02 a livre une piste `chimeric` jamais
# comparee (CASE_wakeup20260924).
AUDIO_HOLDERS = ("audios", "commentary", "audiodesc")


def fabricated_marker_of(audio):
    """La valeur du marqueur, depuis L'UN OU L'AUTRE porteur (`SPEC_ZONE_A.MD`
    s4): la cle en memoire ou le tag re-sonde sous `extra.VMSAM_FABRICATED`.
    Chaine vide = piste intacte. Meme lecture que `tools.keep_best_audio_
    fabricated_status`, qui ne rend que le NOM du porteur, pas la valeur."""
    return str(audio.get("fabricated") or
               (audio.get("extra") or {}).get("VMSAM_FABRICATED") or "")


# LE SEUIL DE "MEME CONTENU" EST CELUI DU REGROUPEMENT GELE, PAS UN NOUVEAU.
# `mergeVideo.find_differences_and_keep_best_audio` (mergeVideo.py:1036-1120)
# decide quelles pistes d'une langue sont la MEME version -- et donc lesquelles
# `keep_best_audio` departage -- avec: `correlate` (audioCorrelation, empreintes
# chromaprint) sur `video.number_cut` fenetres placees par
# `prepare_get_delay_sub`, longueur passee `length_time*2`, puis moyenne des
# fidelites >= 0.90 ET un ensemble de decalages {0}, ou un seul |d| < 128 ms, ou
# deux valeurs toutes deux < 128 ms (mergeVideo.py:1058-1086). Recopie ici,
# constante pour constante, pour que la porte et le regroupement ne puissent
# pas etre en desaccord sur ce qu'est "la meme piste".
SAME_CONTENT_MEAN_FIDELITY = 0.90
SAME_CONTENT_MAX_DELAY_MS = 128


def same_content_verdict(delay_fidelity_values):
    """Le verdict de mergeVideo.py:1058-1086 sur UNE paire, sans les journaux.

    `delay_fidelity_values`: une liste de retours de `correlate`, un par
    fenetre -- (fidelite, _, decalage_ms). Renvoie (bool, moyenne, decalages)."""
    from statistics import mean
    fidelity = mean([fi[0] for fi in delay_fidelity_values])
    delays = set(fi[2] for fi in delay_fidelity_values)
    if fidelity < SAME_CONTENT_MEAN_FIDELITY:
        return False, fidelity, delays
    values = list(delays)
    if len(values) == 1:
        return abs(values[0]) < SAME_CONTENT_MAX_DELAY_MS, fidelity, delays
    if len(values) == 2:
        return (abs(values[0]) < SAME_CONTENT_MAX_DELAY_MS
                and abs(values[1]) < SAME_CONTENT_MAX_DELAY_MS), fidelity, delays
    return False, fidelity, delays


def measure_same_content(master_obj, master_audio, repaired_obj, audio, work_dir):
    """La piste fabriquee est-elle LA MEME VERSION que la piste intacte du maitre?

    MEMES INSTRUMENTS que le regroupement gele: fenetres de
    `video.generate_begin_and_length_by_segment`/`generate_cut_with_begin_length`
    sur la plus courte des deux durees, extraction pcm_s16le stereo (mono si
    l'une est mono, comme `prepare_get_delay_sub`), normalisation
    `video.generate_normalised_file`, `audioCorrelation.correlate` avec
    `length_time*2`. MAIS dans un repertoire PRIVE et sans passer par
    `extract_audio_in_part`: cette methode ecrit `tmpFiles` et
    `audio_pos_file` sur l'objet maitre du pipeline et nomme ses fichiers
    d'apres `fileBaseName` -- l'appeler ici effacerait ou ecraserait les
    extraits dont la fusion a encore besoin.

    Renvoie (verdict_bool_ou_None, detail). None = PAS MESURE: l'appelant ne
    doit pas le lire comme "different"."""
    import shutil
    import tempfile
    from time import strftime, gmtime
    from audioCorrelation import correlate
    private = tempfile.mkdtemp(prefix="fab_gate_", dir=work_dir or tools.tmpFolder)
    try:
        duration = min(float(master_audio["Duration"]), float(audio["Duration"]))
        begin, length_time = video.generate_begin_and_length_by_segment(duration)
        cuts = video.generate_cut_with_begin_length(
            begin, length_time, strftime('%H:%M:%S', gmtime(length_time * 2)))
        channels = "1" if "1" in (str(master_audio.get("Channels")),
                                  str(audio.get("Channels"))) else "2"
        codec_param = ["-c:a", "pcm_s16le", "-ac", channels]

        # EN PARALLELE, comme le pool `ffmpeg_pool_audio_convert` du chemin gele:
        # chaque fenetre est un ffmpeg independant (sortie `-ss` apres `-i`, donc
        # decodee depuis le debut) et en serie la mesure coutait des minutes.
        jobs = []

        def extract(file_path, stream_order, tag):
            out = []
            for number, cut in enumerate(cuts):
                final = path.join(private, f"{tag}.{number}.wav")
                tmp = path.join(private, f"{tag}_tmp.{number}.wav")
                cmd = [tools.software["ffmpeg"], "-y", "-analyzeduration", "1000M",
                       "-probesize", "1000M", "-threads", "3", "-nostdin", "-i",
                       file_path, "-copyts", "-vn", "-dn", "-sn"] + codec_param + [
                       "-map", f"0:{stream_order}", "-ss", cut[0], "-t", cut[1], tmp]
                jobs.append(pool.submit(video.generate_normalised_file, cmd,
                                        codec_param.copy(), final, tmp))
                out.append(final)
            return out

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, int(tools.core_to_use or 1))) as pool:
            master_cuts = extract(master_obj.filePath, master_audio["StreamOrder"], "m")
            fab_cuts = extract(repaired_obj.filePath, audio["StreamOrder"], "f")
            for job in jobs:
                job.result()
            values = list(pool.map(lambda pair: correlate(pair[0], pair[1], length_time * 2),
                                    zip(master_cuts, fab_cuts)))
        verdict, fidelity, delays = same_content_verdict(values)
        return verdict, f"mean_fidelity={fidelity:.4f} delays_ms={sorted(delays)} windows={len(values)}"
    except Exception as error:
        return None, f"unmeasured={type(error).__name__}: {error}"
    finally:
        shutil.rmtree(private, ignore_errors=True)


def gate_fabricated_delivery(repaired_obj, master_obj, work_dir=None,
                             content_probe=None):
    """Aucune piste fabriquee n'atteint la livraison sans avoir ete jugee.

    Sur CHAQUE piste audio du fichier repare (audios, commentaires,
    audio-description), dans cet ordre:

    0. PAS DE RETRAIT POUR CORRECTION DE VITESSE (owner, ADDENDUM 8,
       2026-09-24, qui leve le report de l'ADDENDUM 7): une piste
       `resampled:<f>` est une piste fabriquee comme les autres, jugee par les
       regles ci-dessous, et son marqueur `resampled:<facteur>` voyage avec
       elle jusqu'au produit (ADDENDUM 5 clause c) -- cette porte ne touche
       jamais aux tags, elle ne pose que `keep`.
    1. UN COMMENTAIRE N'EST JAMAIS COURU (owner, 2026-09-24: "un commentaire
       doit etre tague comme tel -- soit par sa piste, soit dans ses
       metadonnees"). Une piste rangee sous `.commentary` y est parce que
       `video.py:113` a lu son titre ("commentary") ou `flag_commentary`: la
       re-sonde du fichier intermediaire la range au meme endroit et
       `generate_merge_command_insert_ID_...` (mergeVideo.py:1556) pose
       `--commentary-flag` au mux final. Gardee, `fabricated_kept
       cause=commentary_tagged`, avec ce qui la porte.
    2. LA PISTE INTACTE GAGNE -- SUR LE MEME CONTENU SEULEMENT (owner,
       2026-09-16, precise le 2026-09-24: "une VFQ chimerique ne sera jamais
       perdue face a une VFF intacte"). Chaque piste intacte de meme langue du
       maitre est comparee par empreinte (`measure_same_content`, seuil du
       regroupement gele). Correle -> course `mergeVideo.keep_best_audio`
       (l'autorite, maitre en COPIE), l'intacte gagne. Ne correle avec aucune
       -> AUTRE VERSION: gardee, taguee VMSAM_FABRICATED, jamais courue.
       Non mesurable -> pas de preuve d'une autre version: course.
       Sans piste intacte de la langue au maitre: gardee (objet de la
       reparation).

    `content_probe`: injectable pour les tests; par defaut `measure_same_content`.
    Renvoie la liste des retraits; journalise chaque decision
    INCONDITIONNELLEMENT (decision de livraison, pas du diagnostic).
    """
    import mergeVideo
    probe = content_probe or measure_same_content
    dropped = []
    master_intact = {}
    # Les adversaires: les pistes PRINCIPALES intactes (et l'audio-description).
    # Jamais un commentaire du maitre: ce n'est pas le meme contenu par nature.
    for holder in ("audios", "audiodesc"):
        for language, audios in (getattr(master_obj, holder, None) or {}).items():
            for audio in audios:
                if not fabricated_marker_of(audio):
                    master_intact.setdefault(language, []).append(audio)

    def say(line, to_stderr=False):
        tools.logs.append(line + "\n")
        if to_stderr:
            sys.stderr.write(line + "\n")

    for holder in AUDIO_HOLDERS:
        for language, audios in (getattr(repaired_obj, holder, None) or {}).items():
            for audio in audios:
                marker = fabricated_marker_of(audio)
                if not marker or not audio.get("keep", True):
                    continue
                where = (f"lang={language} holder={holder} "
                         f"stream={audio.get('StreamOrder')} "
                         f"format={audio.get('Format')} marker={marker}")
                if holder == "commentary":
                    carrier = []
                    if "commentary" in str(audio.get("Title", "")).lower():
                        carrier.append(f"title={audio.get('Title')}")
                    if (audio.get("properties") or {}).get("flag_commentary"):
                        carrier.append("flag_commentary=true")
                    say(f"repair: fabricated_kept cause=commentary_tagged {where} "
                        f"tagged_by={'+'.join(carrier) or 'unknown'} "
                        f"reason=a commentary is never raced against a main "
                        f"track; delivered with --commentary-flag")
                    continue
                opponents = master_intact.get(language, [])
                if not len(opponents):
                    say(f"repair: fabricated_kept cause=no_intact_master_track {where} "
                        f"reason=the master carries no intact {language} track "
                        f"to race it against")
                    continue
                lost_to = None
                measures = []
                for intact in opponents:
                    same, detail = probe(master_obj, intact, repaired_obj, audio, work_dir)
                    measures.append(f"vs_master_stream={intact.get('StreamOrder')}"
                                    f"[same_content={same} {detail}]")
                    if same is False:
                        continue
                    rival = dict(intact)
                    rival["keep"] = True
                    mergeVideo.keep_best_audio([rival, audio], {})
                    if not audio["keep"]:
                        lost_to = (intact, same)
                        break
                if lost_to is None:
                    say(f"repair: fabricated_kept cause=different_version {where} "
                        f"{' '.join(measures)} reason=its fingerprint matches no "
                        f"intact {language} master track: another version, "
                        f"delivered tagged VMSAM_FABRICATED")
                    continue
                intact, same = lost_to
                dropped.append({"kind": "audio", "holder": holder, "language": language,
                                "stream_order": audio.get("StreamOrder"),
                                "format": audio.get("Format"), "marker": marker,
                                "cause": "intact_same_language_wins",
                                "kept_master_stream": intact.get("StreamOrder"),
                                "same_content": same})
                say(f"repair: fabricated_dropped cause=intact_same_language_wins {where} "
                    f"kept_master_stream={intact.get('StreamOrder')} "
                    f"kept_master_format={intact.get('Format')} {' '.join(measures)} "
                    f"reason=same content (or unmeasured), raced by keep_best_audio, "
                    f"intact wins", to_stderr=True)
    return dropped


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
                      f"build={repair_log.build_sha()} "
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
                f" offset_ms={master_fill_offset(region)} "
                # `SPEC_ZONE_A.MD` s4e, ses propres mots: "where each filled
                # region came from -- same-language master, comparison
                # language, or silence". `from=`/`language` ci-dessus disent
                # DEJA la source et la langue; ce champ nomme laquelle des
                # DEUX cas MASTER c'est, sans obliger un lecteur a comparer
                # `from=` a la ligne `lang=` de la piste pour le deduire.
                f"fill_source_class={region.get('fill_source_class') or 'unreported'}"
                # LE JETON DU RAFFINEUR DE CADRES, QUAND IL A DECLINE -- OMIS,
                # PAS `n/a`, QUAND IL NE S'APPLIQUE PAS: ce champ est rare (la
                # plupart des regions ne touchent jamais le raffineur), et
                # l'imprimer partout ajouterait du bruit a chaque ligne pour
                # un cas qui presque jamais ne s'applique -- SA PRESENCE est
                # precisement ce que cette mission demande de rendre visible,
                # pas son apparition universelle.
                f"{' frame_tier_declined_reason=' + str(region['frame_tier_declined_reason']) if region.get('frame_tier_declined_reason') else ''}"
                # LE JETON SEUL NE DIT PAS LEQUEL DES QUATRE. Meme condition
                # que la ligne au-dessus -- il n'existe que quand le raffineur
                # a DECLINE -- mais `could_not_locate_onset` couvre quatre
                # etats distincts dans `frame_compare.locate_match_onset`, et
                # la phrase qui les separe etait deja calculee et jetee. Elle
                # porte les deux lignes de base et le seuil, donc "n'a pas pu
                # calibrer sur ce contenu" cesse de se lire comme "a calibre
                # et n'a rien trouve".
                f"{' frame_tier_declined_evidence=' + repr(str(region['frame_tier_declined_evidence'])) if region.get('frame_tier_declined_evidence') else ''}"
                "\n")
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

    Une fonction et non un dictionnaire en ligne: un dictionnaire construit en
    ligne dans une branche `except` ne se verifie qu'en faisant lever un vrai
    fichier, la ou une fonction se teste directement.

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
            "undelivered_path": getattr(error, "undelivered_path", None)}


def chimeric_cause(error):
    """Le jeton d'un `chimeric_error`, ou une SENTINELLE hors classe acceptee.

    23 sites levent `chimeric_error` et un 24e leve `chimeric_bound_error`, qui
    en est une SOUS-CLASSE et tombe donc sur le meme `isinstance` que les
    autres. CINQ portent un jeton aujourd'hui. Deux sur bornage explicite du
    Lead (R2): ce sont les deux que la production a fait tourner -- 18 et 5
    declins sur les 26 mesures dans 59 artefacts. Le troisieme est scope IN
    par l'Architect (ruling 2026-09-22, RULING_20260922_NO_BAND_ROUTING.MD,
    "RAISE SITE 1001 SCOPED INTO THE TOKENED SET"): `candidate_segment_regression`
    a `merge_video_chimeric.py:1001-1003`, premiere occurrence de production
    2026-09-22 (errid 25, wave table). Le quatrieme est scope IN par ce cas
    (`architect/cases/CASE_errid12_untokened_5367.md`, errid 12, wave table
    pass 8): `delivery_timeline_misalignment` a
    `merge_video_chimeric.py:5360-5397`, la verification post-construction
    contre la timeline du maitre, premiere occurrence de production
    2026-09-22/23. Le cinquieme est scope IN par ce cas
    (`architect/cases/CASE_errid50_untokened_1279.md`, errids 50 et 58, wave
    table pass 10): `candidate_admission_window_exceeded` a
    `merge_video_chimeric.py:1279-1283`, le seul site qui leve
    `chimeric_bound_error` -- la fenetre calculee d'un morceau sort de la
    duree propre du candidat (mesuree sur le flux, jamais un reglage),
    premieres occurrences de production 2026-09-24. Les 19 restants sont
    ATTEIGNABLES depuis le chemin de reparation (mesure statique, zero
    inatteignable) et ont ZERO occurrence en production.

    POURQUOI PAS UN JETON GROSSIER POUR LES 19. Un `assembly_refused` aurait
    rempli la colonne avec une valeur couvrant 19 decisions distinctes et n'en
    classant aucune. Une colonne remplie d'une valeur qui ne classe rien est
    PIRE qu'une colonne vide: elle a l'air notee. Ils sont donc NON COMPTES,
    et non FAUSSEMENT COMPTES.

    POURQUOI PAS `(unstated)`, QUI EXISTE DEJA. Cette sentinelle-la signifie
    *le producteur a tourne et n'a rendu AUCUN jeton alors que son contrat
    l'exige* -- une VIOLATION DE CONTRAT, et elle doit etre bruyante. Les 19
    sites ici sont un MANQUE CONNU, DELIBERE ET AUTORISE. Depenser le signal
    d'alarme sur 19 faux positifs detruit le sens du signal, et c'est ce
    signal-la qui protege la colonne. Deux etats qu'un correctif futur traite
    differemment ne partagent pas une etiquette: c'est la meme regle de
    granularite que pour les jetons, appliquee aux sentinelles.

    LES PARENTHESES SONT LE MECANISME, PAS DE LA PONCTUATION. Le lecteur
    accepte `cause=([A-Za-z0-9_]+)`; une parenthese ne peut pas satisfaire
    cette classe, donc la valeur est exclue PAR CONSTRUCTION et ne peut pas
    gonfler `excluded_with_stated_cause`. Ne pas "ranger" ces parentheses --
    voir le bloc en capitales plus haut, et `tools/check_cause_sentinel.py`,
    qui echoue si l'une ou l'autre sentinelle devient acceptable.

    LE NUMERO DE LIGNE VOYAGE AVEC LA SENTINELLE quand la trace le porte, si
    bien que la liste classee des 20 sites se lit dans les artefacts au lieu de
    demander une seconde mesure. Il est lu sur la trace de l'exception, donc il
    designe le site de LEVEE et pas ce site-ci. Absent, la sentinelle reste
    valide et simplement moins precise -- elle ne devient jamais un jeton.
    """
    cause = getattr(error, "cause", None)
    if cause != None:
        return cause
    line = None
    traceback_entry = getattr(error, "__traceback__", None)
    while traceback_entry != None:
        if traceback_entry.tb_frame.f_code.co_filename.endswith(
                "merge_video_chimeric.py"):
            line = traceback_entry.tb_lineno
        traceback_entry = traceback_entry.tb_next
    if line == None:
        return "(untokened_raise_site)"
    return f"(untokened_raise_site_{line})"


def detail_summary(detail):
    """Le NOYAU DECISIONNEL du `detail`, compact et SANS TABLEAU DE SONDES.

    POURQUOI CE RESUME EXISTE SEPAREMENT DU VIDAGE COMPLET, ET C'EST LE PIEGE
    DE LA PARTIE B. Le proprietaire a demande des informations "en mode
    tools.dev". MAIS L'INSTANCE DE TEST TOURNE `dev: false` -- donc tout ce qui
    est garde y est INVISIBLE, et une consigne suivie a la lettre aurait produit
    exactement l'inverse de ce qu'elle demandait: du diagnostic que le seul
    endroit ou on diagnostique ne voit pas.

    D'ou DEUX LIGNES ET NON UNE:

        ce resume            INCONDITIONNEL -- ce que le pipeline a DECIDE:
                             combien de pistes refusees par politique, quel
                             genre de plan, quel verdict, un controle de sortie
                             a-t-il leve, un artefact est-il reste non livre
        le vidage complet    GARDE par `tools.dev` -- les sondes, les tableaux
                             de verification, les valeurs par flux: du materiau
                             qui fait arriver PLUS VITE a une conclusion deja
                             correcte

    La regle du brief, appliquee ligne a ligne: si un lecteur en tire une
    conclusion differente sur le fait que le pipeline a eu RAISON, c'est une
    DECISION. `declined=3` sur des pistes est une decision -- trois pistes ont
    ete ecartees par politique. `probes=[...]` ne l'est pas: il dit COMMENT on
    l'a su, pas CE QU'ON A FAIT.

    RIEN DE CE QUI EST GARDE N'EST PORTEUR: cette fonction ne lit aucune valeur
    calculee sous un `if tools.dev`, et personne ne lit son resultat -- il est
    ecrit dans le journal et rien d'autre ne le consulte.
    """
    if not detail:
        return ""
    fields = []
    for key in ("plan_kind", "verdict", "plan_source", "marker",
                "undelivered_state"):
        value = detail.get(key)
        if value != None:
            fields.append(f"{key}={value}")
    for key in ("audios", "subtitles", "declined", "failed", "coarse_brackets",
                "fabricated_dropped"):
        value = detail.get(key)
        # UN COMPTE, PAS LE CONTENU. `len` sur une liste de pistes ecartees est
        # la decision; la liste elle-meme est du diagnostic.
        if isinstance(value, (list, tuple)):
            fields.append(f"{key}={len(value)}")
    for key in ("output_check", "verification"):
        value = detail.get(key)
        # PRESENT / ABSENT, ET C'EST UNE DECISION: un controle de sortie qui a
        # LEVE et un controle qui n'a jamais tourne sont deux etats differents,
        # et de l'exterieur ils produisaient le meme silence.
        if value != None:
            fields.append(f"{key}=present")
    return " ".join(fields)


def record(candidate_path, outcome, reason, detail=None, cause=None):
    entry = {"candidate": candidate_path, "outcome": outcome, "reason": reason,
             "detail": detail, "cause": cause}
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
    # THE CAUSE TOKEN GOES BEFORE THE PATH, AND THAT POSITION IS THE WHOLE POINT.
    # Found by `vmsam-dev-4` against its own reader, on a fixture it built before
    # any real material arrived. REPRODUCED HERE BEFORE CHANGING ANYTHING:
    #
    #   path  /srv/<...>/S01: cause=forged_token: E01.mkv
    #   line  repair: declined for <that path>: the produced file does not match
    #   dev-4's hardened parse extracts   cause=forged_token
    #
    # A COLON IS LEGAL IN A FILENAME, so with the token AFTER the path no
    # delimiter closes this: a path can contain any byte but `/` and NUL, and NUL
    # cannot travel in a log line. NO DELIMITER IS SAFE -- ONLY POSITION IS.
    #
    # So the token is now parsed from a PREFIX that ends before the first byte of
    # attacker-controlled text:
    #
    #   repair: <outcome> cause=<token> for <path>: <prose>
    #   ^--------- fixed, closed vocabularies ---------^
    #
    # WHY IT MATTERS MORE THAN A PARSER BUG: a forged token inflates
    # `excluded_with_stated_cause`, WHICH IS THE COLUMN THE OWNER'S END CONDITION
    # IS SCORED ON, and it fabricates a cause NO PRODUCER EMITTED -- my own
    # acceptance R2 arriving on the consumer's side of the same interface.
    # It is silent, and it is unfalsifiable from a corpus holding no real tokens.
    head = f"repair: {outcome}"
    if cause != None:
        head += f" cause={cause}"
    # ROUTED THROUGH `tools.log_always` (owner's order via the Lead,
    # 2026-09-22, wave 3b) -- THE terminal per-candidate verdict, called for
    # every outcome (no_plan/declined/repaired/failed). Deliberately
    # unconditional since the comment two paragraphs up this function
    # explains why (`record()` call is by definition an outcome worth
    # recording); it just never had a stderr half before now.
    tools.log_always(f"{head} for {candidate_path}: {reason}\n")
    # LE `detail` N'ATTEIGNAIT AUCUN ARTEFACT. Mesure, `grep -rn` sur tout
    # `src/`: `last_repair_report` a TROIS occurrences -- sa definition, cet
    # `append`, et le `del` qui le vide au debut de chaque passage. PERSONNE NE
    # LE LIT. Tout ce que `decline_detail` rassemble -- le rapport de controle
    # de sortie, les resultats de verification, l'etat de non-livraison -- etait
    # calcule, empaquete, et jete. C'est la classe de defaut que `AGENT.MD`
    # nomme: ecrit, relu, teste vert, et jamais appele; le seul symptome est que
    # rien ne change.
    #
    # PREFIXE DISTINCT, ET CE N'EST PAS COSMETIQUE. Le jeton est lu sur un
    # PREFIXE A VOCABULAIRE FERME `repair: <outcome> cause=<jeton> for `. Une
    # ligne `repair: detail ... for <chemin>` serait lue par ce parseur comme un
    # OUTCOME nomme `detail` et polluerait le decompte. `repair_detail:` ne peut
    # pas entrer dans cette forme.
    #
    # BORNE A `declined` ET `failed`, qui est exactement le perimetre de
    # l'acceptation pre-enregistree. `repaired` (site 2277) n'est pas touche:
    # changer ce qu'une ligne `repaired` donne a lire changerait une interface
    # que d'autres lecteurs analysent deja, et ce n'est pas ma decision.
    if outcome in ("declined", "failed") and detail:
        summary = detail_summary(detail)
        if summary:
            # DECISION -> INCONDITIONNEL. Visible sur l'instance de test, qui
            # tourne `dev: false`.
            tools.logs.append(f"repair_detail: {outcome} {summary}\n")
        if tools.dev:
            # DIAGNOSTIC -> GARDE. Les sondes et les tableaux par flux, qui font
            # gagner du temps sur une conclusion deja atteignable sans eux.
            # `default=str` parce que le plan porte des `Decimal`: sans lui la
            # serialisation LEVE, et une levee dans un enregistrement de refus
            # remplacerait le refus par une panne.
            try:
                dump = json.dumps(detail, default=str, sort_keys=True)
            except Exception as error:
                dump = f"<undumpable: {type(error).__name__}: {error}>"
            # ROUTED THROUGH `tools.dev_log` (owner's order via the Lead,
            # 2026-09-22, wave 3): same defect class as the other three
            # sites in this file, format unchanged.
            tools.dev_log(f"repair_detail_verbose: {outcome} {dump}\n")
    return entry


def master_intertrack_verdict(best_video, language, cache):
    """STEP 1 (CLASSIFICATION), AND IT LOOKS AT THE MASTER BEFORE IT LOOKS AT
    ANY CANDIDATE. `RULING_20260922_MASTER_INTERTRACK_ADMISSION.MD` + ADDENDUM 2.

    Le defaut mesure (dossier 86): le maitre ne s'accorde pas avec LUI-MEME.
    Ses deux pistes `ja` portent le meme contenu decale de ~131 ms, et
    `change_point_locator.py:1800` prend `master_streams[0]` SANS JAMAIS
    comparer avec `master_streams[1:]`. Onze fichiers ont donc ete refuses pour
    un defaut DU MAITRE, avec une cause qui designait le candidat.

    UNE SEULE FOIS PAR (maitre, langue de comparaison) -- `cache` est ce
    "une seule fois", et il est passe en parametre plutot que garde en global
    parce qu'un etat de module survivrait entre deux appels de
    `repair_not_compatible_videos` sur DEUX MAITRES DIFFERENTS et rendrait le
    verdict du premier au second.

    POURQUOI ICI ET PAS LITTERALEMENT AVANT LA BOUCLE. La langue de comparaison
    est celle que l'appelant passe (ADDENDUM 20: celle de get_delay); la mesure
    reste paresseuse et cachee par langue. Enumerer d'autres langues du maitre
    serait exactement ce que
    l'ADDENDUM 1 interdit ("Never enumerate or probe other languages' track
    pairs"). Le cache donne la propriete que la regle demande vraiment: la
    mesure tourne AU PLUS UNE FOIS par langue, et son verdict precede TOUT
    travail de reparation sur cette langue (appelee avant
    `get_plan_from_locator`, donc avant tout test de vitesse et tout chimerique).

    IMPORT TARDIF ET TOLERANT, meme discipline que `get_plan_from_locator`:
    une capacite dont le module n'est pas deploye n'a pas le droit de casser un
    merge. Un module absent = aucun verdict = comportement d'aujourd'hui.

    Renvoie le dict de verdict de `master_self_check`, ou None quand rien n'a
    pu etre mesure. `None` veut dire *je n'ai pas mesure*, jamais *le maitre
    est sain* -- la meme distinction que la docstring de ce module pose pour
    `change_point_locator`.
    """
    if language in cache:
        return cache[language]
    verdict = None
    try:
        import master_self_check
    except Exception as error:
        tools.dev_log(f"repair: no master_self_check module: {error}\n")
    else:
        tools.dev_log(f"repair: master_intertrack_verdict starting on "
                      f"master={best_video.filePath} language={language}\n")
        try:
            verdict = master_self_check.check_master_intertrack(
                best_video, language)
        except Exception as error:
            # UNE LEVEE DANS UN CONTROLE N'EST PAS UN VERDICT. Elle ne doit ni
            # refuser le fichier (on n'a rien mesure) ni disparaitre en
            # silence. On la nomme et on laisse la chaine continuer comme
            # aujourd'hui.
            tools.dev_log(f"repair: master_intertrack_verdict raised "
                          f"{type(error).__name__}: {error} -- no verdict, the "
                          f"chain continues unchanged\n")
            verdict = None
    cache[language] = verdict
    return verdict


def _drain_audio_pools(objs, site):
    '''STOPGAP, not the fix: removes the fast-decline trigger of the frozen
    fusion.py Pool.terminate() deadlock (CASE id 6) -- the fix is the owner's
    initializer in fusion.py; remove this note when that lands.

    Called immediately before a fast `continue` in
    `repair_not_compatible_videos`. `test_if_constant_good_delay`'s `except`
    (`mergeVideo.py:264-267`) fires ~40 unawaited `apply_async` ffmpeg audio
    extractions on both video objects before re-raising; nobody drains them
    on the fast paths, so they are still in-flight when `fusion.py:399`
    `Pool.terminate()`s a busy worker that swallows SIGTERM and never exits.
    Draining here empties the pool before that teardown runs. `objs` may
    contain `None` (e.g. `candidate_obj` before it exists) -- skipped. A
    drain failure is logged and swallowed: it must not convert a clean
    decline into a crash.
    '''
    for obj in objs:
        if obj is None:
            continue
        try:
            obj.wait_end_ffmpeg_progress_audio()
        except Exception as error:
            tools.dev_log(f"repair: draining pending audio extraction at "
                          f"{site} raised {type(error).__name__}: {error}\n")


def retire_ffmpeg_pools(grace_seconds=300):
    '''STOPGAP for CASE id 6, called only when the merge is over (see the call
    site in mergeVideo.py zone A). close() queues ONE sentinel per worker;
    join() lets each worker read its own and exit. No signal is involved, so
    nothing can swallow it. SIGKILL is the backstop for a worker still stuck in
    ffmpeg past the grace period -- SIGTERM is the signal that does not work
    here, which is the whole case. Never calls terminate(): terminate() IS the
    deadlock.

    POURQUOI PAS UN DRAIN (`_drain_audio_pools` ci-dessus, qui reste en place
    pour ses autres sites d'appel). Mesure, appendice A du dossier: le drain
    attend les `ApplyResult`, ce qui rend le pool SILENCIEUX mais pas MORT, et
    le verrou que `Pool._help_stuff_finish` prend sur `inqueue._rlock` -- sans
    jamais le rendre -- se referme quand meme sur un worker qui se trouve
    ENTRE deux taches. 10 blocages sur 80 executions drainees, contre 0 sur 20
    avec cette retraite-ci, teardown 0.3 ms. La retraite SUBSUME le drain:
    `join()` attend le travail en vol par construction.

    CE QUE CETTE FONCTION EXIGE DE SON APPELANT, et qui n'est pas verifiable
    d'ici: que PLUS RIEN ne reutilise les pools ensuite. Elle les laisse
    FERMES, donc un `apply_async` posterieur leve `ValueError: Pool not
    running` -- bruyamment, jamais en silence. Le seul site d'appel est la
    condition de mergeVideo.py zone A qui dit deja "il ne reste pas de quoi
    fusionner": fusion.py:391 et main.py:81 n'appellent `merge_videos` qu'une
    fois et detruisent les pools juste apres. `main_gestionar_show.process_episode`
    (fige) REESSAIE au contraire des fusions apres cette levee, via
    `process_rejected_files:56`, sur les memes pools -- constat porte au
    dossier, decision du proprietaire.
    '''
    import multiprocessing.pool, video, signal, time, os
    for name in ("ffmpeg_pool_audio_convert", "ffmpeg_pool_big_job"):
        pool = getattr(video, name, None)
        if pool == None:
            continue
        try:
            pool.close()
            deadline = time.monotonic() + grace_seconds
            while (time.monotonic() < deadline
                   and any(p.is_alive() for p in pool._pool)):
                time.sleep(0.1)
            survivors = [p for p in pool._pool if p.is_alive()]
            if len(survivors):
                # LE BACKSTOP SIGKILL NE PEUT PAS SE CONTENTER DE TUER -- MESURE,
                # ET C'EST LE MEME PIEGE QUE CELUI QU'ON SOIGNE. Un worker tue
                # laisse SON `ApplyResult` dans `pool._cache` POUR TOUJOURS
                # (CPython n'a aucune detection de mort de worker). Or:
                #   * `_handle_workers` boucle tant que `cache` n'est pas vide
                #     et REMPLACE les workers disparus (`_repopulate_pool_static`),
                #     donc tuer en fabrique aussitot d'autres, gares sur
                #     `inqueue.get()`;
                #   * `_handle_results` boucle `while cache and state != TERMINATE`
                #     sur un `get()` qui ne rendra plus jamais rien.
                # `pool.join()` joint ces DEUX threads: il ne revenait jamais.
                # Mesure de ce site, grace 2 s contre des taches de 30 s:
                # 5 blocages sur 5 avant ces trois lignes, 0 sur 5 apres, tous
                # les workers morts. On pose les deux etats AVANT de tuer, pour
                # qu'aucun remplacant ne naisse entre le kill et le join.
                #
                # CE N'EST PAS `terminate()`: on ne touche ni a
                # `_help_stuff_finish` ni au verrou `inqueue._rlock` qu'il prend
                # sans le rendre -- c'est lui, le blocage du dossier.
                #
                # ECART ASSUME PAR RAPPORT AU BROUILLON A.6 DU DOSSIER, qui
                # n'avait mesure que le chemin ou le travail se termine tout
                # seul (11 s, 0/20). Le chemin SIGKILL, lui, est atteignable en
                # production: 20 taches de trois passes ffmpeg sur des sources
                # de 580 Mo / 1.2 Go peuvent depasser 300 s.
                pool._worker_handler._state = multiprocessing.pool.TERMINATE
                pool._result_handler._state = multiprocessing.pool.TERMINATE
                pool._change_notifier.put(None)
                for p in survivors:
                    tools.log_always(f"repair: retiring {name}: worker {p.pid} "
                                     f"outlived {grace_seconds}s, SIGKILL\n")
                    os.kill(p.pid, signal.SIGKILL)
            pool.join()
            if len(survivors):
                # ET LE `terminate()` DE fusion.py DOIT REDEVENIR UN NON-EVENEMENT,
                # ce qui est tout l'objet de cette fonction. `_terminate_pool`
                # commence par `if cache and result_handler not alive: raise
                # AssertionError` -- mesure: la levee sortait bien du `terminate()`
                # de fusion.py:399, ou elle est rattrapee ("Error close pool"),
                # mais elle SAUTE le `terminate()` du second pool au passage.
                # Les entrees restantes sont celles des taches tuees; leur
                # `ApplyResult` ne sera jamais rendu ni attendu (la fusion est
                # finie, c'est la condition meme du site d'appel).
                pool._cache.clear()
        except Exception as error:
            tools.dev_log(f"repair: retiring {name} raised "
                          f"{type(error).__name__}: {error}\n")


# ---------------------------------------------------------------------------
# THE SEAM WITH STAGE 5 (plan application), DEFINED HERE BECAUSE THIS ENTRY IS
# ITS ONLY READER.
#
# The owner's `repair()` returns a BOOLEAN (RULING_20260922_ORCHESTRATOR_
# ARCHITECTURE.MD, ADDENDUM point 4: "True = plan trouve ET fichier temporaire
# chimerique cree avec succes"), so the repaired video object cannot travel in
# the return value. It travels in ONE dict this entry hangs on the candidate
# object BEFORE calling `repair()`, and reads back AFTER it returns:
#
#   getattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE) == {
#       "job_start_utc": <ISO-8601 UTC str>   IN  -- written here; VMSAM_ERA
#                                                   (Architect's ruling
#                                                   2026-09-16: the start of
#                                                   the JOB on this candidate)
#       "repaired_obj":  None                 OUT -- stage 5's `apply_plan`
#                                                   sets it to the repaired
#                                                   video object, built to the
#                                                   three requirements of this
#                                                   module's docstring
#                                                   (`delay_same_md5_audio =
#                                                   Decimal('0')`, mediadata
#                                                   read, temp file under
#                                                   `tools.tmpFolder/repair/`)
#                                                   -- exactly what
#                                                   `build_repaired_video_object`
#                                                   returns today
#       "assembly":      None                 OUT -- the assembly report beside
#                                                   it, for stage 5's own
#                                                   `record(..., "repaired",
#                                                   reason, detail)` terminal
#   }
#
# CONTRACT: `repair()` returns True ONLY when `repaired_obj` is set. A True
# with no object is a broken seam and is recorded `failed`
# (cause=repaired_object_missing) -- never attached, never silent. The dict is
# removed after the call, whatever the outcome, so no object can outlive the
# call that produced it and be attached by a later one. `apply_plan` must
# tolerate the attribute's ABSENCE (the orchestrator driven standalone, without
# this entry): it then has no job start to stamp and must say so, not invent
# one. The `repaired` terminal line is stage 5's to write through `record()`;
# this entry writes none, so one repair never reads as two.
# ---------------------------------------------------------------------------
REPAIR_SEAM_ATTRIBUTE = "vmsam_repair_seam"


def _open_repair_seam(candidate_obj, job_start_utc):
    seam = {"job_start_utc": job_start_utc, "repaired_obj": None, "assembly": None}
    setattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE, seam)
    return seam


def _close_repair_seam(candidate_obj):
    seam = getattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE, None)
    if seam is not None:
        delattr(candidate_obj, REPAIR_SEAM_ATTRIBUTE)
    return seam or {}


def _terminal_cause_since(candidate_path, reported_before):
    """The cause token the orchestrator's terminal carried for THIS candidate.

    `repair_orchestrator._terminal` routes every refusal through `record()`
    above, which appends to `last_repair_report`; reading it back here is how
    the boolean's reason reaches the drain site without a second channel. None
    when no terminal was recorded (a True, or an orchestrator that returned
    False without recording -- which its own contract forbids)."""
    for entry in reversed(last_repair_report[reported_before:]):
        if entry.get("candidate") == candidate_path:
            return entry.get("cause")
    return None


def repair_not_compatible_videos(list_not_compatible_video, dict_file_path_obj,
                                 best_video, language):
    """Point d'entree appele depuis la zone A. LA CHAINE EST L'ORCHESTRATEUR.

    Owner, 2026-09-24 (RULING_20260922_ORCHESTRATOR_ARCHITECTURE.MD ADDENDUM 8
    points 4 et 6): remplacement direct, sans drapeau de cohabitation --
    `repair_orchestrator.repair()` par candidat, et l'ancienne chaine
    (`get_plan_from_locator`, le routage par bande, `change_point_locator`)
    est partie dans le meme lot. L'etage 5 (application du plan,
    `repair_orchestrator.apply_plan`) accroche l'objet repare a la couture et
    ecrit lui-meme le terminal `repaired`.

    Renvoie la liste des chemins effectivement repares et raccroches. Les
    fichiers restent retires de `dict_file_path_obj` par la zone A dans tous les
    cas: l'objet repare rejoint le merge par
    `best_video.sameAudioMD5UseForCalculation`, consomme par
    `generate_merge_command_common_md5`, qui ne passe jamais par la machinerie
    de delai.

    LA LANGUE DE COMPARAISON EST PASSEE, JAMAIS REDEVINEE (owner, ADDENDUM 20,
    2026-09-24): `language` est la langue sur laquelle `mergeVideo.get_delay` /
    `get_best_video` ont mesure le delai (la cle normalisee ja/en/fr des objets
    video), transmise par `remove_not_compatible_video`. La re-deduction qui
    vivait ici (`get_delay_language`, relue sur `best_video.delays`) faisait
    comparer a l'orchestrateur D'AUTRES pistes que celles de get_delay (famille
    errid-244); elle est partie, avec son refus `language_undetermined` -- la
    garde vit chez l'appelant, qui ne mesure aucun delai sans langue.

    CE QUI RESTE ICI, ET POURQUOI: le refus que l'orchestrateur ne peut pas voir
    (pas d'objet pour le chemin refuse), le cache du maitre par langue
    (une mesure du maitre par (maitre, langue), passe a `repair()` parce qu'un
    etat de module survivrait entre deux maitres), le drain des pools audio
    (CASE id 6), et la couture avec l'etage 5 (`REPAIR_SEAM_ATTRIBUTE`).
    Chaque candidat laisse une ligne `repair: plan ` (la seule chose que
    `merge_plan_report.is_job_log` teste) et un terminal `record()`.
    """
    import repair_orchestrator
    import merge_video_chimeric
    del last_repair_report[:]
    work_root = path.join(tools.tmpFolder, "repair")
    tools.make_dirs(work_root)
    repaired = []
    # UNE SEULE MESURE DU MAITRE PAR LANGUE DE COMPARAISON, et ce dictionnaire
    # EST ce "une seule": `repair()` le recoit et le passe a
    # `master_intertrack_verdict`. Il vit ici, par appel, parce qu'un appel de
    # cette fonction = un maitre.
    master_intertrack_by_language = {}

    for candidate_path in list_not_compatible_video:
        # VMSAM_ERA (Architect's ruling, 2026-09-16): capture ICI, avant tout
        # declin, parce que c'est le debut du JOB sur CE candidat. Voyage vers
        # l'etage 5 par la couture.
        job_start_utc = datetime.now(timezone.utc).isoformat()
        # WHICH FILE, BEFORE ANY WORK ON IT (owner's decision, 2026-09-22, on a
        # real 7-hour hang whose last line named no file). The orchestrator
        # logs a launch line per step as well; this one precedes even the
        # object lookup.
        tools.dev_log(f"repair: repair_not_compatible_videos starting on "
                      f"{candidate_path}\n")
        candidate_obj = dict_file_path_obj.get(candidate_path)
        if candidate_obj == None:
            # La zone A a refuse un chemin dont elle n'a jamais porte l'objet.
            repair_orchestrator._plan_line("none", candidate_path, step="entry",
                                           cause="candidate_object_absent")
            record(candidate_path, "declined",
                   "the rejected path has no video object in dict_file_path_obj",
                   cause="candidate_object_absent")
            _drain_audio_pools((best_video, candidate_obj),
                               "candidate_object_absent")
            continue
        tools.dev_log(f"repair: comparison language={language} "
                      f"route=passed_by_the_caller(get_delay) for {candidate_path}\n")

        _open_repair_seam(candidate_obj, job_start_utc)
        reported_before = len(last_repair_report)
        try:
            ok = repair_orchestrator.repair(
                best_video, candidate_obj, language,
                work_root=path.join(work_root,
                                    merge_video_chimeric.stable_case_key(candidate_path),
                                    "orchestrator"),
                master_intertrack_cache=master_intertrack_by_language)
        except Exception as error:
            _close_repair_seam(candidate_obj)
            # UNE LEVEE N'EST PAS UN VERDICT, ET ELLE NE DOIT PAS EMPORTER LES
            # CANDIDATS SUIVANTS: la zone A l'attraperait ("The repair raised
            # and was abandoned") et abandonnerait toute la liste. Un refus de
            # l'assemblage (`chimeric_error`, l'etage 5) est un DECLIN avec son
            # jeton pose au site de levee; toute autre levee est une PANNE, avec
            # sa classe dans la prose et un jeton fixe.
            if isinstance(error, tools.decoder_timeout):
                # ADDENDUM 26.3: a decoder past its bound, anywhere in the build, is a NAMED
                # decline -- the file comes back next wave; never a failure of the repair.
                cause = "decoder_timeout"
                repair_orchestrator.log_measurement_class(candidate_path, cause)
                record(candidate_path, "declined", str(error), cause=cause)
            elif isinstance(error, merge_video_chimeric.chimeric_error):
                cause = chimeric_cause(error)
                # B5: THE MEASUREMENT CLASS TRAVELS WITH AN ASSEMBLY REFUSAL TOO.
                repair_orchestrator.log_measurement_class(candidate_path, cause)
                record(candidate_path, "declined", str(error),
                       decline_detail(error), cause=cause)
            else:
                cause = "repair_raised_unhandled"
                record(candidate_path, "failed",
                       f"{type(error).__name__}: {error}", decline_detail(error),
                       cause=cause)
            repair_orchestrator._plan_line("none", candidate_path,
                                           step="orchestrator_raised", cause=cause)
            sys.stderr.write(f"repair: {cause} for {candidate_path}: {error}\n")
            _drain_audio_pools((best_video, candidate_obj), cause)
            continue
        seam = _close_repair_seam(candidate_obj)
        if not ok:
            # Le refus est deja journalise par l'orchestrateur (ligne
            # `repair: plan none ...`, puis `record()` avec son jeton et sa
            # classe de mesure). Il reste le drain, sur TOUT refus et plus
            # seulement sur les trois refus rapides de l'ancienne chaine: un
            # drain sans travail en vol ne coute rien, et le refus le plus
            # rapide -- `master_intertrack_desync`, l'etape 1 -- est justement
            # celui qui a bloque 4 fois sur 3 SHA (CASE id 6).
            cause = _terminal_cause_since(candidate_path, reported_before)
            _drain_audio_pools((best_video, candidate_obj),
                               cause or "orchestrator_declined")
            # FALSIFIEUR (appendice A.7 du dossier id 6): la derniere ligne
            # avant un eventuel silence dit si le drain a rendu la main.
            # `log_always`, parce qu'une ligne qui sert a lire un blocage ne peut
            # pas dependre de `tools.dev`. Meme forme qu'avant la bascule pour
            # `master_intertrack_desync`.
            tools.log_always(f"repair: drain complete at {cause} "
                             f"for {candidate_path}\n")
            continue
        repaired_obj = seam.get("repaired_obj")
        if repaired_obj == None:
            record(candidate_path, "failed",
                   f"the orchestrator returned True but handed over no repaired "
                   f"object through `{REPAIR_SEAM_ATTRIBUTE}['repaired_obj']`: "
                   f"the seam with plan application is broken, nothing is "
                   f"attached and the refusal stands",
                   cause="repaired_object_missing")
            continue
        sys.stdout.write(f"\tRepaired {candidate_path} as "
                         f"{getattr(repaired_obj, 'filePath', repaired_obj)}\n")
        # DERNIER, ET ADJACENT: l'accrochage a `best_video` est la toute
        # derniere instruction parce que c'est la seule que l'appelant peut voir
        # apres une levee.
        repaired.append(candidate_path)
        best_video.sameAudioMD5UseForCalculation.append(repaired_obj)
    return repaired
