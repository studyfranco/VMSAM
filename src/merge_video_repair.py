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

from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
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

# LES TROIS CAUSES QUI DECLENCHENT LE PRODUCTEUR DE STEP 2 (Lead
# authorization, 2026-09-21, sur une mesure de population:
# `dev-step1-classify` a mesure 0/20 des identifiants PAL reels confirmes de
# la campagne atteignant `speed_relation_suspected` -- 12/20 declinent en
# `median_fidelity_below_floor`, 7/20 en `offsets_scattered`, la derive de
# 4.27% de PAL etant assez grande pour transformer la correlation en bruit
# A L'INTERIEUR D'UNE SEULE FENETRE non corrigee, et le bruit n'est pas
# monotone). `confirm_speed_relation_via_resample` ne depend PAS du test de
# monotonicite de la Stage 1: il redérive son propre ratio depuis la DUREE
# seule et reechantillonne LA PISTE ENTIERE avant de sonder -- la derive
# intra-fenetre qui defait la Stage 1 est deja corrigee au moment ou ce
# module mesure. D'ou: essayer ce producteur sur ces trois causes n'est pas
# deux sieges qui rafistolent le meme symptome; si la Stage 1 repare un jour
# son test de monotonicite, `speed_relation_suspected` devient simplement une
# troisieme entree a cote des deux autres, additive et non conflictuelle.
# ON N'OUVRE QUE LA PORTE QU'ON A MESUREE. Le bras de faux positifs du siege
# a tourne sur TROIS residents reels de `median_fidelity_below_floor` (meme
# frequence d'image des deux cotes, donc aucune relation de vitesse possible;
# le confirmateur a essaye LES DEUX hypotheses et decline proprement au seuil
# lui-meme, fidelites 0.5631-0.6107, jamais un delai ni un plantage a une
# etape ulterieure): ZERO faux positif, n=3, 2026-09-21.
# `offsets_scattered` N'EST PAS DANS CET ENSEMBLE, et son absence est une
# mesure et non un oubli: sur ~20 paires reelles depouillees, AUCUN resident
# reel de ce seau-la n'est apparu -- les paires same-fps de l'arbre d'erreurs
# sont surtout de vraies coupes, pas des declins de basse fidelite. Un bras
# mesure sur un seau n'autorise pas l'autre (INSTRUMENT SCOPE LAW appliquee a
# une PORTE): la population franchissant `offsets_scattered` n'a jamais ete
# observee, donc le taux de faux positifs qu'elle produirait est inconnu.
# 7 des 20 PAL confirmes de la campagne declinent par ce jeton, donc la porte
# VAUT d'etre ouverte -- quand elle aura son propre nombre.
SPEED_CONFIRMER_ENTRY_CAUSES = frozenset(
    {"speed_relation_suspected", "median_fidelity_below_floor"})


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

    REND `(ratio, refus, jeton)`. Le troisieme element est un jeton stable en
    snake_case, produit ICI -- a la decision -- et non au site de journalisation.

    ONZE REFUS DISTINCTS PASSENT PAR CETTE FONCTION, et non neuf: quatre sont
    rendus directement ci-dessous, un vient de `check_ratio_convention` et
    TROIS de `check_ratio_labelled`, puis trois autres directement. Un appelant
    ne peut pas les distinguer -- il ne voit qu'une prose libre -- donc un seul
    jeton pose chez lui aurait effondre onze decisions en une. Mesure
    dev-cause 2026-09-15, confirmee par le Lead; le compte de neuf qui
    circulait comptait les `return` de cette fonction, ce qui est exact et
    n'est pas la meme quantite.
    """
    if plan.get("kind") != "speed" and plan.get("speed_ratio") == None:
        return None, None, None
    verdict = plan.get("verdict")
    if verdict == None:
        return None, ("the measurement carries a speed ratio but no verdict; "
                      "AUDIO_SPEED_POLICY.MD requires three outcomes and a decline, "
                      "and applying asetrate on a bare coefficient would let an "
                      "inverting case through undetected"), "speed_verdict_absent"
    if verdict == "leave_alone":
        # PAS DE CORRECTIF A FAIRE SUR CETTE POPULATION, et le jeton doit le
        # dire: la paire va bien. Un seat futur qui compte les fichiers
        # recuperables doit pouvoir SOUSTRAIRE ceux-ci, pas les empiler avec
        # des refus qui attendent un outil.
        return None, ("the measurement says LEAVE IT ALONE: the pair already "
                      "matches and a correction would take it apart"
                      ), "speed_verdict_leave_alone"
    if verdict == "decline":
        return (None, "the measurement declined to name a transformation",
                "speed_verdict_declined")
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
                      "the same correlations is not a third opinion"
                      ), "speed_hypotheses_indeterminate"
    # LA CONVENTION DU RAPPORT, VERIFIEE CONTRE LES DUREES ET NON CONTRE UN NOM.
    #
    # ATTRIBUTION CORRECTED 2026-09-05, AND THE DEFECT IT RECORDS IS UNCHANGED.
    # The line below named `vmsam-dev-1` as having emitted `speed_ratio` in the
    # reciprocal convention. MEASURED AT THE AUTHORITY, with a control:
    #     "speed_ratio" in change_point_locator.py   0
    #     control "quantum_ms"                      12   (the grep fires)
    #     the plan dict it returns carries 25 keys, and NONE is speed_ratio.
    # SO THE ATTRIBUTION NAMED A PRODUCER THAT DOES NOT EXIST HERE -- the same
    # class as the four margin keys documented below, and the same class dev-4
    # filed and corrected to NO WRITER EXISTS. The CONVENTION statement that is
    # correct lives in merge_video_resample.py's own docstring, as an equation.
    #
    # THE HAZARD BELOW IS REAL AND STAYS: two reciprocal conventions for one
    # name, where taking the wrong one stretches a track the wrong way by 8.7 %
    # and nothing inside a sweep can catch it.
    # (historical, kept per the append-only rule:)
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
    # LES JETONS DES DEUX AIDES PASSENT INCHANGES. Ni traduits, ni normalises,
    # et aucun jeton a moi ajoute a cote: c'est la regle que
    # `get_plan_from_locator` applique au producteur de plans, et elle vaut
    # entre deux fonctions du meme fichier pour la meme raison -- celui qui
    # decide nomme, celui qui transporte se tait.
    convention, convention_cause = check_ratio_convention(plan)
    if convention != None:
        return None, convention, convention_cause
    labelled, labelled_cause = check_ratio_labelled(plan)
    if labelled != None:
        return None, labelled, labelled_cause
    if verdict == "rubberband":
        return None, ("the measurement says rubberband -- the inverting case, a "
                      "source already pitch-corrected at origin. Not implemented: "
                      "applying asetrate here would drag the pitch 72.4 cents flat"
                      ), "speed_verdict_rubberband_unimplemented"
    if verdict != "asetrate":
        # LE VERDICT INCONNU VA DANS LA PROSE, LE JETON RESTE FIXE. Regle
        # lexicale 4: un jeton qui porterait `verdict` varierait a chaque
        # valeur inattendue et ne compterait rien.
        return None, f"unknown speed verdict {verdict!r}", "speed_verdict_unknown"
    ratio = plan.get("speed_ratio")
    if ratio == None:
        return (None, "verdict asetrate with no speed_ratio",
                "speed_ratio_absent_for_asetrate")
    return Decimal(str(ratio)), None, None


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

    CORRIGE 2026-09-22 -- DEFAUT TROUVE EN REVUE, PAS EN PRODUCTION: cette
    fonction ne regardait qu'UNE piste -- la premiere trouvee en iterant
    `candidate_obj.audios` seul (ordre d'insertion du dict), IGNORANT
    `.audiodesc`/`.commentary` alors que `iterate_candidate_audios`
    (`merge_video_chimeric.py`, la fonction qui decide reellement ce qui est
    reconstruit) couvre les trois, commentaire compris ("on repare toujours",
    proprietaire 2026-09-16). Et `build_speed_filter_chain`'s facteur EFFECTIF
    depend de la frequence SOURCE (l'arrondi de `asetrate` sur un entier),
    donc deux pistes a des frequences differentes recoivent des facteurs
    EFFECTIFS mesurablement differents (docstring de ce module: 0.35 a
    2.75 ms d'ecart sur 1435 s) -- alors que cette fonction en ecrivait UN
    SEUL, applique IDENTIQUEMENT a chaque piste par `mux_repaired_file`.
    `build_resampled_candidate` (`merge_video_resample.py`) mesure deja ce cas
    et le nomme `"mixed"` (:177-180 de ce fichier); cette fonction ne le
    faisait pas -- la connaissance existait, un seul appelant l'ignorait. Un
    tag faux est "pire que pas de tag du tout" (docstring de ce module, en
    tete): donc mesurer TOUTES les pistes, et REFUSER plutot que deviner
    quand elles ne s'accordent pas -- un DECLIN mesure (`chimeric_error`),
    pas une PANNE d'outil, meme raisonnement que le reste de ce fichier
    (repair.py:2798-2829: `chimeric_error` -> `declined`, tout le reste ->
    `failed`).
    """
    applied_factor = None
    if speed_ratio != None:
        import merge_video_chimeric
        import merge_video_resample
        factors_by_rate = {}
        for audio in merge_video_resample.iter_audio_dicts(candidate_obj):
            rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
            if rate == None:
                continue
            rate = int(float(rate))
            if rate not in factors_by_rate:
                _, applied, _, _ = merge_video_resample.build_speed_filter_chain(
                    rate, speed_ratio)
                factors_by_rate[rate] = merge_video_resample.format_factor(applied)
        if not len(factors_by_rate):
            raise merge_video_chimeric.chimeric_error(
                "no sampling rate is readable on any candidate audio track: "
                "cannot state the applied speed factor for the fabricated marker",
                cause="speed_marker_no_sample_rate")
        distinct_factors = set(factors_by_rate.values())
        if len(distinct_factors) > 1:
            raise merge_video_chimeric.chimeric_error(
                f"the candidate's audio tracks' sample rates "
                f"({sorted(factors_by_rate.keys())}) resolve to DIFFERENT "
                f"effective speed factors ({sorted(distinct_factors)}): one "
                f"marker cannot describe what every stream actually "
                f"received -- refusing rather than tagging some tracks wrong",
                cause="speed_marker_ambiguous_mixed_sample_rates")
        applied_factor = next(iter(distinct_factors))
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


def _candidate_sample_rate_for_speed_test(candidate_obj):
    '''Meme lecture que `get_marker_value_for` (ce fichier): ffprobe d'abord,
    MediaInfo en repli. Duplique volontairement plutot qu'appelle
    `pal_speed_verdict._candidate_sample_rate` -- fonction privee d'un autre
    module, et la meme logique existe deja, publique en pratique, ICI.'''
    for language, audios in candidate_obj.audios.items():
        for audio in audios:
            rate = audio.get("ffprobe", {}).get("sample_rate") or audio.get("SamplingRate")
            if rate != None:
                return int(float(rate))
    return None


def describe_resample_decline(candidate_ratio, gate):
    '''VERDICT_WITHOUT_MEASUREMENT (RULINGS_IN_FORCE.md, 2026-09-21): the
    resample gate's decline must carry the numbers it concluded from, in the
    SAME line a reader actually sees -- not only in a `tools.logs` line gated
    `if tools.dev`, which is dead on the production instance (`dev: false`).

    `gate` is `merge_video_resample.test_speed_ratio_against_master`'s return
    dict. ITS OWN TOP-LEVEL `median_fidelity`/`margin`/`ratio` ARE `None` ON
    EVERY DECLINE, BY THAT FUNCTION'S OWN DESIGN (they carry the WINNING
    hypothesis only, and a decline has no winner) -- reading them here would
    reproduce exactly the blank-reads-as-no-measurement defect this function
    exists to avoid. The real numbers, when they exist, live PER HYPOTHESIS in
    `gate["hypotheses"]["direct"/"reciprocal"]["median"]`.

    Measured 2026-09-22 against real wave declines (Lead's ids
    e8f7bc8a55924482, 79b17a3f34c007df): `merge_video_resample.
    test_speed_ratio_against_master` now distinguishes a per-hypothesis
    `"unmeasurable"` verdict (filter could not be built, or the whole-track
    resample itself failed -- the instrument never ran) from a genuine
    measured `"below"` verdict (a real median was computed and it did not
    clear the floor). This function reads THAT distinction rather than the
    outer `cause` token alone, because `cause="resample_fidelity_below_floor"`
    only guarantees at least one hypothesis has a real median -- it does not
    say the OTHER one does, and picking the best available is what SPEC_ZONE_A
    s4f's own "report by how much it won" asks for on the CONFIRM side; on
    DECLINE the closest miss is the equivalent quantity.
    '''
    import merge_video_resample
    hypotheses = gate.get("hypotheses") or {}
    best_name, best_median = None, None
    for name, result in hypotheses.items():
        median = result.get("median")
        if median is None:
            continue
        if best_median is None or median > best_median:
            best_name, best_median = name, median
    if best_median is None:
        # NEITHER HYPOTHESIS PRODUCED A MEDIAN -- the instrument did not run
        # (filter build / whole-track resample failed on both sides), or ran
        # and stayed inconclusive at the hard ceiling with nothing usable.
        # BRIEF_COMMON rule 5: this is a DIFFERENT answer from a measured
        # negative, and it is said as one.
        return (f"resample fidelity gate ran but produced no usable median "
                f"for either hypothesis (candidate duration ratio={candidate_ratio}): "
                f"{gate.get('cause')}")
    gap = merge_video_resample.RESAMPLE_FIDELITY_FLOOR - best_median
    return (f"resample fidelity gate declined: ratio={candidate_ratio} "
            f"(tested as {best_name}) median_fidelity={best_median} "
            f"floor={merge_video_resample.RESAMPLE_FIDELITY_FLOOR} "
            f"gap={gap:.4f} cause={gate.get('cause')}")


NO_RATE_RELATION_CAUSE = "no_rate_relation"
# THE LEFT HALF of the combined verdict token the no_band route produces when
# the RATE leg found nothing to test: `cause=no_rate_relation+<splice_cause>`
# (RULING_20260922_NO_BAND_ROUTING.MD point 5). It is never emitted alone --
# a token with no splice cause beside it would be a one-leg terminal, which is
# exactly what that ruling made illegal.


def _no_band_terminal(rate_cause, rate_prose, splice_cause, splice_prose):
    '''BOTH LEGS HAVE DECLINED -- the only shape in which the no_band route is
    allowed to end (RULING_20260922_NO_BAND_ROUTING.MD point 5, applying
    VERDICT_WITHOUT_MEASUREMENT).

    The token carries BOTH named causes, joined by `+`, so a census can split
    it and see which leg decided what; the prose carries both legs' deciding
    INSTRUMENTS and their numbers, because a cause without the measurement it
    came from is the defect `describe_resample_decline` above exists to
    refuse.
    '''
    return None, f"{rate_cause}+{splice_cause}", (
        f"no_band route declined on both legs. "
        f"RATE leg [{rate_cause}]: {rate_prose} "
        f"SPLICE leg [{splice_cause}]: {splice_prose}")



def corroborate_with_slope(best_video, candidate_obj, language, measurements,
                           sweep_winner):
    '''THE SLOPE, DEMOTED TO CORROBORATION (RULING_20260922_NO_BAND_ROUTING.MD,
    ADDENDUM 2 -- OWNER OVERRIDE). Returns the derivation dict; GATES NOTHING.

    IT USED TO DECIDE. Until the owner's override, an unsnappable slope meant
    "no rate leg" and the pair went straight to splice without the ladder ever
    running. That made ONE inferred number the gatekeeper of a MEASUREMENT,
    which is backwards: the sweep can ask the ladder about every rate
    combination that exists, and sixteen measured answers do not need an
    inferred one's permission to be heard.

    IT IS STILL COMPUTED, AND THAT IS NOT SENTIMENT. Three things this line
    can say that the sweep cannot:
      * the slope AGREES with the winner -- two independent instruments, one
        answer, which is the strongest evidence this chain ever produces;
      * the slope DISAGREES with the winner -- the sweep won on fidelity and
        the drift says otherwise, which is exactly the row a census should
        pull first;
      * the slope REFUSED (scatter-dominated) while the sweep still found a
        winner -- the owner asked for this case by name, and it is the
        signature of a pair whose per-window offsets are noise while a whole-
        track relation is nonetheless real.

    So this writes a line and returns. Nothing downstream branches on it.
    '''
    import pal_speed_discriminator
    series = (measurements or {}).get("delay_series") or []
    window = (measurements or {}).get("window")
    derivation = pal_speed_discriminator.derive_rate_factor_from_slope(
        series, window=window)
    snapped = pal_speed_discriminator.snap_to_named_rational(derivation["factor"])
    derivation["snapped"] = (None if snapped is None
                             else f"{snapped.numerator}/{snapped.denominator}")
    # HEADERS CORROBORATE, NEVER DECIDE (ruling point 2c). They corroborated
    # nothing that decided anything before the override either; now the whole
    # function is in that category, which is the tidiest place for them.
    master_fps = getattr(best_video, "get_fps", lambda: None)()
    candidate_fps = getattr(candidate_obj, "get_fps", lambda: None)()
    derivation["declared_rate_corroboration"] = \
        pal_speed_discriminator.describe_rate_corroboration(
            snapped, master_fps, candidate_fps)

    if sweep_winner is None:
        agreement = "no sweep winner to compare against"
    elif derivation["factor"] is None:
        agreement = (f"slope REFUSED ({derivation['refusal']}) while the sweep "
                     f"chose {sweep_winner}: no corroboration available, and a "
                     f"scatter-dominated slope beside a real winner is itself "
                     f"worth seeing")
    else:
        winner_value = Decimal(sweep_winner.numerator) / Decimal(sweep_winner.denominator)
        gap = abs(Decimal(str(derivation["factor"])) - winner_value) / winner_value
        agrees = gap <= pal_speed_discriminator.SNAP_RELATIVE_TOLERANCE
        agreement = (f"measured slope factor {derivation['factor']} "
                     f"(snapped {derivation['snapped']}) vs sweep winner "
                     f"{sweep_winner}: relative gap {gap} -> "
                     f"{'AGREE' if agrees else 'DISAGREE'}")
    derivation["sweep_agreement"] = agreement
    tools.dev_log(
        f"repair: corroborate_with_slope on {candidate_obj.filePath} "
        f"language={language} n_points={derivation['n_points']} "
        f"n_used={derivation['n_used']} "
        f"slope_ms_per_s={derivation['slope_ms_per_s']} "
        f"factor={derivation['factor']} "
        f"residual_scatter_ms={derivation['residual_scatter_ms']} "
        f"r_squared={derivation['r_squared']} "
        f"snapped={derivation['snapped']} refusal={derivation['refusal']} "
        f"window={derivation['window']} | {agreement} | "
        f"{derivation['declared_rate_corroboration']}\n")
    return derivation


def run_speed_sweep(best_video, candidate_obj, language):
    '''STEP 2 OF THE OWNER'S DIAGRAM -- the speed test, as a SWEEP over every
    rate combination. Returns `(gate, cause, prose)`; `gate` is None only when
    the sweep could not be started at all.

    `merge_video_resample.sweep_rate_ratios` does the measuring; this function
    is the repair chain's door to it and owns the two reasons the door may not
    open (no readable sampling rate, no work directory).
    '''
    sample_rate = _candidate_sample_rate_for_speed_test(candidate_obj)
    if sample_rate == None:
        return None, "rate_sweep_no_sample_rate", (
            "the rate sweep could not be run: no sampling rate readable on "
            "the candidate")
    import merge_video_resample
    work_dir = path.join(tools.tmpFolder, "repair", "rate_sweep")
    tools.make_dirs(work_dir)
    vocabulary = merge_video_resample.build_rate_ratio_vocabulary()
    tools.dev_log(
        f"repair: run_speed_sweep on {candidate_obj.filePath} language="
        f"{language} sample_rate={sample_rate} factors={len(vocabulary)} "
        f"floor={merge_video_resample.RESAMPLE_FIDELITY_FLOOR} "
        f"vocabulary={[f'{f.numerator}/{f.denominator}' for f in vocabulary]}\n")
    gate = merge_video_resample.sweep_rate_ratios(
        best_video.filePath, candidate_obj.filePath, sample_rate, work_dir,
        vocabulary=vocabulary)
    tools.dev_log(
        f"repair: run_speed_sweep result for {candidate_obj.filePath}: "
        f"verdict={gate['verdict']} winner={gate.get('ratio')} "
        f"median={gate.get('median_fidelity')} margin={gate.get('margin')} "
        f"cause={gate.get('cause')} passing={gate.get('passing')}\n")
    return gate, None, None


def splice_after_rate_normalisation(best_video, candidate_obj, language,
                                    winner, gate):
    '''STEP 3 OF THE OWNER'S DIAGRAM -- chimeric, after rate normalisation.

    RESAMPLE FIRST, THEN SPLICE: the standing pipeline-order invariant,
    applied INSIDE the repair for a mixed rate+content case (ruling point 4).
    The sweep has already picked `winner` on measured fidelity, so the pair is
    normalised on disk with `merge_video_resample.build_resampled_candidate`
    (which exists for exactly this order -- its own docstring says so) and the
    NORMALISED pair is handed to the splice chain's ordinary entry,
    `change_point_locator.locate_change_points`. From that entry's point of
    view this is now a rate-free content-diff case, which is the whole point.

    THE WHOLE-TRACK PASS HAPPENS ONCE, HERE, AND ONLY FOR THE WINNER. The
    sweep itself never muxes a whole track -- it corrects window anchors
    arithmetically instead (`_probe_fidelity_at_ratio`) -- which is what keeps
    sixteen hypotheses affordable. This is the one place a real file is
    written, and by then exactly one factor is in play.

    Returns `(plan, cause, detail)`.
    '''
    import change_point_locator
    import merge_video_resample
    ratio = Decimal(winner.numerator) / Decimal(winner.denominator)
    work_dir = path.join(tools.tmpFolder, "repair", "rate_normalised")
    tools.make_dirs(work_dir)
    out_path = path.join(work_dir, candidate_obj.fileBaseName + ".rate_normalised.mkv")
    tools.dev_log(
        f"repair: splice_after_rate_normalisation building the normalised "
        f"candidate for {candidate_obj.filePath} at the sweep winner "
        f"{winner.numerator}/{winner.denominator} out_path={out_path}\n")
    try:
        normalised_path, applied, seen = merge_video_resample.build_resampled_candidate(
            candidate_obj, ratio, out_path)
        normalised_obj = video.video(path.dirname(normalised_path),
                                     path.basename(normalised_path))
        normalised_obj.get_mediadata()
    except Exception as error:                          # noqa: BLE001 -- see below
        # NARROW BY SCOPE, NOT BY TYPE -- the same rule `merge_video_resample`
        # states at its own whole-track resample site and `change_point_locator`
        # at `_probe`. Three statements are covered and every one of them is a
        # NEW step on a path that was a terminal decline an hour ago: an ffmpeg
        # mux, a mediainfo/mkvmerge read, and a constructor that raises a bare
        # `Exception` when the file it was just handed does not exist. A
        # failure in any of them is a measurement ("this pair could not be
        # normalised"), never a reason to take down a merge that was already
        # going to reject this file.
        tools.dev_log(
            f"repair: splice_after_rate_normalisation could not normalise "
            f"{candidate_obj.filePath}: {type(error).__name__}: {error}\n")
        return None, "rate_normalisation_failed", (
            f"the sweep winner {winner.numerator}/{winner.denominator} "
            f"cleared the fidelity floor but the normalised candidate could "
            f"not be built: {type(error).__name__}")
    # ITS OWN `work_dir`, AS THAT MODULE ASKS. `change_point_locator._probe`
    # states that its probe `tag` is unique only WITHIN a work_dir, and this
    # is the SECOND locator run on one candidate inside one repair -- the
    # first (`get_plan_from_locator`) takes the default `tools.tmpFolder` and
    # writes `cpl_m_s0.wav` there. Sharing it would have the two runs' probe
    # files collide by name.
    locate_dir = path.join(work_dir, "locate")
    tools.make_dirs(locate_dir)
    plan, splice_cause = change_point_locator.locate_change_points(
        best_video, normalised_obj, language, work_dir=locate_dir)
    tools.dev_log(
        f"repair: splice_after_rate_normalisation located on the normalised "
        f"pair for {candidate_obj.filePath}: plan={plan is not None} "
        f"cause={splice_cause} applied_factor={applied}\n")
    if plan is None:
        return None, splice_cause, (
            f"rate normalised at {winner.numerator}/{winner.denominator} "
            f"(applied {applied}, sweep median {gate.get('median_fidelity')} "
            f"against floor {merge_video_resample.RESAMPLE_FIDELITY_FLOOR}), "
            f"then change_point_locator declined on the normalised pair")
    # `verdict` IS NOT DECORATION -- IT IS WHAT MAKES `speed_ratio` READABLE.
    # Found while completing the evidence guard, and it was a silent defect in
    # my own first draft: `get_speed_ratio` (this file) returns
    # `(None, prose, "speed_verdict_absent")` for a plan that carries a
    # `speed_ratio` and NO `verdict`. A None ratio does not raise -- it means
    # "no speed relation", so the guard would never fire and
    # `assemble_on_master_timeline` would build the segments while SILENTLY
    # DROPPING the rate correction the whole route exists to apply. The
    # transform is `asetrate` by `docs/AUDIO_SPEED_POLICY.MD`, which is a
    # DECISION, and this route applies exactly it.
    plan["verdict"] = "asetrate"
    plan["kind"] = "speed_and_splice"
    # THE PLAN IS MEASURED ON THE NORMALISED CANDIDATE, AND IT SAYS SO. Its
    # segment boundaries are positions on a timeline that only exists after
    # the resample, so a consumer must not apply them to the original file
    # without applying the factor too. Both facts travel in the plan.
    plan["speed_ratio"] = str(ratio)
    plan["speed_ratio_convention"] = RATIO_CONVENTION
    plan["speed_ratio_exact"] = f"{winner.numerator}/{winner.denominator}"
    plan["speed_margin"] = gate.get("margin")
    plan["rate_source"] = "rate_sweep"
    plan["rate_sweep_passing"] = gate.get("passing")
    plan["rate_normalised_candidate_path"] = normalised_path
    plan["rate_normalised_applied_factor"] = str(applied)
    plan["resample_gate"] = {k: v for k, v in gate.items() if k != "hypotheses"}
    return plan, None, None


def speed_sweep_then_splice(best_video, candidate_obj, language,
                            discriminator_result, measurements, locator_cause):
    '''THE no_band ROUTE, IN THE SHAPE OF THE OWNER'S DIAGRAM
    (RULING_20260922_NO_BAND_ROUTING.MD, ruling + ADDENDUM 2):

        STEP 1  CLASSIFICATION   -- done by the caller
                                    (`pal_speed_discriminator.discriminate`)
        STEP 2  SPEED TEST       -- `run_speed_sweep`: every rate combination,
                                    one fidelity ladder each, best median wins
        STEP 2b CORROBORATION    -- `corroborate_with_slope`: logged, decides
                                    nothing
        STEP 3  CHIMERIC         -- `splice_after_rate_normalisation` when the
                                    sweep won; the locator's existing verdict
                                    on the un-normalised pair when it did not
        STEP 4  PLAN APPLICATION -- the caller's
                                    `build_repaired_video_object`, behind its
                                    evidence guard

    WHAT THIS REPLACED. `band == "no_band"` used to reach the resample gate
    with the raw DURATION RATIO as its hypothesis, and returned a TERMINAL
    `resample_fidelity_below_floor` when that misaligned every probe window.
    Two things were wrong at once: the band's own hypothesis text says
    "wrong-content suspicion", and the number being tested conflated a real
    rate offset with an unrelated content-length difference -- so the gate was
    asked whether the WRONG factor explained the drift and correctly said no.

    ON THE SPLICE LEG WHEN THERE IS NO RATE LEG, stated rather than hidden:
    the ruling says "route DIRECTLY to the splice chain -- the same entry the
    splice-class cases use today". That entry is
    `change_point_locator.locate_change_points`, and it has already run on
    this exact pair: it is what returned `locator_cause` and sent us here.
    Calling it a second time with the same two files is deterministic and
    would return the same token at the cost of the whole probe grid, so the
    splice leg's verdict is read from that run instead of re-measured. The
    OUTCOME is the ruling's; the mechanism spends nothing to reach it.
    (Deviation B3, accepted by the Architect in ADDENDUM 1.)
    '''
    # ---- STEP 1: classification (already decided by the caller) ----------
    duration_ratio_screened = discriminator_result.get("speed_ratio")
    tools.dev_log(
        f"repair: speed_sweep_then_splice entered for "
        f"{candidate_obj.filePath} language={language} band=no_band "
        f"duration_ratio={duration_ratio_screened} "
        f"(screen only, BANNED as a resample factor on this band) "
        f"locator_cause={locator_cause} "
        f"delay_series_points={len((measurements or {}).get('delay_series') or [])}\n")

    splice_prose = (
        f"change_point_locator already ran on this un-normalised pair and "
        f"declined with {locator_cause}; no rate correction was confirmed, "
        f"so there is nothing to re-measure it on")

    # ---- STEP 2: the speed test -- the SWEEP decides ---------------------
    gate, sweep_cause, sweep_prose = run_speed_sweep(
        best_video, candidate_obj, language)
    if gate is None:
        # The instrument could not be started. Distinct from "it ran and no
        # factor cleared the floor" -- BRIEF_COMMON rule 5.
        corroborate_with_slope(best_video, candidate_obj, language,
                               measurements, None)
        return _no_band_terminal(sweep_cause, sweep_prose,
                                 locator_cause, splice_prose)

    winner = gate.get("ratio") if gate["verdict"] == "confirmed" else None

    # ---- STEP 2b: corroboration only. Nothing below branches on it. ------
    derivation = corroborate_with_slope(best_video, candidate_obj, language,
                                        measurements, winner)

    # ---- STEP 3: chimeric ------------------------------------------------
    if winner is None:
        # NO FACTOR REACHED THE FLOOR -> no rate leg, splice directly.
        rate_prose = (
            f"the rate sweep tested {gate.get('vocabulary_size')} exact rate "
            f"combinations against the unchanged floor "
            f"{_resample_floor()} and none reached it "
            f"({describe_resample_decline('the rate sweep vocabulary', gate)}); "
            f"corroboration: {derivation['sweep_agreement']}; "
            f"{derivation['declared_rate_corroboration']}")
        tools.dev_log(
            f"repair: speed_sweep_then_splice routing {candidate_obj.filePath} "
            f"to the SPLICE leg with no rate leg: sweep cause="
            f"{gate.get('cause')}\n")
        return _no_band_terminal(gate["cause"], rate_prose,
                                 locator_cause, splice_prose)

    tools.dev_log(
        f"repair: speed_sweep_then_splice routing {candidate_obj.filePath} "
        f"to RESAMPLE-THEN-SPLICE at the sweep winner {winner} "
        f"(median={gate.get('median_fidelity')})\n")
    plan, cause, detail = splice_after_rate_normalisation(
        best_video, candidate_obj, language, winner, gate)
    if plan is not None:
        return plan, None, None
    # THE RATE LEG WON AND THE SPLICE LEG DID NOT. Still a two-leg terminal,
    # and the left token says the rate leg CONFIRMED -- a reader must be able
    # to tell this apart from a pair with no rate relation at all, which is
    # the four-states-one-token shape this file has already had to undo twice.
    return _no_band_terminal(
        f"rate_confirmed_{winner.numerator}_{winner.denominator}",
        f"the rate sweep chose {winner} on a median fidelity of "
        f"{gate.get('median_fidelity')} against floor {_resample_floor()} "
        f"(passing factors: {gate.get('passing')}); corroboration: "
        f"{derivation['sweep_agreement']}",
        cause, detail)


def _resample_floor():
    '''The unchanged fidelity floor, read from its owner rather than repeated.

    A second literal `0.90` in this file would be a second definition of one
    number, and the ruling keeps saying "floor unchanged" -- which is only
    checkable if there is exactly one place it lives.
    '''
    import merge_video_resample
    return merge_video_resample.RESAMPLE_FIDELITY_FLOOR



def confirm_speed_relation_via_resample(best_video, candidate_obj, language,
                                        locator_cause=None, measurements=None):
    '''STEP 2 DU PIPELINE DU PROPRIETAIRE -- "Test Reechantillonnage
    (Fidelite > 0,90)" (BRIEF.md; RULINGS_IN_FORCE.md, ligne
    `PIPELINE_CANONICAL`, 2026-09-21). LE PRODUCTEUR MANQUANT: avant cette
    fonction, rien dans le depot n'ecrivait jamais `plan["speed_ratio"]` --
    mesure independamment par `dev-step1-classify` (call graph) et par moi
    (`_decline` de `change_point_locator.py` rend `(None, reason)`, DEUX
    elements, jamais les champs `pal_chain_*` qu'elle journalise).

    Appelee quand `change_point_locator` a decline avec l'une des trois
    causes de `SPEED_CONFIRMER_ENTRY_CAUSES` -- `speed_relation_suspected`
    (derive monotone vue), ou `median_fidelity_below_floor` /
    `offsets_scattered` (la Stage 1 n'a PAS vu de monotonie, mais 19 des 20
    PAL confirmes de la campagne declinent par l'un de ces deux jetons a
    cause du bruit intra-fenetre, voir `SPEED_CONFIRMER_ENTRY_CAUSES`).
    Dans tous les cas la fidelite mediane originale etait sous son propre
    plancher (`MIN_MEDIAN_FIDELITY`, 0.70, UNE AUTRE QUANTITE que celle
    testee ici).

    STEP 2 EST SON PROPRE CONFIRMATEUR (Lead ruling on Q1, 2026-09-21): le
    diagramme du proprietaire dessine DEUX boites -- Classification, puis Test
    Reechantillonnage -- et aucune troisieme. La premiere version de cette
    fonction exigeait D'ABORD `pal_speed_verdict.determine_speed_verdict`
    (pitch + NCC, un plancher NCC_FLOOR=0.80 DIFFERENT et un instrument
    DIFFERENT) avant de tenter mon propre plancher -- mesure sur un vrai
    exemplaire PAL confirme (curated-46, `VMSAM_CORPUS`) que cette chaine
    DECLINE (NCC 0.60 contre son propre plancher 0.80) alors que sa PROPRE
    sonde de hauteur tonale est D'ACCORD et que mon plancher de
    reechantillonnage confirme a 0.93. Le proprietaire n'a jamais dessine
    cette troisieme boite; l'exiger transformait le Test Reechantillonnage en
    second avis sur le verdict d'une autre chaine plutot que le test qu'il a
    specifie. RETIRE. `pal_speed_verdict` reste utile ailleurs (son propre
    journal sur le decline de la Stage 1); il n'est plus un prealable ici.

    LE RATIO CANDIDAT VIENT DE LA DUREE SEULE (`pal_speed_discriminator`,
    deja publique, deuxieme site d'appel de sa fonction privee -- convention
    deja etablie par ce meme module). Aucun filtrage par `band`: la
    classification (`pal_direct`/`pal_inverse`/`no_band`) est une etiquette
    pour LA CHAINE pal_speed_verdict, pas une condition pour ce test -- mon
    propre plancher, avec son bras reciproque et sa mesure sur fichier entier,
    est l'arbitre. Un ratio `None` (duree inutilisable) est le seul cas qui ne
    peut pas etre teste du tout.

    LE BRAS RECIPROQUE RESTE PERMANENT (Lead ruling): appliquer le mauvais
    sens et voir la fidelite RESTER BASSE est ce qui rend ce test une VRAIE
    mesure et non une seconde lecture de la meme correlation -- SPEC_ZONE_A.MD
    s4f l'exige explicitement ("un depatageage tire des memes correlations
    n'est pas un second avis").

    Renvoie (plan, cause, detail). `plan` est None si la relation n'a pas ete
    confirmee. Le plan produit ne porte PAS de `segments`: une relation de
    vitesse pure couvre toute la timeline (`build_repaired_video_object` le
    sait deja construire).

    `detail` est None quand `plan` n'est pas None (rien a expliquer), et une
    PROSE portant les nombres mesures (ratio, fidelite, ecart au plancher)
    quand `cause` n'est pas None -- VERDICT_WITHOUT_MEASUREMENT, voir
    `describe_resample_decline` ci-dessus. Ajoute 2026-09-22: avant ce
    changement seul `cause` voyageait, et le seul site qui portait les
    nombres etait un `tools.logs.append` gate par `if tools.dev` -- mort sur
    l'instance de production (`dev: false`).
    '''
    tools.dev_log(f"repair: confirm_speed_relation_via_resample starting "
                  f"master={best_video.filePath} "
                  f"candidate={candidate_obj.filePath} language={language}\n")
    import pal_speed_discriminator
    discriminator_result, disc_error = pal_speed_discriminator.discriminate_from_videos(
        best_video, candidate_obj, language)
    if disc_error is not None:
        return None, "resample_test_locator_module_absent", (
            f"resample fidelity test could not even start: {disc_error}")
    ratio = discriminator_result.get("speed_ratio")
    if ratio is None:
        return None, "resample_test_duration_unmeasurable", (
            "resample fidelity test could not start: no duration-based "
            "ratio candidate from pal_speed_discriminator")

    # *** THE no_band BAND NO LONGER TESTS THE DURATION RATIO BY RESAMPLE ***
    # RULING_20260922_NO_BAND_ROUTING.MD point 1: on this band the duration
    # ratio is a SCREEN, not a factor. The paragraph above ("Aucun filtrage
    # par `band`") remains the rule for every OTHER band -- pal_direct,
    # pal_inverse and near_unity still reach the gate below unfiltered, and
    # this file's own floor still arbitrates them. What changed is narrower
    # than a filter: on the one band whose hypothesis text already said
    # "wrong-content suspicion", a DIFFERENT and better-founded factor is
    # derived first, and the pair keeps a splice leg either way.
    if discriminator_result.get("band") == "no_band":
        return speed_sweep_then_splice(best_video, candidate_obj, language,
                                       discriminator_result, measurements,
                                       locator_cause)

    sample_rate = _candidate_sample_rate_for_speed_test(candidate_obj)
    if sample_rate == None:
        return None, "resample_test_no_sample_rate", (
            "resample fidelity test could not start: no sampling rate "
            "readable on the candidate")

    import merge_video_resample
    work_dir = path.join(tools.tmpFolder, "repair", "resample_fidelity_test")
    tools.make_dirs(work_dir)
    gate = merge_video_resample.test_speed_ratio_against_master(
        best_video.filePath, candidate_obj.filePath, ratio, sample_rate, work_dir)
    if tools.dev:
        # THE NUMBERS THAT ACTUALLY EXIST ON A DECLINE, not the outer
        # fields that `test_speed_ratio_against_master` sets to None on
        # every decline by design (they carry the WINNING hypothesis only).
        # Reading `gate.get('median_fidelity')` here reproduced exactly the
        # blank-reads-as-no-measurement confusion this file exists to
        # refuse elsewhere -- measured 2026-09-22 against real declines
        # (ids e8f7bc8a55924482, 79b17a3f34c007df) that this line printed
        # as `median=None` while `gate['hypotheses']` held real numbers.
        per_hypothesis = " ".join(
            f"{name}(verdict={r.get('verdict')},median={r.get('median')})"
            for name, r in (gate.get("hypotheses") or {}).items())
        # ROUTED THROUGH `tools.dev_log` (owner's order via the Lead,
        # 2026-09-22, wave 3): this `tools.logs.append` had no stderr half --
        # the exact defect class `change_point_locator._log` had, at a site
        # wave 2 did not reach. `tools.logs` drains only at the end of a
        # merge; a hung process never gets there. Format string unchanged.
        tools.dev_log(
            f"repair: resample fidelity gate for {language}: band="
            f"{discriminator_result.get('band')} verdict={gate['verdict']} "
            f"cause={gate.get('cause')} {per_hypothesis}\n")
    if gate["verdict"] != "confirmed":
        return None, gate["cause"], describe_resample_decline(ratio, gate)

    # THIS BRANCH STAYS REFUSED AT THE EVIDENCE GATE, AND THAT IS DELIBERATE
    # (RULING_20260922_NO_BAND_ROUTING.MD, ADDENDUM: "if that branch produces
    # equivalent evidence, thread it the same way; if it does not, leave it
    # refused and say so rather than widening").
    #
    # IT DOES NOT. Two of the three evidence legs are here -- the ladder
    # confirmed and its median cleared the floor (`resample_gate`) -- and the
    # THIRD IS MISSING BY THIS BRANCH'S OWN DESIGN: `gate["ratio"]` is the
    # MEASURED duration ratio, never a snapped named rational, because
    # `pal_speed_discriminator`'s t112 caveat forbids a band from supplying
    # the number. So `speed_ratio_exact` does not exist here and cannot be
    # fabricated: writing one would mean asserting a rational this branch
    # never recognised.
    #
    # NOT WIDENED HERE. Snapping this branch's ratio too is a plausible next
    # step and it is a DIFFERENT change with its own acceptance. MEASURED
    # 2026-09-22 against the three confirmed real PAL ids this module's own
    # docstring names (errids 70/135/213, ratios 0.959/0.958/0.956) and the
    # 5e-4 snap window around 960/1001 = 0.9590410:
    #     0.959  relative gap 4.271e-5   INSIDE
    #     0.958  relative gap 1.085e-3   outside
    #     0.956  relative gap 3.171e-3   outside
    # So the snap would admit ONE of the three and refuse two, and what to do
    # with the other two -- whose ratios are real, confirmed, and simply not
    # within a whisker of the nominal because a DURATION ratio is not a rate
    # -- is exactly the question this addendum did not answer. Deriving them
    # from a SLOPE instead (this ruling's own instrument) is the shape of the
    # answer, and it is a separate ruling's work, not a quiet edit here.
    return {"kind": "speed", "verdict": "asetrate",
            "speed_ratio": str(gate["ratio"]),
            "speed_ratio_convention": RATIO_CONVENTION,
            "speed_margin": gate["margin"],
            "duration_master_s": float(best_video.video["Duration"]),
            "duration_candidate_s": float(candidate_obj.video["Duration"]),
            "resample_gate": {k: v for k, v in gate.items() if k != "hypotheses"}}, None, None


def get_plan_from_locator(best_video, candidate_obj, language):
    """La mesure de `vmsam-dev-1`, appelee ici et nulle part ailleurs.

    Import tardif et tolerant: le module peut ne pas etre deploye, et une
    capacite fermee par defaut n'a pas le droit de casser un merge parce qu'une
    dependance manque.

    `None` veut dire *je n'ai pas pu mesurer*, jamais *les fichiers sont
    compatibles*. On laisse alors le refus tel quel.

    Renvoie (plan, cause, detail) depuis 2026-09-22 -- `detail` est None
    partout SAUF quand le confirmateur de vitesse (Stage 2) a decline avec des
    nombres a porter (voir `describe_resample_decline`); tous les autres
    chemins de cette fonction gardent leur prose historique, inchangee, au
    site d'appel.
    """
    tools.dev_log(f"repair: get_plan_from_locator starting "
                  f"master={best_video.filePath} "
                  f"candidate={candidate_obj.filePath} language={language}\n")
    try:
        import change_point_locator
    except Exception as error:
        # ROUTED THROUGH `tools.dev_log` (owner's order via the Lead,
        # 2026-09-22, wave 3) -- same reason as every other site in this
        # batch: this used to write ONLY to `tools.logs`, format unchanged.
        tools.dev_log(f"repair: no change_point_locator module: {error}\n")
        # THE MODULE IS NOT DEPLOYED. This is MY OWN process state and I am
        # entitled to state it: no measurement was attempted, because there was
        # nothing to attempt it with. Distinct from "the locator ran and refused".
        # LE JETON EST STABLE; la classe d'exception va dans la PROSE. dev-4
        # classe sur le jeton, donc un jeton qui varie n'est pas un jeton.
        return None, "locator_module_absent", None
    # *** THE PRODUCER HALF LANDED. `locate_change_points` now returns `(plan, cause)`
    # per CAMPAIGN.MD 1153-1164 -- cause is None when a plan is returned, and a stable
    # snake_case token on every one of the ten boundary refusals.
    # UNPACKED, NOT TRUTH-TESTED. The old line read `if plan != None`, and a `(None, token)`
    # TUPLE IS NOT None AND IS TRUTHY -- so leaving that test in place would have read EVERY
    # REFUSAL AS A SUCCESSFUL PLAN. Measured before landing: bool((None, "tok")) is True.
    # That is why both halves are in one commit and why this line changed shape rather than
    # gaining a branch. ***
    # THE OUT-DICT IS THE WHOLE POINT OF THIS LINE'S CHANGE
    # (RULING_20260922_NO_BAND_ROUTING.MD point 2): the locator measures a
    # per-window offset series on its way to a verdict, and until now threw it
    # away on every decline. `confirm_speed_relation_via_resample`'s no_band
    # route needs exactly that series to derive a rate factor from the SLOPE
    # instead of from the conflated duration ratio. Empty on the earliest
    # refusals, which is honest: the locator had not probed yet.
    locator_measurements = {}
    plan, locator_cause = change_point_locator.locate_change_points(
        best_video, candidate_obj, language, measurements=locator_measurements)
    if plan is not None:
        return plan, None, None
    if locator_cause in SPEED_CONFIRMER_ENTRY_CAUSES:
        # STEP 2 OF THE OWNER'S PIPELINE, HERE AND ONLY HERE (BRIEF.md;
        # RULINGS_IN_FORCE.md `PIPELINE_CANONICAL`, 2026-09-21: "Q1 RESOLVED =
        # ACT-and-decide"). Stage 1 (`change_point_locator.py`, not mine) only
        # SUSPECTS a speed relation and declines unconditionally today -- see
        # its own comment at the call site, "No resample, no repair call".
        # `confirm_speed_relation_via_resample` is the missing producer: it
        # confirms the ratio (PAL/NTSC chain) AND gates it against a
        # WHOLE-TRACK RESAMPLE's fidelity (`merge_video_resample`,
        # RESAMPLE_FIDELITY_FLOOR=0.90 -- a DIFFERENT quantity from Stage 1's
        # own MIN_MEDIAN_FIDELITY=0.70, see that module's docstring). A plan
        # returned here still cannot reach a real repair today:
        # `build_repaired_video_object`'s unconditional guard
        # (`speed_transform_not_validated`) refuses any non-None
        # `speed_ratio` until the Lead's own commit lifts it with evidence --
        # this branch stops at "the plan exists and is admissible", which is
        # everything this file can validate on its own.
        #
        # WHY THREE CAUSES, NOT ONE (Lead's authorization, 2026-09-21, on a
        # measured population): `dev-step1-classify` ran all 20 of the
        # campaign's confirmed real PAL ids through the real classifier and
        # found ZERO reach `speed_relation_suspected` -- a 4.27% drift inside
        # one UNCORRECTED probe window is large enough to turn the
        # correlation into noise, and noise is not monotone
        # (`change_point_locator.py:1952-1955`'s own comment, predicted
        # before either seat measured a real file). 12/20 land in
        # `median_fidelity_below_floor`, 7/20 in `offsets_scattered`. This
        # producer does NOT depend on Stage 1's monotone check having
        # succeeded: it re-derives its own ratio from DURATION alone
        # (`pal_speed_discriminator`, unaffected by per-window noise) and
        # gates it by resampling the WHOLE track BEFORE probing -- by the
        # time it measures, the within-window drift that defeats Stage 1 is
        # already corrected away. So trying it on these two additional
        # causes is not two seats patching one symptom: if Stage 1's monotone
        # check is later repaired, `speed_relation_suspected` simply becomes
        # a third entry alongside these two, additive, not conflicting --
        # this producer's own gate stays the decider either way.
        #
        # THE ACCEPTANCE COST OF WIDENING (Lead's mandatory arm, same
        # authorization): `median_fidelity_below_floor` and
        # `offsets_scattered` are BROAD buckets -- every pair with low
        # fidelity for ANY reason lands here, not only PAL. Verified on real,
        # non-PAL error-tree pairs before shipping (see
        # `VMSAM_HELP_AI/dev-step2-resample/`'s task file) that this producer
        # declines correctly on files with no speed relation, at the SAME
        # rate as its already-validated negative controls.
        speed_plan, speed_cause, speed_detail = confirm_speed_relation_via_resample(
            best_video, candidate_obj, language,
            locator_cause=locator_cause, measurements=locator_measurements)
        if speed_plan is not None:
            return speed_plan, None, None
        # NOT CONFIRMED. CORRECTED 2026-09-22 (Lead's fold-in, on his own
        # measurement against real wave declines): this used to fall through
        # to the ORIGINAL `locator_cause` unchanged, on the argument that
        # Stage 1's token was "the finding" and `speed_cause` was merely a
        # log-line detail. That was wrong in a way that cost a census: the
        # terminal `no_plan cause=` line this function's caller writes
        # (`repair_not_compatible_videos`) is the ONLY place anyone reads the
        # outcome, and Stage 1's token only ever said WHY THIS PRODUCER WAS
        # TRIED -- never WHAT IT FOUND when it ran. A pair that never reached
        # the resample gate at all and a pair that reached it and measured
        # 0.5631 median fidelity are two different findings, and collapsing
        # them onto the same `median_fidelity_below_floor`/`offsets_scattered`
        # token is precisely the four-states-one-token shape da10f16c and
        # ca8e3307 already had to undo one layer down in this same file.
        # `locator_cause` survives in the dev log line below for the curious
        # (why entry happened at all); the RETURNED cause and detail are now
        # the confirmer's own -- what actually happened when Stage 2 ran.
        # ROUTED THROUGH `tools.dev_log` (owner's order via the Lead,
        # 2026-09-22, wave 3): this is the site the Lead's own dispatch
        # cited by name (:682-685) -- the exact line that used to narrate
        # the whole probe sequence to stderr and then go silent at its own
        # conclusion, because this one line, the one carrying WHY, only
        # ever reached `tools.logs`. Format unchanged.
        tools.dev_log(
            f"repair: {locator_cause} but speed relation not confirmed "
            f"by resample for {language}: {speed_cause}\n")
        return None, speed_cause, speed_detail
    # THE LOCATOR RAN, RETURNED NO PLAN, AND NOW SAYS WHY.
    #
    # This block used to say the producer half was unlanded and held by dev-1's user,
    # so any cause written here would be INVENTED, NOT READ. That was true when it was
    # written and it is no longer true. THE CAUSE BELOW IS READ, NOT INVENTED.
    #
    # HISTORICAL, AND KEPT BECAUSE THE REASONING STILL DECIDES THINGS: the token that used
    # to stand here was `cause_unavailable`, and it replaced a string claiming no measurement
    # existed. THAT TOKEN IS GONE TOO -- see below -- but the argument for why it beat its
    # predecessor is the argument for why the producer's token beats it in turn.
    # "no measurement available" claims a property of the WORLD -- that no
    # measurement exists. For the fidelity-floor path that is FALSE: the probes
    # RAN, they SUCCEEDED, and they returned a CONCLUSIVE NEGATIVE. A
    # conclusive negative filed as an absence of evidence is exactly the
    # substitution change_point_locator warns about in its own words:
    # "None means I could not measure -- never the files are compatible."
    #
    # This file's own instruction, followed to the letter: READ ITS CAUSE HERE AND PASS
    # IT THROUGH UNCHANGED -- one site, this one. NOT TRANSLATED, NOT NORMALISED, AND NO
    # CAUSE OF MY OWN ADDED BESIDE IT.
    #
    # *** I WROTE `or "cause_unavailable"` HERE AN HOUR AGO, IN THE SAME EDIT WHERE I QUOTED
    # THIS FILE'S INSTRUCTION NOT TO ADD A CAUSE OF MY OWN BESIDE THE PRODUCER'S. THAT `or`
    # IS A CAUSE OF MY OWN. I violated the line I was citing, in the comment citing it.
    #
    # AND IT WAS WORSE THAN UNTIDY. `cause_unavailable` MATCHES dev-4'S ACCEPTING REGEX, so a
    # producer that returned no token would have been counted as A STATED CAUSE -- silently
    # inflating the column the campaign's end condition is scored on, with a non-cause.
    # arch-heir's ruling, and it generalises past this file: *** A FALLBACK TOKEN MUST NOT BE A
    # MEMBER OF THE SET IT FALLS BACK FROM. A sentinel that satisfies the predicate it exists to
    # signal the absence of is not a sentinel, it is a silent pass. *** And "unreachable by
    # construction" is not a defence: reachability is a property of today's call graph, not of
    # the token.
    #
    # THE CAUSE PASSES THROUGH UNCHANGED WHENEVER THERE IS ONE. NOT TRANSLATED, NOT NORMALISED,
    # NOTHING ADDED BESIDE IT.
    #
    # *** THE ONE CASE THAT IS NOT A PASS-THROUGH, AND dev-4 IS RIGHT THAT IT NEEDS A VALUE:
    # the producer RAN and returned NO TOKEN. Under the contract that CANNOT HAPPEN -- every
    # boundary return carries one -- so this is a CONTRACT VIOLATION and it must be LOUD.
    # Passing None here would have emitted NO cause field at all, which is a DIFFERENT and
    # legitimate outcome for other call sites of `record`, and folding the two together would
    # hide a violation inside an ordinary row. I had them collapsed; they are not the same.
    #
    # THE SENTINEL IS LEXICALLY OUTSIDE THE ACCEPTED CLASS, NOT MERELY A DIFFERENT WORD.
    # dev-4 accepts `cause=([A-Za-z0-9_]+)`; parentheses cannot satisfy that class, so
    # `(unstated)` is excluded BY CONSTRUCTION rather than by a denylist in the reader.
    # *** A NAME-BASED EXCLUSION DOWNSTREAM WOULD LEAVE THE SENTINEL A MEMBER OF THE SET AND
    # THE NEXT SENTINEL ANYONE ADDS WOULD WALK STRAIGHT PAST IT. *** That is arch-heir's rule
    # taken literally: not a member, rather than a member that is filtered.
    #
    # ORDERING, STATED BECAUSE IT DECIDES WHO MOVES FIRST: dev-4's reader change is NECESSARY
    # AND NOT SUFFICIENT. Reader-side classification cannot repair a sentinel that is lexically
    # a member of the accepted set, so THIS SIDE HAS TO MOVE FIRST OR THE NEW COLUMN NEVER FILLS.
    if locator_cause is None:
        # *** THE PARENTHESES ARE THE MECHANISM. THEY ARE NOT PUNCTUATION, NOT STYLE, AND NOT
        # DECORATION. The consuming parser accepts `cause=([A-Za-z0-9_]+)`. A parenthesis
        # CANNOT satisfy that class, which is the entire reason this value is excluded.
        #     cause=cause_unavailable  -> ACCEPTED as a stated cause  (the live miscount)
        #     cause=(unstated)         -> NOT ACCEPTED                (this line)
        #     cause=unstated           -> ACCEPTED                    (this line, "tidied")
        # *** DELETING TWO CHARACTERS FOR NEATNESS SILENTLY RESTORES THE INFLATION OF THE
        # COLUMN THE CAMPAIGN'S END CONDITION IS SCORED ON, AND NOTHING AT RUN TIME WILL SAY
        # SO. *** dev-4 asked for this comment; the guard below it is mine, because a rule that
        # has to be remembered is a habit, and this one is two keystrokes from being forgotten.
        # A TEST EXISTS FOR EXACTLY THIS: `lab/ladder_token.sh::sentinel_ladder` in the records
        # repository reads THIS literal and the parser's class FROM SOURCE and fails if the
        # sentinel ever becomes acceptable. If you change this line, run it.
        return None, "(unstated)", None
    return None, locator_cause, None


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


SPEED_EVIDENCE_INSTRUMENTS = frozenset({"rate_sweep"})
# THE DECIDING INSTRUMENTS THIS GUARD RECOGNISES, as a closed vocabulary --
# the same shape as `change_point_locator.DECLINE_REASONS`, and for the same
# reason: a vocabulary kept next to its only consumer cannot drift from it,
# and a new instrument is not admissible without being enumerated in the same
# edit that starts producing it. An unenumerated `rate_source` is REFUSED
# here, not tolerated: this gate stands in front of a destructive transform,
# so the safe direction of an unknown value is "no".


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
      2. THE LADDER MEDIAN >= FLOOR -- `resample_gate["verdict"] ==
         "confirmed"` and a REAL `median_fidelity` at or above
         `merge_video_resample.RESAMPLE_FIDELITY_FLOOR`. Read as a number,
         never as the mere presence of a key: `test_speed_ratio_against_master`
         sets `median_fidelity` to None on every decline BY DESIGN, so a
         truth-test on the key would read a decline as evidence.
      3. THE DECIDING INSTRUMENT -- `rate_source` names one of
         `SPEED_EVIDENCE_INSTRUMENTS`.

    EACH FAILURE HAS ITS OWN TOKEN, because a future correction acts
    differently on each (the Lead's granularity rule R1): a plan with no
    evidence at all is a producer that never ran this route; a plan whose
    rational is not named is a producer inventing factors; a plan whose median
    sits below the floor is a producer shipping a measured negative as a
    confirmation. The token travels in the refusal PROSE -- the raised
    `cause` stays the stable `speed_transform_not_validated`, because
    `chimeric_cause`'s own docstring bounds the tokened `chimeric_error`
    sites and this completion does not add a fourth.

    THIS FUNCTION ONLY EVER SAYS YES TO A NUMBER THAT THREE INDEPENDENT
    THINGS AGREE ON. It cannot say yes to a plan that merely looks confident.
    '''
    import merge_video_resample
    import pal_speed_discriminator

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
    if drift > pal_speed_discriminator.SNAP_RELATIVE_TOLERANCE:
        return False, "speed_evidence_ratio_is_not_the_snapped_rational", (
            f"the plan would apply speed_ratio={speed_ratio} while its "
            f"evidence is for {named} ({nominal}): relative drift {drift} "
            f"exceeds {pal_speed_discriminator.SNAP_RELATIVE_TOLERANCE}. "
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
            "the plan names an instrument but carries no resample_gate: the "
            "fidelity ladder's own result is missing, so the coefficient was "
            "never validated against the floor")
    if gate.get("verdict") != "confirmed":
        return False, "speed_evidence_gate_not_confirmed", (
            f"resample_gate verdict={gate.get('verdict')!r} "
            f"cause={gate.get('cause')!r}: the ladder did not confirm")
    median = gate.get("median_fidelity")
    if not isinstance(median, (int, float)) or isinstance(median, bool):
        # BLANK LAW: a gate that confirmed but carries no median has not shown
        # its measurement, and a missing number is not a passing number.
        return False, "speed_evidence_median_absent", (
            f"resample_gate says confirmed but median_fidelity={median!r} is "
            f"not a measured number")
    if median < merge_video_resample.RESAMPLE_FIDELITY_FLOOR:
        return False, "speed_evidence_median_below_floor", (
            f"ladder median {median} is below the unchanged floor "
            f"{merge_video_resample.RESAMPLE_FIDELITY_FLOOR}")

    return True, "speed_evidence_complete", (
        f"snapped named rational {named} (applied as {speed_ratio}), ladder "
        f"median {median} >= floor "
        f"{merge_video_resample.RESAMPLE_FIDELITY_FLOOR}, deciding instrument "
        f"{instrument}")


def build_repaired_video_object(candidate_obj, master_obj, plan, work_root, job_start_utc):
    '''Construit le fichier repare et l'objet video qui va avec.

    `job_start_utc`: EXIGE, SANS DEFAUT -- voir VMSAM_ERA a l'appelant
    (`repair_not_compatible_videos`). Traverse cette fonction sans etre lu:
    seul `assemble_on_master_timeline` (via `assemble_or_log_the_decline`) en
    a besoin, pour le tag pose au mux.

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

    # WHERE IT PLANTS (owner's decision, 2026-09-22, same hang investigation
    # as the entry log above). If this candidate's repair wedges anywhere
    # downstream, this is the last line that says where on disk its
    # intermediate and final artefacts were headed -- `work_dir` for the
    # per-track extraction/build files `assemble_on_master_timeline` writes,
    # `out_path` for the muxed product. Logged once, here, rather than
    # re-derived from `key` at investigation time: the derivation
    # (`stable_case_key`) is a hash, not something a reader reconstructs by
    # eye from a candidate path under time pressure.
    tools.dev_log(f"repair: build_repaired_video_object starting "
                  f"candidate={candidate_obj.filePath} work_dir={work_dir} "
                  f"out_path={out_path}\n")

    # STOP AND READ BEFORE POPULATING `plan["verdict"]` OR
    # `plan["speed_ratio"]`. Populating either routes THROUGH A DESTRUCTIVE
    # TRANSFORM on a real candidate file, applied below. If you are here for
    # a REPORTING reason -- a chain that wants to log its own speed_ratio or
    # margin -- STOP: emit on YOUR OWN decline line instead (see
    # `pal_speed_verdict.py`'s `speed_margin`, VMSAM_HELP_AI/dev-pal/
    # 013-speed-margin-producer.MD for the investigation that found this the
    # hard way). A hard guard immediately below refuses unconditionally
    # regardless, but the guard is the second line of defence -- this
    # comment is the first, so the next person gets the warning without
    # having to trace it themselves.
    speed_ratio, _refusal, _cause = get_speed_ratio(plan)
    if speed_ratio is not None:
        # HARD GUARD, unconditional -- Lead dispatch 2026-09-21
        # (VMSAM_HELP_AI/dev-pal/013-speed-margin-producer.MD). `speed_ratio`
        # reaching here applies a REAL asetrate transform to a real
        # candidate file, and nothing populates `plan["verdict"]`/
        # `plan["speed_ratio"]` anywhere in this codebase today -- this arm
        # has never fired in production. The investigation that found this
        # was chasing a REPORTING task (emitting `speed_margin` on the PAL
        # chain's own decline line); populating this plan for that reason
        # would have activated a destructive transform nobody reviewed.
        # Refusing here, unconditionally, turns that trap into a named
        # no-op: a well-lit signpost now leads somewhere safe instead of
        # somewhere destructive.
        #
        # WHY THIS IS NOT A PARAMETER ON A FINISHED CAPABILITY
        # (WRITE_ZONES.MD SS4's own rule against exactly that): this is
        # SS4's OTHER case -- a capability under construction, kept out of
        # the library until it is finished, not a setting on one that
        # already works. Removing this guard is the deliberate, reviewed
        # commit that ships Stage-3 confirmation as an APPLIED repair, not a
        # flag anyone flips at runtime. The finished capability ends up
        # unconditional either way, exactly as the no-parameter rule
        # demands -- this refusal is the state BEFORE that commit, not a
        # configuration of the state after it.
        #
        # REMOVAL CONDITION, STATED SO THIS GUARD DOES NOT OUTLIVE ITS
        # PURPOSE: remove this block in the commit that ships Stage 4
        # (resample application) as a reviewed, validated, applied repair
        # path -- not before. A guard without a stated exit outlives its
        # purpose and becomes the thing it was protecting against.
        #
        # `cause="speed_transform_not_validated"` is a FOURTH explicitly
        # tokened `chimeric_error` site. `chimeric_cause`'s own docstring
        # (this file) records that tokened sites were bounded to exactly two
        # by the Lead's own ruling (R2) -- this fourth one is authorized the
        # same way, by the Lead's explicit dispatch naming this exact token,
        # not assumed or added quietly.
        #
        # REVIEWED 2026-09-22, STILL HELD (dev-stage4's mission,
        # VMSAM_HELP_AI/dev-stage4/001-stage4-apply-review.MD): the apply
        # path below WAS reviewed and two real defects WERE found and fixed
        # in this same commit (the decline-path cause fall-through in
        # `get_plan_from_locator`/`confirm_speed_relation_via_resample`, and
        # the fabricated-marker mixed-sample-rate gap in
        # `get_marker_value_for`). The guard was lifted, tested end to end on
        # real media (errid 46 and three more episodes of the same pairing,
        # Rick and Morty S01E01-E04), and put back: all four real attempts
        # DECLINED at `assemble_on_master_timeline`'s own PRE-EXISTING
        # `alignment_contradicts_plan` verification, same cause, same piece,
        # same ~1.9-3.6s window every time -- a real, consistent, explicable
        # population defect in the only confirmed real population available
        # (this pairing needs the composite/resample-first-then-locate path
        # to ever merge, not a single-segment plan). ZERO merges were
        # produced. `alignment_contradicts_plan` verifies that offsets hold;
        # it does not verify the applied speed factor or the written marker
        # are correct -- those are exactly what remains unexercised, per the
        # Lead's ruling (2026-09-22). REMOVAL CONDITION UNCHANGED, now
        # sharpened: lift again in the commit where one real file MERGES and
        # its produced duration/marker match a hand-computed target stated
        # before the run.
        #
        # *** COMPLETED, NOT LIFTED (RULING_20260922_NO_BAND_ROUTING.MD,
        # ADDENDUM, 2026-09-22). *** The guard above refused EVERY non-None
        # `speed_ratio` unconditionally, which was right while no producer
        # could show its work and wrong the moment one could. What the
        # addendum orders is not a relaxation of the bar but a STATEMENT of
        # it: a coefficient is admissible when it arrives with the evidence
        # that validated it -- a snapped named rational, a ladder median at
        # or above the unchanged floor, and a named deciding instrument, all
        # three, all about the SAME number. `speed_plan_evidence` above is
        # that statement, and everything it cannot vouch for still lands on
        # the raise below, with the same stable cause it always had.
        #
        # THE REMOVAL CONDITION FROM THE BLOCK ABOVE IS NOT SATISFIED AND IS
        # NOT BEING TREATED AS SATISFIED. No real file has merged through
        # this path yet. What changed is that the refusal is now a
        # MEASUREMENT of the plan rather than a blanket "not yet": a plan
        # with no evidence is refused for a reason that names what is
        # missing, which is the difference between a wall and a gate.
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
        job_start_utc=job_start_utc,
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
        # STAGE 4's OWN CORROBORATION INPUT (Architect's ruling, 2026-09-17,
        # point i): the pair's own audio quantum, already a TOP-LEVEL plan
        # field the locator emits (`change_point_locator.py`'s own
        # `quantum_ms`) -- consumed here, never re-derived or produced.
        # `None` when the plan carries none, which the frame tier's stage 4
        # reads as "cannot corroborate" and declines named, never silently
        # skips the check.
        quantum_ms=plan.get("quantum_ms"),
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
    # LA PORTE DE LIVRAISON DES PISTES FABRIQUEES (CASE_wakeup20260924, defaut A).
    # Ici et pas dans `keep_best_audio`: c'est le seul point ouvert qui voit
    # TOUTES les pistes construites. En aval, `find_differences_and_keep_best_
    # audio` (gele) ne soumet a `keep_best_audio` que les pistes qui CORRELENT
    # entre elles, et les commentaires ne sont jamais soumis du tout -- une
    # piste fabriquee orpheline y passait sans course, marquee, jusqu'au produit.
    # `keep=False` pose ici est lu par `generate_new_file_audio_config`: la
    # piste n'entre jamais dans le fichier intermediaire.
    assembly["fabricated_dropped"] = gate_fabricated_delivery(
        repaired_obj, master_obj, work_dir=work_dir)
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


def check_ratio_convention(plan):
    """`speed_ratio` est-il dans MA convention? Renvoie `(refus, jeton)`.

    PAIRE `(prose, jeton)`, comme `get_plan_from_locator` rend `(plan, cause)`
    et `locate_change_points` rend `(plan, cause)`: le jeton est produit LA OU
    LA DECISION EST PRISE, jamais au site de journalisation. Un jeton pose chez
    l'appelant ne peut pas etre plus fin que le site d'appel, et ce site-ci
    couvre onze decisions distinctes.

    `(None, None)` quand il n'y a pas de refus.

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
        return None, None
    try:
        r = Decimal(str(ratio))
        expected = Decimal(str(master_s)) / Decimal(str(candidate_s))
    except Exception:
        return None, None
    if r == 0:
        return None, None
    direct = abs(r - expected)
    inverse = abs((Decimal(1) / r) - expected)
    near = expected * Decimal("0.01")
    if inverse <= near and direct > near:
        # JETON DISTINCT DE CELUI DE `check_ratio_labelled`, ET LA DIFFERENCE
        # EST CE SUR QUOI UN CORRECTIF AGIT. Ici RIEN N'EST DECLARE: la
        # convention est DEDUITE des durees du plan. La-bas elle est ECRITE par
        # le producteur. Corriger l'arithmetique d'un producteur muet et ecrire
        # un traducteur pour un producteur qui declare sa convention sont deux
        # gestes differents, donc deux jetons.
        return ("the speed_ratio looks like the RECIPROCAL of this module's "
                "convention: TASKS/009 defines r = master_span / candidate_span, "
                "and the value shipped matches candidate_span / master_span "
                "against the durations in the same plan. Applying it would "
                "resample in the WRONG DIRECTION"), "speed_ratio_reciprocal_vs_durations"
    return None, None


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
    """La convention est-elle DECLAREE, et est-ce la mienne? `(refus, jeton)`.

    TROIS REFUS DISTINCTS ET TROIS JETONS, parce qu'un correctif futur agit
    DIFFEREMMENT sur chacun -- c'est la regle de granularite du Lead (R1), et
    le defaut qu'elle vise n'est pas un jeton qui contredit sa prose, c'est
    deux refus qu'on corrigerait autrement portant la meme etiquette:

        convention declaree INVERSE      -> le producteur SAIT ce qu'il emet et
                                            c'est l'autre sens: un traducteur
                                            est ecrivable sans risque
        convention declaree ILLISIBLE    -> etendre le vocabulaire de
                                            `normalise_convention`
        AUCUNE convention, pres de 1     -> faire EMETTRE le champ au
                                            producteur. C'est le cas
                                            DESTRUCTEUR: 0.999001 contre
                                            1.000999 passe toute borne et
                                            toute tolerance

    Les trois disaient "la direction du coefficient n'est pas etablie", et un
    seul jeton pour les trois aurait rempli la colonne sans rien classer.
    """
    ratio = plan.get("speed_ratio")
    if ratio == None:
        return None, None
    stated = plan.get("speed_ratio_convention")
    if stated != None:
        named = normalise_convention(stated)
        if named == "mine":
            return None, None
        if named == "inverse":
            return (f"the plan states its ratio convention as {stated!r}, which "
                    f"is the RECIPROCAL of {RATIO_CONVENTION!r}; this module will "
                    f"not reinterpret a coefficient whose meaning it did not define"
                    ), "speed_convention_stated_reciprocal"
        # LA VALEUR DECLAREE VA DANS LA PROSE, PAS DANS LE JETON. `stated` est
        # du texte du producteur: un jeton qui la porterait varierait a chaque
        # fichier et ne s'agregerait pas. Regle lexicale 4.
        return (f"the plan states a ratio convention this module does not "
                f"recognise ({stated!r}); it applies {RATIO_CONVENTION!r} and a "
                f"convention it cannot read is not a convention it can trust"
                ), "speed_convention_unrecognised"
    try:
        distance = abs(Decimal(str(ratio)) - Decimal(1))
    except Exception:
        return None, None
    if distance < CONVENTION_FREE_MARGIN:
        return ("the plan carries no speed_ratio_convention and the ratio is "
                "within 1% of unity, where NEITHER the bounds check NOR the "
                "verifier can tell the two directions apart. An unlabelled "
                "near-unity coefficient is not applied"
                ), "speed_convention_absent_near_unity"
    return None, None


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


def check_candidate_admissibility(plan):
    """`RULING_20260916_PLAN_ADMISSIBILITY_NOT_TELEMETRY.MD`, Ruling 2 --
    the EXACT per-boundary invariant, checked ONCE at admission, before
    any extraction or assembly (`RULING_PRECONDITIONS_AT_ADMISSION`: a
    refusal must cost a probe, not a mux).

    NEVER `offset_monotone` (Ruling 1: it is a wrong-signed TELEMETRY
    proxy, schema-agreed as reporting whether `offset_max_abs_ms` is
    comparable across the summary line -- and its own definition counts
    a monotone-DECREASING offset sequence as `true`, which is precisely
    the shape that CAN read the candidate backwards. Gating on it would
    both miss real backward reads it calls `true` and refuse plans E2's
    own consistency gate was built to accept on `false`).

    For consecutive candidate-reading pieces, in the plan's own
    master-timeline order:
        candidate_start(next) >= candidate_end(prev)
    equivalently
        offset(next) - offset(prev) >= -(master_start(next) - master_end(prev))

    A plan may be non-monotone (offset decreasing between pieces) and
    still satisfy this at every boundary, when the master-side gap
    between the pieces absorbs the drop -- E2's own tolerated shape.
    Only a drop LARGER than the gap reads the candidate backwards.

    Returns `None` if every boundary is admissible. Otherwise a dict
    naming the FIRST violating boundary (not every one -- the plan is
    inadmissible after the first, and enumerating the rest would cost a
    probe measuring a boundary the first violation already discards)."""
    segments = parse_segments(plan.get("segments") or [])
    for index in range(1, len(segments)):
        prev_segment, next_segment = segments[index - 1], segments[index]
        master_gap = (next_segment["master_start_ms"]
                      - prev_segment["master_end_ms"])
        offset_delta = (next_segment["candidate_offset_ms"]
                        - prev_segment["candidate_offset_ms"])
        if offset_delta < -master_gap:
            return {"master_boundary_ms": str(prev_segment["master_end_ms"]),
                   "offset_delta_ms": str(offset_delta),
                   "master_gap_ms": str(master_gap),
                   "prev_segment_index": index - 1,
                   "next_segment_index": index}
    return None


def compare_plan_master(plan, best_video):
    """Le plan a-t-il ete mesure contre CE maitre? Rend `(raison, jeton)`.

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

    TROIS REFUS, TROIS JETONS, ET LE REGROUPEMENT SERAIT LE DEFAUT QUE CETTE
    FONCTION EXISTE DEJA POUR EVITER. Deux d'entre eux disent *je n'ai pas pu
    verifier* et le troisieme dit *j'ai verifie, et ils different*. Les fondre
    classerait un NEGATIF CONCLUANT comme une absence de preuve -- exactement
    la substitution que `BRIEF_COMMON.md` regle 5 nomme, et exactement la
    raison pour laquelle l'ancienne ligne unique disait `mesure contre un autre
    maitre` sur un plan qui portait un jeton opaque, ce qui etait FAUX.

    Et les correctifs different: un digest qui ne correspond pas se repare en
    normalisant CE QU'ON HACHE; un jeton incomparable se repare en APPRENANT le
    schema au lecteur; un maitre reellement different veut dire que le plan est
    PERIME et qu'il faut remesurer.
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
            return None, None
        return ("the plan's master path digest does not match this master's. "
                "NOTE: a path digest proves two agents were handed the same "
                "STRING, not the same FILE -- a symlink, a mount prefix or a "
                "different unicode normalisation differs here too, so this is "
                "UNVERIFIED rather than proof of a different master"
                ), "master_digest_mismatch_unverified"
    stated = plan.get("master_path")
    if stated == None or stated == best_video.filePath:
        return None, None
    if not str(stated).startswith("/"):
        return ("the plan names its master with a token this reader cannot "
                "compare to a filesystem path, so the master's identity is "
                "UNVERIFIED -- this is not evidence of a different master"
                ), "master_identity_token_uncomparable"
    # LE SEUL DES TROIS QUI AFFIRME QUELQUE CHOSE SUR LE MONDE. Les deux
    # au-dessus disent `UNVERIFIED`; celui-ci a compare et les chemins
    # different. Le jeton ne porte AUCUN des deux chemins -- une raison qui
    # contient un chemin media voyage avec lui.
    return ("the plan was measured against a different master than the "
            "one selected here"), "master_path_differs"


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
    n'existe pas avant la boucle: c'est `get_delay_language(best_video,
    candidate_obj)` qui la rend, et elle depend du candidat. Appeler avant la
    boucle voudrait dire ENUMERER les langues du maitre -- exactement ce que
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
    # UNE SEULE MESURE DU MAITRE PAR LANGUE DE COMPARAISON, et ce dictionnaire
    # EST ce "une seule". Voir `master_intertrack_verdict` juste au-dessus pour
    # la raison pour laquelle il vit ici et pas plus haut dans la fonction.
    master_intertrack_by_language = {}

    for candidate_path in list_not_compatible_video:
        # VMSAM_ERA (Architect's ruling, 2026-09-16): capture ICI, avant tout
        # declin, parce que c'est le debut du JOB sur CE candidat -- pas le
        # debut du mux, qui peut arriver bien plus tard ou jamais si le
        # candidat decline avant. Un candidat decline avant `mux_repaired_file`
        # calcule cette valeur pour rien (aucun fichier n'existe pour la
        # porter) -- sans cout, et plus honnete qu'un second point de capture
        # plus tard qui laisserait deux definitions possibles de "job start".
        job_start_utc = datetime.now(timezone.utc).isoformat()
        # WHICH FILE, BEFORE ANY WORK ON IT (owner's decision, 2026-09-22,
        # on a real 7-hour hang tonight: two containers logged the
        # "not compatible" line at mergeVideo.py:803, then NOTHING --
        # `repair_not_compatible_videos` is called from inside a bare
        # `except Exception`, and an except clause cannot catch a hang.
        # `merge_video_chimeric.py` carries three unbounded `subprocess.run`
        # calls downstream of here -- traced, not yet confirmed which one
        # runs the process into the ground). This line exists so that WHEN
        # this happens again, the last thing logged before silence names the
        # file, not just the fact that repair was entered at all. Emitted
        # BEFORE the object lookup below, which is itself cheap and cannot
        # hang -- the hang lives further down this loop, in the work that
        # follows once a plan exists.
        tools.dev_log(f"repair: repair_not_compatible_videos starting on "
                      f"{candidate_path}\n")
        candidate_obj = dict_file_path_obj.get(candidate_path)
        if candidate_obj == None:
            # UNE SEULE DECISION ICI, DONC UN SEUL JETON, et il n'est pas
            # grossier: la zone A a refuse un chemin dont elle n'a jamais porte
            # l'objet. Un correctif agit sur la comptabilite de la zone A.
            record(candidate_path, "declined",
                   "the rejected path has no video object in dict_file_path_obj",
                   cause="candidate_object_absent")
            # STOPGAP, not the fix: removes the fast-decline trigger of the
            # frozen fusion.py Pool.terminate() deadlock (CASE id 6) -- the
            # fix is the owner's initializer in fusion.py; remove this note
            # when that lands.
            _drain_audio_pools((best_video, candidate_obj),
                                "candidate_object_absent")
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
                   f"could not tell which language the merge measured on "
                   f"({language_route})", cause="language_undetermined")
            # STOPGAP, not the fix: removes the fast-decline trigger of the
            # frozen fusion.py Pool.terminate() deadlock (CASE id 6) -- the
            # fix is the owner's initializer in fusion.py; remove this note
            # when that lands.
            _drain_audio_pools((best_video, candidate_obj),
                                "language_undetermined")
            continue
        # ---- STEP 1: CLASSIFICATION -- LE MAITRE, CONTRE LUI-MEME ----------
        # Avant le test de vitesse, avant le chimerique, avant toute mesure du
        # candidat: le maitre est-il d'accord avec lui-meme sur LA LANGUE QU'ON
        # MESURE? Si non, le candidat n'a rien fait de mal et le mesurer contre
        # ce maitre produirait un delai qui depend de laquelle de ses propres
        # pistes le localisateur a tiree (`change_point_locator.py:1800`).
        # La classification SORT EN ERREUR ici, avec son jeton et ses nombres.
        master_intertrack = master_intertrack_verdict(
            best_video, language, master_intertrack_by_language)
        if master_intertrack != None and master_intertrack["verdict"] != None:
            # LE JETON EST CELUI DU MODULE QUI L'A MESURE, pas une constante
            # recopiee ici: un jeton duplique est un jeton qui divergera.
            # LES NOMBRES VOYAGENT AVEC LE REFUS -- instrument, decalage,
            # correlation, et LES DEUX PISTES -- parce que c'est exactement ce
            # qui manquait aux onze refus du dossier 86: la cause nommait le
            # candidat pendant que la preuve etait dans le maitre.
            # `verdict` EST DEJA UNE CLE DU VOCABULAIRE DE `detail_summary`
            # (ligne "plan_kind, verdict, plan_source, ..."), donc la ligne
            # `repair_detail: declined verdict=master_intertrack_desync` sort
            # INCONDITIONNELLEMENT sans qu'il faille toucher a ce resume. Le
            # dict complet (les deux pistes, le decalage, la correlation) part
            # dans le vidage verbeux garde par `tools.dev`, ou il a sa place.
            record(candidate_path, "declined", master_intertrack["reason"],
                   detail={"verdict": master_intertrack["verdict"],
                           "master_intertrack": master_intertrack},
                   cause=master_intertrack["verdict"])
            # STOPGAP, not the fix: removes the fast-decline trigger of the
            # frozen fusion.py Pool.terminate() deadlock (CASE id 6) -- the
            # fix is the owner's initializer in fusion.py; remove this note
            # when that lands.
            _drain_audio_pools((best_video, candidate_obj),
                                "master_intertrack_desync")
            # FALSIFIEUR (appendice A.7 du dossier id 6), et RIEN D'AUTRE. Le
            # blocage observe apres le stopgap precedent laissait DEUX lectures
            # indiscernables dans les traces: "le drain a tourne et a perdu la
            # course contre `Pool.terminate()`" ou "le drain lui-meme a bloque,
            # dans `wait_end_ffmpeg_progress_audio` -> `ApplyResult.get()`, sur
            # un worker mort qu'aucun `Pool` de CPython ne detecte". La derniere
            # ligne sortie avant le silence les separe. `log_always` et non
            # `dev_log`: une ligne qui ne sert qu'a lire un blocage ne peut pas
            # dependre d'un drapeau dont la valeur en production est justement
            # ce qu'on ne peut pas verifier pendant le blocage. Elle est posee
            # APRES le drain et AVANT le `continue`, parce que c'est exactement
            # cet intervalle qui est en question.
            tools.log_always(f"repair: drain complete at master_intertrack_desync "
                             f"for {candidate_path}\n")
            continue
        plan, plan_refusal_cause, plan_refusal_detail = get_plan_from_locator(
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
            #
            # LA RAISON, QUAND LE CONFIRMATEUR DE VITESSE EN PORTE UNE. Ajoute
            # 2026-09-22: `plan_refusal_detail` est non-None uniquement quand
            # `get_plan_from_locator` est passe par le confirmateur de Stage 2
            # (`describe_resample_decline`) -- ratio, fidelite mediane et
            # ecart au plancher, la ou avant seule `f"no plan from
            # {plan_source}"` atteignait cette ligne, quel que soit ce que la
            # mesure avait trouve.
            record(candidate_path, "no_plan",
                   plan_refusal_detail if plan_refusal_detail != None
                   else f"no plan from {plan_source}", cause=plan_refusal_cause)
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
                   {"plan_kind": "constant", "plan": plan},
                   # LE JETON NOMME CE QUE LA MESURE A RAPPORTE, PAS CE QUE LE
                   # FICHIER EST. `plan_kind_constant` aurait decrit le champ;
                   # celui-ci decrit la DECISION, et il dit au seat suivant ou
                   # frapper: elargir la couverture du locator, qui ne voit
                   # structurellement pas la tete du fichier. 3 des 26 declins
                   # mesures en production passent ici.
                   cause="plan_reports_no_change_point")
            continue
        speed_ratio, speed_refusal, speed_cause = get_speed_ratio(plan)
        if speed_refusal != None:
            # LE JETON VIENT DE `get_speed_ratio`, QUI L'A PRODUIT A LA
            # DECISION. Onze refus distincts arrivent ici et la prose est la
            # seule chose qui les separe -- un jeton pose sur cette ligne les
            # aurait tous appeles pareil, ce qui remplit la colonne et ne
            # classe rien.
            record(candidate_path, "declined", f"{speed_refusal} ({plan_source})",
                   {"plan_kind": plan.get("kind"), "verdict": plan.get("verdict")},
                   cause=speed_cause)
            continue
        if speed_ratio == None and not len(plan.get("segments") or []):
            record(candidate_path, "declined",
                   f"the measurement returned neither a segment nor a speed "
                   f"relation ({plan_source})",
                   cause="plan_has_neither_segment_nor_speed")
            continue
        master_check, master_cause = compare_plan_master(plan, best_video)
        if master_check != None:
            # LA RAISON NE PORTE PAS LE CHEMIN. `record` ecrit deja le fichier
            # sur sa propre ligne; une RAISON, elle, se cite -- dans un rapport,
            # dans un message a un autre agent, dans un resume -- et une raison
            # qui contient un chemin media voyage avec lui. On redige avant que
            # l'extrait ne parte, pas apres.
            # TROIS JETONS POSSIBLES, PRODUITS PAR `compare_plan_master`. Deux
            # veulent dire `je n'ai pas pu verifier` et un seul `j'ai verifie et
            # ils different`; les confondre remettrait un negatif concluant dans
            # le sac des absences de preuve.
            record(candidate_path, "declined", master_check, cause=master_cause)
            continue
        admissibility_violation = check_candidate_admissibility(plan)
        if admissibility_violation != None:
            # RULING_20260916_PLAN_ADMISSIBILITY_NOT_TELEMETRY.MD, Ruling 2:
            # checked HERE, at admission, before `build_repaired_video_object`
            # does any extraction -- a refusal costs a probe, not a mux.
            # The plan RAN and was MEASURED inadmissible; this is a decline,
            # not a could-not-measure, same as every other check on this path.
            record(candidate_path, "declined",
                   f"plan reads the candidate backwards at master boundary "
                   f"{admissibility_violation['master_boundary_ms']} ms: "
                   f"offset steps by {admissibility_violation['offset_delta_ms']} ms "
                   f"across a {admissibility_violation['master_gap_ms']} ms master gap "
                   f"({plan_source})",
                   admissibility_violation, cause="plan_reads_candidate_backwards")
            continue
        try:
            repaired_obj, assembly = build_repaired_video_object(
                candidate_obj, best_video, plan, work_root, job_start_utc)
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
                       decline_detail(error), cause=chimeric_cause(error))
                sys.stderr.write(f"repair: declined {candidate_path}: {error}\n")
            else:
                # LE MEME DETAIL SUR `failed` QUE SUR `declined`, ET C'EST CE
                # CHEMIN-CI QUI PRODUIT `NOVERDICT`: une panne d'outil apres le
                # mux laisse un artefact renomme, et le seul enregistrement qui
                # peut le nommer est celui-ci. L'omettre remettrait le fichier
                # hors de tout compte rendu, ce que le renommage existe pour
                # empecher.
                # LA CLASSE D'EXCEPTION ENTRE DANS LA PROSE, ET LE JETON RESTE
                # FIXE. `str(error)` seul ne porte PAS le nom de la classe:
                # une `TypeError` et une `OSError` arrivaient ici avec le seul
                # message, donc la partie qui varie n'atteignait AUCUN artefact
                # -- ni le jeton (regle lexicale 4 l'interdit) ni la prose. Le
                # jeton unique n'avait alors RIEN pour etre decoupe.
                #
                # ET C'EST PRECISEMENT PARCE QUE CE SITE N'A JAMAIS TOURNE EN
                # PRODUCTION (0 ligne sur 59 artefacts) QUE CA NE POUVAIT PAS
                # ATTENDRE: le jour ou il tourne, ce premier artefact est
                # TOUTE la base de preuve, et une classe absente est absente
                # pour toujours. On n'ajoute pas un champ a un chemin mort, on
                # rend lisible sa premiere levee. Autorise par le Lead, hors du
                # perimetre des sept tampons.
                record(candidate_path, "failed",
                       f"{type(error).__name__}: {error}", decline_detail(error),
                       cause="repair_raised_unhandled")
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
                  "fabricated_dropped": assembly.get("fabricated_dropped") or [],
                  "verification": assembly["verification"]}
        reason = (f"{len(assembly['audios'])} audio and "
                  f"{len(assembly['subtitles'])} subtitle track(s) rebuilt "
                  f"(rebuilt in the repair object; final delivery decided later "
                  f"by keep_best_audio), "
                  f"{len(assembly['declined'])} declined, "
                  f"{len(assembly['failed'])} failed, "
                  # CASE_wakeup20260924: retirees AVANT la livraison par
                  # `gate_fabricated_delivery` -- chacune a sa ligne
                  # `repair: fabricated_dropped cause=...` plus haut.
                  f"{len(assembly.get('fabricated_dropped') or [])} fabricated "
                  f"dropped before delivery")
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
