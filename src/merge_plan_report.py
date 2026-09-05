"""Le rapport `.merge_plan.log`: ce qui a ete fait au fichier, lisible par un humain.

`SPEC_ZONE_A.MD` s4g. Un diagramme avec les horodatages, les regions perdues en
ROUGE AU-DESSUS des valeurs de decalage, TOUTE valeur de decalage reellement
utilisee -- empruntees comprises -- une table de provenance par region, et le
meme contenu une seconde fois en phrases. Le diagramme et les phrases sont la
meme information deux fois, VOULU: le journal est lu par qui ouvre le journal,
le rapport par qui ouvre le fichier.

CE MODULE NE PRODUIT AUCUNE DONNEE. Il LIT les octets emis par la reparation et
les rend. Un champ manquant est un defaut adresse a son producteur, JAMAIS un
correctif ici. C'est la raison d'etre de la zone: on ne peut pas dessiner une
table de provenance sans decouvrir quelles regions n'en ont aucune.

--- POURQUOI CINQ ETATS PAR CELLULE ET NON UN BLANC ---

Un champ que MON lecteur ne trouve pas doit etre distinguable d'un champ que le
producteur n'emet pas. Le 4 septembre un controle a rapporte "24 sur 24 corrects"
en comparant deux `None` sur des cles abregees. Aujourd'hui ces deux cas
partagent un blanc; ici ils ne le partagent plus:

    PRESENT       lu dans les octets, PAR NOM.
    DERIVED       calcule a partir d'autres champs emis, avec la derivation
                  nommee. Ce n'est PAS `present`: personne ne l'a ecrit.
    COLLAPSED     le producteur l'emet, AGREGE au-dessus de la granularite que
                  cette cellule demande. Un blanc etiquete est un constat; un
                  blanc rempli est un mensonge.
    ABSENT_FORMAT ce journal est d'un format anterieur au champ. Le producteur
                  a HEAD l'emet. NON un defaut -- un artefact plus vieux.
    NO_PRODUCER   aucun producteur ne l'emet a HEAD. C'EST le defaut, et il
                  s'adresse a une fonction nommee.

--- DEUX FAITS D'OCTETS DONT DEPEND CE LECTEUR ---

1. LA PREMIERE LIGNE `repair:` N'EST PAS PRECEDEE D'UNE LIGNE VIDE. Les entrees
   finissent par `\\n` et `fusion.py` les joint par `\\n`, donc toutes les lignes
   `repair:` SONT separees par une ligne vide -- sauf la premiere, qui suit
   `Logs:\\n` directement. Un lecteur qui decoupe sur `"\\n\\nrepair: "` obtient
   19 enregistrements la ou il y a 20 lignes, et CELLE QU'IL PERD dans le format
   courant est `repair: master` -- LA LIGNE QUI NOMME LE MAITRE, celle dont
   l'absence rend une causation indecidable. On decoupe donc EN LIGNES et on
   filtre par PREFIXE.

2. UN FICHIER DE `KEEP` N'EST PAS UN JOURNAL DE TRAVAIL PARCE QU'IL FINIT EN
   `.log`. On rejette PAR STRUCTURE -- absence de ligne `repair: plan` -- et
   jamais par nom ni par compte. Un denominateur defendu par un nom de fichier
   est defendu jusqu'au prochain nom.

--- ET LES CHAMPS SE RESOLVENT PAR NOM, JAMAIS PAR POSITION ---

Un retainer a lu la colonne 2 comme un statut la ou l'ecrivain avait mis
`size_bytes`. Ici s'ajoute un piege propre a ces lignes: `tolerance_ms=500` EST
emis et n'est PAS une des deux portes d'absorption -- c'est la tolerance de
duree de la ligne `output file`. `quantum=129` est le quantum median de
correlation, pas une porte non plus. Trois quantites, deux emises, aucune celle
que s4e demande. Une resolution par forme remplirait la cellule `plateau` avec
500.
"""

from decimal import Decimal
import html
import hashlib
import re

# ---------------------------------------------------------------------------
# LA BASE DE TOUTE AFFIRMATION A L'ECHELLE DU CORPUS
#
# Plusieurs cellules de ce rapport citent des faits qui ne viennent PAS de
# l'artefact rendu -- "zero occurrence sur N artefacts", "le plus court segment
# du corpus fait 80 s". Ce sont des affirmations sur une POPULATION, ecrites
# dans un document qui parle d'UN fichier, et elles etaient des constantes
# codees en dur sans denominateur ni date.
#
# DEUX DEFAUTS DANS UN. Elles perimeront en silence -- un artefact de plus et le
# `zero` est faux, sans que rien ne bouge dans le code. Et elles ne disaient pas
# de quelle population elles parlaient, alors que CETTE POPULATION N'EST PAS
# INDEPENDANTE: mesure ici, 18 journaux pour 15 cas distincts, parce que trois
# journaux decrivent le meme cas 297 par trois chemins de code et deux le meme
# cas 169.
#
# `n_distinct(quoi)` EST LA QUESTION, et le nom du champ la cache. dev-2 a
# publie `n_distinct_release = 12` pour 9 cas -- plus de releases que de cas --
# et l'a lu sans broncher parce que le nombre penchait du cote flatteur. Sa
# garde, gratuite et retenue ici: UN COMPTE DE DISTINCTS NE PEUT JAMAIS EXCEDER
# SA POPULATION, et rien ne le verifiait.
#
# ET L'UNITE RESTE AMBIGUE, ce qui se dit plutot que se tranche en silence: un
# "cas" est ici un couple (maitre, candidat), mais quatre journaux de laboratoire
# ne nomment pas leur candidat et sont regroupes par leur maitre seul. Deux
# candidats differents fusionnes vers un meme maitre compteraient donc pour un.
# UNE CONSTANTE VRAIE EST PIRE QU'UNE CONSTANTE FAUSSE, et c'est la propriete
# que dev-2 a isolee sur ma premiere version. Une constante FAUSSE se rattrape en
# la re-derivant une fois. Une constante VRAIE ne se rattrape qu'en la
# re-derivant PLUS TARD, ce que personne ne planifie -- elle etait exacte le jour
# ou je l'ai ecrite et devient fausse sans que rien ne bouge dans le code.
#
# On retire donc la possibilite au lieu de la surveiller, comme partout ailleurs
# ici: LA BASE N'EST PLUS UNE CONSTANTE, ELLE EST UN PARAMETRE. Qui rend le
# rapport la MESURE et la passe; s'il ne la passe pas, le rapport DIT qu'aucune
# population n'a ete fournie, au lieu de citer un chiffre qui fut vrai.
def measure_corpus(log_paths, read=None):
    """Mesure la population, a l'instant du rendu, sur les fichiers donnes.

    Un "cas" est un couple (maitre, candidat). Quand un journal ne nomme pas son
    candidat -- ce qu'aucun n'a le droit d'emettre en clair -- on retombe sur le
    maitre seul, et DEUX CANDIDATS DISTINCTS VERS UN MEME MAITRE COMPTERAIENT
    POUR UN. L'unite est donc rendue avec son doute plutot que tranchee en
    silence.
    """
    reader = read or (lambda path: open(path, encoding="utf-8",
                                        errors="replace").read())
    keys, logs, rejected, basis = set(), 0, 0, {"digest": 0, "derived": 0,
                                                 "master_only": 0}
    # LA REPARTITION DES BUILDS, PAS SEULEMENT LEUR COHERENCE.
    #
    # dev-2 a mesure que ses 88 lignes de media reel portent UNE SEULE empreinte
    # de build, et que son analyse epingle ce build expres pour que 150 fichiers
    # soient comparables -- donc sa population NE PEUT PAS structurellement
    # exercer un jeton ajoute apres. Il a mis la divulgation DANS l'instrument,
    # au point ou le nombre est produit, plutot que dans un message.
    #
    # Ma ligne CORPUS ne portait que `build_vs_sources`, qui est une COHERENCE
    # et pas une REPARTITION -- et "aucun digest de sources ne porte deux builds
    # differents" se lit tres bien comme "un seul build", ce que ce corpus n'est
    # pas.
    build_spread = {}
    # LE RECENSEMENT DES REMPLISSAGES, CALCULE ICI ET PLUS JAMAIS TAPE.
    #
    # dev-2: `33 of 41` n'existait dans aucun instrument -- le code imprimait des
    # COMPTES et il composait le rapport a la main dans un message. "Le correctif
    # n'est pas une meilleure phrase, c'est de produire le denominateur DANS LE
    # CODE, la ou personne n'a a le composer."
    #
    # Ma narration disait "23 journaux, 42 interieures, 39 d'entre elles, 17 en
    # queue". Mesure aujourd'hui: 28, 55, 55, 20. CHAQUE NOMBRE FAUX, dans un
    # document qui se rend a neuf a chaque fois. Vrais quand tapes, faux une
    # heure plus tard, et rien dans le texte ne vieillissait avec eux.
    census = {"head": [], "interior": [], "tail": []}
    # ON CLASSE LES REJETS AU LIEU D'AFFIRMER LEUR CAUSE -- voir rejected_note.
    rejected_kind = {"died": 0, "merged": 0, "other": 0}
    # DEUX POPULATIONS SOUS UN SEUL COMPTE, et elles se separent PAR STRUCTURE.
    # Un journal de travail de production porte l'enveloppe `Logs:` posee par
    # `fusion.py`; un rejeu de laboratoire commence directement a `repair:`.
    # Les melanger sous `20 journaux` disait qu'ils repondent a la meme question,
    # et ils n'y repondent pas: un rejeu exerce le chemin de reparation, pas une
    # fusion. dev-2 vient de trouver dans un de ses harnais un import qui
    # REMPLACE la configuration d'execution -- la paire a fusionne, le calage a
    # eu lieu, puis la fusion est morte et AUCUN FICHIER N'A ETE ECRIT. Un rejeu
    # peut donc produire un journal sans produire d'artefact, ce qu'un journal de
    # production ne peut pas faire.
    provenance = {"production_job_log": 0, "lab_replay": 0}
    # HOW MANY OPPORTUNITIES THE REPLAY SUBSET ACTUALLY CONTRIBUTES.
    #
    # The provenance row prints `37 production job logs, 8 lab replays` and the
    # census row prints `m = 19`. Both are true, and a reader composes the wrong
    # inference between them: a bigger log count reads as more evidence. Measured
    # by ci and reproduced here -- the 8 replays add 24 fills of which 21 are AT
    # THE BOUND, so the interior denominator grew 70 -> 88 while gaining ZERO
    # chances for the claim to fail. `45 logs` is not more evidence than `37` for
    # the head or interior claim.
    #
    # This is the at-bound exclusion one level up: the members that cannot fail
    # were counted out of the COUNT, and nobody asked whether the CORPUS
    # EXTENSION had added any that could. It survived a re-audit an hour earlier
    # because a published correction feels finished in a way a claim does not.
    #
    # ci's rule, adopted: ask of a corpus extension what the opportunity rule
    # asks of a check -- how many of the files you added could have changed the
    # answer? The qualifier is emitted here, beside the number it qualifies, so
    # it cannot be lost by a reader who reads one row and not the other.
    census_replay = {"head": [], "interior": [], "tail": []}
    # AN ABSENT LINE IS NOT AN EXCLUSION.
    #
    # A merge denominator gets its "excluded" files by SUBTRACTION, and the two
    # populations below are what that subtraction silently merges:
    #   excluded_claimed            -- handed to this render, no plan emitted.
    #                                  These are the files a denominator would
    #                                  subtract.
    #   excluded_with_stated_cause  -- of those, the ones that ALSO emitted a
    #                                  cause anyone can read.
    # MEASURED HERE, 2026-09-05: 31 and 0. Every file that would be subtracted is
    # SILENT, while 8 files WITH a plan DO carry a declared cause -- which is the
    # positive control that the emitter and this parser both work. The causes
    # exist; they are absent precisely on the files being removed from the count.
    #
    # "excluded because completely different" and "declined for some other reason"
    # are THE SAME OBSERVABLE from here: silence. So a silent file is NOT counted
    # as excluded by this reader -- it is counted as UNEXAMINED, and the row says
    # so. Subtracting silent files is how a campaign declares victory over the
    # files it never looked at, and from outside that is indistinguishable from
    # having merged them.
    excluded = {"claimed": 0, "with_stated_cause": 0, "cause_anywhere": 0}
    # L'ETAT DE LA PORTE AU MOMENT OU CES FICHIERS ONT ETE PRODUITS.
    # Le proprietaire a tranche que `enforcing` passe a True. Un artefact portant
    # `would_refuse=True` a donc ete produit SOUS UNE PORTE INERTE et ne serait
    # PAS produit par la configuration qui expedie: il serait DECLINE. Ce corpus
    # n'est donc pas la population que le conteneur produit maintenant, et un
    # lecteur qui compte des artefacts reussis compte des fichiers qui
    # n'existeraient plus.
    gate = {"enforcing_true": 0, "enforcing_false": 0, "would_refuse_true": 0}
    # UN CONTROLE GRATUIT, OFFERT PAR dev-2, ET IL EST DU TYPE QUI SE VIOLE
    # PLUTOT QUE DE SE VERIFIER. `build` couvre ses deux modules, `sources` les
    # 27 fichiers que l'image expedie, ET SES DEUX MODULES SONT DANS LES 27.
    # Donc `build` peut bouger avec `sources`, et `sources` peut bouger seul --
    # mais `build` QUI BOUGE ALORS QUE `sources` NE BOUGE PAS EST IMPOSSIBLE.
    # Pas "non observe": impossible. Si cela arrive, un des deux emetteurs est
    # casse.
    #
    # Cela vaut d'etre code parce que la violation est le seul signal: tant que
    # rien ne se contredit, le controle ne dit rien, et c'est exactement ce
    # qu'on veut d'un invariant.
    by_sources = {}
    brackets = {"compared": 0, "agree": 0, "disagree": 0,
                "emitted_true": 0, "emitted_false": 0, "incomparable": 0}
    for path in log_paths:
        text = reader(path)
        _job_for_exclusion = parse_job_log(text) if is_job_log(text) else {}
        _cause = _job_for_exclusion.get("declined")
        if _cause:
            excluded["cause_anywhere"] += 1
        if not (_job_for_exclusion.get("plan") or {}).get("pieces"):
            excluded["claimed"] += 1
            if _cause:
                excluded["with_stated_cause"] += 1
        if not is_job_log(text):
            # UN FICHIER ECARTE PAR MOI EST UNE ABSENCE QUE JE FABRIQUE. Le
            # rejet par structure est correct -- pas de ligne de plan, pas de
            # geometrie a rendre -- mais le SILENCE l'est pas: une reparation
            # morte AVANT d'avoir planifie emet un journal sans plan, donc mon
            # lecteur l'ecarte et mon corpus ne l'a jamais compte. C'est le "un
            # journal manquant ne laisse aucune trace dans les journaux" de
            # dev-2, UN CRAN PLUS BAS ET DANS MON PROPRE DENOMINATEUR: le
            # fichier existe, c'est moi qui le fais disparaitre.
            rejected += 1
            rejected_kind["died" if "Traceback" in text else
                          "merged" if "first_delay_test" in text else
                          "other"] += 1
            continue
        logs += 1
        is_lab_replay = not any(line.startswith("Logs:")
                                for line in text.splitlines())
        if is_lab_replay:
            provenance["lab_replay"] += 1
        else:
            provenance["production_job_log"] += 1
        job = parse_job_log(text)
        if job.get("sources") and job.get("build"):
            key = job["sources"]["digest"]
            by_sources.setdefault(key, set()).add(
                tuple(sorted(job["build"].items())))
        check = job.get("output_check") or {}
        if check:
            if str(check.get("enforcing")).lower().startswith("true"):
                gate["enforcing_true"] += 1
            else:
                gate["enforcing_false"] += 1
            # `would_refuse=True -- 1 WOULD HAVE BEEN DECLINED (gate inert)` est
            # une forme apparue apres coup: la valeur porte de la prose. On teste
            # donc le PREFIXE et non l'egalite, ce qu'une comparaison stricte
            # aurait rate en silence.
            if str(check.get("would_refuse")).lower().startswith("true"):
                gate["would_refuse_true"] += 1
        # LA QUALITE DE LA CLE VARIE DANS LE CORPUS ET SE DIT. Un digest emis est
        # une LECTURE; un id derive du chemin par moi est une CONSTRUCTION; un
        # maitre seul est une INFERENCE qui regroupe des cas distincts. Publier
        # un compte unique sans dire de quoi chaque cle est faite serait la meme
        # faute qu'un `n_distinct` sans unite.
        pieces = (job.get("plan") or {}).get("pieces") or []
        for index, piece in enumerate(pieces):
            if piece.get("source") != "master":
                continue
            width = int(float(piece["master_end_ms"])) - int(float(piece["master_start_ms"]))
            position = ("head" if index == 0 else
                        "tail" if index == len(pieces) - 1 else "interior")
            census[position].append(width)
            if is_lab_replay:
                census_replay[position].append(width)
        # LES DEUX LECTURES DU MEME CROCHET, COMPTEES ET NON RECITEES.
        #
        # `bound_only` est desormais emis; la signature de largeur que ce
        # rapport utilisait avant qu'il existe est imprimee a cote. Leur accord
        # etait ECRIT EN DUR -- "16 sur 16, 11 vrais, 5 faux" -- mesure sur un
        # corpus de 5 journaux et jamais recompte. Il est compte ici, a chaque
        # rendu, sur les journaux qu'on me tend.
        for entry in job.get("brackets") or []:
            width, flag = entry.get("width_ms"), entry.get("bound_only")
            if width is None or flag is None:
                brackets["incomparable"] += 1
                continue
            brackets["compared"] += 1
            emitted = str(flag).strip().lower().startswith("true")
            derived = abs(float(width) - float(SEARCH_BOUND_MS)) < 1e-9
            brackets["emitted_true" if emitted else "emitted_false"] += 1
            brackets["agree" if emitted == derived else "disagree"] += 1
        if job.get("build"):
            build_spread[tuple(sorted(job["build"].items()))] = (
                build_spread.get(tuple(sorted(job["build"].items())), 0) + 1)
        if job.get("candidate_digest"):
            basis["digest"] += 1
            candidate = ("digest", job["candidate_digest"])
        elif job.get("candidate_opaque_id"):
            basis["derived"] += 1
            candidate = ("derived", job["candidate_opaque_id"])
        else:
            basis["master_only"] += 1
            candidate = None
        keys.add((job.get("master_opaque_id"), candidate))
    # LA GARDE DE dev-2, GRATUITE: un compte de distincts ne peut jamais exceder
    # sa population. dev-2 a publie 12 releases pour 9 cas -- plus de releases
    # que de cas -- et l'a lu sans broncher parce que le nombre penchait du cote
    # flatteur. Rien ne le verifiait.
    # THE SPLIT MUST SUM. A "MEASURED SPLIT" whose parts do not add to its total
    # is a denominator lying in prose, and nothing checked it: the note named two
    # buckets while the classifier had three. It happened to sum only because the
    # third was empty, and an empty branch reads exactly like an absent one.
    assert (rejected_kind["died"] + rejected_kind["merged"]
            + rejected_kind["other"] == rejected), (
        f"reject split does not partition: {rejected_kind} against {rejected} -- "
        f"the note would under-describe the class it claims to have measured")
    assert len(keys) <= logs, (
        f"n_distinct ({len(keys)}) exceeds n ({logs}): a distinct-count cannot "
        f"exceed its population")
    return {
        "logs": logs,
        # LES NOMBRES QUE LA PROSE CITE, RENDUS COMME LIGNE.
        "fills_above_the_bound": (
            f"{sum(1 for w in census['head'] + census['tail'] + census['interior'] if w > SEARCH_BOUND_MS)} "
            f"master fill(s) here exceed the 100 s search bound: "
            f"{sum(1 for w in census['head'] if w > SEARCH_BOUND_MS)} at the head, "
            f"{sum(1 for w in census['tail'] if w > SEARCH_BOUND_MS)} at the tail, "
            f"{sum(1 for w in census['interior'] if w > SEARCH_BOUND_MS)} INTERIOR"),
        # LE DENOMINATEUR NE COMPTE QUE CE QUI POUVAIT TOMBER DE L'AUTRE COTE.
        #
        # 100000 = 25 x 4000, EXACTEMENT. Toute largeur egale a la borne de
        # recherche est donc sur la grille de raffinement PAR CONSTRUCTION et ne
        # peut pas falsifier l'affirmation. Je publiais "interieur 88/88": 69 de
        # ces 88 ne pouvaient pas echouer. Trouve par forensic, et le nombre
        # corrige est PLUS FORT parce qu'il est plus petit -- 19 sur 19 sont des
        # largeurs que le localisateur a REELLEMENT raffinees, et toutes les 19
        # tombent sur son pas.
        #
        # Les fills a la borne portent l'AUTRE affirmation, celle de la
        # signature `bound_only`, verifiee separement contre le champ emis.
        # Deux affirmations, deux populations, aucune ne prete son
        # denominateur a l'autre.
        # THE GRID CLAIM IS RETRACTED HERE, IN THE CODE THAT USED TO ASSERT IT.
        #
        # This row reported "interior 19 of 88 could have gone either way, of
        # which 19 land on the 4000 ms grid". THAT DENOMINATOR IS AN ARITHMETIC
        # IDENTITY AND ITS m IS ZERO. Read at change_point_locator.py (frozen;
        # resolve the blob yourself) -- a refined bracket returns
        # `(last_before, first_after + REFINE_WINDOW)` and probes step by
        # REFINE_STEP, so its width is k*REFINE_STEP + REFINE_WINDOW; a
        # bound-only bracket returns `(region_start, region_end + PROBE_WINDOW)`
        # with coarse probes PROBE_STEP apart, so its width is
        # k*PROBE_STEP + PROBE_WINDOW. ALL FOUR CONSTANTS ARE MULTIPLES OF
        # REFINE_STEP, so EVERY width lands on the grid BY CONSTRUCTION.
        #
        # I corrected this claim twice -- removing the at-bound class, then
        # clearing the above-bound class -- and never asked whether the members
        # that REMAINED could fail. They cannot. A published correction feels
        # finished in a way a published claim does not.
        #
        # And SEARCH_BOUND = 25 x REFINE_STEP was my decomposition; it names no
        # mechanism. SPEC_ZONE_A.MD 4h has the causal one and always did:
        # 100000 = PROBE_STEP + PROBE_WINDOW, the signature of a bracket that
        # was NEVER REFINED. Both arithmetics are true; one names a cause.
        #
        # The counts stay -- they are the DISTRIBUTION, which is informative --
        # but the row no longer offers them as a test that was passed.
        "fill_census": (
            f"master fills by position: "
            f"head {len(census['head'])}, interior {len(census['interior'])}, "
            f"tail {len(census['tail'])}. "
            f"THE GRID PROPERTY IS NOT A TEST AND ITS OPPORTUNITY COUNT IS ZERO: "
            f"a refined bracket's width is k x {REFINE_STEP_MS} + "
            f"{REFINE_WINDOW_MS} ms and an unrefined one's is "
            f"k x {PROBE_STEP_MS} + {PROBE_WINDOW_MS} ms, and ALL FOUR CONSTANTS "
            f"ARE MULTIPLES OF {REFINE_STEP_MS} ms, so EVERY width lands on the "
            f"{REFINE_STEP_MS} ms grid BY CONSTRUCTION. m = 0, UNTESTED, NOT "
            f"PASSED. The {SEARCH_BOUND_MS} ms bound is "
            f"{PROBE_STEP_MS} + {PROBE_WINDOW_MS} -- the signature of a bracket "
            f"that was NEVER REFINED, which is the finding that survives: A FILL "
            f"REGION'S WIDTH IS A PROPERTY OF THE INSTRUMENT AND NOT OF THE "
            f"MEDIA. The distribution is reported below because it is "
            f"informative; it is not offered as a check that passed: "
            + "; ".join(
                (lambda at_bound, rest: (
                    f"{name} {len(widths)} fill(s), of which {len(at_bound)} "
                    f"sit exactly at the bound (never refined) and "
                    f"{len(rest)} do not"))(
                    [w for w in widths if w == SEARCH_BOUND_MS],
                    [w for w in widths if w != SEARCH_BOUND_MS])
                for name, widths in (("head", census["head"]),
                                     ("interior", census["interior"]),
                                     ("tail", census["tail"])))
            + ". distinct interior widths: "
            + " ".join(f"{w}x{census['interior'].count(w)}"
                       for w in sorted(set(census["interior"])))
            + ". tail widths: "
            + " ".join(str(w) for w in sorted(census["tail"]))
            + ((". OF THIS POPULATION, THE LAB REPLAYS CONTRIBUTE: "
                + "; ".join(
                    f"{name} {len(widths)} fill(s) of which "
                    f"{sum(1 for w in widths if w != SEARCH_BOUND_MS)} "
                    f"could have gone either way"
                    for name, widths in (("head", census_replay["head"]),
                                         ("interior", census_replay["interior"]),
                                         ("tail", census_replay["tail"])))
                + f". A LOG COUNT IS NOT AN OPPORTUNITY COUNT: these "
                  f"{provenance['lab_replay']} replays add "
                  f"{sum(len(v) for v in census_replay.values())} fill(s) and "
                  f"{sum(1 for v in census_replay.values() for w in v if w != SEARCH_BOUND_MS)} "
                  f"opportunit(y/ies). Ask of a corpus extension what the "
                  f"opportunity count asks of a check: how many of the files "
                  f"added could have changed the answer?")
               if provenance["lab_replay"] else "")),
        # DEUX NOMBRES, ET LE DENOMINATEUR EST UNE OCCASION DE SE TROMPER.
        # `agree/compared` ne vaut rien si `compared` vaut 0: la signature
        # derivee ne peut pas etre CONTREDITE par un corpus qui n'emet pas le
        # champ. Le compte des deux valeurs emises est donne pour la meme
        # raison: un corpus ou `bound_only` est toujours vrai ne discrimine
        # rien, quel que soit l'accord.
        "bracket_agreement": (
            f"{brackets['agree']} of {brackets['compared']} emitted brackets "
            f"agree with the derived signature `width == the {SEARCH_BOUND_MS} "
            f"ms search bound`"
            + (f"; OF WHICH COULD HAVE DISAGREED: {brackets['compared']} "
               f"({brackets['emitted_true']} emitted True, "
               f"{brackets['emitted_false']} emitted False, so the corpus "
               f"carries both values and the agreement is not a constant)"
               if brackets["compared"] and brackets["emitted_true"]
               and brackets["emitted_false"] else
               f"; OF WHICH COULD HAVE DISAGREED: 0 -- "
               + ("no emitted bracket in this corpus carries both `width_ms` "
                  "and `bound_only`, so the signature is UNTESTED here, not "
                  "confirmed"
                  if not brackets["compared"] else
                  "every emitted `bound_only` in this corpus carries the SAME "
                  "value, so agreement is what a constant reader would also "
                  "score and this corpus does not discriminate"))
            + (f". {brackets['disagree']} DISAGREE -- read the BRACKET rows: "
               f"the locator and this reader have parted company"
               if brackets["disagree"] else "")
            + (f". {brackets['incomparable']} bracket(s) carry only one of the "
               f"two fields and are not scored either way"
               if brackets["incomparable"] else "")),
        "build_spread": (
            f"{len(build_spread)} distinct `repair: build` fingerprint(s) across "
            f"{sum(build_spread.values())} log(s) that carry one; "
            f"{logs - sum(build_spread.values())} carry NONE and are undatable. "
            f"EVERY CORPUS-SCALE CLAIM BELOW SPANS THAT SPREAD -- it is not a "
            f"statement about one build, and this reader cannot say which build "
            f"produced which row: a build value is a content digest with no "
            f"ordering"),
        # ET LA POPULATION EST UNE CONSERVATION, PAS UN RECENSEMENT.
        #
        # dev-2 a quote "123 of 150" toute la nuit, ou 150 est SON PLAFOND
        # D'HORLOGE et le corpus eligible en compte 277 -- un denominateur en
        # forme de PLAN, lu comme une couverture a chaque fois.
        #
        # Le mien est en forme de CONSERVATION. Mine is shaped like a
        # PRESERVATION. Measured BY SYSCALL on the inodes of the logs handed to
        # this render -- `os.stat`, never a directory walk -- 2026-09-05:
        # 45 logs, 45 distinct inodes, st_nlink {1: 38, 2: 7}, so 38 OF 45 are
        # at nlink=1. Their original was deleted and the preserved copy is the
        # only one that exists.
        #
        # THE PREVIOUS FIGURE HERE WAS `21 sur 30`, TYPED, AND IT DRIFTED WITH
        # THE CORPUS -- the proportion held (70% -> 84%) but the number did not.
        # A typed number in a comment is the same defect as a typed number in
        # prose, one hop further from anything that recomputes it. It is dated
        # here rather than silently corrected.
        #
        # AND THE METHOD MATTERS MORE THAN THE FIGURE: a peer's nlink scan was
        # retracted for appending found-paths into a dict and printing len() as
        # though it were st_nlink, while walking only three directories -- so it
        # COULD NOT PRINT ANYTHING BUT 1 for a file linked outside its traversal.
        # The control that separates this measurement from that one is that the
        # distribution above HAS TWO KEYS: it can distinguish nlink=1 from
        # nlink>1, which the retracted scan structurally could not. Donc ce
        # que je compte n'est pas "les journaux produits", c'est "les journaux
        # que quelqu'un a pense a lier avant que le coureur les efface". Un
        # journal jamais preserve ne laisse AUCUNE trace ici, pas meme un trou.
        "population_is_a_preservation": (
            "the logs handed to this render are a PRESERVED SUBSET, not a "
            "census. They survive because something hard-linked them before the "
            "runner deleted its sources; a log never preserved leaves no trace "
            "here, not even a gap. This reader cannot state how many were "
            "produced, only how many it was given"),
        # THE PAIR, EMITTED SIDE BY SIDE SO A DIFFERENCE IS VISIBLE WITHOUT A JOIN.
        "excluded_claimed": excluded["claimed"],
        "excluded_with_stated_cause": excluded["with_stated_cause"],
        "exclusion_note": (
            f"AN ABSENT LINE IS NOT AN EXCLUSION. {excluded['claimed']} file(s) "
            f"handed to this render emitted NO PLAN -- these are the ones a merge "
            f"denominator would SUBTRACT -- and of them "
            f"{excluded['with_stated_cause']} carry a cause anyone can read. "
            + (f"THE OTHER {excluded['claimed'] - excluded['with_stated_cause']} "
               f"ARE SILENT AND THIS READER DOES NOT COUNT THEM AS EXCLUDED: they "
               f"are UNEXAMINED. `excluded because completely different` and "
               f"`declined for some other reason` are the same observable from "
               f"here, and only one of them is an exclusion. "
               if excluded["claimed"] != excluded["with_stated_cause"] else
               "The two counts are EQUAL on this material, which means this "
               "column had no opportunity to discriminate here -- m = 0 for the "
               "difference, and that is UNTESTED, not clean. ")
            + f"POSITIVE CONTROL: {excluded['cause_anywhere']} log(s) in this "
              f"population DO carry a declared cause, so the emitter and this "
              f"parser both fire; an absence below is a property of the file and "
              f"not of the instrument"),
        "rejected_by_structure": rejected,
        # UNE CAUSE AFFIRMEE POUR UNE CLASSE QUI EN A DEUX.
        #
        # Cette note disait "une reparation morte avant de planifier emet
        # exactement cela". Mesure sur les octets: VRAI POUR 21 SUR 23 ET FAUX
        # POUR DEUX -- ceux-la passent par le chemin de FUSION avec un
        # `first_delay_test`, aucune reparation n'est tentee, ET ILS PRODUISENT
        # UN FICHIER. C'est la moitie consequente: "mort avant de planifier"
        # laisse entendre qu'il n'y a pas d'artefact, alors que cette classe en
        # a un, que ce rapport ne decrira jamais.
        #
        # forensic a mesure 48 ms de divergence de tete sur l'un des deux,
        # `824b3b2a213ec817`: trois pistes, deux FLAC a first_pts 0.048 et une
        # AAC a 0.000 -- et c'est l'AAC, la piste PRIMEE, qui commence le plus
        # TOT. La divergence va donc a l'ENVERS de l'artefact de codec, qui ne
        # peut pas l'expliquer.
        #
        # DONC ON COMPTE AU LIEU D'EXPLIQUER.
        "rejected_note": (
            f"handed to this render and NOT counted: they carry no `repair: "
            f"plan` line, so there is no plan in those bytes to describe. "
            f"MEASURED SPLIT of this class, not assumed: {rejected_kind['died']} "
            f"carry a Traceback -- a repair that died before planning, which "
            f"leaves no artefact -- and {rejected_kind['merged']} went through "
            f"the MERGE path with a delay test and NO repair attempted, WHICH "
            f"DO PRODUCE A FILE. Every claim below is blind to both classes, "
            f"and the second has artefacts this report will never describe at "
            f"all. AND A THIRD BUCKET EXISTS AND IS NAMED HERE EVEN WHEN EMPTY: "
            f"{rejected_kind['other']} match NEITHER pattern. "
            + ("Zero today, which is an UNEXERCISED BRANCH and not a "
               "reassurance -- m = 0 for this bucket. A log recording that the "
               "INSTRUMENT DID NOT RUN (a tool that could not read its input) "
               "carries no Traceback and no delay test, so it would land here "
               "and be counted as a refusal by anyone reading the two named "
               "classes as the whole split. `vmsam-ci` measures that class in "
               "its own corpus, where it is not zero: RELAYED, its figure, "
               "about its container substrate, 2026-09-05."
               if not rejected_kind["other"] else
               "THIS BUCKET IS NON-EMPTY: a decline class exists that neither "
               "named pattern describes, and it may be a NON-OBSERVATION "
               "counted as a refusal. Read those logs before quoting any "
               "decline-cause census over this corpus.")),
        "distinct_cases": len(keys),
        "unit": "(master, candidate) pair",
        "key_basis": f"{basis['digest']} from an emitted candidate_digest (read), "
                     f"{basis['derived']} from a path digest I derive myself "
                     f"(constructed), {basis['master_only']} from the master "
                     f"alone (INFERRED: two candidates merged toward one master "
                     f"would count as one)",
        "build_vs_sources": (
            "consistent: no `sources` digest carries two different `build` "
            "digests"
            if all(len(v) <= 1 for v in by_sources.values())
            else "CONTRADICTION: " + ", ".join(
                f"sources={k} carries {len(v)} different build digests"
                for k, v in by_sources.items() if len(v) > 1)
            + ". `build` cannot move while `sources` holds still -- the two "
              "repair modules are inside the 27 shipped files -- so one of the "
              "two emitters is broken")
            if by_sources else "not checkable: no artefact carries both lines",
        "gate_state": f"{gate['enforcing_false']} produced with the duration "
                      f"gate INERT (enforcing=False), {gate['enforcing_true']} "
                      f"with it enforcing. {gate['would_refuse_true']} carry "
                      f"would_refuse=True, so under the SHIPPING configuration "
                      f"they would be DECLINED and would not exist as produced "
                      f"files. THIS CORPUS IS NOT THE POPULATION THE CONTAINER "
                      f"NOW PRODUCES",
        "provenance": f"{provenance['production_job_log']} production job logs "
                      f"(carrying the `Logs:` envelope), "
                      f"{provenance['lab_replay']} lab replays (starting at "
                      f"`repair:`). NOT one population: a replay exercises the "
                      f"repair path and not a merge, and can emit a log without "
                      f"producing an artefact",
        "measured": "at render time, over the logs handed to this render",
        # NI INDEPENDANT NI NON BIAISE, et le second est le plus recent des deux.
        # Ces journaux n'existent que parce qu'une reparation est allee au bout.
        # dev-2 a mesure que son verificateur decode des fenetres de 20 s et que
        # le cout depend du CODEC: quelques millisecondes en EAC3, des dizaines
        # de secondes en TrueHD 7.1 sans perte. Les fichiers chers depassent le
        # delai et ne produisent pas de journal. MON CORPUS HERITE DONC D'UN
        # ECHANTILLONNAGE QUI FAVORISE LES FICHIERS BON MARCHE A DECODER, dans
        # une direction que je ne peux pas quantifier depuis ici.
        # CORRIGE: J'AVAIS AFFIRME UNE PERTE LA OU IL N'Y A QU'UN COUT.
        # J'ecrivais "les fichiers chers depassent le delai et n'emettent aucun
        # journal". dev-2 a retire cette conclusion: son cas 27 a coute des
        # DIZAINES DE MINUTES contre deux a cinq pour les autres -- l'asymetrie
        # de cout est measured -- ET IL A FINI. Un fichier cher qui n'a pas
        # expire etablit le cout, pas la perte. Zero instance connue d'un
        # journal perdu de cette maniere.
        #
        # J'avais repris sa formule "direction connue, ampleur inconnue" et
        # traite la direction comme si elle impliquait des occurrences. Un
        # mecanisme n'est pas une frequence, et je l'ai ecrit dans un rapport
        # comme si ca l'etait.
        # CORRIGE, ET DANS LE SENS FAIBLE. J'allais ecrire qu'un fichier
        # NOVERDICT garde son nom de produit et echappe a tout compte. dev-2 l'a
        # renomme entre-temps: `<nom>.NOVERDICT.<ext>`, classe `unadjudicated`
        # chez ci, non compte comme produit. L'artefact n'est donc plus
        # invisible -- ce qui rend ce caveat MESURABLE au lieu d'etre seulement
        # declare, via `state=NOVERDICT` sur la ligne `undelivered`.
        "caveat": "NOT an independent sample: these logs exist only where a "
                  "repair COMPLETED -- which is not the same as `produced`. A "
                  "run that failed on a tool fault can leave a log and an "
                  "artefact under a NOVERDICT name; that class is countable "
                  "from `undelivered state=` and is not counted here. A COST ASYMMETRY IS MEASURED UPSTREAM -- "
                  "probe decoding is far dearer on lossless multichannel than "
                  "on EAC3 -- but NO LOSS IS ESTABLISHED: zero known instances "
                  "of a log missing for that reason, and the one expensive case "
                  "measured did finish. A mechanism is not a frequency. And "
                  "undatable from any artefact -- no build or timestamp field "
                  "is emitted anywhere",
    }


# ---------------------------------------------------------------------------
# Etats de cellule

PRESENT = "present"
DERIVED = "derived"
COLLAPSED = "collapsed"
ABSENT_FORMAT = "absent-from-this-format"
NO_PRODUCER = "no-producer"
# LA QUANTITE N'EXISTE PAS DANS CE CAS, et la decision a quand meme ete prise --
# par une autre statistique, nommee. Etat distinct parce que L'ACTION est
# distincte, qui est le seul test qui vaille pour un etat: `no-producer` dit
# depose un defaut, `collapsed` dit demande au producteur de faire descendre la
# valeur, et celui-ci ne demande RIEN -- il dit va lire l'autre champ. Le ranger
# sous `collapsed` aurait dit "agrege au-dessus de cette granularite" d'une
# grandeur qui n'est pas agregee mais indefinie: une collision d'etats dans le
# module qui existe pour empecher les collisions.
NOT_DEFINED = "not-defined-here"
# LA BRANCHE EST VIVANTE ET CE CORPUS N'EN PRODUIT PAS L'ENTREE. Distinct de
# `not-defined-here`, qui dit que la grandeur N'EXISTE PAS dans ce cas; celui-ci
# dit qu'elle existerait si le cas se presentait. Et distinct de tous les autres
# par L'ACTION, qui reste le seul test: `no-producer` dit depose un defaut,
# `collapsed` dit demande au producteur, `not-defined-here` dit va lire l'autre
# champ, et celui-ci ne demande RIEN -- il dit ne lis pas ce blanc comme une
# reponse negative, ET SACHE QUE CE CHEMIN N'A JAMAIS SERVI.
#
# Il arrive apres l'avoir rencontre TROIS FOIS en le traitant a chaque fois a la
# main: la boite ambre dont le corpus ne produit pas l'entree, le rendu d'une
# piste acceleree qui n'a jamais tourne, et `frame_rate_original` qui ne se
# declenche qu'en cas de desaccord. Trois cas particuliers etaient un etat.
NOT_EXERCISED = "not-exercised-here"
# AUCUNE POPULATION N'A ETE FOURNIE A CE RENDU. Distinct de tout le reste: ce
# n'est pas une propriete du code NI des donnees, c'est une propriete de
# L'APPEL. Un blanc ici se lirait comme "la population va de soi".
NOT_SUPPLIED = "not-supplied-to-this-render"
# LA MESURE A ETE TENTEE ET N'A PAS ABOUTI. Distinct de `not-exercised-here`, et
# c'est la distinction que dev-2 m'a fait chercher: `not-exercised-here` dit
# L'ENTREE NE S'EST PAS PRESENTEE, celui-ci dit ON NE SAIT PAS si elle s'est
# presentee, parce que la tentative n'a pas fini. L'action differe, seul test
# valable: le premier ne demande rien, le second demande DE REMESURER avec un
# budget qui convient.
#
# Sans lui, un depassement de delai se serait range sous `not-exercised-here` et
# aurait dit `ce cas ne se produit pas` la ou la verite est `on n'a pas
# regarde`. dev-2 a failli me livrer exactement cette phrase -- "aucun SKIPPED
# du lot" -- alors qu'un cas sur trois avait ete mesure et deux tues a 44
# minutes sur un decodage TrueHD.
NOT_MEASURED = "not-measured"

_STATE_MARK = {
    PRESENT: "",
    DERIVED: "~",           # calcule, pas lu
    COLLAPSED: "^",         # emis, mais agrege au-dessus de cette granularite
    ABSENT_FORMAT: "-",     # ce format ne le portait pas
    NO_PRODUCER: "x",       # personne ne l'emet
    NOT_DEFINED: "\u00b7",   # sans objet ici; la decision est ailleurs
    NOT_EXERCISED: "\u25cb",  # branche vivante, entree non produite ici
    NOT_SUPPLIED: "?",       # l'appelant n'a pas fourni la population
    NOT_MEASURED: "\u2049",   # tentee, non aboutie -- a remesurer
}

_STATE_WORD = {
    PRESENT: "read from the bytes by name",
    DERIVED: "derived from other emitted fields",
    COLLAPSED: "emitted, but aggregated above this granularity",
    ABSENT_FORMAT: "absent from this artefact's format",
    NO_PRODUCER: "no producer emits this",
    NOT_DEFINED: "not defined in this case; the decision was made elsewhere",
    NOT_EXERCISED: "the branch is live; this artefact does not produce its input",
    NOT_SUPPLIED: "the caller supplied no population for this render",
    NOT_MEASURED: "the measurement was attempted and did not complete; "
                  "re-measure, do not read as a negative",
}


class Cell:
    """Une valeur ET son etat. Les deux voyagent ensemble ou aucun des deux."""

    __slots__ = ("value", "state", "note")

    def __init__(self, value, state, note=None):
        self.value = value
        self.state = state
        self.note = note

    def __bool__(self):
        return self.state in (PRESENT, DERIVED)

    def text(self):
        if self.state in (PRESENT, DERIVED):
            return f"{_STATE_MARK[self.state]}{self.value}"
        return _STATE_MARK[self.state]

    def long(self):
        if self.state in (PRESENT, DERIVED):
            return f"{self.value}  ({_STATE_WORD[self.state]}"\
                   f"{'; ' + self.note if self.note else ''})"
        return f"{_STATE_WORD[self.state]}"\
               f"{'; ' + self.note if self.note else ''}"


def absent(note=None):
    return Cell(None, ABSENT_FORMAT, note)


def no_producer(note):
    return Cell(None, NO_PRODUCER, note)


# ---------------------------------------------------------------------------
# Lecture. PAR NOM.

def split_fields(text):
    """`key=value` par NOM. Une valeur peut contenir des ESPACES.

    Trouve sur les octets reels et pas sur une fixture: la ligne `output file`
    porte `source=master video Duration (mediainfo)`, quatre mots, sans
    guillemets. Un decoupage sur les espaces rendait `source=master` -- une
    valeur tronquee qui se lit comme une valeur complete, ce qui est pire
    qu'une absence.

    On repere donc les DEBUTS DE CLE a la profondeur zero et on prend pour
    valeur tout ce qui court jusqu'a la cle suivante. Cela tient aussi
    `speed=none(no rate proposed by the measurement)` et
    `fill=master/ja[FILL SOURCE SHORT BY 11.0 ms; ...]`, dont les espaces sont
    proteges par des parentheses ou des crochets.
    """
    depth, starts = 0, []
    for index, character in enumerate(text):
        if character in "[(":
            depth += 1
            continue
        if character in "])":
            depth = max(0, depth - 1)
            continue
        if depth or character != "=":
            continue
        back = index
        while back > 0 and (text[back - 1].isalnum() or text[back - 1] == "_"):
            back -= 1
        name = text[back:index]
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if back > 0 and not text[back - 1].isspace():
            continue
        starts.append((back, name, index + 1))

    fields = {}
    for position, (back, name, value_start) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(text)
        fields.setdefault(name, text[value_start:end].strip())
    return fields


def _decimal(text):
    try:
        return Decimal(str(text))
    except Exception:
        return None


def _trim(value):
    """1479979.000000000 -> 1479979. Les zeros de queue sont du bruit a l'oeil."""
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def is_job_log(text):
    """PAR STRUCTURE. Un `.log` sans ligne de plan n'est pas un journal de travail."""
    return any(line.startswith("repair: plan ") for line in text.splitlines())


def parse_job_log(text):
    """Les octets emis -> enregistrements. Decoupage EN LIGNES, filtre par PREFIXE.

    Ne decoupe JAMAIS sur `"\\n\\nrepair: "`: cela perd `repair: master`.
    """
    lines = text.splitlines()
    job = {
        "master_line_present": False,
        "master_opaque_id": None,
        "master_path": None,
        "candidate_opaque_id": None,
        "candidate_path": None,
        "candidate_digest": None,
        "plan": None,
        "audios": {},
        "subtitles": [],
        "regions_added": {},
        "regions_cut": {},
        "regions_used": {},
        "skipped": [],
        "refused": [],
        "declined": None,
        "failed": None,
        "undelivered": None,
        "build": None,
        "sources": None,
        "unparsed": [],
        "predicted_refusals": [],
        "prediction": None,
        "output_check": None,
        "summary_counts": None,
        "foreign_lines": [],
        "locator_measurements": [],
        "locator_notes": [],
        "brackets": [],
        "segments": [],
        "output_durations": None,
    }

    for line in lines:
        if not line.startswith("repair: "):
            # s4e: un saut hors du flux `repair:` existe et se dit. Un lecteur a
            # prefixe le laisse tomber, donc on le compte au lieu de l'ignorer.
            stripped_line = line.strip()

            # `[change_point_locator]` N'EST PAS DU TEXTE ETRANGER, ET N'EST PAS
            # NON PLUS A FAIRE TAIRE.
            #
            # Demande du Lead: l'ajouter au tuple ci-dessous pour que le compteur
            # `foreign_lines` cesse de tirer a chaque reparation reussie. Le motif
            # est juste -- une ligne TOUJOURS presente est du vocabulaire, pas un
            # saut hors du vocabulaire -- mais l'implementation demandee ne peut
            # pas marcher et le silence serait mon propre pire defaut:
            #
            #   a) `_log` ecrit `f"\t\t[change_point_locator] {message}\n"`
            #      (change_point_locator.py:233). DEUX TABULATIONS. `splitlines`
            #      les garde, donc `line.startswith("[change_point_locator]")`
            #      rend False et le tuple n'aurait rien change. On teste la ligne
            #      DESINDENTEE.
            #   b) une liste de silence est une position qui porte un nom. Trois
            #      fois deja -- `rms_over_floor`, `repair: DECLINED`, `build` --
            #      j'ai laisse tomber une valeur emise parce que ma structure ne
            #      l'attendait pas. Cette ligne PORTE LES MESURES DU LOCALISATEUR:
            #      `offset_ms`, `points`, `quantum_ms`, `window_s`, `segments`,
            #      `change_points`. Les taire pour arreter un compteur echangerait
            #      exactement ce que ce rapport existe pour montrer.
            #
            # Donc: reconnue PAR NOM et rendue, jamais comptee comme etrangere.
            if stripped_line.startswith(_LOCATOR_TAG):
                rest = stripped_line[len(_LOCATOR_TAG):].strip()
                fields = split_fields(rest)
                if fields and "=" not in " ".join(fields.keys()):
                    job["locator_measurements"].append(fields)
                else:
                    # LES VINGT-QUATRE AUTRES APPELS.
                    #
                    # Le Lead et ci decrivent tous deux "une ligne de succes une
                    # fois par paire". Mesure: `_log` a VINGT-CINQ appelants
                    # (grep -c '^\s*_log(' = 25) et la porte `if tools.dev:` est
                    # sur `_log` lui-meme, donc l'ouvrir les ouvre tous. Vingt-
                    # quatre sont du texte libre -- des declins, des probes
                    # ratees -- et :324 est `f"probe at {s}s failed: {error}"` ou
                    # `{error}` est une exception ffprobe ARBITRAIRE, qui porte
                    # couramment le chemin d'entree. C'est le fuite du s8, mot
                    # pour mot. Donc le fait, jamais le texte -- comme dehors.
                    job["locator_notes"].append({
                        "digest": hashlib.md5(
                            stripped_line.encode("utf-8", "replace")).hexdigest()[:12],
                        "chars": len(stripped_line),
                        "carries_path": bool(_PATH.search(stripped_line)),
                        "tail": re.sub(r"[^a-z ]", "",
                                       stripped_line.lower()[-42:]).strip(),
                    })
                continue

            if stripped_line and not line.startswith(("Merged", "Logs:", "We was",
                                                      "Multiple delay")):
                # ON GARDE LE FAIT, PAS LE TEXTE.
                #
                # s4e veut qu'un element ecarte se dise: une omission se lit
                # "il n'y en avait pas". Mais ces lignes sont du texte libre
                # hors du vocabulaire de la reparation, et la ligne reelle
                # observee est
                #   `['<chemin>'] not compatible with the others videos`
                # ou le chemin PORTE DES ESPACES, donc un titre de serie
                # survit a toute redaction par motif -- j'ai mesure
                # `Suicide Squad` traversant la mienne.
                #
                # UN REDACTEUR QUI NE PEUT PAS GARANTIR SON RESULTAT NE DOIT PAS
                # ETRE LA DEFENSE. On enregistre donc l'EXISTENCE de la ligne,
                # sa longueur, son empreinte stable et si elle contient un
                # chemin -- de quoi la retrouver dans le journal, qui lui ne
                # voyage pas -- et jamais son contenu.
                stripped = line.strip()
                job["foreign_lines"].append({
                    "digest": hashlib.md5(
                        stripped.encode("utf-8", "replace")).hexdigest()[:12],
                    "chars": len(stripped),
                    "carries_path": bool(_PATH.search(stripped)),
                    "tail": re.sub(r"[^a-z ]", "", stripped.lower()[-42:]).strip(),
                })
            continue
        body = line[len("repair: "):]

        if body.startswith("master "):
            job["master_line_present"] = True
            # LE NOM A COTE DE L'IDENTIFIANT, JAMAIS A SA PLACE. Un id retire
            # est un id que plus personne ne peut utiliser; un id garde a cote
            # coute seize caracteres et laisse a tout agent citant ce rapport
            # une forme sure sous la main -- il n'a pas a choisir entre citer un
            # nom et ne rien citer.
            job["master_path"] = body[len("master "):].strip()
            job["master_opaque_id"] = opaque_id(job["master_path"])
            continue

        if body.startswith("candidate_digest "):
            # LE CANDIDAT IDENTIFIE SANS ETRE NOMME. dev-2 ne pouvait pas emettre
            # son chemin -- c'est exactement ce que s8 interdit -- donc mon unite
            # de corpus retombait sur le maitre seul et deux candidats vers un
            # meme maitre comptaient pour un. Le digest resout cela SANS emettre
            # le chemin: la cle devient une lecture au lieu d'une inference.
            job["candidate_digest"] = body[len("candidate_digest "):].strip()
            continue

        if body.startswith("plan "):
            rest = body[len("plan "):]
            kind = rest.split(" ", 1)[0]
            fields = split_fields(rest)
            pieces_text = rest.partition("pieces=")[2]

            # BALAYAGE, PAS RUSTINE. dev-2: "diagnostiquer cette classe produit
            # une PHRASE, et une phrase n'enumere pas ses instances". J'avais
            # nomme le defaut sur la ligne TRACK, corrige UNE instance, et il
            # en restait CINQ. Enumerees mecaniquement, puis MESUREES: une seule
            # laissait reellement tomber un champ emis aujourd'hui
            # (`dropped_segments` sur la ligne de plan). Les cinq sont corrigees
            # quand meme -- le defaut est structurel, pas la valeur du jour.
            job["plan"] = dict(fields)
            job["plan"].update({
                "kind": kind,
                "language": fields.get("language"),
                "quantum_ms": fields.get("quantum"),
                "speed_margin": fields.get("speed_margin"),
                # LES TROIS COMPAGNONS DE LA MARGE. Je les avais ajoutes a la
                # LIGNE sans les faire porter par le PLAN: la ligne les
                # demandait, le dictionnaire ne les avait pas, et ils
                # disparaissaient en silence. Trouve par le controle, pas en
                # relisant -- une cle ecrite d'un cote et pas de l'autre ne se
                # voit d'aucun des deux cotes.
                "speed_margin_absent_reason": fields.get("speed_margin_absent_reason"),
                "fidelity_margin": fields.get("fidelity_margin"),
                "decided_by": fields.get("decided_by"),
                "pieces": parse_pieces(pieces_text),
            })
            continue

        matched = re.match(r"audio track (\d+) (.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(2))
            # `residual=probes=4 worst=0.01q quantum=129ms` est un GROUPE: le
            # premier membre se lit `residual=probes=4`, les autres sont freres.
            # L'ancienne forme `residual=(4,0.08,129)` mettait un compte, un
            # rapport et une duree dans une parenthese sans nom ni unite, et un
            # lecteur l'a prise pour un decalage de 4.0 ms -- c'etaient QUATRE
            # SONDES. On resout donc `probes` par NOM, y compris imbrique.
            residual = fields.get("residual")
            if residual is not None and "=" in residual:
                inner, _, value = residual.partition("=")
                fields.setdefault(inner, value)
            job["audios"][order] = fields
            continue

        matched = re.match(r"ADDED audio track (\d+) (\w+) ([\d.]+)-([\d.]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(5))
            # PAR NOM, ET LE RESTE PASSE.
            #
            # Cette liste etait FIXE -- kind, deux bornes, `from` -- et elle a
            # laisse tomber `why=`, `stream=` et `offset_ms=` en silence. dev-2
            # a ajoute `why=` POUR CE RAPPORT, pour qu'un remplissage maitre
            # dise sa CAUSE (`head_gap`, `interior_bracket`, `tail_gap`,
            # `unreported(...)`), et il n'a jamais atteint le rendu: 19
            # occurrences dans mes journaux, ZERO dans mes lignes.
            #
            # C'est le defaut que j'ai nomme et corrige sur la ligne TRACK il y a
            # des heures -- UNE LISTE BLANCHE EST UNE POSITION QUI PORTE UN NOM
            # -- et que je n'ai jamais applique ici. dev-2 demandait si je
            # TRONQUE ou si je SAUTE: je saute, et sa deuxieme hypothese etait la
            # bonne.
            entry = {name: value for name, value in fields.items()}
            entry.update({
                "kind": matched.group(2),
                "master_start_ms": _decimal(matched.group(3)),
                "master_end_ms": _decimal(matched.group(4)),
                "from": fields.get("from"),
            })
            job["regions_added"].setdefault(order, []).append(entry)
            continue

        # LA LIGNE `USED`. Les regions que la sortie PREND DU CANDIDAT, avec le
        # decalage REELLEMENT APPLIQUE, par piste et par region. Elle n'existait
        # pas: `ADDED` couvre le remplissage maitre, `CUT` le candidat jete, et
        # la majorite de chaque fichier -- ce qui vient du candidat -- n'avait
        # aucune ligne. Le decalage etait alors DERIVABLE et pas EMIS, et la
        # derivation dependait du plan ayant coupe quelque chose.
        matched = re.match(r"USED audio track (\d+) master ([\d.-]+)-([\d.-]+) "
                           r"candidate ([\d.-]+)-([\d.-]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(6))
            used = dict(fields)
            used.update({
                "master_start_ms": _decimal(matched.group(2)),
                "master_end_ms": _decimal(matched.group(3)),
                "candidate_start_ms": _decimal(matched.group(4)),
                "candidate_end_ms": _decimal(matched.group(5)),
                # PAR NOM. Un `-` de tete est un signe, pas une absence: un
                # decalage negatif est legitime (le candidat devance le maitre).
                "offset_ms": _decimal(fields.get("offset_ms")),
                "offset_ms_present": "offset_ms" in fields,
            })
            job["regions_used"].setdefault(order, []).append(used)
            continue

        matched = re.match(r"CUT audio track (\d+) candidate ([\d.]+)-([\d.?]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(4))
            end_text = matched.group(3)
            cut = dict(fields)
            cut.update({
                "candidate_start_ms": _decimal(matched.group(2)),
                "candidate_end_ms": None if end_text == "?" else _decimal(end_text),
                "dropped_ms": (None if fields.get("dropped_ms") == "UNMEASURED"
                               else _decimal(fields.get("dropped_ms"))),
                "dropped_unmeasured": fields.get("dropped_ms") == "UNMEASURED",
                "where": fields.get("where"),
            })
            job["regions_cut"].setdefault(order, []).append(cut)
            continue

        matched = re.match(r"subtitle track (\d+) (.*)$", body)
        if matched:
            fields = split_fields(matched.group(2))
            fields["stream_order"] = int(matched.group(1))
            job["subtitles"].append(fields)
            continue

        if body.startswith("sources "):
            # LE DIGEST DES FICHIERS QUI EXPEDIENT, distinct de `build`. dev-2:
            # `build` couvre SES DEUX MODULES, `sources` couvre LES 27 `.py` QUE
            # LE Dockerfile COPIE. L'un bouge quand ses modules bougent, l'autre
            # quand n'importe quoi d'expedie bouge -- NE PAS LIRE L'UN COMME UN
            # RAFFINEMENT DE L'AUTRE.
            rest = body[len("sources "):]
            digest = rest.split()[0] if rest.split() else None
            fields = split_fields(rest)

            # BALAYAGE, PAS RUSTINE. dev-2: "diagnostiquer cette classe produit
            # une PHRASE, et une phrase n'enumere pas ses instances". J'avais
            # nomme le defaut sur la ligne TRACK, corrige UNE instance, et il
            # en restait CINQ. Enumerees mecaniquement, puis MESUREES: une seule
            # laissait reellement tomber un champ emis aujourd'hui
            # (`dropped_segments` sur la ligne de plan). Les cinq sont corrigees
            # quand meme -- le defaut est structurel, pas la valeur du jour.
            job["sources"] = dict(fields)
            job["sources"]["digest"] = digest
            continue

        if body.startswith("build "):
            # L'IDENTITE DE BUILD, ARRIVEE. C'est la cellule `build_identity`
            # deposee le premier jour -- le journal disait ce qui avait ete fait
            # et jamais QUELLE VERSION l'avait fait.
            #
            # ET MON LECTEUR LA JETAIT EN SILENCE. Le repli des lignes non
            # reconnues, ajoute une heure plus tot pour `DECLINED`, l'a attrapee
            # a sa toute premiere execution sur le corpus reel: LA PREMIERE
            # CHOSE QU'IL A SAUVEE EST LA REPONSE A MON PLUS VIEUX DEFAUT FILE,
            # que j'aurais continue a rapporter comme manquant.
            job["build"] = {}
            for token in body[len("build "):].split():
                name, _, digest = token.rpartition(":")
                if name and digest:
                    job["build"][name] = digest
            continue

        # DEUX PREFIXES TERMINAUX ET NON UN, parce que le pilote les classe
        # differemment: `DECLINED` est une DECISION de la porte, `FAILED` est une
        # panne d'outil qui s'est echappee AVANT qu'un verdict existe. Un seul
        # prefixe absorberait chaque echec d'ffprobe dans le cout de la porte --
        # c'est mon `no-producer` contre `not-measured`, dans un nom de ligne.
        #
        # ECRIT CONTRE UNE FORME ANNONCEE. dev-2 les a decrites; je n'ai pas
        # passe ce lecteur sur ses octets, donc la jonction N'EST PAS FAITE.
        if body.startswith("FAILED"):
            job["failed"] = body[len("FAILED"):].strip(": ").strip() or "(no reason given)"
            continue

        if body.startswith("undelivered "):
            # PAR NOM: `state=` et `path=`. Et SON ABSENCE EST UN TROISIEME FAIT
            # et pas une ligne manquante -- elle n'est emise que si un fichier a
            # ete marque, donc pas de ligne veut dire QU'AUCUN ARTEFACT
            # N'EXISTAIT a marquer: la levee est arrivee avant le mux.
            fields = split_fields(body[len("undelivered "):])

            # BALAYAGE, PAS RUSTINE. dev-2: "diagnostiquer cette classe produit
            # une PHRASE, et une phrase n'enumere pas ses instances". J'avais
            # nomme le defaut sur la ligne TRACK, corrige UNE instance, et il
            # en restait CINQ. Enumerees mecaniquement, puis MESUREES: une seule
            # laissait reellement tomber un champ emis aujourd'hui
            # (`dropped_segments` sur la ligne de plan). Les cinq sont corrigees
            # quand meme -- le defaut est structurel, pas la valeur du jour.
            job["undelivered"] = dict(fields)
            job["undelivered"]["path"] = _basename(fields.get("path"))
            continue

        if body.startswith("DECLINED"):
            # LE FICHIER N'A PAS ETE PRODUIT. C'est le verdict le plus important
            # qu'un journal puisse porter et mon lecteur le laissait tomber en
            # silence -- il commence par `repair: `, donc la branche de prefixe
            # le consommait, aucune sous-branche ne le reconnaissait, et il n'y
            # avait AUCUN repli. Pas une cle fantome: une disparition.
            #
            # Sans cela un artefact decline se rendait comme un plan et une
            # piste, sans verdict et sans raison -- UN RAPPORT DECRIVANT UN
            # FICHIER QUI N'EXISTE PAS, avec l'air d'en decrire un qui existe.
            job["declined"] = body[len("DECLINED"):].strip(": ").strip() or "(no reason given)"
            continue

        if body.startswith("SKIPPED "):
            # UN SEGMENT REFUSE A DES BORNES. Le plan AVAIT un candidat ici et
            # l'a jete parce que son decalage etait invalide -- ce n'est PAS la
            # meme chose qu'un endroit ou le plan n'avait pas de candidat, et le
            # producteur le dit dans son propre commentaire. Je collectais la
            # ligne comme une chaine et la figure ne pouvait donc pas la placer.
            matched = re.match(r"SKIPPED segment master ([\d.]+)-([\d.]+)(.*)$", body)
            if matched:
                fields = split_fields(matched.group(3))
                job["refused"].append({
                    "master_start_ms": _decimal(matched.group(1)),
                    "master_end_ms": _decimal(matched.group(2)),
                    "dropped_ms": _decimal(fields.get("dropped_ms")),
                    "reason": matched.group(3).partition("DECLINED:")[2].strip()
                              or None})
            job["skipped"].append(body)
            continue

        # LES TROIS LIGNES QUE dev-1 ET dev-2 ONT FAIT ARRIVER.
        #
        # `bracket_low_ms`, `bracket_high_ms`, `bound_only`, `step_ms` etaient
        # `no-producer` dans mon registre depuis le premier jour. ILS SONT
        # EMIS. Verifie sur les OCTETS REELS et pas sur une fixture:
        #
        #   repair: bracket 2 low_ms=1000000.0 high_ms=1100000.0 width_ms=100000.0
        #           bound_only=True step_ms=-1000.93 step_points=-8
        #
        # ET LE CHIFFRE QUE J'ATTENDAIS DEPUIS LE DEBUT: sur les 16 crochets
        # emis, ma signature derivee (largeur == 100000) reproduit `bound_only`
        # SEIZE FOIS SUR SEIZE -- 11 vrais positifs, 5 vrais negatifs, aucun
        # faux dans un sens ni dans l'autre. La derivation etait juste, et elle
        # cesse d'etre necessaire: on rend LE FAIT et on garde la signature
        # comme repli, en disant lequel des deux a parle.
        #
        # Aucune liste de champs figee: `split_fields` prend ce qui est la.
        if body.startswith("bracket "):
            rest = body[len("bracket "):]
            index, _, tail = rest.partition(" ")
            entry = split_fields(tail)
            entry["index"] = index
            job["brackets"].append(entry)
            continue
        if body.startswith("segment "):
            rest = body[len("segment "):]
            index, _, tail = rest.partition(" ")
            entry = split_fields(tail)
            entry["index"] = index
            job["segments"].append(entry)
            continue
        # LA PORTE QUI S'ANNONCE AVANT DE SE FERMER, ET LE CONTENEUR QUI SE
        # NOTE LUI-MEME.
        #
        # `PREDICTED_REFUSAL` et `prediction` tombaient toutes deux dans
        # UNPARSED -- 7 lignes sur le corpus. VISIBLES, parce que rien n'est
        # jete en silence ici, mais visibles comme du texte etranger et non
        # comme des quantites. `prediction` est le producteur qui teste SA
        # PROPRE porte: il annonce combien de refus il attend, puis dit si
        # l'issue lui a donne raison. C'est la seule ligne du journal qui porte
        # un resultat de controle plutot qu'une mesure, et la perdre dans
        # UNPARSED etait perdre le seul endroit ou le moteur se note.
        if body.startswith("PREDICTED_REFUSAL "):
            job["predicted_refusals"].append(
                split_fields(body[len("PREDICTED_REFUSAL "):]))
            continue
        if body.startswith("prediction "):
            job["prediction"] = split_fields(body[len("prediction "):])
            continue
        if body.startswith("output durations "):
            job["output_durations"] = split_fields(
                body[len("output durations "):])
            continue

        if body.startswith("output file "):
            fields = split_fields(body)
            # `audio 1/1 subtitles 8/8` precede toute cle et n'appartient donc a
            # aucune valeur. Un lecteur par nom les perdrait en silence: on les
            # nomme ici plutot que de les laisser tomber.
            counts = re.search(r"audio (\S+) subtitles (\S+)", body)
            if counts:
                fields.setdefault("audio_tracks", counts.group(1))
                fields.setdefault("subtitle_tracks", counts.group(2))
            job["output_check"] = fields
            continue

        # TOUTE LIGNE `repair:` NON RECONNUE EST CONSERVEE. Le defaut ci-dessus
        # n'etait pas propre a `DECLINED`: mon analyseur avait une LISTE BLANCHE
        # de prefixes et jetait le reste sans trace, ce qui est le meme defaut
        # que la ligne TRACK enumerant ses champs -- une position deguisee en
        # nom, dans l'analyseur cette fois. Le producteur a change de format sept
        # fois en un jour; un lecteur qui perd ce qu'il ne connait pas encore
        # perd exactement les nouveautes.
        matched = re.match(r"repaired for (.*?): (.*)$", body)
        if matched:
            job["candidate_path"] = matched.group(1)
            job["candidate_opaque_id"] = opaque_id(matched.group(1))
            job["summary_counts"] = matched.group(2)
            continue

        job["unparsed"].append(body[:120])

    return job


def parse_pieces(text):
    """`c0-160000 m160000-260000 ...` -> la geometrie DU FICHIER.

    ATTENTION -- ces bornes viennent de `assembly["pieces"]`, l'unique appel a
    `normalize_segments` fait SANS `stream_order`, c'est-a-dire la geometrie DE
    REPLI: celle qu'utilise une piste QUI EMPRUNTE. Les bornes cote MAITRE sont
    en revanche les memes pour toutes les pistes -- invariant enonce par le
    producteur lui-meme (`merge_video_chimeric` :1222) -- et c'est a ce titre,
    et a ce titre seulement, qu'on s'en sert pour situer les regions par piste.
    """
    pieces = []
    for token in text.split():
        matched = re.fullmatch(r"([cm])(-?[\d.]+)-(-?[\d.]+)", token)
        if not matched:
            continue
        pieces.append({
            "source": {"c": "candidate", "m": "master"}[matched.group(1)],
            "master_start_ms": _decimal(matched.group(2)),
            "master_end_ms": _decimal(matched.group(3)),
        })
    return pieces


def opaque_id(path):
    """Un identifiant STABLE et opaque. Aucun nom de media ne quitte ce module.

    Meme construction que la cle de repertoire de travail de
    `merge_video_repair.build_repaired_video_object` (`md5(filePath)[:16]`), pour
    ne pas inventer une seconde convention -- deux conventions d'identifiant sont
    la maniere dont deux classifications de sous-titres ont diverge. PROVISOIRE:
    si le producteur emet un jour cet identifiant, on lit le sien.
    """
    if not path:
        return None
    return hashlib.md5(str(path).strip().encode("utf-8", "replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Generation de format. Ce qui distingue "pas encore emis" de "pas emis".

def format_generation(job):
    """Quel format porte cet artefact -- donc quelles absences sont des defauts.

    Sonde par STRUCTURE, jamais par date de fichier: le mtime d'un artefact dit
    quand il a ete ecrit et pas ce que le producteur savait ecrire.
    """
    if job["plan"] is None:
        return 0, "pre-plan (no `repair: plan` line at all)"
    if not job["regions_cut"] and not job["regions_added"]:
        return 1, "per-track (track lines, no per-region ADDED/CUT lines)"
    if not job["master_line_present"]:
        return 2, "region-level (ADDED/CUT present, master not named)"
    return 3, "current (master named, region-level)"


# ---------------------------------------------------------------------------
# LA RECUPERATION DES DECALAGES. Derivee, et dite comme telle.

def recover_offsets(job, stream_order):
    """Le decalage REELLEMENT UTILISE par cette piste, region par region.

    IL N'EST PAS EMIS SOUS UN NOM. Il est reconstruit depuis des champs qui le
    sont: `candidate_start_ms`/`candidate_end_ms` des lignes CUT, situes contre
    les bornes maitre de la ligne de plan. L'identite inversee est celle du
    producteur -- `candidate_start = master_start + offset`
    (`merge_video_chimeric` :341) -- et le chainage est valide parce que chaque
    fin de morceau predite doit tomber EXACTEMENT sur une borne de coupe emise.
    Une seule discordance et on rend `None` pour la region: on ne rapporte pas
    une valeur qu'on n'a pas su verifier.

    ET LA RECUPERATION EST CONDITIONNELLE: une piste dont le plan ne coupe rien
    n'emet aucune ligne CUT, et alors aucun decalage n'est recuperable. La
    recuperabilite est une propriete DU PLAN et pas de la piste, ce qui est
    precisement l'argument pour demander le champ par son nom.
    """
    plan = job["plan"]
    if not plan:
        return []
    candidate_pieces = [p for p in plan["pieces"] if p["source"] == "candidate"]
    cuts = job["regions_cut"].get(stream_order) or []
    if not candidate_pieces:
        return []
    if not cuts:
        return [Cell(None, NO_PRODUCER,
                     "no CUT line for this track, so nothing to derive from: "
                     "the offset is emitted under no name at all")
                for _ in candidate_pieces]

    head = next((c for c in cuts if c["where"] == "head"), None)
    if head is None or head["candidate_end_ms"] is None:
        return [Cell(None, NO_PRODUCER,
                     "no head CUT line, so the first piece's candidate start is "
                     "not emitted and the chain has no anchor")
                for _ in candidate_pieces]

    results = []
    source_start = head["candidate_end_ms"]
    for index, piece in enumerate(candidate_pieces):
        offset = source_start - piece["master_start_ms"]
        results.append(Cell(_trim(offset), DERIVED,
                            "candidate_start_ms - master_start_ms"))
        source_end = source_start + (piece["master_end_ms"] - piece["master_start_ms"])
        if index + 1 >= len(candidate_pieces):
            break
        following = [c for c in cuts
                     if c["where"] in ("interior", "tail")
                     and c["candidate_start_ms"] == source_end]
        if not following or following[0]["candidate_end_ms"] is None:
            # La chaine casse. On le DIT pour cette region et les suivantes.
            for _ in candidate_pieces[index + 1:]:
                results.append(Cell(None, NO_PRODUCER,
                                    "the CUT chain does not reach this piece: no "
                                    "emitted cut boundary matches the predicted "
                                    "source end"))
            break
        source_start = following[0]["candidate_end_ms"]
    return results


def track_regions(job, stream_order):
    """Une ligne par region, dans l'ordre de la timeline DU MAITRE.

    Trois natures y coexistent et la table doit les separer:
      CANDIDATE  ce que la sortie prend du candidat. C'EST LA MAJORITE DU
                 FICHIER ET AUCUNE LIGNE NE LA DECRIT -- ni source, ni langue,
                 ni fidelite. Sa geometrie n'est ici que par complement.
      MASTER     rempli depuis le maitre; ligne ADDED, avec `from=`.
      SILENCE    rempli par du silence; MEME ligne ADDED, `from=silence`.
      LOST       du candidat qui existe et n'est pas dans la sortie; ligne CUT.
                 Situee sur la timeline DU CANDIDAT, dessinee a son point
                 d'insertion cote maitre.
    """
    plan = job["plan"]
    if not plan:
        return []
    # LA VALEUR EMISE PRIME SUR LA VALEUR DERIVEE, et le desaccord se DIT.
    #
    # `USED` porte le decalage PAR NOM: c'est une lecture, pas une
    # reconstruction, et elle est inconditionnelle -- une piste dont le plan ne
    # coupe rien la porte quand meme. La derivation reste calculee A COTE, et
    # s'ils divergent on montre les DEUX plutot que le plus joli des deux.
    #
    # CE QUE CET ACCORD PROUVE, ET CE QU'IL NE PROUVE PAS -- correction de la
    # mienne. J'ai qualifie l'accord des 20 valeurs de "deux calculs
    # independants". C'EST FAUX. `used_regions` et `cut_regions` sont remplies
    # DANS LA MEME BOUCLE A PARTIR DU MEME `source_start`, donc les deux sorties
    # sont une seule grandeur rendue deux fois et NE PEUVENT PAS diverger.
    # Verifie sur mes propres octets: `step_i == dropped_ms_i - master_gap_i`
    # tient a l'identite sur 10 lignes sur 10, id297 et id169.
    #
    # L'accord etablit donc que MON LECTEUR lit correctement le format de dev-2
    # -- l'ancrage sur la coupe de tete, le chainage, l'appariement des bornes
    # -- ce qui est reel et utile. Ce n'est PAS une mesure du monde, et deux
    # rendus d'un meme calcul qui s'accordent n'apprennent rien sur ce qu'ils
    # calculent. Le controle reste en place parce qu'il attrape un changement de
    # format ou une regression de mon analyseur, pas parce qu'il confirme un
    # decalage.
    derived = recover_offsets(job, stream_order)
    emitted = {}
    for region in job.get("regions_used", {}).get(stream_order) or []:
        if region.get("offset_ms_present"):
            emitted[_trim(region["master_start_ms"])] = region["offset_ms"]
    offsets = derived
    added = {(_trim(r["master_start_ms"]), _trim(r["master_end_ms"])): r
             for r in job["regions_added"].get(stream_order) or []}

    rows, candidate_index = [], 0
    for piece in plan["pieces"]:
        key = (_trim(piece["master_start_ms"]), _trim(piece["master_end_ms"]))
        if piece["source"] == "master":
            region = added.get(key)
            matched_by = None
            if region is None:
                # LA JOINTURE ECHOUAIT SUR UN ARRONDI, ET MA CELLULE ACCUSAIT
                # LE FORMAT.
                #
                # Balayage promis a dev-2, sur SA regle: non pas "cette cause
                # est-elle la plus probable" mais "cette cause peut-elle
                # produire cette absence". Mesure sur mes 30 journaux:
                #
                #   10 artefacts portent des lignes MASTER? (piece sans ADDED)
                #    5 d'entre eux ONT des lignes ADDED
                #
                # Donc sur ces cinq le format PORTE la ligne et ne peut pas etre
                # la cause. La vraie cause, lue sur 49da3f3b:
                #
                #   plan   pieces=m0-983          (entier)
                #   ADDED  master 0-983.54        (fractionnaire)
                #
                # DEUX REPRESENTATIONS DU MEME BORD, emises par le meme
                # producteur sur deux lignes, et ma cle exacte ne les joint pas.
                #
                # On rattrape PAR ARRONDI, jamais en silence: la ligne porte les
                # deux valeurs et dit par quoi elle a joint.
                # UN RATTRAPAGE APPROXIMATIF DOIT ETRE UNIQUE OU NE PAS SE
                # FAIRE. La premiere version prenait LE PREMIER candidat et
                # s'arretait. Mesure aujourd'hui: 128 regions ADDED, ZERO
                # collision de troncature -- donc rien a corriger AUJOURD'HUI.
                # Meme raisonnement que les six listes fixes: le defaut est
                # structurel et pas la valeur du jour. Une jointure floue qui
                # choisit parmi plusieurs candidats est une jointure PAR
                # POSITION portant un nom.
                near = [(ck, cr) for ck, cr in added.items()
                        if ck[0] is not None and ck[1] is not None
                        and str(int(float(ck[0]))) == str(key[0])
                        and str(int(float(ck[1]))) == str(key[1])]
                if len(near) == 1:
                    candidate_key, region = near[0]
                    matched_by = (f"joined by TRUNCATION, not exactly: the plan "
                                  f"says {key[0]}-{key[1]} and the ADDED line "
                                  f"says {candidate_key[0]}-{candidate_key[1]} "
                                  f"-- one boundary, two emitted "
                                  f"representations")
                elif len(near) > 1:
                    matched_by = (f"NOT JOINED: {len(near)} ADDED lines truncate "
                                  f"to {key[0]}-{key[1]} and this reader will "
                                  f"not guess between them")
            if region is None:
                rows.append({
                    "kind": "MASTER?",
                    "master_start_ms": piece["master_start_ms"],
                    "master_end_ms": piece["master_end_ms"],
                    # L'OBSERVABLE D'ABORD, L'ATTENTE ENSUITE -- construction
                    # de dev-2. L'ancienne chaine nommait le FORMAT comme cause
                    # et ne pouvait pas etre vraie sur 5 des 10 artefacts qui la
                    # portent, parce qu'ils ONT des lignes ADDED.
                    "source": absent(
                        "no ADDED line matches this master piece, exactly or by "
                        "rounding. Expected on artefacts whose format carries no "
                        "ADDED lines at all; ON AN ARTEFACT THAT HAS THEM THIS "
                        "IS A FINDING, not an explanation"),
                    # LE SEUL SITE QUE MON PROPRE BALAYAGE N'A PAS VU.
                    #
                    # dev-2: UN DENOMINATEUR PEUT MENTIR EN RETRECISSANT. Mon
                    # motif exigeait un littéral entre guillemets, donc
                    # `absent()` NU n'a jamais ete examine -- 10 sites d'appel,
                    # 6 vus, et j'ai rapporte "six chaines, une instance" comme
                    # si six etait la population.
                    #
                    # Et celui-ci ne nommait pas une cause fausse: il ne disait
                    # RIEN DU TOUT. Le lecteur recevait `offset_state=
                    # absent-from-this-format` sans une syllabe sur le pourquoi,
                    # ce qui est le blanc que ce module entier existe pour ne
                    # pas produire.
                    "offset": absent(
                        "this master piece has no matching ADDED line, so this "
                        "reader cannot say whether it read the candidate at all. "
                        "A FILLED region reads no candidate and would carry "
                        "`n/a`; THIS cell is not that -- it is the offset "
                        "question left unanswered because the region's source "
                        "line is missing"),
                })
                continue
            # `from=silence` contre `from=master/<lang>`. On teste le PREFIXE:
            # la valeur porte une langue accolee pour le maitre et n'en porte
            # pas pour le silence.
            is_silence = str(region["from"] or "").startswith("silence")
            width = piece["master_end_ms"] - piece["master_start_ms"]
            rows.append({
                "fill_width": (
                    "equals the locator's UNREFINED SEARCH BOUND"
                    if width == SEARCH_BOUND_MS else
                    "equals the refine floor: the bracket WAS narrowed"
                    if width == REFINE_FLOOR_MS else None),
                "kind": "SILENCE" if is_silence else "MASTER",
                "master_start_ms": piece["master_start_ms"],
                "master_end_ms": piece["master_end_ms"],
                "source": Cell(region["from"], PRESENT),
                # Une region remplie ne LIT pas le candidat: elle n'a pas de
                # decalage. Absent parce que sans objet, et non parce que
                # manquant -- deux blancs differents.
                "offset": Cell("n/a", PRESENT, "a filled region reads no candidate"),
                # LA CAUSE, EMISE PAR L'ASSEMBLEUR, RENDUE PAR SON NOM.
                #
                # `why=` distingue head_gap / interior_bracket / tail_gap -- la
                # distinction que ma propre prose a passe la nuit a DERIVER de
                # la position dans le plan. dev-2 l'emet, donc on la rend, et on
                # rend AUSSI la derivee a cote pour que les deux se comparent.
                "matched_by": matched_by,
                "why": (region.get("why") or
                        absent("no `why=` on the ADDED line for this region")),
                # UN REPLI QUI NOMME UNE CAUSE EST UNE AFFIRMATION.
                #
                # `unreported(assembly predates the field)` dit au lecteur que
                # L'ASSEMBLAGE est ancien. dev-2 a mesure la verite: le champ
                # etait attache aux `pieces` et la ligne ADDED lit
                # `filled_regions` -- DEUX OBJETS, a trois lignes d'ecart. La
                # valeur etait calculee juste et posee sur un enregistrement que
                # rien n'imprime.
                #
                # 48 sur 48 sur ses propres octets, 19 sur 19 sur les miens:
                # cette chaine n'a JAMAIS ete vraie. Un lecteur qui enquetait
                # serait parti verifier des dates de deploiement.
                #
                # ON NE CORRIGE PAS LA VALEUR D'UN PRODUCTEUR -- elle est rendue
                # telle quelle -- ON DIT CE QU'ON SAIT D'ELLE. Corrige par dev-2
                # en `9a9d164`; les artefacts anterieurs gardent la chaine.
                "why_note": (
                    "THIS FALLBACK IS KNOWN FALSE. It says the assembly predates "
                    "the field; measured cause: the value was attached to the "
                    "plan pieces while the ADDED line reads filled_regions -- two "
                    "objects. Never true at any occurrence (48/48 on the "
                    "producer's bytes, 19/19 on mine). Fixed by its producer in "
                    "`9a9d164` (the crossing) and `85a2614` (the sentence -- "
                    "the string now names no cause, because an old assembly and "
                    "an un-annotated region are the same absence from inside the "
                    "emitter); artefacts built before those carry this string"
                    if str(region.get("why") or "").startswith("unreported(")
                    else None),
                # ET CE QUE LEUR ACCORD PROUVE, QUI EST MOINS QUE CE QUE
                # J'AI ECRIT.
                #
                # Le producteur etiquette PAR POSITION (`head_gap` si
                # `cursor == 0`, `tail_gap` sur la region finale) et ce lecteur
                # etiquette PAR POSITION. MEME REGLE. Leur accord teste donc
                # que les deux LIGNES EMISES -- `ADDED` et `pieces=` -- se
                # correspondent, ce qui est reel et attraperait une divergence
                # emetteur/plan. Il ne confirme RIEN sur la justesse du
                # classement lui-meme: pour cela il faudrait une regle qui ne
                # soit pas la sienne. J'ai ecrit a deux pairs que l'accord
                # "n'est pas une tautologie" en m'appuyant sur des controles qui
                # montrent que les deux TEXTES bougent independamment -- vrai,
                # et pas la meme affirmation. dev-2 a narre exactement cela
                # contre son propre sceau le meme jour.
                "derivation_agreement_proves": (
                    "that the emitted `ADDED` line and the emitted `pieces=` "
                    "geometry agree with each other -- NOT that either "
                    "classification is right. The producer labels by position "
                    "and so does this reader: SAME RULE, so agreement here is "
                    "largely definitional and a divergence would be a fact "
                    "about the log's internal consistency"),
                "position_in_plan_derived_by_this_reader": (
                    "head" if piece is plan["pieces"][0] else
                    "tail" if piece is plan["pieces"][-1] else "interior"),
            })
            continue

        offset = (offsets[candidate_index] if candidate_index < len(offsets)
                  else Cell(None, NO_PRODUCER, "no offset recoverable"))
        read = emitted.get(key[0])
        if read is not None:
            note = None
            if offset.state == DERIVED and str(offset.value) != _trim(read):
                # Les deux instruments divergent. On garde la valeur EMISE --
                # elle est ecrite par le producteur -- et on porte la derivee
                # a cote, nommee, pour que le desaccord soit visible et
                # adressable au lieu d'etre arbitre en silence.
                note = (f"disagrees with the value derived from the CUT bounds "
                        f"({offset.value}); the emitted value is shown")
            offset = Cell(_trim(read), PRESENT, note)
        candidate_index += 1
        rows.append({
            "kind": "CANDIDATE",
            "master_start_ms": piece["master_start_ms"],
            "master_end_ms": piece["master_end_ms"],
            # s4e: la ligne `USED` couvre desormais ces regions. Sans elle, leur
            # seule trace est le jeton `c<a>-<b>` de la ligne de plan -- ni
            # flux, ni decalage, ni fidelite.
            "source": (Cell("candidate", PRESENT) if emitted
                       else no_producer("no line type covers a kept candidate "
                                        "region; only the `c<a>-<b>` token on "
                                        "the plan line")),
            "offset": offset,
        })

    for cut in job["regions_cut"].get(stream_order) or []:
        rows.append({
            "kind": "LOST",
            "candidate_start_ms": cut["candidate_start_ms"],
            "candidate_end_ms": cut["candidate_end_ms"],
            "dropped_ms": cut["dropped_ms"],
            "dropped_unmeasured": cut["dropped_unmeasured"],
            "where": cut["where"],
        })
    return rows


# ---------------------------------------------------------------------------
# LA REDACTION, ET ELLE EST UN CONTROLE ET PAS UNE INTENTION
#
# LE JOURNAL DE PRODUCTION PORTE DE VRAIS CHEMINS, PAR CONSTRUCTION -- douze
# endroits dans `mergeVideo.py` gele. La discipline est donc a LA FRONTIERE: le
# journal ne voyage pas, CE RAPPORT VOYAGE.
#
# Trouve sur les octets reels de dev-2 et pas en relisant mon code: le journal
# passe par le pilote de rejeu commence par
# `['<chemin>'] not compatible with the others videos`, une ligne qui n'a pas le
# prefixe `repair:`. Je la collectais comme "ligne hors reparation" -- pour la
# bonne raison qu'une omission se lirait "il n'y en avait pas" -- et je l'aurais
# RECOPIEE TELLE QUELLE dans le rapport, chemin, titre et identifiant de
# catalogue compris.
#
# On ne compte donc pas sur la vigilance: toute valeur passe par ici, et le
# document fini est RELU avant d'etre rendu. Une fuite dans un artefact qui
# voyage est pire qu'un plantage, donc le controle final LEVE.

# UN CHEMIN A AU MOINS DEUX SEGMENTS. Exiger un seul `/` prenait `</title>`
# pour un chemin -- le controle final refusait d'emettre son propre balisage --
# et aurait mutile `verified=1/1` et `master/ja`, deux valeurs legitimes qui
# portent une barre. Le motif exige donc `/a/b` au minimum.
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:/[^\s'\"\]<>/]+){2,}/?")
_CATALOGUE = re.compile(r"[\[{（(]?\b(?:tvdb|tmdb|imdb|tvdbid|anidb)[-_ ]?\d+\b[\]})）]?",
                        re.IGNORECASE)
_MEDIA = re.compile(r"[^\s/\\]+\.(?:mkv|mp4|avi|m4v|ts|mka|mks|srt|ass|sub|idx)\b",
                    re.IGNORECASE)


def redact(text):
    """Rend le texte sur, et DIT combien de fois il a fallu le rendre sur.

    Un chemin absolu, un identifiant de catalogue ou un nom de fichier media
    devient un jeton opaque STABLE -- le meme chemin donne le meme jeton, donc
    deux mentions restent correlables sans que rien ne soit nomme.
    `master/ja` n'est pas touche: il ne commence pas par `/`.
    """
    if text is None:
        return None, 0
    working, hits = str(text), 0

    def token(match):
        nonlocal hits
        found = match.group(0)
        if len(found) < 3 or found.strip("/") == "":
            return found
        hits += 1
        return "<redacted:" + hashlib.md5(found.encode("utf-8", "replace"))\
                                      .hexdigest()[:8] + ">"

    working = _PATH.sub(token, working)
    working = _CATALOGUE.sub(token, working)
    working = _MEDIA.sub(token, working)
    return working, hits


class _Redactor:
    """Compte les redactions d'un rendu, pour qu'elles se DISENT."""

    def __init__(self):
        self.hits = 0
        # L'OCCASION, PAS SEULEMENT LE RESULTAT.
        #
        # dev-2, ci et moi avons chacun trouve `opportunity zero` dans un
        # instrument different ce soir -- une fixture dont les deux branches
        # rendent le meme texte, un bloc de test inatteignable, un controle de
        # contrat compatible avec un moteur qui refuse tout. La reformulation
        # qui manquait: `m = 0` EST L'ETAT PAR DEFAUT D'UN CONTROLE, et toute
        # occasion non nulle a du etre creee expres.
        #
        # Donc ce compteur porte les DEUX nombres. `hits=0` sur 400 valeurs
        # examinees dit "rien ne correspondait"; `hits=0` sur 0 valeur examinee
        # dit "ce controle n'a pas tourne", et les deux se lisaient pareil.
        self.examined = 0

    def __call__(self, text):
        if not REDACT_MEDIA_NAMES:
            return text
        self.examined += 1
        clean, hits = redact(text)
        self.hits += hits
        return clean


def assert_no_leak(document):
    """Le filet, desormais conditionne par `REDACT_MEDIA_NAMES`.

    RELACHE EN PREMIER ET DELIBEREMENT. L'ordre importe: relacher le FILET seul
    n'a aucun effet visible -- le redacteur en amont a deja remplace le nom --
    tandis que relacher le REDACTEUR seul ferait LEVER le filet et le rapport ne
    serait plus emis du tout. Dans cet ordre chaque etat intermediaire est sur.
    """
    if not REDACT_MEDIA_NAMES:
        return
    """DERNIER CONTROLE, sur le document fini. Il leve; il ne corrige pas.

    Un correctif silencieux ici rendrait la fuite suivante invisible. On
    prefere un plantage bruyant a un artefact qui voyage avec un titre dedans.
    """
    for pattern, label in ((_PATH, "an absolute path"),
                           (_CATALOGUE, "a catalogue id"),
                           (_MEDIA, "a media filename")):
        for match in pattern.finditer(document):
            found = match.group(0)
            if found.startswith("<redacted:") or len(found.strip("/")) < 3:
                continue
            if found.startswith("/") or pattern is not _PATH:
                raise AssertionError(
                    f"merge_plan report would have carried {label} "
                    f"({found[:40]!r}). Refusing to emit: this artefact travels "
                    f"and the log it is built from does not.")


# ---------------------------------------------------------------------------
# LE RENDU
#
# UN SEUL FICHIER, et le rapport EST la page. Decision du proprietaire, pas la
# mienne. Trois consequences qu'il faut tenir ensemble:
#
#   DEUX PUBLICS, ET C'EST LA VRAIE RAISON. Le proprietaire: "le schema c'est
#   bien pour l'humain et le texte c'est un recap pour etre certain de pas tout
#   perdre, et ainsi les agents peuvent rapidement comprendre." LES LIGNES SONT
#   LE CANAL DES AGENTS, le dessin est le canal humain. J'avais justifie l'ordre
#   par la robustesse -- ce qui survit a `cat` -- et cet argument-la s'affaiblit
#   le jour ou le HTML ne casse jamais. L'argument d'audience tient ce jour-la
#   aussi: un agent n'a pas besoin d'un dessin et un humain n'a pas besoin d'un
#   champ.
#
#   RIEN DANS LA SPECIFICATION NE DEPEND DE L'EXISTENCE DU HTML. Les LIGNES
#   portent chaque nombre; le dessin est rendu A PARTIR d'elles. `grep`, `cat`
#   et `diff` donnent tout sans navigateur.
#
#   CHAQUE LIGNE EST UN ENREGISTREMENT COMPLET, `KIND cle=valeur ...`. Aucun
#   lecteur n'a de position a compter. `WRITE_ZONES.MD` s7 demande un en-tete et
#   une resolution PAR NOM; des paires nommees vont plus loin -- il n'y a pas de
#   colonne 2 a confondre avec un statut. Six champs la ou deux etaient prevus
#   devient une amelioration silencieuse au lieu d'un risque.
#
#   AUCUNE REFERENCE EXTERNE. Pas de fonte distante, pas de `<img>`, pas de
#   `<script>`, aucune bibliotheque. Le conteneur n'a aucune garantie de reseau.

_STYLE = """
:root{--ink:#16181d;--paper:#fbfaf7;--rule:#c9c4b8;--faint:#6b6558;
--candidate:#2f6f9f;--master:#b8873a;--silence:#5a5a5a;--lost:#c0392b;--grid:#e4e0d6}
body{background:var(--paper);color:var(--ink);
font-family:ui-monospace,"DejaVu Sans Mono",Menlo,Consolas,monospace;
font-size:13px;line-height:1.5;margin:0;padding:24px}
h1,h2{font-size:14px;font-weight:700;margin:26px 0 8px;letter-spacing:.04em}
h1{font-size:16px;margin-top:0}
pre{white-space:pre;overflow-x:auto;background:transparent;margin:0;
border-left:2px solid var(--rule);padding:6px 0 6px 12px}
.narrative{max-width:78ch;font-family:ui-sans-serif,"DejaVu Sans",system-ui,sans-serif;
font-size:14px}
.narrative p{margin:.5em 0}
.legend span{margin-right:14px;white-space:nowrap}
.sw{display:inline-block;width:11px;height:11px;vertical-align:-1px;margin-right:4px}
.diagram{overflow-x:auto}
.note{color:var(--faint)}
"""


def _escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def clock(ms):
    """Un horodatage, parce que s4g demande des horodatages et pas des nombres."""
    if ms is None:
        return "?"
    total = Decimal(str(ms))
    negative = total < 0
    total = abs(total)
    seconds, millis = divmod(int(total), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{'-' if negative else ''}{hours:d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _row(record_kind, /, _redactor=None, **fields):
    """Un enregistrement = UNE ligne. `KIND cle=valeur`. Jamais de position.

    Une valeur absente n'est pas ecrite comme vide: elle porte son ETAT. Un
    blanc partage entre "mon lecteur ne l'a pas trouve" et "personne ne l'emet"
    est exactement le defaut que ce module existe pour ne pas commettre.
    """
    parts = [record_kind]
    for name, value in fields.items():
        if isinstance(value, Cell):
            if value.state in (PRESENT, DERIVED):
                # LA SCISSION VALAIT POUR LES VALEURS SIMPLES ET PAS POUR LES
                # `Cell`. Teste contre la grammaire que dev-2 vient de publier:
                # les sept tetes, parentheses imbriquees et virgules passent sur
                # un champ ordinaire, et `speed_margin=none(no rate proposed)`
                # -- qui traverse un `Cell` -- sortait ENTIER.
                #
                # Deux chemins pour une regle: exactement le defaut que je
                # viens de corriger en le remontant d'une couche, commis dans
                # la correction elle-meme. Septieme instance.
                split = _split_head(value.value)
                if split:
                    parts.append(f"{name}={split[0]}")
                    parts.append(f"{name}_detail="
                                 f"{_redactor(_wrap(split[1])) if _redactor else _maybe_redact(_wrap(split[1]))}")
                    if value.state == DERIVED:
                        parts.append(f"{name}_state=derived")
                    if value.note:
                        parts.append(
                            f"{name}_because="
                            f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
                    continue
                cleaned = _wrap(value.value)
                parts.append(f"{name}={_redactor(cleaned) if _redactor else _maybe_redact(cleaned)}")
                if value.state == DERIVED:
                    parts.append(f"{name}_state=derived")
                # LA HUITIEME, TROUVEE PAR L'INSTRUMENT QUE dev-2 M'A DIT DE
                # CONSTRUIRE, SUR LA SORTIE QU'IL AVAIT NOMMEE.
                #
                # `Cell(valeur, PRESENT, note)` perdait sa note: la branche
                # scindee l'emettait, celle-ci non, et la difference est LA
                # CHUTE EN FIN DE CORPS -- une sortie sans mot-cle a grepper,
                # que je n'aurais pas cherchee.
                #
                # Ce qui disparaissait: la note de DESACCORD entre le decalage
                # emis et celui que je derive des bornes CUT. Elle existe
                # exactement pour rendre visible un desaccord entre deux
                # instruments, et elle etait le seul endroit du rapport ou ce
                # desaccord se serait dit.
                if value.note:
                    parts.append(
                        f"{name}_because="
                        f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
            else:
                parts.append(f"{name}_state={value.state}")
                # LA RAISON N'ATTEIGNAIT PAS LE LECTEUR. PAS UNE SEULE.
                #
                # Trouve en verifiant que la POPULATION de mon propre balayage
                # atteint une sortie -- extension de "un denominateur peut
                # mentir en retrecissant" (dev-2) d'un cran: j'ai balaye onze
                # chaines `absent(...)`, corrige la seule qui nommait une cause
                # fausse, et reformule les autres AVEC SOIN. Aucune n'etait
                # rendue: cette branche ecrivait l'ETAT et jetait la NOTE.
                #
                # Donc pendant toute la campagne un lecteur a vu
                # `source_state=absent-from-this-format` et JAMAIS le pourquoi,
                # dans le module dont la these entiere est qu'un blanc doit dire
                # laquelle de ses raisons il porte. L'etat sans sa raison EST le
                # blanc ambigu, avec un nom dessus.
                if value.note:
                    parts.append(
                        f"{name}_because="
                        f"{_redactor(_wrap(value.note)) if _redactor else _maybe_redact(_wrap(value.note))}")
            continue
        if value is None:
            continue
        text = _wrap(value)
        # UN JETON `tete(raison)` N'EST PAS UNE VALEUR.
        #
        # Vocabulaire MESURE sur les 30 journaux -- pas devine:
        #   none(...) x33   unreported(...) x19   measured(...) x11
        #   skipped(...) x1   plus absent(...) que dev-2 vient d'ajouter
        #
        # Rendus tels quels, `dropped_segments=unreported(locator did not report
        # it)` met UNE PHRASE la ou un COMPTE est attendu, et un lecteur qui
        # compare des nombres lit du texte. C'est le piege `speed=` pour la
        # troisieme fois -- dev-2 l'a trouve chez lui deux lignes sous les trois
        # champs qu'il venait de corriger, moi chez moi sur deux freres que
        # j'avais laisses.
        #
        # ON NE RECLASSE RIEN. Je ne decide pas lesquelles de ces tetes SIGNIFIENT
        # une absence -- ce serait inferer la semantique du producteur. On SEPARE:
        # la tete reste la valeur comparable, la raison voyage a cote. Uniforme,
        # sans perte, et vrai de toute tete qu'un producteur ajoutera demain.
        head = _split_head(value)
        if head:
            parts.append(f"{name}={head[0]}")
            detail = head[1]
            parts.append(f"{name}_detail="
                         f"{_redactor(_wrap(detail)) if _redactor else _maybe_redact(_wrap(detail))}")
            continue
        parts.append(f"{name}={_redactor(text) if _redactor else _maybe_redact(text)}")
    return " ".join(parts)


# LA COUVERTURE DE `_split_head`, AVEC SON DENOMINATEUR.
#
# dev-2: "a la couche" EST UNE AFFIRMATION DE COUVERTURE ET DEMANDE UN
# DENOMINATEUR COMME TOUTE AUTRE. J'ai dit "une fois, pour tout champ present et
# futur" et la mesure etait "une fois, pour UN des DEUX chemins de `_row`".
#
# Alors compte, plutot que promesse. Les chemins par lesquels une valeur de
# producteur peut atteindre l'octet rendu:
#
#   1  `_row`, valeur simple                  -> _split_head
#   2  `_row`, Cell present/derived           -> _split_head
#   3  `_row`, Cell absent: `_state` + `_because` (aucune valeur de producteur
#      n'y passe: la valeur est None par construction)
#   4  `render_svg(records)`     \  records = parse_rows(rows). LEURS SEULES
#   5  `render_narrative(records)` /  ENTREES SONT LES LIGNES DEJA RENDUES.
#
# 4 et 5 ne peuvent donc PAS voir un jeton non scinde -- structurellement, pas
# par chance -- et la mesure le confirme: 0 jeton `tete(prose)` dans la figure et
# 0 dans la prose, sur les 28 artefacts.
#
# C'est le chemin rendu->figure que j'avais construit pour une autre raison
# ("rien n'apparait dans le dessin qui ne soit dans le texte") qui rend cette
# couverture vraie. IMPOSSIBLE ET NON PAS SEULEMENT NON OBSERVE -- la distinction
# de dev-2, appliquee a ma propre affirmation.
def _split_head(value):
    """`tete(prose)` -> `("tete", "prose")`, sinon None.

    GRAMMAIRE PUBLIEE PAR dev-2 comme contrat, apres que je l'aie
    retro-conçue depuis les octets:

        <field>=<head>(<free text with spaces>)
        head est un ensemble FERME aujourd'hui: unreported, none, measured,
        skipped, absent, unknown, unreadable
        la tete est la valeur comparable; la parenthese est de la prose et peut
        contenir espaces, virgules ET PARENTHESES IMBRIQUEES

    On ne verrouille PAS sur les sept noms: le `.*` glouton jusqu'a la derniere
    parenthese gere l'imbrication, et une tete que dev-2 ajoutera demain marche
    sans que ce module bouge. C'est le point de corriger a la couche.
    """
    if value is None:
        return None
    found = re.match(r"^([A-Za-z_]+)\((.*)\)$", str(value).strip())
    return (found.group(1), found.group(2)) if found else None


def _basename(path):
    """Le nom de fichier seul. `None` reste `None` et ne devient pas ""."""
    if not path:
        return None
    return str(path).rstrip("/").rsplit("/", 1)[-1] or None


def _maybe_redact(text):
    return redact(text)[0] if REDACT_MEDIA_NAMES else text


def _wrap(value):
    """Du texte libre reste LISIBLE et reste analysable.

    Souligner les espaces rendait `fill=master/ja[FILL_SOURCE_SHORT_BY_11.0_ms]`
    -- exact et illisible. Une valeur qui porte un espace ou un `=` est donc
    mise entre crochets: le decoupeur ignore deja les `=` a l'interieur d'un
    crochet, donc `detail=[... tolerance_ms=500 ...]` ne fabrique pas une cle
    fantome, et un humain lit la phrase.
    """
    text = str(value)
    if " " not in text and "=" not in text:
        return text
    if text.startswith("[") and text.endswith("]"):
        return text
    return "[" + text.replace("[", "(").replace("]", ")") + "]"


def plain(value):
    """Retire le crochet d'enrobage, et LUI SEUL.

    `master/ja[FILL SOURCE SHORT BY ...]` n'est pas enrobe -- il commence par
    `master/ja` -- donc la valeur du producteur traverse intacte.
    """
    if value is None:
        return None
    text = str(value)
    if len(text) > 1 and text.startswith("[") and text.endswith("]"):
        return text[1:-1]
    return text


def applied_ratio(track_fields):
    """La relation de vitesse REELLEMENT appliquee, ou None.

    Lue sur `speed=`, qui porte `speed_ratio_applied`. Ce champ EST emis quand
    une relation a ete appliquee -- il ne faut pas le confondre avec
    `speed_margin`, qui n'a aucun producteur: la MARGE par laquelle la
    transformation gagnante l'a emporte est absente, le RATIO ne l'est pas.
    `speed=None` et `speed=none(no rate proposed by the measurement)` disent
    tous deux qu'aucune relation n'a ete proposee.
    """
    value = plain(track_fields.get("speed"))
    if value is None:
        return None
    if value.lower().startswith("none"):
        return None
    try:
        return Decimal(value)
    except Exception:
        return "UNREADABLE"


def lost_reference_frame(ratio):
    """Dans QUELLE timeline `dropped_ms` est exprime, et pourquoi cela tient.

    Les bornes des lignes CUT sont lues sur le candidat DEJA REECHANTILLONNE --
    `assemble_on_master_timeline` multiplie `candidate_duration_ms` par le ratio
    AVANT `normalize_segments`, et la chaine de vitesse est posee avant le
    `atrim` dans le filtergraph, ce que le producteur enonce lui-meme
    (`merge_video_chimeric` :370, "les temps des tranches sont lus sur le
    candidat DEJA reechantillonne").

    Consequence, et elle va CONTRE l'intuition: comme
    `candidate_start = master_start + offset`, cette timeline avance AU MEME
    RYTHME que celle du maitre. Un `dropped_ms` est donc deja une duree
    comparable a du temps maitre, et la barre rouge se dessine a l'echelle 1:1
    SANS correction. Corriger par le ratio introduirait l'erreur qu'on croit
    corriger.

    Ce qui, LUI, differe d'un facteur `ratio`: la quantite de materiau
    D'ORIGINE du candidat qui a ete jetee, soit `dropped_ms / ratio`. Les deux
    sont legitimes et ne sont pas la meme chose, donc on dit laquelle on montre.
    """
    if ratio in (None, "UNREADABLE"):
        return "resampled-candidate (advances at master rate; 1:1 with the axis)"
    return (f"resampled-candidate (advances at master rate; 1:1 with the axis) "
            f"ratio={ratio}")


def seconds_fr(ms, decimals, signed=True):
    """Une duree en SECONDES, point decimal.

    LA VIRGULE EST TOMBEE AVEC LA LANGUE. Le proprietaire a tranche que les
    DOCUMENTS PRODUITS SONT EN ANGLAIS; traduire les mots en gardant `0,125`
    aurait donne un document anglais annoncant une convention francaise et
    l'appliquant -- deux faussetes qui se couvrent l'une l'autre. La ligne
    `CONVENTION` dit desormais `point`.

    Sa locale est francaise et il lit ces figures: `-22,815 s`, jamais
    `-22.815 s`. Le point decimal n'est pas un detail de style ici, c'est un
    nombre qui se lit de travers.

    CONVENTION DE SIGNE, et elle est declaree plutot que subie: la valeur
    AFFICHEE est l'oppose du decalage emis. Un `offset_ms` positif veut dire que
    le candidat est lu PLUS TARD que le maitre, donc qu'il faut l'avancer -- un
    retard NEGATIF au sens ou la figure les ecrit. Le champ affiche est emis
    dans les LIGNES a cote du champ brut, pour que le nombre du dessin existe
    dans le texte: sinon la figure porterait une valeur qu'aucune ligne ne dit.
    """
    if ms is None:
        return None
    value = Decimal(str(ms)) / Decimal("1000")
    if signed:
        value = -value
    quantised = value.quantize(Decimal("1." + "0" * decimals))
    return f"{quantised}"


def borrow_provenance(job, stream_order):
    """EMPRUNTE A QUOI. `BORROWED` seul ne le dit pas.

    La langue de MESURE est sur la ligne de plan, `language=`, et LA SEULE. Une
    correction de dev-2 que je porte ici plutot que dans un message: `offset=`
    distingue mesure d'emprunte et ne dit PAS quelle piste porte la langue du
    plan -- sur un artefact reel deux pistes sont `measured` et une seule est
    dans la langue du plan.

    L'attribution est une INFERENCE et se marque comme telle: le journal ne dit
    nulle part "la piste 2 a emprunte a la piste 5". On la VERIFIE au lieu de
    l'affirmer -- les decalages par region de l'emprunteuse doivent egaler ceux
    de la piste de reference, region par region -- et on rapporte le resultat de
    cette verification, y compris quand elle echoue.
    """
    plan = job.get("plan") or {}
    language = plan.get("language")
    if not language:
        return None
    reference = [order for order, fields in (job.get("audios") or {}).items()
                 if fields.get("lang") == language]
    if len(reference) != 1:
        # Zero ou plusieurs pistes dans la langue du plan: l'attribution n'est
        # pas decidable et on ne devine pas.
        #
        # ET UN ZERO DOIT DIRE DE QUEL COTE IL VIENT. `0 tracks carry the plan
        # language` se lit comme un fait sur LE FICHIER; ce peut etre un fait
        # sur LA JOINTURE. L'aide a mesure la forme generale ailleurs dans la
        # chaine: un cote en ISO-639-1 {de,en,es,fr,...}, l'autre en ISO-639-2
        # {ger,eng,spa,fre,...}, intersection VIDE, et une egalite de chaines
        # rend zero ligne sur chaque fichier sans jamais lever d'erreur.
        #
        # Ici les deux cotes viennent du MEME emetteur et du meme jeu de codes
        # -- mesure sur 40 jointures possibles, 40 avec au moins une ligne,
        # zero vide -- donc le defaut ne m'atteint pas AUJOURD'HUI. Il
        # m'atteindrait en silence le jour ou un cote changerait de jeu, et
        # c'est exactement ce que cette phrase rend impossible.
        codes = sorted({(f or {}).get("lang") for f in (job.get("audios") or {}).values()
                        if (f or {}).get("lang")})
        return {"language": language, "track": None,
                "agreement": f"undecidable: {len(reference)} tracks carry the "
                             f"plan language"
                             + ("" if reference else
                                f". THE PLAN'S CODE `{language}` IS NOT AMONG "
                                f"THE TRACK CODES {codes} -- if those look like "
                                f"the same languages in a different code set "
                                f"(ISO-639-1 against ISO-639-2, `fr` against "
                                f"`fre`), this zero is a fact about THE JOIN and "
                                f"not about the file, and a string-equality "
                                f"match would return zero rows on every file "
                                f"without ever erroring")}
    other = reference[0]
    if other == stream_order:
        return None
    mine = [r["offset_ms"] for r in (job.get("regions_used", {}).get(stream_order) or [])]
    theirs = [r["offset_ms"] for r in (job.get("regions_used", {}).get(other) or [])]
    if not mine or not theirs:
        mine = [c.value for c in recover_offsets(job, stream_order) if c.state == DERIVED]
        theirs = [c.value for c in recover_offsets(job, other) if c.state == DERIVED]
    if not mine or not theirs or len(mine) != len(theirs):
        agreement = "unverified: the two tracks do not expose comparable offsets"
    elif all(str(a) == str(b) for a, b in zip(mine, theirs)):
        agreement = f"offsets identical at {len(mine)} of {len(mine)} regions"
    else:
        agreement = (f"DISAGREES: {sum(1 for a, b in zip(mine, theirs) if str(a) != str(b))} "
                     f"of {len(mine)} regions differ from the reference track")
    return {"language": language, "track": other, "agreement": agreement}


def _speed_margin_cell(plan):
    """`NON EMISE` seulement quand le producteur n'a RIEN dit.

    Une marge indefinie n'est pas une marge absente: quand une seule hypothese
    franchit la porte de fidelite, aucune marge de PLATITUDE n'existe et la
    decision a pourtant ete tranchee -- par la porte, avec sa propre separation
    dans `fidelity_margin`. Confondre les deux imprimerait `NON EMISE` par-dessus
    une decision nette.
    """
    value = plain(plan.get("speed_margin"))
    if not value:
        # `no-producer` ETAIT UNE AFFIRMATION QUE JE NE PEUX PAS SOUTENIR.
        #
        # dev-2 a mesure trois etats dans son emetteur pour DEUX jetons: marge
        # presente, absente AVEC raison, et absente SANS raison -- ce dernier
        # n'ajoutant rien du tout a la ligne. Un blanc y couvre donc "le plan ne
        # la portait pas" ET "cette ligne est anterieure au champ", et je
        # rendais ce blanc comme `no-producer`: PERSONNE NE L'EMET. C'est faux
        # dans le second cas, ou quelqu'un l'emet et pas encore a cette date.
        #
        # dev-2 propose que `repair: build <module:sha12>` serve de
        # discriminant, etant inconditionnel. MESURE SUR MES OCTETS:
        #
        #   28 journaux de travail
        #   12 portent une ligne `repair: build`
        #   16 N'EN PORTENT PAS
        #
        # Le marqueur de provenance est absent precisement des artefacts qu'il
        # devrait dater. Et la ou il est present c'est un DIGEST DE CONTENU --
        # deux valeurs distinctes, aucun ordre -- donc un lecteur ne peut pas
        # dire lequel precede un champ sans une correspondance qui ne vit dans
        # aucun artefact.
        return Cell(None, NO_PRODUCER,
                    "the plan line carries no `speed_margin` key. THIS READER "
                    "CANNOT DISTINGUISH two cases: the plan genuinely had no "
                    "margin, or this line predates the field. The producer "
                    "appends nothing in either case. `repair: build` would date "
                    "it, but it is present on only 12 of the 28 job logs here "
                    "and is a content digest with no ordering")
    if value.lower().startswith("absent("):
        return _plan_absent_cell(value, "speed_margin")
    return Cell(value, PRESENT)


def _plan_absent_cell(value, name):
    """`absent(<raison>)` sur N'IMPORTE QUEL champ de plan, interprete PAR RAISON.

    dev-2 vient d'ajouter `absent(not_in_plan)` sur trois champs. Teste sur des
    octets CONSTRUITS -- il ne pousse pas, donc le jeton n'est encore dans aucun
    journal -- et mon lecteur le traitait mal DEUX FOIS:

      speed_margin     -> `not-defined-here`, avec "-- see decided_by" accole.
                          Or `not_in_plan` ne veut PAS dire "indefinie parce
                          qu'une seule hypothese a franchi la porte": il veut
                          dire QUE LE PLAN NE LA PORTAIT PAS. Deux etats
                          differents sous un seul, et le renvoi vers
                          `decided_by` pointe vers un champ qui porte le meme
                          `absent(not_in_plan)`.
      fidelity_margin  -> rendu TEL QUEL, `fidelity_margin=absent(not_in_plan)`,
      decided_by          comme si la chaine etait une valeur. C'est le piege
                          deja paye sur `speed=`: un SENTINELLE rendu en VALEUR.
    """
    reason = value[len("absent("):].rstrip(")")
    if reason.strip().lower() == "not_in_plan":
        return Cell(None, ABSENT_FORMAT,
                    f"the producer states this plan did not carry "
                    f"`{name}`. THIS IS THE PRODUCER SPEAKING, not this reader "
                    f"inferring: a line with no token at all leaves the same "
                    f"blank ambiguous")
    return Cell(None, NOT_DEFINED, reason + " -- see decided_by")


def _plan_field_cell(plan, name):
    """Un champ de plan qui peut arriver en valeur OU en `absent(<raison>)`."""
    value = plain(plan.get(name))
    if not value:
        return None
    if value.lower().startswith("absent("):
        return _plan_absent_cell(value, name)
    return Cell(value, PRESENT)


def plan_end_ms(job):
    pieces = (job.get("plan") or {}).get("pieces") or []
    return pieces[-1]["master_end_ms"] if pieces else None


def build_rows(job, artefact_id, source_name, n_caveat, corpus=None):
    """Les LIGNES. Tout nombre du rapport est ici, une ligne par enregistrement."""
    generation, description = format_generation(job)
    plan = job.get("plan") or {}
    redactor = _Redactor()
    rows = []

    rows.append(_row("MERGE_PLAN", schema="1", produced_by="merge_plan_report",
                     reads="repair-lines-of-one-job-log"))
    rows.append("# Every record below is one line: KIND key=value ... . Resolve BY NAME;")
    rows.append("# there are no columns and no positions. A value that is not present")
    rows.append("# carries <key>_state instead of <key>, with one of:")
    rows.append("#   derived                 computed from other emitted fields")
    rows.append("#   collapsed               emitted, but aggregated above this granularity")
    rows.append("#   absent-from-this-format this artefact predates the field")
    rows.append("#   no-producer             nothing emits it; this is the defect")
    rows.append("#   not-exercised-here      the branch is live and this artefact does not")
    rows.append("#                           produce its input -- NOT a negative answer.")
    rows.append("#                           ALONE AMONG THESE STATES IT IS A PROPERTY OF")
    rows.append("#                           THE DATA, NOT OF THE CODE: a new corpus file")
    rows.append("#                           turns it into `present` with nobody touching")
    rows.append("#                           producer or consumer. It is therefore scoped")
    rows.append("#                           to ONE artefact and it CANNOT BE DATED from")
    rows.append("#                           this record -- no build, commit or timestamp")
    rows.append("#                           field is emitted anywhere (see the")
    rows.append("#                           build_identity GAP row). Read it as of the")
    rows.append("#                           artefact named on the SOURCE row, never as a")
    rows.append("#                           standing fact about the pipeline.")
    rows.append("#   not-defined-here        the quantity does not exist in this case;")
    rows.append("#                           the decision was made elsewhere, see decided_by")
    # UNE AFFIRMATION DE POPULATION SE DIT, MEME QUAND L'UNITE EST DOUTEUSE.
    # Sans elle un lecteur qui ouvre dix-huit figures voit dix-huit
    # observations. dev-2 a ecrit la sienne, elle etait fausse, et elle a ete
    # corrigee dans l'heure PARCE QU'ELLE ETAIT ECRITE. La mienne etait absente,
    # donc invisible a tout lecteur qui n'etait pas dans la conversation.
    if corpus:
        rows.append(_row("CORPUS", _redactor=redactor, **dict(corpus)))
    else:
        # AUCUNE POPULATION FOURNIE, ET ON LE DIT. Le rapport contient des
        # affirmations a l'echelle du corpus; sans base elles sont sans
        # denominateur, et un blanc ici se lirait comme "la population va de
        # soi".
        rows.append(_row("CORPUS", _redactor=redactor, state=NOT_SUPPLIED,
                         # LE REMEDE DOIT NOMMER SON PUBLIC. Cette note disait
                         # `Call measure_corpus() and pass the result` -- UNE
                         # INSTRUCTION, imprimee dans CHAQUE rapport de
                         # production, disant au lecteur de faire exactement ce
                         # que j'ai explique a l'architecte comme etant pire que
                         # rien: fabriquer une population de 1 et la presenter
                         # comme un denominateur.
                         #
                         # C'est ma propre classe d'un cran de plus: l'ETAT est
                         # juste, et LE REMEDE QUI L'ACCOMPAGNE n'est juste que
                         # pour l'un des deux publics qui le lisent. Et c'est la
                         # note qui voyage, pas le commentaire au point d'appel:
                         # un mainteneur suit la ligne imprimee dans l'artefact
                         # qu'il tient.
                         note="no population was supplied to this render, so "
                              "every corpus-scale claim below is undenominated. "
                              "IN A MULTI-ARTEFACT RENDER, call measure_corpus() "
                              "and pass the result. IN PRODUCTION THIS IS THE "
                              "CORRECT OUTPUT AND NOT A DEFECT: a single merge "
                              "has no population, and inventing one of 1 would "
                              "present a denominator that does not exist"))
    rows.append(_row("SOURCE", artefact=artefact_id, log=source_name,
                     format_generation=generation, format=description))
    if job.get("sources"):
        rows.append(_row("SOURCES", _redactor=redactor,
                         digest=job["sources"]["digest"],
                         files=job["sources"]["files"],
                         scope=job["sources"]["scope"],
                         manifest=job["sources"]["manifest"],
                         derivation="sha256 of the bytes on disk at call time, "
                                    "read per file",
                         covers="the .py files the image ships, AND NOTHING "
                                "ELSE: not the interpreter, not ffmpeg, not "
                                "mkvtoolnix -- all installed unpinned. Two "
                                "artefacts sharing this digest ran identical "
                                "Python; they did not necessarily run in "
                                "identical containers"))
    if job.get("build"):
        rows.append(_row("BUILD", _redactor=redactor,
                         **dict(job["build"]),
                         note="which version of each module produced this. A "
                              "verdict is a claim about a file AND about the "
                              "build that made it"))
    rows.append(_row("IDENTITY",
                     master=job.get("master_opaque_id") or "",
                     candidate=job.get("candidate_opaque_id") or "",
                     # LE BASENAME, PAS LE CHEMIN, et le champ dit ce qu'il
                     # porte. Je rendais `/srv/.../Season 17/X.mkv` dans un champ
                     # nomme `_name`: la valeur etait juste et L'ETIQUETTE
                     # PROMETTAIT AUTRE CHOSE -- meme classe que le jeton d'etat
                     # dans la legende, corrige une heure plus tot.
                     #
                     # Et le perimetre suit l'autorisation plutot que de la
                     # deborder: le proprietaire a autorise les NOMS pour "savoir
                     # qui est quoi". Un chemin absolu donne cela ET la structure
                     # de la bibliotheque, qu'il n'a pas demandee. Le chemin
                     # complet reste dans le journal source, qui ne voyage pas.
                     # LE NOM DU CHAMP PORTE L'AVERTISSEMENT, AU POINT D'USAGE.
                     # La phrase en tete du document est lue par un HUMAIN qui
                     # ouvre le fichier; UN AGENT QUI `grep` NE VOIT JAMAIS LA
                     # TETE. Signale par dev-3, qui est precisement le
                     # consommateur le plus susceptible de grep plutot que de
                     # lire, et qui a prefere le dire plutot que d'etre
                     # l'instance.
                     #
                     # C'est l'inverse exact de mon defaut `_name`: la l'etiquette
                     # promettait autre chose que la valeur, ici l'etiquette est
                     # juste et L'AVERTISSEMENT EST AILLEURS. Le suffixe le
                     # ramene sur la ligne, et `grep master_name` continue de
                     # correspondre par prefixe -- aucun lecteur existant ne
                     # casse.
                     master_name_local_only=(_basename(job.get("master_path"))
                                  if not REDACT_MEDIA_NAMES else None),
                     candidate_name_local_only=(_basename(job.get("candidate_path"))
                                     if not REDACT_MEDIA_NAMES else None),
                     construction="md5(path)[:16]",
                     quote_by=("the opaque ids above, never the `_local_only` "
                               "fields: those must not leave the output "
                               "directory"
                               if not REDACT_MEDIA_NAMES else None),
                     name_fidelity=("square brackets in a name are rendered as "
                                    "parentheses by this row grammar, so a NAME "
                                    "HERE IS NOT BYTE-EXACT -- use the id if you "
                                    "need to match, and the figure's title or "
                                    "the source log if you need the literal name"
                                    if not REDACT_MEDIA_NAMES else None),
                     note=("names carried BESIDE the ids, never instead: an id "
                           "removed is an id nobody can use again"
                           if not REDACT_MEDIA_NAMES else
                           "opaque ids only; no media name travels in this report")))
    if not job.get("master_line_present"):
        rows.append(_row("IDENTITY_LIMIT", master_state=NO_PRODUCER,
                         detail="this_format_does_not_name_the_master;"
                                "_content_correspondence_is_unmeasurable"))
    for caveat in n_caveat:
        rows.append(_row("CAVEAT", text=caveat))

    rows.append(_row("CONVENTION", _redactor=redactor,
                     displayed_offset="the negative of offset_ms, in seconds",
                     reason="a candidate read later than the master must be "
                            "advanced, which the figure writes as a negative delay",
                     decimal_separator="point"))
    rows.append(_row("PLAN", kind=plan.get("kind"), language=plan.get("language"),
                     quantum_ms=plan.get("quantum_ms"),
                     # TROIS CAS ET NON DEUX, et le troisieme est un
                     # SENTINELLE-VALEUR et pas un champ omis:
                     #   speed_margin=12215.4              la marge existe
                     #   speed_margin=absent(<raison>)     elle n'est PAS definie,
                     #                                     et la raison voyage
                     #   champ absent                      le producteur n'a rien dit
                     # Seul le troisieme est `NON EMISE`. Le deuxieme rendu comme
                     # le troisieme qualifierait de sans-marge une decision prise
                     # avec 0,3637 de separation.
                     #
                     # ECRIT CONTRE UN FORMAT ANNONCE ET JAMAIS CONTRE SES
                     # OCTETS: dev-2 m'a decrit ces champs avant de les emettre.
                     # La forme est donc UNE FORME, pas une lecture, et la
                     # jonction ne sera faite que quand j'aurai passe ce lecteur
                     # sur son artefact -- s7 regle 2, et c'est exactement
                     # l'echange que j'ai refuse une fois deja aujourd'hui.
                     speed_margin=_speed_margin_cell(plan),
                     fidelity_margin=_plan_field_cell(plan, "fidelity_margin"),
                     decided_by=_plan_field_cell(plan, "decided_by"),
                     master_end_ms=_trim(plan_end_ms(job)),
                     master_end=clock(plan_end_ms(job)),
                     pieces=len(plan.get("pieces") or []),
                     # ET LE RESTE PASSE. Sixieme et derniere liste fixe du
                     # balayage: `dropped_segments` etait emis sur 2 journaux et
                     # n'atteignait aucune ligne. Le localisateur DIT combien de
                     # segments il a juges inutilisables et ce rapport le taisait.
                     # L'ABSENCE A ACQUIS UN SENS ET NE PEUT PAS ENCORE LE
                     # PORTER. dev-2 emet desormais `dropped_segments=N` quand
                     # il y en a et `unreported(...)` quand le localisateur ne
                     # dit rien -- donc UNE LIGNE SANS LE CHAMP veut dire "zero
                     # segment jete, ET c'etait rapporte".
                     #
                     # Mais c'est vrai des artefacts produits APRES ce
                     # changement seulement, et rien dans l'artefact ne dit de
                     # quel cote il tombe -- `repair: build` manque sur 16 de mes
                     # 28 journaux et ne porte de toute facon aucun ordre. Meme
                     # ambiguite que `speed_margin`, meme raison.
                     dropped_segments_absent_means=(
                         None if "dropped_segments" in plan else
                         "no `dropped_segments` on this plan line. On an "
                         "artefact from a build that emits it, that means ZERO "
                         "segments dropped AND the locator reported so. On an "
                         "older one it means the field did not exist. THIS "
                         "READER CANNOT TELL WHICH: `repair: build` is missing "
                         "from 16 of the 28 job logs here and carries no "
                         "ordering where it is present"),
                     **{name: value for name, value in plan.items()
                        if name not in _PLAN_FIELDS_RENDERED}))
    rows.append(_row("PLAN_GEOMETRY_LIMIT",
                     detail="the_pieces=_token_is_assembly[pieces],_the_one_"
                            "normalize_segments_call_made_with_no_stream_order,"
                            "_i.e._the_geometry_a_BORROWING_track_uses"))

    # L'AFFIRMATION QUE CETTE FIGURE FAIT SANS LA DIRE, AVEC SON OCCASION.
    #
    # Un seul escalier est dessine par GEOMETRIE, pas par piste: la figure
    # AFFIRME que les pistes audio d'un meme fichier partagent une geometrie.
    # Rien de ce que je tiens ne peut la refuter, et une affirmation
    # invisible ne peut pas etre contredite par un lecteur.
    #
    # Elle est donc imprimee AVEC SON COMPTE D'OCCASIONS, y compris la moitie
    # qui vaut ZERO. Une ligne m = 0 laissee de cote pour cause de non-
    # informativite est exactement la faute qui a laisse passer trois tetes
    # "non testees" chez dev-2 hier soir: le lecteur ne voit alors aucune
    # difference entre "verifie" et "jamais mis a l'epreuve".
    if len(job.get("audios") or {}) > 1:
        rows.append(_row(
            "SHARED_GEOMETRY_ASSERTION",
            asserts="this figure draws one staircase per GEOMETRY, not per "
                    "track. IT IS A CORRECT DRAWING OF WHAT THE PIPELINE DID "
                    "AND MUST NOT BE READ AS `THE TRACKS WERE FOUND TO AGREE`",
            answered_from_the_source="NOT unfalsifiable after all, and the "
                 "answer is NO. merge_video_chimeric.py:1611 -- the per-stream "
                 "pairing table covers ONLY the comparison language, so every "
                 "other language BORROWS, silently, with 14 to 32 ms of MEASURED "
                 "error, below the verifier's 100 ms tolerance. The tracks do "
                 "not share a measured geometry: THEY SHARE ONE MEASUREMENT. "
                 "Borrowing is structural because nothing else was ever "
                 "measured, not because two geometries were compared and found "
                 "equal. Found by forensic in the source, verified here at "
                 "merge_video_chimeric.py:1611-1613 and 2211",
            why_no_artefact_can_settle_it="there is no job in which two tracks "
                 "are INDEPENDENTLY measured, so the corpus case this reader "
                 "spent the campaign asking for cannot exist under this code. "
                 "That is the finding, not a gap in the corpus -- and it is why "
                 "`0 logs carry more than one repair: plan line` is not a "
                 "logging defect either: ONE PLAN BECAUSE ONE MEASUREMENT",
            head="what was confirmed at the head is narrower than this row "
                 "claimed an hour ago: two tracks declared filled FROM THE SAME "
                 "master stream cross-correlate at r = 0.996..0.999 over seven "
                 "windows and the identical region stops at the declared "
                 "boundary (forensic, F-58). THAT CONFIRMS THE DECLARED FILL IS "
                 "PRESENT IN THE PRODUCED SAMPLES -- the first check of this "
                 "plan against anything but log text, and worth having. It is "
                 "NOT evidence that two geometries agree: identical bytes "
                 "copied to two tracks correlate at 1 whatever the geometry",
            interior=Cell(None, NOT_MEASURED,
                          "MEASURED AND NON-DISCRIMINATING, and now also "
                          "ANSWERED FROM THE SOURCE ABOVE -- the divergence is "
                          "not unknown, it is 14 to 32 ms and it is hidden "
                          "under the verifier's tolerance BY DESIGN. "
                          "The same instrument on the "
                          "same file returns r = 0.23..0.83 across the interior "
                          "bracket -- inside the band that file produces between "
                          "tracks that are NOT one source, because dubs share a "
                          "music-and-effects bed. OF WHICH COULD HAVE FAILED: 0. "
                          "Measured and non-discriminating is a third outcome, "
                          "not a negative, and this row is printed at m = 0 "
                          "rather than dropped: a dropped m = 0 is how a reader "
                          "stops being able to tell `verified` from `never put "
                          "to the test`"),
            refuted_by_nothing_i_hold="0 logs carry more than one `repair: plan` "
                                      "line, so this reader cannot compare two "
                                      "per-track geometries even in principle"))
    for order in sorted(job.get("audios") or {}):
        fields = job["audios"][order]
        rows.append(_row("TRACK", track=order, kind="audio",
                         lang=fields.get("lang"),
                         fill=fields.get("fill"),
                         filled_ms=fields.get("filled_ms"),
                         silence_ms=fields.get("silence_ms"),
                         head_pad_ms=fields.get("head_pad_ms"),
                         speed=fields.get("speed"),
                         offset=fields.get("offset"),
                         verify=fields.get("verify"),
                         probes=fields.get("probes"),
                         worst=fields.get("worst"),
                         r_min=fields.get("r_min"),
                         verified=fields.get("verified"),
                         # TOUT CHAMP QUE JE NE CONNAIS PAS PASSE QUAND MEME.
                         #
                         # J'enumerais une liste fixe, donc un champ ajoute par
                         # le producteur DISPARAISSAIT SANS TRACE. Mesure sur les
                         # octets de dev-2 et pas en relisant: `rms_over_floor`
                         # etait sur la ligne de piste, mon analyseur le lisait
                         # PAR NOM correctement, et ma ligne le jetait.
                         #
                         # C'est le pire endroit pour cette perte: resoudre par
                         # nom existe pour qu'une amelioration du producteur
                         # traverse en silence, et j'avais recree un lecteur
                         # POSITIONNEL a l'emission -- une liste blanche est une
                         # position deguisee en nom. Le format de dev-2 a change
                         # sept fois en un jour; un lecteur qui doit etre modifie
                         # a chaque ajout n'est pas un lecteur par nom.
                         **{name: value for name, value in fields.items()
                            if name not in _TRACK_FIELDS_RENDERED}))
        label = plain(fields.get("offset")) or ""
        if label.startswith("BORROWED"):
            borrow = borrow_provenance(job, order)
            if borrow:
                # ET LA RAISON IMPRIMEE PAR LE PRODUCTEUR N'EST PAS TESTEE
                # PAR LE PRODUCTEUR.
                #
                # `BORROWED[le maitre ne porte aucun flux fr]` a ete verifie sur
                # le fichier: LE MAITRE PORTE UN FLUX fr, index 2, ac3 2ch --
                # exactement l'index et le codec que le champ `fill=` nomme.
                # La branche qui ecrit cette phrase est un FOURRE-TOUT, atteint
                # quand la table d'appariement n'a pas de ligne pour ce flux, et
                # RIEN sur ce chemin n'inspecte les flux du maitre.
                #
                # La vraie raison vit dans la CONDITION, la raison imprimee vit
                # dans le MESSAGE, et rien ne les tient en phase. UN FOURRE-TOUT
                # QUI NOMME UNE CAUSE PRECISE EST PIRE QU'UN QUI DIT `INCONNU`:
                # il est falsifiable et faux, et une raison ferme la question
                # qu'un blanc aurait ouverte. Trouve par forensic (F-60).
                rows.append(_row("BORROW", _redactor=redactor, track=order,
                                 plan_language=borrow["language"],
                                 from_track=borrow["track"],
                                 attribution="inferred, not stated by the log",
                                 printed_reason_is_not_checked_by_its_writer=(
                                     "the `BORROWED[...]` text on the TRACK row "
                                     "is the producer's and is rendered "
                                     "verbatim, never laundered -- but its "
                                     "final branch is a CATCH-ALL reached when "
                                     "the pairing table has no row for this "
                                     "stream, and nothing on that path inspects "
                                     "what it names. One such reason has been "
                                     "measured FALSE against the file: it said "
                                     "the master carried no stream in that "
                                     "language and the master carried one, at "
                                     "the index and codec the `fill=` field "
                                     "names. READ IT AS A LABEL, NOT AS A "
                                     "FINDING"),
                                 check=borrow["agreement"]))

        # UN PLAN SANS `language=`, ET SEPT PISTES DESSINEES DEPUIS LUI.
        #
        # `borrow_provenance` rend None quand le plan ne nomme pas sa langue de
        # mesure, et la ligne BORROW n'est de toute facon tiree que si `offset=`
        # commence par BORROWED. Sur les trois artefacts a SEPT pistes dont le
        # plan ne porte AUCUN `language=`, ni l'une ni l'autre condition n'est
        # remplie: le rapport dessinait la MEME geometrie pour les sept pistes
        # sans une ligne pour le dire. UN BLANC QUI SE LIT `RIEN D'EMPRUNTE ICI`.
        #
        # C'est mon propre pire defaut, dans ma propre machinerie d'emprunt:
        # une entree absente produit du silence au lieu d'un enonce.
        #
        # ET LES PISTES NE SONT PAS IDENTIQUES, ce journal le montre lui-meme:
        # `head_pad_ms` vaut 48.0 sur la piste 1 et 1103.0 sur les six autres.
        # Elles ne peuvent donc pas partager la geometrie de tete que ce rapport
        # leur dessine a toutes.
        audios = job.get("audios") or {}
        pads = {order: (f or {}).get("head_pad_ms") for order, f in audios.items()}
        distinct = {str(v) for v in pads.values() if v is not None}
        plan_language = (job.get("plan") or {}).get("language")
        # DEUX RAISONS DISTINCTES, ET MA PREMIERE VERSION N'EN VOYAIT QU'UNE.
        #
        # Je ne tirais cette ligne que si le plan ne nommait pas sa langue.
        # Enquete elargie a TOUS les journaux multi-pistes que je detiens:
        #
        #   12 journaux multi-pistes
        #    4 ont des head_pad_ms DIFFERENTS entre pistes
        #    3 de ces quatre ont un plan sans `language=`  -> ma ligne tirait
        #    1, cb354a77e0088e7e, a un plan `language=ja`  -> ELLE NE TIRAIT PAS
        #
        # Or c'est l'ECART DES head_pad qui prouve que les pistes ne partagent
        # pas la geometrie de tete, pas l'absence de langue. L'absence de langue
        # rend l'ATTRIBUTION indecidable; l'ecart rend la GEOMETRIE fausse. Deux
        # defauts, et je conditionnais le second sur le premier.
        if len(audios) > 1 and (not plan_language or len(distinct) > 1):
            # `borrow` n'existe que dans la branche BORROWED; ici on redemande.
            attribution = borrow_provenance(job, order)
            own = [o for o, f in audios.items()
                   if plan_language and (f or {}).get("lang") == plan_language]
            if attribution is None and own == [order]:
                attribution = {"track": "THIS track: it carries the plan's own "
                                        "measurement language"}
            rows.append(_row("PLAN_ATTRIBUTION_LIMIT", track=order,
                             plan_lines_read=absent(
                                 "this reader keeps ONE plan per job and would "
                                 "silently overwrite a second; it cannot report "
                                 "how many the log carried"),
                             audio_tracks=len(audios),
                             plan_language=(plan_language or
                                            absent("this plan line carries no "
                                                   "`language=` key")),
                             reference_track=(
                                 (attribution or {}).get("track")
                                 if plan_language and attribution else
                                 absent("undecidable: no track can be "
                                        "identified as the one the plan was "
                                        "measured on")),
                             head_pad_ms_distinct_values=len(distinct),
                             head_pad_ms_seen=" ".join(sorted(distinct)) or None,
                             reads_as=("THIS ROW'S GEOMETRY IS DRAWN FROM THE "
                                       "ONE PLAN IN THIS LOG AND WAS NOT "
                                       "MEASURED FOR THIS TRACK"
                                       if len(distinct) > 1 else
                                       "drawn from the one plan in this log; "
                                       "no per-track geometry is emitted"),
                             evidence=("the tracks are NOT identical at the "
                                       "head: head_pad_ms takes "
                                       f"{len(distinct)} different values "
                                       "across them, so they cannot share the "
                                       "head geometry drawn for all of them"
                                       if len(distinct) > 1 else None),
                             state=NOT_MEASURED))

        # UN MANQUE MESURE N'EST PAS UN MANQUE ATTRIBUE.
        #
        # La ligne de piste porte `[FILL SOURCE SHORT BY x ms; TRACK LOST y ms;
        # UNEXPLAINED z ms]`, et je la rendais telle quelle, y compris dans la
        # phrase francaise. Un lecteur y lit que LA REPARATION a perdu z ms.
        #
        # LE LEAD A MESURE LE CONTRAIRE, CINQ FOIS SUR CINQ: tout defaut mesure
        # sur un fichier produit est HERITE -- le maitre est court par rapport a
        # sa propre image et l'artefact le suit a la milliseconde. LA CAMPAGNE
        # N'A AUCUNE INSTANCE CONFIRMEE DU PIPELINE INTRODUISANT UN DEFAUT.
        #
        # Et ce journal ne peut pas trancher: il NOMME le maitre et ne donne
        # aucune de ses durees. FIDELITE A UNE SOURCE N'EST PAS CORRECTION D'UNE
        # SORTIE, et tout ce que ces lignes mesurent est de la fidelite. On emet
        # donc le manque AVEC son indecidabilite plutot que seul.
        shortfall = re.search(r"FILL SOURCE SHORT BY ([\d.-]+) ms; "
                              r"TRACK LOST ([\d.-]+) ms; UNEXPLAINED ([\d.-]+) ms",
                              plain(fields.get("fill")) or "")
        if shortfall:
            rows.append(_row("SHORTFALL", _redactor=redactor, track=order,
                             fill_source_short_by_ms=shortfall.group(1),
                             track_lost_ms=shortfall.group(2),
                             unexplained_ms=shortfall.group(3),
                             attribution=NOT_MEASURED,
                             detail="a shortfall MEASURED here is not a "
                                    "shortfall ATTRIBUTED. This log names the "
                                    "master and carries none of its durations, "
                                    "so nothing here can say whether the repair "
                                    "lost this or inherited it. Every produced "
                                    "defect measured in this campaign so far -- "
                                    "5 of 5 -- was INHERITED, and there is no "
                                    "confirmed instance of the pipeline "
                                    "introducing one. Fidelity to a source is "
                                    "not correctness of an output. THE NARROW "
                                    "FIX, named by forensic: the `repair: "
                                    "master` line names the master and emits "
                                    "none of its per-track durations. Three "
                                    "numbers on that line and this cell becomes "
                                    "measurable at render time",
                             # LE SEUL NOMBRE DE CE RAPPORT QUE JE N'AI PAS
                             # MESURE, donc le seul qui doit porter sa source et
                             # son sens de derive. `5 sur 5` vient du Lead, pas
                             # de mes octets, et il NE PEUT BOUGER QUE DANS UN
                             # SENS: une seule instance de defaut introduit
                             # l'invalide, et personne ne re-verifie un nombre
                             # qui arrange. C'est la meme asymetrie que mon
                             # compte de cellules vides qui ne pouvait que
                             # baisser -- et je ne peux pas le dater, parce
                             # qu'aucun artefact ne porte de champ de date.
                             # ET SA SOURCE EST UN MESSAGE, PAS UN DOCUMENT.
                             # Verifie par l'architecte: zero occurrence de ce
                             # chiffre dans MEASURING.MD, WRITE_ZONES.MD et
                             # AGENT.MD. Il n'existe que dans la couche de
                             # messages -- ET DANS CE RAPPORT. Donc le porteur le
                             # plus durable d'une affirmation non promue est un
                             # artefact qui n'appartient pas a son auteur, et un
                             # lecteur ne peut PAS remonter la citation.
                             #
                             # Le rapport le dit au lieu de le masquer. La partie
                             # DERIVEE DE MES OCTETS -- ce journal ne porte
                             # aucune duree du maitre, donc l'attribution est
                             # indecidable ici -- se tient sans ce chiffre; le
                             # chiffre n'est qu'un contexte emprunte, et il est
                             # marque comme tel.
                             count_source="vmsam-lead, message layer only: ZERO "
                                          "occurrences in MEASURING.MD, "
                                          "WRITE_ZONES.MD or AGENT.MD (checked). "
                                          "This report is currently its most "
                                          "durable carrier and there is no "
                                          "citation to follow back",
                             count_direction="can only be invalidated, never "
                                             "confirmed further: one introduced "
                                             "defect ends it, and a number that "
                                             "flatters is not re-checked",
                             count_dated=NOT_MEASURED))

        # LE RYTHME, ET CE QUE `speed=` NE DIT PAS.
        rate = applied_ratio(fields)
        speed_text = plain(fields.get("speed"))
        if isinstance(rate, Decimal):
            rows.append(_row("SPEED", _redactor=redactor, track=order,
                             ratio_applied=str(rate),
                             margin_state=NO_PRODUCER,
                             note="the winning transform's margin has no "
                                  "producer, so how decisively it won is unknown"))
        else:
            rows.append(_row("SPEED", _redactor=redactor, track=order,
                             ratio_applied_state=NO_PRODUCER,
                             rendering_of_an_accelerated_track=NOT_EXERCISED,
                             emitted=speed_text,
                             note="NOBODY ASKED. `no rate proposed by the "
                                  "measurement` is not a verdict of `no rate "
                                  "problem` -- no producer of a speed verdict "
                                  "exists in src/, so this track was never "
                                  "assessed for one"))

        # LA TETE, ET SES TROIS ETATS. Champ obtenu apres que j'ai depose
        # `container_start_time`: `head_pad_ms=0` couvrait `pas de decalage`,
        # `non mesure` et `un decalage que le plan lit par-dessus` avec un seul
        # chiffre. dev-2 a scinde le champ; les trois se disent maintenant.
        # Le champ ABSENT reste distinct des trois -- l'assemblage precede le
        # champ -- et c'est le quatrieme etat, pas un zero.
        head_pad = fields.get("head_pad")
        if head_pad:
            counts = {}
            for piece in plain(head_pad).split(","):
                name, _, value = piece.partition("=")
                if name.strip():
                    counts[name.strip()] = value.strip()
            rows.append(_row("HEAD", _redactor=redactor, track=order,
                             head_pad_ms=fields.get("head_pad_ms"),
                             none=counts.get("none"), padded=counts.get("padded"),
                             read_past=counts.get("read_past"),
                             unreported=counts.get("unreported"),
                             note="read_past is a real container offset the plan "
                                  "reads over: it costs no padding and used to "
                                  "print the same zero as no offset at all"))
        else:
            rows.append(_row("HEAD", _redactor=redactor, track=order,
                             head_pad_ms=fields.get("head_pad_ms"),
                             breakdown_state=ABSENT_FORMAT,
                             note="this assembly predates the per-piece head "
                                  "breakdown, so head_pad_ms=0 here still covers "
                                  "three states with one digit"))

        # s3.3: UNE etiquette pour toute la piste. Quand les regions portent des
        # decalages differents, on le DIT plutot que de laisser l'etiquette
        # passer pour une propriete de chaque region.
        rows.append(_row("TRACK_LABEL_LIMIT", track=order,
                         offset_label=fields.get("offset"),
                         offset_fidelity_state=COLLAPSED,
                         detail="offset_measured_is_all()_over_segments_and_"
                                "offset_fidelity_is_the_first_non-None;_both_"
                                "are_per-TRACK_over_per-REGION_values"))

        candidate_rows = []
        for index, region in enumerate(track_regions(job, order)):
            if region["kind"] == "CANDIDATE":
                candidate_rows.append(region)
            if region["kind"] == "LOST":
                ratio = applied_ratio(fields)
                original = None
                if isinstance(ratio, Decimal) and ratio and not region["dropped_unmeasured"] \
                        and region["dropped_ms"] is not None:
                    # La MEME perte, comptee en materiau d'origine du candidat.
                    original = _trim((region["dropped_ms"] / ratio).quantize(Decimal("0.01")))
                rows.append(_row("LOST", track=order, seq=index,
                                 timeline=lost_reference_frame(ratio),
                                 original_candidate_ms=original,
                                 candidate_start_ms=_trim(region["candidate_start_ms"]),
                                 candidate_end_ms=_trim(region["candidate_end_ms"]),
                                 candidate_start=clock(region["candidate_start_ms"]),
                                 candidate_end=clock(region["candidate_end_ms"]),
                                 dropped_ms=("UNMEASURED" if region["dropped_unmeasured"]
                                             else _trim(region["dropped_ms"])),
                                 # LE NOMBRE DU DESSIN EXISTE DANS LE TEXTE.
                                 dropped_s=(None if region["dropped_unmeasured"]
                                            else seconds_fr(region["dropped_ms"], 1)),
                                 where=region["where"]))
                continue
            rows.append(_row("REGION", track=order, seq=index, kind=region["kind"],
                             master_start_ms=_trim(region["master_start_ms"]),
                             master_end_ms=_trim(region["master_end_ms"]),
                             master_start=clock(region["master_start_ms"]),
                             master_end=clock(region["master_end_ms"]),
                             source=region["source"], offset_ms=region["offset"],
                             fill_width=region.get("fill_width"),
                             # LE NOMBRE QUE LA FIGURE DESSINE, DANS LE TEXTE.
                             #
                             # L'assertion que je viens d'ecrire a leve sur un
                             # vrai rendu: le dessin porte `22:40 -> 24:39
                             # +1:59` et AUCUNE ligne ne contenait `1:59`. La
                             # duree est calculee dans la figure et n'existait
                             # nulle part dans le texte grepable -- exactement
                             # ce que ce rapport promet de ne pas faire, promis
                             # depuis le debut et jamais verifie.
                             #
                             # Ajoute ici plutot que retire de la figure: le
                             # proprietaire lit le dessin, la duree lui sert, et
                             # la regle est "rien dans le dessin qui ne soit
                             # dans le texte", pas "rien de derive dans le
                             # dessin".
                             duration=short_clock(region["master_end_ms"]
                                                  - region["master_start_ms"]),
                             offset_s=(seconds_fr(region["offset"].value, 3)
                                       if region["offset"].state in (PRESENT, DERIVED)
                                       and region["offset"].value not in (None, "n/a")
                                       else None),
                             # ET LE RESTE PASSE, PAR NOM. Troisieme liste fixe
                             # de la nuit dans ce module, et la meme correction:
                             # ce qui n'est pas nomme ici doit sortir quand meme,
                             # sinon un champ ajoute en amont pour ce rapport
                             # n'y arrive jamais -- ce qui est exactement ce qui
                             # est arrive a `why=`.
                             **{name: value for name, value in region.items()
                                if name not in _REGION_FIELDS_RENDERED}))

        # LE PAS DE DECALAGE D'UNE REGION A LA SUIVANTE, qui est ce que le
        # proprietaire ecrit en saumon au-dessus de la regle. VERIFIE sur ses
        # deux images plutot que suppose: sur l'image 1 les trois nombres
        # saumon -23,0 -30,6 -27,2 sont exactement les pas -23,022 -30,613
        # -27,195 entre etiquettes bleues consecutives, et les deux pas NON
        # etiquetes sont les petits, -0,3 et +0,2. Ce n'est donc NI la duree
        # jetee NI le decalage de la region -- deux lectures que j'avais
        # proposees, et la seconde tombait pres par coincidence sur un artefact.
        steps = []
        for previous, following in zip(candidate_rows, candidate_rows[1:]):
            if not (previous["offset"].state in (PRESENT, DERIVED)
                    and following["offset"].state in (PRESENT, DERIVED)):
                continue
            try:
                delta = Decimal(str(following["offset"].value)) - \
                        Decimal(str(previous["offset"].value))
            except Exception:
                continue
            steps.append((previous["master_end_ms"], delta))
            rows.append(_row("STEP", _redactor=redactor, track=order,
                             at_master_ms=_trim(previous["master_end_ms"]),
                             at_master=clock(previous["master_end_ms"]),
                             step_ms=_trim(delta), step_s=seconds_fr(delta, 1)))

    for fields in job.get("subtitles") or []:
        rows.append(_row("SUBTITLE", track=fields.get("stream_order"),
                         lang=fields.get("lang"), format=fields.get("format"),
                         kept_cues=fields.get("kept_cues"),
                         dropped_cues=fields.get("dropped_cues")))

    # UNE MARQUE ABSENTE DOIT S'ANNONCER, exactement comme une cellule vide.
    # C'est le meme defaut d'un cran au-dessus: la ou une cellule vide risquait
    # de se lire comme `zero`, une MARQUE ENTIERE qui ne se dessine jamais
    # risque de se lire comme `rien n'est jamais refuse`. Ce n'est PAS un manque
    # -- `log_assembly` emet bien `repair: SKIPPED` -- c'est un cas que cet
    # artefact n'exerce pas, et les deux ne partagent pas un blanc.
    if not (job.get("refused") or []):
        # LA PRECONDITION SE MESURE, donc on la donne au lieu de dire seulement
        # que le cas ne s'est pas produit. `repair: SKIPPED segment` vient de
        # `drop_unverified_segments`: un segment PLUS COURT QUE LA FENETRE DE
        # SONDE est jete et rempli depuis le maitre. `PROBE_WINDOW_SECONDS` vaut
        # 60 s et le plus court segment candidat du corpus fait 80 s -- donc la
        # precondition n'a jamais ete remplie, de 20 s. Ce n'est ni du code mort
        # ni une branche inatteignable: c'est une branche dont ce corpus ne
        # produit pas l'entree, et la marge est etroite.
        shortest = None
        for piece in plan.get("pieces") or []:
            if piece["source"] != "candidate":
                continue
            length = piece["master_end_ms"] - piece["master_start_ms"]
            if shortest is None or length < shortest:
                shortest = length
        rows.append(_row("REFUSED_NONE", _redactor=redactor,
                         count=0, state=NOT_EXERCISED,
                         producer="merge_video_repair.drop_unverified_segments "
                                  "then log_assembly `repair: SKIPPED segment`",
                         precondition="a plan segment SHORTER than the "
                                      "measurement's probe window "
                                      "(PROBE_WINDOW_SECONDS=60s)",
                         shortest_candidate_segment_ms=_trim(shortest),
                         note="this artefact refuses no candidate segment, so "
                              "the figure draws no dashed amber box. NOT a claim "
                              "that nothing is ever refused. The mark has never "
                              "been drawn against real refused material"))
        # TROIS SILENCES DERRIERE UN MEME BLANC, et elles ne se valent pas.
        # Enumerees ici plutot que dans un message, parce que la figure ne peut
        # pas les distinguer et qu'un lecteur les lira toutes comme "rien de
        # refuse". Une seule porte un nombre.
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="segment-level: drop_unverified_segments never "
                                "fired", applies="YES on this artefact",
                         bound=f"measurable -- probe window 60 s, shortest "
                               f"candidate segment here "
                               f"{_trim(shortest) if shortest else '?'} ms"))
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="per-track: declined/failed needs SOME tracks to "
                                "succeed while OTHERS fail on one file",
                         applies=("YES on this artefact"
                                  if (job.get("audios") or {}) else "unknown"),
                         bound="0 per-track SKIPPED on 3 of 3 speed-mismatch "
                               "cases (dev-2, measured: 26 no_plan, 27 declined, "
                               "29 no_plan). The population most likely to "
                               "produce mixed success produced none -- a "
                               "measured negative, on 3 cases of one release"))
        rows.append(_row("SILENCE", _redactor=redactor, mark="amber refused box",
                         reason="no tracks were attempted: the repair died "
                                "before any track was built",
                         applies="CANNOT APPEAR HERE",
                         bound="such a run emits no `repair: plan` line, so this "
                               "reader rejects it by structure and it is absent "
                               "from every count in this report -- see "
                               "rejected_by_structure on the CORPUS row"))

    if job.get("failed"):
        rows.append(_row("FAILED", _redactor=redactor, file_produced="NO",
                         reason=job["failed"],
                         note="a TOOL FAULT escaped before any verdict existed. "
                              "Not a decision about the media: nobody decided"))
    if job.get("undelivered"):
        rows.append(_row("UNDELIVERED", _redactor=redactor,
                         state=job["undelivered"]["state"],
                         file_local_only=job["undelivered"]["path"],
                         note="REFUSED = the gate decided against it. NOVERDICT "
                              "= nobody decided. The artefact exists on disk "
                              "under this name and is NOT counted as produced"))
    elif job.get("failed") or job.get("declined"):
        rows.append(_row("UNDELIVERED", _redactor=redactor, state=NOT_EXERCISED,
                         note="no `undelivered` line: no artefact existed to "
                              "mark, so the raise came before the mux. An "
                              "absence here is a THIRD fact, not a missing line"))
    if job.get("declined"):
        # LE VERDICT EN PREMIER, pas en fin de liste: ce rapport decrit
        # normalement un FICHIER PRODUIT, et ici il n'y en a pas.
        rows.append(_row("DECLINED", _redactor=redactor,
                         file_produced="NO",
                         reason=job["declined"],
                         note="this report describes a repair that was REFUSED. "
                              "No file was produced. Everything below describes "
                              "what the plan WOULD have done, not what any "
                              "artefact contains"))
    if job.get("plan") and not job.get("output_check"):
        # L'ABSENCE DE LIGNE DE RESUME SE DIT, avec sa raison quand on l'a.
        rows.append(_row("NO_OUTPUT_CHECK", _redactor=redactor,
                         reason=("the repair was DECLINED: the `output file` "
                                 "summary is written past the point where the "
                                 "gate raises, so it does not exist"
                                 if job.get("declined") else
                                 "no `output file` line on this artefact and no "
                                 "decline recorded either -- unexplained"),
                         state=(PRESENT if job.get("declined") else NOT_MEASURED)))
    for entry in job.get("unparsed") or []:
        rows.append(_row("UNPARSED", _redactor=redactor, line=entry,
                         # ALARME SUR CE LECTEUR, PAS FAIT SUR L'ARTEFACT.
                         # Construction prise a dev-2, qui l'a appliquee a son
                         # propre classificateur: une valeur qu'aucune entree ne
                         # peut atteindre est etiquetee de sorte qu'un ZERO ne
                         # puisse pas se lire comme une mesure et qu'un NON-ZERO
                         # se lise comme "l'instrument est perime".
                         #
                         # Ici c'est le sens inverse et la meme idee: une ligne
                         # `UNPARSED` NE DIT RIEN SUR L'ARTEFACT. Elle dit que
                         # CE lecteur ne connait pas encore cette ligne. Trois
                         # fois ce soir -- `DECLINED`, `build`, `sources` -- et
                         # les trois fois le producteur avait raison et moi du
                         # retard.
                         reads_as="an alarm about THIS READER, not a fact about "
                                  "the artefact: the producer emitted a line "
                                  "this parser does not know yet",
                         note="kept rather than dropped, because a reader that "
                              "discards what it does not know discards exactly "
                              "what is new"))

    for entry in job.get("refused") or []:
        # CE QUI EST REFUSE DU CANDIDAT -- reglage du proprietaire. C'est CETTE
        # region que la boite ambre pointillee marque, et pas un remplissage
        # depuis le maitre. Une region peut etre l'un, l'autre, les deux ou
        # aucun; les confondre rendrait `refuse` et `rempli depuis le maitre`
        # indistinguables, qui est le defaut meme que ce rapport combat.
        rows.append(_row("REFUSED", _redactor=redactor,
                         master_start_ms=_trim(entry["master_start_ms"]),
                         master_end_ms=_trim(entry["master_end_ms"]),
                         master_start=clock(entry["master_start_ms"]),
                         master_end=clock(entry["master_end_ms"]),
                         dropped_ms=_trim(entry["dropped_ms"]),
                         reason=entry.get("reason")))

    for entry in job.get("skipped") or []:
        rows.append(_row("SKIPPED", _redactor=redactor,
                         detail=entry.replace(" ", "_")))

    check = job.get("output_check")
    if check:
        rows.append(_row("CHECK", **{k: v for k, v in check.items()}))
        # `audio_tracks=1/1` NE DIT PAS QUE LE FICHIER A UNE PISTE.
        #
        # Le denominateur compte ce que la reparation a RECONSTRUIT, pas ce que
        # le fichier porte. forensic a mesure HUIT flux audio dans un fichier
        # dont cette ligne rend `1/1`. Une fraction dont les deux membres sont
        # egaux se lit comme une completude, et celle-ci est une completude a
        # propos du travail et non a propos du fichier.
        if check.get("audio_tracks"):
            rows.append(_row("CHECK_DENOMINATOR_LIMIT",
                             audio_tracks=check.get("audio_tracks"),
                             counts="tracks the repair REBUILT",
                             does_not_count="tracks present in the produced file",
                             measured="a file rendering audio_tracks=1/1 was "
                                      "measured to carry EIGHT audio streams",
                             reads_as="an n/n here is completeness about the "
                                      "WORK, never about the FILE",
                             state=COLLAPSED))
    # LE SEUL ENDROIT DU JOURNAL OU LE MOTEUR SE NOTE LUI-MEME.
    #
    # `prediction` porte un RESULTAT DE CONTROLE et non une mesure: le
    # producteur annonce combien de refus il attend, puis dit si l'issue lui a
    # donne raison. Rendue telle quelle, avec son propre denominateur devant --
    # `agreement=held` sur zero refus predit est un accord qu'un predicteur
    # constant obtiendrait aussi, et cette ligne doit le dire elle-meme plutot
    # que laisser un lecteur le supposer.
    prediction = job.get("prediction")
    if prediction:
        predicted = prediction.get("predicted")
        try:
            opportunities = int(str(predicted).strip())
        except (TypeError, ValueError):
            opportunities = None
        rows.append(_row(
            "PREDICTION",
            **dict(prediction),
            reads="the producer testing ITS OWN duration gate: it announces the "
                  "refusals it expects, then reports whether the outcome agreed",
            of_which_could_have_disagreed=(
                predicted if opportunities else
                "0 -- a prediction of no refusals AGREES with any run that "
                "refuses nothing, which a constant predictor would also score. "
                "UNTESTED on this artefact, not confirmed"
                if opportunities == 0 else
                Cell(NOT_MEASURED, "`predicted` is not a number this reader can "
                                   "read, so it cannot say what the agreement "
                                   "was worth"))))
    for entry in job.get("predicted_refusals") or []:
        rows.append(_row("PREDICTED_REFUSAL", **dict(entry),
                         reads="announced BEFORE the mux, by the producer, "
                               "naming the track and the shortfall that will "
                               "trip the gate"))
    if job.get("summary_counts"):
        rows.append(_row("SUMMARY_COUNTS", text=job["summary_counts"].replace(" ", "_"),
                         note="a_count_is_a_statement_about_work_done,_not_about_a_file"))

    for entry in job.get("brackets") or []:
        # LE FAIT, ET LA SIGNATURE A COTE PLUTOT QU'A LA PLACE.
        #
        # `bound_only` est desormais EMIS. Ma signature derivee reste rendue
        # sur la meme ligne, non pas comme un repli silencieux mais pour que le
        # lecteur puisse les voir DIVERGER: le jour ou elles ne s'accordent
        # plus, c'est un fait sur le localisateur et pas sur ce rapport.
        width = entry.get("width_ms")
        try:
            derived = float(width) == float(SEARCH_BOUND_MS)
        except (TypeError, ValueError):
            derived = None
        rows.append(_row("BRACKET", _redactor=redactor,
                         agrees_with_my_derived_signature=(
                             None if derived is None else
                             str(derived) == str(entry.get("bound_only"))),
                         derived_signature=(
                             None if derived is None else
                             "width == the 100 s search bound" if derived else
                             "width != the search bound"),
                         note="the locator's OWN bracket for this change point. "
                              "`bound_only` is now emitted and this row prints "
                              "it; the derived signature beside it is kept so a "
                              "reader can watch the two agree",
                         **{k: v for k, v in entry.items()}))
    for entry in job.get("segments") or []:
        rows.append(_row("SEGMENT", _redactor=redactor,
                         note="the locator's own segmentation, with the offset "
                              "it measured for each -- per stream where the "
                              "producer gives it per stream",
                         **{k: v for k, v in entry.items()}))
    if job.get("output_durations"):
        rows.append(_row("OUTPUT_DURATIONS", _redactor=redactor,
                         note="measured on the PRODUCED FILE: container length, "
                              "longest a/v stream, and the length expected of "
                              "it. A per-FILE quantity -- still not per-track",
                         **{k: v for k, v in job["output_durations"].items()}))

    for entry in job.get("locator_measurements") or []:
        # TOUTE VALEUR EMISE, PAR NOM. Aucune liste fixe: si dev-1 ajoute un
        # septieme nombre demain il apparait, et s'il en retire un la colonne
        # disparait au lieu de rendre un vide qui se lirait comme un zero.
        rows.append(_row("LOCATOR", _redactor=redactor,
                         note="the locator's own measurement of this pair, "
                              "read by name from the line it emits",
                         # UNE VALEUR VIDE N'EST PAS UNE VALEUR.
                         #
                         # Mesure sur `offset_ms= points=0`: `split_fields` rend
                         # `{'offset_ms': ''}`, et une cellule vide SE LIT COMME
                         # UN ZERO. C'est exactement le defaut deja paye une fois
                         # -- `RESAMPLED x` sans rapport, pendant que la prose a
                         # cote niait le reechantillonnage. Un f-string sur une
                         # valeur absente produit ceci sans que personne le
                         # veuille, donc on le dit en toutes lettres au lieu de
                         # rendre un blanc.
                         **{k: (v if v != "" else
                                "the producer emitted this key with nothing "
                                "after it")
                            for k, v in entry.items()}))
    for entry in job.get("locator_notes") or []:
        rows.append(_row("LOCATOR_NOTE", _redactor=redactor,
                         digest=entry["digest"], chars=entry["chars"],
                         carries_path=entry["carries_path"],
                         tail=entry["tail"] or None,
                         reads_as="a locator line that is prose and not "
                                  "`key=value` -- a decline, or a probe that "
                                  "failed",
                         note="content withheld: `probe at ...s failed: {error}` "
                              "interpolates an arbitrary ffprobe exception, and "
                              "those carry the input path"))

    for entry in job.get("foreign_lines") or []:
        # Le FAIT qu'une ligne hors reparation existe, et de quoi la retrouver
        # dans le journal. Jamais son texte.
        rows.append(_row("OUTSIDE_REPAIR", _redactor=redactor,
                         digest=entry["digest"], chars=entry["chars"],
                         carries_path=entry["carries_path"],
                         tail=entry["tail"] or None,
                         note="content withheld: free text outside the repair "
                              "vocabulary cannot be redacted with a guarantee"))

    # LES CONTROLES DE CE RAPPORT DECLARENT S'ILS ONT SERVI.
    #
    # Meme discipline que pour les champs des producteurs, retournee sur moi.
    # Un controle presente comme un succes avant d'avoir jamais rien attrape est
    # exactement la marque ambre qui ne se dessine jamais: une absence lue comme
    # une reponse. `0 redaction` ne dit PAS `rien a rediger` -- il dit `rien ne
    # correspondait aux motifs`, et mes motifs ont deja rate un titre de serie
    # qu'un chemin a espaces portait.
    #
    # Et le compte est PAR RENDU, pas historique: le journal ne porte ni build
    # ni horodatage, donc ce rapport ne peut rien affirmer sur ses propres
    # executions passees -- meme cellule `build_identity`, revenue encore.
    for gap in blank_cells(job, corpus):
        # UN ETAT QUI PEUT PERIMER SANS QUE PERSONNE NE TOUCHE AU CODE PORTE SA
        # PORTEE. Les six autres etats sont des proprietes du code et restent
        # vraies tant que le code ne bouge pas; celui-ci est une propriete DU
        # CORPUS, et un `not-exercised-here` lu dans six mois se prendrait pour
        # un fait sur le pipeline. Il ne peut pas non plus etre DATE d'ici --
        # aucun champ de build, de commit ou d'horodatage n'est emis nulle part,
        # ce qui est la cellule `build_identity` deposee le premier jour,
        # revenant comme une consequence concrete plutot qu'un principe.
        rows.append(_row("GAP", _redactor=redactor,
                         quantity=gap["quantity"], state=gap["state"],
                         addressed_to=gap["address"], detail=gap["detail"],
                         **({"observed_on": artefact_id,
                             "scope": "this artefact only; a property of the "
                                      "corpus and not of the code, and undatable "
                                      "from this record"}
                            if gap["state"] == NOT_EXERCISED else {})))
    # LES CONTROLES SE DECLARENT EN DERNIER, APRES TOUT CE QU'ILS COUVRENT.
    # Ecrits avant les lignes GAP, ils rapportaient un compte de redactions
    # PRIS AVANT une partie du travail -- un controle qui publie un chiffre
    # arrete plus tot que ce qu il mesure. Le defaut exact que je viens de
    # decrire chez les autres, commis en l ecrivant.
    rows.append(_row("CONTROL", _redactor=redactor, name="redaction",
                     fired_on_this_render=redactor.hits,
                     # LE DENOMINATEUR A TROUVE SON PREMIER DEFAUT
                     # IMMEDIATEMENT, ET C'ETAIT CELUI-CI.
                     #
                     # `fired_on_this_render=0` s'imprime dans CHAQUE rapport
                     # depuis le debut et se lit "rien n'avait besoin d'etre
                     # redige". La verite est que `REDACT_MEDIA_NAMES` vaut
                     # False -- reglage du proprietaire, les noms de media SONT
                     # permis ici -- donc le redacteur rend son texte AVANT de
                     # compter quoi que ce soit. LE CONTROLE NE TOURNE PAS.
                     #
                     # Un zero de resultat sans son denominateur ne distingue
                     # pas "rien a attraper" de "l'instrument est eteint", et
                     # c'est la lecture flatteuse qui gagne par defaut.
                     values_examined=redactor.examined,
                     could_have_fired=(
                         redactor.examined if redactor.examined else
                         "ZERO. REDACT_MEDIA_NAMES is False on this build -- "
                         "media names are permitted in this report by the "
                         "owner's ruling -- so this control DID NOT RUN. The "
                         "zero above is the instrument being off, not a clean "
                         "result"),
                     checks="absolute paths, catalogue ids and media filenames "
                            "in emitted values, replaced by a stable opaque token",
                     limit="pattern-based, and a pattern cannot survive text it "
                           "does not own: a path containing spaces let a show "
                           "title through, measured. The real control is that "
                           "free text outside the repair vocabulary is never "
                           "reproduced at all"))
    rows.append(_row("CONTROL", _redactor=redactor, name="leak_assertion",
                     fired_on_this_render=0,
                     could_have_fired="the whole finished document, re-read",
                     checks="the finished document is re-read and the render "
                            "RAISES rather than corrects if anything survives",
                     limit="it raises, so a zero here is the only outcome a "
                           "reader can ever see -- this line records that the "
                           "control ran, not that there was nothing to catch"))
    rows.append(_row("CONTROL", _redactor=redactor, name="corpus_sanity",
                     fired_on_this_render=0 if corpus else None,
                     could_have_fired=("1 comparison: n_distinct against n"
                                       if corpus else
                                       "0 -- NO POPULATION, so this control had "
                                       "no opportunity to fire at all"),
                     state=None if corpus else NOT_SUPPLIED,
                     checks="a distinct-count can never exceed its population",
                     limit="it passes on a WRONG count as readily as a right "
                           "one: 15 and 16 both satisfy it, and 15 was the "
                           "wrong unit. NEVER FIRED"))

    return rows


# Les noms que la ligne REGION rend DEJA explicitement. Tout autre nom present
# sur la region sort tel quel: la liste dit ce qui est DEJA PRIS, jamais ce qui
# a le droit d'exister.
# Les noms que la ligne PLAN rend deja explicitement, plus ceux consommes
# ailleurs (`pieces` devient un COMPTE, la geometrie est rendue region par
# region).
_PLAN_FIELDS_RENDERED = frozenset((
    "kind", "language", "quantum", "quantum_ms", "speed_margin",
    "speed_margin_absent_reason", "fidelity_margin", "decided_by", "pieces",
))

_REGION_FIELDS_RENDERED = frozenset((
    "kind", "master_start_ms", "master_end_ms", "source", "offset",
    "fill_width", "from", "candidate_start_ms", "candidate_end_ms",
    "dropped_ms", "dropped_unmeasured", "where",
))

_TRACK_FIELDS_RENDERED = frozenset((
    "lang", "fill", "filled_ms", "silence_ms", "head_pad_ms", "speed", "offset",
    "verify", "probes", "worst", "r_min", "verified",
    # portes par d'autres lignes, pas perdus:
    "residual", "quantum", "head_pad"))


def blank_cells(job, corpus=None):
    """LE REGISTRE DES CELLULES VIDES. C'est la sortie la plus utile du module.

    Chaque entree est une quantite que s4e ou s4g exige et qui n'arrive pas
    jusqu'ici, avec SON ETAT et SON ADRESSE. Une cellule vide etiquetee est un
    constat; une cellule vide remplie est un mensonge. Trois defauts de
    divulgation ont ete trouves le 4 septembre et LES TROIS PAR CHANCE -- un
    rendu les trouve par construction.
    """
    generation, _ = format_generation(job)
    entries = []

    if not job.get("master_line_present"):
        entries.append({
            "quantity": "master_identity",
            "state": ABSENT_FORMAT if generation < 3 else PRESENT,
            "address": "merge_video_repair.log_assembly",
            "detail": "the master is not named, so a produced deficit cannot "
                      "be attributed to the merge or to the master"})

    used = bool(job.get("regions_used") or {})
    if not used:
        entries.append({
            "quantity": "per region offset by name",
            "state": ABSENT_FORMAT if generation >= 2 else NO_PRODUCER,
            "address": "merge_video_repair.log_assembly (USED_line)",
            "detail": "derived here from CUT bounds against plan bounds; "
                      "derivation is conditional on the plan having cut something"})

    entries.append({
        "quantity": "per region offset_fidelity",
        "state": COLLAPSED,
        "address": "merge_video_chimeric.assemble_on_master_timeline",
        "detail": "offset_fidelity is the first non-None across segments and "
                  "offset_measured is all() over them; per-segment values exist "
                  "upstream in candidate offset_fidelity by stream"})
    entries.append({
        "quantity": "kept candidate region provenance",
        "state": NO_PRODUCER if not used else PRESENT,
        "address": "merge_video_repair.log_assembly",
        # L'ETAT ET SA RAISON DOIVENT S'ACCORDER. `state=present` sous un detail
        # qui dit "aucune ligne ne les couvre" est une cellule qui se contredit,
        # et c'est pire qu'une cellule vide: un lecteur en croit une moitie.
        "detail": ("the USED line covers them: master bounds, candidate bounds "
                   "and the offset applied, per track and per region"
                   if used else
                   "no line type covers the regions the output takes from the "
                   "candidate, which are the majority of every file")})
    # LA CELLULE QUE MA NARRATION RENVOYAIT ET QUI N'EXISTAIT PAS.
    #
    # Le paragraphe "cette figure ne peut pas dire POURQUOI" dit depuis le
    # debut: "voir `gap_is_filled_because_unsure` pour l'adresse". IL N'Y AVAIT
    # AUCUNE ENTREE DE CE NOM. Un renvoi vers une cellule absente, dans le
    # document meme dont le sujet est qu'une absence doit se dire. Trouve en
    # cherchant ou basculer l'etat, pas par une relecture.
    #
    # ET ELLE ARRIVE `present`, PAS `no-producer`: dev-1 et dev-2 ont fait
    # atterrir la ligne. Verifie sur les octets emis, 16 crochets sur 5
    # journaux.
    emitted_brackets = len(job.get("brackets") or [])
    entries.append({
        "quantity": "gap_is_filled_because_unsure",
        "state": PRESENT if emitted_brackets else NO_PRODUCER,
        "address": "change_point_locator -> merge_video_repair "
                   "(repair: bracket line)",
        "detail": (
            f"EMITTED ON THIS ARTEFACT: {emitted_brackets} `repair: bracket` "
            f"line(s) carrying low_ms, high_ms, width_ms, bound_only, step_ms "
            f"and step_points. `bound_only=True` is the locator SAYING it could "
            f"not narrow the bracket, which is the question this report existed "
            f"to answer and could previously only infer. THE INFERENCE WAS "
            f"RIGHT, AND THE COUNT IS MEASURED AT RENDER TIME: "
            + (corpus["bracket_agreement"] if corpus else
               "NO POPULATION SUPPLIED to this render, so this report cannot "
               "say how often the derived signature and the emitted "
               "`bound_only` agree -- it can only show you both on the BRACKET "
               "rows of THIS artefact. The corpus-scale figure this sentence "
               "used to recite was measured on five logs and typed in; it is "
               "gone")
            + f". Both are printed on the BRACKET row so a reader can watch "
              f"them diverge"
            if emitted_brackets else
            "NOT ON THIS ARTEFACT. The locator computes it; this log carries no "
            "`repair: bracket` line, so for this file the width signature is "
            "all there is. Other artefacts in this corpus DO carry it -- an "
            "absence here is a fact about this record and not about the field")})
    entries.append({
        "quantity": "plateau_tolerance_ms",
        "state": NO_PRODUCER,
        "address": "change_point_locator.locate_change_points",
        "detail": "PLATEAU_TOLERANCE_MS=50.0 is defined and never returned; it "
                  "is the gate that shifts the plateau mean. NOT tolerance_ms=500,"
                  " which is the duration-enforcement tolerance on the CHECK row"})
    entries.append({
        "quantity": "step_floor_ms",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly",
        "detail": "MIN_STEP_MS=60.0 IS returned by the locator and is never "
                  "printed; the gate reaches the plan dict and dies at the emitter"})
    entries.append({
        "quantity": "speed_margin",
        "state": NO_PRODUCER,
        # L'ADRESSE A CHANGE ET L'ETAT NON. dev-1 PRODUIT desormais la marge --
        # et deux d'entre elles, `speed_margin` en ms de platitude et
        # `fidelity_margin` sans dimension, parce qu'une seule cle portant
        # "la statistique qui a tranche" aurait une UNITE VARIABLE d'une ligne a
        # l'autre. Mais elles vivent dans un JSON de transfert et AUCUNE
        # n'atteint le journal: zero occurrence de `speed_margin` dans les
        # treize artefacts. Le defaut n'est donc plus "personne ne la mesure",
        # c'est "elle ne traverse pas jusqu'ici", et l'adresse est le chemin
        # entre les deux.
        # UN DEFAUT DEPOSE CHEZ LE MAUVAIS PRODUCTEUR EST UN DEFAUT QUE
        # PERSONNE NE CORRIGERA.
        #
        # J'ecrivais "la couture: dev-1 l'emet dans un JSON de passage". Trace
        # par dev-2 et VERIFIE ICI SUR L'ARBRE: aucune assignation de
        # `speed_margin`, `speed_margin_absent_reason`, `fidelity_margin` ni
        # `decided_by` n'existe NULLE PART dans src/. Chaque occurrence hors de
        # ce module est une LECTURE -- `plan.get(...)` -- ou un commentaire. Le
        # plan vient de `change_point_locator.locate_change_points`, dont le
        # dictionnaire retourne ne porte aucune de ces cles.
        #
        # Donc la branche de raison n'est pas EN ATTENTE d'un emetteur: elle est
        # INATTEIGNABLE pour tout plan que cette chaine peut produire
        # aujourd'hui. UNE BRANCHE DE LECTEUR N'EST PAS UNE OCCASION SI AUCUN
        # ECRIVAIN NE PEUT L'ATTEINDRE.
        "address": "NO WRITER EXISTS. Verified across src/: every occurrence of "
                   "`speed_margin`, `speed_margin_absent_reason`, "
                   "`fidelity_margin` and `decided_by` outside this module is a "
                   "READ (`plan.get(...)`) or a comment; nothing assigns them. "
                   "The plan comes from change_point_locator.locate_change_points "
                   "and its returned dict carries none of these keys. Addressed "
                   "THERE -- NOT to the emitter in merge_video_repair, which is "
                   "correct and simply never fed. THE TOKEN IS A MARKER OF AN "
                   "OPEN DECISION AND IT HAS AN OBSERVABLE END -- not a "
                   "permanent record, which is what this cell said an hour ago "
                   "and what the ruling it quoted has since amended. WHILE the "
                   "question of whether the locator should originate these keys "
                   "is open, the token is the only artefact-visible evidence "
                   "that it IS open: as a measurement of the plan it carries "
                   "nothing, as a record it is the sole trace that four keys "
                   "were designed, consumed and never originated, and cutting "
                   "it leaves that fact only in observers like this row, never "
                   "in the produced record. WHEN the question closes: if the "
                   "keys are to be originated, the fields FILL and there is "
                   "nothing to cut; if the design is dropped, the fields COME "
                   "OUT and this cell changes BEFORE the bytes do, so no "
                   "artefact ever renders a state this reader cannot account "
                   "for. A PERMANENT FIELD EMITTING ONE CONSTANT FOREVER WOULD "
                   "BE THE SAME DEFECT IN ANOTHER COSTUME, which is why the "
                   "end condition is written here rather than assumed. THIS "
                   "CELL HAS BEEN WRONG TWICE TODAY IN OPPOSITE DIRECTIONS -- "
                   "first recommending a cut, then forbidding one -- and both "
                   "corrections came from the field's producer rather than from "
                   "this reader re-reading itself",
        "detail": (f"0 occurrences across {corpus['logs']} logs / "
                   f"{corpus['distinct_cases']} distinct cases (see the CORPUS "
                   f"row: NOT an independent sample). "
                   if corpus else
                   "not seen on this artefact; NO POPULATION SUPPLIED, so this "
                   "report cannot say how many artefacts were looked at. ")
                  + "AND THE EMITTER WILL FAIL ON "
                  "ARRIVAL, PRECISELY WHERE IT MATTERS: log_assembly emits the "
                  "field CONDITIONAL ON ITS TRUTHINESS, and a margin that is "
                  "undefined is None, which is falsy. So on exactly the files "
                  "where the fidelity gate decided and no flatness margin "
                  "exists, the line prints nothing and this report would say NON "
                  "EMISE over a decision with a 0.3637 separation. "
                  "speed_margin_absent_reason, fidelity_margin and decided_by "
                  "need emitters OF THEIR OWN rather than riding on "
                  "speed_margin's truthiness -- otherwise the reason never "
                  "travels in the one case it exists to explain. NOTE for any "
                  "caption: fidelity_margin is winner MINUS best hypothesis that "
                  "did NOT clear the gate -- a separation across the selection "
                  "boundary, not a spread among the accepted, and wording it as "
                  "a spread would make it circular"})
    entries.append({
        "quantity": "verification_probe_positions",
        "state": COLLAPSED,
        "address": "merge_video_chimeric.verify_on_master_timeline",
        "detail": "per-probe master_position_ms lag_ms and correlation are built "
                  "and summarised to probes worst r_min; a max() has no position "
                  "so it cannot land on a timeline"})
    # UNE QUANTITE QUE PERSONNE N'AVAIT EN TETE, ET C'EST LA LIMITE DU REGISTRE
    # LUI-MEME. forensic a mesure QUATRE CHAPITRES dans un fichier produit,
    # bornes et titres identiques a ceux du maitre. Ce rapport ne les mentionne
    # nulle part -- ni ligne, ni cellule, ni absence admise -- et `mergeVideo`
    # passe `--no-chapters`, les deux modules de reparation ne contiennent pas
    # le mot, et aucun site du depot ne les ajoute.
    #
    # UN REGISTRE DE CELLULES VIDES ENUMERE CE QUE LA SPECIFICATION NOMME. IL NE
    # PEUT PAS PORTER `no-producer` POUR UNE GRANDEUR A LAQUELLE PERSONNE N'A
    # PENSE. C'est la limite structurelle de tout ce module: il rend visible ce
    # qui manque PARMI CE QU'ON SAIT DEVOIR ETRE LA, et il est aveugle a ce que
    # le fichier contient et que personne n'a nomme.
    #
    # Trouve par forensic, dont les entrees sont LE FICHIER PRODUIT, la ou les
    # miennes sont le journal. Un journal ne mentionne pas ce qu'aucun code ne
    # revendique avoir ecrit.
    entries.append({
        "quantity": "audio_normalisation_gain_and_its_output_rate",
        "state": NO_PRODUCER,
        "address": "video.generate_normalised_file -- neither the gain nor the "
                   "output sample rate is emitted on a job that SUCCEEDS",
        "detail": (
            "A SECOND INSTANCE OF THE CELL BELOW, FOUND WHILE IT WAS BEING "
            "WRITTEN. The pipeline applies `highpass=f=60,lowpass=f=16000,"
            "volume=<gain>dB` whenever |gain| >= 0.5 dB (video.py:528-538), and "
            "forensic measured that chain pinning samples to full scale as a "
            "function of the OUTPUT RATE: at 32000 Hz a 0.50 dB gain pins "
            "79.86% of samples, while at 44100 and 48000 the same chain pins "
            "0.00% even at 20 dB. THE 0.5 dB EDGE IS A BRANCH CONDITION, NOT A "
            "PHYSICAL THRESHOLD -- 0.499 takes the `anull` path and 0.500 takes "
            "the filter. "
            "MEASURED ON THE LOGS THIS READER CAN SEE: 7 carry the chain, and "
            "ALL SEVEN gains are >= 0.5 in absolute value (-5.81 -3.96 -0.53 "
            "1.43 1.61 4.74 4.87), so the filter branch is the only one "
            "exercised here. The rates observed are 48000 (x4) and 44100 (x1); "
            "32000 DOES NOT OCCUR in anything this reader holds, and at the "
            "rates that do occur the measured pinning is 0.00%. "
            "WHY THE CELL EXISTS ANYWAY -- KIND: TRACE, read and derived, "
            "not run, and this claim has already been narrowed once. The rate "
            "is NOT fixed in the code: `exportParam['SamplingRate']` "
            "(video.py:268) is `audioParam['SamplingRate']`, derived at "
            "mergeVideo.py:583 as the MINIMUM OVER THE SELECTED LANGUAGE'S "
            "STREAMS ON BOTH SIDES, clamped only from above at 44100. "
            "*** THAT DESCRIBES STAGE 2 AND IS NOT A PROPERTY OF THE DEFECT. "
            "THIS CELL SAID `A 32 kHz SOURCE IS NECESSARY AND NOT SUFFICIENT: "
            "THE ROUTE MUST ALSO SELECT THAT LANGUAGE`, ATTRIBUTING THE "
            "INSUFFICIENCY TO THE PAIR MINIMUM AND THE CLAMP. WITHDRAWN -- "
            "forensic retraction 22, and it is withdrawn as UNSCOPED rather "
            "than as false. *** THERE ARE TWO ROUTES TO THE SAME DEGENERATE "
            "FILTER. STAGE 2 uses the pair minimum and the one-sided clamp, "
            "where the old wording holds. STAGE 1 SETS NO `-ar` AT ALL: "
            "`prepare_get_delay_sub` builds its parameter dict WITHOUT a "
            "`SamplingRate` key, `compare_video.__init__` takes a `.copy()` so "
            "stage 2's assignment cannot propagate back, and video.py appends "
            "`-ar` ONLY `if 'SamplingRate' in exportParam`. SO A STREAM "
            "EXTRACTED AT STAGE 1 KEEPS ITS OWN RATE, and a natively-32000 one "
            "meets `lowpass=f=16000` at exactly its own Nyquist with NO pair, "
            "NO minimum and NO clamp involved. KIND: TRACE, traced by forensic "
            "and RE-VERIFIED HERE SITE BY SITE against the authority's "
            "mergeVideo.py blob 942269ba39dc37f820e1d79a96b8d2a63b6213fd and "
            "video.py blob 7808e1858ef0e9e2005de871604cb0a9025fe2e5. "
            "AND THE HALF THE BROAD FORM DOES NOT SAY, CHECKED HERE RATHER "
            "THAN ASSUMED: `extract_audio_in_part` iterates "
            "`self.audios[language]` and `-map`s each stream, so LANGUAGE "
            "STILL GATES WHICH STREAMS REACH STAGE 1 AT ALL. The exposure "
            "condition is therefore NEITHER the old narrow form NOR an "
            "unqualified broad one: A STREAM IS EXPOSED IF IT IS IN THE "
            "SELECTED LANGUAGE AND IS NATIVELY 32000 -- the other side's rate, "
            "the pair minimum and the clamp are all irrelevant to it. "
            "TWO NOTATION AND REACHABILITY TRAPS ON THAT CONDITION, BOTH "
            "VERIFIED HERE AND BOTH OF A KIND THAT FAILS SILENTLY. (i) THE "
            "LANGUAGE KEY IS TWO-LETTER. `self.audios` is keyed at ingest by "
            "`Lang(data['Language']).pt1` with a macro fallback (video.py "
            "blob 7808e1858ef0..., the `is_language` branch), so the key is "
            "ISO-639-1 -- `fr`, `ja` -- and NEVER the three-letter tag. An "
            "instrument that ffprobes a PRODUCED ARTEFACT reads `fre`/`jpn` and "
            "will not match this key: the join returns zero rows and never "
            "errors. Resolve exposure from the LOG's language or by stream "
            "index; never join a raw container tag against this key without "
            "normalising both sides. (ii) THE `compatible` GUARD IS INERT AND "
            "IS NOT PART OF THE CONDITION. Both stage-1 extraction loops are "
            "wrapped in `if audio[\"compatible\"]:`, which reads as a filter and "
            "is not one: across every tracked `.py` at the authority there is "
            "EXACTLY ONE write to that key, `data[\"compatible\"] = True`, and "
            "NOTHING ANYWHERE SETS IT FALSE (positive control on the same "
            "pattern shape fires). `remove_not_compatible_audio` does not touch "
            "it -- it works on video paths and a different structure, and its "
            "NAME INVITES THE OPPOSITE CONCLUSION. Found by forensic, which "
            "nearly filed it as a further narrowing before checking what writes "
            "it; recorded here so the next reader tracing this path does not "
            "re-derive it as a filter. "
            "WHAT THAT CHANGES FOR THE TWO KNOWN SUB-44100 FILES -- and the "
            "tags below are quoted as the LOG carries them, three-letter, "
            "while the code matches on two: the one "
            "whose 32000 stream is tagged `fre` (key `fr`) while the route "
            "took `ja` is "
            "STILL NOT EXPOSED, because `fre` is never extracted -- the old "
            "cell's CONCLUSION about it survives while the MECHANISM it gave "
            "does not. The one whose streams are all `jpn` including the 32000 "
            "IS EXPOSED UNDER A STRICTLY WEAKER CONDITION THAN THIS CELL USED "
            "TO STATE: it no longer needs the pair minimum to land on 32000, "
            "only its own rate. "
            "AND `grid_hz` IS NOT A SUBSTITUTE, FOR A REASON WORSE THAN ITS "
            "BEING A DIFFERENT QUANTITY: the locator PINNED 44100 until "
            "2026-09-05, so on any log written before that day the comparison "
            "grid and the output rate DIVERGE -- and they diverge precisely "
            "below 44100, which is the only region where any of this matters. "
            "On the current build they read the same variable and agree. FOUR "
            "OF THE SEVEN LOGS HERE AGREE FOR THAT REASON AND NOT BY LUCK: "
            "they are all post-change and all 44100, the one region where the "
            "two agreed even beforehand. A reader applying the agreement to an "
            "older log gets it wrong exactly where the rule exists. "
            "AND THE CHAIN IS NOW CLOSED END TO END THROUGH SHIPPED CODE, "
            "WITH THE DECISIVE STEP IN A PLACE THIS READER HAD NOT LOOKED. "
            "`codec_param` is built ONCE and fed to BOTH commands: "
            "`baseCommand.extend(codec_param)` (video.py, the EXTRACT) and "
            "`codec_param.copy()` into the normaliser. SO THE TEMPORARY FILE IS "
            "ALREADY AT THE GRID RATE BEFORE THE FILTER EVER RUNS -- the filter "
            "never resamples, it inherits. And at 32000 Hz `lowpass=f=16000` "
            "sits EXACTLY on Nyquist. Found by ci-pair; verified here against "
            "the authority's `src/video.py` "
            "blob 7808e1858ef0e9e2005de871604cb0a9025fe2e5. "
            "KIND: RUN, AND NOT MINE -- ci-pair reproduced it on REAL CORPUS "
            "AUDIO (id 108 MASTER, +11.58 dB: 86.80% of samples pinned at "
            "32000, 0.00% at 44100). Every earlier figure in this cell, "
            "including the 79.86%, came from synthetic noise. THIS READER "
            "MEASURED NONE OF IT: attributed, kind stated, unverified by me. "
            "AND THIS READER WOULD NOT SEE IT: the chain is visible "
            "only because it appears inside an ECHOED FFMPEG COMMAND ON A "
            "FAILED JOB. A job that succeeds emits neither the gain nor the "
            "rate, so on exactly the files that ship, this report can say "
            "nothing at all. `grid_hz=44100` on the locator line is NOT this "
            "rate -- it is the comparison grid, a different quantity, and "
            "reading one as the other is the wrong-slot error this register "
            "exists to make visible"),
    })
    entries.append({
        "quantity": "file_properties_nobody_named",
        "state": NO_PRODUCER,
        "address": "this register, and the spec it enumerates",
        "detail": "MEASURED INSTANCE: a produced file carried 4 chapters with "
                  "boundaries and titles identical to the master's. This report "
                  "does not mention chapters anywhere, no code path claims to "
                  "write them, and mergeVideo passes --no-chapters. A register "
                  "of empty cells enumerates what the SPEC names; it cannot "
                  "hold `no-producer` for a quantity nobody thought of. THIS "
                  "REPORT IS BLIND TO PROPERTIES OF THE FILE THAT NOBODY HAS "
                  "NAMED, and only an instrument reading the FILE rather than "
                  "the LOG can find them"})
    # LE PREMIER ECART QUE LA CHAINE INTRODUIT AU LIEU DE L'HERITER, ET LE
    # JOURNAL NE PORTE RIEN QUI PUISSE LE MONTRER.
    #
    # forensic a mesure, sur trois artefacts et sur LE FICHIER: sept pistes
    # audio en copie a 66657 paquets, la piste RECONSTRUITE a 66658. Une trame
    # AAC a 48 kHz vaut 1024/48000 = 21.3333 ms, et l'ecart de duree mesure fait
    # 22 ms a la resolution de l'etiquette. Le temoin qui en fait une decouverte
    # est negatif: deux artefacts a NEUF pistes E-AC-3 rendent UN SEUL compte
    # distinct, reconstruites comprises. Donc ce n'est pas une propriete de la
    # reparation, c'est une propriete de la reparation QUI EMET DE L'AAC.
    #
    # CE QUE CE RAPPORT PEUT EN DIRE: rien, et c'est le point. Aucune cle emise
    # ne porte une duree ni un compte de trames PAR PISTE -- verifie sur les
    # octets, la liste complete des cles du journal n'en contient aucune. Le
    # seul controle de duree est `expected_ms` contre `tolerance_ms=500`, et
    # 21,33 ms passe vingt-trois fois sous ce seuil PAR CONSTRUCTION.
    entries.append({
        "quantity": "produced_track_duration_or_frame_count",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": "RETRACTED AND REPLACED, and the retraction is the better "
                  "evidence for this cell. It was reported that a rebuilt AAC "
                  "track carries ONE FRAME MORE than its passthrough siblings "
                  "-- 66658 packets against 66657 -- and read as the pipeline "
                  "making a track longer. IT DOES NOT. The extra packet is the "
                  "AAC encoder's PRIMING frame at the head, the container "
                  "declares it as initial_padding=1024, and start_time is "
                  "0.000000 on every track, so a correct decoder discards it. "
                  "THE TRACK IS NOT LONGER: its packet count and its DURATION "
                  "tag include a frame the decoded audio does not. CONTAINER "
                  "PACKETS AND DECODED AUDIO ARE DIFFERENT QUANTITIES. The "
                  "E-AC-3 comparison that appeared to confirm the first reading "
                  "was not a control: E-AC-3 has no priming because it was "
                  "never encoded, which agrees with BOTH explanations and "
                  "discriminates neither. WHY THIS CELL EXISTS: no emitted key "
                  "carries a per-track duration or frame count, so this report "
                  "could not have told you either way -- and the only length "
                  "check is expected_ms against tolerance_ms=500, which 21.33 "
                  "ms clears by a factor of 23 by construction. A quantity "
                  "nobody emits was read wrong twice from outside the log"})
    entries.append({
        "quantity": "borrowed_placement_tag",
        "state": NO_PRODUCER,
        "address": "merge_video_chimeric.mux_repaired_file",
        "detail": "s4g requires a tag DISTINCT from VMSAM_FABRICATED; the marker "
                  "value is the single string chimeric on every marked track of "
                  "every produced file measured -- see the CORPUS row for "
                  "what that population is and is not"})
    # DEUX QUESTIONS DERRIERE UN SEUL NOM DE CHAMP, separees en deux cellules.
    # Le TAUX est emis; le DESACCORD entre le taux d'origine et le taux final ne
    # l'est que quand les deux different. Une seule cellule aurait lu `present`
    # sur chaque artefact portant un taux et n'aurait JAMAIS montre le cas pour
    # lequel `frame_rate_original` existe -- si bien que le jour ou ce champ
    # arrive enfin, il arrive dans une cellule deja marquee complete. Trouve par
    # dev-2 sur ma propre cellule, et c'est la meme collision que je passe la
    # journee a defaire, un cran au-dessus.
    entries.append({
        "quantity": "video_frame_rate_disagreement",
        "state": (PRESENT if (job.get("output_check") or {}).get("frame_rate_original")
                  else NOT_EXERCISED if (job.get("output_check") or {}).get("frame_rate")
                  else ABSENT_FORMAT),
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": "frame_rate_original= is emitted ONLY when the original and "
                  "final rates differ. It has never fired: it can only fire on "
                  "files that are VFR, which are likely to be declined instead. "
                  "A blank here is not agreement between two rates -- it is one "
                  "rate and no second to compare it with"})
    entries.append({
        "quantity": "video_frame_rate",
        # CELLULE COMBLEE PAR SON PRODUCTEUR, et on le dit par artefact plutot
        # que globalement: les artefacts anterieurs restent sans le champ, et
        # `absent de ce format` n'est pas `personne ne l'emet`.
        "state": (PRESENT if (job.get("output_check") or {}).get("frame_rate")
                  else NO_PRODUCER),
        "address": "merge_video_repair.log_assembly (output file line)",
        "detail": ("emitted on the output line as frame_rate=RATE(MODE). "
                   "Any statement about a delay landing on the video grid now "
                   "divides by a MEASURED rate rather than by an assumption -- "
                   "and the mode matters: two CFR files at 23.839 and 47.281 "
                   "take the snap branch and snap to a fabricated grid"
                   if (job.get("output_check") or {}).get("frame_rate") else
                   "no fps, frame_rate or FrameRate key on this artefact. Any "
                   "statement about a delay landing on the video grid -- "
                   "rounding, snapping, a drawn grid -- divides by a rate this "
                   "record does not carry, so it divides by an assumption")})
    entries.append({
        "quantity": "container_start_time",
        "state": NO_PRODUCER,
        "address": "merge_video_chimeric.get_stream_start_ms",
        # ABSENT N'EST PAS ZERO, et ici les deux portent le meme chiffre.
        # `head_pad_ms` est `stream_start_ms - source_start_ms` ECRETE a zero:
        # quand le flux commence AVANT le point de lecture, la fonction rend ""
        # et le champ vaut 0. Donc `head_pad_ms=0` -- 11 occurrences sur le
        # corpus -- couvre deux etats: le conteneur n'a pas de decalage, et le
        # conteneur en a un que le plan lit par-dessus. On ne peut pas inverser
        # le champ pour retrouver la quantite.
        # LA MOITIE DE CETTE CELLULE A ETE COMBLEE PAR SON PRODUCTEUR et on le
        # dit, plutot que de laisser la cellule inchangee ou de la supprimer.
        # L'AMBIGUITE que j'avais filee -- un zero pour trois etats -- est
        # resolue par `head_pad=`. LA GRANDEUR ne l'est pas: aucun champ ne
        # porte le `start_time` du conteneur en millisecondes, donc une phase
        # constante hors grille reste inexplicable a partir de ce journal.
        "detail": ("the AMBIGUITY is resolved: head_pad= now separates none, "
                   "padded and read_past, so a zero no longer covers three "
                   "states. The MAGNITUDE is still not emitted -- no start_time, "
                   "Delay or container-offset key exists -- so a constant "
                   "off-grid phase in the offsets still cannot be explained "
                   "from this record"
                   if any((job.get("audios") or {}).get(order, {}).get("head_pad")
                          for order in (job.get("audios") or {}))
                   else
                   "no start_time, Delay or container-offset key is emitted. "
                   "head_pad_ms is a one-sided derivative clamped at zero, so "
                   "head_pad_ms=0 means EITHER no container offset OR an offset "
                   "the plan reads past -- it cannot be inverted. A constant "
                   "off-grid phase in the offsets cannot be explained without it")})
    entries.append({
        "quantity": "build_identity",
        # COMBLEE PAR SON PRODUCTEUR, par artefact. Filee le premier jour, emise
        # depuis, et mon propre lecteur la jetait jusqu'a ce qu'un repli des
        # lignes inconnues l'attrape.
        "state": PRESENT if (job.get("build") or job.get("sources"))
                 else NO_PRODUCER,
        # L'ADRESSE SUIT L'ETAT. Filee contre l'en-tete de journal; le
        # producteur qui a repondu est la ligne `repair: build` de la
        # reparation, et nommer l'ancienne adresse sur une cellule comblee
        # enverrait le lecteur au mauvais endroit.
        "address": ("merge_video_repair (repair: build line)" if job.get("build")
                    else "gestionar_show.fusion (job log header)"),
        # UN ETAT ET SA RAISON QUI DIVERGENT SONT PIRES QU'UNE CELLULE VIDE:
        # un lecteur en croit une moitie. Ce detail disait encore `aucune cle de
        # build` sous un `state=present`, deux heures apres que j'aie corrige
        # exactement cela sur une autre cellule.
        "detail": ("CONTENT-DERIVED, confirmed by its producer and reproduced "
                   "independently: sha256 of the modules' own bytes read from "
                   "disk at call time, not a label passed in. So this is a "
                   "MEASUREMENT of which code ran, which is what the deploy "
                   "gate is not -- that verifies ARG VMSAM_GIT_COMMIT against "
                   "nothing. `build` covers 2 modules and `sources` the 27 .py "
                   "files the image ships; different scopes, NOT a refinement "
                   "of one another. Neither covers the interpreter, ffmpeg or "
                   "mkvtoolnix. And a digest is not a DATE, so the "
                   "`not-exercised-here` cells stay undatable"
                   if job.get("build") else
                   (f"no commit build version or image key occurs in any of the "
                    f"{corpus['logs']} logs; " if corpus else
                    "no commit build version or image key occurs on this "
                    "artefact, and no population was supplied to say how many "
                    "were checked; ")
                   + "the log records what was done and not which build did it")})
    return entries


def parse_rows(rows):
    """Relit les LIGNES en enregistrements. Le dessin part de LA, pas des objets.

    Le proprietaire a tranche: le dessin est rendu A PARTIR des lignes. Le faire
    litteralement donne une propriete verifiable et pas une intention -- RIEN NE
    PEUT APPARAITRE SUR LE SCHEMA QUI NE SOIT PAS DANS LE TEXTE. Si le HTML
    disparait, il ne manque aucune quantite; c'est le test qui prime.
    """
    records = []
    for line in rows:
        if not line or line.startswith("#"):
            continue
        kind, _, rest = line.partition(" ")
        records.append((kind, split_fields(rest)))
    return records


def _x(value, span, width):
    if not span:
        return 0.0
    return float(Decimal(str(value)) / span) * width


def _text(x, y, content, fill, size=11, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}"{extra}>{_escape(content)}</text>')


def _clip(text, limit):
    """Coupe pour la MISE EN PAGE, jamais pour la donnee.

    La valeur entiere est toujours sur sa ligne; ceci ne protege que la
    largeur du dessin. Une coupe se voit -- elle finit par une ellipse -- parce
    qu'un texte tronque silencieusement se lit comme un texte complet, ce qui
    est le meme defaut que `source=master` tronque d'une valeur de quatre mots.
    """
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "\u2026"


def short_clock(ms):
    """m:ss sous l'heure, h:mm:ss au-dela. Pour les etiquettes serrees."""
    if ms is None:
        return "?"
    total = int(Decimal(str(ms)))
    seconds, _ = divmod(abs(total), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sign = "-" if total < 0 else ""
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{seconds:02d}"
    return f"{sign}{minutes}:{seconds:02d}"


def _geometry_key(regions):
    """Deux pistes ont la MEME geometrie quand leurs decalages par region sont
    identiques chiffre pour chiffre. C'est ce qui distingue une mesure d'un
    emprunt sans lire une etiquette."""
    return tuple((f.get("master_start_ms"), f.get("offset_s") or f.get("offset_ms_state"))
                 for f in regions)


def render_svg(records):
    """LA FIGURE QUE LE PROPRIETAIRE A DESSINEE, sur ses propres octets.

    Fond sombre, aucune grille, aucune legende, aucun chrome. UN axe fin en bas
    portant `0` et le total. Des REGLES SAUMON pleine hauteur a chaque coupe --
    la marque dominante: l'oeil lit LES COUPES d'abord. Des BARRES BLEU PALE,
    une par segment apparie, a leur vraie place et largeur, ET DECALEES VERS LE
    BAS EN ESCALIER. Des RECTANGLES AMBRE POINTILLES la ou rien ne correspond,
    avec `aucune correspondance` dessous.

    L'escalier fait un vrai travail: des segments consecutifs a des hauteurs
    differentes se suivent COMME UNE FORME. Ma premiere version mettait chaque
    region sur une seule ligne par piste -- exact, et cela se lisait comme un
    TABLEAU. Le proprietaire a rejete la figure et pas les nombres.

    VIRGULE DECIMALE. Il lit ces figures et sa locale est francaise.

    -- LE PROBLEME DE CONCEPTION: SES DEUX IMAGES DESSINENT UNE PISTE, L'ARTEFACT
       EN A SEPT --

    Une copie litterale ne survit pas au compte de pistes. Et sept escaliers
    seraient pires que lourds: sur a9111493 cinq pistes portent les decalages de
    la piste 1 AU CHIFFRE PRES, donc la figure dessinerait six fois la meme forme
    et AFFIRMERAIT SEPT MESURES LA OU DEUX ONT ETE PRISES.

    Donc UN ESCALIER PAR GEOMETRIE DISTINCTE, et les pistes qui empruntent
    s'accrochent a l'escalier qu'elles empruntent, dans une marque differente et
    nommee. L'exigence qui prime -- UN DECALAGE EMPRUNTE ET UN DECALAGE MESURE
    NE DOIVENT PAS SE RESSEMBLER SUR L'AXE -- devient alors structurelle et non
    decorative: on ne peut pas dessiner un emprunt comme une mesure parce qu'un
    emprunt N'A PAS D'ESCALIER A LUI.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    if not plan or not plan.get("master_end_ms"):
        return ('<p class="note">No plan geometry in this artefact, so there is '
                'no master timeline to draw. The rows above carry everything '
                'this format emitted.</p>')
    span = Decimal(plan["master_end_ms"])

    tracks, regions, lost, borrows, steps, refused = [], {}, {}, {}, {}, []
    for kind, fields in records:
        if kind == "TRACK" and fields.get("kind") == "audio":
            tracks.append(fields)
        elif kind == "REGION":
            regions.setdefault(fields["track"], []).append(fields)
        elif kind == "LOST":
            lost.setdefault(fields["track"], []).append(fields)
        elif kind == "BORROW":
            borrows[fields["track"]] = fields
        elif kind == "STEP":
            steps.setdefault(fields["track"], []).append(fields)
        elif kind == "REFUSED":
            # PAR FICHIER et non par piste: un segment refuse l'est par le PLAN,
            # avant que les pistes existent.
            refused.append(fields)

    # Regrouper par geometrie. L'ordre de premiere apparition tient lieu d'ordre.
    groups = []
    for track in tracks:
        key = _geometry_key(regions.get(track["track"], []))
        for group in groups:
            if group["key"] == key:
                group["tracks"].append(track)
                break
        else:
            groups.append({"key": key, "tracks": [track]})

    ink, paper = "#c6ccd4", "#0f1115"
    salmon, blue, bar_blue, amber, faint = ("#e8836b", "#8fbcd4", "#a3cadd",
                                            "#c9a227", "#6f7681")
    left, right, width = 44, 44, 1080
    plot = width - left - right
    top = 48

    def x_of(value):
        return left + float(Decimal(str(value)) / span) * plot

    # --- L'INFO DU PLAN, demandee explicitement. Discrete, jamais du chrome. --
    # `kind=piecewise_constant` avec `quantum_ms=129` est exactement ce qui
    # explique POURQUOI l'escalier a la forme qu'il a: un plan constant par
    # morceaux descend par paliers, un plan constant serait une seule marche.
    # Rien de tout cela n'atteignait le dessin.
    body = []
    plan_line = (f"plan: {plan.get('kind')} \u00b7 measured on "
                 f"{plan.get('language')} \u00b7 quantum {plan.get('quantum_ms')} ms \u00b7 "
                 f"{plan.get('pieces')} pieces \u00b7 master duration "
                 f"{plain(plan.get('master_end')) or '?'}")
    if plan.get("decided_by"):
        plan_line += f" \u00b7 decided by {plan['decided_by']}"
    # `NON EMISE` UNIQUEMENT QUAND LE PRODUCTEUR N'A RIEN DIT.
    #
    # Je viens de deposer ce defaut exact contre l'emetteur de dev-2 et je le
    # tenais dans mon propre en-tete: le test portait sur la PRESENCE d'un etat,
    # donc `not-defined-here` et `no-producer` -- deux etats que je venais de
    # separer -- retombaient sur la meme phrase. Filer un defaut ne le retire
    # pas de chez soi, et un etat n'existe que si quelque chose le LIT.
    #
    # Quand la marge n'est pas definie, `tranche par : fidelity_gate` dit deja
    # ce qui a decide; ecrire `NON EMISE` a cote qualifierait de sans-marge une
    # decision prise avec 0,3637 de separation.
    if plan.get("speed_margin_state") == NO_PRODUCER:
        # s4f: la marge par laquelle la transformation gagnante l'a emporte.
        # On nomme le sujet et on marque la valeur manquante -- on ne laisse pas
        # un blanc se lire comme une marge nulle, qui serait deux hypotheses a
        # egalite.
        plan_line += " \u00b7 speed margin: NOT EMITTED"
    elif plan.get("speed_margin_state") == NOT_DEFINED:
        plan_line += " \u00b7 flatness margin: NOT APPLICABLE here"
    elif plan.get("speed_margin"):
        plan_line += f" \u00b7 marge {plan['speed_margin']}"
    if plan.get("fidelity_margin"):
        plan_line += f" \u00b7 fidelity separation {plan['fidelity_margin']}"
    body.append(_text(left, top - 18, plan_line, faint, 10))
    y = top + 4

    for group in groups:
        lead = next((t for t in group["tracks"]
                     if not (plain(t.get("offset")) or "").startswith("BORROWED")),
                    group["tracks"][0])
        number = lead["track"]
        mine = regions.get(number, [])
        candidates = [f for f in mine if f.get("kind") == "CANDIDATE"]
        filled = [f for f in mine if f.get("kind") in ("MASTER", "SILENCE", "MASTER?")]
        others = [t for t in group["tracks"] if t["track"] != number]

        panel_top = y + 16
        step = 30 if len(candidates) < 8 else max(14, int(200 / max(1, len(candidates))))
        stair = max(1, len(candidates)) * step
        panel_bottom = panel_top + stair + 10
        master_bar_y = panel_bottom + 16
        panel_end = master_bar_y + 16 + (12 * len(others))

        # --- l'en-tete: le plan, la piste, et si elle a ete acceleree ---------
        # UNE CHAINE VIDE NE COMMENCE PAS PAR `none`. Un champ `speed=` ABSENT
        # rendait donc `resampled=True` et dessinait `ACCELEREE x` AVEC UN RATIO
        # VIDE -- pendant que la narration du MEME document disait "aucune piste
        # n'a recu de correction de rythme". LE DESSIN AFFIRMAIT, LA PROSE
        # NIAIT, sur les memes octets.
        #
        # C'est mon propre defaut de la boite ambre, un champ plus loin, et dans
        # la direction que j'avais appelee "en pire": une affirmation portee par
        # une marque PRESENTE. Et il etait ARME PAR UNE ABSENCE, la classe que
        # ce module entier existe pour rendre impossible.
        #
        # Trouve par l'architecte parce que SA fixture omettait le champ.
        # `merge_video_repair.py:959` ecrit toujours `speed=`, donc ce n'etait
        # pas atteignable en production -- LATENT, pas vivant. Il le devient au
        # premier producteur qui omet le champ, ce qui est exactement ce que
        # `absent-from-this-format` existe pour dire.
        speed = plain(lead.get("speed"))
        if speed is None:
            resampled, speed = None, ""
        else:
            resampled = not speed.lower().startswith("none")
        head = (f"track {number} · {lead.get('lang')} · "
                f"{'measured' if not (plain(lead.get('offset')) or '').startswith('BORROWED') else 'BORROWED'}")
        if resampled:
            # s4f: on dit QUE la piste a ete reechantillonnee, et on marque la
            # confiance MANQUANTE. `speed_ratio_applied` est emis,
            # `speed_margin` n'a AUCUN PRODUCTEUR: on n'omet pas le fait parce
            # que la moitie manque, et on n'imprime pas la moitie manquante
            # comme si c'etait une valeur.
            #
            # LE RATIO SEUL DANS L'EN-TETE, JAMAIS LA PROSE. dev-2 s'apprete a
            # emettre un refus en toutes lettres DANS `speed=none(...)` -- la
            # convention de dev-1 est l'inverse de la sienne et rien ne separe
            # 0.999001 de 1.000999. Cette phrase atterrirait ICI, sur la seule
            # ligne du dessin qui n'a aucune marge. On coupe donc AVANT qu'elle
            # arrive: l'en-tete ne prend que ce qui ressemble a un nombre, et le
            # texte entier reste sur la ligne TRACK, qui le porte deja en
            # entier. Reparer une mise en page apres l'avoir vue casser aurait
            # voulu dire la voir casser chez le proprietaire.
            ratio = speed.split("(")[0].strip()
            head += (f" \u00b7 RESAMPLED \u00d7{_clip(ratio, 12)}"
                     f" \u00b7 winning margin: NOT EMITTED")
        elif resampled is None:
            # NI acceleree NI non-acceleree: le champ n'est pas la. On ne
            # devine pas dans un sens plutot que dans l'autre.
            head += " \u00b7 rate: FIELD ABSENT from this artefact"
        elif "(" in speed and not speed.lower().startswith("none("):
            head += " \u00b7 speed: see the TRACK row"
        else:
            # `not resampled` ETAIT UNE REPONSE FAUSSE ET AFFIRMATIVE, et c'est
            # la collision que `MEASURING.MD` enregistre deja: `speed=none(no
            # rate proposed)` sur 112 fichiers se lit comme 112 fichiers SANS
            # PROBLEME DE RYTHME et signifie 112 fichiers QUE PERSONNE N'A
            # INTERROGES -- aucun producteur de verdict de vitesse n'existe dans
            # `src/`. J'ecrivais donc un SILENCE comme un VERDICT NEGATIF, dans
            # le seul document qu'il lit.
            #
            # C'est le meme defaut que la boite ambre, un champ plus loin, et en
            # pire: l'ambre AFFIRMAIT par une marque absente, ceci affirme par
            # une marque presente. Meme question de `MEASURING.MD`: la valeur
            # avec laquelle la sentinelle entre en collision est-elle une valeur
            # que cette mesure peut legitimement produire? Oui -- une piste peut
            # reellement n'avoir aucun rythme a corriger -- donc la sentinelle
            # est un defaut et pas une commodite.
            head += " \u00b7 rate: NOT MEASURED"
        if lead.get("verify"):
            head += f" · verify {_clip(plain(lead['verify']), 24)}"
        body.append(_text(left, y + 10, _clip(head, 132), faint, 10))

        # --- les regions sans correspondance: ambre, pointille ---------------
        # LA BOITE AMBRE POINTILLEE MARQUE CE QUI EST REFUSE DU CANDIDAT.
        # Reglage du proprietaire, et je dessinais le mauvais ensemble: je la
        # nourrissais des regions REMPLIES DEPUIS LE MAITRE. Les deux marques
        # disent des choses differentes et il en faut deux -- la boite dit
        # `ce materiau du candidat a ete REFUSE`, la barre dit `voici ce qui est
        # VENU DU MAITRE`. Une region peut etre l'un, l'autre, les deux ou
        # aucun, et les confondre rendait `refuse` et `rempli` indistinguables:
        # exactement le defaut que le producteur nomme dans son propre
        # commentaire, "le plan n'avait pas de candidat ici" contre "le plan en
        # avait un et on l'a jete".
        #
        # ET LA HAUTEUR RESTE BASSE. Pleine hauteur, l'ambre pesait plus que les
        # regles et l'oeil tombait dessus en premier; ses images donnent la
        # figure AUX REGLES. Meme position, meme etendue, une bande.
        band = 18
        band_y = panel_top + max(0.0, (stair - band) / 2)
        for fields in refused:
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            body.append(f'<rect x="{x0:.1f}" y="{panel_top:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="{stair:.1f}" '
                        f'fill="{amber}" fill-opacity="0.05" stroke="none"/>')
            body.append(f'<rect x="{x0:.1f}" y="{band_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="{band}" '
                        f'fill="none" stroke="{amber}" stroke-width="1" '
                        f'stroke-dasharray="4 3"/>')
            if x1 - x0 > 108:
                body.append(_text((x0 + x1) / 2, band_y + 13,
                                  "no correspondence", amber, 9, "middle"))

        # --- les coupes: regles saumon pleine hauteur, la marque dominante ----
        # Une coupe retire du materiau du CANDIDAT; sur la timeline du MAITRE
        # c'est un EVENEMENT SANS LARGEUR. On trace donc UNE regle a son point
        # d'insertion et on ecrit son ampleur, plutot que de dessiner une
        # longueur sur un axe qui n'est pas le sien.
        for index, fields in enumerate(lost.get(number, [])):
            where = fields.get("where")
            if where == "head" and candidates:
                boundary = Decimal(candidates[0]["master_start_ms"])
            elif where == "tail" and candidates:
                boundary = Decimal(candidates[-1]["master_end_ms"])
            else:
                inner = [f for f in lost.get(number, []) if f.get("where") == "interior"]
                position = inner.index(fields) if fields in inner else 0
                boundary = (Decimal(candidates[position]["master_end_ms"])
                            if position < len(candidates) else span)
            x = x_of(boundary)
            body.append(f'<line x1="{x:.1f}" y1="{panel_top - 4:.1f}" x2="{x:.1f}" '
                        f'y2="{panel_bottom:.1f}" stroke="{salmon}" stroke-width="1.6"/>')
            # UNE VALEUR ET SON ABSENCE NE DOIVENT PAS IMPRIMER LE MEME JETON --
            # et au-dessus d'une regle le jeton de l'absence etait DU BLANC. Les
            # coupes de tete et de queue n'ont pas de pas, rien ne les encadre,
            # donc elles n'ont a juste titre pas de nombre; mais une regle sans
            # rien au-dessus se lit comme une etiquette MANQUANTE et non comme un
            # silence voulu. C'est la regle de la campagne arrivant dans un
            # dessin, et il a fallu un rendu pour la voir.
            if not any(_trim(Decimal(f["at_master_ms"])) == _trim(boundary)
                       for f in steps.get(number, [])):
                body.append(f'<circle cx="{x:.1f}" cy="{panel_top - 14:.1f}" r="2" '
                            f'fill="none" stroke="{salmon}" stroke-width="1">'
                            f'<title>no step here: head or tail cut, '
                            f'nothing flanks it</title></circle>')

        # LE NOMBRE SAUMON EST LE PAS DE DECALAGE A TRAVERS LA COUPE, et pas la
        # duree jetee. Mesure sur les deux images du proprietaire: quatre de ses
        # cinq nombres saumon sont exactement la difference entre deux
        # etiquettes bleues consecutives, a la decimale imprimee. Les coupes de
        # tete et de queue n'ont pas de pas -- rien ne les encadre -- et restent
        # donc des regles NUES, ce qui est aussi ce qu'il a dessine: huit
        # regles, trois nombres.
        for fields in steps.get(number, []):
            x = x_of(fields["at_master_ms"])
            anchor, at = "middle", x
            if x < left + 44:
                anchor, at = "start", left
            elif x > left + plot - 44:
                anchor, at = "end", left + plot
            body.append(_text(at, panel_top - 10, f"{fields['step_s']} s",
                              salmon, 11, anchor))

        # --- l'escalier: une barre par segment apparie, decalee vers le bas ---
        for index, fields in enumerate(candidates):
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            bar_y = panel_top + index * step + 4
            body.append(f'<rect x="{x0:.1f}" y="{bar_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="12" '
                        f'fill="{bar_blue}"/>')
            offset = fields.get("offset_s")
            if not offset:
                body.append(_text(x1 + 6, bar_y + 10, "decalage non emis", salmon, 10))
                continue
            # `s` sur la PREMIERE etiquette d'une serie, puis abandonnee: c'est
            # ce qu'il a dessine.
            label = f"{offset} s" if index == 0 else offset
            span_px = x1 - x0
            estimate = len(label) * 6.4
            if span_px > estimate + 16:
                body.append(_text((x0 + x1) / 2, bar_y - 3, label, blue, 11, "middle"))
            elif x1 + 8 + estimate < left + plot:
                body.append(_text(x1 + 8, bar_y + 10, label, blue, 11))
            else:
                body.append(_text(x0 - 8, bar_y + 10, label, blue, 11, "end"))

        # --- LA BARRE EN PLUS: ce qui a ete RAJOUTE DU MAITRE, et d'ou --------
        # Nommee par le proprietaire comme ce qu'il aime. Et la langue REELLE de
        # remplissage est la plus revelatrice de l'artefact: six pistes disant
        # `master/ja` et une disant `master/fr` est le remplissage inter-langue
        # de SPEC_ZONE_A s4c qui se produit sous les yeux du lecteur.
        # A QUEL MOMENT EST LE RAJOUT ET QUAND IL FINIT -- demande du
        # proprietaire, et l'etiquette appartient A CETTE BARRE. Elle etait sur
        # la boite ambre, parce que je croyais l'ambre egale au remplissage
        # maitre; le proprietaire a tranche que l'ambre est le REFUS. `Le
        # rajout` est ce qui vient du maitre, donc son horodatage suit la barre
        # du maitre et pas la boite.
        #
        # SUR la barre et non laissee a l'axe: l'axe repond "ou en est-on dans
        # le programme", la barre repond "quand commence et finit CE rajout".
        # Sinon le lecteur doit redescendre vers l'axe et deviner quel cran lui
        # appartient.
        sources, filled_total, placed_marks = [], Decimal("0"), []
        for fields in filled:
            duration = (Decimal(fields["master_end_ms"])
                        - Decimal(fields["master_start_ms"]))
            filled_total += duration
            x0, x1 = x_of(fields["master_start_ms"]), x_of(fields["master_end_ms"])
            body.append(f'<rect x="{x0:.1f}" y="{master_bar_y:.1f}" '
                        f'width="{max(2.0, x1 - x0):.1f}" height="7" fill="{amber}" '
                        f'fill-opacity="0.75"/>')
            # LA MARQUE SUR LA FIGURE: le proprietaire lit le dessin, et un
            # remplissage large de la borne de recherche exacte doit SE VOIR.
            if "UNREFINED SEARCH BOUND" in (plain(fields.get("fill_width")) or ""):
                body.append(f'<rect x="{x0:.1f}" y="{master_bar_y - 2:.1f}" '
                            f'width="{max(2.0, x1 - x0):.1f}" height="11" '
                            f'fill="none" stroke="{salmon}" stroke-width="1.4"/>')
                body.append(_text((x0 + x1) / 2, master_bar_y + 20,
                                  "search bound \u2014 bracket never narrowed",
                                  salmon, 9, "middle"))
            timing = (f"{short_clock(fields['master_start_ms'])} \u2192 "
                      f"{short_clock(fields['master_end_ms'])}  "
                      f"+{short_clock(duration)}")
            estimate = len(timing) * 5.2
            middle = (x0 + x1) / 2
            if not placed_marks or middle - estimate / 2 > placed_marks[-1] + 8:
                placed_marks.append(middle + estimate / 2)
                body.append(_text(middle, master_bar_y - 4, timing, amber, 9, "middle"))
            elif x1 - x0 > 34:
                body.append(_text(middle, master_bar_y - 4,
                                  f"+{short_clock(duration)}", amber, 9, "middle"))
            # UN ETAT N'EST PAS UN NOM DE SOURCE. Je rendais `source_state` a la
            # place du nom quand le champ manquait, et la legende disait
            # `added from the master: absent-from-this-format` -- un jeton de
            # registre pose la ou un humain attend une piste. Trouve par
            # l'architecte dans SA sortie, sur un artefact dont le format ne
            # porte pas encore `from=`.
            #
            # C'est exactement le defaut que ce module poursuit, arrive DANS LE
            # CANAL HUMAIN: mes etats sont faits pour les LIGNES, ou un lecteur
            # a la legende sous les yeux. Dans la figure ils n'ont aucun sens et
            # se lisent comme une reponse.
            name = plain(fields.get("source"))
            if not name:
                name = ("source not recorded in this format"
                        if fields.get("source_state") == ABSENT_FORMAT
                        else "source not emitted")
            if name not in sources:
                sources.append(name)
        # UNE LEGENDE QUI NOMME UNE SOURCE SANS SA QUANTITE invite le lecteur a
        # estimer la quantite sur la largeur de la barre. On la donne.
        body.append(_text(left, master_bar_y + 20,
                          "added from the master: "
                          + (" \u00b7 ".join(sources) or "nothing")
                          + f" \u00b7 {len(filled)} region(s)"
                          + f" \u00b7 {short_clock(filled_total)}",
                          amber, 10))

        # --- les pistes qui EMPRUNTENT cette geometrie ------------------------
        for index, other in enumerate(others):
            strip = master_bar_y + 28 + index * 12
            borrow = borrows.get(other["track"], {})
            # LE POIDS D'UNE BANDE SUIT SON ACCORD, et c'est une correction
            # avant panne plutot qu'apres. Cinq bandes IDENTIQUES se lisent
            # comme UNE annotation repetee et restent subordonnees a l'escalier
            # -- la subordination est portee par leur MEMETE et non par leur
            # place. Donnez-en une qui differe et elle saute hors du groupe et
            # se met a concurrencer l'escalier.
            #
            # Alors qu'elle le fasse EXPRES: une bande dont le controle dit
            # `identical at N of N` reste discrete, et TOUTE AUTRE prend le
            # poids. Une emprunteuse divergente qui attire l'oeil est le
            # comportement voulu, pas une regression de mise en page.
            # ETAT: CONSTRUIT, JAMAIS EXERCE. Toutes les emprunteuses
            # observees a ce jour s'accordent avec leur reference, donc la
            # branche saumon ci-dessous n'a jamais ete dessinee. Elle est vivante
            # et le corpus n'en produit pas l'entree -- `not-exercised-here`,
            # dit ici plutot que dans un message parce qu'un message ne survit
            # pas a une session.
            check = plain(borrow.get("check")) or ""
            agrees = bool(re.match(r"offsets identical at (\d+) of \1 regions$",
                                   check))
            tone = faint if agrees else salmon
            body.append(f'<rect x="{left:.1f}" y="{strip:.1f}" width="{plot:.1f}" '
                        f'height="6" fill="none" stroke="{tone}" '
                        f'stroke-width="{1 if agrees else 1.4}" '
                        f'stroke-dasharray="{"3 3" if agrees else "none"}"/>')
            detail = (f"track {other['track']} · {other.get('lang')} · EMPRUNTE "
                      f"cette geometrie · verify "
                      f"{plain(other.get('verify')) or '?'}")
            if check:
                detail += f" · {check}"
            body.append(_text(left + 6, strip + 5.5, _clip(detail, 150), tone, 9))
        y = panel_end + 18

    # --- l'axe: un seul, fin, en bas. `0` et le total. Aucune graduation. -----
    axis = y + 6
    height = axis + 38
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;height:auto" role="img" '
           f'aria-label="master timeline, one staircase per distinct geometry">',
           '<title>merge_plan</title>',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="{paper}"/>']
    out.extend(body)
    out.append(f'<line x1="{left}" y1="{axis:.1f}" x2="{left + plot}" '
               f'y2="{axis:.1f}" stroke="{faint}" stroke-width="1"/>')

    # LES HORODATAGES, demandes par le proprietaire: "sur la figure il faut les
    # timing aussi". Ils vont SUR L'AXE et non au-dessus des regles, et la
    # raison n'est pas une preference.
    #
    # UNE COUPE POSE DEUX QUESTIONS A UN LECTEUR ET ELLES VEULENT DES ENDROITS
    # DIFFERENTS. `-12,1 s` en saumon au-dessus de la regle repond COMBIEN;
    # `0:02:40` en gris sur l'axe repond QUAND. Les empiler au-dessus de la
    # regle ferait porter DEUX GRANDEURS A UNE SEULE COULEUR, qui est
    # exactement le defaut que ce rapport existe pour empecher.
    #
    # Cela contredit le "aucune graduation intermediaire" de ses images. Il a
    # amende sa propre specification en le demandant, et entre ses images et
    # son instruction, L'INSTRUCTION EST LA PLUS RECENTE.
    #
    # Aucun champ nouveau: `master_start` et `master_end` sont deja sur chaque
    # ligne REGION. C'est la deuxieme fois qu'un des ajouts demandes se revele
    # etre un manque DU DESSIN et pas un manque D'EMISSION.
    boundaries = set()
    for group in groups:
        for fields in regions.get(group["tracks"][0]["track"], []):
            boundaries.add(Decimal(fields["master_start_ms"]))
            boundaries.add(Decimal(fields["master_end_ms"]))
    placed = []
    for position in sorted(boundaries):
        x = x_of(position)
        out.append(f'<line x1="{x:.1f}" y1="{axis:.1f}" x2="{x:.1f}" '
                   f'y2="{axis + 5:.1f}" stroke="{faint}" stroke-width="1"/>')
        label = clock(position)
        if label.endswith(".000"):
            label = label[:-4]
        # ON NE LAISSE PAS DEUX HORODATAGES SE CHEVAUCHER: un nombre illisible
        # n'est pas un nombre publie. Un cran sans etiquette garde sa position.
        half = len(label) * 3.2
        if placed and x - half < placed[-1] + 4:
            continue
        placed.append(x + half)
        out.append(_text(x, axis + 17, label, ink, 10, "middle"))
    out.append(_text(left, axis + 30, "master timeline", faint, 9))
    out.append('</svg>')
    return "\n".join(out)


def render_narrative(records):
    """LES MEMES FAITS EN PHRASES.

    s4g: le diagramme et les phrases sont la meme information deux fois, VOULU.
    Un lecteur doit voir ce qui a ete fait au fichier SANS lire une table.

    EN ANGLAIS: le proprietaire a tranche pour les documents produits. Ce qui
    distingue cette moitie des LIGNES n'est donc plus la langue mais LE ROLE --
    le dessin et ces phrases sont le canal HUMAIN, les lignes sont l'assurance
    contre la perte et le canal des AGENTS. Meme document, meme langue, deux
    vues des memes octets, et chacune dit laquelle elle est.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    source = next((f for k, f in records if k == "SOURCE"), {})
    identity = next((f for k, f in records if k == "IDENTITY"), {})
    said = []

    declined = next((f for k, f in records if k == "DECLINED"), None)
    if declined:
        said.append(
            f"<p><b>THIS FILE WAS NOT PRODUCED.</b> The repair was "
            f"<b>REFUSED</b>: {_escape(plain(declined.get('reason', '?')))}. "
            f"Everything below describes what the plan WOULD have done, not the "
            f"contents of an artefact \u2014 there is no artefact.</p>")
    if plan:
        said.append(
            f"<p>Artefact <b>{_escape(plain(source.get('artefact', '?')))}</b> "
            f"was rebuilt on the master's timeline, which runs to "
            f"<b>{_escape(plain(plan.get('master_end', '?')))}</b>. The "
            f"measurement called the relationship "
            f"<b>{_escape(plan.get('kind', '?'))}</b> and took it on the "
            f"<b>{_escape(plan.get('language', '?'))}</b> track, at a quantum of "
            f"{_escape(plan.get('quantum_ms', '?'))} ms.</p>")
    if identity.get("master"):
        said.append(f"<p>The master is named in this log, as "
                    f"<b>{_escape(identity['master'])}</b>. That is what makes a "
                    f"deficit measured in the produced file <i>attributable</i> "
                    f"to the merge or to the master, rather than merely "
                    f"observable.</p>")
    else:
        said.append("<p><b>The master is not named in this artefact's format.</b> "
                    "A deficit measured in the produced file can therefore be "
                    "attributed to nobody: content correspondence to its source "
                    "is unmeasurable, and that is a property of the record and "
                    "not of the file.</p>")

    # LES DEUX PHRASES QUE LE PROPRIETAIRE DOIT LIRE, AVANT LES PISTES, PARCE
    # QU'ELLES PORTENT SUR CHAQUE REGION REMPLIE DU FICHIER.
    # LA PORTEE VIENT DES LIGNES, PAS D'UN ARGUMENT.
    #
    # Trouve en rendant AU DEFAUT DE PRODUCTION, ce que ce module ne faisait
    # jamais: un rapport de production dit `not-supplied-to-this-render` et
    # `toute affirmation a l'echelle du corpus ci-dessous est sans
    # denominateur` -- PUIS IMPRIMAIT QUATRE PARAGRAPHES a l'echelle du corpus,
    # avec des chiffres durs (23 journaux, 42 remplissages) mesures sur MON
    # corpus de laboratoire, dans un rapport qui porte sur UN fichier.
    #
    # "dans ce corpus" dans un rapport de fusion unique nomme une population que
    # le lecteur n'a pas. C'est la meme classe que tout le reste de la nuit: une
    # propriete de mon instrument imprimee la ou elle ne tient pas.
    corpus_supplied = not any(k == "CORPUS" and plain(f.get("state")) == NOT_SUPPLIED
                              for k, f in records)
    corpus_said = []
    bound_only = [f for k, f in records if k == "REGION"
                  and "UNREFINED SEARCH BOUND" in (plain(f.get("fill_width")) or "")]
    if bound_only:
        seen = {(f["master_start_ms"], f["master_end_ms"]) for f in bound_only}
        said.append(
            f"<p><b>{len(seen)} filled region(s) here are exactly 100 s wide "
            f"\u2014 the locator's UNREFINED SEARCH BOUND.</b> Its coarse pass "
            f"steps 40 s through a 60 s window, so a change point it never "
            f"managed to narrow stays bracketed by exactly 100 s \u2014 AND "
            f"THAT BRACKET IS WHAT GETS FILLED WITH MASTER AUDIO. A width "
            f"landing on the search constant to the millisecond is not a "
            f"measured hole; it is the measurement's uncertainty, delivered as "
            f"sound. EVIDENCE, NOT PROOF: a genuine 100 s hole would look the "
            f"same. But real boundaries in these files are numbers like 1442240 "
            f"and 152056, and never round.</p>")
        # LE TEMOIN NEGATIF, ET IL EST DANS LES MEMES FICHIERS.
        #
        # forensic a demande mon compte INTERIEUR seul. En le separant, la
        # derniere region de chaque plan est apparue comme un temoin que je
        # n'avais pas construit: si 100 s etait une largeur de contenu
        # plausible, elle se verrait AUSSI en fin de plan. Corpus du
        # 2026-09-05, 23 journaux, un plan par travail:
        #
        #   interieur  42 regions   30 a 100000   9 a 12000   42/42 RONDES
        #   tete       17 regions    9 a 100000   0 a 12000   11/17 rondes
        #   queue      17 regions    0 a 100000   0 a 12000    1/17 ronde
        #
        # Les queues mesurent 31, 38, 1989, 152056, 287947 ms. C'est la forme
        # d'un contenu mesure. Les interieures sont rondes A LA SECONDE SANS
        # EXCEPTION. Un seul jeu de fichiers, deux distributions.
        # LA PROSE CITE LA LIGNE `CORPUS`, ELLE NE RECITE PLUS DE MEMOIRE.
        #
        # Ces phrases portaient "23 journaux, 42 interieures, 39, 17". Vrai
        # quand tape; 28 / 55 / 55 / 20 une heure plus tard. Le nombre vit
        # desormais dans `fill_census`, produit par `measure_corpus`, et la
        # narration LE LIT -- donc il ne peut plus diverger de la mesure, et il
        # est dans le texte grepable par construction.
        census = next((plain(f.get("fill_census")) for k, f in records
                       if k == "CORPUS" and f.get("fill_census")), None)
        corpus_said.append(
            "<p><b>The same files carry their own control.</b> The census is on "
            "the CORPUS row above and this sentence reads it rather than "
            "restating it: <code>" + (census or "not supplied") + "</code>. "
            "INTERIOR fills sit on the locator's refine grid; fills at the END "
            "of a plan do not, and their widths look like measured content. If "
            "100 s were a plausible width for real missing content, it would "
            "appear at the end of a plan too. It never does. That is the same "
            "files disagreeing with themselves depending on whether the locator "
            "had a bracket to close.</p>")
        # ET LA FORME NETTE, TROUVEE EN NOMMANT LES TROIS RESTANTES.
        #
        # J'allais ecrire "39 sur 42 sont une des deux constantes, plus trois
        # autres". LES TROIS AUTRES SONT 16000, 16000, 20000 -- le plancher
        # d'affinage 12000 plus UN et DEUX pas de 4000. Il n'y a pas de reste.
        #
        #   interieur  42 regions,  QUATRE largeurs distinctes:
        #        12000 x9    plancher + 0 pas
        #        16000 x2    plancher + 1 pas
        #        20000 x1    plancher + 2 pas
        #       100000 x30   borne de recherche, jamais retrecie
        #     42/42 sont un multiple EXACT du pas d'affinage de 4 s
        #   queue      17 regions, DIX-SEPT largeurs distinctes, 0/17 multiples
        #
        # C'est le meme test que le controle ci-dessus, en plus tranchant: la
        # question n'est plus "cette largeur est-elle ronde" mais "cette
        # largeur est-elle sur la GRILLE DE RECHERCHE". A l'interieur, toujours.
        # En queue, jamais.
        corpus_said.append(
            "<p><b>Interior gap widths in this corpus are not measurements of "
            "the media. They are positions on the locator's search grid.</b> "
            "Every interior master fill is an exact multiple of the 4 s refine "
            "step and they take only a handful of distinct values, each one a "
            "locator constant \u2014 the 100 s un-narrowed search bound, and the "
            "12 s refine floor plus zero, one or two refine steps. The fills at "
            "the end of a plan take a DIFFERENT width every time and NOT ONE is "
            "a multiple of the refine step. The counts are on the CORPUS row; "
            "this sentence does not carry its own copy of them. Nothing in the "
            "pipeline quantises a master fill to 4 s \u2014 if it did, the ends "
            "of plans would be quantised too.</p>"
            "<p><b>And most of that interior count could not have gone the "
            "other way.</b> The 100 s search bound is 25 x 4 s exactly, so "
            "every fill that IS the bound sits on the refine grid by "
            "construction and tests nothing. Those are counted out on the "
            "CORPUS row, and the remainder — the widths the locator "
            "actually narrowed — is the population this claim rests on. "
            "It is a much smaller number and it is the honest one; a fill at "
            "the bound belongs to the OTHER claim, the one the emitted "
            "<code>bound_only</code> field now settles directly. The "
            "end-of-plan column is unaffected: not one of those fills is at "
            "the bound, so every one of them could have landed on the grid and "
            "none did. Raised by forensic against the earlier wording, which "
            "counted all of them together.</p>")
        # LA BORNE, TROUVEE PAR forensic ET VERIFIEE PAR MOI SUR LES OCTETS.
        #
        # J'ai ecrit "QUATRE valeurs, il n'y a pas de reste" -- vrai de MA
        # population et faux comme enonce. Ma population exclut les REFUS, donc
        # elle ne pouvait pas voir ou le motif s'arrete. forensic garde les
        # refus et y a trouve deux largeurs hors des quatre. Mesure faite
        # ensuite sur mes propres octets, pas reprise de son message:
        #
        #   23 fichiers .error sur le disque, 20 rejetes par structure,
        #   3 portent un plan:
        #       100000 x2
        #       900000 x1   \  toutes deux dans UN SEUL enregistrement,
        #       932000 x1   /   ad5ed63a2a6ff055.error
        #
        # LES DEUX SONT SUR LA GRILLE DE 4 s (k=222 et k=230). Donc le test de
        # QUANTIFICATION les admet -- 46/46 interieures sur les deux
        # populations -- et le test des QUATRE VALEURS les exclut. Deux
        # affirmations emboitees, et un lecteur doit savoir laquelle il tient.
        #
        # C'est la troisieme fois ce soir que j'ecris une propriete de mon
        # instrument comme une propriete du monde. Celle-ci est corrigee avant
        # d'avoir voyage.
        corpus_said.append(
            "<p><b>And that four-value list is a property of files that "
            "SHIPPED.</b> Refusal records are not in this corpus. In the "
            "refused records on disk that carry a plan, interior fills of 900 s "
            "and 932 s were measured — 15 minutes of master audio in one "
            "record — and both are still exact multiples of the 4 s refine "
            "step. THAT COUNT IS NOT PRODUCED BY THIS RENDER: refusal records "
            "are outside the population handed in, so it was measured "
            "separately and is dated by that measurement rather than by this "
            "artefact. So the GRID claim holds across both populations and the "
            "FOUR-VALUE claim holds only where a file was delivered. The pattern's edge is in the files that did not "
            "ship, which is the argument for keeping refusals in a census "
            "rather than the argument for excluding them.</p>")
        # ET LA BORNE SUPERIEURE, QUI EST LA MEME PHRASE VUE DE HAUT.
        #
        # ci a ecrit "tout remplissage au-dessus de 100 000 ms sur ce disque est
        # dans un refus". Teste sur mes octets: FAUX tel quel -- HUIT
        # remplissages livres depassent 100 000, dont une queue de 287947 et une
        # tete de 220000. VRAI restreint a l'INTERIEUR, et c'est la que ca dit
        # quelque chose: zero interieure livree au-dessus de la borne, deux
        # refusees. Quatrieme enonce de la nuit plus large que son test.
        corpus_said.append(
            "<p><b>And no delivered file in this corpus has an interior fill "
            "wider than the search bound.</b> The count is on the CORPUS row: "
            "<code>" + (next((plain(f.get("fills_above_the_bound"))
                              for k, f in records
                              if k == "CORPUS" and f.get("fills_above_the_bound")),
                             None) or "not supplied") + "</code>. Fills DO "
            "exceed 100 s, and every one of them is at the start or the end of "
            "a plan, where widths are not on the grid at all. Interior fills "
            "stop at 100 s in every file that shipped.</p>")
    if corpus_supplied:
        said.extend(corpus_said)
    elif corpus_said:
        # PAS DE SILENCE NON PLUS: une omission se lit "il n'y en avait pas".
        said.append(
            "<p><b>A corpus-scale finding about this file's gap widths exists "
            "and is NOT shown here, because no population was supplied to this "
            "render.</b> It concerns whether an interior gap width is a "
            "measurement of the media or a position on the locator's search "
            "grid, and it cannot be stated from one artefact: it needs a "
            "population, and the numbers behind it were measured elsewhere. "
            "Render this log together with others and the finding appears with "
            "its denominator. WHAT THIS REPORT CAN STILL TELL YOU ABOUT THIS "
            "FILE ALONE is above: the width of each filled region, and whether "
            "it equals a locator constant.</p>")
    # CE PARAGRAPHE EST DEVENU FAUX SUR UNE PARTIE DU CORPUS.
    #
    # Il dit "le localisateur calcule lequel des deux et cela n'atteint pas ce
    # rapport". Ce n'est plus vrai des artefacts qui portent des lignes
    # `repair: bracket`. Une phrase juste pendant trois semaines et fausse
    # depuis une heure, dans un document qui voyage: exactement le "perime en
    # place" que j'ai signale au Lead pour les depots.
    bracket_score = next((plain(f.get("bracket_agreement")) for k, f in records
                          if k == "CORPUS" and f.get("bracket_agreement")), None)
    if [f for k, f in records if k == "BRACKET"]:
        said.append(
            "<p><b>And on this artefact it CAN tell you why.</b> The locator "
            "now emits its own bracket for each change point, with "
            "<code>bound_only</code> — its statement that it could not narrow "
            "the search. Read the BRACKET rows: where <code>bound_only=True</code>, "
            "the master audio filling that gap is the width of the "
            "measurement's uncertainty and not the width of a hole in the "
            "candidate. The derived signature this report used before the field "
            "existed is printed beside it"
            + ((", and across the population handed to this render: "
                + html.escape(bracket_score) + ".</p>") if bracket_score
               else ". No population was supplied to this render, so this "
                    "paragraph says nothing about how often the two agree "
                    "elsewhere.</p>"))
    else:
        said.append(
        "<p><b>This figure cannot tell you WHY a gap was filled.</b> A region "
        "taken from the master because the candidate genuinely had nothing "
        "there, and a region taken because the measurement could not pin down "
        "the change point, are drawn IDENTICALLY. The locator computes which is "
        "which and it does not reach this report; the width above is a derived "
        "signature standing in for it. See "
        "<code>gap_is_filled_because_unsure</code> for the address. OTHER "
        "ARTEFACTS IN THIS CORPUS DO CARRY IT: the field has shipped, and its "
        "absence here is a fact about this record.</p>")

    tracks = [f for k, f in records if k == "TRACK" and f.get("kind") == "audio"]
    for track in tracks:
        number = track["track"]
        regions = [f for k, f in records if k == "REGION" and f["track"] == number]
        lost = [f for k, f in records if k == "LOST" and f["track"] == number]
        borrow = next((f for k, f in records
                       if k == "BORROW" and f["track"] == number), None)
        kept = [f for f in regions if f.get("kind") == "CANDIDATE"]
        filled = [f for f in regions if f.get("kind") in ("MASTER", "SILENCE")]
        offsets = [f.get("offset_s") for f in kept if f.get("offset_s")]

        sentence = [f"<p><b>Track {_escape(number)} "
                    f"({_escape(track.get('lang', '?'))}).</b> "]
        if kept:
            sentence.append(
                f"It takes {len(kept)} region(s) from the candidate and fills "
                f"{len(filled)} from "
                f"{_escape(plain(track.get('fill', 'the master')))}. ")
        if offsets:
            unique = []
            for value in offsets:
                if value not in unique:
                    unique.append(value)
            if len(unique) > 1:
                sentence.append(
                    f"<b>It reads the candidate at {len(unique)} different "
                    f"offsets</b> \u2014 "
                    f"{', '.join(_escape(v) + ' s' for v in unique)} \u2014 "
                    f"while the track line carries a single label, "
                    f"<code>offset={_escape(plain(track.get('offset', '?')))}</code>. "
                    f"One token stands over {len(unique)} values. ")
            else:
                sentence.append(f"It reads the candidate at "
                                f"{_escape(unique[0])} s throughout. ")
        else:
            sentence.append("<b>No offset is recoverable for this track</b>: "
                            "this format emits none by name, and the plan cut "
                            "nothing to derive one from. ")
        if borrow:
            sentence.append(
                f"<b>It BORROWS</b> the geometry of track "
                f"{_escape(borrow.get('from_track', '?'))}, the one carrying the "
                f"measurement language: "
                f"{_escape(plain(borrow.get('check', '?')))}. The log states "
                f"this nowhere \u2014 it is an inference, verified rather than "
                f"asserted. ")
        if lost:
            total = sum(Decimal(f["dropped_ms"]) for f in lost
                        if f.get("dropped_ms") not in (None, "UNMEASURED"))
            places = ", ".join(sorted({_escape(f.get("where", "?")) for f in lost}))
            sentence.append(f"<b>{_escape(seconds_fr(total, 1, signed=False))} s "
                            f"of candidate material is not in the output</b>, "
                            f"across {len(lost)} region(s), at the {places}. ")
        verify = plain(track.get("verify"))
        if verify:
            sentence.append(f"Verification says <code>{_escape(verify)}</code>")
            if track.get("probes"):
                sentence.append(f" on {_escape(track['probes'])} probe(s), worst "
                                f"{_escape(track.get('worst', '?'))}")
            if track.get("verified"):
                sentence.append(f", and {_escape(track['verified'])} of the "
                                f"file's audio tracks were verified at all")
            sentence.append(". ")
        sentence.append("</p>")
        said.append("".join(sentence))

    shortfalls = [f for k, f in records if k == "SHORTFALL"]
    if shortfalls:
        said.append(
            "<p><b>This file declares a shortfall, and this report CANNOT say "
            "where it came from.</b> The track line quantifies how short the "
            "fill source was and what the track lost; it does not say whether "
            "the repair caused it or <i>inherited</i> it from an already-short "
            "master. This log names the master and carries none of its "
            "durations, so the question is not decidable here. To date, <b>all "
            "five defects measured on produced files were inherited</b> "
            "\u2014 a <i>borrowed</i> figure, not measured here and recorded in "
            "no reference document \u2014 and no instance of the pipeline "
            "introducing one has been confirmed. Fidelity to a source is not "
            "correctness of an output.</p>")

    speeds = [f for k, f in records if k == "SPEED"]
    if speeds and all(f.get("ratio_applied_state") for f in speeds):
        said.append(
            "<p><b>No track in this artefact received a rate correction</b>, "
            "and that is not the same as \u201cnone needed one\u201d. The "
            "measurement proposed no rate, and <b>no producer of a speed "
            "verdict exists</b>: nobody asked the question. The figure has "
            "therefore never been drawn for a resampled track \u2014 that "
            "display path has never run.</p>")

    refused = [f for k, f in records if k == "REFUSED"]
    none_row = next((f for k, f in records if k == "REFUSED_NONE"), None)
    if refused:
        total = sum(Decimal(f["dropped_ms"]) for f in refused
                    if f.get("dropped_ms"))
        said.append(
            f"<p><b>{len(refused)} candidate region(s) were REFUSED</b>, "
            f"{_escape(seconds_fr(total, 1, signed=False))} s in all: the plan "
            f"had a candidate there and discarded it. They carry the dashed "
            f"amber box on the figure. That is a different thing from a fill "
            f"taken from the master, and both marks coexist.</p>")
    elif none_row:
        shortest = none_row.get("shortest_candidate_segment_ms")
        said.append(
            "<p><b>No candidate region was refused in this artefact</b>, so the "
            "figure carries no dashed amber box. Read that as \u201cthis case "
            "did not occur here\u201d and not as \u201cnothing is ever "
            "refused\u201d."
            + (f" A segment is refused only if it is SHORTER than the probe "
               f"window, which is 60 s; the shortest segment in this plan is "
               f"{_escape(seconds_fr(shortest, 0, signed=False))} s. The "
               f"condition is therefore not met"
               + (", and not by much" if Decimal(str(shortest)) < 120000 else "")
               + "." if shortest else "")
            + " The mark has never yet been drawn against genuinely refused "
              "material.</p>")

    gaps = [f for k, f in records if k == "GAP"]
    missing = [f for f in gaps if f.get("state") == NO_PRODUCER]
    if missing:
        said.append(
            f"<p><b>{len(missing)} quantity(ies) this report must show have no "
            f"producer at all:</b> "
            f"{', '.join('<code>' + _escape(plain(f['quantity'])) + '</code>' for f in missing)}. "
            f"They are listed above with their addresses. An empty cell here is "
            f"a FINDING and not a formatting accident \u2014 and it is "
            f"deliberately distinguishable from a field this format simply did "
            f"not carry yet.</p>")
    return "\n".join(said)


class merge_plan_error(Exception):
    """Un appel qui ne peut pas produire de rapport correct, refuse par son nom."""


def _job_contract():
    """LES CLES QU'UN `job` DOIT PORTER, DERIVEES ET NON RECOPIEES.

    Prises sur ce que `parse_job_log` construit, donc elles ne peuvent pas
    diverger de lui: ajouter une cle au lecteur l'ajoute au contrat le meme
    jour. Une liste recopiee a la main serait une SECONDE IMPLEMENTATION du
    dictionnaire, exactement ce que j'ai refuse a dev-1 pour sa configuration et
    a moi-meme pour le compte du corpus.
    """
    return frozenset(parse_job_log("").keys())


def validate_job(job):
    """REFUSE FORT PLUTOT QUE DE RENDRE A MOITIE. Signale par son nom.

    ci a livre son premier appel avec un `job` epars plausible et a obtenu
    `KeyError: 'plan'`. Quinze cles sont lues AVEC DES CROCHETS, et TREIZE
    d'entre elles sont lues aussi avec `.get` ailleurs -- donc QU'UNE CLE
    ABSENTE PLANTE OU NON DEPEND DE LA BRANCHE QUI S'EXECUTE. Une fusion qui
    marche sur dix fichiers peut mourir sur le onzieme par la meme cle absente.
    Seules `regions_added` et `regions_cut` ne sont jamais gardees nulle part.

    ET LE REMEDE N'EST PAS UN `.get()` GENERALISE AVEC DES DEFAUTS. Un rapport
    qui se rend avec un champ silencieusement absent est de la meme classe qu'un
    document tronque qui s'ouvre quand meme -- et le prefixe de longueur du
    transport existe precisement pour rendre cela impossible.
    """
    if not isinstance(job, dict):
        raise merge_plan_error(
            f"job must be the dict `parse_job_log` returns, not "
            f"{type(job).__name__}. Build it with `parse_job_log(<the emitted "
            f"bytes>)` and never by hand: a hand-built dict tests a fixture and "
            f"not the output, which is the failure this zone exists for")
    missing = sorted(_job_contract() - set(job))
    if missing:
        raise merge_plan_error(
            f"job is missing {len(missing)} key(s) this report reads: "
            f"{', '.join(missing)}. Pass `parse_job_log(<the emitted bytes>)`. "
            f"NOTE: 13 of the 15 keys are read both with brackets and with "
            f"`.get` in different branches, so an absent key crashes ONLY ON "
            f"SOME INPUTS -- a call that works on ten files can die on the "
            f"eleventh. This refusal is deliberate and is not fixed by "
            f"defaulting the value: a report rendered with a silently absent "
            f"field is the same class as a truncated document that still opens")
    return job


def render_report(job, artefact_id, source_name, caveats=(), corpus=None):
    """LE FICHIER. Un seul, et le rapport EST la page.

    L'ordre est deliberé: LES LIGNES D'ABORD. Le test qui prime est que rien
    dans la specification ne depende de l'existence du HTML -- alors on met en
    tete ce qui survit a `cat`, et le dessin apres, rendu depuis ces lignes.
    """
    validate_job(job)
    rows = build_rows(job, artefact_id, source_name, list(caveats), corpus)
    records = parse_rows(rows)
    generation, description = format_generation(job)

    document = [
        # LE DOCTYPE EN PREMIER OCTET. Un commentaire AVANT lui fait basculer
        # certains navigateurs en quirks mode, et ce fichier est fait pour etre
        # EXTRAIT D'UN JOURNAL PUIS OUVERT PAR DOUBLE-CLIC: il n'y a personne
        # pour diagnostiquer un rendu degrade.
        "<!doctype html>",
        # `lang="en"`: LE PROPRIETAIRE A TRANCHE QUE LES DOCUMENTS PRODUITS SONT
        # EN ANGLAIS. Je remplace la JUSTIFICATION et pas seulement la chaine --
        # laissee en place elle re-autoriserait le francais au prochain passage.
        #
        # ET CETTE DECISION FAIT TOMBER CE QUI SEPARAIT MES DEUX PUBLICS. J'avais
        # les lignes en anglais pour les agents et la prose en francais pour
        # l'humain: LA LANGUE PORTAIT LA DISTINCTION. Elle ne la porte plus, donc
        # elle est portee par LE ROLE, que le proprietaire a lui-meme enonce --
        # "le schema c'est bien pour l'humain et le texte c'est un recap pour
        # etre certain de pas tout perdre, et ainsi les agents peuvent rapidement
        # comprendre". Un seul document, une seule langue, DEUX ROLES DECLARES:
        # le dessin pour l'humain, les lignes contre la perte et pour les agents.
        '<html lang="en"><head><meta charset="utf-8">',
        "<!-- VMSAM merge_plan report. SPEC_ZONE_A.MD s4g.",
        "     THE ROWS BELOW ARE THE REPORT. The diagram is rendered from them and",
        "     adds no quantity of its own: `grep`, `cat` and `diff` give every",
        "     number in this file with no browser. Resolve fields BY NAME.",
        f"     artefact={artefact_id} format_generation={generation} ({description})",
        "     Opaque ids only: no media filename, title or catalogue id appears here.",
        "-->",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>merge_plan {_escape(artefact_id)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>merge_plan — artefact {_escape(artefact_id)}</h1>",
        ('<p class="note"><b>This file carries media names.</b> It is generated '
         'by VMSAM, written beside the produced file in the output directory, '
         'and never enters the repository. <b>Do not copy it, or lines from it, '
         'outside that directory</b> — quote the opaque id on the IDENTITY row '
         'instead, which is carried beside every name for exactly that purpose. '
         'The reader of this file is the one most likely to be tempted to cite '
         'it.</p>' if not REDACT_MEDIA_NAMES else ''),
        '<p class="note">Every record is one line, <code>KIND key=value …</code>. '
        'Resolve by name; there are no columns. A value that is not present carries '
        '<code>&lt;key&gt;_state</code> instead, so a field this format predates is '
        'never confused with a field nothing emits.</p>',
        "<h2>Rows</h2>", "<pre>",
    ]
    document.extend(_escape(row) for row in rows)
    document.append("</pre>")

    document.append("<h2>Timeline</h2>")
    document.append(
        '<p class="note legend">'
        '<span><i class="sw" style="background:var(--candidate)"></i>from the candidate</span>'
        '<span><i class="sw" style="background:var(--master)"></i>filled from the master</span>'
        '<span><i class="sw" style="background:var(--silence)"></i>filled with silence</span>'
        '<span><i class="sw" style="background:var(--lost)"></i>lost — candidate material '
        'not in the output</span></p>')
    document.append(
        '<p class="note">The axis is the <b>master</b> timeline. A lost region lives on '
        'the <b>candidate</b> timeline and is drawn at its insertion point on the master, '
        'at the same scale, so its extent compares with what was kept — a length drawn on '
        'an axis that is not its own, said rather than assumed. A <code>~</code> before an '
        'offset means it was <b>derived</b> from other emitted fields, not read.</p>')
    document.append(
        '<p class="note">Dashed amber marks <b>candidate material that was '
        'refused</b> \u2014 the plan had a candidate there and dropped it. That '
        'is a different thing from the master-contribution bar under the '
        'staircase, which shows what came <b>from the master</b>; a region can '
        'be either, both or neither. <b>If no dashed box appears, this artefact '
        'refused nothing</b> \u2014 a case not exercised here, and not a claim '
        'that nothing is ever refused.</p>')
    document.append(f'<div class="diagram">{render_svg(records)}</div>')

    document.append("<h2>What was done to this file</h2>")
    document.append(f'<div class="narrative">{render_narrative(records)}</div>')

    # LA PROPRIETE QUE CE MODULE REVENDIQUE DEPUIS LE DEBUT ET NE VERIFIAIT PAS:
    # AUCUN NOMBRE DU DESSIN QUI NE SOIT DANS LE TEXTE.
    #
    # dev-2, apres avoir attrape sa propre table faussee non par sa valeur mais
    # parce qu'un corpus rendait DEUX familles la ou l'echantillon en avait
    # CINQ: "un compte faux est invisible, une FORME fausse ne l'est pas --
    # enoncez des proprietes dont la violation est structurelle et non
    # numerique".
    #
    # Celle-ci l'est: un nombre dessine absent des lignes ne peut pas etre une
    # petite erreur, c'est une valeur venue d'ailleurs que du texte grepable, et
    # c'est exactement la chose que ce rapport promet de ne pas faire. Elle LEVE
    # au lieu de corriger, comme le filet de fuite -- un rendu qui ment sur sa
    # propre construction ne doit pas etre emis.
    _assert_figure_says_nothing_new(document)
    document.append("</body></html>")
    rendered = "\n".join(document)
    # LE CONTROLE FINAL, sur le document fini. Il leve, il ne corrige pas: un
    # correctif silencieux ici rendrait la fuite suivante invisible.
    assert_no_leak(rendered)
    return rendered


def report_for_log(text, artefact_id, source_name, caveats=(), corpus=None):
    """Octets d'un journal de travail -> le rapport. Rejette PAR STRUCTURE."""
    if not is_job_log(text):
        raise ValueError(
            f"{source_name} carries no `repair: plan` line, so it is not a job "
            f"log. Rejected by STRUCTURE and not by name: a `.log` suffix is not "
            f"evidence, and a denominator defended by a filename is defended "
            f"until the next filename.")
    return render_report(parse_job_log(text), artefact_id, source_name,
                         caveats, corpus)


# ---------------------------------------------------------------------------
# LA DESTINATION, ET LA COPIE DE TRANSPORT
#
# s4g: le rapport va A COTE DU FICHIER PRODUIT, et pas dans un repertoire de
# travail que quelque chose efface ensuite. Un manifeste compte 195 artefacts
# produits puis supprimes, 45 fusionnes et jamais verifies, un fusionne alors
# que le pipeline le disait DESYNCHRONISE. Aucun ne peut plus etre examine.
#
# LE FICHIER SUR DISQUE EST L'ORIGINAL. L'entree de journal est une COPIE DE
# TRANSPORT, une commodite pour un lecteur qui n'a pas encore d'acces disque.
# L'ordre n'est pas une preference: si la copie devient la seule, on a migre
# une source de verite au lieu d'ajouter un lecteur, et c'est exactement ce que
# la regle interdit. Le jour ou l'outillage lit le disque, l'append disparait
# et rien n'est perdu.

# LES LARGEURS QUI SONT DES CONSTANTES DU LOCALISATEUR ET NON DES MESURES.
#
#   100 000 ms = PROBE_STEP 40 000 + PROBE_WINDOW 60 000   borne de recherche,
#                                                          JAMAIS RESSERREE
#    12 000 ms = REFINE_WINDOW 8 000 + REFINE_STEP 4 000    plancher, RESSERRE
#
# dev-3 a etabli que l'intervalle du localisateur EST le trou rempli: sa largeur
# devient de l'audio maitre insere. Un trou dont la largeur EGALE EXACTEMENT la
# borne de recherche n'est donc pas un manque mesure, C'EST L'INCERTITUDE DE LA
# MESURE LIVREE COMME DU SON.
#
# CONSTANTES COPIEES, PAS LUES: le journal n'en porte aucune. Si dev-1 les
# change, ces nombres deviennent faux EN SILENCE, et le rapport le dit a cote du
# drapeau. Meme defaut que mes constantes de corpus, meme seule defense.
# EMPRUNTE AU PRODUCTEUR, PAS DEVINE: `change_point_locator.py:233` ecrit
# exactement `f"\t\t[change_point_locator] {message}\n"`. Le tag est lu apres
# desindentation, donc les deux tabulations ne le cachent pas.
_LOCATOR_TAG = "[change_point_locator]"

SEARCH_BOUND_MS = 100000
# Le pas d'affinage du localisateur, en ms. Emprunte, pas devine.
REFINE_STEP_MS = 4000
# Named so the census can state the CONSTRUCTION rather than assert a result.
REFINE_WINDOW_MS = 8000
PROBE_STEP_MS = 40000
PROBE_WINDOW_MS = 60000

REFINE_FLOOR_MS = 12000

TRANSPORT_KEYWORD = "MERGE_PLAN_HTML"

# LE PROPRIETAIRE A AUTORISE LES NOMS DE MEDIA DANS CE DOCUMENT.
#
#   "Ces journaux peuvent avoir les noms des medias. C'est tres utile de savoir
#    qui est quoi. Comme c'est genere par VMSAM, pas de soucis. Ce n'est pas
#    quelque chose qui finit dans le repo de VMSAM."
#
# SA PREMISSE EST JUSTE ET MON FILET ETAIT CALIBRE SUR UNE AUTRE. Mon message
# d'assertion disait "this artefact travels and the log it is built from does
# not". Celui-ci NE VOYAGE PAS: il est ecrit dans le repertoire de sortie, a
# cote du media, et n'entre jamais dans le depot.
#
# MAIS LE FILET FAISAIT DEUX CHOSES ET UNE SEULE EST CADUQUE. Il empechait le
# nom d'entrer dans le FICHIER -- caduc. Il rendait aussi IMPOSSIBLE PAR
# CONSTRUCTION qu'un agent lisant ce rapport recopie un nom dans un message, un
# commit ou une tache, parce qu'il n'y avait rien a recopier. PERSONNE N'A
# ANNULE CETTE SECONDE PROPRIETE, et c'est celle sur laquelle cette campagne a
# deja perdu deux fois -- un titre traversant un redacteur dans une phrase
# anglaise, dix-neuf codes d'episode dans un WORKLOG.
#
#   LA GARANTIE PASSE DE "IMPOSSIBLE" A "TOUT LE MONDE Y PENSE".
#   C'est un affaiblissement REEL, il est AUTORISE, et il n'est pas muet.
#
# Une seule ligne a rebasculer pour revenir en arriere, et le redacteur reste
# construit et teste sous elle.
REDACT_MEDIA_NAMES = False


def _assert_figure_says_nothing_new(document):
    """Tout nombre VISIBLE dans la figure existe dans les lignes."""
    text = "".join(document)
    figure = re.search(r"<svg.*?</svg>", text, re.S)
    if not figure:
        return
    rows = text[figure.end():] + text[:figure.start()]
    drawn = set()
    for label in re.findall(r"<text[^>]*>([^<]*)</text>", figure.group(0)):
        drawn |= set(re.findall(r"\d+[.,]?\d*", html.unescape(label)))
    plain_rows = html.unescape(re.sub(r"<[^>]+>", " ", rows))
    missing = sorted(n for n in drawn
                     if n not in plain_rows and n.replace(",", ".") not in plain_rows)
    if missing:
        raise merge_plan_error(
            "the figure draws numbers that are in no row: "
            + ", ".join(missing[:12])
            + ". This report's construction is that the drawing shows nothing "
              "the grep-able text does not; a render that breaks it is not "
              "emitted")


def transport_entry(document, artefact_id):
    """UNE entree, PREFIXEE PAR SA LONGUEUR EN OCTETS. Jamais de delimiteur.

    Un extracteur qui cherche une balise de fin applique un motif a un texte
    qu'il ne controle pas, et si la charge utile peut contenir ce motif il
    tronque -- en produisant un fichier a moitie ecrit QUI S'OUVRE QUAND MEME.
    C'est la classe qui a coute deux fois a cette campagne aujourd'hui: un
    redacteur mort sur une espace, un `sed` laissant passer un titre pendant des
    heures. On retire donc la possibilite au lieu de la surveiller: le lecteur
    prend EXACTEMENT n octets et ne cherche rien.

    EN OCTETS ET NON EN CARACTERES, ET L'EXTRACTEUR DOIT DECOUPER DES OCTETS.
    Le document est en UTF-8 et ses etiquettes sont francaises: `é`, `→`, `·` et
    `×` font plusieurs octets chacun. MESURE, et sur ma propre erreur: mon
    premier controle a decoupe la chaine par CARACTERES, a rendu 17467 octets
    la ou l'en-tete en annoncait 17466, et le document ne correspondait plus.
    Le prefixe etait juste; le lecteur ne l'etait pas. Un extracteur qui
    decoupe des caracteres produit un fichier corrompu QUI S'OUVRE QUAND MEME,
    ce qui est exactement le mode de defaillance que ce prefixe existe pour
    supprimer -- et il a suffi d'un tour pour que je le commette.

    UNE SEULE ENTREE. Repartie sur plusieurs `append`, la charge devrait etre
    rassemblee par le lecteur, et tout retour a la ligne ou toute troncature
    que le journal applique entre deux morceaux la corrompt de maniere
    invisible.
    """
    payload = document.encode("utf-8")
    return (f"{TRANSPORT_KEYWORD} {artefact_id} bytes={len(payload)}\n"
            + document + "\n")


def report_path(produced_file_path):
    """`<nom du produit>.merge_plan.html`, A COTE du fichier produit."""
    # `.html` PARCE QUE C'EN EST UN. Le proprietaire: "comme ce fichier est un
    # HTML, l'extension peut l'etre je pense." Mesure par l'architecte avant le
    # changement: RIEN dans `src/` ne globbe `*.log` dans le repertoire de
    # sortie -- les `.log` y sont ecrits et jamais relus par le code -- donc
    # aucun consommateur ne casse.
    return str(produced_file_path) + ".merge_plan.html"


# ---------------------------------------------------------------------------
# LE POINT D'APPEL, QUI EXISTE MAINTENANT -- et il n'est pas ou je l'avais ecrit
#
# CE BLOC DISAIT: "ce module n'est appele par rien" et "le point d'appel unique
# appartient au chemin de reparation". LE PREMIER EST PERIME ET LE SECOND ETAIT
# FAUX. Le proprietaire a tranche le cablage:
#
#     mergeVideo.py   remplit `merge_plan` avec les octets REELLEMENT EMIS,
#                     "".join(tools.logs[position_avant_reparation:]), et
#                     seulement si une version reparee existe.
#     fusion.py       appelle `parse_job_log` puis `write_report`.
#
# ET FUSION A RAISON CONTRE MOI, pour une raison que le chemin de reparation ne
# peut pas contourner: FUSION EST LE SEUL ENDROIT QUI CONNAIT LE CHEMIN PUBLIE.
# En mode test le fichier est deplace vers `VMSAM_TEST_OUTPUT_DIR` APRES le
# merge; la reparation n'a jamais vu cette destination. Un rapport ecrit depuis
# le chemin de reparation atterrirait a cote d'un fichier qui a demenage --
# c'est-a-dire exactement le defaut que s4g interdit, "pas dans un repertoire de
# travail que quelque chose efface ensuite", produit par ma propre regle.
#
# CE QUE L'APPELANT PASSE, et `validate_job` refuse par son nom si le dict ne
# porte pas les quinze cles:
#
#     job        = parse_job_log(<les octets emis>). JAMAIS un dict construit a
#                  la main: ce serait tester une fixture et pas la sortie, la
#                  faute qui a cree cette zone.
#     artefact_id        un identifiant OPAQUE, jamais un nom de media.
#     source_name        d'ou viennent ces octets.
#     produced_file_path le chemin PUBLIE du fichier produit. Le rapport s'ecrit
#                        a cote, `<ce chemin>.merge_plan.log`.
#     caveats    ce que l'appelant sait et que les octets ne portent pas.
#     corpus     `measure_corpus(...)` ou RIEN -- auquel cas le rapport DIT
#                qu'aucune population n'a ete fournie plutot que d'inventer.
#
# LE TRANSPORT EST EN RESERVE, PAS RETIRE. Le proprietaire a prefere un pointeur
# d'une ligne dans le `.log` au document entier: "je voulais que le plan soit
# sauvegarde dans les logs pour que je puisse les extraire. Mais j'ai meme
# mieux" -- le fichier a cote plutot que quatorze kilo-octets de HTML au milieu
# d'un journal qui se lit. `transport_entry()` reste construit et teste; la
# ligne a changer pour revenir a l'extraction est l'appel a `tools.logs.append`
# cote appelant, pas quoi que ce soit ici.
#
# UNE CONSEQUENCE ASSUMEE ET QUI EST UN TROU CONNU: `merge_plan` n'est rempli
# que si une version reparee existe, donc UNE REPARATION QUI TOURNE ET REFUSE NE
# PRODUIT AUCUN RAPPORT. C'est precisement la classe qui passe de zero a neuf
# artefacts avec `enforcing=True`, et c'est dans `mergeVideo.py` que cela se
# decide, pas ici.
#
# CE MODULE N'APPELLE JAMAIS `tools.logs` LUI-MEME, et c'est delibere: un
# producteur qui ne voit pas sa propre sortie ne teste pas sa sortie mais sa
# fixture.
def write_report(job, artefact_id, source_name, produced_file_path, caveats=(),
                 corpus=None):
    """Ecrit le rapport A SA DESTINATION, puis rend la copie de transport.

    L'appelant ecrit d'abord, transporte ensuite. Rien ici n'appelle
    `tools.logs`: le point d'appel unique appartient au chemin de reparation et
    pas a ce module, et un producteur qui ne voit pas sa propre sortie ne teste
    pas sa sortie mais sa fixture.

    Renvoie (chemin, entree_de_transport).
    """
    validate_job(job)
    document = render_report(job, artefact_id, source_name, caveats, corpus)
    destination = report_path(produced_file_path)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(document)
    return destination, transport_entry(document, artefact_id)
