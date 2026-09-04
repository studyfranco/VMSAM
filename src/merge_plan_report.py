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
import hashlib
import re

# ---------------------------------------------------------------------------
# Etats de cellule

PRESENT = "present"
DERIVED = "derived"
COLLAPSED = "collapsed"
ABSENT_FORMAT = "absent-from-this-format"
NO_PRODUCER = "no-producer"

_STATE_MARK = {
    PRESENT: "",
    DERIVED: "~",           # calcule, pas lu
    COLLAPSED: "^",         # emis, mais agrege au-dessus de cette granularite
    ABSENT_FORMAT: "-",     # ce format ne le portait pas
    NO_PRODUCER: "x",       # personne ne l'emet
}

_STATE_WORD = {
    PRESENT: "read from the bytes by name",
    DERIVED: "derived from other emitted fields",
    COLLAPSED: "emitted, but aggregated above this granularity",
    ABSENT_FORMAT: "absent from this artefact's format",
    NO_PRODUCER: "no producer emits this",
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
        "candidate_opaque_id": None,
        "plan": None,
        "audios": {},
        "subtitles": [],
        "regions_added": {},
        "regions_cut": {},
        "regions_used": {},
        "skipped": [],
        "refused": [],
        "output_check": None,
        "summary_counts": None,
        "foreign_lines": [],
    }

    for line in lines:
        if not line.startswith("repair: "):
            # s4e: un saut hors du flux `repair:` existe et se dit. Un lecteur a
            # prefixe le laisse tomber, donc on le compte au lieu de l'ignorer.
            if line.strip() and not line.startswith(("Merged", "Logs:", "We was",
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
            job["master_opaque_id"] = opaque_id(body[len("master "):].strip())
            continue

        if body.startswith("plan "):
            rest = body[len("plan "):]
            kind = rest.split(" ", 1)[0]
            fields = split_fields(rest)
            pieces_text = rest.partition("pieces=")[2]
            job["plan"] = {
                "kind": kind,
                "language": fields.get("language"),
                "quantum_ms": fields.get("quantum"),
                "speed_margin": fields.get("speed_margin"),
                "pieces": parse_pieces(pieces_text),
            }
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
            job["regions_added"].setdefault(order, []).append({
                "kind": matched.group(2),
                "master_start_ms": _decimal(matched.group(3)),
                "master_end_ms": _decimal(matched.group(4)),
                "from": fields.get("from"),
            })
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
            job["regions_used"].setdefault(order, []).append({
                "master_start_ms": _decimal(matched.group(2)),
                "master_end_ms": _decimal(matched.group(3)),
                "candidate_start_ms": _decimal(matched.group(4)),
                "candidate_end_ms": _decimal(matched.group(5)),
                # PAR NOM. Un `-` de tete est un signe, pas une absence: un
                # decalage negatif est legitime (le candidat devance le maitre).
                "offset_ms": _decimal(fields.get("offset_ms")),
                "offset_ms_present": "offset_ms" in fields,
            })
            continue

        matched = re.match(r"CUT audio track (\d+) candidate ([\d.]+)-([\d.?]+)(.*)$", body)
        if matched:
            order = int(matched.group(1))
            fields = split_fields(matched.group(4))
            end_text = matched.group(3)
            job["regions_cut"].setdefault(order, []).append({
                "candidate_start_ms": _decimal(matched.group(2)),
                "candidate_end_ms": None if end_text == "?" else _decimal(end_text),
                "dropped_ms": (None if fields.get("dropped_ms") == "UNMEASURED"
                               else _decimal(fields.get("dropped_ms"))),
                "dropped_unmeasured": fields.get("dropped_ms") == "UNMEASURED",
                "where": fields.get("where"),
            })
            continue

        matched = re.match(r"subtitle track (\d+) (.*)$", body)
        if matched:
            fields = split_fields(matched.group(2))
            fields["stream_order"] = int(matched.group(1))
            job["subtitles"].append(fields)
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

        matched = re.match(r"repaired for (.*?): (.*)$", body)
        if matched:
            job["candidate_opaque_id"] = opaque_id(matched.group(1))
            job["summary_counts"] = matched.group(2)
            continue

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
            if region is None:
                rows.append({
                    "kind": "MASTER?",
                    "master_start_ms": piece["master_start_ms"],
                    "master_end_ms": piece["master_end_ms"],
                    "source": absent("no ADDED line for this master piece in "
                                     "this artefact's format"),
                    "offset": absent(),
                })
                continue
            # `from=silence` contre `from=master/<lang>`. On teste le PREFIXE:
            # la valeur porte une langue accolee pour le maitre et n'en porte
            # pas pour le silence.
            is_silence = str(region["from"] or "").startswith("silence")
            rows.append({
                "kind": "SILENCE" if is_silence else "MASTER",
                "master_start_ms": piece["master_start_ms"],
                "master_end_ms": piece["master_end_ms"],
                "source": Cell(region["from"], PRESENT),
                # Une region remplie ne LIT pas le candidat: elle n'a pas de
                # decalage. Absent parce que sans objet, et non parce que
                # manquant -- deux blancs differents.
                "offset": Cell("n/a", PRESENT, "a filled region reads no candidate"),
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

    def __call__(self, text):
        clean, hits = redact(text)
        self.hits += hits
        return clean


def assert_no_leak(document):
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
#   RIEN DANS LA SPECIFICATION NE DEPEND DE L'EXISTENCE DU HTML. Les LIGNES
#   portent chaque nombre; le dessin est rendu A PARTIR d'elles. `grep`, `cat`
#   et `diff` donnent tout sans navigateur, et un diagramme qu'on ne peut pas
#   ouvrir ne retire rien.
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
                cleaned = _wrap(value.value)
                parts.append(f"{name}={_redactor(cleaned) if _redactor else redact(cleaned)[0]}")
                if value.state == DERIVED:
                    parts.append(f"{name}_state=derived")
            else:
                parts.append(f"{name}_state={value.state}")
            continue
        if value is None:
            continue
        text = _wrap(value)
        parts.append(f"{name}={_redactor(text) if _redactor else redact(text)[0]}")
    return " ".join(parts)


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
    """Une duree en SECONDES, virgule decimale, comme le proprietaire les lit.

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
    return f"{quantised}".replace(".", ",")


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
        return {"language": language, "track": None,
                "agreement": f"undecidable: {len(reference)} tracks carry the "
                             f"plan language"}
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


def plan_end_ms(job):
    pieces = (job.get("plan") or {}).get("pieces") or []
    return pieces[-1]["master_end_ms"] if pieces else None


def build_rows(job, artefact_id, source_name, n_caveat):
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
    rows.append(_row("SOURCE", artefact=artefact_id, log=source_name,
                     format_generation=generation, format=description))
    rows.append(_row("IDENTITY",
                     master=job.get("master_opaque_id") or "",
                     candidate=job.get("candidate_opaque_id") or "",
                     construction="md5(path)[:16]",
                     note="opaque_ids_only;_no_media_name_travels_in_this_report"))
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
                     decimal_separator="comma (fr)"))
    rows.append(_row("PLAN", kind=plan.get("kind"), language=plan.get("language"),
                     quantum_ms=plan.get("quantum_ms"),
                     speed_margin=(Cell(plan["speed_margin"], PRESENT)
                                   if plan.get("speed_margin")
                                   else no_producer("no_key_produces_it")),
                     master_end_ms=_trim(plan_end_ms(job)),
                     master_end=clock(plan_end_ms(job)),
                     pieces=len(plan.get("pieces") or [])))
    rows.append(_row("PLAN_GEOMETRY_LIMIT",
                     detail="the_pieces=_token_is_assembly[pieces],_the_one_"
                            "normalize_segments_call_made_with_no_stream_order,"
                            "_i.e._the_geometry_a_BORROWING_track_uses"))

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
                         verified=fields.get("verified")))
        label = plain(fields.get("offset")) or ""
        if label.startswith("BORROWED"):
            borrow = borrow_provenance(job, order)
            if borrow:
                rows.append(_row("BORROW", _redactor=redactor, track=order,
                                 plan_language=borrow["language"],
                                 from_track=borrow["track"],
                                 attribution="inferred, not stated by the log",
                                 check=borrow["agreement"]))

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
                             offset_s=(seconds_fr(region["offset"].value, 3)
                                       if region["offset"].state in (PRESENT, DERIVED)
                                       and region["offset"].value not in (None, "n/a")
                                       else None)))

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
        rows.append(_row("REFUSED_NONE", _redactor=redactor,
                         count=0,
                         producer="merge_video_repair.log_assembly emits "
                                  "`repair: SKIPPED segment` for these",
                         note="this artefact records no refused candidate "
                              "segment, so the figure draws no dashed amber box. "
                              "NOT EXERCISED, not missing -- and the mark has "
                              "never been drawn against real refused material"))

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
    if job.get("summary_counts"):
        rows.append(_row("SUMMARY_COUNTS", text=job["summary_counts"].replace(" ", "_"),
                         note="a_count_is_a_statement_about_work_done,_not_about_a_file"))

    for entry in job.get("foreign_lines") or []:
        # Le FAIT qu'une ligne hors reparation existe, et de quoi la retrouver
        # dans le journal. Jamais son texte.
        rows.append(_row("OUTSIDE_REPAIR", _redactor=redactor,
                         digest=entry["digest"], chars=entry["chars"],
                         carries_path=entry["carries_path"],
                         tail=entry["tail"] or None,
                         note="content withheld: free text outside the repair "
                              "vocabulary cannot be redacted with a guarantee"))

    if redactor.hits:
        rows.append(_row("REDACTED", occurrences=redactor.hits,
                         note="values carrying an absolute path, catalogue id or "
                              "media filename were replaced by a stable opaque "
                              "token; the same input yields the same token"))

    for gap in blank_cells(job):
        rows.append(_row("GAP", quantity=gap["quantity"], state=gap["state"],
                         addressed_to=gap["address"], detail=gap["detail"]))
    return rows


def blank_cells(job):
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
        "address": "the seam: dev-1 emits it in a handoff JSON, "
                   "merge_video_repair.log_assembly has the conditional emitter, "
                   "nothing carries it between them",
        "detail": "0 occurrences in 13 artefacts. AND WHEN IT ARRIVES, `NON "
                  "EMISE` MUST BE CONDITIONED: dev-1 also emits "
                  "speed_margin_absent_reason and decided_by, because a margin "
                  "can be undefined while the decision was still decisive -- one "
                  "hypothesis clearing the fidelity gate leaves no flatness "
                  "margin, and printing that as unemitted would call a 0.36 "
                  "fidelity separation margin-less"})
    entries.append({
        "quantity": "verification_probe_positions",
        "state": COLLAPSED,
        "address": "merge_video_chimeric.verify_on_master_timeline",
        "detail": "per-probe master_position_ms lag_ms and correlation are built "
                  "and summarised to probes worst r_min; a max() has no position "
                  "so it cannot land on a timeline"})
    entries.append({
        "quantity": "borrowed_placement_tag",
        "state": NO_PRODUCER,
        "address": "merge_video_chimeric.mux_repaired_file",
        "detail": "s4g requires a tag DISTINCT from VMSAM_FABRICATED; the marker "
                  "value is the single string chimeric on every marked track of "
                  "every produced file measured"})
    entries.append({
        "quantity": "video_frame_rate",
        "state": NO_PRODUCER,
        "address": "merge_video_repair.log_assembly",
        "detail": "no fps, frame_rate or FrameRate key occurs in any artefact. "
                  "Any statement about a delay landing on the video grid -- "
                  "rounding, snapping, a drawn grid -- divides by a rate this "
                  "record does not carry, so it divides by an assumption"})
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
        "state": NO_PRODUCER,
        "address": "gestionar_show.fusion (job log header)",
        "detail": "no commit build version or image key occurs in any artefact; "
                  "the log records what was done and not which build did it"})
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
    plan_line = (f"plan : {plan.get('kind')} \u00b7 langue de mesure "
                 f"{plan.get('language')} \u00b7 quantum {plan.get('quantum_ms')} ms \u00b7 "
                 f"{plan.get('pieces')} morceaux \u00b7 duree maitre "
                 f"{plain(plan.get('master_end')) or '?'}")
    if plan.get("speed_margin_state"):
        # s4f: la marge par laquelle la transformation gagnante l'a emporte.
        # AUCUN PRODUCTEUR. On nomme le sujet et on marque la valeur manquante
        # -- on ne laisse pas un blanc se lire comme une marge nulle, qui serait
        # deux hypotheses a egalite.
        plan_line += " \u00b7 marge de vitesse : NON EMISE"
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
        speed = plain(lead.get("speed")) or ""
        resampled = not speed.lower().startswith("none")
        head = (f"piste {number} · {lead.get('lang')} · "
                f"{'mesuree' if not (plain(lead.get('offset')) or '').startswith('BORROWED') else 'EMPRUNTEE'}")
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
            head += (f" \u00b7 ACC\u00c9L\u00c9R\u00c9E \u00d7{_clip(ratio, 12)}"
                     f" \u00b7 marge de victoire : NON \u00c9MISE")
        elif "(" in speed and not speed.lower().startswith("none("):
            head += " \u00b7 vitesse : voir la ligne TRACK"
        else:
            # `non acceleree` ETAIT UNE REPONSE FAUSSE ET AFFIRMATIVE, et c'est
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
            head += " \u00b7 rythme : NON MESUR\u00c9"
        if lead.get("verify"):
            head += f" · verification {_clip(plain(lead['verify']), 24)}"
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
                                  "aucune correspondance", amber, 9, "middle"))

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
                            f'<title>pas de pas ici : coupe de tete ou de queue, '
                            f'rien ne l\u0027encadre</title></circle>')

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
            name = plain(fields.get("source")) or fields.get("source_state")
            if name and name not in sources:
                sources.append(name)
        # UNE LEGENDE QUI NOMME UNE SOURCE SANS SA QUANTITE invite le lecteur a
        # estimer la quantite sur la largeur de la barre. On la donne.
        body.append(_text(left, master_bar_y + 20,
                          "rajout\u00e9 du ma\u00eetre : "
                          + (" \u00b7 ".join(sources) or "rien")
                          + f" \u00b7 {len(filled)} r\u00e9gion(s)"
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
            check = plain(borrow.get("check")) or ""
            agrees = bool(re.match(r"offsets identical at (\d+) of \1 regions$",
                                   check))
            tone = faint if agrees else salmon
            body.append(f'<rect x="{left:.1f}" y="{strip:.1f}" width="{plot:.1f}" '
                        f'height="6" fill="none" stroke="{tone}" '
                        f'stroke-width="{1 if agrees else 1.4}" '
                        f'stroke-dasharray="{"3 3" if agrees else "none"}"/>')
            detail = (f"piste {other['track']} · {other.get('lang')} · EMPRUNTE "
                      f"cette geometrie · verification "
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
    out.append(_text(left, axis + 30, "timeline du ma\u00eetre", faint, 9))
    out.append('</svg>')
    return "\n".join(out)


def render_narrative(records):
    """LES MEMES FAITS EN PHRASES, ET EN FRANCAIS.

    s4g: le diagramme et les phrases sont la meme information deux fois, VOULU.
    Un lecteur doit voir ce qui a ete fait au fichier SANS lire une table.

    EN FRANCAIS PARCE QUE C'EST LE LECTEUR. La figure porte deja la virgule
    decimale pour la meme raison; laisser les phrases en anglais a cote d'une
    figure francaise, dans un document dont l'attribut `lang` dit `fr`, serait
    une incoherence que seul le destinataire paierait. LES LIGNES restent en
    anglais: elles sont lues par des agents et par `grep`, et leurs noms de
    champs sont ceux du producteur.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    source = next((f for k, f in records if k == "SOURCE"), {})
    identity = next((f for k, f in records if k == "IDENTITY"), {})
    said = []

    if plan:
        said.append(
            f"<p>L\u2019artefact <b>{_escape(plain(source.get('artefact', '?')))}</b> "
            f"a \u00e9t\u00e9 reconstruit sur la timeline du ma\u00eetre, qui "
            f"court jusqu\u2019\u00e0 <b>{_escape(plain(plan.get('master_end', '?')))}</b>. "
            f"La mesure a qualifi\u00e9 la relation de "
            f"<b>{_escape(plan.get('kind', '?'))}</b> et l\u2019a prise sur la "
            f"piste <b>{_escape(plan.get('language', '?'))}</b>, \u00e0 un quantum "
            f"de {_escape(plan.get('quantum_ms', '?'))} ms.</p>")
    if identity.get("master"):
        said.append(f"<p>Le ma\u00eetre est nomm\u00e9 dans ce journal, sous "
                    f"<b>{_escape(identity['master'])}</b>. C\u2019est ce qui "
                    f"rend un manque mesur\u00e9 dans le fichier produit "
                    f"<i>attribuable</i> \u00e0 la fusion ou au ma\u00eetre, et "
                    f"pas seulement constatable.</p>")
    else:
        said.append("<p><b>Le ma\u00eetre n\u2019est pas nomm\u00e9 dans ce "
                    "format.</b> Un manque mesur\u00e9 dans le fichier produit "
                    "ne peut donc \u00eatre attribu\u00e9 \u00e0 personne\u00a0: "
                    "la correspondance du contenu avec sa source est "
                    "immesurable, et c\u2019est une propri\u00e9t\u00e9 de "
                    "l\u2019enregistrement et non du fichier.</p>")

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

        sentence = [f"<p><b>Piste {_escape(number)} "
                    f"({_escape(track.get('lang', '?'))}).</b> "]
        if kept:
            sentence.append(
                f"Elle prend {len(kept)} r\u00e9gion(s) du candidat et en remplit "
                f"{len(filled)} depuis "
                f"{_escape(plain(track.get('fill', 'le ma\u00eetre')))}. ")
        if offsets:
            unique = []
            for value in offsets:
                if value not in unique:
                    unique.append(value)
            if len(unique) > 1:
                sentence.append(
                    f"<b>Elle lit le candidat \u00e0 {len(unique)} d\u00e9calages "
                    f"diff\u00e9rents</b> \u2014 "
                    f"{', '.join(_escape(v) + '\u00a0s' for v in unique)} \u2014 "
                    f"alors que la ligne de piste ne porte qu\u2019une seule "
                    f"\u00e9tiquette, <code>offset={_escape(plain(track.get('offset', '?')))}</code>. "
                    f"Un jeton se tient au-dessus de {len(unique)} valeurs. ")
            else:
                sentence.append(f"Elle lit le candidat \u00e0 "
                                f"{_escape(unique[0])}\u00a0s d\u2019un bout "
                                f"\u00e0 l\u2019autre. ")
        else:
            sentence.append("<b>Aucun d\u00e9calage n\u2019est r\u00e9cup\u00e9rable "
                            "pour cette piste</b>\u00a0: ce format n\u2019en "
                            "\u00e9met aucun par son nom, et le plan n\u2019a "
                            "rien coup\u00e9 d\u2019o\u00f9 en d\u00e9river un. ")
        if borrow:
            sentence.append(
                f"<b>Elle EMPRUNTE</b> la g\u00e9om\u00e9trie de la piste "
                f"{_escape(borrow.get('from_track', '?'))}, celle de la langue "
                f"de mesure\u00a0: {_escape(plain(borrow.get('check', '?')))}. "
                f"Le journal ne le dit nulle part \u2014 c\u2019est une "
                f"inf\u00e9rence, v\u00e9rifi\u00e9e et pas affirm\u00e9e. ")
        if lost:
            total = sum(Decimal(f["dropped_ms"]) for f in lost
                        if f.get("dropped_ms") not in (None, "UNMEASURED"))
            places = ", ".join(sorted({_escape(f.get("where", "?")) for f in lost}))
            sentence.append(f"<b>{_escape(seconds_fr(total, 1, signed=False))}\u00a0s "
                            f"de mati\u00e8re du candidat ne sont pas dans la "
                            f"sortie</b>, sur {len(lost)} r\u00e9gion(s), en "
                            f"{places}. ")
        verify = plain(track.get("verify"))
        if verify:
            sentence.append(f"La v\u00e9rification dit <code>{_escape(verify)}</code>")
            if track.get("probes"):
                sentence.append(f" sur {_escape(track['probes'])} sonde(s), pire "
                                f"\u00e9cart {_escape(track.get('worst', '?'))}")
            if track.get("verified"):
                sentence.append(f", et {_escape(track['verified'])} des pistes "
                                f"audio du fichier ont \u00e9t\u00e9 "
                                f"v\u00e9rifi\u00e9es tout court")
            sentence.append(". ")
        sentence.append("</p>")
        said.append("".join(sentence))

    refused = [f for k, f in records if k == "REFUSED"]
    none_row = next((f for k, f in records if k == "REFUSED_NONE"), None)
    if refused:
        total = sum(Decimal(f["dropped_ms"]) for f in refused
                    if f.get("dropped_ms"))
        said.append(
            f"<p><b>{len(refused)} r\u00e9gion(s) du candidat ont \u00e9t\u00e9 "
            f"REFUS\u00c9ES</b>, soit "
            f"{_escape(seconds_fr(total, 1, signed=False))}\u00a0s\u00a0: le plan "
            f"y avait un candidat et l\u2019a \u00e9cart\u00e9. Elles portent la "
            f"bo\u00eete ambre en pointill\u00e9 sur la figure. C\u2019est autre "
            f"chose qu\u2019un remplissage depuis le ma\u00eetre, et les deux "
            f"marques coexistent.</p>")
    elif none_row:
        said.append(
            "<p><b>Aucune r\u00e9gion du candidat n\u2019a \u00e9t\u00e9 "
            "refus\u00e9e dans cet artefact</b>, donc la figure ne porte aucune "
            "bo\u00eete ambre en pointill\u00e9. \u00c0 lire comme "
            "\u00ab\u00a0ce cas ne s\u2019est pas produit ici\u00a0\u00bb et "
            "non comme \u00ab\u00a0rien n\u2019est jamais refus\u00e9\u00a0\u00bb"
            "\u00a0: le producteur \u00e9met bien ces lignes, et la marque "
            "n\u2019a encore jamais \u00e9t\u00e9 dessin\u00e9e contre de la "
            "mati\u00e8re r\u00e9ellement refus\u00e9e.</p>")

    speeds = [f for k, f in records if k == "SPEED"]
    if speeds and all(f.get("ratio_applied_state") for f in speeds):
        said.append(
            "<p><b>Aucune piste de cet artefact n\u2019a re\u00e7u de correction "
            "de rythme</b>, et ce n\u2019est pas la m\u00eame chose que "
            "\u00ab\u00a0aucune n\u2019en avait besoin\u00a0\u00bb. La mesure "
            "n\u2019a propos\u00e9 aucun rythme, et <b>aucun producteur de "
            "verdict de vitesse n\u2019existe</b>\u00a0: personne n\u2019a pos\u00e9 "
            "la question. La figure n\u2019a donc jamais \u00e9t\u00e9 "
            "dessin\u00e9e pour une piste acc\u00e9l\u00e9r\u00e9e \u2014 ce "
            "chemin d\u2019affichage n\u2019a jamais servi.</p>")

    gaps = [f for k, f in records if k == "GAP"]
    missing = [f for f in gaps if f.get("state") == NO_PRODUCER]
    if missing:
        said.append(
            f"<p><b>{len(missing)} grandeur(s) que ce rapport doit montrer "
            f"n\u2019ont aucun producteur\u00a0:</b> "
            f"{', '.join('<code>' + _escape(plain(f['quantity'])) + '</code>' for f in missing)}. "
            f"Elles sont list\u00e9es plus haut avec leur adresse. Une cellule "
            f"vide ici est un CONSTAT et pas un accident de mise en forme "
            f"\u2014 et elle se distingue volontairement d\u2019un champ que "
            f"ce format-ci ne portait simplement pas encore.</p>")
    return "\n".join(said)


def render_report(job, artefact_id, source_name, caveats=()):
    """LE FICHIER. Un seul, et le rapport EST la page.

    L'ordre est deliberé: LES LIGNES D'ABORD. Le test qui prime est que rien
    dans la specification ne depende de l'existence du HTML -- alors on met en
    tete ce qui survit a `cat`, et le dessin apres, rendu depuis ces lignes.
    """
    rows = build_rows(job, artefact_id, source_name, list(caveats))
    records = parse_rows(rows)
    generation, description = format_generation(job)

    document = [
        # LE DOCTYPE EN PREMIER OCTET. Un commentaire AVANT lui fait basculer
        # certains navigateurs en quirks mode, et ce fichier est fait pour etre
        # EXTRAIT D'UN JOURNAL PUIS OUVERT PAR DOUBLE-CLIC: il n'y a personne
        # pour diagnostiquer un rendu degrade.
        "<!doctype html>",
        # `lang="fr"`: la figure et les phrases sont en francais parce que c'est
        # le lecteur. Les LIGNES restent en anglais -- elles sont lues par des
        # agents et par `grep`, et leurs noms de champs sont ceux du producteur.
        '<html lang="fr"><head><meta charset="utf-8">',
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
    document.append("</body></html>")
    rendered = "\n".join(document)
    # LE CONTROLE FINAL, sur le document fini. Il leve, il ne corrige pas: un
    # correctif silencieux ici rendrait la fuite suivante invisible.
    assert_no_leak(rendered)
    return rendered


def report_for_log(text, artefact_id, source_name, caveats=()):
    """Octets d'un journal de travail -> le rapport. Rejette PAR STRUCTURE."""
    if not is_job_log(text):
        raise ValueError(
            f"{source_name} carries no `repair: plan` line, so it is not a job "
            f"log. Rejected by STRUCTURE and not by name: a `.log` suffix is not "
            f"evidence, and a denominator defended by a filename is defended "
            f"until the next filename.")
    return render_report(parse_job_log(text), artefact_id, source_name, caveats)


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

TRANSPORT_KEYWORD = "MERGE_PLAN_HTML"


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
    """`<nom du candidat>.merge_plan.log`, A COTE du fichier produit."""
    return str(produced_file_path) + ".merge_plan.log"


def write_report(job, artefact_id, source_name, produced_file_path, caveats=()):
    """Ecrit le rapport A SA DESTINATION, puis rend la copie de transport.

    L'appelant ecrit d'abord, transporte ensuite. Rien ici n'appelle
    `tools.logs`: le point d'appel unique appartient au chemin de reparation et
    pas a ce module, et un producteur qui ne voit pas sa propre sortie ne teste
    pas sa sortie mais sa fixture.

    Renvoie (chemin, entree_de_transport).
    """
    document = render_report(job, artefact_id, source_name, caveats)
    destination = report_path(produced_file_path)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(document)
    return destination, transport_entry(document, artefact_id)
