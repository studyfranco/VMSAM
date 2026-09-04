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
    # coupe rien la porte quand meme. La derivation reste calculee A COTE, comme
    # controle croise: deux instruments qui tombent d'accord valent mieux qu'un,
    # et s'ils divergent on montre les deux plutot que le plus joli des deux.
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
        # s3.3: UNE etiquette pour toute la piste. Quand les regions portent des
        # decalages differents, on le DIT plutot que de laisser l'etiquette
        # passer pour une propriete de chaque region.
        rows.append(_row("TRACK_LABEL_LIMIT", track=order,
                         offset_label=fields.get("offset"),
                         offset_fidelity_state=COLLAPSED,
                         detail="offset_measured_is_all()_over_segments_and_"
                                "offset_fidelity_is_the_first_non-None;_both_"
                                "are_per-TRACK_over_per-REGION_values"))

        for index, region in enumerate(track_regions(job, order)):
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
                                 where=region["where"]))
                continue
            rows.append(_row("REGION", track=order, seq=index, kind=region["kind"],
                             master_start_ms=_trim(region["master_start_ms"]),
                             master_end_ms=_trim(region["master_end_ms"]),
                             master_start=clock(region["master_start_ms"]),
                             master_end=clock(region["master_end_ms"]),
                             source=region["source"], offset_ms=region["offset"]))

    for fields in job.get("subtitles") or []:
        rows.append(_row("SUBTITLE", track=fields.get("stream_order"),
                         lang=fields.get("lang"), format=fields.get("format"),
                         kept_cues=fields.get("kept_cues"),
                         dropped_cues=fields.get("dropped_cues")))

    for entry in job.get("skipped") or []:
        rows.append(_row("SKIPPED", detail=entry.replace(" ", "_")))

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
        "address": "change_point_locator.locate_change_points",
        "detail": "log_assembly has a conditional emitter for it, so it would "
                  "print the moment something produced it; the absence is upstream"})
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


def render_svg(records):
    """s4g: la timeline du maitre avec HORODATAGES, une barre par segment a son
    decalage, les regions perdues en ROUGE AU-DESSUS des valeurs de decalage, et
    TOUTE valeur de decalage utilisee.

    Ecrit a la main, en balises. Pas de bibliotheque, pas de `<script>`, pas de
    `<img>`, aucune fonte distante: le conteneur n'a pas de garantie de reseau.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    if not plan or not plan.get("master_end_ms"):
        return ('<p class="note">No plan geometry in this artefact, so there is '
                'no master timeline to draw. The rows above carry everything '
                'this format emitted.</p>')
    span = Decimal(plan["master_end_ms"])

    tracks = []
    for kind, fields in records:
        if kind == "TRACK" and fields.get("kind") == "audio":
            tracks.append(fields["track"])
    regions = {}
    lost = {}
    for kind, fields in records:
        if kind == "REGION":
            regions.setdefault(fields["track"], []).append(fields)
        elif kind == "LOST":
            lost.setdefault(fields["track"], []).append(fields)

    left, right, width = 78, 18, 916
    plot = width - left - right
    lost_band, label_band, bar_height, spacer = 16, 15, 24, 20
    block = lost_band + label_band + bar_height + spacer
    top, axis = 26, 34
    height = top + block * max(1, len(tracks)) + axis

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'style="max-width:{width}px;height:auto" role="img" '
           f'aria-label="master timeline, one row per audio track">',
           '<title>Master timeline: every region and every offset used</title>']

    # Graduations, en TEMPS. Un axe en millisecondes brutes n'est pas un
    # horodatage, et s4g demande des horodatages.
    ticks = 8
    for index in range(ticks + 1):
        position = span * index / ticks
        x = left + _x(position, span, plot)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                   f'y2="{height - axis + 6:.0f}" stroke="var(--grid)" '
                   f'stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{height - axis + 20:.0f}" '
                   f'font-size="10" fill="var(--faint)" text-anchor="middle">'
                   f'{clock(position)}</text>')

    for row_index, track in enumerate(tracks):
        base = top + block * row_index
        bars_y = base + lost_band + label_band
        out.append(f'<text x="0" y="{bars_y + 16:.0f}" font-size="11" '
                   f'fill="var(--ink)">track {_escape(track)}</text>')

        for fields in regions.get(track, []):
            start = Decimal(fields["master_start_ms"])
            end = Decimal(fields["master_end_ms"])
            x = left + _x(start, span, plot)
            bar = max(1.0, _x(end - start, span, plot))
            colour = {"CANDIDATE": "var(--candidate)", "MASTER": "var(--master)",
                      "SILENCE": "var(--silence)"}.get(fields.get("kind"),
                                                       "var(--faint)")
            out.append(f'<rect x="{x:.1f}" y="{bars_y}" width="{bar:.1f}" '
                       f'height="{bar_height}" fill="{colour}"/>')
            # LA VALEUR DU DECALAGE, ECRITE. s4g la demande explicitement, et
            # une barre a la bonne place ne la dit pas.
            offset = fields.get("offset_ms")
            if offset and offset != "n/a":
                mark = "~" if fields.get("offset_ms_state") == "derived" else ""
                out.append(f'<text x="{x + bar / 2:.1f}" y="{bars_y - 4:.0f}" '
                           f'font-size="10" fill="var(--ink)" '
                           f'text-anchor="middle">{mark}{_escape(offset)} ms</text>')
            elif fields.get("offset_ms_state"):
                out.append(f'<text x="{x + bar / 2:.1f}" y="{bars_y - 4:.0f}" '
                           f'font-size="10" fill="var(--lost)" '
                           f'text-anchor="middle">no offset emitted</text>')

        # LES REGIONS PERDUES, EN ROUGE, AU-DESSUS DES VALEURS DE DECALAGE.
        # Elles vivent sur la timeline DU CANDIDAT; on les dessine a leur point
        # d'insertion cote maitre, a la meme echelle, pour que leur AMPLEUR se
        # compare a ce qui a ete garde. Le rapport le dit en toutes lettres --
        # une longueur dessinee sur un axe qui n'est pas le sien doit s'annoncer.
        candidates = [f for f in regions.get(track, []) if f.get("kind") == "CANDIDATE"]
        for fields in lost.get(track, []):
            where = fields.get("where")
            if where == "head" and candidates:
                boundary = Decimal(candidates[0]["master_start_ms"])
            elif where == "tail" and candidates:
                boundary = Decimal(candidates[-1]["master_end_ms"])
            else:
                index = [f for f in lost.get(track, [])
                         if f.get("where") == "interior"].index(fields)
                boundary = (Decimal(candidates[index]["master_end_ms"])
                            if index < len(candidates) else span)
            dropped = fields.get("dropped_ms")
            x = left + _x(boundary, span, plot)
            if dropped == "UNMEASURED" or dropped is None:
                out.append(f'<text x="{x:.1f}" y="{base + 11}" font-size="10" '
                           f'fill="var(--lost)" text-anchor="middle">'
                           f'lost, extent UNMEASURED</text>')
                continue
            rule = max(2.0, _x(Decimal(dropped), span, plot))
            out.append(f'<rect x="{x - rule / 2:.1f}" y="{base + 4}" '
                       f'width="{rule:.1f}" height="4" fill="var(--lost)"/>')
            out.append(f'<line x1="{x:.1f}" y1="{base + 8}" x2="{x:.1f}" '
                       f'y2="{bars_y - 13:.0f}" stroke="var(--lost)" '
                       f'stroke-width="1" stroke-dasharray="2 2"/>')
            out.append(f'<text x="{x:.1f}" y="{base + 2}" font-size="9" '
                       f'fill="var(--lost)" text-anchor="middle">'
                       f'-{_escape(dropped)} ms</text>')

    out.append('</svg>')
    return "\n".join(out)


def render_narrative(records):
    """LES MEMES FAITS EN PHRASES. Voulu: un lecteur doit voir ce qui a ete fait
    au fichier SANS lire une table. Le journal est lu par qui ouvre le journal;
    le rapport par qui ouvre le fichier.
    """
    plan = next((f for k, f in records if k == "PLAN"), None)
    source = next((f for k, f in records if k == "SOURCE"), {})
    identity = next((f for k, f in records if k == "IDENTITY"), {})
    said = []

    if plan:
        said.append(
            f"<p>Artefact <b>{_escape(source.get('artefact', '?'))}</b> was rebuilt "
            f"on the master's timeline, which runs to "
            f"<b>{_escape(plan.get('master_end', '?'))}</b>. The measurement called "
            f"the relationship <b>{_escape(plan.get('kind', '?'))}</b> and measured it "
            f"on the <b>{_escape(plan.get('language', '?'))}</b> track, at a quantum of "
            f"{_escape(plan.get('quantum_ms', '?'))} ms.</p>")
    if identity.get("master"):
        said.append(f"<p>The master is named in this log, as "
                    f"<b>{_escape(identity['master'])}</b>. That is what makes a "
                    f"deficit in the produced file attributable to the merge or to "
                    f"the master rather than merely present.</p>")
    else:
        said.append('<p><b>The master is not named in this artefact\'s format.</b> '
                    'A deficit measured in the produced file therefore cannot be '
                    'attributed: content correspondence to the source is '
                    'unmeasurable, and that is a property of the record, not of '
                    'the file.</p>')

    tracks = [f for k, f in records if k == "TRACK" and f.get("kind") == "audio"]
    for track in tracks:
        number = track["track"]
        regions = [f for k, f in records if k == "REGION" and f["track"] == number]
        lost = [f for k, f in records if k == "LOST" and f["track"] == number]
        kept = [f for f in regions if f.get("kind") == "CANDIDATE"]
        filled = [f for f in regions if f.get("kind") in ("MASTER", "SILENCE")]
        offsets = [f.get("offset_ms") for f in kept if f.get("offset_ms")]

        sentence = [f"<p><b>Track {_escape(number)} "
                    f"({_escape(track.get('lang', '?'))}).</b> "]
        if kept:
            sentence.append(
                f"It takes {len(kept)} region(s) from the candidate and fills "
                f"{len(filled)} from {_escape(plain(track.get("fill", "the master")))}. ")
        if offsets:
            unique = []
            for value in offsets:
                if value not in unique:
                    unique.append(value)
            if len(unique) > 1:
                sentence.append(
                    f"<b>It reads the candidate at {len(unique)} different offsets</b> "
                    f"— {', '.join(_escape(v) + ' ms' for v in unique)} — while the "
                    f"track line carries the single label "
                    f"<code>offset={_escape(track.get('offset', '?'))}</code>. "
                    f"One token stands over {len(unique)} values. ")
            else:
                sentence.append(f"It reads the candidate at {_escape(unique[0])} ms "
                                f"throughout. ")
        else:
            sentence.append("<b>No offset is recoverable for this track</b>: this "
                            "format emits none by name, and the plan cut nothing "
                            "to derive one from. ")
        if lost:
            total = sum(Decimal(f["dropped_ms"]) for f in lost
                        if f.get("dropped_ms") not in (None, "UNMEASURED"))
            places = ", ".join(sorted({_escape(f.get("where", "?")) for f in lost}))
            sentence.append(f"<b>{_trim(total)} ms of candidate material is not in "
                            f"the output</b>, across {len(lost)} region(s) at the "
                            f"{places}. ")
        verify = track.get("verify")
        if verify:
            sentence.append(f"Verification says <code>{_escape(verify)}</code>")
            if track.get("probes"):
                sentence.append(f" on {_escape(track['probes'])} probe(s), worst "
                                f"{_escape(track.get('worst', '?'))}")
            if track.get("verified"):
                sentence.append(f", with {_escape(track['verified'])} of the file's "
                                f"audio tracks verified at all")
            sentence.append(". ")
        sentence.append("</p>")
        said.append("".join(sentence))

    gaps = [f for k, f in records if k == "GAP"]
    missing = [f for f in gaps if f.get("state") == NO_PRODUCER]
    if missing:
        said.append(
            f"<p><b>{len(missing)} quantity(ies) this report is required to show "
            f"have no producer at all</b>: "
            f"{', '.join('<code>' + _escape(f['quantity']) + '</code>' for f in missing)}. "
            f"They are listed with their addresses in the rows above. An empty cell "
            f"here is a finding, not a formatting accident — and it is deliberately "
            f"distinguishable from a field this artefact's format simply predates.</p>")
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
        "<!-- VMSAM merge_plan report. SPEC_ZONE_A.MD s4g.",
        "     THE ROWS BELOW ARE THE REPORT. The diagram is rendered from them and",
        "     adds no quantity of its own: `grep`, `cat` and `diff` give every",
        "     number in this file with no browser. Resolve fields BY NAME.",
        f"     artefact={artefact_id} format_generation={generation} ({description})",
        "     Opaque ids only: no media filename, title or catalogue id appears here.",
        "-->",
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
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
