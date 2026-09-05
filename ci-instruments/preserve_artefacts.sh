#!/bin/bash
# COPY EVERY ARTEFACT BEFORE THE RUNNER CAN DELETE IT.
# The running runner cannot be told to stop deleting without restarting it, and a
# restart costs the file currently in flight. This copies out of band instead: the
# producer keeps its behaviour, the evidence survives, and nothing in the run changes.
# Hard-link where possible (free, same filesystem), copy otherwise.
mkdir -p /config/output/KEEP
# SECOND DIRECTORY, NEVER KEEP. A refused artefact must not be hard-linked into KEEP
# and counted as produced -- that exclusion is deliberate and stays. But EXCLUDED FROM
# THE COUNT and DELETED are different things, and I had built only the first.
# The gate now refuses files for real (id 44: a track 2128 ms short of the master --
# the id 134 class, caught), and every one of those is evidence that the gate fired.
mkdir -p /config/output/DECLINED
# RETENTION, per the owner: delete under DISK PRESSURE, not after verification.
#   * .log and .log.error are NEVER evicted -- the log is the durable record and the
#     file is the perishable one. A deleted file WITH its plan is a fact; a deleted
#     file without one is an absence, and 195 artefacts have already been destroyed
#     leaving nothing to say what they contained.
#   * .mkv is evicted only below the floor, and in this order:
#       validated first  (forensic has read and accepted it -- cheapest to lose)
#       unread next      (only under real pressure)
#       REFUTED LAST     (evidence of a defect nobody has explained; most expensive)
#   * status comes from a file forensic writes, NOT from anything I decide.
FLOOR_GB=30
GRACE_H=6          # nothing younger than this is evicted under an empty oracle
STATUS=/config/output/KEEP/validation-status.tsv
LEDGER=/config/output/KEEP/ci-preservation-ledger.tsv
# THE SAME HEADER/ROW MISMATCH THAT BIT THE DECLINED LEDGER, HERE AND UNFIXED.
# I found it there, migrated it, added a guard -- AND DID NOT ASK WHETHER THE OTHER
# LEDGER HAD IT. It did: 62 rows of 10 fields and 35 of 8 under an EIGHT-column header,
# so `csv.DictReader` silently dropped `size_at_capture` and `mtime_at_capture` into a
# None key on every recent row.
#
# And it hid an answer: the one "MISSING from disk" row I have reported all night as an
# unexplained gap carries its own explanation in a NINTH field -- "WITHDRAWN: ci test
# probe, NOT an artefact" -- which the reader threw away.
LEDGER_SCHEMA='keep_name\tkind\tstem_sha256_16\tinode\tdev\tnlink_at_capture\tbasis\tcaptured_utc\tsize_at_capture\tmtime_at_capture\tnote'
if [ -s "$LEDGER" ]; then
  _h=$(head -1 "$LEDGER")
  if [ "$_h" != "$(printf '%b' "$LEDGER_SCHEMA")" ]; then
    echo "$(date -Is) LEDGER SCHEMA MISMATCH in $LEDGER"
    echo "  header  : $_h"
    echo "  expected: $(printf '%b' "$LEDGER_SCHEMA")"
    echo "  APPENDING WOULD ADD A SHAPE THE HEADER DOES NOT DESCRIBE. Migrate first."
  fi
fi
PROTECT=/config/output/KEEP/protected-stems.tsv     # "<keep-hash>\tvalidated|refuted"
DECLINED_DIR=/config/output/DECLINED
DECLINED_LEDGER=$DECLINED_DIR/ci-declined-ledger.tsv
# A HEADER THAT NO LONGER DESCRIBES ITS ROWS. I added `source_zone`, then `observer`,
# to a LIVE APPEND-ONLY ledger -- and the header is only written `if [ ! -s ]`, so it
# stayed at the original ten columns while rows arrived with eleven and then twelve.
# Three schemas in one file; a header-driven parser read `source_zone` as
# `src_sha256_16` on 5 of 20 rows and reported a digest of "lab".
#
# THE FILE LOOKED COMPLETE AND EVERY ROW WAS WELL FORMED. Nothing was missing; the
# LABELS had stopped describing the values. That is the populated-column defect again,
# one layer out: not a key that matches nothing, but a header that names the wrong
# fields.
#
# So: check the header against the schema every start, and say so LOUDLY rather than
# appending a fourth shape underneath it.
DECLINED_SCHEMA='declined_name\tkind\tsource_zone\tobserver\tsrc_sha256_16\tinode\tdev\tnlink_at_capture\tbasis\tcaptured_utc\tsize_at_capture\tmtime_at_capture'
if [ -s "$DECLINED_LEDGER" ]; then
  _have=$(head -1 "$DECLINED_LEDGER")
  if [ "$_have" != "$(printf '%b' "$DECLINED_SCHEMA")" ]; then
    echo "$(date -Is) LEDGER SCHEMA MISMATCH in $DECLINED_LEDGER"
    echo "  header : $_have"
    echo "  expected: $(printf '%b' "$DECLINED_SCHEMA")"
    echo "  APPENDING WOULD ADD A SHAPE THE HEADER DOES NOT DESCRIBE. Migrate first."
  fi
fi
touch "$STATUS"

evict_if_pressured() {
  local avail_gb st
  avail_gb=$(df -BG --output=avail /config/output 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -z "$avail_gb" ] && return 0
  [ "$avail_gb" -ge "$FLOOR_GB" ] && return 0
  echo "$(date -Is) DISK PRESSURE: ${avail_gb}GB free, floor ${FLOOR_GB}GB"

  # FOUR STATUSES, forensic's, in eviction order:
  #   VALIDATED            first  -- it has yielded its information
  #   NOT_EXAMINED         middle -- not reached, AND THAT IS NOT A PASS
  #   COULD_NOT_DETERMINE  late   -- measured and undecidable; someone should keep it
  #   REFUTED              last   -- evidence of a defect somebody will re-examine
  #
  # AN EMPTY ORACLE IS NOT MISSING DATA. It is a complete answer: nothing has been
  # validated, so everything is NOT_EXAMINED and the ordinary order applies. The
  # degraded mode I wrote earlier is DELETED -- it was a special case for a situation
  # that needs none, and a special case that never fires is untested code in a path
  # that only runs when something is already going wrong.
  #
  # THE ONE FORBIDDEN RESOLUTION: an absent row must NEVER read as VALIDATED, because
  # validated is cheapest to lose. A file nobody has read is not a file somebody
  # approved. This is could-not-determine versus safe-to-delete, aimed at an oracle
  # instead of a measurement, and it is the one place tonight where getting it wrong
  # deletes something.
  for class in VALIDATED NOT_EXAMINED COULD_NOT_DETERMINE REFUTED; do
    # OLDEST FIRST WITHIN EACH CLASS. A glob is not an order. Resolving every file
    # to one status -- which is what an empty oracle does, correctly -- collapses the
    # four-way order into ONE undifferentiated pass, and that pass had no order at
    # all. So "absent = NOT_EXAMINED" is right and INSUFFICIENT: it makes the status
    # honest and leaves the queue arbitrary. Sorting by mtime means the order is
    # defined even when every file shares a status, which is exactly the case the
    # ruling produces.
    for f in $(ls -1tr /config/output/KEEP/*.mkv 2>/dev/null); do
      [ -e "$f" ] || continue
      h=$(basename "$f" .mkv)
      # `ext` WAS NEVER ASSIGNED IN THIS LOOP. It carried whatever value the PRESERVE
      # loop left behind -- "log" or "error" on any pass where those were seen last --
      # and the protected-stem lookup below builds "$h.$ext". A stale "log" makes that
      # key match nothing, so the protection SILENTLY DOES NOT FIRE. This loop only
      # ever walks *.mkv, so the value is knowable; it was simply never stated.
      ext=mkv
      # FIND THE STATUS COLUMN BY ITS HEADER, NOT BY POSITION. I specified the format
      # as "<hash>\t<status>" and forensic wrote "file size_bytes duration date status".
      # My reader took $2 -- the SIZE -- as the status, matched no class, and would
      # have resolved EVERY file to NOT_EXAMINED including the REFUTED one, evicting
      # the single most expensive artefact in the directory under the ordering built
      # to protect it. A positional read of someone else's file is a convention I did
      # not own; the header is the only thing that says what a column means.
      # THE FILE IS TAB-SEPARATED AND awk DEFAULTS TO WHITESPACE. Reading the column
      # by header fixed the POSITION and not the SPLIT: the moment a value contained a
      # space -- "pre-f878454 (superseded)" -- it became two fields and every column
      # after it shifted by one. My reader returned the IMAGE where the STATUS lives,
      # so a REFUTED artefact read as "(SUPERSEDED)", matched no class, and fell to
      # NOT_EXAMINED. The same 2.7 GB file, the same wrong eviction, through a
      # different door -- and I had "fixed" this once already.
      # FS is part of a format. Finding the header does not help if the fields are
      # not the fields.
      st=$(awk -F'\t' -v k="$h" '
             /^#/ { next }
             !col { for(i=1;i<=NF;i++) if (tolower($i)=="status") col=i; if (col) next }
             col && ($1==k || $1==k".mkv") { print toupper($col); exit }
           ' "$STATUS" 2>/dev/null | head -1)
      if [ -z "$st" ]; then
        # NAMED BRANCH, not a default value. A default is invisible at the call site;
        # a branch says which of the four states an unlisted file is in.
        st=NOT_EXAMINED
        echo "$(date -Is) UNLISTED $h -> NOT_EXAMINED (never VALIDATED)"
      fi
      # NEVER EVICT A PROTECTED STEM, at any pressure and in any class. A peer needs
      # the uninstrumented artefact of specific ids to compare against a later
      # instrumented run, and THE COMPARISON ONLY EXISTS IF BOTH HALVES SURVIVE.
      # Checked by STEM, not by keep-name, so it holds for the .mkv, the .log and the
      # .log.error of the same source alike.
      # MATCH ON THE KEEP NAME AS THE LEDGER WRITES IT -- WITH its extension.
      # `h` is the basename with .mkv stripped; ledger column 1 keeps it. `$1==$h`
      # could never match, so the guard could never fire -- and my first control
      # passed because I wrote "$h.mkv" in the CONTROL and "$h" in the SCRIPT.
      # A control that tests an expression the code does not contain is not a control.
      pstem=$(awk -F'\t' -v kn="$h.$ext" '!/^#/ && $1==kn {print $3}' "$LEDGER" 2>/dev/null | head -1)
      if [ -n "$pstem" ] && grep -q "^$pstem	" "$PROTECT" 2>/dev/null; then
        echo "$(date -Is) PROTECTED $h stem=$pstem -- not evicted"
        continue
      fi
      # A GATE-DECLINED ARTEFACT RANKS WITH REFUTED. The Lead: "ci preserves what ships;
      # NOTHING PRESERVES WHAT STOPS." With enforcing on, a declined file is evidence of
      # a defect nobody has explained -- the same thing REFUTED means -- and it is the
      # only class that would otherwise never be kept at all.
      dk=$(awk -F'\t' -v kn="$h.$ext" '!/^#/ && $1==kn {print $2}' "$LEDGER" 2>/dev/null | head -1)
      if [ "$dk" = "declined_by_gate" ] || [ "$dk" = "unadjudicated" ]; then
        st=REFUTED
        echo "$(date -Is) $dk $h -> ranked REFUTED (evicted last)"
      fi
      [ "$st" != "$class" ] && continue
      avail_gb=$(df -BG --output=avail /config/output 2>/dev/null | tail -1 | tr -dc '0-9')
      [ "$avail_gb" -ge "$FLOOR_GB" ] && return 0
      echo "$(date -Is) EVICT $h class=$class"
      rm -f "$f"
    done
  done
}

while true; do
  evict_if_pressured
  # SKIP DOT-FILES. I wrote a 20-byte test probe to /config/output/srv to verify the
  # ledger, deleted the source, and the hard link SURVIVED BY DESIGN into a directory
  # three other agents count files in. A peer's published denominator went stale by one
  # and another had to learn to reject the file by STRUCTURE rather than by name.
  # A TEST ARTEFACT IN A SHARED DIRECTORY IS INDISTINGUISHABLE FROM EVIDENCE, and the
  # preserver is exactly the mechanism that makes it durable.
  find /config/output/srv -type f ! -name '.*' \( -name '*.mkv' -o -name '*.log' -o -name '*.log.error' -o -name '*.REFUSED.*' -o -name '*.NOVERDICT.*' -o -name '*.merge_plan.html' \) 2>/dev/null |
  while read -r f; do
    b=$(printf '%s' "$f" | sha256sum | cut -c1-16)
    ext="${f##*.}"
    # A GATE-DECLINED FILE IS EVIDENCE, NOT WASTE -- and it is the only kind I was
    # about to stop keeping. With enforcing=True six of sixteen files STOP, and a
    # stopped file never reaches forensic, so its population becomes PRE-FILTERED BY
    # AN INSTRUMENT IT ALREADY AGREES WITH. A gate-versus-column disagreement is
    # exactly what a declined file would carry, and forensic measured 11 of 11
    # concordance with ZERO cases in the band where the two could separate -- so that
    # evidence does not exist yet and does not grow unless these are kept.
    #
    # I told dev-2 an hour ago I would EXCLUDE these so they were not counted as
    # produced. Wrong in the direction that matters: EXCLUSION IS WHAT MAKES A DECLINE
    # INVISIBLE. They are kept, in their own class, so they are never counted as
    # shipped and never absent.
    # THREE STATES, THREE TOKENS. `declined_by_gate` says THE GATE DECIDED.
    # `unadjudicated` says NOBODY DECIDED -- a tool fault escaped before any verdict
    # was reached (ffprobe non-zero, or json.loads on its stdout), so the file is on
    # disk claimed by no report. dev-2 declined to reuse .REFUSED for it and was right:
    # nothing refused it, and a tool fault is not a verdict about the media.
    #
    # Both are PRESERVED and both are EXCLUDED FROM PRODUCED COUNTS. They differ in
    # what they assert, not in how they are kept -- and the difference is the whole
    # reason for a third token: "refused" and "never adjudicated" are not the same
    # fact, and one token for both would recreate the collapse this is fixing.
    case "$f" in
      *.REFUSED.*)        kind="declined_by_gate" ;;
      *.NOVERDICT.*)      kind="unadjudicated" ;;
      # THE DELIVERABLE THE OWNER VALIDATED AND WILL OPEN IN A BROWSER, AND I WAS NOT
      # KEEPING IT. I raised an alarm about the report OVER-filling the preserved logs;
      # the architect measured that the transport copy is never written -- the .log
      # gains 81 bytes and a pointer -- so that risk did not exist. The real one is the
      # inverse and it was in my own sentence: my scan matched *.log and *.mkv, so the
      # .html fell through entirely.
      #
      # A report never PRESERVED and a report never PRODUCED are indistinguishable in
      # KEEP. That is `absent never zero` on the deliverable itself, and I built the
      # rule and then failed it on the one artefact the owner has validated.
      *.merge_plan.html)  kind="merge_plan_report" ;;
      # A REFUSAL RECORD IS KINDED BY ITS EXTENSION, WHICH SAYS NOTHING ABOUT IT.
      # The Lead found id 31's complete refusal record -- the six-piece plan, both
      # digests, `undelivered state=REFUSED` and the full decline text -- sitting in
      # KEEP as `kind=error`, indistinguishable from any other error log. It WAS
      # enumerated and preserved (21 of 21 on disk are in the ledger); it was never
      # CLASSIFIED. So `declined_by_gate` and `unadjudicated` counted zero while the
      # records were on disk under a generic token.
      #
      # THE .mkv IS DESTROYED ON EVERY RECREATE AND THIS RECORD IS NOT. It is the only
      # surviving account of those refusals, and there are more of them than there are
      # produced files.
      #
      # Content, not name: the marker lives inside the file. Cheap -- these are a few
      # kB and the sweep already has the path open in the ledger step.
      *.log|*.error|*.log.error)
        if grep -qE 'undelivered state=(REFUSED|NOVERDICT)' "$f" 2>/dev/null; then
          if grep -q 'undelivered state=NOVERDICT' "$f" 2>/dev/null; then
            kind="unadjudicated_record"
          else
            kind="declined_by_gate_record"
          fi
        else
          kind="$ext"
        fi ;;
      *)                  kind="$ext" ;;
    esac
    dst="/config/output/KEEP/${b}.${ext}"
    [ -e "$dst" ] && continue
    if ln "$f" "$dst" 2>/dev/null; then basis="hardlink"; else cp -p "$f" "$dst" 2>/dev/null; basis="copy"; fi
    # WRITE THE BINDING WHILE THE WINDOW IS OPEN.
    # A peer proved four of five artefact<->log pairings by INODE IDENTITY -- two files
    # can share a size, they cannot share an inode by accident. The fifth was already
    # nlink=1: the runner had deleted its copy, the log's output path no longer
    # resolved, and THE PROOF WAS GONE PERMANENTLY. The hard link does not preserve the
    # pairing; IT PRESERVES A WINDOW IN WHICH THE PAIRING CAN BE CAPTURED, and nlink is
    # the clock. Every artefact of this run passes through nlink=2 and then drops to 1.
    # Scraping after the fact catches only what happens to still be open when someone
    # looks. Writing it HERE cannot be missed: at this instant the loop necessarily
    # holds both paths and the link is 2 by construction.
    # The STEM is what binds an artefact to its log -- both derive from one source
    # path -- so it is recorded rather than left to be re-derived by a reader.
    # THE REPORT COULD NEVER BIND TO ITS OWN ARTEFACT. The stem strips `.log.error`,
    # `.log` and `.mkv` -- and a report is named `<produced>.mkv.merge_plan.html`, which
    # matches NONE of them. So every merge_plan_report row carried a stem no other row
    # could share, and `vmsam-dev-4` reported the symptom: seven production reports and
    # not one `.log` they could associate with any of them.
    #
    # The binding column existed, was populated, and joined nothing -- the same shape as
    # the three classifiers nothing routed into. A KEY THAT MATCHES NOTHING IS NOT A KEY.
    stem=$(basename "$f")
    stem="${stem%.merge_plan.html}"
    stem="${stem%.log.error}"; stem="${stem%.log}"; stem="${stem%.mkv}"
    stemhash=$(printf '%s' "$stem" | sha256sum | cut -c1-16)
    if [ ! -s "$LEDGER" ]; then
      printf '%b\n' "$LEDGER_SCHEMA" > "$LEDGER"
    fi
      # SIZE AND MTIME AT CAPTURE. A HARD LINK IS NOT A SNAPSHOT.
      # forensic found KEEP/1853b392becad5e5.mkv had grown from 1853126458 to
      # 1858565468 bytes ON THE SAME INODE. The preserver never re-linked -- it skips
      # existing names -- the PRODUCER rewrote the original in place, and a hard link
      # shares the inode, so the "preserved" copy was rewritten with it.
      #
      # My ledger recorded inode, dev and nlink: three facts about IDENTITY and none
      # about CONTENT, so drift was undetectable from it. forensic caught it only
      # because ITS rows carry size at verdict time, and by accident at that. The
      # DURATION was identical to the millisecond, so a duration check would have
      # missed it too. Only the byte size differed.
      #
      # AN INODE PROVES SAME FILE OBJECT. IT DOES NOT PROVE SAME BYTES.
      # ONE stat CALL, NOT FOUR. Four separate `stat` processes read the metadata at
      # FOUR DIFFERENT INSTANTS, so a row could carry a size from one moment and an
      # mtime from another -- A TORN READ OF A FILE'S METADATA.
      #
      # MEASURED, on `9c22d7d987840f1f.mkv`: size_at_capture 1 224 802 304 against a
      # file that is 1 610 057 582 today, WITH THE MTIME IDENTICAL. The producer was
      # still writing when I hard-linked it; the %s call caught 1.22 GB mid-write and
      # the %Y call, microseconds later, caught the final timestamp. `captured_utc` and
      # `mtime_at_capture` are the SAME SECOND.
      #
      # The hard link was fine -- the file completed on the same inode and forensic has
      # validated it at its full size. WHAT WAS WRONG WAS MY RECORD OF IT: a size that
      # was never the final size, sitting beside an mtime that was.
      _st=$(stat -c '%i %d %h %s %Y' "$dst" 2>/dev/null)
      set -- $_st
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${b}.${ext}" "$kind" "$stemhash" \
        "${1:-}" "${2:-}" "${3:-}" "$basis" "$(date -u +%FT%TZ)" \
        "${4:-}" "${5:-}" "" >> "$LEDGER"
  done

  # ---- THE WHITELIST'S FAILURE MODE IS SILENCE ---------------------------------
  # The main sweep is a whitelist of extensions. A file type nobody thought of is not
  # reported as unhandled -- it is simply never mentioned, which is indistinguishable
  # from not existing. `.error` was invisible to two independent instruments for
  # exactly this reason.
  #
  # So: enumerate what is actually there, and NAME anything the whitelist does not
  # cover. An empty line here is then a measurement rather than an absence.
  UNSEEN=$(find /config/output/srv -type f ! -name '.*' \
             ! -name '*.mkv' ! -name '*.log' ! -name '*.error' \
             ! -name '*.REFUSED.*' ! -name '*.NOVERDICT.*' ! -name '*.html' \
             2>/dev/null | sed -E 's|.*/[^/]*\.||' | sort -u | tr '\n' ' ')
  if [ -n "${UNSEEN// /}" ]; then
    echo "$(date -Is) SWEEP WHITELIST GAP: unenumerated extension(s) under srv: $UNSEEN"
  fi

  # ---- PUBLISH THE FAMILY TABLE WHEN THE LEDGER MOVES ---------------------------
  # An artefact and its log share no part of their names; the binding is the ledger's
  # stem column. `vmsam-forensic` filed a verdict as UNVERIFIABLE for want of a log that
  # was on disk, because it could not resolve that binding and I had never published it.
  # Regenerated only when the ledger's mtime changes -- a consumer reading between two
  # captures gets the last complete table, and the write is atomic.
  _lm=$(stat -c %Y "$LEDGER" 2>/dev/null || echo 0)
  if [ "${_lm:-0}" != "${_last_ledger_mtime:-}" ]; then
    python3 "$(dirname "$0")/publish_families.py" >/dev/null 2>&1 && _last_ledger_mtime="$_lm"
  fi

  # ---- SECOND SWEEP: the declined class, wherever it lives ----------------------
  # The main sweep roots at /config/output/srv. Every *.REFUSED.* and *.NOVERDICT.*
  # on this disk is somewhere else, so the classification existed and NEVER FIRED --
  # `declined_by_gate` and `unadjudicated` are defined in the case statement above and
  # appear zero times in the ledger. A kind that is never assigned is not a kind.
  #
  # Rooting the MAIN sweep at /config/output instead would drag 509 files belonging to
  # other agents into KEEP. So: same root, but ONLY the two declined patterns, into a
  # separate directory with a separate ledger. KEEP and every count over it are
  # unchanged.
  #
  # NOVERDICT IS THE PRIORITY OF THE TWO. A REFUSED file is a gate working and the log
  # says why. A NOVERDICT file is a tool fault that escaped before any verdict existed,
  # so the artefact is the only record that the case occurred at all.
  #
  # WHAT THIS DOES NOT REACH, stated because a partial fix that looks total is worse
  # than none: the pipeline's own undelivered repairs are written to
  # /tmp/gestionar_show_<container-start>/ INSIDE THE CONTAINER. That path is not
  # visible from this host and is destroyed on every recreate. This sweep saves the
  # declined artefacts that reach the shared disk; it cannot save those.
  find /config/output \
       -type d \( -name KEEP -o -name DECLINED \) -prune -o \
       -type f ! -name '.*' \( -name '*.REFUSED.*' -o -name '*.NOVERDICT.*' \) -print 2>/dev/null |
  while read -r f; do
    case "$f" in
      *.REFUSED.*)   dkind="declined_by_gate"; mark="REFUSED" ;;
      *.NOVERDICT.*) dkind="unadjudicated";    mark="NOVERDICT" ;;
      *) continue ;;
    esac
    db=$(printf '%s' "$f" | sha256sum | cut -c1-16)
    dext="${f##*.}"
    ddst="$DECLINED_DIR/${db}.${mark}.${dext}"
    [ -e "$ddst" ] && continue
    if ln "$f" "$ddst" 2>/dev/null; then dbasis="hardlink"; else cp -p "$f" "$ddst" 2>/dev/null; dbasis="copy"; fi
    [ -e "$ddst" ] || continue
    # WHOSE REFUSAL IS THIS? The ledger records a HASH of the source path and nothing
    # about WHERE it came from -- so a lab artefact and a container refusal are
    # indistinguishable in the only record that outlives them. `vmsam-dev-2` came one
    # restart from writing its own scan's refusals into this exact directory: its
    # durable store now defaults to /config/output/undelivered/, which is MY evidence
    # path, and only an accident of module-load timing kept 14 lab artefacts out of it.
    #
    # A test artefact in a shared directory is indistinguishable from evidence -- my own
    # rule, and I had built the directory without the column that enforces it.
    case "$f" in
      /config/output/undelivered/*) zone="container" ;;
      /config/output/srv/*)         zone="container" ;;
      *)                            zone="lab" ;;
    esac
    # WHICH CONTAINER OBSERVED THE REFUSAL. dev-2 put the observer in the NAME rather
    # than the path (283dab7), so `case_key` stays the sole location of a case and one
    # case is not fragmented across two directories:
    #     <case key>/<name>.<observer>.REFUSED.mkv
    # My DECLINED name is sha256(source path)[:16] + marker + ext, WHICH DROPS THE
    # OBSERVER. Two observers give two different source paths so my COUNT is right --
    # but the directory would not say which container produced which, and that is the
    # whole point of the token.
    #
    # `no-observer-token` rather than blank when the name carries none: a file predating
    # dev-2's change and a file whose observer is unknown are different facts, and dev-2
    # emits `unknown-observer` precisely so those two worlds stay distinguishable.
    obs=$(basename "$f" | sed -nE 's/^[^.]*\.([^.]+)\.(REFUSED|NOVERDICT).*/\1/p')
    [ -z "$obs" ] && obs="no-observer-token"
    if [ ! -s "$DECLINED_LEDGER" ]; then
      printf 'declined_name\tkind\tsource_zone\tobserver\tsrc_sha256_16\tinode\tdev\tnlink_at_capture\tbasis\tcaptured_utc\tsize_at_capture\tmtime_at_capture\n' > "$DECLINED_LEDGER"
    fi
    # ONE stat CALL -- see the KEEP row above for the torn read this prevents.
    _dst_stat=$(stat -c '%i %d %h %s %Y' "$ddst" 2>/dev/null)
    set -- $_dst_stat
    _ds1="${1:-}"; _ds2="${2:-}"; _ds3="${3:-}"; _ds4="${4:-}"; _ds5="${5:-}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${db}.${mark}.${dext}" "$dkind" "$zone" "$obs" "$db" \
      "${_ds1:-}" "${_ds2:-}" \
      "${_ds3:-}" "$dbasis" "$(date -u +%FT%TZ)" \
      "${_ds4:-}" "${_ds5:-}" >> "$DECLINED_LEDGER"
  done
  sleep 5
done
