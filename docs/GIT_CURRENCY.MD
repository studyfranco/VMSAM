# Git currency — how to know what a store holds RIGHT NOW

Distilled from campaign 2's `COMING_BACK.MD`. Rules with their consequence, no narrative.

1. **`git ls-remote <URL>` is a pure read.** It needs no configured remote, writes nothing, and
   asks the store itself rather than a cache. Name the authority by URL — `origin` names a
   different store at different seats, so `ls-remote origin` is not one instruction, it is two.
2. **Rows that differ prove you are stale. Rows that agree prove nothing.** If the hop never
   fetches, both sides go stale together and still match — agreement is not freshness.
3. **A content claim is permanent; a role claim expires.** "Content `X` is these bytes" cannot
   drift; "`<path>` is blob `X`" is a claim about a role, and roles move. A rebuild uses only the
   second kind — the one that expires.
4. **`git rev-parse -q --verify` before any read, and an EMPTY answer means ABSENT, not
   different.** An id you report as a blob must also pass `git cat-file -t` → `blob`: trees and
   commits pass every shape check a blob does.
5. **Never `git fetch && git show <cached-ref>:<path>` for governance.** A cached tracking ref
   serves blobs that are real and well-formed — for a tree that may no longer carry the file at
   all. Read governance from the authority's tip, by URL.
6. **Before any `reset --hard`, take a snapshot you can come back from.** Two artefacts:
   `git diff --binary > snap.patch` and
   `git ls-files --others --exclude-standard | tar czf untracked.tgz -T -`.
   Uncommitted work is the only thing a reset destroys.
7. **After a history rewrite, test the blob, not the commit list.** Ahead/behind counts measure
   reachability, and a rewrite changes reachability by design. Compare
   `git rev-parse -q --verify HEAD:<file>` against `origin/<branch>:<file>`, treating empty as
   ABSENT — then `format-patch` → `reset --hard` → `am`.
8. **When you kill a producer, kill its waiters.** Nothing warns you at the moment your own
   action makes a wait's premise false, and an `until [ -f … ]` loop on a file that will never
   appear runs forever.
