# VMSAM Web Interface — AI Agent Context & Directives

You are assisting with the `web-interface` branch of VMSAM: a small Rust server that
serves a static single-page UI and proxies it to the VMSAM `gestionar_show` API.
Read this before the first edit. The engine's own rules (`AGENT.MD` on `master`) do
not apply here, except the privacy rule, which applies everywhere.

## 🎯 What this is

VMSAM (`master`) watches a download folder, matches file names against per-folder
regex rules, renames and merges episodes into a library. Its API is a FastAPI app on
port 8080 with no UI. This branch is the UI. The page has two sections, **Shows** and
**Movies** (see "Sections" below); the Shows tabs are:

| tab | what it does | VMSAM routes |
| :--- | :--- | :--- |
| Home | one tile per tab, the whole tile is the link (see "Home tiles") | — |
| Folder Creation | browses `CREATE_ROOT`, creates a library folder | `POST /folder/` |
| Bulk Regex | one rule per group of downloaded files (grouped by name, regroupable by drag, "Auto-group", "Split all"); **the only place a regex is edited**: "Edit regex" on an Existing Rules row opens a *replace card* (see below) | `POST /regex/`, `POST /regex/replace/`, `GET /regex_folder/` |
| Index Folder | registers an existing library directory ("Select a folder…" opens the directory picker), hides files already indexed, proposes a rule per group, then renames and indexes | `POST /folder/`, `GET /episodes_folder/`, `GET /regex_folder/`, `POST /regex/`, `POST /index_folder/` |
| Move Folder | moves/renames a registered folder: destination typed relative to `CREATE_ROOT` (free "New path" input) or chosen with "Browse…" in the directory picker (choosing a directory proposes `<dir>/<folder name>`), live old → new preview; VMSAM moves the directory, rewrites the folder and episode paths and deletes the old parent directories left empty (`removed_directories`, shown relative to the root) | `POST /folders/move/`, `GET /episodes_folder/` |
| Episodes | lists the registered episodes of a folder (number, file name, weight); renames one inline (the extension stays apart, visible), or a batch: ticked episodes, a pattern holding `{<episode>}` (prefilled from the folder's rule rename patterns, or proposed from the names), live old → new preview with the episode on two digits as VMSAM writes it, the file's own extension when the pattern has none, confirmation, one call. VMSAM renames all or nothing; no rule is stored | `GET /episodes_folder/`, `GET /regex_folder/`, `POST /episodes/rename/` |
| Folder Settings | path (relative to `CREATE_ROOT`, editable, "Browse…" in the directory picker), original language, number cut, max episode, cut ratio of a folder, current values shown. `POST /folder/` is keyed by the path and takes the FULL data, so a new path is a move first (`confirmDialog`, removed directories shown), then, if a parameter changed, the save with every field under the new path; an unchanged path is only the save. A folder stored outside the root (or not as root + `/` + relative path) cannot be saved here: the save would create another folder | `GET /folder/`, `POST /folders/move/`, `POST /folder/` |
| Rules | every rule of a folder: **weight, destination folder (dropdown, a move is confirmed first) and rename pattern** (empty = none, sent as `null`), saved with `POST /regex/` and the FULL data (same `regex_pattern`, the chosen folder's `destination_path`, an `example_filename`); the example is built (`proposeRuleExample`: the edited then the stored rename pattern with episode 01 when the regex matches it, else `sampleFromRegex`), and only when none matches the user types a real file name, checked with `testPattern`. The regex is read-only, with "Edit regex in Bulk Regex" (`openRegexEdit`). Delete after confirmation | `GET /regex_folder/`, `POST /regex/`, `POST /regex/delete/` |
| Errors | rejected files (`incompatible_files`) grouped by folder, with the registered episode they failed to merge with (the master). Each folder group folds (its header is a button: path, file count); "Expand all" / "Collapse all" act on the folders shown; the search box keeps the folders whose path or one of whose file names contains the text (case-insensitive; a folder matched by file names shows those rows only). On arrival everything is listed and unfolded (the overview), a reload after a delete keeps the folds and the search. "Relaunch fusion" queues a merge, "Delete" removes the file, its `.log.error` and its row, and its error directory only when just `.error` files remain (VMSAM's lists of deleted files and removed directories are shown). A Fusion queue panel polls `GET /fusion` every 5 s while the tab is shown (the timer stops itself once the tab is hidden) | `GET /errors`, `GET /folders/`, `GET /episodes_folder/`, `POST /fusion`, `GET /fusion`, `POST /errors/delete/` |
| Incrementaller | rules that shift an episode number before the folder rules run | `POST /incrementaller/`, `GET /incrementaller_list/` |

### Sections

`index.html` holds two navs under the Shows / Movies switch (`switchSection`). Shows
tabs live in `app.js` (nav buttons before `<!-- SHOWS-NAV-END -->`, sections before
`<!-- SHOWS-SECTIONS-END -->`); Movies tabs live in `movie.js` (nav buttons before
`<!-- MOVIES-NAV-END -->`, `<section id="movie-…-tab">` before
`<!-- MOVIES-SECTIONS-END -->`), loaded after `app.js` and reusing its helpers. Both
register their init function in the shared `TAB_INIT` (tab id → function), which
`switchTab` calls on every visit; `SECTION_HOME` names the home tab of each section, and
the section of a tab is the `data-section` of the nav holding its button. Movies tabs:

The movie side is a separate VMSAM instance (public port 8081, `VMSAM_MOVIE_API_HOST`),
reached through `/api/movie/<name>` (`MOVIE_GET_ROUTES` / `MOVIE_POST_ROUTES` in
`api.rs`; `movies`, `movie_move` and `folders_move` take paths relative to
`CREATE_ROOT`, the relay prefixes them and refuses `..`). A movie is one kept file (a
film or a show special). Its **`title` is its name in VMSAM**: globally unique, the key
of `movies`, what every search and picker looks for first (`movieTitleRank`); it is
**not the file name**. Directory and suffix are shared freely ("Movie A (2001)" and
"Movie A (2001) extended" may share both). **The file name comes from the rule**: each
movie regex has a `rename_pattern` holding `{<suffix>}` (`MOVIE_SUFFIX_PLACEHOLDER`, the
movies' `{<episode>}`), e.g. `Movie A (2001){<suffix>}.mkv`; final file = the movie's
directory / the rename pattern with `{<suffix>}` replaced by the movie's suffix
(`movieRenamedName`); no rename pattern = the file keeps its name. A new suffix or rename
pattern never renames a file already in place: it applies to the next integration or
merge (the UI says so). Rules have **no episode group**: they only have to match the
downloaded names.

Movie routes (name → VMSAM): GET `movies_list` → `/movies/` (`{movies}`, 404 when
empty = no movie), `movies_title_like` → `/movies/title_like/` (`?title_like=`),
`movies_like` → `/movies/destination_like/`, `movie` → `/movie/` (`?movie_id=` or
`?title=`, 404 = unknown), `folders_list` → `/folders/` (`{folders: [{destination_path,
movies_count}]}`), `regex_movie`, `regex_list` (with `rename_pattern`, `title`, `suffix`,
`destination_path`), `errors`, `fusion_status`, `metadata_providers`, `health`; POST
`movies` → `/movie/`, `movie_move` → `/movie/move/`, `movies_delete`, `regex`,
`regex_replace` → `/regex/replace/`, `regex_delete`, `index_movie`, `folders_move`,
`errors_delete`, `fusion`, `metadata_tags`. A movie object is keyed **`movie_id`**
(never `id`): `{movie_id, title, destination_path, suffix, original_language,
number_cut, cut_file_to_get_delay_second_method, metadata_ids, file_path, file_weight}`.
`movies` **creates or updates by title, database only** (`{title, destination_path,
suffix, original_language, number_cut, cut_file_to_get_delay_second_method,
metadata_ids}`, no `movie_id`; never touches a file; 400 "use /movie/move/" when the
movie has a file and the directory differs); answers the movie + `message`.
`movie_move` (`{movie_id, new_destination_path}`) moves one movie's file (same name) to
another directory; answers the movie + `message`, `file_moved`, `removed_directories`.
`regex` creates a rule or, for an existing pattern, changes its movie, rename pattern and
weight (`{regex_pattern, movie_id, rename_pattern, weight, example_filename}`, the example
must match; answers `final_file_name`); `regex_replace` changes the pattern
(`{regex_pattern, new_regex_pattern, example_filename}`, plus `rename_pattern` /
`weight` / `movie_id` only when given: absent = kept). `index_movie` renames the chosen
file with its rule's rename pattern. `movies_update` / `regex_update` no longer exist.
**Roles**: the Rules tab changes a rule's weight, movie and rename pattern, never its
regex; the regex is changed in Bulk Regex, checked on real files of the watched folder.

The **rename pattern field** (`createMovieRenameField`) is the same everywhere (Bulk
Regex cards, replace cards, Index "+ Rule", Rules editor): input, validation
(`movieRenameProblem`: empty allowed = name kept, else it must hold `{<suffix>}`, no
`/`), live preview of the final file with the linked movie's directory and suffix.
`movieRenameFor(names)` proposes it: the first name with the wild spans of
`buildTemplate` dropped (separators merged like `buildRename`) and `{<suffix>}` right
before the extension (`Movie.G.2005.1080p.BluRay-AAA.mkv` + `Movie.G.2005.720p.WEB-BBB.mkv`
→ `Movie.G.2005.1080p{<suffix>}.mkv`; one file → its own name + `{<suffix>}`).

| tab (`movie-…-tab`) | what it does | movie routes |
| :--- | :--- | :--- |
| Home | the shared `.home-tile[data-tab]` tiles (no button: the whole tile, click / Enter / Space) | — |
| Movie Creation | directory typed or chosen with "Browse…" (the shared `openDirectoryPicker`, name filter; a sub-path that does not exist may be typed), **title** (explained: the movie's unique name in VMSAM, not the file name), suffix (replaces `{<suffix>}`), options; the **metadata generator**: one checkbox per app and one input per provider from `metadata_providers`, an input enabled only when a ticked app knows the provider, a chip per app (`plex: tmdb` / `jellyfin: unsupported`); the tag string is built by VMSAM (`metadata_tags`, never guessed by the UI) and can be used as the suffix, appended to it, or appended to the directory name. The typed title is looked up (`movie?title=`, debounced): "New title" → "Create Movie", a known title → "Update movie #id" (`movies` updates by title), refused here when that movie has a file and the directory differs (the Movies tab moves it). Summary: movie, directory, `…{<suffix>}.<ext> → …<suffix>.<ext>`. Full data sent, ids as `metadata_ids` (null when none) | `GET metadata_providers`, `POST metadata_tags`, `GET movie`, `POST movies` |
| Bulk Regex | files from the file modal, grouped with `groupByNamePattern`; a file an existing movie rule matches is listed apart under that rule ("Propose a new rule anyway"). One rule per group, `movieRegexFor`: one file → its escaped name anchored, several → `buildRegex(buildTemplate(names), null)` (no episode), the names as alternatives if the template missed one; checked in the browser on every file (`checkMovieRegex`). Each card has a **Rename Pattern** (proposed by `movieRenameFor`, "Re-propose", live final name with the linked movie's directory and suffix, empty or holding `{<suffix>}`, sent as `rename_pattern`, empty = `null`), links to a movie (searchable picker, title first) or "+ Create a movie" (Movie Creation with a title guessed from the name, the new movie linked back to the card), weight, the movie's existing rules (with their rename patterns) and the selected files they already match. **Editing a regex happens here**: "Edit regex" on an existing rule (card's rules panel, "already matched" panel, or the Rules tab's "Edit regex in Bulk Regex") opens an "Edit rule" card (`startMovieRegexReplace`, mode `replace`, kept out of Auto-group / Split all / drag) with the current regex and the watched-folder files it matches; "+ Add files" adds others; the new regex must match every file of the card and differ; the rule's rename pattern is shown in its field and kept (sent only when edited there; a rename pattern alone is changed in Rules); saved with `regex_replace`, `example_filename` = the first file. No real file → no save (explained on the card). A refusal (a rule of another movie matches the example, or a show rule takes it as an episode) is shown as VMSAM wrote it, on the card | `GET regex_list`, `GET movies_list`, `GET regex_movie`, `/api/fs/list` (watched folder), `POST regex`, `POST regex_replace` |
| Index | a movie (picker) or a directory (search over `folders_list`, or "Browse the directories…": the shared picker where movie directories show their count, a directory with movies further down says "movies inside", and a click on a directory without movie opens it) then one of its movies; the movie's directory, suffix, registered file and rules (each with its rename pattern and the name it gives); the directory's files with their state (registered file of this movie / of another one / matched by a rule, with the new name that rule gives / no rule); "+ Rule" opens an inline rule form for a file (pattern, weight, rename pattern proposed from the name); "Index movie now" → report: indexed (old → new name, from the rule's rename pattern), skipped with VMSAM's reason, unmatched, already registered | `GET movie`, `GET folders_list`, `GET regex_movie`, `/api/fs/list`, `POST regex`, `POST index_movie` |
| Move Folder | a directory holding movies (search or "Browse the directories…", as in Index), the movies moved with it; the destination: free "New path" input, or "Browse…" (shared picker: the chosen directory receives the moved one under its name; the moved directory is marked and cannot hold itself), live preview, `confirmDialog`; report: moved movies with their new directory, deleted empty directories | `GET folders_list`, `POST folders_move` |
| Movies | table (id, title, suffix, directory, file, weight, metadata ids) with a search over all of them; edit: the **title is read-only** (the key; renaming a movie = delete + create, said on the card), directory (typed or "Browse…"), suffix, options, metadata ids with the generator ("Use as suffix" / "Append to suffix" / "Append to the directory name"); shown: the current file and, per rule, the name its rename pattern gives with the suffix being edited (a new suffix does not rename the file in place). Save: a new directory for a movie that has a file is a move first (`confirmDialog`, `movie_move`: same name, `file_moved`, removed directories listed on the card), then, if anything else changed, `movies` with the title and the full data; a movie without file changes directory through `movies` only. The reopened card and the toast carry VMSAM's messages; delete after confirmation (VMSAM refuses a movie that still has rules or incompatible files: its reason is shown) | `GET movies_list`, `GET regex_movie`, `POST movie_move`, `POST movies`, `POST movies_delete` |
| Rules | every movie rule (pattern, movie title, rename pattern, directory, weight) with a filter and a "test a file name" box marking the rules that match it; edit **weight, linked movie (picker by title) and rename pattern** (validated, preview with the chosen movie's suffix; relink confirmed first) via `regex` with the full data (`rename_pattern` empty = `null`) and an example the regex matches: the tested name, else one built from the regex (`movieSampleFromRegex`: `sampleFromRegex`, per alternative for `^(?:a|b)$`), else a real name typed in; the regex is read-only there, with "Edit regex in Bulk Regex"; delete after confirmation | `GET regex_list`, `POST regex`, `POST regex_delete` |
| Errors | incompatible files grouped by movie, one collapsible group per movie (header click / Enter), "Expand all" / "Collapse all", a search (contains, case-insensitive) on title, directory and file names; everything listed and expanded on arrival; the movie's registered file; "Relaunch fusion", "Delete" (confirmed; VMSAM removes the file, its `.log.error`, the row, and the error directory when only `.error` files remain; reported); fusion queue panel polling every 5 s while the tab is shown | `GET errors`, `POST fusion`, `GET fusion_status`, `POST errors_delete` |

`movie.js` shares the global scope of `app.js` (two classic scripts): every top-level
name it declares carries `movie` / `Movie` (`movieRegexTab`, `createMovieSearchPicker`…),
since a `let`/`const` declared twice breaks the whole page. Its styles are in
`movie.css`.

## 🗺️ Files

| file | role |
| :--- | :--- |
| `src/main.rs` | poem server: `/api` routes, static files, CORS, `PORT` |
| `src/api.rs` | `/api/fs/list` (directory listing under a root), `/api/config`, `/api/vmsam/folders` and `/api/vmsam/folders_move` (paths relative to `CREATE_ROOT`, prefixed by the relay, `..` refused), and the generic `/api/vmsam/<name>` relay driven by `GET_ROUTES` / `POST_ROUTES` |
| `static/index.html` | the section switch, both tab navs, every tab section (home included) and the shared file modal; every element the JS looks up by id lives here |
| `static/app.js` | one plain script: API client, **filename analysis engine** (pure functions), section/tab switching, shared widgets, group board, one init function per Shows tab |
| `static/movie.js` | the Movies tabs, loaded after `app.js`: movie API client, search picker, directory pickers (on `openDirectoryPicker`), metadata generator, one init per tab |
| `static/movie.css` | styles of the Movies tabs, loaded after `style.css`, its colour variables only |
| `static/style.css` | dark theme; every colour is a variable on `:root` |
| `Dockerfile`, `docker-compose.yml` | multi-stage build; runtime needs only `ca-certificates` |

## 📜 Rules

**No external resource, ever.** No CDN script, no font, no stylesheet, no analytics.
The page must work on a LAN with no internet. A `<script src=…>` or `<link>` that does
not point at `static/` is a defect, whatever it brings.

**Vanilla only.** Plain JS, CSS and HTML; no bundler, no framework, no npm. Two
classic scripts share one global scope: `app.js` (shared helpers, engine, Shows tabs)
then `movie.js` (Movies tabs, reusing app.js helpers). New Shows functionality is a
new section in `app.js`, new Movies functionality goes in `movie.js`; never a new
toolchain. Every top-level name in `movie.js` carries `movie`/`Movie`: a same-named
`let`/`const` breaks the whole page, and a same-named `function` silently replaces
app.js's. Build nodes with `createElement` / `textContent`: file names carry
brackets, quotes and braces, and one `innerHTML` with a file name in it is an
injection.

**Naming.** JS and CSS keep the `camelCase` / kebab-case they already use. Rust is
`snake_case`, `cargo fmt` formatted, and **must pass `cargo build`, `cargo clippy` and
`cargo fmt --check` with no warning** before a commit.

**Every VMSAM call goes through the relay.** The browser never knows the VMSAM host.
The Shows names (`/api/vmsam/<name>`, `api` in `app.js`): `folders_list` → `GET /folders/`,
`folders_like` → `GET /folders/destination_like/`, `folder` → `GET /folder/?folder_id=`,
POST `folders` → `POST /folder/` (create **or update**, FULL data, `destination_path` relative
to `CREATE_ROOT`; answers the folder object + `message` "Folder created" / "Folder updated"),
POST `folders_move` → `POST /folders/move/`, POST `regex` → `POST /regex/` (creates, or updates
an existing pattern: folder via `destination_path`, `rename_pattern`, `weight`; the
`example_filename` must match and yield an episode ≥ 1 even for an update), POST
`regex_replace` → `POST /regex/replace/` (`{regex_pattern, new_regex_pattern,
example_filename}`: the pattern only). `folders_update` and `regex_update` are gone. A
folder object carries **`folder_id`** (never `id`): pickers, `folderCache`, every lookup by
id use it.
Adding a VMSAM route is one line in `GET_ROUTES` or `POST_ROUTES` in `api.rs`; the
relay forwards the upstream status and JSON body unchanged, so the UI can show
VMSAM's own `detail` message. Do not summarise an upstream error into "request
failed": the user needs the reason VMSAM gave.

**`CREATE_ROOT` is VMSAM's library root, on the same path.** Folder creation prepends
it to the relative path chosen in the browser and VMSAM stores the result as the
folder's `destination_path`; Index Folder does the reverse and strips it to list a
folder VMSAM already knows. Mount the library at the same path in both containers, or
nothing that reads a registered folder works.

**Privacy — the repository is potentially public.** No real show title, release
group, media file name or library path in a tracked file, a commit message or a
screenshot. Examples in code, help text and tests use invented names (`Show A`,
`[Grp]`). Real names are for manual testing only and stay outside the repository.

**HTTP client.** `reqwest` with `rustls-tls`, deliberately: no OpenSSL at build time,
and the upstream is plain HTTP on a private network anyway.

## 🧠 The filename engine (`app.js`, section 3)

The part worth understanding before touching any tab. Pure functions, exposed as
`window.vmsamEngine` for tests.

1. `tokenizeName` cuts a name into digit runs, letter runs and single other characters.
2. `buildTemplate` aligns every name of a group on the first one with a **weighted**
   LCS (`lcsPairs`): identical tokens weigh most, two digit runs of different values
   still align (so an episode number stays aligned when the title next to it
   changes), two words that differ only by case (`Of` / `of`) align too, and among
   equal alignments the one with fewest holes wins. A shared token stays literal
   (`caseFold` marks the letters whose case changes); an aligned digit run whose
   value changes is `varying`; anything else that differs is `wild`.
   `absorbBetweenWilds` merges a short literal run caught between two wild spans
   (shared title words) and a checksum glued to a wild word; `settleWildRuns` then
   measures each wild run against every name: a separator that opens or closes it
   in every name stays literal (` - ` before a title), and the run is `optional`
   only when it is empty in some name.
3. `findMarkerSpan` / `findEpisodeMarker`: when **every** name carries the same kind
   of season/episode marker — `S<n>E<n>`, `<n>x<nn>` (`3x01`, never `1920x1080`) or
   `Saison|Season <n> Épisode|Episode <n>` — its episode digits are the episode,
   unconditionally (+1000). Otherwise `scoreEpisodeSlot`
   scores each digit run: an `ep` word, a dash, zero padding and changing between
   files score up; resolution (`1080p`), codec (`x264`), version (`v2`), season
   (`S2`), year, CRC, brackets and the extension score down. Below −100 nothing is
   chosen and the user clicks a chip. A chip click always overrides.
4. `buildRegex` emits an anchored **Python** pattern: literals escaped, a letter
   whose case changes as a class (`[oO]f`), varying digits
   `\d+`, wild spans `.+` (`.*` when optional, lazy `.+?` right before the episode),
   the episode `(?P<episode>\d+)`. `anchorAfterEpisode` keeps the separator that
   follows the episode number literal when every name has it, before the wild span
   (`S01E(?P<episode>\d+)\..+`, or `\.(?:.+\.)?MULTi` when the title is sometimes
   absent); a variant is kept only if it extracts the same episodes. `toJsRegex`
   converts `(?P<` to `(?<` so the browser can check it on every file of the group.
5. `buildRename` is the first name (its spelling, case included) with the episode
   replaced by `{<episode>}` and every wild span (per-episode titles, tags) dropped,
   separators merged. Then the episode stands out (owner's rule): the episode with
   its marker glued (`S01E`, `1x`, `E`, `Ep 03`, a whole `Saison 1 Épisode 03`) is one
   unit, and a separator run on either side of it that holds no dash becomes ` - `:
   `Show.S01E03.1080p.WEB.mkv` → `Show - S01E{<episode>} - 1080p.WEB.mkv`,
   `Show 03 VOSTFR.mp4` → `Show - {<episode>} - VOSTFR.mp4`. A run that already holds a
   dash is kept (never ` -  - `); nothing is added at the start, at the end, before the
   extension, or where the unit touches a bracket or a word directly (`[03]`).

Default grouping: **Bulk Regex and Index Folder group with `groupByNamePattern`**;
**Incrementaller starts with one file per card**. A word is a run between
separators (space `. / , - ( ) _ ; : [ ] { } + ~ ! '`), compared case-insensitively;
the extension is not a word but must be equal. (1) Files gather by prefix: with a
season/episode marker (`findMarkerSpan`), the marker kind and the words before it;
without one, the first two words, where an episode-like word (`01`, `E05`, `ep3`,
`07v2`, at most three digits) only counts as "a number here". The **separators** of
that prefix (and the one right after it) are part of the key: `Show Z 01` and
`Show_Z_01` never group, one regex cannot cover both styles (owner's rule). (2) Inside a prefix,
the tail (last three words after the marker, or of the name; 8-hex CRC words left
out) splits files only when at least two files share it: two releases stay apart,
while a season whose names end with a per-episode title stays in one pooled group.
(3) A file with no marker and no episode-like word leaves a group where others have
one (a special among numbered episodes) for its own card. (4) `refineGroupsByPairs`
aligns every pair of a group of 4 to 200 names; pairs that keep the same extra words
literal (beyond the whole group's) form a class, classes become sub-groups (largest
first, leftovers join the sub-group that still gains with them). The split is kept
only when every sub-group has two names or more, gains a word outside the
per-episode title (`partGain`: a word that opens the title area right after the
episode, or a fixed number after a wild span, does not count), the sub-group
regexes differ, and no more groups lose a valid episode: two releases (`MULTi` /
`DUAL`, `[GrpA]` / `[GrpB]`), two seasons, OP / ED separate; a season whose titles
share a first word or an arc name does not. Groups keep the order of
their first file. Every board has "Auto-group" (regroup all its files that way) and
"Split all" (one file per card); both drop the manual edits of the groups. Drag a
file onto another card to merge. `−` takes a file out (`findEjectTarget`): a card
of several files whose current regex (manual edit included) yields a valid episode
from it takes it, a one-file card with the same `nameShape` prefix takes it when the
pair's proposal is valid; among several, the regex with most literal characters
wins (a toast names the card); otherwise the file gets its own card. A manual edit of
the regex or rename sticks until "Re-propose". `groupByPrefix` (first five characters) stays exported, unused.

**The controls live in VMSAM, and the UI surfaces them as-is.** `POST /regex/` checks
the rule on the example file name (it must compile, match it and yield an episode ≥ 1),
for an update too; a new pattern is refused when another rule matches the example
(`Conflict with existing regex`). `POST /regex/replace/` checks the new pattern the same
way, refuses an unchanged or already existing pattern and an example another rule
matches.
`POST /episodes/rename/` checks the whole batch (name, extension, file on disk, target
free, no two episodes on one name) before renaming anything. `POST /fusion` answers
404 without a master episode, 409 for a job already queued, 503 when the internal
worker is unreachable. Every refusal carries VMSAM's `detail`: show it (toast, and on
the card or row concerned), never retry silently. `POST /regex/` lets a rule move to
another folder (the Rules tab), deliberately.

**Rules tab vs Bulk Regex.** The Rules tab never changes a regex: weight, folder and
rename pattern only. A regex is edited in Bulk Regex, against real files: "Edit regex"
on an Existing Rules row (or "Edit regex in Bulk Regex" in the Rules tab, which selects
the folder there) puts a **replace card** on the board (`board.addReplaceGroup`,
`group.replace`), holding the rule's current regex and the files of the watch folder
(top level) it already matches; "+ Add files" adds more from the file modal. The card
validates that the new regex yields a valid episode from **every** file of the card
(the real names are the control) and saves with `regex_replace`, `example_filename` =
its first file. Rename pattern and weight are read-only on it. A replace card stays on
the board without files but cannot be saved (it says why); "Auto-group" / "Split all"
leave it alone.

Every rule the UI sends must extract a valid episode number from every file of its
group and its rename pattern must contain `{<episode>}`: VMSAM refuses anything else
with a 400, and the UI refuses first.

## 🧩 Shared widgets (`app.js`, section 6, the end of section 4 and the top of section 1)

- **Header title** (`#title-home`, a `<button class="title-link">` inside the `<h1>`):
  `goToSectionHome()` opens the home tab of the current section (`SECTION_HOME`:
  Shows → `home-tab`, Movies → `movie-home-tab`); keyboard accessible as a button.
- **Home tiles**, both sections: `<div class="card home-tile text-center"
  data-tab="<tab id>" role="button" tabindex="0">` with no `onclick` and no button
  inside; a document listener in `app.js` opens `data-tab` on a click anywhere on the
  tile, Enter or Space; `style.css` gives the pointer, hover lift, focus ring.
- **`openDirectoryPicker({ title, hint, startPath, chooseLabel, hereLabel,
  allowCurrent(path), decorate(row, item), onNavigate(path), onPick(path) })`**: the
  directory browser over `CREATE_ROOT` in a large modal (the file modal's size), ids
  `dir-picker-*`: a click selects a row (Enter too, Enter again chooses), "Open →",
  a double click or → goes inside, ".." goes up, the name filter (`applyDirFilter`) is
  kept while navigating; "Choose" picks the selected row, the optional `hereLabel`
  button the directory shown; Escape / ✕ / Cancel / a click outside close without
  picking. Used by Index Folder ("Select a folder…", `registered` marks), Move Folder and
  Folder Settings ("Browse…", `this folder` mark, `<dir>/<folder name>` proposed). No
  directory browser is inline any more in those tabs.

- **`confirmDialog(message, { title, okText, cancelText })`**: the in-page, themed
  replacement of `window.confirm()`, a Promise of `true` / `false`. OK has the focus
  (Enter = OK), Escape or a click outside = Cancel, Tab cycles between the two buttons.
  The message is plain text with its line breaks kept (file names carry brackets).
  Every action that writes to disk or deletes asks through it (move a folder, rename
  episodes, move or delete a rule, delete a rejected file).
- **`createFolderPicker(prefix, onSelect, onClear)`**: search box + dropdown over the
  folders VMSAM knows (`fetchFolders`, cached in `folderCache`: set it to `null` after
  a change so the pickers reload). Ids `<prefix>-folder-search`, `<prefix>-folder-list`,
  `<prefix>-selected-folder`; `picker.select(folder)` picks one from code.
- **Directory filters** (`applyDirFilter`): the `.dir-filter` input above a directory
  browser (the directory picker, the Movies browsers) hides the listed directories
  whose name does not contain the text (case-insensitive, no regex); the text is kept
  while navigating and a note counts the hidden directories.
- **`showRemovedDirectories(box, removed, root)`** lists the directories a move deleted,
  relative to the root (Move Folder, Folder Settings); **`syncFolderInTabs(folder,
  except)`** re-selects a changed folder in the other Shows tabs that have it.
- **Files under existing rules** (`findCoveringRule`, `showRuleFiles`): in Bulk Regex
  and Index Folder, a file one of the folder's rules already matches (valid episode) is
  listed under that rule in the "Existing Rules" panel instead of getting a new card,
  with its extracted episode and a "Propose a new rule anyway" button that puts it on
  the board. After a save the panel reloads, and a file no rule matches any more goes
  back to the board.

## 🧪 Testing

Two things are worth checking on every change, and neither needs the real library:

- **the engine** — a page that loads `app.js`, runs `vmsamEngine.analyzeGroup` on a
  table of invented names with the expected episode per name, and prints failures.
  Run it in headless Chromium (`--dump-dom`); keep the table outside the repository.
- **the tabs against a local VMSAM API** — VMSAM's `gestionar_show.api` runs from a
  Python venv with a sqlite `DATABASE_URL` and a temporary library tree; point the
  server at it with `VMSAM_API_HOST`, `CREATE_ROOT`, `FILES_ROOT`. Every proxied route
  must answer; the tabs can be driven by a copy of `index.html` with a script
  appended, in headless Chromium. `--screenshot` and `--dump-dom` in one invocation
  load the page twice: use one or the other.
  `GET` / `POST /fusion` go through VMSAM's internal worker (`127.0.0.1:42085`, not
  started by `gestionar_show.api`): without it VMSAM answers 503, which the Errors tab
  shows as-is; a small stand-in answering `/internal/fusion` there exercises the queue.

Never POST to a production VMSAM instance to test: `/folder/`, `/regex/`,
`/regex/replace/`, `/regex/delete/`, `/incrementaller/` write to its database,
`/index_folder/`, `/episodes/rename/` and `/folders/move/` rename or move files,
`/errors/delete/` deletes files and `/fusion` queues a merge.

## ⚙️ Runtime

| variable | default | meaning |
| :--- | :--- | :--- |
| `PORT` | `3000` | listen port |
| `VMSAM_API_HOST` | `http://vmsam-api:8000` | the VMSAM public API (production uses port 8080) |
| `CREATE_ROOT` | `/srv/media` | library root, see the rule above |
| `FILES_ROOT` | `/srv/downloads` | the folder VMSAM watches; the file modal lists its top level only |
| `RUST_LOG` | `poem=debug` | tracing filter |
