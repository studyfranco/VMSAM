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
| Folder Creation | browses `CREATE_ROOT`, creates a library folder | `POST /folders/` |
| Bulk Regex | one rule per group of downloaded files (grouped by name, regroupable by drag, "Auto-group", "Split all") | `POST /regex/`, `GET /regex_folder/` |
| Index Folder | registers an existing library directory, hides files already indexed, proposes a rule per group, then renames and indexes | `POST /folders/`, `GET /episodes_folder/`, `GET /regex_folder/`, `POST /regex/`, `POST /index_folder/` |
| Move Folder | moves/renames a registered folder: destination typed relative to `CREATE_ROOT` or picked in a directory browser (a click proposes `<dir>/<folder name>`), live old → new preview; VMSAM moves the directory, rewrites the folder and episode paths and deletes the old parent directories left empty (`removed_directories`, shown relative to the root) | `POST /folders/move/`, `GET /episodes_folder/` |
| Episodes | lists the registered episodes of a folder (number, file name, weight); renames one inline (the extension stays apart, visible), or a batch: ticked episodes, a pattern holding `{<episode>}` (prefilled from the folder's rule rename patterns, or proposed from the names), live old → new preview with the episode on two digits as VMSAM writes it, the file's own extension when the pattern has none, confirmation, one call. VMSAM renames all or nothing; no rule is stored | `GET /episodes_folder/`, `GET /regex_folder/`, `POST /episodes/rename/` |
| Folder Settings | original language, number cut, max episode, cut ratio of a folder, current values shown, only the changed fields sent; the path is read-only, with a button to the Move Folder tab | `GET /folder/`, `POST /folders/update/` |
| Rules | every rule of a folder, editable: regex, rename pattern (empty = none, sent as `null`), weight, folder (dropdown of all folders, a move is confirmed first); the example file name VMSAM requires is typed or prefilled (the rename pattern with episode 01 when the regex matches it, else a name built from the regex, `sampleFromRegex`) and checked with `testPattern` before sending; delete after confirmation | `GET /regex_folder/`, `POST /regex/update/`, `POST /regex/delete/` |
| Errors | rejected files (`incompatible_files`) grouped by folder, with the registered episode they failed to merge with (the master); "Relaunch fusion" queues a merge, "Delete" removes the file, its logs and its error sub-folder once no video remains (the removed directories are listed). A Fusion queue panel polls `GET /fusion` every 5 s while the tab is shown (the timer stops itself once the tab is hidden) | `GET /errors`, `GET /folders_list/`, `GET /episodes_folder/`, `POST /fusion`, `GET /fusion`, `POST /errors/delete/` |
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
`api.rs`; `movies` and `folders_move` take paths relative to `CREATE_ROOT`, the relay
prefixes it and refuses `..`). A movie is one kept file (a film or a show special):
`<destination_path>/<title><suffix>.<ext>`, several movies per directory allowed. Its
rules have **no episode group**: they only have to match the downloaded names.

| tab (`movie-…-tab`) | what it does | movie routes |
| :--- | :--- | :--- |
| Home | cards to every movie tab | — |
| Movie Creation | directory browser over `CREATE_ROOT` with the name filter (a sub-path that does not exist may be typed), title, suffix, options; the **metadata generator**: one checkbox per app and one input per provider from `metadata_providers`, an input enabled only when a ticked app knows the provider, a chip per app (`plex: tmdb` / `jellyfin: unsupported`); the tag string is built by VMSAM (`metadata_tags`, never guessed by the UI) and can become the suffix, be appended to it, or to the last directory of the destination; preview `<dir>/<title><suffix>.<ext>`; every non-empty id is stored as `metadata_ids` | `GET metadata_providers`, `POST metadata_tags`, `POST movies` |
| Bulk Regex | files from the file modal, grouped with `groupByNamePattern`; a file an existing movie rule matches is listed apart under that rule ("Propose a new rule anyway"). One rule per group, `movieRegexFor`: one file → its escaped name anchored, several → `buildRegex(buildTemplate(names), null)` (no episode), the names as alternatives if the template missed one; checked in the browser on every file (`checkMovieRegex`). Each card links to a movie (searchable picker) or "+ Create a movie" (Movie Creation with a title guessed from the name, the new movie linked back to the card), weight, the movie's existing rules with the selected files they already match. A refusal (a rule of another movie matches the example, or a show rule takes it as an episode) is shown as VMSAM wrote it, on the card | `GET regex_list`, `GET movies_list`, `GET regex_movie`, `POST regex` |
| Index | a movie (picker) or a directory (`folders_list`) then one of its movies; the directory's files with their state (registered file of this movie / of another one / matched by a rule / no rule); "+ Rule" opens an inline rule form for a file; "Index movie now" → report: indexed (old → new name), skipped with VMSAM's reason, unmatched, already registered | `GET movie`, `GET regex_movie`, `/api/fs/list`, `POST regex`, `POST index_movie` |
| Move Folder | a directory holding movies (`folders_list`), the movies moved with it, a `CREATE_ROOT` browser with filter (a click proposes `<dir>/<name>`, "Move here"), live preview, `confirmDialog`; report: moved movies with their new directory, deleted empty directories | `GET folders_list`, `POST folders_move` |
| Movies | table (id, title, suffix, directory, file, weight, metadata ids) with a search over all of them; edit (title / suffix rename the registered file on disk, options, metadata ids with the generator; only changed fields sent), delete after confirmation (VMSAM refuses a movie that still has rules or incompatible files: its reason is shown) | `GET movies_list`, `POST movies_update`, `POST movies_delete` |
| Rules | every movie rule with a filter and a "test a file name" box marking the rules that match it; inline edit (pattern, weight, movie, the example file name VMSAM requires, checked first), delete after confirmation | `GET regex_list`, `POST regex_update`, `POST regex_delete` |
| Errors | incompatible files grouped by movie, with the movie's registered file; "Relaunch fusion", "Delete" (confirmed; deleted files and removed directories reported); fusion queue panel polling every 5 s while the tab is shown | `GET errors`, `POST fusion`, `GET fusion_status`, `POST errors_delete` |

`movie.js` shares the global scope of `app.js` (two classic scripts): every top-level
name it declares carries `movie` / `Movie` (`movieRegexTab`, `createMovieSearchPicker`…),
since a `let`/`const` declared twice breaks the whole page. Its styles are in
`movie.css`.

## 🗺️ Files

| file | role |
| :--- | :--- |
| `src/main.rs` | poem server: `/api` routes, static files, CORS, `PORT` |
| `src/api.rs` | `/api/fs/list` (directory listing under a root), `/api/config`, `/api/vmsam/folders`, and the generic `/api/vmsam/<name>` relay driven by `GET_ROUTES` / `POST_ROUTES` |
| `static/index.html` | the section switch, both tab navs, every tab section (home included) and the shared file modal; every element the JS looks up by id lives here |
| `static/app.js` | one plain script: API client, **filename analysis engine** (pure functions), section/tab switching, shared widgets, group board, one init function per Shows tab |
| `static/movie.js` | the Movies tabs, loaded after `app.js`: movie API client, search picker, directory browser, metadata generator, one init per tab |
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
   separators merged.

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

**The controls live in VMSAM, and the UI surfaces them as-is.** `POST /regex/update/`
checks the edited rule on the example file name (it must compile, match it and yield
an episode ≥ 1) and, when the regex itself changes, refuses a pattern that already
exists or an example another rule matches (`Conflict with existing regex`).
`POST /episodes/rename/` checks the whole batch (name, extension, file on disk, target
free, no two episodes on one name) before renaming anything. `POST /fusion` answers
404 without a master episode, 409 for a job already queued, 503 when the internal
worker is unreachable. Every refusal carries VMSAM's `detail`: show it (toast, and on
the card or row concerned), never retry silently. `POST /regex/` and
`POST /regex/update/` let a rule move to another folder, deliberately.

Every rule the UI sends must extract a valid episode number from every file of its
group and its rename pattern must contain `{<episode>}`: VMSAM refuses anything else
with a 400, and the UI refuses first.

## 🧩 Shared widgets (`app.js`, section 6 and the top of section 1)

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
  browser (Index Folder "Register an existing folder", Move Folder) hides the listed
  directories whose name does not contain the text (case-insensitive, no regex); the
  text is kept while navigating and a note counts the hidden directories.
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

Never POST to a production VMSAM instance to test: `/regex/`, `/regex/update/`,
`/regex/delete/`, `/folders/update/`, `/incrementaller/` write to its database,
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
