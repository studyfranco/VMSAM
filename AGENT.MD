# VMSAM Web Interface — AI Agent Context & Directives

You are assisting with the `web-interface` branch of VMSAM: a small Rust server that
serves a static single-page UI and proxies it to the VMSAM `gestionar_show` API.
Read this before the first edit. The engine's own rules (`AGENT.MD` on `master`) do
not apply here, except the privacy rule, which applies everywhere.

## 🎯 What this is

VMSAM (`master`) watches a download folder, matches file names against per-folder
regex rules, renames and merges episodes into a library. Its API is a FastAPI app on
port 8080 with no UI. This branch is the UI:

| tab | what it does | VMSAM routes |
| :--- | :--- | :--- |
| Folder Creation | browses `CREATE_ROOT`, creates a library folder | `POST /folders/` |
| Bulk Regex | one rule per group of downloaded files (grouped by name, regroupable by drag, "Auto-group", "Split all") | `POST /regex/`, `GET /regex_folder/` |
| Index Folder | registers an existing library directory, hides files already indexed, proposes a rule per group, then renames and indexes | `POST /folders/`, `GET /episodes_folder/`, `GET /regex_folder/`, `POST /regex/`, `POST /index_folder/` |
| Move Folder | moves/renames a registered folder: destination typed relative to `CREATE_ROOT` or picked in a directory browser (a click proposes `<dir>/<folder name>`), live old → new preview; VMSAM moves the directory, rewrites the folder and episode paths and deletes the old parent directories left empty (`removed_directories`, shown relative to the root) | `POST /folders/move/`, `GET /episodes_folder/` |
| Specials | exact-name rename of a special (`destination_path` is a control: a new name an existing rule already catches must belong to that folder, and the answer carries that rule for editing), plus the rule that catches the new name | `POST /special/`, `GET /special_list/`, `POST /regex/` |
| Incrementaller | rules that shift an episode number before the folder rules run | `POST /incrementaller/`, `GET /incrementaller_list/` |

## 🗺️ Files

| file | role |
| :--- | :--- |
| `src/main.rs` | poem server: `/api` routes, static files, CORS, `PORT` |
| `src/api.rs` | `/api/fs/list` (directory listing under a root), `/api/config`, `/api/vmsam/folders`, and the generic `/api/vmsam/<name>` relay driven by `GET_ROUTES` / `POST_ROUTES` |
| `static/index.html` | the seven tabs (home included) and the shared file modal; every element the JS looks up by id lives here |
| `static/app.js` | one plain script: API client, **filename analysis engine** (pure functions), shared widgets, group board, one init function per tab |
| `static/style.css` | dark theme; every colour is a variable on `:root` |
| `Dockerfile`, `docker-compose.yml` | multi-stage build; runtime needs only `ca-certificates` |

## 📜 Rules

**No external resource, ever.** No CDN script, no font, no stylesheet, no analytics.
The page must work on a LAN with no internet. A `<script src=…>` or `<link>` that does
not point at `static/` is a defect, whatever it brings.

**Vanilla only.** Plain JS, CSS and HTML; no bundler, no framework, no npm. The JS is
one file loaded once; new functionality is a new section in `app.js`, not a new
toolchain. Build nodes with `createElement` / `textContent`: file names carry
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
   changes), and among equal alignments the one with fewest holes wins. A shared
   token stays literal; an aligned digit run whose value changes is `varying`;
   anything else that differs is `wild` (`optional` when empty in some name).
   `absorbBetweenWilds` merges a short literal run caught between two wild spans
   (shared title words) and a checksum glued to a wild word.
3. `findMarkerSpan` / `findEpisodeMarker`: when **every** name carries the same kind
   of season/episode marker — `S<n>E<n>`, `<n>x<nn>` (`3x01`, never `1920x1080`) or
   `Saison|Season <n> Épisode|Episode <n>` — its episode digits are the episode,
   unconditionally (+1000). Otherwise `scoreEpisodeSlot`
   scores each digit run: an `ep` word, a dash, zero padding and changing between
   files score up; resolution (`1080p`), codec (`x264`), version (`v2`), season
   (`S2`), year, CRC, brackets and the extension score down. Below −100 nothing is
   chosen and the user clicks a chip. A chip click always overrides.
4. `buildRegex` emits an anchored **Python** pattern: literals escaped, varying digits
   `\d+`, wild spans `.+` (`.*` when optional, lazy `.+?` right before the episode),
   the episode `(?P<episode>\d+)`. `toJsRegex` converts `(?P<` to `(?<` so the
   browser can check it on every file of the group.
5. `buildRename` is the first name with the episode replaced by `{<episode>}` and
   every wild span (per-episode titles, tags) dropped, separators merged.

Default grouping: **Bulk Regex and Index Folder group with `groupByNamePattern`**;
**Specials and Incrementaller start with one file per card**. A word is a run between
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
one (a special among numbered episodes) for its own card. Groups keep the order of
their first file. Every board has "Auto-group" (regroup all its files that way) and
"Split all" (one file per card); both drop the manual edits of the groups. Drag a
file onto another card to merge, `−` to give it back its own card. A manual edit of
the regex or rename sticks until "Re-propose". `groupByPrefix` (first five characters) stays exported, unused.

**The control around specials lives in VMSAM, and the UI surfaces it as-is.**
`POST /special/` takes the folder and refuses a new name that a regex of *another*
folder catches (wrong destination, or a cross-match between shows), and one that an
increment rule catches (the daemon would rename it again). It answers with the
heaviest regex of the folder that catches the new name, so the Specials tab edits
that rule instead of creating a second one. Every refusal is a 400 with the reason:
show it, never retry silently. `POST /regex/` has no such control, deliberately: a
rule may be moved to another folder.

Every rule the UI sends must extract a valid episode number from every file of its
group and its rename pattern must contain `{<episode>}`: VMSAM refuses anything else
with a 400, and the UI refuses first.

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

Never POST to a production VMSAM instance to test: `/regex/`, `/special/`,
`/incrementaller/` write to its database and `/index_folder/` renames files.

## ⚙️ Runtime

| variable | default | meaning |
| :--- | :--- | :--- |
| `PORT` | `3000` | listen port |
| `VMSAM_API_HOST` | `http://vmsam-api:8000` | the VMSAM public API (production uses port 8080) |
| `CREATE_ROOT` | `/srv/media` | library root, see the rule above |
| `FILES_ROOT` | `/srv/downloads` | the folder VMSAM watches; the file modal lists its top level only |
| `RUST_LOG` | `poem=debug` | tracing filter |
