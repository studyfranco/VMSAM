# `gestionar_movie` — API reference

A **movie** is any single file to keep: a film, or a standalone show episode. Its
`title` is its **name in the database**, unique (`The Rabbits Crusade` and `The Rabbits
Crusade extended` are two movies); it is not the file name. `destination_path` and
`suffix` are not unique. The file name comes from the regex that caught the file, as
for an episode: its `rename_pattern` with `{<suffix>}` replaced by the movie's suffix
(the pattern carries its extension), or the file's own name when the regex has none.

## Instances and ports

Started by `main_gestionar.py` through `main_gestionar_movie.py`, beside the two show
instances (public 8080, internal 42085). The ports are fixed literals, like the show's:

| instance | bind | workers | where the port is written |
| :--- | :--- | :--- | :--- |
| public (`api.py`) | `0.0.0.0:8081` | 5 | `main_gestionar_movie.run_uvicorn_public` |
| internal (`internal_api.py`) | `127.0.0.1:42086` | **1** (owns the queue) | `main_gestionar_movie.run_uvicorn_internal`, `internal_client.build_internal_url` |

The internal ports must differ (show 42085, movie 42086): all instances run in one
container. The public app imports neither the merge engine (`fusion`, `mergeVideo`,
numpy) nor `gestionar_show`; the show tables it cross-checks are read with plain SQL.

## Routes

Every route answers JSON; an error is an HTTP status with `{"detail": "..."}`. A movie
is always described by `movie_to_infos`: `{movie_id, title, destination_path, suffix,
original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids,
file_path, file_weight}`.

### Public instance

| method | route | body / query | does |
| :--- | :--- | :--- | :--- |
| POST | `/movie/` | `{title, destination_path, suffix="", original_language, number_cut=10, cut_file_to_get_delay_second_method=2, metadata_ids?}` | creates the movie (and its directory), or updates the movie with that title. Full data every time, **database only**: no file is moved or renamed, a new suffix only names the next file integrated or merged. A movie that has a file keeps its `destination_path` (400, use `/movie/move/`). Checks in `model.test_conformity_movie_param`. Answers `movie_to_infos` + `message` |
| POST | `/movie/move/` | `{movie_id, new_destination_path}` | moves the movie's file (same name) into the new directory, created if needed; refused when the target exists or is another movie's file; a failed commit moves it back. Removes the old directory when left empty and not another movie's destination nor a show folder. Answers `movie_to_infos` + `message`, `file_moved`, `removed_directories` |
| GET | `/movie/` | `?movie_id=` or `?title=` (exact) | one movie |
| GET | `/movies/` | — | `{movies: [...]}`, 404 when there is none |
| GET | `/movies/title_like/` | `?title_like=` | movies whose title contains the text, case-insensitive |
| GET | `/movies/destination_like/` | `?destination_like=` | movies whose destination contains the text |
| POST | `/movies/delete/` | `{movie_id}` | deletes a movie without regex nor incompatible file; its file stays on disk |
| POST | `/regex/` | `{regex_pattern, movie_id, rename_pattern?, weight=1, example_filename}` | creates the regex, or updates the movie, rename pattern and weight of an existing one. Must compile and match the example; no other movie regex may match it, no show regex may take it as an episode (the movies run first). A rename pattern holds `{<suffix>}`; empty means none. Answers `{message, regex_pattern, movie_id, rename_pattern, weight, final_file_name}`, the name the example would get |
| POST | `/regex/replace/` | `{regex_pattern, new_regex_pattern, example_filename, rename_pattern?, weight?, movie_id?}` | changes the pattern (the primary key): new row and old row deleted in one commit. Absent fields keep their value; `rename_pattern: null` removes it. Answers the `/regex/` fields + `previous_regex_pattern` |
| POST | `/regex/delete/` | `{regex_pattern}` | deletes a regex |
| GET | `/regex_movie/` | `?movie_id=` | the regexes of a movie |
| GET | `/regex_list/` | — | every movie regex with its rename pattern and movie (`title`, `suffix`, `destination_path`) |
| POST | `/index_movie/` | `{movie_id}` | registers one file already in the destination directory: the movie's regexes match it (heaviest regex, then largest file); renamed by its regex (or kept), nothing merged. Others listed in `skipped` |
| GET | `/folders/` | — | the destination directories with their number of movies |
| POST | `/folders/move/` | `{destination_path, new_destination_path}` | moves a whole directory and rewrites every movie inside it |
| GET | `/errors` | — | the incompatible files |
| POST | `/errors/delete/` | `{error_file_path}` | deletes the file, its `.log.error` and its row; the directory is removed when only `.error` files remain in it |
| POST / GET | `/fusion` | `{error_file_path}` / — | queues a fusion with the movie file / the queue (relayed to the internal instance) |
| GET | `/metadata_providers/` | — | the two metadata files as loaded |
| POST | `/metadata_tags/` | `{apps, ids}` | tags such as plex `{tmdb-1255}`, jellyfin `[tmdbid-1255]` |
| GET | `/health` | — | version, mode, dev flag, internal instance state |

### Internal instance

| method | route | does |
| :--- | :--- | :--- |
| GET | `/internal/health` | worker state, queue length, `fusion_enabled` |
| GET | `/internal/fusion` | the queue |
| POST | `/internal/fusion` | queues a job already validated by the public instance |

`GET /errors` is how an agent obtains the path of a file in error. **Ask it; never
write a path down** — see `AGENT.MD`, Privacy.

A movie regex is refused when a show regex would take its example as an episode: the
movie step runs first in each round. Metadata tags come from `src/metadata_providers.json`
and `src/metadata_brackets.json` (overridable with `VMSAM_METADATA_PROVIDERS` /
`VMSAM_METADATA_BRACKETS`); a malformed file stops the public instance at startup.

## Daemon and fusion

- Each round, files of the watch folder go to the first matching movie regex (heaviest
  first). A movie is one pool job taking the lock `movie_<id>` non-blocking. Files are
  taken heaviest first (weight, then size), with the show's `process_episode` rules:
  no file yet → moved to the name given by its regex and registered; a file exists →
  merged, the heavier as `forced_best_video` (the registered file wins a tie). If the
  registered file weighs at least as much, the result keeps its path; otherwise it
  takes the name given by the candidate's regex. A merge produces Matroska, so the
  result's name gets the `.mkv` extension; a previous file under another name is
  removed. A target name held by another file leaves the candidate in the watch folder.
- A failed merge sends the lighter file to `<error root>/<destination_path>/<title>/`,
  after trying it against the movie's existing incompatible files, like the show's
  `process_rejected_files`.
- `POST /fusion` is queued on the internal instance: one job at a time, waits for the
  database first, replays a job that lost it before its merge, takes the movie lock
  blocking. The result is named as in the daemon; the rejected file's regex is found
  by its weight (then by matching its base name), as in the show's
  `resolve_rename_pattern`. Temp trees: daemon `<tmp>/movie_<id>`, fusion
  `<tmp>/fusion_movie_<id>`. Only `VMSAM_MODE=production` replaces the movie file;
  anything else writes under `VMSAM_TEST_OUTPUT_DIR` and touches nothing (without it,
  fusions are refused with 503).
- Changing the suffix or a rename pattern never renames a file already in place.
