# `gestionar_show` — API reference

**This subsystem is frozen** (`WRITE_ZONES.MD` §1). The document lives here so the
description sits beside the code it describes; it is read, not edited.

## Routes

Every route answers JSON; an error is an HTTP status with `{"detail": "..."}`.
A folder is always described by `folder_to_infos`:
`{folder_id, destination_path, original_language, number_cut,
cut_file_to_get_delay_second_method, max_episode_number}`.

### Public instance

| method | route | body / query | does |
| :--- | :--- | :--- | :--- |
| POST | `/folder/` | `{destination_path, original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number}` | creates the folder (and its directory), or updates the parameters of the folder registered at that path. Full data every time |
| GET | `/folder/` | `?folder_id=` | one folder |
| GET | `/folders/` | — | `{folders: [...]}`, 404 when there is none |
| GET | `/folders/destination_like/` | `?destination_like=` | folders whose path contains the text |
| POST | `/folders/move/` | `{folder_id, new_destination_path}` | moves the directory on disk, rewrites the folder and episode paths, removes the old parent if left empty. `removed_directories` in the answer |
| POST | `/regex/` | `{regex_pattern, rename_pattern, weight, example_filename, destination_path}` | creates the regex, or updates the folder, rename pattern and weight of an existing one. `example_filename` must match and give an episode ≥ 1 in both cases |
| POST | `/regex/replace/` | `{regex_pattern, new_regex_pattern, example_filename, rename_pattern?, weight?, folder_id?}` | changes the pattern (the primary key): new row and old row deleted in one commit. Absent fields keep their value; `rename_pattern: null` removes it |
| POST | `/regex/delete/` | `{regex_pattern}` | deletes a regex |
| GET | `/regex_folder/` | `?folder_id=` | the regexes of a folder |
| POST | `/incrementaller/` | `{regex_pattern, rename_pattern, episode_incremental, example_filename}` | creates or updates an increment rule |
| GET | `/incrementaller_list/` | — | every increment rule |
| POST | `/index_folder/` | `{folder_id}` | renames and registers the files a folder already holds, through its regexes. A file whose episode is already registered is skipped; nothing is merged |
| GET | `/episodes_folder/` | `?folder_id=` | the registered episodes of a folder; an empty list is a 200 |
| POST | `/episodes/rename/` | `{folder_id, renames: [{episode_number, new_file_name}]}` | renames episode files in place, all or nothing. No regex is stored |
| GET | `/errors` | — | the incompatible files |
| POST | `/errors/delete/` | `{error_file_path}` | deletes the error file, its `.log.error` and its row; the error directory is removed when only `.error` files remain in it |
| POST | `/fusion` | `{error_file_path}` | queues a fusion of an incompatible file with its master episode (relayed to the internal instance) |
| GET | `/fusion` | — | the fusion queue (relayed) |
| GET | `/health` | — | version, mode, dev flag, internal instance state |

### Internal instance

| method | route | does |
| :--- | :--- | :--- |
| GET | `/internal/health` | worker state, queue length, `fusion_enabled` |
| GET | `/internal/fusion` | the queue |
| POST | `/internal/fusion` | queues a job already validated by the public instance |

`GET /errors` is how an agent obtains the path of a file in error. **Ask it; never
write a path down** — see `AGENT.MD`, Privacy.

## Architecture
`main_gestionar.py` forks four uvicorn instances: the two show instances below
(`main_gestionar_show.run_uvicorn_public` / `run_uvicorn_internal`) and the two movie
instances described in `src/gestionar_movie/API_REFERENCE.MD` (public `0.0.0.0:8081`,
internal `127.0.0.1:42086`).

| instance | bind | workers | owns |
| :--- | :--- | :--- | :--- |
| public | `0.0.0.0:8080` | 5 | external routes, database validation. Stateless. |
| internal | `127.0.0.1:42085` | **1** | the fusion queue, the worker, execution settings |

Each round of the watch loop: wait for the database (`gestionar_common.database.wait_for_database`),
then the movie integrator, then the incrementaller, then the show integrator. A file a
movie regex takes never reaches the show regexes.

`workers=1` on the internal instance is **mandatory**: the queue lives in process
memory, and at that setting uvicorn serves in-process so the app inherits the forked
`tools` globals. Raising it splits the queue *and* empties `tools.software`.

The public app must not import `fusion`, directly or transitively — it would pull
`mergeVideo` → `audioCorrelation` → numpy/scipy into all five workers. Verify with an
import-closure walk, not by grepping one file: the first violation hid one module
deep, inside the proxy client.

The proxy uses stdlib `urllib` with an **empty `ProxyHandler`** — the image ships no
HTTP client library, and bare `urlopen` honours `http_proxy` even for loopback.

Shared code lives in `gestionar_common/`: `settings.py` (`DATABASE_URL` for every
instance) and `database.py` (connection arguments, `wait_for_database`,
`is_database_unavailable`).

## Execution mode

`POST /fusion` is queued, never synchronous. The worker runs one job at a time
through a `ProcessPoolExecutor(max_workers=1)`, and takes the episode lock blocking
— see `AGENT.MD`, Concurrency. Before each job it waits for the database; a job that
loses the database before the merge starts is replayed, up to 10 times.

`VMSAM_MODE` decides what a finished job does with its output: only the exact string
`production` replaces the library file. Everything else, including an empty value,
writes to `VMSAM_TEST_OUTPUT_DIR` and touches nothing.
