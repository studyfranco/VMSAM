# `gestionar_show` — API reference

**This subsystem is frozen** (`WRITE_ZONES.MD` §1). The document lives here so the
description sits beside the code it describes; it is read, not edited.

## Routes

| instance | method | route |
| :--- | :--- | :--- |
| public | POST | `/folders/`, `/regex/`, `/incrementaller/`, `/fusion` |
| public | GET | `/folders_list/`, `/folders_infos/`, `/regex_folder/`, `/errors`, `/fusion`, `/health` |
| internal | GET | `/internal/health`, `/internal/fusion` |
| internal | POST | `/internal/fusion` |

`GET /errors` is how an agent obtains the path of a file in error. **Ask it; never
write a path down** — see `AGENT.MD`, Privacy.

## Architecture
Two uvicorn instances, both forked by `main_gestionar_show.py`:

| instance | bind | workers | owns |
| :--- | :--- | :--- | :--- |
| public | `0.0.0.0:8080` | 5 | external routes, database validation. Stateless. |
| internal | `127.0.0.1:42085` | **1** | the fusion queue, the worker, execution settings |

`workers=1` on the internal instance is **mandatory**: the queue lives in process
memory, and at that setting uvicorn serves in-process so the app inherits the forked
`tools` globals. Raising it splits the queue *and* empties `tools.software`.

The public app must not import `fusion`, directly or transitively — it would pull
`mergeVideo` → `audioCorrelation` → numpy/scipy into all five workers. Verify with an
import-closure walk, not by grepping one file: the first violation hid one module
deep, inside the proxy client.

The proxy uses stdlib `urllib` with an **empty `ProxyHandler`** — the image ships no
HTTP client library, and bare `urlopen` honours `http_proxy` even for loopback.


## Execution mode

`POST /fusion` is queued, never synchronous. The worker runs one job at a time
through a `ProcessPoolExecutor(max_workers=1)`, and takes the episode lock blocking
— see `AGENT.MD`, Concurrency.

`VMSAM_MODE` decides what a finished job does with its output: only the exact string
`production` replaces the library file. Everything else, including an empty value,
writes to `VMSAM_TEST_OUTPUT_DIR` and touches nothing.
