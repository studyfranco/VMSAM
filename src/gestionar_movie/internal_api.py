"""Internal fusion worker instance of the movie side.

Single uvicorn process bound to 127.0.0.1:42086. It owns the only movie fusion
queue and worker thread; the public movie API validates a request, then
forwards it here. workers must stay at 1: the queue lives in this process'
memory.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sys import stderr

import tools

from . import fusion
from gestionar_common.settings import Settings

# False while the worker could not start: a fusion is then refused with its
# reason, instead of accepting jobs nobody will consume.
fusion_enabled = False


class InternalFusionRequest(BaseModel):
    error_file_path: str

app = FastAPI(
    title="Gestionar Movie Internal Fusion Worker",
    description="Internal instance running the movie fusions one at a time",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    global fusion_enabled
    settings = Settings()
    # Rebuild the merge runtime state whatever the start method: under
    # forkserver (default since Python 3.14) nothing is inherited, and the
    # failure would be silent (locks in another tree, empty tools.software).
    tools.load_merge_runtime_from_env()

    if fusion.is_test_mode() and (not len(fusion.get_test_output_dir())):
        fusion_enabled = False
        stderr.write("VMSAM_TEST_OUTPUT_DIR is not set: the internal movie fusion endpoint stays disabled, "
                     "the rest of the internal movie instance starts normally\n")
        return

    fusion_enabled = True
    fusion.start_worker(settings.DATABASE_URL)

@app.on_event("shutdown")
def on_shutdown():
    fusion.stop_worker()

@app.get("/internal/health")
def get_internal_health():
    """Worker state, read by the public GET /health"""
    return {
        "status": "ok",
        "git_commit": tools.get_git_commit(),
        "mode": tools.get_execution_mode(),
        "is_running": fusion.is_job_running(),
        "queue_length": fusion.get_fusion_status()["queue_length"],
        "fusion_enabled": fusion_enabled
    }

@app.get("/internal/fusion")
def get_internal_fusion_status():
    """State of the sequential worker and of its in-memory queue"""
    return fusion.get_fusion_status()

@app.post("/internal/fusion")
def create_internal_fusion_job(fusion_request: InternalFusionRequest):
    """Queue a fusion already validated by the public movie instance.

    No database check here: the worker resolves everything again when the job runs.
    """
    if not fusion_enabled:
        raise HTTPException(status_code=503, detail="Fusion is disabled: VMSAM_TEST_OUTPUT_DIR is not set in test mode")

    try:
        position = fusion.enqueue_fusion_job(fusion_request.error_file_path)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "message": "Fusion job queued",
        "error_file_path": fusion_request.error_file_path,
        "mode": tools.get_execution_mode(),
        "queue_position": position
    }
