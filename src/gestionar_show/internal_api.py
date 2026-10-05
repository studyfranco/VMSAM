"""Internal merge worker instance.

A single uvicorn process bound to 127.0.0.1 that owns the only merge queue and
worker thread. The public API validates requests and forwards them here. It
runs with workers=1 so one process holds the merge runtime state.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sys import stderr

import tools

from . import fusion
from .settings import Settings

# False until the worker has started; requests are then refused with the reason.
fusion_enabled = False


class InternalFusionRequest(BaseModel):
    """Request body to queue a merge retry for an error file."""
    error_file_path: str

app = FastAPI(
    title="Gestionar Show Internal Fusion Worker",
    description="Instance interne dédiée à l'exécution séquentielle des fusions",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    """Load the merge runtime and start the worker unless test output is unconfigured."""
    global fusion_enabled
    settings = Settings()
    # A forkserver-started process inherits nothing, so rebuild the runtime state.
    tools.load_merge_runtime_from_env()

    # In test mode without VMSAM_TEST_OUTPUT_DIR a merge has nowhere to write; keep the
    # worker off but serve /internal/health so it can report why.
    if fusion.is_test_mode() and (not len(fusion.get_test_output_dir())):
        fusion_enabled = False
        stderr.write("VMSAM_TEST_OUTPUT_DIR is not set: the internal fusion endpoint stays disabled, "
                     "the rest of the internal instance starts normally\n")
        return

    fusion_enabled = True
    # The child process opens its own session from the URL: a sessionmaker does not survive fork.
    fusion.start_worker(settings.DATABASE_URL)

@app.on_event("shutdown")
def on_shutdown():
    """Stop the worker thread."""
    fusion.stop_worker()

@app.get("/internal/health")
def get_internal_health():
    """Report worker state; read by the public GET /health."""
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
    """Return the worker and in-memory queue status."""
    return fusion.get_fusion_status()

@app.post("/internal/fusion")
def create_internal_fusion_job(fusion_request: InternalFusionRequest):
    """Queue a merge already validated by the public instance.

    No database check here; the worker resolves everything when the job runs.
    """
    # The test output directory is checked here, where the worker runs.
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
