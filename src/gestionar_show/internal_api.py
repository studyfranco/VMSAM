"""Internal fusion worker instance.

Runs as a single uvicorn process bound to 127.0.0.1 and owns the only fusion
queue and worker thread in the deployment. It is never exposed outside the
container: the public API validates a request first, then forwards it here.

Running with workers=1 is what makes the merge engine usable at all. uvicorn
skips its multiprocess supervisor at that setting and serves in-process, so this
app inherits the fully initialised `tools` module from the fork done by
main_gestionar_show (software paths, merge rules, core count, temp folders).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sys import stderr

import tools

from . import fusion
from .settings import Settings

# Faux tant que le worker n'a pas pu demarrer: la fusion est alors refusee avec
# sa raison, au lieu d'accepter des jobs que personne ne consommera.
fusion_enabled = False


class InternalFusionRequest(BaseModel):
    error_file_path: str

app = FastAPI(
    title="Gestionar Show Internal Fusion Worker",
    description="Instance interne dédiée à l'exécution séquentielle des fusions",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    global fusion_enabled
    settings = Settings()
    # Reconstruire l'etat runtime des merges sans dependre de la methode de
    # demarrage: sous forkserver (defaut depuis Python 3.14) rien n'est herite,
    # et l'echec serait silencieux -- verrous d'episode dans un autre arbre,
    # tools.software vide.
    tools.load_merge_runtime_from_env()

    # En mode test, une fusion n'a nulle part ou deposer son resultat sans
    # VMSAM_TEST_OUTPUT_DIR. On refuse alors de demarrer le worker plutot que de
    # laisser chaque job echouer un par un apres avoir consomme un merge complet.
    # Le reste de l'instance demarre: /internal/health reste interrogeable et dit
    # pourquoi la fusion est indisponible.
    if fusion.is_test_mode() and (not len(fusion.get_test_output_dir())):
        fusion_enabled = False
        stderr.write("VMSAM_TEST_OUTPUT_DIR is not set: the internal fusion endpoint stays disabled, "
                     "the rest of the internal instance starts normally\n")
        return

    fusion_enabled = True
    # Le worker passe l'URL au process fils, qui ouvre sa propre session: un
    # sessionmaker ne survit pas proprement au fork.
    fusion.start_worker(settings.DATABASE_URL)

@app.on_event("shutdown")
def on_shutdown():
    fusion.stop_worker()

@app.get("/internal/health")
def get_internal_health():
    """Etat du worker, consommé par le GET /health public"""
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
    """Etat du worker séquentiel et de la file d'attente en mémoire"""
    return fusion.get_fusion_status()

@app.post("/internal/fusion")
def create_internal_fusion_job(fusion_request: InternalFusionRequest):
    """Met en file une fusion déjà validée par l'instance publique.

    Aucune revalidation en base ici: l'instance publique a déjà vérifié
    l'existence de l'entrée incompatible_files et de l'épisode maître, et le
    worker refera la résolution complète au moment de l'exécution.
    """
    # The worker is the authority on execution settings: the public instance only
    # screens out an unknown VMSAM_MODE, the test output directory is checked here.
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
