"""Optional REST service with the Azure design's endpoints: pip install fastapi uvicorn; uvicorn scripts.api:app"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foundry import pipeline as P, store as S  # noqa: E402

F = P.Foundry(S.Store(os.path.join(ROOT, "data", "foundry.db")), os.path.join(ROOT, "data"))
app = FastAPI(title="PID Foundry inference service")


@app.post("/pid/{pid_id}/symbol-detection")
def post_symbol_detection(pid_id: str, sheets: list[int] | None = None):
    return F.symbol_detection(pid_id, sheet_nos=sheets)


@app.post("/pid/{pid_id}/text-detection")
def post_text_detection(pid_id: str, corrected_symbols: dict | None = None):
    if not F.results(pid_id, "symbol_detection"):
        raise HTTPException(422, "symbol detection has not been run for this P&ID")
    return F.text_detection(pid_id, corrected_symbols)


@app.post("/pid/{pid_id}/graph-construction", status_code=202)
def post_graph_construction(pid_id: str, corrected_text: dict | None = None):
    if not F.results(pid_id, "text_detection"):
        raise HTTPException(422, "text detection has not been run for this P&ID")
    F.post_graph_construction(pid_id, corrected_text)
    return {"status": "Submitted"}


@app.get("/pid/{pid_id}/graph-construction/status")
def get_job_status(pid_id: str):
    s = F.job_status(pid_id)
    if s.get("status") == "NotFound":
        raise HTTPException(404, "no job")
    return s


@app.post("/pid/{pid_id}/graph-persistence", status_code=201)
def post_graph_persistence(pid_id: str, corrected_graph: dict | None = None):
    return F.graph_persistence(pid_id, corrected_graph)


@app.get("/pid/{pid_id}/results/{step}")
def get_results(pid_id: str, step: str):
    r = F.results(pid_id, step)
    if r is None:
        raise HTTPException(404, "no output for this step yet")
    return r


@app.get("/pid/{pid_id}/images/{step}/{idx}")
def get_image(pid_id: str, step: str, idx: int):
    imgs = F.images(pid_id, step)
    if idx >= len(imgs):
        raise HTTPException(404, "no image")
    return FileResponse(imgs[idx])
