"""Batch run: python scripts/run_pipeline.py data/sources/<drawing>.dxf --pid STR-PID-R5 [--bom data/seed/BOM.xlsx] [--sheets 3 4] [--train]"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foundry import pipeline as P, store as S, training as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dxf")
ap.add_argument("--pid")
ap.add_argument("--bom", default=os.path.join(ROOT, "data", "seed", "BOM_single_tube_reactor.xlsx"))
ap.add_argument("--sheets", type=int, nargs="*")
ap.add_argument("--train", action="store_true", help="(re)seed from the legend sheet and train a model first")
ap.add_argument("--algo", default="rf")
a = ap.parse_args()
data = os.path.join(ROOT, "data")
store = S.Store(os.path.join(data, "foundry.db"))
F = P.Foundry(store, data)
pid, sheets = F.ingest(a.dxf, a.pid)
print("ingested", pid, len(sheets), "sheets")
if a.train or not store.one("SELECT 1 FROM models WHERE task='symbol' AND active=1"):
    g = F.geometry(pid)
    leg = [s for s in g["sheets"] if s["kind"] == "legend"]
    if leg:
        print("seed samples:", T.seed_from_legend(store, g["ents"], leg[0], pid)[0])
    print("harvested approved:", T.harvest_approved(store))
    mid, m = T.train_symbol_model(store, algo=a.algo)
    print("model", mid, "hold-out acc", m["holdout_accuracy"], "macro F1", m["macro_f1"])
r = F.symbol_detection(pid, sheet_nos=a.sheets)
print("symbols", len(r["symbols"]))
print("texts", len(F.text_detection(pid)["texts"]))
r = F.graph_construction(pid)
print("lines", r["lines"], "edges", len(r["edges"]))
print(F.graph_persistence(pid, bom_path=a.bom if os.path.exists(a.bom) else None))
