"""Retrain: python scripts/train.py [--algo rf|svm]. Harvests approved reviews first (active learning)."""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foundry import store as S, training as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--algo", default="rf")
a = ap.parse_args()
st = S.Store(os.path.join(ROOT, "data", "foundry.db"))
print("harvested", T.harvest_approved(st))
mid, m = T.train_symbol_model(st, algo=a.algo)
print("model", mid, {k: v for k, v in m.items() if k != "per_class"})
