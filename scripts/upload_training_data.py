"""Funnel 2 - batch upload labelled crops from any labelling tool.

  python scripts/upload_training_data.py <folder>          # <folder>/<Category>__<Sub>__<Name>/*.png
  python scripts/upload_training_data.py labels.csv        # columns: path,label   (label = Category/Sub/Name)
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foundry import store as S, training as T  # noqa: E402

src = sys.argv[1]
st = S.Store(os.path.join(ROOT, "data", "foundry.db"))
n = 0
if src.endswith(".csv"):
    for r in csv.DictReader(open(src)):
        T.add_upload(st, r["label"], open(r["path"], "rb").read(), r["path"]); n += 1
else:
    for d in sorted(os.listdir(src)):
        lab = d.replace("__", "/")
        for f in sorted(os.listdir(os.path.join(src, d))):
            if f.lower().endswith(".png"):
                T.add_upload(st, lab, open(os.path.join(src, d, f), "rb").read(), f"{d}/{f}"); n += 1
print("uploaded", n)
