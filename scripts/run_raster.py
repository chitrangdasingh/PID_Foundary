"""Raster path for PDFs / scans: python scripts/run_raster.py drawing.pdf --pid CAT-PID"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foundry import raster as RA, store as S, training as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--pid", required=True)
a = ap.parse_args()
data = os.path.join(ROOT, "data")
st = S.Store(os.path.join(data, "foundry.db"))
mid, clf = T.load_active(st)
out = os.path.join(data, "inference-results", a.pid, "raster-pages")
pages = RA.pdf_to_images(a.path, out) if a.path.lower().endswith(".pdf") else [a.path]
res = []
for i, p in enumerate(pages, 1):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    syms, texts, segs, edges, stats = RA.process_page(img, clf, i)
    res.append(dict(page=i, symbols=syms, texts=texts, edges=edges, stats=stats))
    print(f"page {i}: {len(syms)} symbols, {len(texts)} words, {len(segs)} lines, {len(edges)} candidate edges")
json.dump(res, open(os.path.join(out, "response.json"), "w"), default=str)
