"""
raster.py - the image path, for scanned P&IDs and PDF pages (the route the Azure / AWS samples take).

  page image -> binarise -> text detection (Tesseract word boxes) -> symbol candidates (line removal + connected
  components) -> classification (same symbol model) -> line detection (Hough transform on the line mask) -> graph.
Outputs land in the same tables and folders as the DXF path, with pixel coordinates (y up) and sheet = page.
"""
import csv
import io
import json
import os
import shutil
import subprocess

import cv2
import numpy as np

from . import graph as GR, symbols as Y, text as TX, training as TR


def pdf_to_images(pdf, out_dir, dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm (poppler-utils) is needed for PDF input; upload PNG pages instead")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", pdf, os.path.join(out_dir, "page")], check=True)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))


def ocr_words(img):
    if not shutil.which("tesseract"):
        return []
    tmp = "/tmp/_pid_page.png"
    cv2.imwrite(tmp, img)
    out = subprocess.run(["tesseract", tmp, "stdout", "--psm", "11", "tsv"], capture_output=True, text=True).stdout
    words = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t"):
        t = (r.get("text") or "").strip()
        try:
            conf = float(r.get("conf", -1))
        except ValueError:
            conf = -1
        if t and conf > 40:
            x, y, w, h = int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"])
            words.append(dict(text=t.upper(), box=(x, y, w, h), conf=conf / 100))
    return words


def process_page(img_gray, clf, page_no, px=1.0):
    H, W = img_gray.shape
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    words = ocr_words(bw)
    clean = bw.copy()
    for w in words:
        x, y, ww, hh = w["box"]
        if hh < H * 0.05:
            clean[y:y + hh, x:x + ww] = 255
    scale = max(H, W) / 2400.0
    cands, _ = Y.candidates(clean, px=6 * scale)
    syms = []
    for i, b in enumerate(cands):
        crop = Y.crop(clean, b, px=6 * scale)
        if crop.size == 0:
            continue
        P = clf.predict_proba([Y.features(Y.square(crop))])[0] if clf is not None else None
        if P is None:
            continue
        j = int(P.argmax()); cls = str(clf.classes_[j])
        if cls.split("/")[0] in ("Text", "Line"):
            continue
        x, y, w, h = b
        syms.append(dict(symbol_id=f"S{page_no}-{i:03d}", sheet_no=page_no, cls=cls, category=cls.split("/")[0], confidence=round(float(P[j]), 3),
                         bbox=[x, H - (y + h), x + w, H - y], tag=None, method="raster-model", props={}, handles=[]))
    ink = (clean < 128).astype(np.uint8) * 255
    k = max(15, int(20 * scale))
    mask = cv2.bitwise_or(cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))),
                          cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))))
    hl = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40, minLineLength=k, maxLineGap=int(4 * scale) + 2)
    segs = []
    for l in (hl if hl is not None else []):
        x1, y1, x2, y2 = map(int, l[0])
        segs.append(dict(p=(x1, H - y1), q=(x2, H - y2), len=float(np.hypot(x2 - x1, y2 - y1)), handle=None, style="solid", dash=False))
    texts = [dict(text=w["text"], bbox=[w["box"][0], H - w["box"][1] - w["box"][3], w["box"][0] + w["box"][2], H - w["box"][1]],
                  label=TX.rule_label(w["text"]), confidence=w["conf"], sheet=page_no, source="ocr_raster") for w in words]
    edges, stats = GR.build([dict(id=s["symbol_id"], bbox=s["bbox"], kind="symbol") for s in syms], segs, snap=3 * scale, attach=6 * scale, sheet_no=page_no)
    return syms, texts, segs, edges, stats
