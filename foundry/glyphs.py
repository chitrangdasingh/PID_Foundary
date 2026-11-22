"""
glyphs.py - text detection for drawings whose lettering was exploded into vector strokes (PDF imports).

1. A candidate is a glyph when it is letter-sized, made only of open strokes, with no circle / fill.
2. Glyphs are chained into words along a baseline (horizontal, or vertical for rotated labels).
3. Each word is rendered and read by Tesseract (single-line mode); vertical words are rotated first.
Words with a plausible reading become text spans (source 'ocr_glyph'); the rest go back to symbol detection.
"""
import re

import cv2
import numpy as np

from .bubbles import ocr_image
from .render import render

WORD_OK = re.compile(r"^[A-Z0-9][A-Z0-9\-/\.\"#()]*$")


def is_glyph(c):
    b = c["bbox"]; w, h = b[2] - b[0], b[3] - b[1]
    if max(w, h) > 2.7 or max(w, h) < 0.5:
        return False
    for m in c["members"]:
        if m["type"] not in ("LWPOLYLINE", "LINE", "ARC") or m.get("closed"):
            return False
    if len(c["members"]) == 1 and len(c["members"][0].get("pts", [])) == 2 and max(w, h) > 1.4:
        return False          # a lone dash of a dashed line
    return True


def chain_words(glyphs):
    """Returns list of words: dict(members, bbox, vertical)."""
    used = [False] * len(glyphs)
    words = []
    order = sorted(range(len(glyphs)), key=lambda i: (glyphs[i]["bbox"][0], glyphs[i]["bbox"][1]))
    for i in order:
        if used[i]:
            continue
        used[i] = True
        cur = [i]
        b = list(glyphs[i]["bbox"])
        gw, gh = b[2] - b[0], b[3] - b[1]
        vertical = gw > gh * 1.15 and gh < 1.6       # a rotated letter is wider than tall
        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                o = glyphs[j]["bbox"]
                if not vertical:
                    h = max(b[3] - b[1], 0.6)
                    same_row = abs((o[1] + o[3]) / 2 - (b[1] + b[3]) / 2) < 0.45 * h and abs((o[3] - o[1]) - h) < 0.7 * h
                    gap = max(o[0] - b[2], b[0] - o[2])
                    if same_row and gap < 0.75 * h:
                        b = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]; cur.append(j); used[j] = True; changed = True
                else:
                    w = max(b[2] - b[0], 0.6)
                    same_col = abs((o[0] + o[2]) / 2 - (b[0] + b[2]) / 2) < 0.45 * w
                    gap = max(o[1] - b[3], b[1] - o[3])
                    if same_col and gap < 0.75 * w:
                        b = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]; cur.append(j); used[j] = True; changed = True
        words.append(dict(idx=cur, members=[m for k in cur for m in glyphs[k]["members"]], bbox=tuple(b), vertical=vertical))
    return words


def read_word(word, px=None):
    b = word["bbox"]
    h = (b[2] - b[0]) if word["vertical"] else (b[3] - b[1])
    px = px or 40.0 / max(h, 0.5)
    img = render(word["members"], b, px_per_unit=px, pad=0.3, thickness=2)
    if word["vertical"]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    txt = ocr_image(img, psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/.\"#()")
    if word["vertical"] and not WORD_OK.match(txt or ""):
        txt2 = ocr_image(cv2.rotate(img, cv2.ROTATE_180), psm=7)
        if WORD_OK.match(txt2 or ""):
            txt = txt2
    return (txt or "").replace(" ", "")


def detect(cands, ocr=True):
    """cands: vector candidates. Returns (words, remaining_candidates)."""
    gl, rest = [], []
    for c in cands:
        (gl if is_glyph(c) else rest).append(c)
    words = chain_words(gl)
    out, back = [], []
    for w in words:
        t = read_word(w) if ocr else ""
        if t and WORD_OK.match(t) and (len(t) >= 2 or len(w["idx"]) == 1):
            w["text"] = t
            out.append(w)
        else:
            back.extend(gl[k] for k in w["idx"])
    return out, rest + back
