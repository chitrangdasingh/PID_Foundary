"""
bubbles.py - read ISA-5.1 instrument bubbles.

In this drawing (a PDF import) bubble letters are vector glyph outlines, not text, so the tag is read by
rendering the inside of each circle and running Tesseract OCR on it. Where MTEXT numbers sit inside the
bubble they are used directly (exact) and OCR only supplies the letters.
Bubble style tells the location (ISA-5.1): plain circle = field, circle in square = shared display (DCS/PLC),
horizontal bar = main panel / control room.
"""
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from .render import render, in_box

TAG_RE = re.compile(r"^([A-Z]{1,5})\s*-?\s*(\d{2,4}[A-Z]?)$")
LETTER_FIX = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
NUM_FIX = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
FIRST_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def has_tesseract():
    return shutil.which("tesseract") is not None


def ocr_image(img, psm=6, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"):
    if not has_tesseract():
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, img)
        try:
            out = subprocess.run(["tesseract", f.name, "stdout", "--psm", str(psm), "-c",
                                  "tessedit_char_whitelist=" + whitelist], capture_output=True, text=True, timeout=20)
            return out.stdout.strip()
        except Exception:
            return ""


def ocr_vote(glyphs, box, r, part, settings=((40, 3, 1), (24, 2, 2), (30, 3, 2), (40, 2, 1))):
    """Read the top (letters) or bottom (number) half of a bubble several ways and take the majority reading.
    The CAD stroke font makes single reads unstable (3 vs 5, stray strokes); voting fixes most of it."""
    votes = []
    pat = re.compile(r"^[A-Z]{1,4}$") if part == "top" else re.compile(r"^\d{2,4}[A-Z]?$")
    wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if part == "top" else "0123456789AB"
    for px, th, sc in settings:
        img = render(glyphs, box, px_per_unit=px, pad=0.0, thickness=th)
        H, W = img.shape
        mask = np.zeros_like(img); cv2.circle(mask, (W // 2, H // 2), int(r * px * 0.86), 255, -1)
        img = np.where(mask > 0, img, 255).astype(np.uint8)
        rows = (img < 128).sum(axis=1)
        img[np.where(rows > W * 0.55)[0], :] = 255
        half = img[:H // 2, :] if part == "top" else img[H // 2:, :]
        big = cv2.copyMakeBorder(cv2.resize(half, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        t = ocr_image(big, psm=7, whitelist=wl)
        if part == "bottom" and re.fullmatch(r"\d{4}", t or "") and t.endswith("5") and t[:3].endswith("3"):
            t = t[:3]                      # stroke-font '3' read as '35'
        if pat.match(t or ""):
            votes.append(t)
    if not votes:
        return "", 0.0
    best = max(set(votes), key=lambda v: (votes.count(v), -votes.index(v)))
    return best, votes.count(best) / len(settings)


def bubble_style(img):
    """img: rendered crop slightly larger than the circle. Returns 'field' | 'shared_display' | 'panel'."""
    h, w = img.shape
    edge = np.concatenate([img[2, :], img[-3, :], img[:, 2], img[:, -3]])
    boxed = (edge < 128).mean() > 0.5
    mid = img[h // 2 - 2:h // 2 + 3, int(w * .2):int(w * .8)]
    bar = ((mid < 128).mean(axis=1) > 0.85).any()
    if boxed:
        return "shared_display"
    return "panel" if bar else "field"


def normalise_tag(letters, number):
    letters = "".join(LETTER_FIX.get(c, c) for c in re.sub(r"[^A-Z0-9]", "", letters.upper()))
    number = "".join(NUM_FIX.get(c, c) for c in re.sub(r"[^0-9A-Za-z]", "", number))
    number = re.sub(r"[^0-9A-Z]", "", number)
    return letters, number


def read_bubbles(ents, sheet_of=None, r_min=2.5, r_max=3.6, px=24):
    """ents: model-space entities. Returns list of dicts (tag, letters, number, x, y, r, handle, style, method, confidence)."""
    circles = [e for e in ents if e["type"] == "CIRCLE" and r_min <= e.get("r", 0) <= r_max]
    small = [e for e in ents if e["type"] in ("LWPOLYLINE", "LINE", "MTEXT", "TEXT", "SOLID", "ARC")]
    # coarse grid index for speed
    grid = {}
    for e in small:
        b = e.get("bbox")
        if not b or (b[2] - b[0]) > 10 or (b[3] - b[1]) > 10:
            continue
        grid.setdefault((int(b[0] // 10), int(b[1] // 10)), []).append(e)
    out = []
    for c in circles:
        r = c["r"]
        box = (c["x"] - r, c["y"] - r, c["x"] + r, c["y"] + r)
        cand = []
        for gx in range(int(box[0] // 10) - 1, int(box[2] // 10) + 2):
            for gy in range(int(box[1] // 10) - 1, int(box[3] // 10) + 2):
                cand += grid.get((gx, gy), [])
        inside = [e for e in cand if in_box(e, box, 0.05)]
        frame = [e for e in cand if in_box(e, (box[0] - .6, box[1] - .6, box[2] + .6, box[3] + .6)) and e not in inside]
        styled = render(inside + frame + [c], box, px_per_unit=px, pad=0.4)
        style = bubble_style(styled)
        glyphs = [e for e in inside if e["type"] not in ("MTEXT", "TEXT")]
        mtext = [e for e in inside if e["type"] in ("MTEXT", "TEXT") and e.get("text")]
        # render glyphs only, mask circle edge, split top / bottom halves
        img = render(glyphs, box, px_per_unit=px, pad=0.0, thickness=2)
        H, W = img.shape
        mask = np.zeros_like(img); cv2.circle(mask, (W // 2, H // 2), int(r * px * 0.86), 255, -1)
        img = np.where(mask > 0, img, 255).astype(np.uint8)
        rows = (img < 128).sum(axis=1)
        mid = H // 2
        # if a bar exists, remove it
        bar_rows = np.where(rows > W * 0.55)[0]
        img[bar_rows, :] = 255
        top = img[:mid, :]; bot = img[mid:, :]
        big = lambda a: cv2.copyMakeBorder(cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        letters_raw, lv = ocr_vote(glyphs, box, r, "top")
        nums = [m["text"] for m in mtext if re.fullmatch(r"[0-9]{2,4}[A-Z]?", m["text"])]
        nv = 1.0
        if nums:
            number_raw, method = nums[0], "mtext+ocr"
        else:
            (number_raw, nv), method = ocr_vote(glyphs, box, r, "bottom"), "ocr"
        txt_letters = [m["text"] for m in mtext if re.fullmatch(r"[A-Z]{1,5}", m["text"])]
        if txt_letters:
            letters_raw, method = txt_letters[0], ("text" if nums else "text+ocr")
        letters, number = normalise_tag(letters_raw, number_raw)
        empty = (img < 128).sum() < 20 and not mtext
        conf = 0.0
        if letters and number:
            conf = 0.9 if method == "text" else 0.6 + 0.3 * lv if method == "mtext+ocr" else 0.45 + 0.2 * lv + 0.2 * nv
            if letters[0] not in FIRST_LETTERS or len(letters) > 4:
                conf -= 0.3
        elif letters:
            conf = 0.35
        out.append(dict(tag=f"{letters}-{number}" if letters and number else (letters or None), letters=letters,
                        number=number, x=c["x"], y=c["y"], r=r, handle=c.get("handle"), style=style,
                        method=method if not empty else "empty", confidence=round(max(conf, 0), 2),
                        sheet=sheet_of(c["x"], c["y"]) if sheet_of else None))
    return out
