"""
symbols.py - symbol detection.

1. Candidate regions: the sheet is rendered (text and the title block removed), long straight runs (pipes,
   signal lines, equipment walls) are removed with morphological opening, and what is left is grouped into
   connected components: valves, bubbles, reliefs, filters, arrows, glyphs.
2. Each candidate crop is classified by the active symbol model (HOG features + SVM / random forest, trained
   from the legend seed set, augmentations, data-scientist uploads and approved reviews - see training.py).
3. Low-confidence results go to the review queue; approved ones become new training samples (active learning).
The same code runs on raster input (scans / PDF pages): only step 1's rendering is replaced by binarisation.
"""
import cv2
import numpy as np

from .render import render

CROP = 64


def hog(img, cell=8, block=2, bins=9):
    """Histogram of oriented gradients in plain numpy (same layout as OpenCV's HOGDescriptor 64x64 / 16x16 / 8 / 8 / 9):
    unsigned orientations, 2x2-cell blocks with 1-cell stride, L2-Hys normalisation. No OpenCV build dependency."""
    f = img.astype(np.float32)
    gx = np.zeros_like(f); gy = np.zeros_like(f)
    gx[:, 1:-1] = f[:, 2:] - f[:, :-2]
    gy[1:-1, :] = f[2:, :] - f[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / bins)
    lo = np.floor(ang).astype(int) % bins
    hi = (lo + 1) % bins
    w_hi = ang - np.floor(ang)
    H, W = f.shape
    ch, cw = H // cell, W // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    ci = (np.arange(H) // cell)[:, None].repeat(W, 1)
    cj = (np.arange(W) // cell)[None, :].repeat(H, 0)
    ok = (ci < ch) & (cj < cw)
    np.add.at(hist, (ci[ok], cj[ok], lo[ok]), (mag * (1 - w_hi))[ok])
    np.add.at(hist, (ci[ok], cj[ok], hi[ok]), (mag * w_hi)[ok])
    out = []
    for i in range(ch - block + 1):
        for j in range(cw - block + 1):
            v = hist[i:i + block, j:j + block].ravel()
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            v = np.minimum(v, 0.2)
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            out.append(v)
    return np.concatenate(out)


def square(img, size=CROP, pad_frac=0.08):
    """Pad to square (white), add margin, resize. img: uint8, black ink on white."""
    h, w = img.shape[:2]
    s = int(max(h, w) * (1 + 2 * pad_frac)) + 2
    out = np.full((s, s), 255, np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def features(img):
    """HOG + coarse ink grid + aspect. img: square CROP x CROP grayscale."""
    h = hog(img)
    ink = (img < 128).astype(np.float32)
    grid = cv2.resize(ink, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([h, grid, [ink.mean()]]).astype(np.float32)


def sheet_raster(ents, box, px=6.0, exclude_title=True):
    geo = [e for e in ents if e["type"] not in ("TEXT", "MTEXT", "IMAGE", "OLE2FRAME") and "bbox" in e
           and box[0] - 1 <= e["bbox"][0] and e["bbox"][2] <= box[2] + 1 and box[1] - 1 <= e["bbox"][1] and e["bbox"][3] <= box[3] + 1]
    # drop the frame, the title block and the bottom equipment strip
    x0, y0, x1, y1 = box
    keep = []
    for e in geo:
        b = e["bbox"]
        if (b[2] - b[0]) > 0.9 * (x1 - x0) or (b[3] - b[1]) > 0.9 * (y1 - y0):
            continue
        if exclude_title and ((b[0] > x1 - 95 and b[3] < y0 + 75) or b[3] < y0 + 30 or b[1] > y1 - 8 or b[2] < x0 + 8 or b[0] > x1 - 8):
            continue
        keep.append(e)
    return render(keep, box, px_per_unit=px), keep


def candidates(img, px=6.0, min_u=1.2, max_u=26.0, line_u=9.0):
    """img: rendered sheet. Returns list of (x, y, w, h) pixel boxes."""
    ink = (img < 128).astype(np.uint8) * 255
    k = max(9, int(line_u * px))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    rest = cv2.subtract(ink, cv2.bitwise_or(hor, ver))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blob = cv2.dilate(rest, np.ones((int(px * 0.9), int(px * 0.9)), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if min_u * px <= max(w, h) <= max_u * px and a > (px * px):
            out.append((int(x), int(y), int(w), int(h)))
    return out, rest


def crop(img, b, px=6.0, margin_u=0.8):
    x, y, w, h = b
    m = int(margin_u * px)
    H, W = img.shape
    return img[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]


def px_to_cad(b, box, px):
    x, y, w, h = b
    return (box[0] + x / px, box[3] - (y + h) / px, box[0] + (x + w) / px, box[3] - y / px)


def vector_candidates(ents, box, max_u=14.0, long_u=7.0, tol=0.35):
    """Vector-native candidates for CAD input: union small non-text entities whose boxes touch.
    Returns list of dict(bbox, members) in drawing units."""
    x0, y0, x1, y1 = box
    small = []
    for e in ents:
        if e["type"] in ("TEXT", "MTEXT", "IMAGE", "OLE2FRAME", "HATCH") or "bbox" not in e:
            continue
        b = e["bbox"]
        if not (x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1):
            continue
        if (b[0] > x1 - 95 and b[3] < y0 + 75) or b[3] < y0 + 30:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if max(w, h) > max_u:
            continue
        if e["type"] in ("LINE",) and max(w, h) > long_u and min(w, h) < 0.05:
            continue
        if e["type"] == "LWPOLYLINE" and len(e.get("pts", [])) == 2 and max(w, h) > long_u and min(w, h) < 0.05:
            continue
        small.append(e)
    n = len(small)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    cell = 4.0
    grid = {}
    for i, e in enumerate(small):
        b = e["bbox"]
        for gx in range(int(b[0] // cell), int(b[2] // cell) + 1):
            for gy in range(int(b[1] // cell), int(b[3] // cell) + 1):
                grid.setdefault((gx, gy), []).append(i)
    for idxs in grid.values():
        for a in range(len(idxs)):
            ba = small[idxs[a]]["bbox"]
            for b_ in idxs[a + 1:]:
                bb = small[b_]["bbox"]
                if ba[0] - tol <= bb[2] and bb[0] - tol <= ba[2] and ba[1] - tol <= bb[3] and bb[1] - tol <= ba[3]:
                    ra, rb = find(idxs[a]), find(b_)
                    if ra != rb:
                        parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(small[i])
    out = []
    for mem in groups.values():
        b = (min(m["bbox"][0] for m in mem), min(m["bbox"][1] for m in mem), max(m["bbox"][2] for m in mem), max(m["bbox"][3] for m in mem))
        w, h = b[2] - b[0], b[3] - b[1]
        if max(w, h) < 1.0:
            continue
        if max(w, h) > 2.2 * max_u:
            # an over-merged cluster (e.g. bubbles tied by leaders to a hatched heater): keep its bubbles
            for c in [m for m in mem if m["type"] == "CIRCLE" and 2.5 <= m.get("r", 0) <= 3.6]:
                cb = (c["x"] - c["r"], c["y"] - c["r"], c["x"] + c["r"], c["y"] + c["r"])
                inside = [m for m in mem if m is c or (cb[0] - .05 <= m["bbox"][0] and m["bbox"][2] <= cb[2] + .05 and cb[1] - .05 <= m["bbox"][1] and m["bbox"][3] <= cb[3] + .05)]
                out.append(dict(bbox=cb, members=inside))
            continue
        out.append(dict(bbox=b, members=mem))
    return out


def render_candidate(c, px=None):
    b = c["bbox"]
    s = max(b[2] - b[0], b[3] - b[1], 0.5)
    px = px or (CROP * 0.85 / s)
    return square(render(c["members"], b, px_per_unit=px, pad=0.15, thickness=1 if px < 20 else 2))
