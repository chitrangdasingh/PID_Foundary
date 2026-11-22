"""render.py - draw DXF geometry (lines, polylines, circles, arcs, solids, hatches, text boxes) to an image with OpenCV."""
import math

import cv2
import numpy as np


def render(ents, box, px_per_unit=4.0, pad=0.0, draw_text=False, thickness=1):
    """ents: entity dicts; box: (x0, y0, x1, y1) in drawing units. Returns a white uint8 grayscale image."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    W, H = max(1, int((x1 - x0) * px_per_unit)), max(1, int((y1 - y0) * px_per_unit))
    img = np.full((H, W), 255, np.uint8)
    P = lambda x, y: (int(round((x - x0) * px_per_unit)), int(round((y1 - y) * px_per_unit)))
    for e in ents:
        t = e["type"]
        if t == "LINE":
            cv2.line(img, P(e.get("x", 0), e.get("y", 0)), P(e.get("x2", 0), e.get("y2", 0)), 0, thickness)
        elif t in ("LWPOLYLINE", "POLYLINE") and e.get("pts"):
            pts = np.array([P(*p) for p in e["pts"]], np.int32)
            cv2.polylines(img, [pts], bool(e.get("closed")), 0, thickness)
        elif t == "CIRCLE":
            cv2.circle(img, P(e["x"], e["y"]), max(1, int(e.get("r", 0) * px_per_unit)), 0, thickness)
        elif t == "ARC":
            c = P(e["x"], e["y"]); r = max(1, int(e.get("r", 0) * px_per_unit))
            a1, a2 = e.get("rot", 0.0), e.get("a2", 360.0)
            if a2 < a1:
                a2 += 360
            cv2.ellipse(img, c, (r, r), 0, -a2, -a1, 0, thickness)
        elif t in ("SOLID", "TRACE"):
            q = [(e.get("x"), e.get("y")), (e.get("x2"), e.get("y2")), (e.get("x4", e.get("x3")), e.get("y4", e.get("y3"))), (e.get("x3"), e.get("y3"))]
            q = [P(a, b) for a, b in q if a is not None and b is not None]
            if len(q) >= 3:
                cv2.fillPoly(img, [np.array(q, np.int32)], 0)
        elif t == "HATCH" and e.get("pts"):
            cv2.fillPoly(img, [np.array([P(*p) for p in e["pts"]], np.int32)], 0)
        elif draw_text and t in ("TEXT", "MTEXT") and e.get("text"):
            h = max(6, int(e.get("height", 1) * px_per_unit))
            cv2.putText(img, e["text"][:40], P(e["x"], e["y"]), cv2.FONT_HERSHEY_SIMPLEX, h / 28, 0, 1, cv2.LINE_AA)
    return img


def in_box(e, box, margin=0.0):
    b = e.get("bbox")
    if not b:
        return False
    return b[0] >= box[0] - margin and b[1] >= box[1] - margin and b[2] <= box[2] + margin and b[3] <= box[3] + margin
