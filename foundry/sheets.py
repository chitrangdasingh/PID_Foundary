"""sheets.py - split a multi-sheet model space into drawing sheets using border frames, and read each title block."""
import re

FRAME_W, FRAME_H = (370, 430), (270, 320)
DWG_RE = re.compile(r"[A-Z]{2,5}/\d{4}-\d{2}/[A-Z0-9-]+/PID/R\d+")
SECTION_RE = re.compile(r"\(\s*([^()]*(SECTION|SYSTEM|SHEET|UNIT|HEADER)[^()]*)\)", re.I)


def find_sheets(ents):
    """Returns list of sheets: dict(sheet_no, box, title, drawing_no, revision, kind)."""
    frames = []
    for e in ents:
        if e["type"] != "LWPOLYLINE" or "bbox" not in e:
            continue
        b = e["bbox"]; w, h = b[2] - b[0], b[3] - b[1]
        if FRAME_W[0] < w < FRAME_W[1] and FRAME_H[0] < h < FRAME_H[1]:
            frames.append(b)
    frames.sort()
    sheets = []
    for b in frames:  # keep the outermost frame of each nested group
        if sheets and b[0] < sheets[-1][2] - 50:
            s = sheets[-1]
            sheets[-1] = (min(s[0], b[0]), min(s[1], b[1]), max(s[2], b[2]), max(s[3], b[3]))
        else:
            sheets.append(b)
    out = []
    for i, b in enumerate(sheets, 1):
        out.append(dict(sheet_no=i, box=b, kind="process"))
    # anything substantial outside every frame (legend pages drawn without a frame) becomes one extra sheet
    right = max((s[2] for s in sheets), default=-1e18)
    rest = [e for e in ents if "bbox" in e and e["bbox"][0] > right + 5]
    if len(rest) > 50:
        texts = [e for e in rest if e["type"] in ("TEXT", "MTEXT")]
        leg = next((t for t in texts if "LEGEND" in t.get("text", "").upper()), None)
        if leg is not None:   # legend page drawn without a closed frame: page = text extent up to its title block
            x0 = min(t["x"] for t in texts if abs(t["y"] - leg["y"]) < 300 and t["x"] < leg["x"]) - 12
            box = (x0, min(0.0, leg["y"] - 12), leg["x"] + 62, max(t["y"] for t in texts if t["x"] < leg["x"] + 62) + 12)
        else:
            xs = sorted(v for e in rest for v in (e["bbox"][0], e["bbox"][2]))
            ys = sorted(v for e in rest for v in (e["bbox"][1], e["bbox"][3]))
            k = max(1, len(xs) // 500)
            box = (xs[k], ys[k], xs[-k], ys[-k])
        out.append(dict(sheet_no=len(out) + 1, box=box, kind="legend"))
    for s in out:
        texts = [e for e in ents if e["type"] in ("TEXT", "MTEXT") and _inside(e["bbox"], s["box"], 2)]
        s["title"], s["drawing_no"], s["revision"] = _title_block(texts, s)
        if s["title"] and "LEGEND" in s["title"].upper():
            s["kind"] = "legend"
    return out


def _inside(b, box, m=0.5):
    return b[0] >= box[0] - m and b[1] >= box[1] - m and b[2] <= box[2] + m and b[3] <= box[3] + m


def _title_block(texts, s):
    title = None
    for t in sorted(texts, key=lambda t: t["y"]):
        m = SECTION_RE.search(t["text"])
        if m:
            title = m.group(1).strip().title(); break
    dwg = next((DWG_RE.search(t["text"]).group(0) for t in texts if DWG_RE.search(t["text"])), None)
    x0, y0, x1, y1 = s["box"]
    revs = [int(t["text"]) for t in texts if re.fullmatch(r"\d{2}", t["text"]) and x1 - 85 < t["x"] < x1 - 55 and y0 + 30 < t["y"] < y0 + 90]
    return title, dwg, (f"{max(revs):02d}" if revs else None)


def sheet_lookup(sheets):
    def f(x, y):
        for s in sheets:
            b = s["box"]
            if b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1:
                return s["sheet_no"]
        return None
    return f
