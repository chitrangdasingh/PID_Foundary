"""
dxf_reader.py - a dependency-free, streaming DXF reader (ASCII DXF R12 to 2018).

Reads the ENTITIES section (and BLOCKS definitions) into flat records:
  handle, type, layer, block (owner block name or '*MODEL'), geometry (points / centre / radius),
  text (TEXT, MTEXT, ATTRIB), insert name / position / scale / rotation, bbox.
It keeps every entity handle so each extracted fact can be traced back to the drawing.
ezdxf is the better choice when it can be installed; this reader exists so the platform runs anywhere.
"""
import hashlib
import math
import re
from collections import Counter

MTEXT_CODES = re.compile(r"\\[ACcFfHhLlOoQqTtWw][^;\\]*;|\\[PNXLlOoKk~]|[{}]|\\U\+[0-9A-Fa-f]{4}")


def clean_mtext(s):
    s = s.replace("\\P", " ").replace("\\~", " ")
    s = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), s)
    s = MTEXT_CODES.sub("", s)
    s = s.replace("%%c", "Ø").replace("%%C", "Ø").replace("%%d", "°").replace("%%D", "°").replace("%%p", "±")
    return re.sub(r"\s+", " ", s).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs(path):
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        while True:
            code = f.readline()
            if not code:
                return
            val = f.readline()
            try:
                yield int(code.strip()), val.rstrip("\r\n")
            except ValueError:
                continue


def _num(v, default=0.0):
    try:
        return float(v)
    except ValueError:
        return default


def read(path):
    """Returns dict(header, entities, blocks, stats). entities: list of dicts (model space + block contents)."""
    header, entities, blocks = {}, [], {}
    section = None
    cur = None
    block_name = None
    hvar = None
    in_ent = False

    def finish():
        nonlocal cur
        if cur is not None:
            _finalise(cur)
            if cur["type"] not in ("SEQEND", "ENDBLK", "BLOCK", "VERTEX"):
                entities.append(cur)
            cur = None

    last_poly = None
    for code, val in _pairs(path):
        if code == 0:
            if val == "SECTION":
                finish(); section = "?"; continue
            if val == "ENDSEC":
                finish(); section = None; continue
            if section in ("ENTITIES", "BLOCKS"):
                if cur is not None and cur["type"] == "VERTEX" and last_poly is not None:
                    last_poly["pts"].append((cur.get("x", 0), cur.get("y", 0)))
                    cur = None
                elif cur is not None and cur["type"] == "POLYLINE":
                    last_poly = cur
                    entities.append(cur); cur = None
                else:
                    if cur is not None and cur["type"] == "SEQEND":
                        if last_poly is not None:
                            _finalise(last_poly)
                        last_poly = None
                        cur = None
                    finish()
                if val == "BLOCK":
                    cur = dict(type="BLOCK", block_def=True)
                    continue
                if val == "ENDBLK":
                    block_name = None
                    continue
                cur = dict(type=val, layer="0", block=block_name or "*MODEL", pts=[])
            continue
        if section == "?":
            if code == 2:
                section = val
            continue
        if section == "HEADER":
            if code == 9:
                hvar = val
            elif hvar:
                header.setdefault(hvar, val if code not in (10, 20, 30) else None)
                if code in (10, 20):
                    header[hvar + ("_x" if code == 10 else "_y")] = _num(val)
            continue
        if cur is None:
            continue
        t = cur["type"]
        if t == "BLOCK":
            if code == 2:
                block_name = val
                cur["name"] = val
                blocks[val] = dict(name=val, base=(0.0, 0.0))
            elif code == 10 and block_name:
                blocks[block_name]["base"] = (_num(val), blocks[block_name]["base"][1])
            elif code == 20 and block_name:
                blocks[block_name]["base"] = (blocks[block_name]["base"][0], _num(val))
            continue
        if code == 5:
            cur["handle"] = val
        elif code == 8:
            cur["layer"] = val
        elif code == 2 and t in ("INSERT",):
            cur["name"] = val
        elif code in (1, 3) and t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            cur["raw"] = cur.get("raw", "") + val if code == 3 or "raw" not in cur else cur["raw"] + val
            if code == 1 and t != "MTEXT":
                cur["raw"] = val
        elif code == 98 and t == "HATCH":
            cur["seeds"] = True
        elif code in (10, 20) and t == "HATCH" and cur.get("seeds"):
            pass
        elif code == 10:
            if t in ("LWPOLYLINE", "SPLINE", "LEADER", "HATCH"):
                cur["pts"].append([_num(val), None])
            else:
                cur["x"] = _num(val)
        elif code == 20:
            if t in ("LWPOLYLINE", "SPLINE", "LEADER", "HATCH") and cur["pts"] and cur["pts"][-1][1] is None:
                cur["pts"][-1][1] = _num(val)
            else:
                cur["y"] = _num(val)
        elif code == 11:
            cur["x2"] = _num(val)
        elif code == 21:
            cur["y2"] = _num(val)
        elif code in (12, 13):
            cur["x%d" % (code - 9)] = _num(val)
        elif code in (22, 23):
            cur["y%d" % (code - 19)] = _num(val)
        elif code == 40:
            cur.setdefault("r", _num(val))         # radius / text height
        elif code == 41:
            cur["sx"] = _num(val)
        elif code == 42:
            cur["sy"] = _num(val)
        elif code == 50:
            cur["rot"] = _num(val)
        elif code == 51:
            cur["a2"] = _num(val)
        elif code == 70 and t == "LWPOLYLINE":
            cur["closed"] = bool(int(_num(val)) & 1)
    finish()
    ents = [e for e in entities if e.get("type") not in ("BLOCK",)]
    stats = dict(total=len(ents), by_type=Counter(e["type"] for e in ents), by_layer=Counter(e["layer"] for e in ents),
                 model_space=sum(1 for e in ents if e["block"] == "*MODEL"))
    return dict(header=header, entities=ents, blocks=blocks, stats=stats)


def _finalise(e):
    t = e["type"]
    if t in ("TEXT", "MTEXT", "ATTRIB"):
        e["text"] = clean_mtext(e.get("raw", ""))
        e["height"] = e.get("r", 0.0)
    pts = [tuple(p) for p in e.get("pts", []) if p[1] is not None]
    if t == "HATCH" and not e.get("_hatch_done"):
        pts = pts[1:]                 # first code-10 point of a HATCH is its elevation point, not boundary
        e["_hatch_done"] = True
    e["pts"] = pts
    xs, ys = [], []
    if t == "LINE":
        xs, ys = [e.get("x", 0), e.get("x2", 0)], [e.get("y", 0), e.get("y2", 0)]
    elif t in ("CIRCLE", "ARC"):
        r = e.get("r", 0)
        xs, ys = [e.get("x", 0) - r, e.get("x", 0) + r], [e.get("y", 0) - r, e.get("y", 0) + r]
    elif t in ("SOLID", "TRACE", "3DFACE"):
        xs = [e.get(k) for k in ("x", "x2", "x3", "x4") if e.get(k) is not None]
        ys = [e.get(k) for k in ("y", "y2", "y3", "y4") if e.get(k) is not None]
    elif pts:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    elif "x" in e:
        xs, ys = [e["x"]], [e.get("y", 0)]
        if t in ("TEXT", "MTEXT"):
            w = 0.6 * e.get("height", 1) * max(len(e.get("text", "")), 1)
            xs, ys = [e["x"], e["x"] + w], [e.get("y", 0), e.get("y", 0) + e.get("height", 1)]
    if xs:
        e["bbox"] = (min(xs), min(ys), max(xs), max(ys))
    return e


def explode_inserts(doc, max_depth=3):
    """Returns model-space entities with every INSERT replaced by its transformed block geometry.
    Exploded entities keep the INSERT handle in 'insert' and the block name in 'from_block'."""
    by_block = {}
    for e in doc["entities"]:
        if e["block"] != "*MODEL":
            by_block.setdefault(e["block"], []).append(e)
    out = []

    def tf(pt, ins, base):
        x, y = (pt[0] - base[0]) * ins.get("sx", 1.0), (pt[1] - base[1]) * ins.get("sy", 1.0)
        a = math.radians(ins.get("rot", 0.0))
        return (ins.get("x", 0) + x * math.cos(a) - y * math.sin(a), ins.get("y", 0) + x * math.sin(a) + y * math.cos(a))

    def expand(ins, depth, root):
        name = ins.get("name")
        base = doc["blocks"].get(name, {}).get("base", (0.0, 0.0))
        for e in by_block.get(name, []):
            if e["type"] == "INSERT":
                if depth < max_depth:
                    sub = dict(e); sub["x"], sub["y"] = tf((e.get("x", 0), e.get("y", 0)), ins, base)
                    sub["sx"] = e.get("sx", 1) * ins.get("sx", 1); sub["sy"] = e.get("sy", 1) * ins.get("sy", 1)
                    sub["rot"] = e.get("rot", 0) + ins.get("rot", 0)
                    expand(sub, depth + 1, root)
                continue
            n = dict(e, block="*MODEL", insert=root.get("handle"), from_block=root.get("name"), layer=root.get("layer", e["layer"]))
            for kx, ky in (("x", "y"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")):
                if kx in e:
                    n[kx], n[ky] = tf((e[kx], e.get(ky, 0)), ins, base)
            if e.get("pts"):
                n["pts"] = [tf(p, ins, base) for p in e["pts"]]
            if e["type"] in ("CIRCLE", "ARC"):
                n["r"] = e.get("r", 0) * abs(ins.get("sx", 1))
                if e["type"] == "ARC":
                    n["rot"] = e.get("rot", 0) + ins.get("rot", 0); n["a2"] = e.get("a2", 360) + ins.get("rot", 0)
            n.pop("bbox", None)
            _finalise(n)
            out.append(n)

    for e in doc["entities"]:
        if e["block"] != "*MODEL":
            continue
        if e["type"] == "INSERT":
            expand(e, 0, e)
        else:
            out.append(e)
    return out
