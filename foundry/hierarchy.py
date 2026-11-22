"""
hierarchy.py - map extracted objects onto an ISA-95 style asset hierarchy and reconcile with the BOM / index.

  Enterprise (Thermax Limited) > Site > Area (the skid) > Process cell (drawing sheet / section)
     > Unit (equipment) > Control module (instrument loop, valve, relief device)

Instruments are placed under equipment by graph distance (fewest hops through candidate connections), else by
nearest equipment on the same sheet; the method is stored so an engineer can see why. Loops are formed from
instrument tags sharing a loop number (TE-202, TIC-202, TY-202, HTR-202 -> loop 202).
Reconciliation compares every tag with the BOM: matched, only on the drawing, only in the BOM.
"""
import json
import math
import re
from collections import defaultdict

import networkx as nx

SECTION_DEFAULT = "Unassigned section"
ISA_FIRST = {"A": "Analysis", "F": "Flow", "L": "Level", "P": "Pressure", "T": "Temperature", "W": "Weight", "H": "Hand",
             "K": "Time", "S": "Speed", "Z": "Position", "E": "Voltage", "J": "Power", "I": "Current", "Y": "Event"}


def norm(tag):
    return re.sub(r"\s+", "", (tag or "").upper())


def load_bom(path):
    if not path:
        return []
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["BOM"] if "BOM" in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h or "").strip().lower() for h in rows[0]]
    out = []
    for r in rows[1:]:
        d = dict(zip(head, r))
        if d.get("tag"):
            out.append(dict(tag=norm(d["tag"]), category=d.get("category"), description=d.get("description"),
                            sheet=d.get("sheet"), item=d.get("item")))
    return out


def build(store, pid_id, symbols, edges, bom_path=None, site="Site (to be confirmed)"):
    sheets = {r["sheet_no"]: r for r in store.q("SELECT * FROM sheets WHERE pid_id=?", (pid_id,))}
    src = store.one("SELECT * FROM sources WHERE pid_id=?", (pid_id,)) or {}
    assets = []
    A = lambda **k: assets.append(dict(dict(parent_id=None, tag=None, name=None, asset_class=None, sheet_no=None, bom_item=None,
                                            match_status=None, confidence=None, props={}), **k))
    ent, site_id, area = "ENT-THERMAX", "SITE-1", f"AREA-{pid_id}"
    A(asset_id=ent, level="Enterprise", name="Thermax Limited")
    A(asset_id=site_id, parent_id=ent, level="Site", name=site)
    A(asset_id=area, parent_id=site_id, level="Area", name=src.get("title") or "Process area", props=dict(drawing_no=src.get("drawing_no")))
    for n, s in sorted(sheets.items()):
        if s["kind"] == "process":
            A(asset_id=f"CELL-{n}", parent_id=area, level="Process cell", name=f"Sheet {n}: {s['title'] or SECTION_DEFAULT}", sheet_no=n)
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["src"], e["dst"])
    equip = [o for o in symbols if o["category"] == "Equipment"]
    eq_by_sheet = defaultdict(list)
    for o in equip:
        eq_by_sheet[o["sheet_no"]].append(o)
        A(asset_id=o["symbol_id"], parent_id=f"CELL-{o['sheet_no']}", level="Unit", tag=o["tag"], name=o["tag"], asset_class=o["cls"],
          sheet_no=o["sheet_no"], confidence=o["confidence"])
    center = lambda o: ((o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2)
    loops = defaultdict(list)
    for o in symbols:
        if o["category"] not in ("Instrument", "Piping") or o["cls"].startswith("Annotation"):
            continue
        parent, method = f"CELL-{o['sheet_no']}", "sheet"
        cands = [(nx.shortest_path_length(G, o["symbol_id"], q["symbol_id"]), q) for q in eq_by_sheet.get(o["sheet_no"], [])
                 if o["symbol_id"] in G and q["symbol_id"] in G and nx.has_path(G, o["symbol_id"], q["symbol_id"])]
        if cands:
            d, q = min(cands, key=lambda t: t[0])
            parent, method = q["symbol_id"], f"graph ({d} hops)"
        elif eq_by_sheet.get(o["sheet_no"]):
            q = min(eq_by_sheet[o["sheet_no"]], key=lambda q: math.dist(center(o), center(q)))
            if math.dist(center(o), center(q)) < 60:
                parent, method = q["symbol_id"], "nearest equipment"
        tag = o.get("tag")
        m = re.match(r"^([A-Z]+)-(\d{3})", tag or "")
        if m and o["category"] == "Instrument":
            loops[(o["sheet_no"], m.group(2), m.group(1)[0])].append(o["symbol_id"])
        A(asset_id=o["symbol_id"], parent_id=parent, level="Control module", tag=tag, name=tag or o["cls"].split("/")[-1],
          asset_class=o["cls"], sheet_no=o["sheet_no"], confidence=o.get("confidence"), props=dict(placement=method))
    for (sh, num, var), ids in loops.items():
        if len(ids) >= 2:
            lid = f"LOOP-{ISA_FIRST.get(var, var)[:4].upper()}-{num}"
            parent = next((a["parent_id"] for a in assets if a["asset_id"] == ids[0]), f"CELL-{sh}")
            A(asset_id=lid, parent_id=parent, level="Loop", tag=f"{var}-{num}", name=f"{ISA_FIRST.get(var, var)} loop {num}", sheet_no=sh,
              props=dict(members=ids))
            for a in assets:
                if a["asset_id"] in ids:
                    a["parent_id"] = lid
    # BOM reconciliation
    bom = load_bom(bom_path)
    bom_by = {b["tag"]: b for b in bom}
    found = defaultdict(list)
    for a in assets:
        if a["tag"] and a["level"] in ("Unit", "Control module"):
            found[norm(a["tag"])].append(a)
    rows = []
    for tag, b in bom_by.items():
        if tag in found:
            for a in found[tag]:
                a["bom_item"], a["match_status"] = str(b.get("item")), "matched"
            rows.append(dict(tag=tag, bom_category=b["category"], bom_sheet=b["sheet"], status="matched",
                             drawing_sheets=sorted({a["sheet_no"] for a in found[tag]})))
        else:
            rows.append(dict(tag=tag, bom_category=b["category"], bom_sheet=b["sheet"], status="in BOM, not found on drawing", drawing_sheets=[]))
    for tag, lst in found.items():
        if tag not in bom_by:
            for a in lst:
                a["match_status"] = "not in BOM" if bom else None
            if bom:
                rows.append(dict(tag=tag, bom_category=None, bom_sheet=None, status="on drawing, not in BOM", drawing_sheets=sorted({a["sheet_no"] for a in lst})))
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["bom_category"]:
            by_cat[r["bom_category"]][1] += 1
            by_cat[r["bom_category"]][0] += r["status"] == "matched"
    summary = dict(bom_tags=len(bom_by), matched=sum(r["status"] == "matched" for r in rows),
                   bom_only=sum(r["status"].startswith("in BOM") for r in rows), drawing_only=sum(r["status"].startswith("on drawing") for r in rows),
                   recall_by_category={k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())})
    store.x("DELETE FROM assets WHERE pid_id=?", (pid_id,))
    store.many("INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               [(pid_id, a["asset_id"], a["parent_id"], a["level"], a["tag"], a["name"], a["asset_class"], a["sheet_no"], a["bom_item"],
                 a["match_status"], a["confidence"], json.dumps(a["props"])) for a in assets])
    store.kv_set(f"recon:{pid_id}", dict(summary=summary, rows=rows))
    return assets, dict(summary=summary, rows=rows)


def drawing_checks(symbols, edges, assets):
    """Rule checks an engineer should look at (not confirmed defects): duplicate tags, untagged bubbles,
    incomplete control loops, unconnected equipment, safety devices without a tag."""
    out = []
    tags = defaultdict(list)
    for o in symbols:
        if o.get("tag") and o["category"] in ("Instrument", "Piping", "Equipment"):
            tags[o["tag"]].append(o)
    for t, lst in tags.items():
        ids = sorted({o["symbol_id"] for o in lst})
        if len(ids) > 1 and lst[0]["category"] != "Equipment":
            out.append(dict(rule="Duplicate tag", severity="high", tag=t, sheets=sorted({o["sheet_no"] for o in lst}), objects=ids,
                            detail=f"{t} appears {len(ids)} times"))
    for o in symbols:
        if o["cls"].startswith("Instrument/Bubble") and not o.get("tag"):
            out.append(dict(rule="Bubble without readable tag", severity="medium", tag=None, sheets=[o["sheet_no"]], objects=[o["symbol_id"]], detail="OCR could not read the bubble"))
        if o["category"] == "Piping" and ("Safety" in o["cls"] or "Rupture" in o["cls"]) and not o.get("tag"):
            out.append(dict(rule="Relief device without tag", severity="medium", tag=None, sheets=[o["sheet_no"]], objects=[o["symbol_id"]], detail=o["cls"]))
    loops = defaultdict(set)
    for o in symbols:
        m = re.match(r"^([A-Z]+)-(\d{3})", o.get("tag") or "")
        if m and o["category"] == "Instrument":
            loops[(m.group(2), m.group(1)[0])].add(m.group(1))
    for (num, var), fns in loops.items():
        if any(f.endswith("IC") for f in fns) and not any(f.endswith("E") or f.endswith("T") for f in fns):
            out.append(dict(rule="Controller without sensor", severity="medium", tag=f"{var}-{num}", sheets=[], objects=sorted(fns), detail=f"loop {num} has {sorted(fns)}"))
    connected = {e["src"] for e in edges} | {e["dst"] for e in edges}
    for o in symbols:
        if o["category"] == "Equipment" and o["symbol_id"] not in connected:
            out.append(dict(rule="Equipment with no detected connection", severity="low", tag=o.get("tag"), sheets=[o["sheet_no"]], objects=[o["symbol_id"]],
                            detail="outline not found or lines end short of it"))
    return out
