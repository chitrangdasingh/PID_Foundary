"""
legend.py - read the legend sheet into a symbol dictionary (the seed training set).

A P&ID legend is a table: Sr. # | DESCRIPTION | SYMBOL (repeated in columns). Each row gives a labelled example of a
symbol drawn in the same CAD style as the process sheets, so it is the natural seed set (the Azure design seeds with
32 standard symbols; here the drawing's own legend gives ~60). Labels follow <Category>/<Subcategory>/<Display>.
"""
import re

from .render import render, in_box

TAXONOMY = [  # (regex on description, category, subcategory)
    (r"PROCESS LINE|HEAT TRACED|HOSE", "Piping", "Line"),
    (r"SIGNAL|SUPPLY|CAPILLARY|DATA LINK", "Instrument", "SignalLine"),
    (r"SAFETY VALVE|RELIEF VALVE", "Piping", "SafetyValve"),
    (r"RUPTURE", "Piping", "RuptureDisc"),
    (r"REGULATOR", "Piping", "Regulator"),
    (r"ACTUATED VALVE", "Piping", "ControlValve"),
    (r"VALVE", "Piping", "Valve"),
    (r"FILTER|STRAINER", "Piping", "Filter"),
    (r"PUMP|COMPRESSOR", "Equipment", "Rotating"),
    (r"MASS FLOW CONTROLLER", "Instrument", "MFC"),
    (r"FLOW ELEMENT|FLOW INDICATOR", "Instrument", "Flow"),
    (r"LEVEL GLASS|LEVEL SWITCH|LEVEL TRANSMITTER", "Instrument", "Level"),
    (r"TEMPERATURE ELEMENT", "Instrument", "Temperature"),
    (r"PRESSURE TRANSMITTER", "Instrument", "Pressure"),
    (r"INSTRUMENT|FUNCTION|PLC", "Instrument", "Bubble"),
    (r"INSULATION", "Piping", "Insulation"),
]


def taxonomy(desc):
    d = desc.upper()
    for rx, cat, sub in TAXONOMY:
        if re.search(rx, d):
            return cat, sub
    return "Other", "Misc"


def read_legend(ents, sheet):
    """Returns rows: dict(sr, description, label, category, subcategory, box, handles)."""
    box = sheet["box"]
    texts = [e for e in ents if e["type"] in ("TEXT", "MTEXT") and in_box(e, box, 1)]
    heads_sym = sorted(t["x"] for t in texts if t["text"].upper() == "SYMBOL")
    heads_sr = sorted(t["x"] for t in texts if t["text"].upper().startswith("SR"))
    head_y = max((t["y"] for t in texts if t["text"].upper() == "SYMBOL"), default=box[3])
    srs = [t for t in texts if re.fullmatch(r"\d{1,2}\.", t["text"]) and t["y"] < head_y - 2]
    rows = []
    for sr in srs:
        # column = nearest Sr header on the left
        col_sr = max([x for x in heads_sr if x <= sr["x"] + 3], default=None)
        if col_sr is None:
            continue
        nxt_sr = min([x for x in heads_sr if x > col_sr + 5], default=col_sr + 100)
        sym_x = min([x for x in heads_sym if x > col_sr], default=None)
        if sym_x is None or sym_x > nxt_sr:
            continue        # this column has no symbol cell (tag code table)
        same_col = sorted([s for s in srs if abs(s["x"] - sr["x"]) < 4], key=lambda s: -s["y"])
        i = same_col.index(sr)
        y_hi = (same_col[i - 1]["y"] + sr["y"]) / 2 if i > 0 else head_y - 2
        y_lo = (same_col[i + 1]["y"] + sr["y"]) / 2 if i + 1 < len(same_col) else sr["y"] - 7
        y_hi = min(y_hi, sr["y"] + 6); y_lo = max(y_lo, sr["y"] - 9)
        desc_parts = sorted([t for t in texts if col_sr + 5 < t["x"] < sym_x - 12 and y_lo < t["y"] < y_hi + 1
                             and not re.fullmatch(r"\d{1,2}\.", t["text"])], key=lambda t: -t["y"])
        desc = " ".join(t["text"] for t in desc_parts).strip()
        if not desc:
            continue
        cell = (sym_x - 20, y_lo, nxt_sr - 2, y_hi)
        geo = [e for e in ents if e["type"] not in ("TEXT", "MTEXT") and in_box(e, cell, 0.2)]
        cat, sub = taxonomy(desc)
        rows.append(dict(sr=int(sr["text"][:-1]), description=desc, category=cat, subcategory=sub,
                         label=f"{cat}/{sub}/{desc.title()}", box=cell, n_geom=len(geo),
                         handles=[g.get("handle") for g in geo]))
    return sorted(rows, key=lambda r: r["sr"])


def render_cell(ents, row, px=12):
    geo = [e for e in ents if e["type"] not in ("TEXT", "MTEXT") and in_box(e, row["box"], 0.2)]
    if not geo:
        return None
    b = (min(g["bbox"][0] for g in geo), min(g["bbox"][1] for g in geo), max(g["bbox"][2] for g in geo), max(g["bbox"][3] for g in geo))
    return render(geo, b, px_per_unit=px, pad=0.5), b
