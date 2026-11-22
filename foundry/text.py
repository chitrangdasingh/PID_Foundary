"""
text.py - text detection for vector drawings: every TEXT/MTEXT span with its box, classified into
equipment tag / instrument tag / line number / pipe size / valve tag / off-page connector / note / title-block.
Rules give weak labels; a character n-gram model (models.TextClassifier) learns from them and from reviews.
"""
import re

EQUIP_RE = re.compile(r"^(V|HE|SM|FRN|R|CH|THY|P|C|K|E|T|TK|F|FL|D|CL|HTR|BL|CMP|PK|COL|X)-\d{3}[A-Z]?$")
INSTR_RE = re.compile(r"^([A-Z]{2,4})-(\d{2,4}[A-Z]?)$")
VALVE_RE = re.compile(r"^(BV|NV|TV|NRV|GV|GLV|CV|MV|SV|PSV|PRV|BPR|RD|XV|FV|PCV|TCV|LCV|SOV|MFC|F)-?\d{2,4}[A-Z]?$")
LINE_RE = re.compile(r"^\d{1,3}(/\d)?\"?-[A-Z]{1,4}-\d{2,4}(-[A-Z0-9]+){0,3}$")
SIZE_RE = re.compile(r"^\d/\d{1,2}\"?$|^\d{1,2}\"$|^X$")
OPC_RE = re.compile(r"^(I-\d{1,2}|[A-H])$")
NUM_RE = re.compile(r"^\d{2,4}[A-Z]?$")
TITLE_HINT = re.compile(r"THERMAX|DRAWING|REVIS|APPROV|CHECKED|DRAWN|SCALE|NTS|SKID MOUNTED|SPL/|DATE|^\d{2}/\d{2}/\d{4}$|AS BUILT|HAZOP|CLIENT")

CLASSES = ["equipment_tag", "instrument_tag", "valve_tag", "line_number", "pipe_size", "offpage_connector",
           "loop_number", "title_block", "note"]


def rule_label(text, in_title_block=False):
    t = text.strip().upper()
    if in_title_block or TITLE_HINT.search(t):
        return "title_block"
    if EQUIP_RE.match(t):
        return "equipment_tag"
    if VALVE_RE.match(t):
        return "valve_tag"
    if INSTR_RE.match(t):
        return "instrument_tag"
    if LINE_RE.match(t):
        return "line_number"
    if SIZE_RE.match(t):
        return "pipe_size"
    if OPC_RE.match(t):
        return "offpage_connector"
    if NUM_RE.match(t):
        return "loop_number"
    return "note"


def spans(ents, sheets, sheet_of):
    """Returns text spans with sheet, box, weak label."""
    out = []
    boxes = {s["sheet_no"]: s["box"] for s in sheets}
    for e in ents:
        if e["type"] not in ("TEXT", "MTEXT") or not e.get("text"):
            continue
        sh = sheet_of(e["x"], e["y"])
        tb = False
        if sh in boxes:
            x0, y0, x1, y1 = boxes[sh]
            tb = e["x"] > x1 - 95 and e["y"] < y0 + 70 or e["y"] < y0 + 28
        lab = rule_label(e["text"], tb)
        if tb and EQUIP_RE.match(e["text"].strip()):
            lab = "title_block"          # the equipment strip under the title block repeats tags
        out.append(dict(handle=e.get("handle"), text=e["text"], x=e["x"], y=e["y"], height=e.get("height", 0),
                        bbox=e.get("bbox"), sheet=sh, label=lab, source="dxf_text", confidence=1.0))
    return out
