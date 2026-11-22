"""
pipeline.py - the inference workflow, one function per step (mirrors the Azure P&ID design endpoints):

  ingest                 register the drawing (bronze), split sheets, cache parsed geometry
  symbol_detection       candidates -> glyph words removed -> bubbles / arrows by geometry -> classifier for the rest
  text_detection         DXF text + OCR of vector glyphs, labelled, and tied to symbols (tags)
  graph_construction     line detection + candidate connections (runs as a background job, job_status.json)
  graph_persistence      graph tables, ISA-95 asset hierarchy, BOM reconciliation, search index

Every step writes request.json / response.json / output_*.png under inference-results/<pid_id>/<step>/ and a
versioned run record in the store. A step reads the *corrected* output of the previous step when one exists, so an
engineer can fix results between steps (corrected.json), exactly as the design's API contract describes.
"""
import json
import math
import os
import pickle
import re
import threading
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

from . import bubbles as BB, dxf_reader as R, glyphs as GL, graph as GR, sheets as SH, symbols as Y, text as TX, training as TR
from .render import render

STEP_DIRS = {"symbol_detection": "symbol-detection", "text_detection": "text-detection",
             "graph_construction": "graph-construction", "graph_persistence": "graph-persistence"}
DEBUG = os.environ.get("DEBUG", "0") == "1"
LEXICON = {"516": "316", "S5": "SS", "5S": "SS", "SEPREX": "SEPREX", "C01": "CO1", "C03": "CO3", "JOV": "10V", "OSBL": "OSBL"}


def flow_role(o):
    """How flow behaves at an object: in-line piping items pass it on, instruments and equipment stop it,
    FROM / FEED connectors are sources, TO / VENT / DRAIN connectors are destinations."""
    if o["category"] == "Piping":
        return "through"
    if o["category"] == "Connector":
        t = (o.get("tag") or "").upper()
        if re.search(r"\bFROM\b|\bFEED\b", t):
            return "source"
        if re.search(r"\bTO\b|VENT|DRAIN|VACUUM|FLARE", t):
            return "to"
    return "sink"


ISA_ORDER = [(r"^[A-Z]E$|^[A-Z]G$", 0), (r"^[A-Z]T$|^[A-Z]DT$", 1), (r"^[A-Z]I$|^[A-Z]IC$|^[A-Z]C$", 2), (r"^[A-Z]Y$", 3),
             (r"^[A-Z]V$|^[A-Z]CV$|^HTR$|^[A-Z]Z$", 4)]


def isa_signal_direction(a, b, tag_of):
    """Signal direction inside one loop by ISA-5.1 function: element -> transmitter -> indicator/controller -> relay -> final element."""
    ma = re.match(r"^([A-Z]+)-(\d{3})", tag_of.get(a) or ""); mb = re.match(r"^([A-Z]+)-(\d{3})", tag_of.get(b) or "")
    if not ma or not mb or ma.group(2) != mb.group(2):
        return None
    ra = next((r for rx, r in ISA_ORDER if re.match(rx, ma.group(1))), None)
    rb = next((r for rx, r in ISA_ORDER if re.match(rx, mb.group(1))), None)
    if ra is None or rb is None or ra == rb:
        return None
    return f"{a}->{b}" if ra < rb else f"{b}->{a}"


STYLE_CLS = {"field": "Instrument/Bubble/Field-Discrete", "panel": "Instrument/Bubble/Panel-Discrete",
             "shared_display": "Instrument/Bubble/Field-SharedDisplay"}


class Foundry:
    def __init__(self, store, root):
        self.store = store
        self.root = root
        self.jobs = {}

    # ---------------- paths / io
    def step_dir(self, pid_id, step):
        d = os.path.join(self.root, "inference-results", pid_id, STEP_DIRS[step])
        os.makedirs(d, exist_ok=True)
        return d

    def _write(self, pid_id, step, name, obj):
        with open(os.path.join(self.step_dir(pid_id, step), name), "w") as f:
            json.dump(obj, f, indent=1, default=str)

    def _read(self, pid_id, step, name):
        p = os.path.join(self.step_dir(pid_id, step), name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def results(self, pid_id, step):
        """GetInferenceResults: corrected output first, otherwise the original response."""
        return self._read(pid_id, step, "corrected.json") or self._read(pid_id, step, "response.json")

    def correct(self, pid_id, step, obj):
        self._write(pid_id, step, "corrected.json", obj)

    def images(self, pid_id, step):
        d = self.step_dir(pid_id, step)
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))

    def cache_path(self, pid_id):
        return os.path.join(self.root, "cache", f"{pid_id}.pkl")

    def geometry(self, pid_id):
        with open(self.cache_path(pid_id), "rb") as f:
            return pickle.load(f)

    # ---------------- step 0: ingest
    def ingest(self, path, pid_id=None, title=None):
        sha = R.sha256(path)
        pid_id = pid_id or "PID-" + sha[:8].upper()
        doc = R.read(path)
        ents = [e for e in R.explode_inserts(doc) if "bbox" in e]
        sheets = SH.find_sheets(ents)
        os.makedirs(os.path.dirname(self.cache_path(pid_id)), exist_ok=True)
        with open(self.cache_path(pid_id), "wb") as f:
            pickle.dump(dict(ents=ents, sheets=sheets, header=doc["header"]), f)
        st = self.store
        st.x("DELETE FROM sheets WHERE pid_id=?", (pid_id,))
        st.many("INSERT INTO sheets VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(pid_id, s["sheet_no"], s.get("title"), s["kind"], s.get("drawing_no"), s.get("revision"), *s["box"]) for s in sheets])
        dwg = next((s["drawing_no"] for s in sheets if s.get("drawing_no")), None)
        rev = max((s["revision"] for s in sheets if s.get("revision")), default=None)
        st.x("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?)",
             (pid_id, os.path.basename(path), sha, "DXF " + str(doc["header"].get("$ACADVER")), os.path.getsize(path),
              title or "Skid Mounted Single Tube Reactor Unit", dwg, rev, len(sheets), st_now(),
              json.dumps(dict(entities=doc["stats"]["total"], by_type=dict(doc["stats"]["by_type"].most_common(12))))))
        return pid_id, sheets

    # ---------------- step 1: symbol detection
    def symbol_detection(self, pid_id, sheet_nos=None, equipment_boxes=None, ocr=True):
        st = self.store
        try:
            model_id, clf = TR.load_active(st)
        except Exception as ex:      # e.g. a model pickled with another scikit-learn version: retrain in the Models tab
            print("model could not be loaded:", ex)
            model_id, clf = None, None
        run, ver = st.start_run(pid_id, "symbol_detection", dict(sheets=sheet_nos, ocr=ocr), model_id)
        g = self.geometry(pid_id)
        ents, sheets = g["ents"], [s for s in g["sheets"] if s["kind"] == "process"]
        if sheet_nos:
            sheets = [s for s in sheets if s["sheet_no"] in sheet_nos]
        out_symbols, out_words, crops = [], [], []
        for s in sheets:
            t0 = time.time()
            box = s["box"]
            sub = [e for e in ents if box[0] - 1 <= e["bbox"][0] and e["bbox"][2] <= box[2] + 1 and box[1] - 1 <= e["bbox"][1] and e["bbox"][3] <= box[3] + 1]
            cands = Y.vector_candidates(sub, box, tol=0.15)
            ends = self._pipe_ends(sub)
            words, rest = GL.detect(cands, ocr=ocr)
            for w in words:
                out_words.append(dict(sheet_no=s["sheet_no"], text=LEXICON.get(w["text"], w["text"]), bbox=w["bbox"], vertical=w["vertical"],
                                      handles=[m.get("handle") for m in w["members"]][:20]))
            bub = {b["handle"]: b for b in BB.read_bubbles(sub, lambda x, y, n=s["sheet_no"]: n)} if ocr else {}
            k = 0
            for c in rest:
                b = c["bbox"]
                types = Counter(m["type"] for m in c["members"])
                circles = [m for m in c["members"] if m["type"] == "CIRCLE" and 2.5 <= m.get("r", 0) <= 3.6]
                img = Y.render_candidate(c)
                tag, method, props = None, "model", {}
                if len(circles) > 1:
                    # touching bubbles (TE / TIC / TY of one loop) come as one group: one symbol per circle
                    for ci in circles[1:]:
                        bi = bub.get(ci.get("handle"), {})
                        k += 1
                        out_symbols.append(dict(symbol_id=f"S{s['sheet_no']}-{k:03d}", sheet_no=s["sheet_no"], cls=STYLE_CLS[bi.get("style", "field")],
                                                category="Instrument", confidence=round(max(bi.get("confidence", 0.5), 0.5), 3), tag=bi.get("tag"), method="geometry+ocr",
                                                bbox=[round(v, 3) for v in (ci["x"] - ci["r"], ci["y"] - ci["r"], ci["x"] + ci["r"], ci["y"] + ci["r"])],
                                                props=dict(letters=bi.get("letters"), number=bi.get("number"), ocr_method=bi.get("method")), handles=[ci.get("handle")]))
                if circles:
                    bi = bub.get(circles[0].get("handle"), {})
                    style = bi.get("style") or BB.bubble_style(img)
                    cls = STYLE_CLS[style]
                    conf, tag, method = max(bi.get("confidence", 0.5), 0.5), bi.get("tag"), "geometry+ocr"
                    props = dict(letters=bi.get("letters"), number=bi.get("number"), ocr_method=bi.get("method"))
                    b = (circles[0]["x"] - circles[0]["r"], circles[0]["y"] - circles[0]["r"], circles[0]["x"] + circles[0]["r"], circles[0]["y"] + circles[0]["r"])
                elif set(types) == {"SOLID"} or (types.get("SOLID", 0) >= 1 and max(b[2] - b[0], b[3] - b[1]) < 3.2 and len(c["members"]) <= 3):
                    cls, conf, method = "Annotation/Arrow/FlowArrow", 0.9, "geometry"
                    sol = [m for m in c["members"] if m["type"] == "SOLID"][0]
                    pts = [(sol.get(kx), sol.get(ky)) for kx, ky in (("x", "y"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")) if sol.get(kx) is not None]
                    cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
                    tip = max(pts, key=lambda p: math.dist(p, cen))
                    props = dict(tip=tip, base=cen)
                elif clf is not None:
                    P = clf.predict_proba([Y.features(img)])[0]
                    j = int(P.argmax()); cls, conf = str(clf.classes_[j]), float(P[j])
                else:
                    cls, conf = "Unknown/Unknown/Unclassified", 0.0
                if cls.split("/")[0] in ("Text", "Line") and clf is not None and not circles:
                    if self._inline(c["bbox"], ends) >= 2 and max(b[2] - b[0], b[3] - b[1]) >= 1.2:
                        # sits between two pipe ends: an in-line item even if the model thinks it is text
                        P = clf.predict_proba([Y.features(img)])[0]
                        order = [j for j in P.argsort()[::-1] if str(clf.classes_[j]).split("/")[0] not in ("Text", "Line", "Annotation")]
                        cls, conf, method = str(clf.classes_[order[0]]), float(P[order[0]]) * 0.5, "inline-rule+model"
                    else:
                        continue
                k += 1
                sid = f"S{s['sheet_no']}-{k:03d}"
                out_symbols.append(dict(symbol_id=sid, sheet_no=s["sheet_no"], cls=cls, category=cls.split("/")[0], confidence=round(conf, 3),
                                        bbox=[round(v, 3) for v in b], tag=tag, method=method, props=props,
                                        handles=[m.get("handle") for m in c["members"]][:30]))
                crops.append((sid, img))
            # line patterns (heat tracing, jackets) repeat the same small mark along a line: not symbols
            out_symbols[:] = self._drop_patterns(out_symbols, s["sheet_no"])
            # equipment: tag labels + the outline they label
            out_symbols += self._equipment(s, sub)
            print(f"  sheet {s['sheet_no']}: {len(cands)} candidates, {len(words)} words, {time.time() - t0:.1f}s", flush=True)
        st.x("DELETE FROM symbols WHERE pid_id=? AND version=?", (pid_id, ver))
        st.many("INSERT INTO symbols VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(pid_id, o["symbol_id"], ver, o["sheet_no"], o["cls"], o["category"], o["confidence"], *o["bbox"], o.get("tag"),
                  o["method"], model_id, json.dumps(o["handles"]), "candidate", json.dumps(o.get("props", {}))) for o in out_symbols])
        cls_of = {o["symbol_id"]: o["cls"] for o in out_symbols}
        crops = [(sid, im) for sid, im in crops if sid in cls_of]
        st.x("DELETE FROM training_samples WHERE funnel='inference' AND pid_id=?", (pid_id,))
        st.many("INSERT INTO training_samples(label, funnel, pid_id, source_ref, png, approved, created) VALUES(?,?,?,?,?,?,?)",
                [(cls_of[sid], "inference", pid_id, sid, TR.to_png(im), 0, st_now()) for sid, im in crops])
        resp = dict(pid_id=pid_id, version=ver, model_id=model_id, symbols=out_symbols, glyph_words=out_words)
        self._write(pid_id, "symbol_detection", "request.json", dict(pid_id=pid_id, sheets=sheet_nos, equipment_boxes=equipment_boxes))
        self._write(pid_id, "symbol_detection", "response.json", resp)
        for s in sheets:
            self._overlay(pid_id, "symbol_detection", s, out_symbols, f"output_{pid_id}_S{s['sheet_no']:02d}_symbol_detection.png")
        summary = dict(symbols=len(out_symbols), words=len(out_words), by_category=Counter(o["category"] for o in out_symbols))
        st.finish_run(run, outputs=summary)
        return resp

    @staticmethod
    def _drop_patterns(objs, sheet_no, min_run=4):
        mine = [o for o in objs if o["sheet_no"] == sheet_no and o["method"] in ("model", "inline-rule+model")]
        drop = set()
        for axis in (0, 1):
            groups = defaultdict(list)
            for o in mine:
                b = o["bbox"]
                c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                groups[(round(c[axis] / 0.6), round((b[2] - b[0]) / 0.4), round((b[3] - b[1]) / 0.4))].append(o)
            for g in groups.values():
                if len(g) >= min_run:
                    drop.update(o["symbol_id"] for o in g)
        return [o for o in objs if o["symbol_id"] not in drop]

    @staticmethod
    def _pipe_ends(sub, min_len=3.0):
        grid = defaultdict(list)
        for e in sub:
            if e["type"] == "LINE":
                pts = [(e.get("x", 0), e.get("y", 0)), (e.get("x2", 0), e.get("y2", 0))]
            elif e["type"] == "LWPOLYLINE" and len(e.get("pts", [])) >= 2 and not e.get("closed"):
                pts = [e["pts"][0], e["pts"][-1]]
            else:
                continue
            if math.dist(pts[0], pts[-1]) < min_len and e["type"] == "LINE":
                continue
            b = e["bbox"]
            if max(b[2] - b[0], b[3] - b[1]) < min_len:
                continue
            for p in pts:
                grid[(int(p[0] // 5), int(p[1] // 5))].append(p)
        return grid

    @staticmethod
    def _inline(b, grid, tol=0.6):
        n = 0
        for gx in range(int((b[0] - tol) // 5), int((b[2] + tol) // 5) + 1):
            for gy in range(int((b[1] - tol) // 5), int((b[3] + tol) // 5) + 1):
                for p in grid.get((gx, gy), []):
                    if b[0] - tol <= p[0] <= b[2] + tol and b[1] - tol <= p[1] <= b[3] + tol:
                        n += 1
        return n

    def _equipment(self, s, sub):
        box = s["box"]
        labels = [e for e in sub if e["type"] in ("TEXT", "MTEXT") and TX.EQUIP_RE.match(e.get("text", "").strip())
                  and e.get("height", 0) >= 2.0 and e["y"] > box[1] + 28]
        big = [e for e in sub if e["type"] in ("LWPOLYLINE", "LINE", "ARC", "CIRCLE", "ELLIPSE") and 6 < max(e["bbox"][2] - e["bbox"][0], e["bbox"][3] - e["bbox"][1]) < 160
               and e["bbox"][1] > box[1] + 28]
        out, seen = [], set()
        for i, t in enumerate(labels):
            tag = t["text"].strip()
            if tag in seen:
                continue
            seen.add(tag)
            best = None
            for e in big:
                b = e["bbox"]
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                inside = b[0] <= t["x"] <= b[2] and b[1] <= t["y"] <= b[3]
                d = 0 if inside else math.dist((cx, cy), (t["x"], t["y"]))
                score = d - (0.02 * (b[2] - b[0]) * (b[3] - b[1]) if inside else 0)
                if d < 35 and (best is None or score < best[0]):
                    best = (score, e)
            if best:
                e = best[1]
                # grow the outline with touching big entities (vessel = shell + heads)
                bb = list(e["bbox"])
                core = list(bb)
                grow = (core[2] - core[0]) * 0.3 + 1, (core[3] - core[1]) * 0.3 + 1
                for o in big:
                    ob = o["bbox"]
                    oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
                    if not (core[0] - grow[0] <= oc[0] <= core[2] + grow[0] and core[1] - grow[1] <= oc[1] <= core[3] + grow[1]):
                        continue
                    if ob[2] - ob[0] > 2 * (core[2] - core[0]) + 4 or ob[3] - ob[1] > 2 * (core[3] - core[1]) + 4:
                        continue
                    if ob[0] <= bb[2] + 0.3 and bb[0] <= ob[2] + 0.3 and ob[1] <= bb[3] + 0.3 and bb[1] <= ob[3] + 0.3 and max(ob[2] - ob[0], ob[3] - ob[1]) < 80:
                        bb = [min(bb[0], ob[0]), min(bb[1], ob[1]), max(bb[2], ob[2]), max(bb[3], ob[3])]
                conf = 0.85 if best[0] <= 0 else 0.6
            else:
                bb, conf = [t["x"] - 5, t["y"] - 5, t["x"] + 15, t["y"] + 5], 0.4
            out.append(dict(symbol_id=f"E{s['sheet_no']}-{tag}", sheet_no=s["sheet_no"], cls="Equipment/Vessel-or-Unit/" + tag.split("-")[0],
                            category="Equipment", confidence=conf, bbox=[round(v, 3) for v in bb], tag=tag, method="text+outline",
                            props=dict(label_xy=(t["x"], t["y"])), handles=[t.get("handle")] + ([best[1].get("handle")] if best else [])))
        return out

    def _overlay(self, pid_id, step, s, objs, name, edges=None):
        g = self.geometry(pid_id)
        box = s["box"]; px = 4.0
        sub = [e for e in g["ents"] if box[0] - 1 <= e["bbox"][0] and e["bbox"][2] <= box[2] + 1 and box[1] - 1 <= e["bbox"][1] and e["bbox"][3] <= box[3] + 1]
        img = cv2.cvtColor(render(sub, box, px_per_unit=px), cv2.COLOR_GRAY2BGR)
        col = {"Instrument": (200, 90, 0), "Piping": (0, 140, 0), "Equipment": (0, 0, 220), "Annotation": (150, 150, 150), "Text": (180, 0, 180)}
        P = lambda x, y: (int((x - box[0]) * px), int((box[3] - y) * px))
        for o in objs:
            if o.get("sheet_no") != s["sheet_no"]:
                continue
            b = o["bbox"]
            c = col.get(o.get("category"), (0, 120, 220))
            cv2.rectangle(img, P(b[0], b[3]), P(b[2], b[1]), c, 2)
            lab = o.get("tag") or o.get("text") or o["cls"].split("/")[-1]
            cv2.putText(img, str(lab)[:14], P(b[0], b[3] + 0.6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)
        for e in edges or []:
            a, b = e.get("_a"), e.get("_b")
            if a and b:
                col_e = (0, 165, 255) if e["kind"] == "process" else (255, 0, 255)
                d = e.get("direction") or "unknown"
                if "->" in d:
                    if d.split("->")[0] != e["src"]:
                        a, b = b, a
                    col_e = (0, 120, 0) if e["kind"] == "process" else (200, 0, 200)
                    cv2.arrowedLine(img, P(*a), P(*b), col_e, 2, cv2.LINE_AA, tipLength=min(0.3, 18.0 / max(1.0, math.dist(P(*a), P(*b)))))
                else:
                    cv2.line(img, P(*a), P(*b), col_e, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.step_dir(pid_id, step), name), img)

    # ---------------- step 2: text detection
    def text_detection(self, pid_id, corrected_symbols=None):
        st = self.store
        run, ver = st.start_run(pid_id, "text_detection", {})
        if corrected_symbols is not None:
            self.correct(pid_id, "symbol_detection", corrected_symbols)
        sym = self.results(pid_id, "symbol_detection")
        g = self.geometry(pid_id)
        sheets = g["sheets"]
        f = SH.sheet_lookup(sheets)
        spans = TX.spans(g["ents"], sheets, f)
        for w in sym.get("glyph_words", []):
            b = w["bbox"]
            spans.append(dict(handle=(w.get("handles") or [None])[0], text=w["text"], x=b[0], y=b[1], height=b[3] - b[1], bbox=b,
                              sheet=w["sheet_no"], label=TX.rule_label(w["text"]), source="ocr_glyph", confidence=0.7))
        symbols = sym["symbols"]
        # tie text to symbols: equipment tags already carry theirs; valves / reliefs take prefix + number nearby
        by_sheet = defaultdict(list)
        for i, t in enumerate(spans):
            by_sheet[t["sheet"]].append(i)
        for o in symbols:
            if o["category"] not in ("Piping", "Instrument") or o.get("tag"):
                continue
            b = o["bbox"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            near = sorted((math.dist((cx, cy), ((spans[i]["bbox"][0] + spans[i]["bbox"][2]) / 2, (spans[i]["bbox"][1] + spans[i]["bbox"][3]) / 2)), i)
                          for i in by_sheet[o["sheet_no"]])
            prefix = next((spans[i]["text"].rstrip("-") for d, i in near if d < 6 and re.fullmatch(r"(BV|NV|NRV|PRV|PSV|RD|BPR|PCV|FV|KV|F|MFC|TV)-?", spans[i]["text"])), None)
            number = next((spans[i]["text"] for d, i in near if d < 7 and re.fullmatch(r"\d{3}[A-Z]?", spans[i]["text"])), None)
            if prefix and number:
                o["tag"] = f"{prefix}-{number}"; o.setdefault("props", {})["tag_method"] = "prefix+number within 7 units"
            elif prefix:
                o.setdefault("props", {})["tag_prefix"] = prefix
        for i, t in enumerate(spans):
            t["text_id"] = f"T{t['sheet'] or 0}-{i:04d}"
        st.x("DELETE FROM texts WHERE pid_id=? AND version=?", (pid_id, ver))
        st.many("INSERT INTO texts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(pid_id, t["text_id"], ver, t["sheet"], t["text"], t["label"], t["confidence"], *(t["bbox"] or (t["x"], t["y"], t["x"], t["y"])), t["source"], t["handle"], None) for t in spans])
        # tags found by text association go back onto the symbols table
        sv = st.latest_version(pid_id, "symbol_detection")
        st.many("UPDATE symbols SET tag=?, props=? WHERE pid_id=? AND symbol_id=? AND version=?",
                [(o.get("tag"), json.dumps(o.get("props", {})), pid_id, o["symbol_id"], sv) for o in symbols if o.get("tag")])
        resp = dict(pid_id=pid_id, version=ver, texts=spans, symbols=symbols)
        self._write(pid_id, "text_detection", "request.json", dict(pid_id=pid_id, corrected_symbols=corrected_symbols is not None))
        self._write(pid_id, "text_detection", "response.json", resp)
        for s in [s for s in sheets if s["kind"] == "process"]:
            objs = [dict(t, sheet_no=t["sheet"], category="Text", cls=t["label"]) for t in spans if t["bbox"] and t["label"] not in ("title_block",)]
            self._overlay(pid_id, "text_detection", s, objs, f"output_{pid_id}_S{s['sheet_no']:02d}_text_detection.png")
        st.finish_run(run, outputs=dict(texts=len(spans), by_label=Counter(t["label"] for t in spans), tagged_symbols=sum(1 for o in symbols if o.get("tag"))))
        return resp

    # ---------------- step 3: line detection + graph construction (job)
    def post_graph_construction(self, pid_id, corrected_text=None):
        if corrected_text is not None:
            self.correct(pid_id, "text_detection", corrected_text)
        self._write(pid_id, "graph_construction", "job_status.json", dict(status="Submitted", ts=st_now()))
        th = threading.Thread(target=self._graph_job, args=(pid_id,), daemon=True)
        self.jobs[pid_id] = th
        th.start()
        return 202

    def job_status(self, pid_id):
        return self._read(pid_id, "graph_construction", "job_status.json") or dict(status="NotFound")

    def _graph_job(self, pid_id):
        try:
            self._write(pid_id, "graph_construction", "job_status.json", dict(status="Running", ts=st_now()))
            self.graph_construction(pid_id)
            self._write(pid_id, "graph_construction", "job_status.json", dict(status="Succeeded", ts=st_now()))
        except Exception as ex:
            self._write(pid_id, "graph_construction", "job_status.json", dict(status="Failed", error=str(ex), ts=st_now()))

    def graph_construction(self, pid_id):
        st = self.store
        run, ver = st.start_run(pid_id, "graph_construction", {})
        td = self.results(pid_id, "text_detection")
        g = self.geometry(pid_id)
        symbols = td["symbols"]
        texts = td["texts"]
        all_edges, all_lines, stats = [], [], {}
        for s in [s for s in g["sheets"] if s["kind"] == "process"]:
            box = s["box"]
            sub = [e for e in g["ents"] if box[0] - 1 <= e["bbox"][0] and e["bbox"][2] <= box[2] + 1 and box[1] - 1 <= e["bbox"][1] and e["bbox"][3] <= box[3] + 1]
            objs = [o for o in symbols if o["sheet_no"] == s["sheet_no"] and o["category"] != "Annotation"]
            arrows = [dict(tip=tuple(o["props"]["tip"]), base=tuple(o["props"]["base"])) for o in symbols
                      if o["sheet_no"] == s["sheet_no"] and o["cls"] == "Annotation/Arrow/FlowArrow" and o.get("props", {}).get("tip")]
            # off-page connectors: I-n / A..H labels and TO/FROM boxes become connector objects
            for t in texts:
                if t["sheet"] == s["sheet_no"] and t["label"] == "offpage_connector" and t["bbox"]:
                    b = t["bbox"]
                    objs.append(dict(symbol_id=f"C{s['sheet_no']}-{t['text']}-{t['text_id'][-4:]}", sheet_no=s["sheet_no"], cls="Connector/OffPage/" + t["text"],
                                     category="Connector", bbox=[b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2], tag=t["text"], confidence=0.7))
            objs += self._connectors(s, sub, texts)
            excl = [o["bbox"] for o in objs if o["category"] in ("Piping", "Instrument") and (o["bbox"][2] - o["bbox"][0]) < 15]
            excl += [t["bbox"] for t in texts if t["sheet"] == s["sheet_no"] and t["bbox"] and t["source"] == "ocr_glyph"]
            eq_boxes = [o["bbox"] for o in objs if o["category"] == "Equipment"]
            segs = GR.segments_from_entities(sub, box, excl)
            # drop segments drawn inside an equipment outline (internals, hatching)
            segs = [sg for sg in segs if not any(b[0] + 0.5 < sg["p"][0] < b[2] - 0.5 and b[1] + 0.5 < sg["p"][1] < b[3] - 0.5 and
                                                 b[0] + 0.5 < sg["q"][0] < b[2] - 0.5 and b[1] + 0.5 < sg["q"][1] < b[3] - 0.5 for b in eq_boxes)]
            edges, stt = GR.build([dict(id=o["symbol_id"], bbox=o["bbox"], role=flow_role(o),
                                        kind=("equipment" if o["category"] == "Equipment" else "offpage" if o["category"] == "Connector" else "symbol")) for o in objs],
                                  segs, arrows, sheet_no=s["sheet_no"])
            tag_of = {o["symbol_id"]: o.get("tag") for o in objs}
            for e in edges:
                if e["direction"] == "unknown":
                    d = isa_signal_direction(e["src"], e["dst"], tag_of)
                    if d:
                        e["direction"], e["direction_method"], e["direction_confidence"] = d, "ISA-5.1 loop order", 0.6
            stats[s["sheet_no"]] = stt
            cen = {o["symbol_id"]: ((o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2) for o in objs}
            for e in edges:
                e["sheet_no"] = s["sheet_no"]
                if e.get("hub") and e["hub"] not in cen:
                    pts = [cen[x] for x in (e["src"], e["dst"]) if x in cen]
                    cen[e["hub"]] = pts[0] if pts else (0, 0)
                e["_a"], e["_b"] = cen.get(e["src"]), cen.get(e["dst"])
            for i, sg in enumerate(segs):
                all_lines.append(dict(line_id=f"L{s['sheet_no']}-{i:04d}", sheet_no=s["sheet_no"], kind="signal" if sg["style"] == "dashed" else "process",
                                      style=sg["style"], p=sg["p"], q=sg["q"], handles=sg.get("handles") or [sg.get("handle")]))
            for o in objs:
                if o["category"] == "Connector" and o not in symbols:
                    symbols.append(o)
            all_edges += edges
            self._overlay(pid_id, "graph_construction", s, objs, f"output_{pid_id}_S{s['sheet_no']:02d}_graph-construction.png", edges)
        # cross-sheet off-page links: same connector label on two sheets
        conns = defaultdict(list)
        for o in symbols:
            if o["category"] == "Connector":
                conns[o["tag"]].append(o["symbol_id"])
        for lab, ids in list(conns.items()):
            if not re.fullmatch(r"I-\d+|[A-H]", lab or ""):
                continue
            sh = sorted(set(ids))
            if 2 <= len(sh) <= 3:
                for a, b in zip(sh, sh[1:]):
                    if a.split("-")[0] != b.split("-")[0]:
                        all_edges.append(dict(src=a, dst=b, kind="offpage", direction="unknown", confidence=0.55, evidence=[], sheet_no=None))
        eq_by_tag = {o["tag"]: o["symbol_id"] for o in symbols if o["category"] == "Equipment"}
        for o in symbols:
            tgt = (o.get("props") or {}).get("target") if o["category"] == "Connector" else None
            if tgt and tgt in eq_by_tag and eq_by_tag[tgt].split("-")[0][1:] != str(o["sheet_no"]):
                d = f"{o['symbol_id']}->{eq_by_tag[tgt]}" if o["cls"].endswith("To") else f"{eq_by_tag[tgt]}->{o['symbol_id']}" if o["cls"].endswith("From") else "unknown"
                all_edges.append(dict(src=o["symbol_id"], dst=eq_by_tag[tgt], kind="offpage", direction=d, confidence=0.7, evidence=o.get("handles", []), sheet_no=o["sheet_no"],
                                      direction_method="TO/FROM connector" if d != "unknown" else None, direction_confidence=0.8 if d != "unknown" else 0.0))
        for i, e in enumerate(all_edges):
            e["edge_id"] = f"G{i:05d}"
        st.x("DELETE FROM lines WHERE pid_id=? AND version=?", (pid_id, ver))
        st.many("INSERT INTO lines VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [(pid_id, l["line_id"], ver, l["sheet_no"], l["kind"], l["style"], l["p"][0], l["p"][1], l["q"][0], l["q"][1], json.dumps(l["handles"][:20])) for l in all_lines])
        resp = dict(pid_id=pid_id, version=ver, lines=len(all_lines), edges=[{k: v for k, v in e.items() if not k.startswith("_")} for e in all_edges],
                    symbols=symbols, stats=stats)
        self._write(pid_id, "graph_construction", "request.json", dict(pid_id=pid_id))
        self._write(pid_id, "graph_construction", "response_line-detection.json", dict(lines=[dict(l, p=list(l["p"]), q=list(l["q"])) for l in all_lines[:20000]]))
        self._write(pid_id, "graph_construction", "response_graph-construction.json", resp)
        self._write(pid_id, "graph_construction", "response.json", resp)
        st.finish_run(run, outputs=dict(lines=len(all_lines), edges=len(all_edges), by_kind=Counter(e["kind"] for e in all_edges)))
        return resp

    def _connectors(self, s, sub, texts):
        """Off-page connector boxes: a pentagon / arrow outline holding 'TO x' or 'FROM x'."""
        out = []
        polys = [e for e in sub if e["type"] == "LWPOLYLINE" and 5 <= len(e.get("pts", [])) <= 9 and 8 < (e["bbox"][2] - e["bbox"][0]) + (e["bbox"][3] - e["bbox"][1]) < 60]
        words = [t for t in texts if t["sheet"] == s["sheet_no"] and t["bbox"]]
        k = 0
        for pl in polys:
            b = pl["bbox"]
            inside = [t for t in words if b[0] - 0.3 <= (t["bbox"][0] + t["bbox"][2]) / 2 <= b[2] + 0.3 and b[1] - 0.3 <= (t["bbox"][1] + t["bbox"][3]) / 2 <= b[3] + 0.3]
            txt = " ".join(t["text"] for t in sorted(inside, key=lambda t: (t["bbox"][0], -t["bbox"][1])))
            if not re.search(r"\b(TO|FROM)\b|VENT|DRAIN|OSBL|FEED", txt):
                continue
            k += 1
            target = re.search(r"([A-Z]{1,4}-\d{3}[A-Z]?)", txt.replace(" -", "-").replace("- ", "-"))
            page = re.search(r"PAGE\s*(\d+)", txt)
            out.append(dict(symbol_id=f"C{s['sheet_no']}-{k:02d}", sheet_no=s["sheet_no"], cls="Connector/OffPage/" + ("To" if "TO" in txt.split() else "From" if "FROM" in txt else "Terminal"),
                            category="Connector", bbox=list(b), tag=txt[:40], confidence=0.75, method="outline+text",
                            props=dict(target=target.group(1) if target else None, page=int(page.group(1)) if page else None), handles=[pl.get("handle")]))
        return out

    # ---------------- step 4: persistence + hierarchy + reconciliation
    def graph_persistence(self, pid_id, corrected_graph=None, bom_path=None, site="R&D Pilot Facility (to be confirmed)"):
        from . import hierarchy as HI
        st = self.store
        run, ver = st.start_run(pid_id, "graph_persistence", {})
        if corrected_graph is not None:
            self.correct(pid_id, "graph_construction", corrected_graph)
        gc = self.results(pid_id, "graph_construction")
        symbols, edges = gc["symbols"], gc["edges"]
        status = {r["symbol_id"]: r["status"] for r in st.q("SELECT symbol_id, status FROM symbols WHERE pid_id=? AND version=?", (pid_id, st.latest_version(pid_id, "symbol_detection")))}
        nodes = []
        for o in symbols:
            b = o["bbox"]
            nodes.append((pid_id, ver, o["symbol_id"], o["category"], o.get("tag") or o["cls"].split("/")[-1], o["cls"], o["sheet_no"],
                          (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, o.get("confidence"), status.get(o["symbol_id"], "candidate"),
                          json.dumps(dict(bbox=b, method=o.get("method"), props=o.get("props", {})))))
        hubs = {e["hub"] for e in edges if e.get("hub")}
        for h in hubs:
            nodes.append((pid_id, ver, h, "Junction", h, "Piping/Junction/Header", int(h.split("-")[0][1:]), None, None, 0.6, "candidate", "{}"))
        st.x("DELETE FROM graph_nodes WHERE pid_id=? AND version=?", (pid_id, ver))
        st.x("DELETE FROM graph_edges WHERE pid_id=? AND version=?", (pid_id, ver))
        st.many("INSERT INTO graph_nodes VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", nodes)
        st.many("INSERT INTO graph_edges VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(pid_id, ver, e["edge_id"], e["src"], e["dst"], e["kind"], e["direction"], e["confidence"], "candidate",
                  json.dumps(dict(handles=(e.get("evidence") or [])[:30], direction_method=e.get("direction_method"),
                                  direction_confidence=e.get("direction_confidence")))) for e in edges])
        assets, recon = HI.build(st, pid_id, symbols, edges, bom_path=bom_path, site=site)
        checks = HI.drawing_checks(symbols, edges, assets)
        st.kv_set(f"checks:{pid_id}", checks)
        # search index
        if st.fts:
            st.x("DELETE FROM search WHERE pid_id=?", (pid_id,))
        rows = [(pid_id, "asset", a["asset_id"], " ".join(str(x) for x in (a["tag"], a["name"], a["asset_class"], a["level"]) if x), a["sheet_no"]) for a in assets]
        rows += [(pid_id, "text", r["text_id"], r["text"], r["sheet_no"]) for r in st.q("SELECT text_id, text, sheet_no FROM texts WHERE pid_id=? AND version=? AND label!='title_block'", (pid_id, st.latest_version(pid_id, "text_detection")))]
        st.index(rows)
        out = dict(nodes=len(nodes), edges=len(edges), assets=len(assets), checks=len(checks), reconciliation=recon["summary"])
        self._write(pid_id, "graph_persistence", "response.json", out)
        st.finish_run(run, outputs=out)
        return out

    def run_all(self, path, pid_id=None, bom_path=None, ocr=True, sheet_nos=None):
        pid_id, _ = self.ingest(path, pid_id)
        self.symbol_detection(pid_id, sheet_nos=sheet_nos, ocr=ocr)
        self.text_detection(pid_id)
        self.graph_construction(pid_id)
        return pid_id, self.graph_persistence(pid_id, bom_path=bom_path)


def st_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
