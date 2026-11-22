"""
copilot.py - the P&ID chatbot.

Two modes over the same tools (all read the Foundry store: symbols, graph, hierarchy, BOM reconciliation, checks):
  * offline  - a small intent model (TF-IDF + logistic regression, trained on example questions at start-up) picks a
               tool, tags / sheet numbers are pulled from the question, and the answer is written from the result.
               No key, no network.
  * Claude   - with ANTHROPIC_API_KEY (Streamlit Secrets), Claude plans tool calls and writes the answer; it may only
               state what the tools return.
Every answer carries its evidence: sheet, object id, confidence, review status.
"""
import difflib
import json
import os
import re
from collections import Counter, defaultdict

import networkx as nx

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 8
TAG_RE = re.compile(r"\b([A-Z]{1,4}-\d{2,4}[A-Z]?)\b")

SYSTEM = """You are the P&ID copilot for a digitised Thermax single tube reactor P&ID (skid mounted unit, 9 process sheets).
Answer only from tool results. Everything extracted is a candidate until an engineer approves it: say so when it matters,
and give confidence and sheet numbers. When a tag is missing, say what the tools show (similar tags, OCR limits), never guess.
Be brief and practical; engineers will check your answer against the drawing."""

TOOLS = [
    dict(name="find_tag", description="Everything known about one tag or object id: class, sheet, confidence, review status, parent in the asset hierarchy, BOM status, connections, rule checks.",
         input_schema=dict(type="object", properties=dict(tag=dict(type="string")), required=["tag"])),
    dict(name="connections", description="Objects connected to a tag through candidate process / signal / off-page connections, up to N hops.",
         input_schema=dict(type="object", properties=dict(tag=dict(type="string"), hops=dict(type="integer")), required=["tag"])),
    dict(name="flow", description="Upstream or downstream items of a tag following flow directions (arrows, FROM/TO connectors, ISA loop order). direction: upstream | downstream.",
         input_schema=dict(type="object", properties=dict(tag=dict(type="string"), direction=dict(type="string"), hops=dict(type="integer")), required=["tag", "direction"])),
    dict(name="path", description="Shortest connection path between two tags.",
         input_schema=dict(type="object", properties=dict(start=dict(type="string"), end=dict(type="string")), required=["start", "end"])),
    dict(name="hierarchy", description="Children of a tag or asset in the ISA-95 hierarchy (equipment -> loops -> instruments / valves).",
         input_schema=dict(type="object", properties=dict(tag=dict(type="string")), required=["tag"])),
    dict(name="loops", description="Instrument control loops (tags sharing a loop number), optionally for one sheet.",
         input_schema=dict(type="object", properties=dict(sheet=dict(type="integer")))),
    dict(name="sheet_summary", description="What is on a sheet: title, equipment, instruments, valves, connections, open checks.",
         input_schema=dict(type="object", properties=dict(sheet=dict(type="integer")), required=["sheet"])),
    dict(name="bom_status", description="BOM reconciliation. status: matched | bom_only | drawing_only | all; optional category (Valve, Instrument, Safety, Equipment, Filter, Heater).",
         input_schema=dict(type="object", properties=dict(status=dict(type="string"), category=dict(type="string")))),
    dict(name="why_missing", description="Explain why a BOM tag was not found on the drawing: similar tags read on the drawing, text near the expected sheet.",
         input_schema=dict(type="object", properties=dict(tag=dict(type="string")), required=["tag"])),
    dict(name="drawing_checks", description="Rule checks to review: duplicate tags, untagged bubbles, relief devices without tag, controllers without sensor, unconnected equipment.",
         input_schema=dict(type="object", properties=dict(rule=dict(type="string")))),
    dict(name="review_queue", description="Lowest-confidence detections waiting for review.",
         input_schema=dict(type="object", properties=dict(limit=dict(type="integer")))),
    dict(name="search", description="Full-text search over tags, classes and drawing text.",
         input_schema=dict(type="object", properties=dict(query=dict(type="string")), required=["query"])),
    dict(name="stats", description="Overall digitisation statistics.", input_schema=dict(type="object", properties={})),
    dict(name="propose_correction", description="Record a proposed correction for an engineer to approve (does not change data). target = tag or object id.",
         input_schema=dict(type="object", properties=dict(target=dict(type="string"), new_class=dict(type="string"), new_tag=dict(type="string"),
                                                          reason=dict(type="string")), required=["target", "reason"])),
]

EXAMPLES = {
    "find_tag": ["what is TE-202", "tell me about R-300", "details of PSV-201", "where is V-200", "show TIC-401", "info on BV-213",
                 "which sheet has HE-400", "what kind of item is PT-201", "how confident is LI-201", "is NRV-202 approved"],
    "connections": ["what is connected to TIC-202", "connections of V-500", "what feeds R-400", "neighbours of PT-201",
                    "what is TE-205 linked to", "lines from FRN-200", "neighbors of HE-200"],
    "flow": ["what is upstream of R-400", "downstream of V-200", "where does the flow from SM-200 go", "what feeds into HE-400 upstream",
             "flow direction after NRV-202", "what comes before V-500", "what is after FRN-200 in the flow", "source of flow to R-300"],
    "path": ["path from V-200 to R-300", "how is SM-200 connected to FRN-200", "route between HE-400 and V-500", "trace from FROM MFC to R-300"],
    "hierarchy": ["what is under R-300", "children of V-200", "instruments on R-400", "what belongs to HE-400", "hierarchy of FRN-300",
                  "valves of V-500", "which instruments are on the reactor"],
    "loops": ["loops on sheet 3", "list control loops", "temperature loops", "which loops are on sheet 4", "show all loops", "control loops of the furnace"],
    "sheet_summary": ["what is on sheet 5", "summarise sheet 3", "sheet 7 contents", "overview of sheet 2", "describe sheet 6", "what does sheet 4 show"],
    "bom_status": ["which BOM items are missing", "valves not found on drawing", "BOM reconciliation", "what is only on the drawing",
                   "how many BOM tags matched", "missing instruments", "compare with BOM", "which safety valves matched"],
    "why_missing": ["why is TE-306 missing", "why was BV-208 not found", "where is FV-101 on the drawing", "why can't you find KV-501",
                    "TE-307 not detected why", "explain missing PRV-101"],
    "drawing_checks": ["any problems in the drawing", "duplicate tags", "drawing checks", "which relief devices have no tag", "errors found",
                       "untagged bubbles", "what should I check", "issues on the P&ID"],
    "review_queue": ["what needs review", "lowest confidence items", "review queue", "what should I approve first", "uncertain detections"],
    "search": ["search needle valve", "find vent", "look for SS 316", "search for chiller", "find nitrogen", "anything about vacuum"],
    "stats": ["how good is the digitisation", "statistics", "how many symbols were found", "summary of results", "how accurate is it", "overall status"],
    "help": ["help", "what can you do", "hi", "hello", "what questions can I ask"],
}


class Copilot:
    def __init__(self, store, pid_id):
        self.S, self.pid = store, pid_id
        self.v = {k: store.latest_version(pid_id, k) for k in ("symbol_detection", "graph_persistence")}
        self.nodes = {n["node_id"]: n for n in store.q("SELECT * FROM graph_nodes WHERE pid_id=? AND version=?", (pid_id, self.v["graph_persistence"]))}
        self.edges = store.q("SELECT * FROM graph_edges WHERE pid_id=? AND version=? AND status!='rejected'", (pid_id, self.v["graph_persistence"]))
        self.assets = {a["asset_id"]: a for a in store.q("SELECT * FROM assets WHERE pid_id=?", (pid_id,))}
        self.sheets = {s["sheet_no"]: s for s in store.q("SELECT * FROM sheets WHERE pid_id=?", (pid_id,))}
        self.recon = store.kv_get(f"recon:{pid_id}", {"rows": [], "summary": {}})
        self.checks = store.kv_get(f"checks:{pid_id}", [])
        self.D = nx.DiGraph()
        for e in self.edges:
            d = e.get("direction") or "unknown"
            if "->" in d:
                a, b = d.split("->")
                ev = json.loads(e.get("evidence") or "{}")
                ev = ev if isinstance(ev, dict) else {}
                self.D.add_edge(a, b, kind=e["kind"], how=ev.get("direction_method"), conf=ev.get("direction_confidence"), id=e["edge_id"])
        self.G = nx.Graph()
        for n in self.nodes:
            self.G.add_node(n)
        for e in self.edges:
            self.G.add_edge(e["src"], e["dst"], kind=e["kind"], conf=e["confidence"], id=e["edge_id"])
        self.by_tag = defaultdict(list)
        for n in self.nodes.values():
            if n["label"]:
                self.by_tag[n["label"].upper()].append(n["node_id"])
        self._nlu = None

    # ------------------------------------------------------------------ helpers
    def _ids(self, tag):
        t = (tag or "").strip().upper()
        if t in self.by_tag:
            return self.by_tag[t]
        if tag in self.nodes:
            return [tag]
        return []

    def _brief(self, nid):
        n = self.nodes.get(nid, {})
        return dict(id=nid, tag=n.get("label"), cls=(n.get("cls") or "").split("/")[-1], sheet=n.get("sheet_no"),
                    confidence=n.get("confidence"), status=n.get("status"))

    def _bom_row(self, tag):
        return next((r for r in self.recon.get("rows", []) if r["tag"] == (tag or "").upper()), None)

    # ------------------------------------------------------------------ tools
    def find_tag(self, tag):
        ids = self._ids(tag)
        if not ids:
            near = difflib.get_close_matches(tag.upper(), list(self.by_tag), n=5, cutoff=0.6)
            return dict(found=False, tag=tag, similar_tags_on_drawing=near, bom=self._bom_row(tag))
        out = []
        for i in ids:
            n = self.nodes[i]
            a = self.assets.get(i, {})
            parent = self.assets.get(a.get("parent_id"), {})
            props = json.loads(n.get("props") or "{}")
            out.append(dict(self._brief(i), full_class=n["cls"], method=props.get("method"),
                            parent=dict(id=parent.get("asset_id"), tag=parent.get("tag") or parent.get("name"), level=parent.get("level")),
                            placement=json.loads(a.get("props") or "{}").get("placement"),
                            sheet_title=(self.sheets.get(n["sheet_no"]) or {}).get("title"),
                            connections=[dict(self._brief(m), via=self.G.edges[i, m]["kind"]) for m in self.G.neighbors(i)][:15],
                            checks=[c for c in self.checks if i in c.get("objects", []) or c.get("tag") == n["label"]]))
        return dict(found=True, tag=tag, objects=out, bom=self._bom_row(tag))

    def connections(self, tag, hops=1):
        ids = self._ids(tag)
        if not ids:
            return dict(found=False, tag=tag, similar=difflib.get_close_matches(tag.upper(), list(self.by_tag), n=5, cutoff=0.6))
        res = []
        for i in ids:
            reach = nx.single_source_shortest_path_length(self.G, i, cutoff=max(1, min(int(hops or 1), 3)))
            res.append(dict(object=self._brief(i), connected=[dict(self._brief(m), hops=d) for m, d in sorted(reach.items(), key=lambda kv: kv[1]) if m != i][:25],
                            edges=[dict(edge=self.G.edges[i, m]["id"], kind=self.G.edges[i, m]["kind"], confidence=self.G.edges[i, m]["conf"]) for m in self.G.neighbors(i)][:15]))
        return dict(found=True, results=res, note="Connections are candidates from line tracing unless their status is approved.")

    def flow(self, tag, direction="downstream", hops=3):
        ids = [i for i in self._ids(tag) if i in self.D]
        if not self._ids(tag):
            return dict(found=False, tag=tag)
        if not ids:
            return dict(found=True, tag=tag, items=[], note="No flow direction is known for the connections of this item yet.")
        G = self.D.reverse(copy=False) if direction.lower().startswith("up") else self.D
        out = []
        for i in ids:
            reach = nx.single_source_shortest_path_length(G, i, cutoff=max(1, min(int(hops or 3), 6)))
            for m, d in sorted(reach.items(), key=lambda kv: kv[1]):
                if m == i:
                    continue
                path = nx.shortest_path(G, i, m)
                e = G.edges[path[0], path[1]]
                out.append(dict(self._brief(m), hops=d, first_step_method=e.get("how"), first_step_confidence=e.get("conf")))
        return dict(found=True, tag=tag, direction=direction, items=out[:30],
                    note="Directions come from flow arrows, FROM/TO connectors and ISA loop order; lower confidence after junctions.")

    def path(self, start, end):
        a, b = self._ids(start), self._ids(end)
        if not a or not b:
            return dict(found=False, missing=[t for t, x in ((start, a), (end, b)) if not x])
        best = None
        for i in a:
            for j in b:
                if nx.has_path(self.G, i, j):
                    p = nx.shortest_path(self.G, i, j)
                    if best is None or len(p) < len(best):
                        best = p
        return dict(found=bool(best), path=[self._brief(x) for x in best or []])

    def hierarchy(self, tag):
        roots = [k for k, a in self.assets.items() if (a["tag"] or "").upper() == tag.upper() or k == tag or (a["name"] or "").upper() == tag.upper()]
        if not roots:
            return dict(found=False, tag=tag)
        kids = defaultdict(list)
        for a in self.assets.values():
            kids[a["parent_id"]].append(a)
        out = []

        def walk(k, d):
            for c in kids.get(k, []):
                out.append(dict(depth=d, id=c["asset_id"], level=c["level"], tag=c["tag"], cls=(c["asset_class"] or "").split("/")[-1],
                                placement=json.loads(c["props"] or "{}").get("placement")))
                if d < 4:
                    walk(c["asset_id"], d + 1)
        walk(roots[0], 1)
        return dict(found=True, root=roots[0], level=self.assets[roots[0]]["level"], children=out[:60], total=len(out))

    def loops(self, sheet=None):
        L = [a for a in self.assets.values() if a["level"] == "Loop" and (sheet is None or a["sheet_no"] == int(sheet))]
        return dict(count=len(L), loops=[dict(loop=a["tag"], name=a["name"], sheet=a["sheet_no"],
                                              members=[self.nodes.get(m, {}).get("label") for m in json.loads(a["props"] or "{}").get("members", [])])
                                         for a in sorted(L, key=lambda a: a["tag"] or "")])

    def sheet_summary(self, sheet):
        sheet = int(sheet)
        s = self.sheets.get(sheet)
        if not s:
            return dict(found=False, sheet=sheet, sheets=sorted(self.sheets))
        ns = [n for n in self.nodes.values() if n["sheet_no"] == sheet]
        by = defaultdict(list)
        for n in ns:
            by[n["kind"]].append(n["label"])
        ed = [e for e in self.edges if self.nodes.get(e["src"], {}).get("sheet_no") == sheet]
        return dict(found=True, sheet=sheet, title=s["title"], kind=s["kind"],
                    equipment=sorted(set(by["Equipment"])), instruments=sorted({x for x in by["Instrument"] if re.match(TAG_RE, x or "")}),
                    untagged_instruments=sum(1 for x in by["Instrument"] if not TAG_RE.match(x or "")),
                    piping_items=Counter(x if TAG_RE.match(x or "") else "untagged" for x in by["Piping"]).most_common(20),
                    connectors=sorted(set(by["Connector"]))[:20], connections=Counter(e["kind"] for e in ed),
                    checks=[c["detail"] for c in self.checks if sheet in c.get("sheets", [])][:10])

    def bom_status(self, status="all", category=None):
        rows = self.recon.get("rows", [])
        m = {"matched": "matched", "bom_only": "in BOM", "drawing_only": "on drawing"}.get((status or "all").lower())
        if m:
            rows = [r for r in rows if r["status"].startswith(m)]
        if category:
            rows = [r for r in rows if (r.get("bom_category") or "").lower() == category.lower()]
        return dict(summary=self.recon.get("summary"), count=len(rows), rows=rows[:60])

    def why_missing(self, tag):
        t = tag.upper()
        b = self._bom_row(t)
        found = self._ids(t)
        if found:
            return dict(tag=t, found_on_drawing=True, objects=[self._brief(i) for i in found])
        pre, num = (t.split("-") + [""])[:2]
        same_num = sorted({l for l in self.by_tag if l.endswith("-" + num)})
        same_pre = sorted({l for l in self.by_tag if l.startswith(pre + "-")})
        sim = difflib.get_close_matches(t, list(self.by_tag), n=6, cutoff=0.6)
        sheet = b.get("bom_sheet") if b else None
        texts = self.S.q("SELECT text, source FROM texts WHERE pid_id=? AND version=(SELECT MAX(version) FROM texts WHERE pid_id=?) AND sheet_no=? AND (text=? OR text=? OR text LIKE ?)",
                         (self.pid, self.pid, sheet, num, pre, f"%{num}%")) if sheet else []
        reasons = []
        if not same_num and not texts:
            reasons.append(f"no text '{num}' was read on sheet {sheet}: either the tag is drawn as strokes the OCR missed, or the BOM number differs from the drawing")
        if same_num:
            reasons.append(f"the drawing has other items numbered {num}: {same_num[:6]} - the prefix may be read or written differently")
        if sim:
            reasons.append(f"closest tags read on the drawing: {sim}")
        if pre in ("BV", "NV", "NRV", "PRV", "FV", "KV", "F", "BPR"):
            reasons.append("valve and filter tags on this drawing are stroke lettering next to the symbol; OCR reads the prefix and number separately and pairs them only within 7 drawing units")
        return dict(tag=t, found_on_drawing=False, bom=b, numbers_read_on_bom_sheet=[x["text"] for x in texts][:10],
                    same_number=same_num[:10], same_prefix_count=len(same_pre), similar=sim, likely_reasons=reasons)

    def drawing_checks(self, rule=None):
        c = [x for x in self.checks if not rule or rule.lower() in x["rule"].lower()]
        return dict(by_rule=Counter(x["rule"] for x in self.checks), items=c[:40])

    def review_queue(self, limit=15):
        rows = self.S.q("SELECT symbol_id, sheet_no, cls, tag, confidence, method FROM symbols WHERE pid_id=? AND version=? AND status='candidate' ORDER BY confidence LIMIT ?",
                        (self.pid, self.v["symbol_detection"], int(limit or 15)))
        return dict(items=rows, waiting=self.S.one("SELECT COUNT(*) n FROM symbols WHERE pid_id=? AND version=? AND status='candidate' AND confidence<0.6",
                                                   (self.pid, self.v["symbol_detection"]))["n"])

    def search(self, query):
        return dict(results=self.S.search(query, limit=25))

    def stats(self):
        kinds = Counter(n["kind"] for n in self.nodes.values())
        tagged = sum(1 for n in self.nodes.values() if TAG_RE.match(n["label"] or ""))
        st = Counter(n["status"] for n in self.nodes.values())
        return dict(sheets=len(self.sheets), nodes=len(self.nodes), by_kind=kinds, tagged=tagged, connections=len(self.edges),
                    connection_kinds=Counter(e["kind"] for e in self.edges), review_status=st, bom=self.recon.get("summary"),
                    checks=Counter(c["rule"] for c in self.checks),
                    model=self.S.one("SELECT model_id, algo, metrics FROM models WHERE task='symbol' AND active=1"))

    def propose_correction(self, target, reason, new_class=None, new_tag=None):
        ids = self._ids(target) or [target]
        self.S.review(self.pid, "symbol", ids[0], "propose", None, dict(cls=new_class, tag=new_tag), reviewer="copilot", comment=reason)
        return dict(recorded=True, object=ids[0], note="Saved as a proposal in the review log; an engineer must approve it.")

    def call(self, name, args):
        f = getattr(self, name, None)
        if not f or name.startswith("_") or name not in {t["name"] for t in TOOLS}:
            return dict(error=f"unknown tool {name}")
        try:
            return f(**(args or {}))
        except Exception as ex:
            return dict(error=str(ex))

    # ------------------------------------------------------------------ offline mode
    def nlu(self):
        if self._nlu is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline, make_union
            X, y = [], []
            for k, v in EXAMPLES.items():
                for s in v:
                    X.append(TAG_RE.sub(" TAGX ", s.upper()).lower()); y.append(k)
            self._nlu = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))),
                                      LogisticRegression(C=8, max_iter=2000)).fit(X, y)
        return self._nlu

    def offline(self, q):
        tags = TAG_RE.findall(q.upper())
        sheet = re.search(r"sheet\s*(\d+)", q, re.I)
        clf = self.nlu()
        P = clf.predict_proba([TAG_RE.sub(" TAGX ", q.upper()).lower()])[0]
        intent = clf.classes_[P.argmax()]
        steps = []

        def run(name, **a):
            r = self.call(name, a); steps.append((name, a, r)); return r
        if intent in ("find_tag", "connections", "hierarchy", "why_missing") and not tags:
            intent = "search" if len(q.split()) <= 4 else "help"
        if intent == "flow" and not tags:
            intent = "help"
        if intent == "flow":
            up = bool(re.search(r"upstream|before|feeds|source|into|comes from", q, re.I))
            r = run("flow", tag=tags[0], direction="upstream" if up else "downstream")
            if not r["found"]:
                return f"{tags[0]} is not on the extracted graph.", steps
            if not r["items"]:
                return f"{tags[0]}: {r['note']}", steps
            return (f"**{'Upstream' if up else 'Downstream'} of {tags[0]}**:\n\n" +
                    "\n".join(f"- {x['tag']} (sheet {x['sheet']}, {x['hops']} step{'s' if x['hops'] > 1 else ''}) · via {x['first_step_method']} · conf {x['first_step_confidence']}" for x in r["items"][:20]) +
                    "\n\n_" + r["note"] + "_"), steps
        if intent == "path" and len(tags) < 2:
            intent = "connections" if tags else "help"
        if intent == "sheet_summary" and not sheet:
            intent = "stats"
        if intent == "find_tag":
            r = run("find_tag", tag=tags[0])
            if not r["found"]:
                b = r.get("bom")
                txt = f"**{tags[0]}** was not found on the drawing." + (f" It is in the BOM ({b['bom_category']}, sheet {b['bom_sheet']})." if b else "")
                if r["similar_tags_on_drawing"]:
                    txt += f" Closest tags read: {', '.join(r['similar_tags_on_drawing'])}."
                if b:
                    w = run("why_missing", tag=tags[0])
                    txt += "\n\n" + "\n".join("- " + x for x in w["likely_reasons"])
                return txt, steps
            lines = []
            for o in r["objects"]:
                lines.append(f"**{o['tag']}** — {o['full_class']} on sheet {o['sheet']} ({o['sheet_title']}), confidence {o['confidence']}, status *{o['status']}*.")
                if o["parent"]["id"]:
                    lines.append(f"Parent: {o['parent']['tag']} ({o['parent']['level']}) — placed by {o['placement']}.")
                if o["connections"]:
                    lines.append("Connected to: " + ", ".join(f"{c['tag']} ({c['via']})" for c in o["connections"][:10]) + ".")
                for c in o["checks"]:
                    lines.append(f"⚠ {c['rule']}: {c['detail']}")
            b = r.get("bom")
            lines.append(f"BOM: {b['status']} ({b['bom_category']})." if b else "BOM: not listed.")
            return "\n\n".join(lines), steps
        if intent == "connections":
            r = run("connections", tag=tags[0], hops=2 if re.search(r"upstream|downstream|2 hop|two hop|feeds|whole", q, re.I) else 1)
            if not r["found"]:
                return f"{tags[0]} is not on the extracted graph. Similar: {', '.join(r['similar']) or 'none'}.", steps
            out = []
            for x in r["results"]:
                c = x["connected"]
                out.append(f"**{x['object']['tag']}** (sheet {x['object']['sheet']}) connects to {len(c)} item(s): " +
                           ", ".join(f"{y['tag']}" + (f" [{y['hops']} hops]" if y["hops"] > 1 else "") for y in c[:20]) + ".")
            return "\n\n".join(out) + "\n\n_" + r["note"] + "_", steps
        if intent == "path":
            r = run("path", start=tags[0], end=tags[1])
            if not r["found"]:
                return f"No connected path between {tags[0]} and {tags[1]} in the candidate graph (the trace may break at a symbol or sheet edge).", steps
            return "Path: " + " → ".join(f"{p['tag']} (s{p['sheet']})" for p in r["path"]), steps
        if intent == "hierarchy":
            r = run("hierarchy", tag=tags[0])
            if not r["found"]:
                return f"{tags[0]} is not in the asset hierarchy.", steps
            return (f"**{tags[0]}** ({r['level']}) has {r['total']} item(s) below it:\n\n" +
                    "\n".join("  " * (c["depth"] - 1) + f"- {c['tag'] or c['id']} · {c['level']} · {c['cls']}" + (f" · {c['placement']}" if c["placement"] else "") for c in r["children"][:40])), steps
        if intent == "loops":
            r = run("loops", **({"sheet": int(sheet.group(1))} if sheet else {}))
            return f"{r['count']} loop(s)" + (f" on sheet {sheet.group(1)}" if sheet else "") + ":\n\n" + \
                "\n".join(f"- **{l['loop']}** ({l['name']}, sheet {l['sheet']}): {', '.join(x for x in l['members'] if x)}" for l in r["loops"][:40]), steps
        if intent == "sheet_summary":
            r = run("sheet_summary", sheet=int(sheet.group(1)))
            if not r["found"]:
                return f"There is no sheet {sheet.group(1)}; sheets are {r['sheets']}.", steps
            t = (f"**Sheet {r['sheet']} — {r['title']}**\n\n- Equipment: {', '.join(r['equipment']) or 'none found'}\n"
                 f"- Instruments ({len(r['instruments'])} tagged, {r['untagged_instruments']} untagged): {', '.join(r['instruments'][:30])}\n"
                 f"- Piping items: {', '.join(f'{k}×{v}' if v > 1 else k for k, v in r['piping_items'])}\n"
                 f"- Connections: {dict(r['connections'])}\n")
            if r["checks"]:
                t += "- Checks: " + "; ".join(r["checks"])
            return t, steps
        if intent == "why_missing":
            r = run("why_missing", tag=tags[0])
            if r["found_on_drawing"]:
                return f"{tags[0]} is on the drawing: " + ", ".join(f"{o['id']} (sheet {o['sheet']})" for o in r["objects"]), steps
            b = r["bom"]
            return (f"**{tags[0]}** " + (f"is in the BOM ({b['bom_category']}, sheet {b['bom_sheet']}) but " if b else "") +
                    "was not found on the drawing.\n\n" + "\n".join("- " + x for x in r["likely_reasons"]) +
                    ("\n\nNumbers read on that sheet that contain it: " + ", ".join(r["numbers_read_on_bom_sheet"]) if r["numbers_read_on_bom_sheet"] else "")), steps
        if intent == "bom_status":
            cat = next((c for c in ("Valve", "Instrument", "Safety", "Equipment", "Filter", "Heater") if c.lower() in q.lower()), None)
            stt = "bom_only" if re.search(r"missing|not found|absent", q, re.I) else "drawing_only" if re.search(r"only on the drawing|not in (the )?bom|extra", q, re.I) else \
                "matched" if re.search(r"match", q, re.I) and not re.search(r"how many", q, re.I) else "all"
            r = run("bom_status", status=stt, category=cat)
            s = r["summary"] or {}
            t = f"BOM: {s.get('matched')}/{s.get('bom_tags')} tags matched, {s.get('bom_only')} only in BOM, {s.get('drawing_only')} only on drawing. By category: {s.get('recall_by_category')}."
            if stt != "all":
                t += f"\n\n{r['count']} {stt.replace('_', ' ')}" + (f" ({cat})" if cat else "") + ": " + ", ".join(x["tag"] for x in r["rows"][:50])
            return t, steps
        if intent == "drawing_checks":
            rule = next((k for k in ("duplicate", "bubble", "relief", "controller", "equipment") if k in q.lower()), None)
            r = run("drawing_checks", rule=rule)
            return (f"Checks: {dict(r['by_rule'])}\n\n" + "\n".join(f"- [{c['severity']}] {c['rule']}: {c['detail']} (sheets {c['sheets']}, {', '.join(c['objects'][:4])})" for c in r["items"][:20]) +
                    "\n\n_These are rule flags to look at, not confirmed defects._"), steps
        if intent == "review_queue":
            r = run("review_queue", limit=15)
            return f"{r['waiting']} candidates below 0.6 confidence. Lowest first:\n\n" + \
                "\n".join(f"- {x['symbol_id']} (sheet {x['sheet_no']}): {x['cls'].split('/')[-1]} {x['tag'] or ''} · {x['confidence']} · {x['method']}" for x in r["items"]), steps
        if intent == "search":
            term = re.sub(r"^(search( for)?|find|look for|anything about)\s+", "", q.strip(), flags=re.I)
            r = run("search", query=term)
            return (f"{len(r['results'])} result(s) for '{term}':\n\n" + "\n".join(f"- {x['kind']} {x['ref']}: {x['text']} (sheet {x['sheet_no']})" for x in r["results"][:20])) if r["results"] else f"Nothing found for '{term}'.", steps
        if intent == "stats":
            r = run("stats")
            m = json.loads(r["model"]["metrics"]) if r.get("model") else {}
            b = r["bom"] or {}
            return (f"{r['sheets']} sheets · {r['nodes']} graph nodes ({dict(r['by_kind'])}) · {r['tagged']} tagged · {r['connections']} candidate connections {dict(r['connection_kinds'])}.\n\n"
                    f"BOM match {b.get('matched')}/{b.get('bom_tags')} ({b.get('recall_by_category')}). Review status: {dict(r['review_status'])}. "
                    f"Symbol model hold-out accuracy {m.get('holdout_accuracy')} (on augmented legend samples; field accuracy comes from reviews). Checks: {dict(r['checks'])}."), steps
        return HELP, steps

    # ------------------------------------------------------------------ Claude mode
    def ask(self, question, history=None, api_key=None, on_step=None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self.offline(question)
        try:
            import anthropic
        except ImportError:
            t, s = self.offline(question)
            return t + "\n\n_(anthropic package not installed - answered offline)_", s
        client = anthropic.Anthropic(api_key=api_key)
        msgs = list(history or []) + [dict(role="user", content=question)]
        steps = []
        for _ in range(MAX_STEPS):
            rep = client.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM + f"\nDrawing id: {self.pid}.", tools=TOOLS, messages=msgs)
            msgs.append(dict(role="assistant", content=rep.content))
            uses = [b for b in rep.content if getattr(b, "type", "") == "tool_use"]
            if not uses:
                return "".join(b.text for b in rep.content if getattr(b, "type", "") == "text"), steps
            results = []
            for u in uses:
                res = self.call(u.name, u.input)
                steps.append((u.name, u.input, res))
                if on_step:
                    on_step(u.name, u.input, res)
                results.append(dict(type="tool_result", tool_use_id=u.id, content=json.dumps(res, default=str)[:12000]))
            msgs.append(dict(role="user", content=results))
        return "Stopped after the maximum number of tool calls; ask a narrower question.", steps


HELP = """I answer from the digitised P&ID (works offline; with a Claude key I can handle free-form questions). Try:
- **what is TE-202** · **what is connected to TIC-202** · **path from V-200 to R-300** · **downstream of NRV-202** · **upstream of R-400**
- **what is under R-300** · **loops on sheet 3** · **what is on sheet 5**
- **which BOM valves are missing** · **why is TE-306 missing**
- **duplicate tags** · **what needs review** · **search needle valve** · **how good is the digitisation**"""
