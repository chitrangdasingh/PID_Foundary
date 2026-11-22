"""PID Foundry - Streamlit front end. Run: streamlit run app.py"""
import json
import os
import re
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from foundry import copilot as CP, exports as EX, hierarchy as HI, pipeline as PL, store as STO, training as TR  # noqa: E402

DATA = os.path.join(ROOT, "data")
DEFAULT_DXF = os.path.join(DATA, "sources", "Thermax_Single_Tube_Reactor_System_PID.dxf")
DEFAULT_BOM = os.path.join(DATA, "seed", "BOM_single_tube_reactor.xlsx")
st.set_page_config(page_title="PID Foundry", layout="wide")


@st.cache_resource
def get_foundry():
    store = STO.Store(os.path.join(DATA, "foundry.db"))
    return PL.Foundry(store, DATA)


F = get_foundry()
S = F.store


def ensure_cache(pid_id):
    """Parsed geometry is not committed (it is rebuilt from the source drawing in about a second)."""
    if os.path.exists(F.cache_path(pid_id)):
        return True
    src = S.one("SELECT file_name FROM sources WHERE pid_id=?", (pid_id,))
    path = os.path.join(DATA, "sources", src["file_name"]) if src else None
    if path and os.path.exists(path):
        F.ingest(path, pid_id)
        return True
    return False


# ---------------- sidebar
st.sidebar.title("PID Foundry")
st.sidebar.caption("Static P&ID → governed engineering knowledge graph")
sources = S.q("SELECT pid_id, file_name, title FROM sources ORDER BY ingested_at DESC")
if not sources:
    st.sidebar.warning("No drawing ingested yet. Use the Ingest tab.")
pid = st.sidebar.selectbox("Drawing", [s["pid_id"] for s in sources], format_func=lambda p: next(f"{p} · {s['title']}" for s in sources if s["pid_id"] == p)) if sources else None
sheets = S.q("SELECT * FROM sheets WHERE pid_id=? ORDER BY sheet_no", (pid,)) if pid else []
proc_sheets = [s for s in sheets if s["kind"] == "process"]
vs = {k: S.latest_version(pid, k) for k in ("symbol_detection", "text_detection", "graph_construction", "graph_persistence")} if pid else {}
if pid:
    st.sidebar.markdown("**Pipeline versions**")
    for k, v in vs.items():
        st.sidebar.write(f"{'✅' if v else '⬜'} {k.replace('_', ' ')} · v{v}")
    try:
        TR.load_active(S)
    except Exception:
        st.sidebar.error("The stored model does not load with this scikit-learn version. Retrain it in Models & training (about 10 s).")
    m = S.one("SELECT model_id, name, algo, metrics FROM models WHERE task='symbol' AND active=1")
    if m:
        st.sidebar.write(f"Active model: #{m['model_id']} {m['algo']} (hold-out acc {json.loads(m['metrics'])['holdout_accuracy']})")

tabs = st.tabs(["Overview", "Copilot", "Ingest & pipeline", "Symbols", "Text", "Graph", "Review queue", "Asset hierarchy",
                "BOM reconciliation", "Search & ask", "Models & training", "Exports", "Architecture & API"])


def sheet_label(n):
    s = next((x for x in sheets if x["sheet_no"] == n), {})
    return f"Sheet {n}: {s.get('title') or '-'}"


def sym_df():
    return S.df("SELECT symbol_id, sheet_no, category, cls, tag, confidence, method, status, x0, y0, x1, y1 FROM symbols WHERE pid_id=? AND version=?", (pid, vs.get("symbol_detection", 0)))


# ---------------- Overview
with tabs[0]:
    st.header("Single Tube Reactor P&ID — digitisation overview")
    if not pid:
        st.info("Ingest a drawing first.")
    else:
        src = S.one("SELECT * FROM sources WHERE pid_id=?", (pid,))
        sd = sym_df()
        ne = S.one("SELECT COUNT(*) n FROM graph_edges WHERE pid_id=? AND version=?", (pid, vs["graph_persistence"]))["n"]
        rec = S.kv_get(f"recon:{pid}", {}).get("summary", {})
        c = st.columns(6)
        c[0].metric("Sheets", len(sheets))
        c[1].metric("Symbols", len(sd))
        c[2].metric("Tagged objects", int(sd["tag"].notna().sum()) if len(sd) else 0)
        c[3].metric("Candidate connections", ne)
        c[4].metric("BOM tags matched", f"{rec.get('matched', 0)}/{rec.get('bom_tags', 0)}")
        c[5].metric("Reviewed", S.one("SELECT COUNT(*) n FROM reviews WHERE pid_id=?", (pid,))["n"])
        st.caption(f"Source: {src['file_name']} · {src['fmt']} · sha256 {src['sha256'][:16]}… · drawing {src['drawing_no']} rev {src['revision']}")
        st.subheader("Sheets")
        st.dataframe(pd.DataFrame(sheets)[["sheet_no", "title", "kind", "drawing_no", "revision"]], hide_index=True)
        if len(sd):
            st.subheader("Detections by sheet and category")
            st.dataframe(sd.pivot_table(index="sheet_no", columns="category", values="symbol_id", aggfunc="count", fill_value=0))
        st.info("Everything extracted is a **candidate** until an engineer approves it in the Review queue. "
                "Approved items become training data (active learning) and are the only items exported as 'approved'.")

# ---------------- Copilot
with tabs[1]:
    st.header("P&ID copilot")
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        key = None
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    c1, c2 = st.columns([3, 1])
    use_ai = c2.toggle("Use Claude", value=bool(key), disabled=not key, help="Add ANTHROPIC_API_KEY in Streamlit Secrets to enable")
    c1.caption(("Claude mode: free-form questions, answered only from the digitised drawing. " if use_ai else
                "Offline mode: a trained intent model answers common questions; no key or network needed. ") +
               "Every answer shows the data it used. Extracted items are candidates until reviewed.")
    if not pid or not vs.get("graph_persistence"):
        st.info("Run the pipeline first.")
    else:
        @st.cache_resource
        def get_copilot(p, v):
            return CP.Copilot(S, p)
        cop = get_copilot(pid, vs["graph_persistence"])
        hist = st.session_state.setdefault("chat", [])
        ex = st.columns(4)
        starters = ["What is on sheet 3?", "Why is TE-306 missing?", "Duplicate tags", "What is connected to TIC-202?"]
        clicked = None
        for i, q in enumerate(starters):
            if ex[i].button(q, key=f"starter{i}"):
                clicked = q
        for m in hist:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                for name, args, res in m.get("steps", []):
                    with st.expander(f"data: {name}({json.dumps(args)})"):
                        st.json(res)
        q = st.chat_input("Ask about tags, connections, loops, sheets, BOM gaps, checks…") or clicked
        if q:
            hist.append(dict(role="user", content=q))
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Looking it up…"):
                    prior = [dict(role=m["role"], content=m["content"]) for m in hist[:-1]][-8:]
                    try:
                        ans, steps = cop.ask(q, history=prior, api_key=key) if use_ai else cop.offline(q)
                    except Exception as ex_:
                        ans, steps = cop.offline(q)
                        ans += f"\n\n_(Claude call failed: {ex_}; answered offline)_"
                st.markdown(ans)
                for name, args, res in steps:
                    with st.expander(f"data: {name}({json.dumps(args)})"):
                        st.json(res)
            hist.append(dict(role="assistant", content=ans, steps=steps))
        if hist and st.button("Clear conversation"):
            st.session_state["chat"] = []
            st.rerun()

# ---------------- Ingest & pipeline
with tabs[2]:
    st.header("Ingest and run the pipeline step by step")
    st.markdown("Steps follow the inference API of the Azure P&ID design: **symbol detection → text detection → "
                "graph construction (line detection + connections, a background job) → graph persistence**. "
                "Between steps you can correct the output; the next step reads the corrected version.")
    up = st.file_uploader("Upload a P&ID (DXF; PDF / PNG go through the raster path)", type=["dxf", "pdf", "png", "jpg"])
    c1, c2 = st.columns(2)
    if up is not None and c1.button("Register drawing"):
        dest = os.path.join(DATA, "sources", up.name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(up.getbuffer())
        if up.name.lower().endswith(".dxf"):
            new_pid, sh = F.ingest(dest)
            st.success(f"Registered {new_pid} with {len(sh)} sheets. Select it in the sidebar.")
        else:
            st.warning("Raster input: run scripts/run_raster.py on it (OCR + Hough + the active model). The DXF path is the recommended one.")
    if c2.button("Re-register the bundled reactor DXF") and os.path.exists(DEFAULT_DXF):
        F.ingest(DEFAULT_DXF, "STR-PID-R5")
        st.success("Re-registered STR-PID-R5.")
    if pid:
        ensure_cache(pid)
        st.subheader(f"Run steps for {pid}")
        sel = st.multiselect("Sheets (empty = all process sheets)", [s["sheet_no"] for s in proc_sheets], format_func=sheet_label)
        b = st.columns(4)
        if b[0].button("1 · Symbol detection"):
            with st.spinner("Detecting symbols and reading bubbles (OCR)…"):
                r = F.symbol_detection(pid, sheet_nos=sel or None)
            st.success(f"{len(r['symbols'])} symbols, {len(r['glyph_words'])} OCR words · v{r['version']}")
        if b[1].button("2 · Text detection"):
            r = F.text_detection(pid)
            st.success(f"{len(r['texts'])} text spans · v{r['version']}")
        if b[2].button("3 · Graph construction (job)"):
            st.info(f"HTTP {F.post_graph_construction(pid)} accepted; check job status below.")
        if b[3].button("4 · Graph persistence"):
            r = F.graph_persistence(pid, bom_path=DEFAULT_BOM if os.path.exists(DEFAULT_BOM) else None)
            st.success(json.dumps(r))
        st.write("Graph construction job:", F.job_status(pid))
        st.subheader("Lineage: every run")
        st.dataframe(S.df("SELECT run_id, step, version, status, model_id, datetime(started,'unixepoch') started, round(finished-started,1) seconds, outputs FROM runs WHERE pid_id=? ORDER BY run_id DESC", (pid,)), hide_index=True)
        st.subheader("Stored outputs (inference-results/<pid_id>/…)")
        base = os.path.join(DATA, "inference-results", pid)
        if os.path.exists(base):
            tree = [f"{os.path.relpath(os.path.join(dp, f), DATA)}" for dp, _, fs in os.walk(base) for f in sorted(fs)]
            st.code("\n".join(tree[:80]))


def show_step_image(step, key):
    if not pid:
        return None
    imgs = [p for p in F.images(pid, step) if os.path.basename(p).startswith("output_")]
    if not imgs:
        st.info("Run this step first.")
        return None
    names = [os.path.basename(p) for p in imgs]
    pick = st.selectbox("Sheet image", names, key=key)
    st.image(imgs[names.index(pick)])
    m = re.search(r"_S(\d+)_", pick)
    return int(m.group(1)) if m else None

# ---------------- Symbols
with tabs[3]:
    st.header("Symbol detection")
    st.caption("Bubbles: circle geometry + OCR of the vector letters. Arrows: filled triangles. Equipment: tag text + the outline it labels. "
               "Everything else: the active symbol model (HOG + classifier trained from the legend sheet and approved reviews).")
    sn = show_step_image("symbol_detection", "symimg")
    if pid:
        sd = sym_df()
        if sn:
            sd = sd[sd.sheet_no == sn]
        cats = st.multiselect("Category", sorted(sd.category.unique()), default=list(sorted(sd.category.unique())))
        sd = sd[sd.category.isin(cats)]
        ed = st.data_editor(sd, hide_index=True, disabled=["symbol_id", "sheet_no", "confidence", "method", "x0", "y0", "x1", "y1"], key="symed")
        if st.button("Save corrections as reviews"):
            n = 0
            for (_, a), (_, b) in zip(sd.iterrows(), ed.iterrows()):
                if a["cls"] != b["cls"] or (a["tag"] or "") != (b["tag"] or "") or a["status"] != b["status"]:
                    dec = "reject" if b["status"] == "rejected" else "approve" if (b["status"] == "approved" and a["cls"] == b["cls"]) else "correct"
                    S.review(pid, "symbol", a["symbol_id"], dec, dict(cls=a["cls"], tag=a["tag"]), dict(cls=b["cls"], tag=b["tag"], category=str(b["cls"]).split("/")[0]))
                    n += 1
            st.success(f"{n} reviews saved. Rebuild the graph (steps 2-4) to use them; retrain the model to learn from them.")

# ---------------- Text
with tabs[4]:
    st.header("Text detection")
    st.caption("DXF TEXT/MTEXT is read exactly; lettering that was exploded into strokes (PDF import) is grouped into words and read by OCR.")
    sn = show_step_image("text_detection", "txtimg")
    if pid:
        td = S.df("SELECT text_id, sheet_no, text, label, source, confidence FROM texts WHERE pid_id=? AND version=?", (pid, vs.get("text_detection", 0)))
        if sn:
            td = td[td.sheet_no == sn]
        st.dataframe(td.groupby(["label", "source"]).size().rename("count").reset_index(), hide_index=True)
        lab = st.multiselect("Labels", sorted(td.label.unique()), default=[x for x in sorted(td.label.unique()) if x != "title_block"])
        st.dataframe(td[td.label.isin(lab)], hide_index=True)

# ---------------- Graph
with tabs[5]:
    st.header("Line detection and graph construction")
    sn = show_step_image("graph_construction", "grimg")
    if pid and vs["graph_persistence"]:
        ge = S.df("""SELECT e.edge_id, e.kind, e.direction, e.confidence, e.status, a.sheet_no, a.label src_label, b.label dst_label, e.src, e.dst, e.evidence
                     FROM graph_edges e JOIN graph_nodes a ON a.pid_id=e.pid_id AND a.version=e.version AND a.node_id=e.src
                     JOIN graph_nodes b ON b.pid_id=e.pid_id AND b.version=e.version AND b.node_id=e.dst WHERE e.pid_id=? AND e.version=?""", (pid, vs["graph_persistence"]))
        lab_of = dict(zip(ge.src, ge.src_label)) | dict(zip(ge.dst, ge.dst_label))

        def _ev(x):
            try:
                v = json.loads(x or "{}")
                return v if isinstance(v, dict) else {}
            except Exception:
                return {}
        ge["flow"] = [(" → ".join(lab_of.get(t, t) for t in d.split("->")) if "->" in (d or "") else "unknown") for d in ge.direction]
        ge["direction_method"] = [_ev(x).get("direction_method") for x in ge.evidence]
        ge["direction_conf"] = [_ev(x).get("direction_confidence") for x in ge.evidence]
        ge = ge.drop(columns=["evidence"])
        if sn:
            ge = ge[ge.sheet_no == sn]
        c = st.columns(3)
        c[0].metric("Connections", len(ge)); c[1].metric("With flow direction", int((ge.flow != "unknown").sum()))
        c[2].metric("Direction from arrows / connectors / ISA", f"{ge.direction_method.fillna('').str.startswith('arrow').sum()} / {ge.direction_method.fillna('').str.startswith('connector').sum()} / {ge.direction_method.fillna('').str.startswith('ISA').sum()}")
        st.caption("Green arrows on the sheet image = flow direction. Direction comes from flow arrows drawn on the line, FROM/TO connectors "
                   "(FROM = source, TO / VENT = destination), carried along the pipe through in-line valves and up to 3 junctions; "
                   "signal direction inside a loop follows ISA-5.1 (element → transmitter → controller → relay → final element). "
                   "Runs where two seeds disagree get confidence 0.3.")
        fe = st.selectbox("Confirm or flip the direction of a connection", ["-"] + [f"{r.edge_id}: {r.flow if r.flow != 'unknown' else r.src_label + ' — ' + r.dst_label}" for r in ge.itertuples()][:300])
        if fe != "-":
            eid = fe.split(":")[0]
            row = ge[ge.edge_id == eid].iloc[0]
            cc = st.columns(3)
            if cc[0].button("Confirm direction") and row.direction != "unknown":
                S.review(pid, "edge", eid, "approve", dict(direction=row.direction)); st.rerun()
            if cc[1].button(f"Set {row.src_label} → {row.dst_label}"):
                S.review(pid, "edge", eid, "correct", dict(direction=row.direction), dict(direction=f"{row.src}->{row.dst}")); st.rerun()
            if cc[2].button(f"Set {row.dst_label} → {row.src_label}"):
                S.review(pid, "edge", eid, "correct", dict(direction=row.direction), dict(direction=f"{row.dst}->{row.src}")); st.rerun()
        st.dataframe(ge, hide_index=True)
        G = EX.load_graph(S, pid)
        if sn:
            keep = [n for n, d in G.nodes(data=True) if d["sheet"] == sn]
            H = G.subgraph(keep)
            dot = ["digraph G { rankdir=LR; node [shape=box, fontsize=9, style=rounded];"]
            col = {"Equipment": "#d9534f", "Instrument": "#337ab7", "Piping": "#5cb85c", "Connector": "#f0ad4e", "Junction": "#999999"}
            for n, d in H.nodes(data=True):
                dot.append(f'"{n}" [label="{d["label"][:18]}", color="{col.get(d["kind"], "#777")}"];')
            for a, b, d in H.edges(data=True):
                arrow = "normal" if "->" in d["direction"] else "none"
                dot.append(f'"{a}" -> "{b}" [arrowhead={arrow}, style={"dashed" if d["kind"] == "signal" else "solid"}];')
            dot.append("}")
            st.graphviz_chart("\n".join(dot))

# ---------------- Review queue
with tabs[6]:
    st.header("Review queue (human in the loop → active learning)")
    if pid:
        thr = st.slider("Show candidates below confidence", 0.0, 1.0, 0.6, 0.05)
        q = S.q("""SELECT s.symbol_id, s.sheet_no, s.cls, s.tag, s.confidence, s.method, t.png FROM symbols s
                   LEFT JOIN training_samples t ON t.pid_id=s.pid_id AND t.source_ref=s.symbol_id AND t.funnel='inference'
                   WHERE s.pid_id=? AND s.version=? AND s.status='candidate' AND s.confidence < ? ORDER BY s.confidence LIMIT 30""",
                (pid, vs["symbol_detection"], thr))
        classes = json.loads((S.one("SELECT classes FROM models WHERE task='symbol' AND active=1") or {"classes": "[]"})["classes"])
        st.write(f"{len(q)} shown (lowest confidence first).")
        for r in q:
            c = st.columns([1, 3, 3, 2])
            if r["png"]:
                c[0].image(TR.from_png(r["png"]), width=64)
            c[1].write(f"**{r['symbol_id']}** · sheet {r['sheet_no']}  \n{r['cls']} · {r['tag'] or ''}  \nconf {r['confidence']} · {r['method']}")
            new = c[2].selectbox("Correct class", [r["cls"]] + [k for k in classes if k != r["cls"]], key="c" + r["symbol_id"])
            tag = c[2].text_input("Tag", r["tag"] or "", key="t" + r["symbol_id"])
            if c[3].button("Approve / save", key="a" + r["symbol_id"]):
                S.review(pid, "symbol", r["symbol_id"], "approve" if new == r["cls"] and tag == (r["tag"] or "") else "correct",
                         dict(cls=r["cls"], tag=r["tag"]), dict(cls=new, tag=tag or None, category=new.split("/")[0]))
                st.rerun()
            if c[3].button("Reject", key="r" + r["symbol_id"]):
                S.review(pid, "symbol", r["symbol_id"], "reject", dict(cls=r["cls"]))
                st.rerun()
        st.subheader("Drawing checks (rules)")
        ck = S.kv_get(f"checks:{pid}", [])
        if ck:
            ckd = pd.DataFrame(ck)
            st.dataframe(ckd.groupby(["rule", "severity"]).size().rename("count").reset_index(), hide_index=True)
            st.dataframe(ckd, hide_index=True)
        st.subheader("Edges to confirm")
        qe = S.q("SELECT edge_id, src, dst, kind, confidence FROM graph_edges WHERE pid_id=? AND version=? AND status='candidate' ORDER BY confidence LIMIT 15", (pid, vs["graph_persistence"]))
        lab = {n["node_id"]: n["label"] for n in S.q("SELECT node_id, label FROM graph_nodes WHERE pid_id=? AND version=?", (pid, vs["graph_persistence"]))}
        for e in qe:
            c = st.columns([5, 1, 1])
            c[0].write(f"{e['edge_id']} · {lab.get(e['src'], e['src'])} — {lab.get(e['dst'], e['dst'])} · {e['kind']} · conf {e['confidence']}")
            if c[1].button("✓", key="ea" + e["edge_id"]):
                S.review(pid, "edge", e["edge_id"], "approve"); st.rerun()
            if c[2].button("✗", key="er" + e["edge_id"]):
                S.review(pid, "edge", e["edge_id"], "reject"); st.rerun()
        st.subheader("Review log")
        st.dataframe(S.df("SELECT review_id, object_type, object_id, decision, new_value, reviewer, ts, used_for_training FROM reviews WHERE pid_id=? ORDER BY review_id DESC", (pid,)), hide_index=True)

# ---------------- Asset hierarchy
with tabs[7]:
    st.header("ISA-95 asset hierarchy")
    st.caption("Enterprise › Site › Area › Process cell (sheet) › Unit (equipment) › Loop › Control module (instrument / valve). "
               "Placement method is shown so the parent can be checked.")
    if pid:
        A = S.q("SELECT * FROM assets WHERE pid_id=?", (pid,))
        kids = {}
        for a in A:
            kids.setdefault(a["parent_id"], []).append(a)
        lines = []

        def walk(pid_, depth):
            for a in sorted(kids.get(pid_, []), key=lambda a: (a["level"], a["tag"] or a["name"] or "")):
                pl = json.loads(a["props"] or "{}").get("placement", "")
                lines.append(dict(tree="　" * depth + ("└ " if depth else "") + (a["tag"] or a["name"] or a["asset_id"]), level=a["level"],
                                  asset_class=(a["asset_class"] or "").split("/")[-1], sheet=a["sheet_no"], placement=pl, bom=a["match_status"], asset_id=a["asset_id"]))
                walk(a["asset_id"], depth + 1)
        walk(None, 0)
        st.dataframe(pd.DataFrame(lines), hide_index=True, height=600)
        st.download_button("Asset hierarchy + relationships (Excel)", EX.assets_xlsx(S, pid), f"{pid}_asset_hierarchy.xlsx")

# ---------------- BOM
with tabs[8]:
    st.header("BOM / index reconciliation")
    if pid:
        rec = S.kv_get(f"recon:{pid}", {})
        if rec:
            s_ = rec["summary"]
            c = st.columns(4)
            c[0].metric("BOM tags", s_["bom_tags"]); c[1].metric("Matched", s_["matched"]); c[2].metric("Only in BOM", s_["bom_only"]); c[3].metric("Only on drawing", s_["drawing_only"])
            st.write("Found / BOM tags by category:", s_["recall_by_category"])
            df = pd.DataFrame(rec["rows"])
            pick = st.multiselect("Status", sorted(df.status.unique()), default=sorted(df.status.unique()))
            st.dataframe(df[df.status.isin(pick)], hide_index=True)
            st.caption("'Only in BOM' items are what to look for on the sheet (often valve tags drawn as strokes the OCR missed); "
                       "'only on drawing' are OCR misreads or items missing from the BOM — both go to review.")

# ---------------- Search & ask
with tabs[9]:
    st.header("Search and ask")
    qtext = st.text_input("Search tags, classes, notes (e.g. 'TE-202', 'R-300', 'needle valve', 'VENT')")
    if qtext and pid:
        st.dataframe(pd.DataFrame(S.search(qtext)), hide_index=True)
    st.subheader("Ask the graph")
    ask = st.text_input("Try: 'connected to TIC-202', 'under R-300', 'loops on sheet 3', 'what is on sheet 5'")
    if ask and pid:
        G = EX.load_graph(S, pid).to_undirected()
        lab = {n: d["label"] for n, d in G.nodes(data=True)}
        m = re.search(r"([A-Z]{1,4}-\d{3}[A-Z]?)", ask.upper())
        if "CONNECT" in ask.upper() and m:
            ids = [n for n, l in lab.items() if l == m.group(1)]
            st.write({lab[i]: sorted({lab[n] for n in G.neighbors(i)}) for i in ids} or "tag not found")
        elif "UNDER" in ask.upper() and m:
            A = S.q("SELECT * FROM assets WHERE pid_id=?", (pid,))
            root = [a["asset_id"] for a in A if a["tag"] == m.group(1)]
            out, frontier = [], root
            while frontier:
                ch = [a for a in A if a["parent_id"] in frontier]
                out += ch; frontier = [a["asset_id"] for a in ch]
            st.dataframe(pd.DataFrame(out)[["asset_id", "level", "tag", "name", "sheet_no"]] if out else pd.DataFrame(), hide_index=True)
        elif "LOOP" in ask.upper():
            n = re.search(r"SHEET\s*(\d+)", ask.upper())
            st.dataframe(S.df("SELECT tag, name, sheet_no, props FROM assets WHERE pid_id=? AND level='Loop'" + (" AND sheet_no=?" if n else ""), (pid, int(n.group(1))) if n else (pid,)), hide_index=True)
        elif "SHEET" in ask.upper():
            n = int(re.search(r"(\d+)", ask).group(1))
            st.dataframe(S.df("SELECT category, cls, tag, confidence FROM symbols WHERE pid_id=? AND version=? AND sheet_no=? ORDER BY category", (pid, vs["symbol_detection"], n)), hide_index=True)
        else:
            st.write("Patterns: 'connected to <TAG>', 'under <TAG>', 'loops on sheet N', 'what is on sheet N'.")

# ---------------- Models
with tabs[10]:
    st.header("Models and training data")
    st.markdown("**Funnels**: `seed` = legend sheet symbols · `upload` = data-scientist labelled crops · `approved` = reviewed detections (active learning) · `inference` = crops kept for review (not trained on until approved).")
    st.dataframe(S.df("SELECT funnel, label, COUNT(*) n FROM training_samples GROUP BY funnel, label ORDER BY funnel, n DESC"), hide_index=True)
    seeds = S.q("SELECT label, source_ref, png FROM training_samples WHERE funnel='seed' LIMIT 60")
    if seeds:
        st.subheader("Seed gallery (from the legend sheet)")
        cols = st.columns(10)
        for i, r in enumerate(seeds):
            cols[i % 10].image(TR.from_png(r["png"]), caption=r["label"].split("/")[-1][:18], width=64)
    c = st.columns(3)
    algo = c[0].selectbox("Algorithm", ["rf", "svm"])
    if c[1].button("Harvest approved reviews into training data"):
        st.success(f"{TR.harvest_approved(S)} new samples")
    if c[2].button("Train and register a new model"):
        with st.spinner("Training…"):
            mid, met = TR.train_symbol_model(S, algo=algo)
        st.success(f"Model #{mid}: hold-out accuracy {met['holdout_accuracy']}, macro F1 {met['macro_f1']}")
    up = st.file_uploader("Upload labelled crops (PNG). File name pattern: <Category>__<Sub>__<Name>__anything.png", type=["png"], accept_multiple_files=True)
    if up and st.button("Add uploads"):
        for f in up:
            parts = f.name.split("__")
            if len(parts) >= 3:
                TR.add_upload(S, "/".join(parts[:3]), f.getvalue(), f.name)
        st.success("Added.")
    st.subheader("Model registry")
    reg = S.df("SELECT model_id, name, algo, n_train, data_hash, active, created, metrics FROM models WHERE task='symbol' ORDER BY model_id DESC")
    st.dataframe(reg, hide_index=True)
    if len(reg):
        act = st.selectbox("Activate model", reg.model_id.tolist())
        if st.button("Set active"):
            S.x("UPDATE models SET active = (model_id = ?) WHERE task='symbol'", (int(act),)); st.rerun()
        met = json.loads(reg.iloc[0]["metrics"])
        st.write("Per-class F1 (latest model, hold-out):")
        st.dataframe(pd.Series(met.get("per_class", {}), name="f1").sort_values())

# ---------------- Exports
with tabs[11]:
    st.header("Exports")
    if pid and vs["graph_persistence"]:
        G = EX.load_graph(S, pid)
        src = S.one("SELECT * FROM sources WHERE pid_id=?", (pid,))
        A = S.q("SELECT * FROM assets WHERE pid_id=?", (pid,))
        c = st.columns(4)
        c[0].download_button("GraphML", EX.graphml(G), f"{pid}.graphml")
        c[1].download_button("Neo4j Cypher", EX.cypher(G, A), f"{pid}.cypher")
        c[2].download_button("DEXPI-shaped JSON", json.dumps(EX.dexpi_json(G, src), indent=1), f"{pid}_dexpi.json")
        c[3].download_button("Asset register (Excel)", EX.assets_xlsx(S, pid), f"{pid}_assets.xlsx")
        st.caption(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (rejected items excluded).")

# ---------------- Architecture
with tabs[12]:
    st.header("Architecture, API and references")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style=rounded, fontsize=10];
      src [label="DXF / DWG→DXF\\nPDF / scan"]; ing [label="Ingest\\nsheets, title blocks\\n(bronze)"];
      sym [label="1 Symbol detection\\ngeometry rules + model\\nbubble OCR"]; txt [label="2 Text detection\\nDXF text + glyph OCR\\ntag association"];
      gc [label="3 Graph construction\\nline detection, junctions,\\nattach, direction (job)"]; gp [label="4 Graph persistence\\nSQL graph, ISA-95,\\nBOM reconciliation"];
      rv [label="Review queue\\n(corrections)"]; tr [label="Training funnels\\nseed · upload · approved"]; reg [label="Model registry"];
      ex [label="Exports\\nGraphML · Cypher · DEXPI JSON · Excel"];
      src->ing->sym->txt->gc->gp->ex; sym->rv; gc->rv; rv->tr->reg->sym; rv->gp; }""")
    st.subheader("API contract (Python functions; wrap with FastAPI for a service)")
    st.table(pd.DataFrame([
        ["PostSymbolDetection", "Foundry.symbol_detection(pid_id, sheet_nos, equipment_boxes)", "symbol-detection/response.json + output_*.png"],
        ["PostTextDetection", "Foundry.text_detection(pid_id, corrected_symbols)", "text-detection/response.json"],
        ["PostGraphConstruction (202)", "Foundry.post_graph_construction(pid_id, corrected_text)", "graph-construction/job_status.json"],
        ["GetGraphConstructionJobStatus", "Foundry.job_status(pid_id)", "Submitted / Running / Succeeded / Failed"],
        ["PostGraphPersistence (201)", "Foundry.graph_persistence(pid_id, corrected_graph)", "graph_nodes, graph_edges, assets"],
        ["GetInferenceResults", "Foundry.results(pid_id, step)", "corrected.json first, else response.json"],
        ["GetInferenceResultImages", "Foundry.images(pid_id, step)", "output_*.png (debug_* when DEBUG=1)"]], columns=["Endpoint", "Function", "Stored as"]))
    st.markdown(open(os.path.join(ROOT, "docs", "REFERENCES.md")).read() if os.path.exists(os.path.join(ROOT, "docs", "REFERENCES.md")) else "")
