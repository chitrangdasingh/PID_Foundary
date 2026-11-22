"""Smoke tests: DXF reader, legend seed, store, app import with a stub UI. Run: python -m pytest -q tests"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DXF = os.path.join(ROOT, "data", "sources", "Thermax_Single_Tube_Reactor_System_PID.dxf")
from foundry.sources import unpack_sources  # noqa: E402
unpack_sources(os.path.dirname(DXF))


def test_reader_and_sheets():
    from foundry import dxf_reader as R, sheets as S
    d = R.read(DXF)
    assert d["stats"]["total"] > 10000
    ents = [e for e in R.explode_inserts(d) if "bbox" in e]
    sh = S.find_sheets(ents)
    assert len([s for s in sh if s["kind"] == "process"]) == 9
    assert any(s["kind"] == "legend" for s in sh)


def test_legend_seed():
    from foundry import dxf_reader as R, sheets as S, legend as L
    d = R.read(DXF)
    ents = [e for e in R.explode_inserts(d) if "bbox" in e]
    leg = [s for s in S.find_sheets(ents) if s["kind"] == "legend"][0]
    rows = L.read_legend(ents, leg)
    assert sum(1 for r in rows if r["n_geom"]) >= 45


def test_text_rules():
    from foundry.text import rule_label
    assert rule_label("R-300") == "equipment_tag"
    assert rule_label("BV-201") == "valve_tag"
    assert rule_label("I-5") == "offpage_connector"


def test_store_review_and_search(tmp_path):
    from foundry.store import Store
    s = Store(str(tmp_path / "t.db"))
    s.x("INSERT INTO symbols(pid_id, symbol_id, version, sheet_no, cls, category, confidence, x0, y0, x1, y1) VALUES('P','S1',1,1,'Piping/Valve/BallValve','Piping',0.4,0,0,1,1)")
    s.review("P", "symbol", "S1", "correct", {}, {"cls": "Piping/Valve/NeedleValve"})
    assert s.one("SELECT cls, status FROM symbols")["status"] == "corrected"
    s.index([("P", "asset", "A1", "TE-202 temperature element", 3)])
    assert s.search("TE-202")


def test_app_imports_with_stub():
    import importlib
    import tests.st_stub as stub
    sys.modules["streamlit"] = stub._st
    sys.path.insert(0, ROOT)
    if "app" in sys.modules:
        del sys.modules["app"]
    importlib.import_module("app")
    # the Graph tab draws the clickable knowledge graph with BOM details for every node
    page = stub._v1.rendered[-1] if stub._v1.rendered else ""
    assert "BOM entry" in page and '"TIC-202"' in page


def test_copilot_offline():
    from foundry import copilot as C
    from foundry.store import Store
    st = Store(os.path.join(ROOT, "data", "foundry.db"))
    cp = C.Copilot(st, "STR-PID-R5")
    for q, tool in [("what is TE-202", "find_tag"), ("loops on sheet 3", "loops"), ("why is TE-306 missing", "why_missing"),
                    ("duplicate tags", "drawing_checks"), ("what is on sheet 5", "sheet_summary"), ("downstream of NRV-202", "flow")]:
        ans, steps = cp.offline(q)
        assert steps and steps[0][0] == tool, (q, steps)


def test_flow_direction_present():
    from foundry.store import Store
    st = Store(os.path.join(ROOT, "data", "foundry.db"))
    v = st.latest_version("STR-PID-R5", "graph_persistence")
    n = st.one("SELECT COUNT(*) n FROM graph_edges WHERE pid_id='STR-PID-R5' AND version=? AND direction LIKE '%->%'", (v,))["n"]
    assert n > 100
