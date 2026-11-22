"""
store.py - the Foundry data store: one SQLite file (data/foundry.db) holding

  sources        every ingested drawing (sha256, size, format, title) = the bronze landing zone
  runs           pipeline step executions with parameters, model version, status, timings (lineage)
  sheets         drawing sheets with title block facts
  symbols        detected symbols (bbox, class, confidence, model version, evidence handles, status)
  texts          detected text (DXF text + OCR), label, association to a symbol
  lines          detected line segments (process / signal, solid / dashed)
  graph_nodes    persisted graph (SQL-graph style node table) per source + version
  graph_edges    persisted graph edges with evidence and direction
  assets         ISA-95 asset hierarchy (enterprise > site > area > unit > equipment > instrument)
  reviews        human decisions (approve / correct / reject) = corrected outputs + active-learning labels
  training_samples  labelled crops (seed legend, data-scientist uploads, approved inference)
  models         model registry: pickled model, classes, metrics, active flag
  search         FTS5 index over tags, descriptions, notes

File outputs per drawing follow the Azure design's directory layout (inference-results/<pid_id>/...); this database
adds versioning, which that design lists as missing.
"""
import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(pid_id TEXT PRIMARY KEY, file_name TEXT, sha256 TEXT, fmt TEXT, size_bytes INTEGER,
    title TEXT, drawing_no TEXT, revision TEXT, n_sheets INTEGER, ingested_at TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, pid_id TEXT, step TEXT, version INTEGER,
    params TEXT, model_id INTEGER, status TEXT, message TEXT, started REAL, finished REAL, outputs TEXT);
CREATE TABLE IF NOT EXISTS sheets(pid_id TEXT, sheet_no INTEGER, title TEXT, kind TEXT, drawing_no TEXT, revision TEXT,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, PRIMARY KEY(pid_id, sheet_no));
CREATE TABLE IF NOT EXISTS symbols(pid_id TEXT, symbol_id TEXT, version INTEGER, sheet_no INTEGER, cls TEXT, category TEXT,
    confidence REAL, x0 REAL, y0 REAL, x1 REAL, y1 REAL, tag TEXT, method TEXT, model_id INTEGER, handles TEXT,
    status TEXT DEFAULT 'candidate', props TEXT, PRIMARY KEY(pid_id, symbol_id, version));
CREATE TABLE IF NOT EXISTS texts(pid_id TEXT, text_id TEXT, version INTEGER, sheet_no INTEGER, text TEXT, label TEXT,
    confidence REAL, x0 REAL, y0 REAL, x1 REAL, y1 REAL, source TEXT, handle TEXT, symbol_id TEXT,
    PRIMARY KEY(pid_id, text_id, version));
CREATE TABLE IF NOT EXISTS lines(pid_id TEXT, line_id TEXT, version INTEGER, sheet_no INTEGER, kind TEXT, style TEXT,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, handles TEXT, PRIMARY KEY(pid_id, line_id, version));
CREATE TABLE IF NOT EXISTS graph_nodes(pid_id TEXT, version INTEGER, node_id TEXT, kind TEXT, label TEXT, cls TEXT,
    sheet_no INTEGER, x REAL, y REAL, confidence REAL, status TEXT, props TEXT, PRIMARY KEY(pid_id, version, node_id));
CREATE TABLE IF NOT EXISTS graph_edges(pid_id TEXT, version INTEGER, edge_id TEXT, src TEXT, dst TEXT, kind TEXT,
    direction TEXT, confidence REAL, status TEXT, evidence TEXT, PRIMARY KEY(pid_id, version, edge_id));
CREATE TABLE IF NOT EXISTS assets(pid_id TEXT, asset_id TEXT, parent_id TEXT, level TEXT, tag TEXT, name TEXT,
    asset_class TEXT, sheet_no INTEGER, bom_item TEXT, match_status TEXT, confidence REAL, props TEXT,
    PRIMARY KEY(pid_id, asset_id));
CREATE TABLE IF NOT EXISTS reviews(review_id INTEGER PRIMARY KEY AUTOINCREMENT, pid_id TEXT, object_type TEXT,
    object_id TEXT, decision TEXT, old_value TEXT, new_value TEXT, reviewer TEXT, comment TEXT, ts TEXT,
    used_for_training INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS training_samples(sample_id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, funnel TEXT,
    pid_id TEXT, source_ref TEXT, png BLOB, approved INTEGER DEFAULT 1, created TEXT);
CREATE TABLE IF NOT EXISTS models(model_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, task TEXT, algo TEXT,
    classes TEXT, metrics TEXT, n_train INTEGER, data_hash TEXT, blob BLOB, active INTEGER DEFAULT 0, created TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""
FTS = "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(pid_id, kind, ref, text, sheet_no UNINDEXED)"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        # One connection per thread: Streamlit serves sessions from several threads, and a single shared
        # sqlite3 connection fails under concurrent use ("bad parameter or other API misuse").
        self._local = threading.local()
        self._write = threading.RLock()
        db = self.db
        db.executescript(SCHEMA)
        try:
            db.execute(FTS)
            self.fts = True
        except sqlite3.OperationalError:
            self.fts = False
        db.commit()

    @property
    def db(self):
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            self._local.con = c
        return c

    @staticmethod
    def _args(args):
        """Plain Python values only: numpy ints/floats are rejected by newer sqlite3 bindings."""
        out = []
        for a in (args or ()):
            if hasattr(a, "item") and not isinstance(a, (str, bytes)):
                a = a.item()
            out.append(a)
        return tuple(out)

    # generic helpers
    def q(self, sql, args=()):
        return [dict(r) for r in self.db.execute(sql, self._args(args)).fetchall()]

    def one(self, sql, args=()):
        r = self.db.execute(sql, self._args(args)).fetchone()
        return dict(r) if r else None

    def x(self, sql, args=()):
        with self._write:
            cur = self.db.execute(sql, self._args(args)); self.db.commit(); return cur

    def many(self, sql, rows):
        with self._write:
            self.db.executemany(sql, [self._args(r) for r in rows]); self.db.commit()

    def df(self, sql, args=()):
        """DataFrame from a query, built directly (works the same on pandas 2 and 3)."""
        import pandas as pd
        cur = self.db.execute(sql, self._args(args))
        cols = [d[0] for d in cur.description]
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)

    # lineage
    def start_run(self, pid_id, step, params=None, model_id=None):
        v = self.latest_version(pid_id, step) + 1
        cur = self.x("INSERT INTO runs(pid_id, step, version, params, model_id, status, started) VALUES(?,?,?,?,?,?,?)",
                     (pid_id, step, v, json.dumps(params or {}), model_id, "running", time.time()))
        return cur.lastrowid, v

    def finish_run(self, run_id, status="succeeded", message="", outputs=None):
        self.x("UPDATE runs SET status=?, message=?, finished=?, outputs=? WHERE run_id=?",
               (status, message, time.time(), json.dumps(outputs or {}), run_id))

    def latest_version(self, pid_id, step):
        r = self.one("SELECT MAX(version) v FROM runs WHERE pid_id=? AND step=? AND status='succeeded'", (pid_id, step))
        return (r or {}).get("v") or 0

    # key / value
    def kv_get(self, k, default=None):
        r = self.one("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(r["v"]) if r else default

    def kv_set(self, k, v):
        self.x("INSERT OR REPLACE INTO kv(k, v) VALUES(?,?)", (k, json.dumps(v)))

    # reviews
    def review(self, pid_id, object_type, object_id, decision, old_value=None, new_value=None, reviewer="engineer", comment=""):
        self.x("INSERT INTO reviews(pid_id, object_type, object_id, decision, old_value, new_value, reviewer, comment, ts) VALUES(?,?,?,?,?,?,?,?,?)",
               (pid_id, object_type, object_id, decision, json.dumps(old_value), json.dumps(new_value), reviewer, comment, now()))
        if object_type == "symbol":
            st = {"approve": "approved", "correct": "corrected", "reject": "rejected"}.get(decision, "candidate")
            sets, args = ["status=?"], [st]
            if decision == "correct" and isinstance(new_value, dict):
                for k in ("cls", "tag", "category"):
                    if k in new_value:
                        sets.append(f"{k}=?"); args.append(new_value[k])
            self.x(f"UPDATE symbols SET {', '.join(sets)} WHERE pid_id=? AND symbol_id=?", (*args, pid_id, object_id))
        if object_type == "edge":
            st = {"approve": "approved", "reject": "rejected", "correct": "corrected"}.get(decision, "candidate")
            self.x("UPDATE graph_edges SET status=? WHERE pid_id=? AND edge_id=?", (st, pid_id, object_id))
            if decision == "correct" and isinstance(new_value, dict) and new_value.get("direction"):
                self.x("UPDATE graph_edges SET direction=? WHERE pid_id=? AND edge_id=?", (new_value["direction"], pid_id, object_id))

    # search
    def index(self, rows):
        if self.fts:
            self.many("INSERT INTO search(pid_id, kind, ref, text, sheet_no) VALUES(?,?,?,?,?)", rows)

    def search(self, query, limit=50):
        if not query.strip():
            return []
        if self.fts:
            toks = " OR ".join('"' + t.replace('"', '') + '"*' for t in query.split())
            try:
                return self.q("SELECT pid_id, kind, ref, text, sheet_no, bm25(search) score FROM search WHERE search MATCH ? ORDER BY score LIMIT ?", (toks, limit))
            except sqlite3.OperationalError:
                pass
        like = f"%{query}%"
        return self.q("SELECT pid_id, 'symbol' kind, symbol_id ref, COALESCE(tag, cls) text, sheet_no FROM symbols WHERE tag LIKE ? OR cls LIKE ? LIMIT ?", (like, like, limit))
