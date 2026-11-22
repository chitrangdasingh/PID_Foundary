"""
training.py - the training & development workflow and the model registry.

Training data funnels (as in the Azure P&ID design, adapted to CAD input):
  1. seed       - every symbol on the drawing's own legend sheet (labelled by its legend description), plus
                  synthetic negatives (glyphs, line pieces, flow arrows) so text and lines are not called symbols
  2. upload     - data scientists add labelled crops (PNG + label) from any labelling tool  (scripts/upload_training_data.py)
  3. approved   - symbols an engineer approved or corrected in the review queue (active learning)
Every sample is stored as PNG in training_samples, so a model can always be rebuilt and audited.
Models are pickled into the models table with classes, metrics and a data hash; one model per task is active.
"""
import hashlib
import io
import json
import pickle
import random
import re
import zlib

import cv2
import numpy as np

from . import symbols as Y
from .legend import read_legend, taxonomy
from .render import render, in_box
from .store import now

BUBBLE_MAP = [  # legend description -> collapsed bubble class (location x function)
    (r"AUXIL", "Auxiliary"), (r"PANEL", "Panel"), (r"SKID|FIELD", "Field")]
FUNC_MAP = [(r"COMPUTER", "Computer"), (r"PLC", "PLC"), (r"DISTRIBUTED", "SharedDisplay"), (r"ANALOG|INSTRUMENT", "Discrete")]


def class_for(desc):
    cat, sub = taxonomy(desc)
    d = desc.upper()
    if sub == "Bubble":
        loc = next((v for k, v in BUBBLE_MAP if re.search(k, d)), "Field")
        fn = next((v for k, v in FUNC_MAP if re.search(k, d)), "Discrete")
        return f"Instrument/Bubble/{loc}-{fn}"
    if sub in ("Line", "SignalLine", "Insulation"):
        return None                      # line types are recognised by the line stage, not as symbols
    name = re.sub(r"\s*\(.*?\)", "", desc).title().replace(" ", "")
    return f"{cat}/{sub}/{name}"


NEGATIVES = ["Text/Glyph/Character", "Line/Fragment/Segment", "Annotation/Arrow/FlowArrow"]


def to_png(img):
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def from_png(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)


def seed_from_legend(store, ents, legend_sheet, pid_id=None):
    rows = read_legend(ents, legend_sheet)
    n = 0
    store.x("DELETE FROM training_samples WHERE funnel='seed' AND (pid_id=? OR ? IS NULL)", (pid_id, pid_id))
    for r in rows:
        cls = class_for(r["description"])
        if not cls:
            continue
        geo = [e for e in ents if e["type"] not in ("TEXT", "MTEXT") and in_box(e, r["box"], 0.2)]
        if not geo:
            continue
        # strip pipe stubs that run to the cell edge so the seed looks like an in-line symbol on a sheet
        b = (min(g["bbox"][0] for g in geo), min(g["bbox"][1] for g in geo), max(g["bbox"][2] for g in geo), max(g["bbox"][3] for g in geo))
        s = max(b[2] - b[0], b[3] - b[1], 0.5)
        img = Y.square(render(geo, b, px_per_unit=Y.CROP * 0.85 / s, pad=0.15))
        store.x("INSERT INTO training_samples(label, funnel, pid_id, source_ref, png, created) VALUES(?,?,?,?,?,?)",
                (cls, "seed", pid_id, f"legend #{r['sr']}: {r['description']}", to_png(img), now()))
        n += 1
    return n, rows


def synth_negative(kind, rng):
    img = np.full((96, 96), 255, np.uint8)
    if kind.startswith("Text"):
        chars = "ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789-/"
        s = "".join(rng.choice(chars) for _ in range(rng.choice([1, 1, 2, 3])))
        sc = rng.uniform(1.4, 2.6)
        cv2.putText(img, s, (rng.randint(4, 20), rng.randint(55, 80)), cv2.FONT_HERSHEY_SIMPLEX, sc, 0, rng.choice([1, 2]), cv2.LINE_AA)
    elif kind.startswith("Line"):
        for _ in range(rng.choice([1, 1, 2])):
            if rng.random() < 0.5:
                y = rng.randint(10, 86); cv2.line(img, (rng.randint(0, 30), y), (rng.randint(60, 95), y), 0, rng.choice([1, 2]))
            else:
                x = rng.randint(10, 86); cv2.line(img, (x, rng.randint(0, 30)), (x, rng.randint(60, 95)), 0, rng.choice([1, 2]))
        if rng.random() < 0.3:
            cv2.line(img, (10, 10), (85, 85), 0, 1)
    else:
        c = (48, 48); a = rng.choice([0, 90, 180, 270]); L = rng.randint(18, 34)
        pts = np.array([[L, 0], [-L, L * 0.55], [-L, -L * 0.55]])
        R = cv2.getRotationMatrix2D((0, 0), a, 1)[:, :2]
        pts = (pts @ R.T + c).astype(np.int32)
        cv2.fillPoly(img, [pts], 0)
        if rng.random() < 0.6:
            cv2.line(img, (0, 48), (96, 48), 0, 1) if a in (0, 180) else cv2.line(img, (48, 0), (48, 96), 0, 1)
    ys, xs = np.where(img < 128)
    if len(xs):
        img = img[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    return Y.square(img)


def augment(img, rng):
    out = img.copy()
    k = rng.choice([0, 0, 1, 2, 3])
    if k:
        out = np.rot90(out, k).copy()
    if rng.random() < 0.3:
        out = np.fliplr(out).copy()
    if rng.random() < 0.5:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))       # thicker ink
    s = rng.uniform(0.75, 1.0)
    small = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full_like(out, 255)
    oy, ox = rng.randint(0, out.shape[0] - small.shape[0]), rng.randint(0, out.shape[1] - small.shape[1])
    canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    if rng.random() < 0.4:   # a pipe stub through the symbol, as on the process sheets
        y = Y.CROP // 2 + rng.randint(-3, 3)
        if rng.random() < 0.5:
            cv2.line(canvas, (0, y), (rng.randint(4, 14), y), 0, 1)
        else:
            cv2.line(canvas, (Y.CROP - rng.randint(4, 14), y), (Y.CROP, y), 0, 1)
    if rng.random() < 0.3:
        noise = (np.random.default_rng(rng.randint(0, 10**6)).random(canvas.shape) < 0.01)
        canvas[noise] = 0
    return canvas


def build_dataset(store, per_class=60, neg_per_class=150, seed=7, include=("seed", "upload", "approved")):
    rng = random.Random(seed)
    rows = store.q(f"SELECT label, funnel, png FROM training_samples WHERE approved=1 AND funnel IN ({','.join('?' * len(include))})", include)
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append((r["funnel"], from_png(r["png"])))
    X, y, groups = [], [], []
    for lab, items in by.items():
        k = max(per_class, 6 * len(items))
        for i in range(k):
            f, im = items[i % len(items)]
            X.append(Y.features(im if i < len(items) else augment(im, rng))); y.append(lab); groups.append(f)
    for neg in NEGATIVES:
        for _ in range(neg_per_class):
            X.append(Y.features(augment(synth_negative(neg, rng), rng))); y.append(neg); groups.append("synthetic")
    h = hashlib.sha256(json.dumps(sorted((r["label"], len(r["png"])) for r in rows)).encode()).hexdigest()[:16]
    return np.array(X), np.array(y), groups, h, {k: len(v) for k, v in by.items()}


def train_symbol_model(store, algo="svm", name="symbol-classifier", activate=True, per_class=60):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    X, y, groups, h, counts = build_dataset(store, per_class=per_class)
    if len(set(y)) < 2:
        raise ValueError("need at least two classes of training samples")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    if algo == "rf":
        clf = RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=2, max_features="sqrt",
                                     random_state=0, n_jobs=-1, class_weight="balanced")
    else:
        from sklearn.decomposition import PCA
        clf = make_pipeline(StandardScaler(), PCA(n_components=120, random_state=0),
                            CalibratedClassifierCV(SVC(C=10, gamma="scale", class_weight="balanced", random_state=0), ensemble=False, cv=3))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True, zero_division=0)
    metrics = dict(holdout_accuracy=round(float(accuracy_score(yte, pred)), 4),
                   macro_f1=round(float(f1_score(yte, pred, average="macro")), 4),
                   n_classes=len(set(y)), n_samples=len(y), source_counts=counts,
                   per_class={k: round(v["f1-score"], 3) for k, v in rep.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")},
                   note="Hold-out is drawn from augmented seed / upload / approved samples; field accuracy comes from the review queue.")
    clf.fit(X, y)   # final model on all data
    blob = zlib.compress(pickle.dumps(clf), 6)
    if activate:
        store.x("UPDATE models SET active=0 WHERE task='symbol'")
    cur = store.x("INSERT INTO models(name, task, algo, classes, metrics, n_train, data_hash, blob, active, created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (name, "symbol", algo, json.dumps(sorted(set(y))), json.dumps(metrics), len(y), h, blob, 1 if activate else 0, now()))
    return cur.lastrowid, metrics


def load_active(store, task="symbol"):
    r = store.one("SELECT model_id, blob, classes FROM models WHERE task=? AND active=1 ORDER BY model_id DESC LIMIT 1", (task,))
    if not r:
        return None, None
    b = r["blob"]
    try:
        b = zlib.decompress(b)
    except zlib.error:
        pass
    return r["model_id"], pickle.loads(b)


def harvest_approved(store):
    """Active learning: turn reviewed symbols into training samples (funnel 'approved')."""
    rows = store.q("""SELECT r.review_id, r.pid_id, r.object_id, r.decision, r.new_value, s.cls FROM reviews r
                      JOIN symbols s ON s.pid_id=r.pid_id AND s.symbol_id=r.object_id
                      WHERE r.object_type='symbol' AND r.used_for_training=0 AND r.decision IN ('approve','correct')""")
    n = 0
    for r in rows:
        crop = store.one("SELECT png FROM training_samples WHERE funnel='inference' AND source_ref=? AND pid_id=?", (r["object_id"], r["pid_id"]))
        if not crop:
            continue
        lab = r["cls"]
        if r["decision"] == "correct":
            try:
                lab = json.loads(r["new_value"]).get("cls", lab)
            except Exception:
                pass
        store.x("INSERT INTO training_samples(label, funnel, pid_id, source_ref, png, created) VALUES(?,?,?,?,?,?)",
                (lab, "approved", r["pid_id"], r["object_id"], crop["png"], now()))
        store.x("UPDATE reviews SET used_for_training=1 WHERE review_id=?", (r["review_id"],))
        n += 1
    return n


def add_upload(store, label, png_bytes, source_ref="upload"):
    store.x("INSERT INTO training_samples(label, funnel, source_ref, png, created) VALUES(?,?,?,?,?)",
            (label, "upload", source_ref, png_bytes, now()))
