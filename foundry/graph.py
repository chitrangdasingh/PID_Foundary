"""
graph.py - line detection (vector) and graph construction.

Line detection: LINE entities and every LWPOLYLINE segment become segments; short collinear dashes are chained into
dashed (signal) lines; segments that lie inside a symbol, a word or an equipment outline are removed.
Graph construction: segment endpoints are snapped, T-junctions are joined, each connected line network is attached to
the symbols / equipment / connectors its ends touch, and every pair of attached objects becomes a candidate
connection. Direction is taken from flow arrows when an arrow sits on the network; otherwise it stays 'unknown'.
Every edge keeps the entity handles that support it and starts with status 'candidate'.
"""
import math
from collections import defaultdict

import networkx as nx


def segments_from_entities(ents, box, exclude_boxes, min_len=0.25):
    segs = []
    x0, y0, x1, y1 = box
    for e in ents:
        if e["type"] == "LINE":
            pts = [(e.get("x", 0), e.get("y", 0)), (e.get("x2", 0), e.get("y2", 0))]
        elif e["type"] in ("LWPOLYLINE", "POLYLINE") and len(e.get("pts", [])) >= 2:
            pts = list(e["pts"]) + ([e["pts"][0]] if e.get("closed") else [])
        else:
            continue
        b = e.get("bbox")
        if not b or b[0] < x0 or b[2] > x1 or b[1] < y0 or b[3] > y1:
            continue
        if (b[2] - b[0]) > 0.9 * (x1 - x0) or (b[3] - b[1]) > 0.9 * (y1 - y0):
            continue
        if (b[0] > x1 - 95 and b[3] < y0 + 75) or b[3] < y0 + 30:
            continue
        for p, q in zip(pts, pts[1:]):
            L = math.dist(p, q)
            if L < min_len:
                continue
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            if any(bx[0] <= mx <= bx[2] and bx[1] <= my <= bx[3] and bx[0] <= p[0] <= bx[2] and bx[0] <= q[0] <= bx[2]
                   and bx[1] <= p[1] <= bx[3] and bx[1] <= q[1] <= bx[3] for bx in exclude_boxes):
                continue
            segs.append(dict(p=p, q=q, len=L, handle=e.get("handle"), dash=L < 2.4))
    return chain_dashes(segs)


def chain_dashes(segs, gap=1.6):
    """Merge collinear dashes into dashed lines. Solid segments pass through."""
    solid = [s for s in segs if not s["dash"]]
    dashes = [s for s in segs if s["dash"]]
    used = set()
    out = [dict(s, style="solid") for s in solid]
    def direction(s):
        dx, dy = s["q"][0] - s["p"][0], s["q"][1] - s["p"][1]
        return "h" if abs(dx) >= abs(dy) * 3 else "v" if abs(dy) >= abs(dx) * 3 else "d"
    groups = defaultdict(list)
    for i, s in enumerate(dashes):
        d = direction(s)
        if d == "d":
            continue
        key = (d, round((s["p"][1] if d == "h" else s["p"][0]) / 0.3))
        groups[key].append(i)
    for (d, _), idx in groups.items():
        k = 0 if d == "h" else 1
        idx.sort(key=lambda i: min(dashes[i]["p"][k], dashes[i]["q"][k]))
        run = []
        for i in idx + [None]:
            if i is not None and run:
                last = dashes[run[-1]]
                if min(dashes[i]["p"][k], dashes[i]["q"][k]) - max(last["p"][k], last["q"][k]) <= gap:
                    run.append(i); continue
            if len(run) < 3:
                for j in run:
                    out.append(dict(dashes[j], style="solid")); used.add(j)
            if len(run) >= 3:
                lo = min(min(dashes[j]["p"][k], dashes[j]["q"][k]) for j in run)
                hi = max(max(dashes[j]["p"][k], dashes[j]["q"][k]) for j in run)
                c = dashes[run[0]]["p"][1 - k]
                p, q = ((lo, c), (hi, c)) if k == 0 else ((c, lo), (c, hi))
                out.append(dict(p=p, q=q, len=hi - lo, handle=dashes[run[0]]["handle"], style="dashed", dash=False,
                                handles=[dashes[j]["handle"] for j in run]))
                used.update(run)
            run = [i] if i is not None else []
    out += [dict(s, style="solid") for i, s in enumerate(dashes) if i not in used]
    return out


def _near_seg(pt, s, tol):
    (x1, y1), (x2, y2) = s["p"], s["q"]
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return False, 0
    t = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / L2
    if t <= 0.02 or t >= 0.98:
        return False, t
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(pt[0] - px, pt[1] - py) <= tol, t


def split_at_tees(segs, tol=0.3, cell=5.0):
    """Split a segment where another segment's end touches its interior, so tees become real graph nodes."""
    grid = defaultdict(list)
    for i, s in enumerate(segs):
        xs, ys = sorted((s["p"][0], s["q"][0])), sorted((s["p"][1], s["q"][1]))
        for gx in range(int(xs[0] // cell), int(xs[1] // cell) + 1):
            for gy in range(int(ys[0] // cell), int(ys[1] // cell) + 1):
                grid[(gx, gy)].append(i)
    cuts = defaultdict(set)
    for j, s in enumerate(segs):
        for pt in (s["p"], s["q"]):
            for i in grid.get((int(pt[0] // cell), int(pt[1] // cell)), []):
                if i != j:
                    ok, t = _near_seg(pt, segs[i], tol)
                    if ok:
                        cuts[i].add(round(t, 4))
    out = []
    for i, s in enumerate(segs):
        ts = [0.0] + sorted(cuts.get(i, ())) + [1.0]
        (x1, y1), (x2, y2) = s["p"], s["q"]
        for a, b in zip(ts, ts[1:]):
            if b - a < 1e-3:
                continue
            p = (x1 + a * (x2 - x1), y1 + a * (y2 - y1)); q = (x1 + b * (x2 - x1), y1 + b * (y2 - y1))
            out.append(dict(s, p=p, q=q, len=math.dist(p, q)))
    return out


ROLE_THROUGH, ROLE_SINK, ROLE_SOURCE = "through", "sink", "source"


def build(objects, segs, arrows=(), snap=0.35, attach=0.7, sheet_no=None):
    """objects: [{id, bbox, kind, role}], segs: line segments, arrows: [{tip:(x,y), base:(x,y)}].
    role: 'through' (in-line valve / filter: flow passes), 'sink' (instrument, equipment: flow stops),
    'source' (FROM connector), 'sink' (TO connector).
    Returns (edges, stats). Edge direction: 'a->b' or 'unknown', with direction_method and confidence."""
    segs = split_at_tees(segs, tol=snap * 0.8)
    # merge end points that coincide (within snap) into one junction node
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    key = lambda p: (round(p[0] / snap), round(p[1] / snap))
    buckets = defaultdict(list)
    pts = {}
    for i, sg in enumerate(segs):
        for k, p in ((0, sg["p"]), (1, sg["q"])):
            pts[(i, k)] = p
            buckets[key(p)].append((i, k))
    for (i, k), p in pts.items():
        kx, ky = key(p)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for m in buckets.get((kx + ox, ky + oy), ()):
                    if m != (i, k) and math.dist(pts[m], p) <= snap:
                        ra, rb = find((i, k)), find(m)
                        if ra != rb:
                            parent[ra] = rb
    node_of = {ik: ("p",) + find(ik) for ik in pts}
    G = nx.Graph()
    for i, sg in enumerate(segs):
        u, v = node_of[(i, 0)], node_of[(i, 1)]
        G.add_node(u, pt=sg["p"]); G.add_node(v, pt=sg["q"])
        if u != v:
            G.add_edge(u, v, seg=i)
    obj_nodes = {}
    role = {o["id"]: o.get("role", ROLE_SINK) for o in objects}
    for o in objects:
        bb = o["bbox"]; t = attach if o.get("kind") != "equipment" else attach * 1.5
        for n, d in G.nodes(data=True):
            x, y = d["pt"]
            if bb[0] - t <= x <= bb[2] + t and bb[1] - t <= y <= bb[3] + t:
                obj_nodes.setdefault(o["id"], set()).add(n)
    comp_of = {}
    for ci, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_of[n] = ci
    line_degree = {n: G.degree(n) for n in G.nodes}
    for oid, ns in obj_nodes.items():
        for n in ns:
            G.add_edge(("o", oid), n, attach=True)
    # ---- flow orientation: seeds from arrows and FROM / TO connectors, then propagate along the lines
    MAX_JUNCTIONS = 3
    flow, method, conflicts = {}, {}, []

    def set_dir(u, v, how):
        k = frozenset((u, v))
        if k in flow:
            if flow[k] != (u, v) and method[k][2] != how[2] and method[k][1] == 0 and how[1] == 0:
                conflicts.append((u, v, how))      # two independent seeds disagree on the same run
            return False
        flow[k] = (u, v); method[k] = how
        return True

    q = []
    for ar in arrows:
        best = None
        for u, v, d in G.edges(data=True):
            if d.get("seg") is None:
                continue
            (x1, y1), (x2, y2) = G.nodes[u]["pt"], G.nodes[v]["pt"]
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            for pt in (ar["tip"], ar["base"]):
                t = max(0, min(1, ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / L2))
                dd = math.hypot(pt[0] - (x1 + t * dx), pt[1] - (y1 + t * dy))
                if dd < 0.8 and (best is None or dd < best[0]):
                    best = (dd, u, v)
        if best is None:
            continue
        _, u, v = best
        (x1, y1), (x2, y2) = G.nodes[u]["pt"], G.nodes[v]["pt"]
        dot = (ar["tip"][0] - ar["base"][0]) * (x2 - x1) + (ar["tip"][1] - ar["base"][1]) * (y2 - y1)
        if abs(dot) < 1e-9:
            continue
        up, dn = (u, v) if dot > 0 else (v, u)
        sid = f"arrow@{ar['tip'][0]:.1f},{ar['tip'][1]:.1f}"
        if set_dir(up, dn, ("arrow", 0, sid)):
            q.append((dn, up, "fwd", ("arrow", 0, sid)))
            q.append((up, dn, "bwd", ("arrow", 0, sid)))
    for oid, r in role.items():
        n = ("o", oid)
        if n not in G:
            continue
        for m in list(G.neighbors(n)):
            if r == ROLE_SOURCE and set_dir(n, m, ("connector", 0, oid)):
                q.append((m, n, "fwd", ("connector", 0, oid)))
            elif r == "to" and set_dir(m, n, ("connector", 0, oid)):
                q.append((m, n, "bwd", ("connector", 0, oid)))
    stop = {("o", oid) for oid, r in role.items() if r in (ROLE_SINK, "to", ROLE_SOURCE)}
    while q:
        node, came, mode, (src, nj, sid) = q.pop(0)
        if node in stop:
            continue
        if node[0] == "p" and line_degree.get(node, 0) > 2:
            nj += 1
            if nj > MAX_JUNCTIONS:
                continue
        for m in list(G.neighbors(node)):
            if m == came:
                continue
            ok = set_dir(node, m, (src, nj, sid)) if mode == "fwd" else set_dir(m, node, (src, nj, sid))
            if ok:
                q.append((m, node, mode, (src, nj, sid)))

    def label(h):
        return h[0] if h[1] == 0 else f"{h[0]} via {h[1]} junction{'s' if h[1] > 1 else ''}"

    # ---- object-level connections
    comps = defaultdict(set)
    for oid, ns in obj_nodes.items():
        for n in ns:
            comps[comp_of[n]].add(oid)
    style_of = defaultdict(set)
    handles_of = defaultdict(list)
    for i, sg in enumerate(segs):
        c = comp_of[node_of[(i, 0)]]
        style_of[c].add(sg["style"])
        handles_of[c].append(sg.get("handle"))

    def obj_dir(a, ci):
        """'out' if flow leaves object a into network ci, 'in' if it enters a, else None."""
        outs = ins = 0
        hows = set()
        for n in obj_nodes.get(a, ()):
            if comp_of[n] != ci:
                continue
            k = frozenset((("o", a), n))
            if k in flow:
                hows.add(method[k])
                if flow[k][0] == ("o", a):
                    outs += 1
                else:
                    ins += 1
        if outs and not ins:
            return "out", hows
        if ins and not outs:
            return "in", hows
        return None, hows

    conflict_comps = set()
    for u, v, _ in conflicts:
        for n in (u, v):
            if n in comp_of:
                conflict_comps.add(comp_of[n])

    def best_how(hows):
        if not hows:
            return ""
        return label(min(hows, key=lambda h: (h[1], h[0] != "arrow"))[:2])

    edges = []
    for ci, oids in comps.items():
        oids = sorted(oids)
        if len(oids) < 2:
            continue
        style = "signal" if style_of[ci] == {"dashed"} else "process"
        dirs = {a: obj_dir(a, ci) for a in oids}
        pairs = [(a, b) for k, a in enumerate(oids) for b in oids[k + 1:]]
        hub = None
        if len(oids) > 4:
            hub = f"J{sheet_no}-{ci}"
            pairs = [(hub, a) for a in oids]
        for a, b in pairs:
            direction, how = "unknown", ""
            if style == "process":
                if hub:
                    d, hw = dirs[b]
                    if d == "out":
                        direction, how = f"{b}->{a}", best_how(hw)
                    elif d == "in":
                        direction, how = f"{a}->{b}", best_how(hw)
                else:
                    (da, ha), (db, hb) = dirs[a], dirs[b]
                    if da == "out" and db != "out":
                        direction, how = f"{a}->{b}", best_how(ha | hb)
                    elif db == "out" and da != "out":
                        direction, how = f"{b}->{a}", best_how(ha | hb)
                    elif da == "in" and db != "in":
                        direction, how = f"{b}->{a}", best_how(ha | hb)
                    elif db == "in" and da != "in":
                        direction, how = f"{a}->{b}", best_how(ha | hb)
            conf = 0.75 if style == "process" else 0.6
            if hub:
                conf -= 0.1 if len(oids) <= 8 else 0.4
            if "offpage" in (role.get(a), role.get(b)):
                conf -= 0.05
            dconf = 0.0
            if direction != "unknown" and ci in conflict_comps:
                how += " (conflicting seeds on this run)"
            if direction != "unknown":
                dconf = {"arrow": 0.85, "connector": 0.75}.get(how, 0.6 if "1 junction" in how else 0.5)
                if "conflicting" in how:
                    dconf = 0.3
            edges.append(dict(src=a, dst=b, kind=style, direction=direction, direction_method=how or None, direction_confidence=dconf,
                              confidence=round(conf, 2), evidence=[h for h in handles_of[ci] if h][:30], hub=hub))
    stats = dict(segments=len(segs), dashed=sum(1 for s in segs if s["style"] == "dashed"),
                 networks=len(set(comp_of.values())), attached_objects=len(obj_nodes),
                 networks_with_2plus=sum(1 for v in comps.values() if len(v) >= 2),
                 arrows=len(arrows), oriented_segments=sum(1 for k in flow if all(x[0] == "p" for x in k)),
                 direction_conflicts=len(conflicts))
    return edges, stats
