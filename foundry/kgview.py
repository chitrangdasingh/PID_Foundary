"""Interactive knowledge-graph view: one self-contained HTML page (SVG + plain JS, no CDN) shown inside Streamlit.

Click any node to see its detection data, the matching BOM row, its ISA-95 hierarchy path and its connections
(upstream / downstream / undirected); click a connection to jump to that asset. Pan with drag, zoom with the wheel,
search by tag, filter by kind, and switch between the drawing layout (nodes where they sit on the sheets) and a
network layout (force-directed)."""
import json
import math
import os

import networkx as nx

KIND_COLOURS = {"Equipment": "#C0392B", "Instrument": "#2471A3", "Piping": "#27AE60", "Connector": "#E67E22",
                "Junction": "#8E8E8E", "Annotation": "#B7950B"}


def _bom_rows(bom_path):
    """Tag -> BOM row (all columns of the 'BOM' sheet), or {} when no workbook is available."""
    if not bom_path or not os.path.exists(bom_path):
        return {}
    try:
        import pandas as pd
        x = pd.ExcelFile(bom_path)
        sh = "BOM" if "BOM" in x.sheet_names else x.sheet_names[0]
        df = pd.read_excel(x, sheet_name=sh)
    except Exception:
        return {}
    tag_col = next((c for c in df.columns if str(c).strip().lower() == "tag"), None)
    if tag_col is None:
        return {}
    out = {}
    for r in df.to_dict("records"):
        t = str(r.get(tag_col) or "").strip().upper()
        if t:
            out[t] = {str(k): ("" if (v is None or (isinstance(v, float) and math.isnan(v))) else (int(v) if isinstance(v, float) and v.is_integer() else v))
                      for k, v in r.items()}
    return out


def build_payload(store, pid_id, G, bom_path=None, sheet=None, include_annotations=False):
    """Nodes, edges and per-node details for the viewer."""
    keep = [n for n, d in G.nodes(data=True)
            if (include_annotations or d.get("kind") != "Annotation") and (not sheet or d.get("sheet") == sheet)]
    H = G.subgraph(keep).copy()

    # layout 1: drawing positions, sheets placed on a grid
    sheets = sorted({d.get("sheet") or 0 for _, d in H.nodes(data=True)})
    cols = max(1, math.ceil(math.sqrt(len(sheets))))
    box = {}
    for s in sheets:
        xs = [d["x"] for _, d in H.nodes(data=True) if (d.get("sheet") or 0) == s]
        ys = [d["y"] for _, d in H.nodes(data=True) if (d.get("sheet") or 0) == s]
        box[s] = (min(xs), max(xs), min(ys), max(ys)) if xs else (0, 1, 0, 1)
    pos_d = {}
    for n, d in H.nodes(data=True):
        s = d.get("sheet") or 0
        i = sheets.index(s)
        x0, x1, y0, y1 = box[s]
        u = (d["x"] - x0) / max(x1 - x0, 1e-9)
        v = 1 - (d["y"] - y0) / max(y1 - y0, 1e-9)      # CAD y points up, screen y points down
        pos_d[n] = ((i % cols) * 1.15 + u, (i // cols) * 0.8 + v * 0.7)
    # layout 2: force-directed network
    U = nx.Graph(H)
    pos_n = nx.spring_layout(U, seed=7, k=1.6 / math.sqrt(max(U.number_of_nodes(), 1)), iterations=80) if U.number_of_nodes() else {}

    assets = {a["asset_id"]: a for a in store.q("SELECT * FROM assets WHERE pid_id=?", (pid_id,))}
    by_tag = {}
    for a in assets.values():
        if a.get("tag"):
            by_tag.setdefault((a["tag"].upper(), a.get("sheet_no")), a)
    rec = store.kv_get(f"recon:{pid_id}", {}) or {}
    recon = {str(r.get("tag", "")).upper(): r for r in rec.get("rows", [])}
    bom = _bom_rows(bom_path)

    def path_of(a):
        out, seen = [], set()
        while a and a["asset_id"] not in seen:
            seen.add(a["asset_id"])
            out.append(f'{a["level"]}: {a.get("tag") or a.get("name") or a["asset_id"]}')
            a = assets.get(a.get("parent_id"))
        return list(reversed(out))

    nodes, details = [], {}
    for n, d in H.nodes(data=True):
        tag = (d.get("label") or "").strip()
        a = assets.get(n) or by_tag.get((tag.upper(), d.get("sheet")))
        r = recon.get(tag.upper())
        b = bom.get(tag.upper())
        bom_status = (r or {}).get("status") or ((a or {}).get("match_status")) or ("matched" if b else "not in BOM")
        nodes.append(dict(id=n, t=tag or n, k=d.get("kind", ""), s=d.get("sheet") or 0,
                          dx=round(pos_d[n][0], 4), dy=round(pos_d[n][1], 4),
                          nx=round(float(pos_n[n][0]), 4) if n in pos_n else 0, ny=round(float(pos_n[n][1]), 4) if n in pos_n else 0,
                          b=1 if b else 0))
        ups, downs, links = [], [], []
        for a_, b_, ed in H.in_edges(n, data=True):
            (ups if "->" in ed.get("direction", "") else links).append([a_, H.nodes[a_].get("label") or a_, ed.get("kind", "")])
        for a_, b_, ed in H.out_edges(n, data=True):
            (downs if "->" in ed.get("direction", "") else links).append([b_, H.nodes[b_].get("label") or b_, ed.get("kind", "")])
        details[n] = dict(tag=tag or n, kind=d.get("kind", ""), cls=d.get("cls", ""), sheet=d.get("sheet") or 0,
                          confidence=round(float(d.get("confidence") or 0), 2), status=d.get("status") or "candidate",
                          bom_status=bom_status, bom=b or {}, path=path_of(a) if a else [],
                          up=ups, down=downs, links=links)
    edges = []
    for a_, b_, ed in H.edges(data=True):
        edges.append([a_, b_, 1 if "->" in ed.get("direction", "") else 0, 1 if ed.get("kind") == "signal" else 0])
    return dict(nodes=nodes, edges=edges, details=details, sheets=sheets,
                counts=dict(nodes=len(nodes), edges=len(edges), bom=sum(x["b"] for x in nodes)))


def html(payload, height=720):
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data).replace("__H__", str(height)).replace("__COL__", json.dumps(KIND_COLOURS))


TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;font:13px/1.4 system-ui,-apple-system,Segoe UI,Arial,sans-serif;color:#1f2a30;background:#fff}
.wrap{display:grid;grid-template-columns:1fr 340px;height:__H__px;border:1px solid #dde3e6;border-radius:8px;overflow:hidden}
.left{position:relative;background:#fafbfc}svg{width:100%;height:100%;cursor:grab;display:block}svg:active{cursor:grabbing}
.bar{position:absolute;top:8px;left:8px;right:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;z-index:2}
.bar input[type=text]{padding:5px 8px;border:1px solid #c9d2d6;border-radius:6px;width:150px;font:inherit}
.bar button,.bar label{background:#fff;border:1px solid #c9d2d6;border-radius:6px;padding:4px 8px;font:inherit;cursor:pointer}
.bar button.on{background:#1f6f78;color:#fff;border-color:#1f6f78}
.bar label{display:flex;gap:4px;align-items:center}#legend{display:flex;flex-wrap:wrap;gap:6px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.stats{position:absolute;bottom:8px;left:8px;color:#5c6b72;font-size:12px;background:#ffffffcc;padding:2px 6px;border-radius:4px}
.panel{border-left:1px solid #dde3e6;padding:12px 14px;overflow:auto;background:#fff}
.panel h3{margin:0 0 2px;font-size:18px}.muted{color:#5c6b72}.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:600;border:1px solid}
.ok{color:#1e7d46;border-color:#1e7d46}.warn{color:#a35f00;border-color:#a35f00}.no{color:#8e8e8e;border-color:#8e8e8e}
table{border-collapse:collapse;width:100%;margin:4px 0 10px}td{padding:3px 4px;border-bottom:1px solid #eef1f3;vertical-align:top}td:first-child{color:#5c6b72;width:42%}
h4{margin:12px 0 4px;font-size:13px;color:#1f6f78}.lnk{color:#1f5fa8;cursor:pointer;text-decoration:underline}
ol{margin:2px 0 8px;padding-left:18px}li{margin:1px 0}
</style></head><body><div class="wrap"><div class="left">
<div class="bar"><input id="q" type="text" placeholder="Find tag, e.g. TIC-202" list="tags"><datalist id="tags"></datalist>
<button id="lay">Network layout</button><button id="fit">Fit</button><span id="legend"></span></div>
<svg id="svg"><g id="vp"><g id="E"></g><g id="N"></g></g></svg><div class="stats" id="stats"></div></div>
<div class="panel" id="panel"><h3>Click an asset</h3><p class="muted">Click any node in the graph to see its BOM entry, where it sits in the asset hierarchy and what it connects to.</p>
<p class="muted">Ringed nodes have a BOM entry. Arrows show flow direction; dashed lines are signal lines.</p></div></div>
<script>
const D=__DATA__, COL=__COL__;
const svg=document.getElementById('svg'),vp=document.getElementById('vp'),gE=document.getElementById('E'),gN=document.getElementById('N');
const NS='http://www.w3.org/2000/svg',byId={};D.nodes.forEach(n=>byId[n.id]=n);
let mode='d',sel=null,hidden=new Set(['Junction']),T={x:0,y:0,k:1};
const W=()=>svg.clientWidth||800,H=()=>svg.clientHeight||600;
function xy(n){return mode==='d'?[n.dx,n.dy]:[n.nx,n.ny]}
function extent(){let a=[1e9,1e9,-1e9,-1e9];D.nodes.forEach(n=>{if(hidden.has(n.k))return;const[x,y]=xy(n);a=[Math.min(a[0],x),Math.min(a[1],y),Math.max(a[2],x),Math.max(a[3],y)]});return a}
let S=1,OX=0,OY=0;
function scale(){const[a,b,c,d]=extent();const w=Math.max(c-a,1e-6),h=Math.max(d-b,1e-6);S=Math.min((W()-60)/w,(H()-90)/h);OX=30-a*S+((W()-60)-w*S)/2;OY=60-b*S+((H()-90)-h*S)/2}
function P(n){const[x,y]=xy(n);return[x*S+OX,y*S+OY]}
function el(t,a){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e}
const defs=el('defs',{});const mk=el('marker',{id:'ar',viewBox:'0 0 10 10',refX:'16',refY:'5',markerWidth:'4',markerHeight:'4',markerUnits:'userSpaceOnUse',orient:'auto'});
mk.appendChild(el('path',{d:'M0,0L10,5L0,10z',fill:'#7a8a92'}));defs.appendChild(mk);svg.insertBefore(defs,vp);
let nodeEls={},edgeEls=[];
function draw(){gE.innerHTML='';gN.innerHTML='';nodeEls={};edgeEls=[];scale();const r=Math.max(2.5,Math.min(6,260/Math.sqrt(D.nodes.length)));
 D.edges.forEach(([a,b,dir,sig])=>{const A=byId[a],B=byId[b];if(!A||!B||hidden.has(A.k)||hidden.has(B.k))return;const[x1,y1]=P(A),[x2,y2]=P(B);
  const l=el('line',{x1,y1,x2,y2,stroke:'#9aa8ae','stroke-width':1,'stroke-dasharray':sig?'3,3':'','vector-effect':'non-scaling-stroke'});if(dir)l.setAttribute('marker-end','url(#ar)');l.dataset.a=a;l.dataset.b=b;gE.appendChild(l);edgeEls.push(l)});
 D.nodes.forEach(n=>{if(hidden.has(n.k))return;const[x,y]=P(n);const g=el('g',{transform:`translate(${x},${y})`,cursor:'pointer'});
  const c=el('circle',{r:n.k==='Equipment'?r*1.7:r,fill:COL[n.k]||'#777',stroke:n.b?'#111':'#fff','stroke-width':n.b?1.6:0.8});g.appendChild(c);
  const tt=el('title',{});tt.textContent=`${n.t} · ${n.k} · sheet ${n.s}`;g.appendChild(tt);
  if(n.k==='Equipment'||D.nodes.length<160){const t=el('text',{x:r*1.9,y:3,'font-size':n.k==='Equipment'?10:8,fill:'#1f2a30'});t.textContent=n.t;g.appendChild(t)}
  g.addEventListener('click',ev=>{ev.stopPropagation();pick(n.id)});gN.appendChild(g);nodeEls[n.id]=g});
 apply();if(sel)highlight(sel);
 document.getElementById('stats').textContent=`${D.counts.nodes} nodes · ${D.counts.edges} connections · ${D.counts.bom} with a BOM entry · drag to pan, wheel to zoom`}
function apply(){vp.setAttribute('transform',`translate(${T.x},${T.y}) scale(${T.k})`);const s=1/Math.sqrt(T.k);Object.values(nodeEls).forEach(g=>{const m=g.getAttribute('transform').match(/translate\(([^)]+)\)/);g.setAttribute('transform',`translate(${m[1]}) scale(${s})`)})}
function highlight(id){const nb=new Set([id]);edgeEls.forEach(l=>{const on=l.dataset.a===id||l.dataset.b===id;if(on){nb.add(l.dataset.a);nb.add(l.dataset.b)}
  l.setAttribute('stroke',on?'#1f6f78':'#9aa8ae');l.setAttribute('stroke-width',on?2.2:1);l.setAttribute('opacity',on?1:0.25)});
 Object.entries(nodeEls).forEach(([k,g])=>{g.setAttribute('opacity',nb.has(k)?1:0.25);g.firstChild.setAttribute('stroke-width',k===id?3:(byId[k].b?1.6:0.8));g.firstChild.setAttribute('stroke',k===id?'#000':(byId[k].b?'#111':'#fff'))})}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function rows(o){return '<table>'+Object.entries(o).filter(([k,v])=>v!==''&&v!==null).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')+'</table>'}
function list(a){return a.length?'<ol>'+a.map(([id,t,k])=>`<li><span class="lnk" data-id="${esc(id)}">${esc(t)}</span> <span class="muted">${k==='signal'?'(signal)':''}</span></li>`).join('')+'</ol>':'<p class="muted">none</p>'}
function pick(id){sel=id;const d=D.details[id];if(!d)return;highlight(id);
 const st=d.bom_status||'',cls=/match/.test(st)?'ok':(/bom/i.test(st)?'warn':'no');
 let h=`<h3>${esc(d.tag)}</h3><div class="muted">${esc(d.kind)} · ${esc((d.cls||'').split('/').join(' › '))}</div><p><span class="pill ${cls}">BOM: ${esc(st)}</span></p>`;
 h+='<h4>BOM entry</h4>'+(Object.keys(d.bom).length?rows(d.bom):'<p class="muted">No row for this tag in the BOM workbook.</p>');
 h+='<h4>On the drawing</h4>'+rows({'Sheet':d.sheet,'Confidence':d.confidence,'Review status':d.status});
 h+='<h4>Asset hierarchy (ISA-95)</h4>'+(d.path.length?'<ol>'+d.path.map(p=>`<li>${esc(p)}</li>`).join('')+'</ol>':'<p class="muted">Not placed in the hierarchy.</p>');
 h+='<h4>Upstream</h4>'+list(d.up)+'<h4>Downstream</h4>'+list(d.down)+'<h4>Connected, direction not confirmed</h4>'+list(d.links);
 const p=document.getElementById('panel');p.innerHTML=h;p.querySelectorAll('.lnk').forEach(x=>x.onclick=()=>{const t=x.dataset.id;if(byId[t]&&hidden.has(byId[t].k)){hidden.delete(byId[t].k);legend();draw()}pick(t);centre(t)})}
function centre(id){const n=byId[id];if(!n||hidden.has(n.k))return;const[x,y]=P(n);T.k=Math.max(T.k,2.5);T.x=W()/2-x*T.k;T.y=H()/2-y*T.k;apply()}
// pan + zoom
let drag=null;svg.addEventListener('mousedown',e=>{drag=[e.clientX-T.x,e.clientY-T.y]});window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{if(!drag)return;T.x=e.clientX-drag[0];T.y=e.clientY-drag[1];apply()});
svg.addEventListener('wheel',e=>{e.preventDefault();const r=svg.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.2:1/1.2;
 const k=Math.min(40,Math.max(0.3,T.k*f));T.x=mx-(mx-T.x)*k/T.k;T.y=my-(my-T.y)*k/T.k;T.k=k;apply()},{passive:false});
document.getElementById('fit').onclick=()=>{T={x:0,y:0,k:1};apply()};
document.getElementById('lay').onclick=e=>{mode=mode==='d'?'n':'d';e.target.textContent=mode==='d'?'Network layout':'Drawing layout';T={x:0,y:0,k:1};draw()};
const dl=document.getElementById('tags');[...new Set(D.nodes.map(n=>n.t))].sort().forEach(t=>{const o=document.createElement('option');o.value=t;dl.appendChild(o)});
document.getElementById('q').addEventListener('change',e=>{const v=e.target.value.trim().toUpperCase();const n=D.nodes.find(n=>n.t.toUpperCase()===v)||D.nodes.find(n=>n.t.toUpperCase().includes(v));
 if(n){if(hidden.has(n.k)){hidden.delete(n.k);legend();draw()}pick(n.id);centre(n.id)}});
function legend(){const kinds=[...new Set(D.nodes.map(n=>n.k))];document.getElementById('legend').innerHTML=kinds.map(k=>`<label><input type="checkbox" data-k="${k}" ${hidden.has(k)?'':'checked'}><span class="dot" style="background:${COL[k]||'#777'}"></span>${k}</label>`).join('');
 document.querySelectorAll('#legend input').forEach(c=>c.onchange=()=>{c.checked?hidden.delete(c.dataset.k):hidden.add(c.dataset.k);draw()})}
legend();draw();window.addEventListener('resize',draw);
</script></body></html>"""
