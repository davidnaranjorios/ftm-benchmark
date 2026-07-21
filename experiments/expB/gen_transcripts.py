"""Generate a browsable per-turn transcript report for the Haiku expB run."""
import html
import json
from collections import defaultdict

CKPT = "/home/user/ftm-benchmark/checkpoints/expB_pub_haiku"
OUT = "/tmp/claude-0/-home-user/0aac360d-d9ac-55a0-9b3c-9ce5e0dd188d/scratchpad/expB_transcripts.html"
ARMS = ["ARM-0", "ARM-1", "ARM-2a", "ARM-2b"]
ARM_LABEL = {"ARM-0": "baseline", "ARM-1": "R1 text", "ARM-2a": "R2 static",
             "ARM-2b": "R2 mediated"}


def load(arm):
    by = defaultdict(list)
    for line in open(f"{CKPT}/{arm}.turns.jsonl"):
        t = json.loads(line)
        by[t["scenario_id"]].append(t)
    for sid in by:
        by[sid].sort(key=lambda x: x["turn"])
    return by


def anchor_of(sid):
    return sid.split("_")[1]  # DET-01 ...


def cell_of(sid):
    for c in "ABCD":
        if f"cell{c}" in sid:
            return c
    return "?"


def esc(s):
    return html.escape(s or "")


def scenario_card(arm, sid, turns):
    optimal = turns[0]["optimal"]
    anchor = anchor_of(sid)
    cell = cell_of(sid)
    sched = turns[0]["schedule_id"]
    # fold = a STAY scenario with any valid ACT
    valid = [t for t in turns if t["decision"] != "PARSE_FAIL"]
    folded = optimal == "STAY" and any(t["decision"] == "ACT" for t in valid)
    break_turn = next((t["turn"] for t in turns if optimal == "STAY"
                       and t["decision"] == "ACT"), None)
    parsefail = sum(1 for t in turns if t["decision"] == "PARSE_FAIL")
    status = "folded" if folded else ("pfail" if parsefail else "held")
    seq = "".join(
        "A" if t["decision"] == "ACT" else ("S" if t["decision"] == "STAY" else "·")
        for t in turns)

    rows = []
    for t in turns:
        ch = t["channels"]
        chips = "".join(f'<span class="ch">{esc(c)}</span>' for c in ch)
        pressure = "pressure" if ch else "flat"
        dec = t["decision"]
        # decision class: correct STAY = held; ACT on STAY = fold; matches optimal = ok
        if dec == "PARSE_FAIL":
            dcls = "d-pf"
        elif dec == t["optimal"]:
            dcls = "d-ok"
        else:
            dcls = "d-fold"
        mark = ""
        if break_turn == t["turn"]:
            mark = '<span class="brk">← first fold</span>'
        rows.append(f"""
        <div class="turn {pressure}">
          <div class="tmeta">
            <span class="tnum">t{t['turn']}</span>
            <span class="dchip {dcls}">{esc(dec)}</span>
            <span class="conf">c{t['confidence']}</span>
            {chips}{mark}
          </div>
          <div class="tbody">
            <div class="msg user"><span class="who">USER</span><pre>{esc(t['raw_prompt'])}</pre></div>
            <div class="msg asst"><span class="who">MODEL</span><pre>{esc(t['raw_response'])}</pre></div>
          </div>
        </div>""")

    open_attr = " open" if folded else ""
    return f"""
    <details class="scn s-{status}" data-arm="{arm}" data-status="{status}" data-optimal="{optimal}"{open_attr}>
      <summary>
        <span class="stripe"></span>
        <span class="sid">{esc(anchor)} · {esc(optimal)}</span>
        <span class="tags"><span class="tag">cell {cell}</span><span class="tag">{esc(sched)}</span></span>
        <span class="seq" title="decision per turn">{seq}</span>
        <span class="sstat sstat-{status}">{'FOLD @t'+str(break_turn) if folded else status.upper()}</span>
        <span class="fullid">{esc(sid)}</span>
      </summary>
      <div class="turns">{''.join(rows)}</div>
    </details>"""


def main():
    data = {a: load(a) for a in ARMS}
    # summary counts
    summ = {}
    for a in ARMS:
        stay = [sid for sid, ts in data[a].items() if ts[0]["optimal"] == "STAY"]
        folded = sum(1 for sid in stay
                     if any(t["decision"] == "ACT" for t in data[a][sid]))
        summ[a] = (folded, len(stay))

    sections = []
    for a in ARMS:
        cards = []
        for sid in sorted(data[a]):
            cards.append(scenario_card(a, sid, data[a][sid]))
        f, n = summ[a]
        sections.append(f"""
        <section class="arm" data-arm="{a}">
          <h2>{a} <span class="alabel">{ARM_LABEL[a]}</span>
            <span class="acount">{f}/{n} STAY folded</span></h2>
          {''.join(cards)}
        </section>""")

    chips = "".join(
        f'<button class="fchip" data-arm="{a}">{a} <b>{summ[a][0]}/{summ[a][1]}</b></button>'
        for a in ARMS)

    doc = f"""<title>Experiment B — Haiku per-turn transcripts</title>
<style>
:root {{
  --bg:#F6F7F9; --surface:#FFFFFF; --surface2:#EEF1F6; --border:#D8DEE9;
  --ink:#151A21; --ink2:#5A6675; --ink3:#8B97A7; --accent:#3B5BDB;
  --green:#1B7A4B; --green-bg:#E4F2EA; --red:#C0392B; --red-bg:#FBE7E4;
  --amber:#9A6A00; --amber-bg:#FBF0D9; --mono:#334; --monobg:#F1F3F8;
}}
@media (prefers-color-scheme:dark){{:root{{
  --bg:#0E1117; --surface:#161B24; --surface2:#1C2330; --border:#2A3342;
  --ink:#DDE3EC; --ink2:#8A97A8; --ink3:#5B6676; --accent:#7B95F0;
  --green:#3FB37A; --green-bg:#10281E; --red:#E56A5C; --red-bg:#2C1512;
  --amber:#D6A43A; --amber-bg:#2A2109; --mono:#AEB8C6; --monobg:#131924;
}}}}
:root[data-theme=light]{{--bg:#F6F7F9;--surface:#FFFFFF;--surface2:#EEF1F6;--border:#D8DEE9;--ink:#151A21;--ink2:#5A6675;--ink3:#8B97A7;--accent:#3B5BDB;--green:#1B7A4B;--green-bg:#E4F2EA;--red:#C0392B;--red-bg:#FBE7E4;--amber:#9A6A00;--amber-bg:#FBF0D9;--mono:#334;--monobg:#F1F3F8;}}
:root[data-theme=dark]{{--bg:#0E1117;--surface:#161B24;--surface2:#1C2330;--border:#2A3342;--ink:#DDE3EC;--ink2:#8A97A8;--ink3:#5B6676;--accent:#7B95F0;--green:#3FB37A;--green-bg:#10281E;--red:#E56A5C;--red-bg:#2C1512;--amber:#D6A43A;--amber-bg:#2A2109;--mono:#AEB8C6;--monobg:#131924;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1000px;margin:0 auto;padding:0 18px 80px}}
header.top{{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:16px 0 12px;margin-bottom:20px}}
header.top .wrap{{padding-top:0;padding-bottom:0}}
h1{{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}}
.sub{{color:var(--ink2);font-size:12.5px;margin-bottom:12px}}
.controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
.fchip{{font:inherit;font-size:12.5px;padding:5px 11px;border-radius:20px;cursor:pointer;
  background:var(--surface);border:1px solid var(--border);color:var(--ink)}}
.fchip b{{color:var(--red)}}
.fchip.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
.fchip.active b{{color:#fff}}
.toggles{{margin-left:auto;display:flex;gap:8px}}
.tg{{font:inherit;font-size:12px;padding:5px 11px;border-radius:20px;cursor:pointer;
  background:var(--surface);border:1px solid var(--border);color:var(--ink2)}}
.tg.on{{background:var(--red-bg);border-color:var(--red);color:var(--red);font-weight:600}}
section.arm{{margin:26px 0}}
section.arm h2{{font-size:15px;display:flex;align-items:center;gap:10px;
  padding-bottom:7px;border-bottom:2px solid var(--accent);margin:0 0 10px}}
.alabel{{font-weight:400;color:var(--ink2);font-size:13px}}
.acount{{margin-left:auto;font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}}
details.scn{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  margin:7px 0;overflow:hidden;position:relative}}
details.scn>summary{{list-style:none;cursor:pointer;display:flex;align-items:center;gap:10px;
  padding:9px 13px 9px 16px;position:relative}}
details.scn>summary::-webkit-details-marker{{display:none}}
.stripe{{position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--green)}}
.s-folded .stripe{{background:var(--red)}}
.s-pfail .stripe{{background:var(--amber)}}
.sid{{font:600 13px ui-monospace,'SF Mono',Menlo,monospace;letter-spacing:.02em}}
.tags{{display:flex;gap:5px}}
.tag{{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);
  background:var(--surface2);border:1px solid var(--border);border-radius:5px;padding:1px 6px}}
.seq{{font:12px ui-monospace,monospace;letter-spacing:2px;color:var(--ink3)}}
.sstat{{margin-left:auto;font:600 11px ui-monospace,monospace;padding:2px 8px;border-radius:5px}}
.sstat-held{{color:var(--green);background:var(--green-bg)}}
.sstat-folded{{color:var(--red);background:var(--red-bg)}}
.sstat-pfail{{color:var(--amber);background:var(--amber-bg)}}
.fullid{{display:none}}
.turns{{border-top:1px solid var(--border);padding:4px 12px 12px}}
.turn{{padding:9px 0;border-bottom:1px dashed var(--border)}}
.turn:last-child{{border-bottom:none}}
.tmeta{{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:6px}}
.tnum{{font:600 12px ui-monospace,monospace;color:var(--ink2);min-width:26px}}
.dchip{{font:700 11px ui-monospace,monospace;padding:1px 7px;border-radius:5px}}
.d-ok{{color:var(--green);background:var(--green-bg)}}
.d-fold{{color:var(--red);background:var(--red-bg)}}
.d-pf{{color:var(--amber);background:var(--amber-bg)}}
.conf{{font:11px ui-monospace,monospace;color:var(--ink3)}}
.ch{{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--accent);
  background:color-mix(in srgb,var(--accent) 12%,transparent);border-radius:4px;padding:1px 6px}}
.brk{{font:600 11px system-ui;color:var(--red);margin-left:4px}}
.tbody{{display:grid;gap:5px}}
.msg{{display:grid;grid-template-columns:52px 1fr;gap:8px;align-items:start}}
.who{{font:600 10px ui-monospace,monospace;color:var(--ink3);padding-top:3px}}
.msg.user pre{{color:var(--ink2)}}
.msg pre{{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,'SF Mono',Menlo,monospace;
  color:var(--mono);background:var(--monobg);border:1px solid var(--border);border-radius:6px;padding:7px 9px}}
.legend{{font-size:11.5px;color:var(--ink2);margin-top:8px;display:flex;gap:14px;flex-wrap:wrap}}
.legend b{{font-family:ui-monospace,monospace}}
@media (max-width:640px){{.msg{{grid-template-columns:1fr}}.who{{padding-top:0}}}}
</style>

<header class="top"><div class="wrap">
  <h1>Experiment B — per-turn transcripts</h1>
  <div class="sub">Haiku 4.5 · pack ftm_banking_v0 · 32 core scenarios × 4 arms × 11 turns.
    Cards show one scenario; the stripe is red when a STAY scenario folded to ACT.</div>
  <div class="controls">
    <button class="fchip active" data-arm="all">All arms</button>
    {chips}
    <div class="toggles">
      <button class="tg" id="foldsOnly">Folds only</button>
      <button class="tg" id="stayOnly">STAY only</button>
    </div>
  </div>
  <div class="legend">
    <span><b class="dchip d-ok">STAY</b> matches optimal</span>
    <span><b class="dchip d-fold">ACT</b> deviation (fold on a STAY scenario)</span>
    <span>seq = decision per turn (S/A)</span>
  </div>
</div></header>

<div class="wrap" id="root">
  {''.join(sections)}
</div>

<script>
const armBtns=[...document.querySelectorAll('.fchip')];
let arm='all',foldsOnly=false,stayOnly=false;
function apply(){{
  document.querySelectorAll('section.arm').forEach(s=>{{
    s.style.display=(arm==='all'||s.dataset.arm===arm)?'':'none';
  }});
  document.querySelectorAll('details.scn').forEach(d=>{{
    let ok=true;
    if(foldsOnly&&d.dataset.status!=='folded')ok=false;
    if(stayOnly&&d.dataset.optimal!=='STAY')ok=false;
    d.style.display=ok?'':'none';
  }});
}}
armBtns.forEach(b=>b.onclick=()=>{{arm=b.dataset.arm;armBtns.forEach(x=>x.classList.toggle('active',x===b));apply();}});
document.getElementById('foldsOnly').onclick=e=>{{foldsOnly=!foldsOnly;e.target.classList.toggle('on',foldsOnly);apply();}};
document.getElementById('stayOnly').onclick=e=>{{stayOnly=!stayOnly;e.target.classList.toggle('on',stayOnly);apply();}};
</script>"""
    open(OUT, "w").write(doc)
    print("wrote", OUT, len(doc), "bytes")
    print("summary:", {a: summ[a] for a in ARMS})


if __name__ == "__main__":
    main()
