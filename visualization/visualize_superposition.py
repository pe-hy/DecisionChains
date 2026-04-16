"""
Generate an interactive HTML visualization of letter probability distributions.

Supports both 2-step chains (original) and N-step extended chains (3-5 steps).
For each example, shows bar charts at each decision point, with correct
letters (from decision functions f, g) highlighted in green/blue.
A checkbox filters to only examples where the model chose an unseen tuple.

Called from inference scripts or standalone:
    python visualize_superposition.py --npz_path results/decision_point_probs.npz
"""

import os
import json
import argparse
import numpy as np

# ── Decision functions (same as inference scripts) ──

LETTERS = [chr(ord("a") + i) for i in range(20)]


def is_even(x):
    return 1 if x % 2 == 0 else 0


def decision_func_f(vec):
    return is_even(vec[0]) * 10 + vec[1]


def decision_func_g(vec):
    return is_even(vec[3]) * 10 + vec[4]


# ── HTML generation ──

def build_html(examples_data):
    """Build the full HTML string from a list of example dicts.

    Each example dict has:
        idx: int
        vec: list[int]
        n_steps: int
        steps: list of {probs: list[float], idx_f: int, idx_g: int,
                         gen_letter: str|None, label: str}
        gen_letters: list[str|None]
        chosen_in_train: bool|None
        tuple_str: str  (e.g. "(a,b,c)")
    """
    data_json = json.dumps(examples_data)
    letters_json = json.dumps(LETTERS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Superposition — Letter Probability Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #222; padding: 20px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 12px; }}
  .controls {{ background: #fff; padding: 14px 18px; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
               display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  .controls label {{ font-weight: 600; font-size: 0.9rem; }}
  .controls select, .controls input {{ font-size: 0.9rem; padding: 4px 8px; }}
  #numVisible {{ width: 60px; }}
  .example-card {{ background: #fff; border-radius: 8px; padding: 16px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 14px; }}
  .example-card h3 {{ font-size: 0.95rem; margin-bottom: 10px; }}
  .example-card h3 span.tag {{ font-size: 0.78rem; padding: 2px 6px; border-radius: 4px;
                                margin-left: 6px; font-weight: 500; }}
  .tag-f {{ background: #d4edda; color: #155724; }}
  .tag-g {{ background: #cce5ff; color: #004085; }}
  .tag-unseen {{ background: #fff3e0; color: #e65100; border: 1px solid #ff9800; }}
  .tag-seen {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #4caf50; }}
  .charts {{ display: flex; gap: 14px; overflow-x: auto; padding-bottom: 6px; }}
  .chart-box {{ flex: 0 0 500px; min-width: 500px; }}
  .chart-box h4 {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}
  canvas {{ width: 100% !important; height: 260px !important; }}
  .legend {{ font-size: 0.78rem; color: #555; margin-top: 4px; }}
  .legend span {{ display: inline-block; width: 12px; height: 12px;
                  border-radius: 2px; vertical-align: middle; margin-right: 3px; }}
  #statsPanel {{ display: none; background: #fff; border-radius: 8px; padding: 18px 22px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
                 font-size: 0.88rem; line-height: 1.6; }}
  #statsPanel.visible {{ display: block; }}
  #statsPanel h2 {{ font-size: 1.05rem; margin-bottom: 10px; }}
  #statsPanel h3 {{ font-size: 0.92rem; margin: 14px 0 6px; color: #333;
                    border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  #statsPanel table {{ border-collapse: collapse; margin: 6px 0; }}
  #statsPanel th, #statsPanel td {{ text-align: left; padding: 3px 14px 3px 0;
                                    font-size: 0.85rem; }}
  #statsPanel th {{ color: #666; font-weight: 600; }}
  #statsPanel td.num {{ font-family: "SF Mono", Consolas, monospace; text-align: right; }}
  .stats-btn {{ background: #1976d2; color: #fff; border: none; border-radius: 5px;
                padding: 5px 14px; cursor: pointer; font-size: 0.88rem; font-weight: 500; }}
  .stats-btn:hover {{ background: #1565c0; }}
  .stats-btn.active {{ background: #0d47a1; }}
</style>
</head>
<body>
<h1>Letter probability distributions (a-t)</h1>

<div class="controls">
  <label for="exampleSelect">Example:</label>
  <select id="exampleSelect"></select>

  <label for="numVisible">Show N:</label>
  <input type="number" id="numVisible" value="3" min="1" max="50">

  <label><input type="checkbox" id="logScale"> Log scale</label>
  <label><input type="checkbox" id="unseenOnly"> Unseen tuples only</label>

  <button id="btnPrev">&#9664; Prev</button>
  <button id="btnNext">Next &#9654;</button>
  <button class="stats-btn" id="btnStats">Statistics</button>
  <span id="rangeInfo" style="font-size:0.85rem; color:#666;"></span>
</div>

<div id="statsPanel"></div>
<div id="cardsContainer"></div>

<div class="legend" style="margin-top:8px;">
  <span style="background:#4caf50;"></span> Correct letter (from f)&nbsp;&nbsp;
  <span style="background:#2196f3;"></span> Correct letter (from g)&nbsp;&nbsp;
  <span style="background:#00bcd4;"></span> Correct letter (both f&amp;g)&nbsp;&nbsp;
  <span style="background:#ccc;"></span> Other letter&nbsp;&nbsp;
  <span style="background:transparent; border:2px solid #f44336; width:10px; height:10px;"></span> Model's choice
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<script>
Chart.register(ChartDataLabels);

const DATA = {data_json};
const LETTERS = {letters_json};

let filteredIndices = [];
let cursorPos = 0;
let activeCharts = [];

const sel = document.getElementById("exampleSelect");

function isLogScale() {{
  return document.getElementById("logScale").checked;
}}
function isUnseenOnly() {{
  return document.getElementById("unseenOnly").checked;
}}
function getNumVisible() {{
  return Math.max(1, Math.min(50, parseInt(document.getElementById("numVisible").value) || 3));
}}
function destroyCharts() {{
  activeCharts.forEach(c => c.destroy());
  activeCharts = [];
}}

function rebuildDropdown() {{
  const unseen = isUnseenOnly();
  filteredIndices = [];
  sel.innerHTML = "";
  DATA.forEach((ex, i) => {{
    if (unseen && ex.chosen_in_train !== false) return;
    filteredIndices.push(i);
    const opt = document.createElement("option");
    opt.value = filteredIndices.length - 1;
    const tupleLabel = ex.tuple_str
        ? "  " + ex.tuple_str
            + (ex.chosen_in_train === false ? " UNSEEN" : " TRAIN")
        : "";
    opt.textContent = "#" + i + "  vec=[" + ex.vec.join(",") + "]"
        + "  " + ex.n_steps + "-step" + tupleLabel;
    sel.appendChild(opt);
  }});
  cursorPos = 0;
  sel.value = 0;
}}

function render() {{
  destroyCharts();
  const useLog = isLogScale();
  const nv = getNumVisible();
  const total = filteredIndices.length;
  const end = Math.min(cursorPos + nv, total);
  document.getElementById("rangeInfo").textContent =
      total === 0 ? "No matching examples"
      : "Showing " + (cursorPos + 1) + "-" + end + " of " + total
        + (isUnseenOnly() ? " (unseen tuples)" : "");

  const container = document.getElementById("cardsContainer");
  container.innerHTML = "";

  for (let fi = cursorPos; fi < end; fi++) {{
    const dataIdx = filteredIndices[fi];
    const ex = DATA[dataIdx];
    const card = document.createElement("div");
    card.className = "example-card";

    let title = "<h3>#" + dataIdx + " &nbsp; vec = [" + ex.vec.join(", ") + "]"
        + " &nbsp; " + ex.n_steps + "-step";
    if (ex.tuple_str) {{
        if (ex.chosen_in_train === false) {{
            title += '<span class="tag tag-unseen">chose ' + ex.tuple_str + ' UNSEEN</span>';
        }} else if (ex.chosen_in_train === true) {{
            title += '<span class="tag tag-seen">chose ' + ex.tuple_str + ' TRAIN</span>';
        }} else {{
            title += '<span class="tag" style="background:#eee;">chose ' + ex.tuple_str + '</span>';
        }}
    }}
    title += "</h3>";
    card.innerHTML = title;

    const chartsDiv = document.createElement("div");
    chartsDiv.className = "charts";

    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (step.probs) {{
        chartsDiv.appendChild(makeChartBox(
            step.probs, step.label,
            step.idx_f, step.idx_g,
            step.gen_letter,
            "c_" + fi + "_" + s, useLog));
      }} else {{
        const noData = document.createElement("div");
        noData.className = "chart-box";
        noData.innerHTML = "<h4>" + step.label + "</h4>"
            + "<p style='color:#999;font-size:0.85rem;'>No data</p>";
        chartsDiv.appendChild(noData);
      }}
    }}

    card.appendChild(chartsDiv);
    container.appendChild(card);
  }}
}}

function makeChartBox(probs, title, idxF, idxG, genLetter, canvasId, useLog) {{
  const box = document.createElement("div");
  box.className = "chart-box";
  box.innerHTML = "<h4>" + title + "</h4>";
  const canvas = document.createElement("canvas");
  canvas.id = canvasId;
  box.appendChild(canvas);

  const colors = probs.map((_, j) => {{
    const isF = (idxF !== null && idxF !== undefined && j === idxF);
    const isG = (idxG !== null && idxG !== undefined && j === idxG);
    if (isF && isG) return "#00bcd4";
    if (isF) return "#4caf50";
    if (isG) return "#2196f3";
    return "#ccc";
  }});

  const borders = probs.map((_, j) => {{
    return (LETTERS[j] === genLetter) ? "#f44336" : "transparent";
  }});
  const borderWidths = probs.map((_, j) => {{
    return (LETTERS[j] === genLetter) ? 3 : 0;
  }});

  const displayData = useLog
      ? probs.map(p => p > 0 ? Math.log10(p) : -6)
      : probs;

  const yConfig = useLog
      ? {{ title: {{ display: true, text: "log10 P" }},
           grid: {{ color: "#eee" }}, min: -6, max: 0 }}
      : {{ beginAtZero: true,
           title: {{ display: true, text: "P(letter)" }},
           grid: {{ color: "#eee" }} }};

  setTimeout(() => {{
    const chart = new Chart(canvas, {{
      type: "bar",
      data: {{
        labels: LETTERS,
        datasets: [{{ data: displayData, backgroundColor: colors,
                      borderColor: borders, borderWidth: borderWidths }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        layout: {{ padding: {{ top: 20 }} }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: (ctx) => {{
                const j = ctx.dataIndex;
                const p = probs[j];
                let label = LETTERS[j] + ": " + p.toFixed(4);
                if (idxF !== null && idxF !== undefined && j === idxF) label += " [f]";
                if (idxG !== null && idxG !== undefined && j === idxG) label += " [g]";
                return label;
              }}
            }}
          }},
          datalabels: {{
            anchor: "end",
            align: "top",
            font: {{ size: 9 }},
            color: "#444",
            display: (ctx) => {{
              const p = probs[ctx.dataIndex];
              return p >= 0.005;
            }},
            formatter: (val, ctx) => {{
              const p = probs[ctx.dataIndex];
              if (p >= 0.01) return p.toFixed(2);
              return p.toFixed(3);
            }},
          }},
        }},
        scales: {{
          x: {{ grid: {{ display: false }} }},
          y: yConfig,
        }},
      }},
    }});
    activeCharts.push(chart);
  }}, 0);

  return box;
}}

// Navigation
document.getElementById("btnPrev").onclick = () => {{
  cursorPos = Math.max(0, cursorPos - getNumVisible());
  sel.value = cursorPos;
  render();
}};
document.getElementById("btnNext").onclick = () => {{
  cursorPos = Math.min(Math.max(0, filteredIndices.length - 1), cursorPos + getNumVisible());
  sel.value = cursorPos;
  render();
}};
sel.onchange = () => {{
  cursorPos = parseInt(sel.value) || 0;
  render();
}};
document.getElementById("numVisible").onchange = render;
document.getElementById("logScale").onchange = render;
document.getElementById("unseenOnly").onchange = () => {{
  rebuildDropdown();
  render();
}};

// Statistics
function computeStats() {{
  const N = DATA.length;
  if (N === 0) return "<p>No data.</p>";

  // Group by chain length
  const byLen = {{}};
  DATA.forEach(ex => {{
    const k = ex.n_steps;
    if (!byLen[k]) byLen[k] = [];
    byLen[k].push(ex);
  }});
  const lengths = Object.keys(byLen).map(Number).sort((a,b) => a - b);

  // Tuple origin
  let trainCount = 0, unseenCount = 0, unknownCount = 0;
  const trainPerLen = {{}}, unseenPerLen = {{}};
  lengths.forEach(l => {{ trainPerLen[l] = 0; unseenPerLen[l] = 0; }});
  DATA.forEach(ex => {{
    if (ex.chosen_in_train === true) {{ trainCount++; trainPerLen[ex.n_steps]++; }}
    else if (ex.chosen_in_train === false) {{ unseenCount++; unseenPerLen[ex.n_steps]++; }}
    else unknownCount++;
  }});

  // Per-step probability stats
  const maxSteps = Math.max(...lengths);
  const probF = {{}}, probG = {{}}, mass = {{}};
  for (let s = 0; s < maxSteps; s++) {{ probF[s] = []; probG[s] = []; mass[s] = []; }}
  DATA.forEach(ex => {{
    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (!step.probs || step.idx_f === null || step.idx_f === undefined) continue;
      const pf = step.probs[step.idx_f];
      const pg = (step.idx_g !== null && step.idx_g !== undefined) ? step.probs[step.idx_g] : 0;
      probF[s].push(pf);
      probG[s].push(pg);
      const m = (step.idx_f === step.idx_g) ? pf : pf + pg;
      mass[s].push(m);
    }}
  }});

  // Per-step: model chose f-letter, g-letter, or other
  const choiceF = {{}}, choiceG = {{}}, choiceOther = {{}};
  for (let s = 0; s < maxSteps; s++) {{ choiceF[s] = 0; choiceG[s] = 0; choiceOther[s] = 0; }}
  DATA.forEach(ex => {{
    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (!step.gen_letter || step.idx_f === null || step.idx_f === undefined) continue;
      const gi = LETTERS.indexOf(step.gen_letter);
      if (gi === step.idx_f || gi === step.idx_g) {{
        if (gi === step.idx_f) choiceF[s]++;
        if (gi === step.idx_g) choiceG[s]++;
      }} else {{
        choiceOther[s]++;
      }}
    }}
  }});

  // Top tuples per length
  const tupleCounts = {{}};
  lengths.forEach(l => tupleCounts[l] = {{}});
  DATA.forEach(ex => {{
    if (!ex.tuple_str) return;
    const c = tupleCounts[ex.n_steps];
    c[ex.tuple_str] = (c[ex.tuple_str] || 0) + 1;
  }});

  const mean = arr => arr.length ? (arr.reduce((a,b) => a+b, 0) / arr.length) : NaN;
  const pct = (a, b) => b > 0 ? (100 * a / b).toFixed(1) + "%" : "—";
  const fmt = v => isNaN(v) ? "—" : v.toFixed(4);

  // Build HTML
  let h = "<h2>Dataset Statistics</h2>";

  // Overview
  h += "<h3>Overview</h3><table>";
  h += "<tr><th>Total examples</th><td class='num'>" + N + "</td></tr>";
  lengths.forEach(l => {{
    h += "<tr><th>" + l + "-step</th><td class='num'>" + byLen[l].length
      + " (" + pct(byLen[l].length, N) + ")</td></tr>";
  }});
  h += "</table>";

  // Tuple origin
  h += "<h3>Tuple Origin</h3><table>";
  h += "<tr><th></th><th>Train</th><th>Unseen</th><th>Unknown</th></tr>";
  h += "<tr><th>Overall</th><td class='num'>" + trainCount + " (" + pct(trainCount, N) + ")</td>"
    + "<td class='num'>" + unseenCount + " (" + pct(unseenCount, N) + ")</td>"
    + "<td class='num'>" + unknownCount + "</td></tr>";
  lengths.forEach(l => {{
    const tot = byLen[l].length;
    h += "<tr><th>" + l + "-step</th><td class='num'>" + trainPerLen[l]
      + " (" + pct(trainPerLen[l], tot) + ")</td>"
      + "<td class='num'>" + unseenPerLen[l]
      + " (" + pct(unseenPerLen[l], tot) + ")</td><td></td></tr>";
  }});
  h += "</table>";

  // Per-step probabilities
  h += "<h3>Per-Step Decision Probabilities</h3><table>";
  h += "<tr><th>Step</th><th>Mean P(f)</th><th>Mean P(g)</th><th>Mean mass</th><th>N</th></tr>";
  let allPf = [], allPg = [];
  for (let s = 0; s < maxSteps; s++) {{
    if (probF[s].length === 0) continue;
    allPf = allPf.concat(probF[s]);
    allPg = allPg.concat(probG[s]);
    h += "<tr><td>Step " + (s+1) + "</td>"
      + "<td class='num'>" + fmt(mean(probF[s])) + "</td>"
      + "<td class='num'>" + fmt(mean(probG[s])) + "</td>"
      + "<td class='num'>" + fmt(mean(mass[s])) + "</td>"
      + "<td class='num'>" + probF[s].length + "</td></tr>";
  }}
  if (allPf.length > 0) {{
    h += "<tr style='border-top:1px solid #ccc;'><th>Grand avg</th>"
      + "<td class='num'><b>" + fmt(mean(allPf)) + "</b></td>"
      + "<td class='num'><b>" + fmt(mean(allPg)) + "</b></td>"
      + "<td></td><td></td></tr>";
  }}
  h += "</table>";

  // Per-step model choice
  h += "<h3>Model's Letter Choice</h3><table>";
  h += "<tr><th>Step</th><th>Chose f-letter</th><th>Chose g-letter</th><th>Chose other</th></tr>";
  for (let s = 0; s < maxSteps; s++) {{
    const tot = choiceF[s] + choiceG[s] + choiceOther[s];
    if (tot === 0) continue;
    h += "<tr><td>Step " + (s+1) + "</td>"
      + "<td class='num'>" + choiceF[s] + " (" + pct(choiceF[s], tot) + ")</td>"
      + "<td class='num'>" + choiceG[s] + " (" + pct(choiceG[s], tot) + ")</td>"
      + "<td class='num'>" + choiceOther[s] + " (" + pct(choiceOther[s], tot) + ")</td></tr>";
  }}
  h += "</table>";

  // Top tuples per length
  h += "<h3>Top Tuples</h3>";
  lengths.forEach(l => {{
    const counts = tupleCounts[l];
    const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]).slice(0, 10);
    if (sorted.length === 0) return;
    const tot = byLen[l].length;
    h += "<b>" + l + "-step</b> (top 10):<table>";
    h += "<tr><th>Tuple</th><th>Count</th><th>%</th><th>Origin</th></tr>";
    sorted.forEach(([tup, cnt]) => {{
      const origin = DATA.find(ex => ex.tuple_str === tup && ex.n_steps === l);
      const tag = origin
        ? (origin.chosen_in_train === true ? "TRAIN" : origin.chosen_in_train === false ? "UNSEEN" : "?")
        : "?";
      h += "<tr><td>" + tup + "</td><td class='num'>" + cnt + "</td>"
        + "<td class='num'>" + pct(cnt, tot) + "</td>"
        + "<td>" + tag + "</td></tr>";
    }});
    h += "</table>";
  }});

  return h;
}}

document.getElementById("btnStats").onclick = () => {{
  const panel = document.getElementById("statsPanel");
  const btn = document.getElementById("btnStats");
  if (panel.classList.contains("visible")) {{
    panel.classList.remove("visible");
    btn.classList.remove("active");
  }} else {{
    if (!panel.innerHTML) panel.innerHTML = computeStats();
    panel.classList.add("visible");
    btn.classList.add("active");
  }}
}};

rebuildDropdown();
render();
</script>
</body>
</html>"""
    return html


def build_html_with_entropy(examples_data, token_level_data, unique_inputs, predictions, metrics=None,
                            per_step_idx_f=None, per_step_idx_g=None,
                            is_valid_chain=None, is_valid_solution=None, has_different_final=None,
                            is_entropy_monotone=None, entropy_violation_count=None, per_step_entropy=None,
                            mono_data_all=None):
    """Build HTML with both bar chart view and entropy token view.

    Data is written to companion .js files next to save_path to keep HTML small.
    mono_data_all: optional list of dicts with per_step_entropy/is_entropy_monotone/chain_len
                   for ALL examples (not limited by html_samples) — used for the mono view.
    """
    # Inject entropy monotonicity info into bar chart examples data
    if is_entropy_monotone is not None:
        for i, ex in enumerate(examples_data):
            if i < len(is_entropy_monotone):
                ex["is_entropy_monotone"] = is_entropy_monotone[i]
            if entropy_violation_count and i < len(entropy_violation_count):
                ex["entropy_violation_count"] = entropy_violation_count[i]
            if per_step_entropy and i < len(per_step_entropy):
                ex["per_step_entropy"] = per_step_entropy[i]

    # Prepare entropy data
    entropy_samples = []
    for i, (input_str, chain_len, input_vec) in enumerate(unique_inputs or []):
        tokens = token_level_data[i] if token_level_data and i < len(token_level_data) else []
        pred = predictions[i] if predictions and i < len(predictions) else ""
        idx_f_list = per_step_idx_f[i] if per_step_idx_f and i < len(per_step_idx_f) else []
        idx_g_list = per_step_idx_g[i] if per_step_idx_g and i < len(per_step_idx_g) else []
        v_chain = is_valid_chain[i] if is_valid_chain and i < len(is_valid_chain) else False
        v_solution = is_valid_solution[i] if is_valid_solution and i < len(is_valid_solution) else False
        has_diff_final = has_different_final[i] if has_different_final and i < len(has_different_final) else False
        ent_mono = is_entropy_monotone[i] if is_entropy_monotone and i < len(is_entropy_monotone) else True
        ent_violations = entropy_violation_count[i] if entropy_violation_count and i < len(entropy_violation_count) else 0
        step_ent = per_step_entropy[i] if per_step_entropy and i < len(per_step_entropy) else []
        entropy_samples.append({
            "input_str": input_str,
            "chain_len": chain_len,
            "prediction": pred,
            "tokens": tokens,
            "idx_f_per_step": idx_f_list,
            "idx_g_per_step": idx_g_list,
            "is_valid_chain": v_chain,
            "is_valid_solution": v_solution,
            "has_different_final": has_diff_final,
            "is_entropy_monotone": ent_mono,
            "entropy_violation_count": ent_violations,
            "per_step_entropy": step_ent,
        })

    # Serialize data inline (file:// protocol blocks external script loading)
    data_json = json.dumps(examples_data)
    letters_json = json.dumps(LETTERS)
    metrics_json = json.dumps(metrics or {})
    entropy_json = json.dumps(entropy_samples)

    # MONO_DATA: lightweight, ALL examples (not html-limited) — tiny payload
    mono_payload = mono_data_all if mono_data_all else [
        {"per_step_entropy": s.get("per_step_entropy", []),
         "is_entropy_monotone": s.get("is_entropy_monotone", True),
         "chain_len": s.get("chain_len", 0),
         "entropy_violation_count": s.get("entropy_violation_count", 0)}
        for s in entropy_samples
    ]
    mono_json = json.dumps(mono_payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Superposition — Letter Probability & Entropy Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #222; padding: 20px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 12px; color: #333; }}

  /* View Toggle */
  .view-toggle {{ display: flex; gap: 10px; margin-bottom: 16px; }}
  .view-btn {{ padding: 10px 20px; font-size: 0.95rem; background: #fff; color: #1976d2;
               border: 2px solid #1976d2; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
  .view-btn:hover {{ background: #1976d2; color: #fff; }}
  .view-btn.active {{ background: #1976d2; color: #fff; font-weight: bold; }}

  .view {{ display: none; }}
  .view.active {{ display: block; }}

  /* Bar chart view styles (light theme preserved) */
  #barChartView {{ background: #f5f5f5; color: #222; padding: 20px; border-radius: 10px; }}
  #barChartView .controls {{ background: #fff; padding: 14px 18px; border-radius: 8px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
               display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  #barChartView .controls label {{ font-weight: 600; font-size: 0.9rem; }}
  #barChartView .controls select, #barChartView .controls input {{ font-size: 0.9rem; padding: 4px 8px; }}
  #barChartView .controls select {{ width: auto; max-width: 80vw; font-family: "SF Mono", Consolas, monospace; font-size: 0.82rem; }}
  #barChartView #numVisible {{ width: 60px; }}
  #barChartView .example-card {{ background: #fff; border-radius: 8px; padding: 16px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 14px; }}
  #barChartView .example-card h3 {{ font-size: 0.95rem; margin-bottom: 10px; color: #222; }}
  #barChartView .example-card h3 span.tag {{ font-size: 0.78rem; padding: 2px 6px; border-radius: 4px;
                                margin-left: 6px; font-weight: 500; }}
  .tag-f {{ background: #d4edda; color: #155724; }}
  .tag-g {{ background: #cce5ff; color: #004085; }}
  .tag-unseen {{ background: #fff3e0; color: #e65100; border: 1px solid #ff9800; }}
  .tag-seen {{ background: #e8f5e9; color: #2e7d32; border: 1px solid #4caf50; }}
  .charts {{ display: flex; gap: 14px; overflow-x: auto; padding-bottom: 6px; }}
  .chart-box {{ flex: 0 0 500px; min-width: 500px; }}
  .chart-box h4 {{ font-size: 0.82rem; color: #666; margin-bottom: 6px; }}
  canvas {{ width: 100% !important; height: 260px !important; }}
  #barChartView .legend {{ font-size: 0.78rem; color: #555; margin-top: 4px; }}
  #barChartView .legend span {{ display: inline-block; width: 12px; height: 12px;
                  border-radius: 2px; vertical-align: middle; margin-right: 3px; }}
  #barChartView #statsPanel {{ display: none; background: #fff; border-radius: 8px; padding: 18px 22px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
                 font-size: 0.88rem; line-height: 1.6; color: #222; }}
  #barChartView #statsPanel.visible {{ display: block; }}
  #barChartView #statsPanel h2 {{ font-size: 1.05rem; margin-bottom: 10px; }}
  #barChartView #statsPanel h3 {{ font-size: 0.92rem; margin: 14px 0 6px; color: #333;
                    border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  #barChartView #statsPanel table {{ border-collapse: collapse; margin: 6px 0; }}
  #barChartView #statsPanel th, #barChartView #statsPanel td {{ text-align: left; padding: 3px 14px 3px 0;
                                    font-size: 0.85rem; }}
  #barChartView #statsPanel th {{ color: #666; font-weight: 600; }}
  #barChartView #statsPanel td.num {{ font-family: "SF Mono", Consolas, monospace; text-align: right; }}
  .stats-btn {{ background: #1976d2; color: #fff; border: none; border-radius: 5px;
                padding: 5px 14px; cursor: pointer; font-size: 0.88rem; font-weight: 500; }}
  .stats-btn:hover {{ background: #1565c0; }}
  .stats-btn.active {{ background: #0d47a1; }}

  /* Entropy view styles (light theme) */
  #entropyView {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .entropy-controls {{ background: #f8f9fa; padding: 14px 18px; border-radius: 8px;
               margin-bottom: 16px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
               border: 1px solid #e0e0e0; }}
  .entropy-controls label {{ font-weight: 600; font-size: 0.9rem; color: #333; }}
  .entropy-selector {{ padding: 8px 12px; font-size: 0.9rem; background: #fff; color: #333;
                       border: 1px solid #1976d2; border-radius: 6px; min-width: 300px;
                       width: auto; max-width: 80vw; font-family: "SF Mono", Consolas, monospace; font-size: 0.82rem; }}
  .entropy-nav-btn {{ padding: 8px 16px; font-size: 0.9rem; background: #fff; color: #1976d2;
                      border: 1px solid #1976d2; border-radius: 6px; cursor: pointer; }}
  .entropy-nav-btn:hover {{ background: #1976d2; color: #fff; }}

  .entropy-legend {{ display: flex; align-items: center; gap: 15px; margin: 15px 0; padding: 12px;
                     background: #f8f9fa; border-radius: 8px; flex-wrap: wrap; font-size: 0.85rem;
                     border: 1px solid #e0e0e0; color: #333; }}
  .entropy-legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .entropy-legend-color {{ width: 30px; height: 14px; border-radius: 3px; }}

  .token-container {{ display: flex; flex-wrap: wrap; gap: 4px; line-height: 3.2em; padding: 12px;
                      background: #f8f9fa; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e0e0e0; }}
  .token-wrapper {{ display: inline-flex; flex-direction: column; align-items: center;
                    position: relative; cursor: pointer; transition: transform 0.1s; }}
  .token-wrapper:hover {{ transform: translateY(-2px); }}
  .token-wrapper.selected {{ outline: 2px solid #1976d2; outline-offset: 2px; }}
  .token-wrapper.decision-point .entropy-bar {{ box-shadow: 0 0 4px rgba(255, 152, 0, 0.8); }}
  .entropy-bar {{ width: 100%; height: 12px; border-radius: 3px 3px 0 0; margin-bottom: 2px;
                  min-width: 16px; }}
  .p-stop-bar {{ width: 100%; height: 4px; min-width: 16px; margin-top: 1px; border-radius: 0 0 2px 2px; }}
  .token {{ padding: 3px 5px; background: #fff; border-radius: 0 0 3px 3px;
            font-family: "SF Mono", Consolas, monospace; font-size: 13px; white-space: pre;
            border: 1px solid #ddd; text-align: center; min-width: 16px; max-width: 120px;
            overflow: hidden; text-overflow: ellipsis; color: #333; }}

  .tooltip {{ display: none; position: fixed; background: #fff; border: 1px solid #1976d2;
              border-radius: 8px; padding: 10px; min-width: 180px; max-width: 220px; z-index: 10000;
              box-shadow: 0 4px 15px rgba(0,0,0,0.15); font-size: 0.85em;
              pointer-events: none; }}
  .tooltip-header {{ font-weight: bold; color: #1976d2; margin-bottom: 8px; }}
  .tooltip-row {{ display: flex; justify-content: space-between; margin: 4px 0; }}
  .tooltip-label {{ color: #666; }}
  .tooltip-value {{ color: #333; font-family: monospace; }}
  .alternatives {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }}
  .alternatives-title {{ color: #666; font-size: 0.85em; margin-bottom: 5px; }}
  .alt-token {{ display: flex; justify-content: space-between; font-size: 0.85em; padding: 2px 0; }}
  .alt-token-str {{ font-family: monospace; color: #555; }}
  .alt-prob {{ color: #1976d2; }}

  /* Sidebar */
  .sidebar {{ position: fixed; top: 0; right: -420px; width: 420px; height: 100vh;
              background: #fff; border-left: 2px solid #1976d2;
              box-shadow: -5px 0 20px rgba(0,0,0,0.15); transition: right 0.3s ease;
              z-index: 1000; overflow-y: auto; padding: 20px; }}
  .sidebar.open {{ right: 0; }}
  .sidebar-close {{ position: absolute; top: 10px; right: 10px; background: #f8f9fa;
                    border: 1px solid #1976d2; color: #1976d2; font-size: 22px; width: 32px;
                    height: 32px; border-radius: 50%; cursor: pointer; display: flex;
                    align-items: center; justify-content: center; }}
  .sidebar-close:hover {{ background: #1976d2; color: #fff; }}
  .sidebar-header {{ font-size: 1.1em; font-weight: bold; color: #1976d2; margin-bottom: 18px;
                     padding-right: 40px; }}
  .sidebar-section {{ margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #eee; }}
  .sidebar-section:last-child {{ border-bottom: none; }}
  .sidebar-section-title {{ color: #1976d2; font-weight: bold; margin-bottom: 10px; font-size: 0.95em; }}

  .example-info {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e0e0e0; }}
  .example-info-label {{ color: #666; font-size: 0.85em; margin-bottom: 3px; }}
  .example-info-value {{ font-family: monospace; font-size: 0.9em; word-break: break-all; color: #333; }}

  /* Summary view styles */
  #summaryView {{ background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 25px; }}
  .summary-card {{ background: #f8f9fa; border-radius: 10px; padding: 20px; border: 1px solid #e0e0e0; }}
  .summary-card h3 {{ color: #1976d2; font-size: 1rem; margin-bottom: 15px; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px; }}
  .metric-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e0e0e0; position: relative; }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ color: #666; font-size: 0.9em; cursor: help; border-bottom: 1px dotted #999; }}
  .metric-label[title]:hover {{ color: #1976d2; }}
  .metric-value {{ color: #1976d2; font-weight: bold; font-family: monospace; }}
  .metric-highlight {{ background: #4caf50; color: #fff; padding: 2px 8px; border-radius: 4px; }}
  .metric-warning {{ background: #ff9800; color: #fff; padding: 2px 8px; border-radius: 4px; }}

  /* Invalid letter marker (not from f/g at decision point) */
  .token.invalid-letter {{ position: relative; }}
  .token.invalid-letter::after {{ content: "●"; position: absolute; top: -2px; right: 2px;
                                  font-size: 8px; color: #f44336; font-weight: bold; }}

  /* Tag for non-monotone entropy */
  .tag-ent-up {{ background: #fce4ec; color: #c62828; border: 1px solid #ef9a9a; }}

  /* Rich metric tooltips */
  .metric-label {{ position: relative; }}
  .metric-tooltip {{ display: none; position: fixed; background: #fff; border: 1px solid #1976d2;
                     border-radius: 8px; padding: 12px 14px; width: 400px; max-width: calc(100vw - 32px);
                     z-index: 10000; box-shadow: 0 4px 20px rgba(0,0,0,0.18); font-size: 0.88em;
                     line-height: 1.5; color: #333; pointer-events: none; }}
  .metric-tooltip .katex {{ font-size: 1.05em; }}
  .metric-tooltip p {{ margin: 0; }}
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
</head>
<body>
<div id="globalTooltip" class="tooltip"></div>
<div id="globalMetricTooltip" class="metric-tooltip"></div>
<h1>Superposition — Letter Probability & Entropy Viewer</h1>

<!-- View Toggle -->
<div class="view-toggle">
  <button class="view-btn" onclick="switchView('summary')">Summary</button>
  <button class="view-btn active" onclick="switchView('barChart')">Bar Chart View</button>
  <button class="view-btn" onclick="switchView('entropy')">Entropy Token View</button>
  <button class="view-btn" onclick="switchView('mono')">Entropy Monotonicity</button>
</div>

<!-- Entropy Token Sidebar -->
<div id="token-sidebar" class="sidebar">
  <div class="sidebar-close" onclick="closeSidebar()">&times;</div>
  <div id="sidebar-content"></div>
</div>

<!-- Summary View -->
<div id="summaryView" class="view">
  <div class="summary-grid">
    <div class="summary-card">
      <h3>Overall Metrics</h3>
      <div id="overall-metrics"></div>
    </div>
    <div class="summary-card">
      <h3>Valid Solutions</h3>
      <div id="valid-solution-metrics"></div>
    </div>
    <div class="summary-card">
      <h3>Tuple Classification</h3>
      <div id="tuple-metrics"></div>
    </div>
    <div class="summary-card">
      <h3>Per Chain Length</h3>
      <div id="length-metrics"></div>
    </div>
    <div class="summary-card">
      <h3>Entropy Monotonicity</h3>
      <div id="entropy-mono-metrics"></div>
    </div>
    <div class="summary-card">
      <h3>STOP Metrics</h3>
      <div id="stop-metrics"></div>
    </div>
  </div>
</div>

<!-- Bar Chart View -->
<div id="barChartView" class="view active">
  <div class="controls">
    <label for="exampleSelect">Example:</label>
    <select id="exampleSelect"></select>
    <label for="numVisible">Show N:</label>
    <input type="number" id="numVisible" value="3" min="1" max="50">
    <label><input type="checkbox" id="logScale"> Log scale</label>
    <label><input type="checkbox" id="unseenOnly"> Unseen tuples only</label>
    <button id="btnPrev">&#9664; Prev</button>
    <button id="btnNext">Next &#9654;</button>
    <button class="stats-btn" id="btnStats">Statistics</button>
    <span id="rangeInfo" style="font-size:0.85rem; color:#666;"></span>
  </div>
  <div id="statsPanel"></div>
  <div id="cardsContainer"></div>
  <div class="legend" style="margin-top:8px;">
    <span style="background:#4caf50;"></span> Correct letter (from f)&nbsp;&nbsp;
    <span style="background:#2196f3;"></span> Correct letter (from g)&nbsp;&nbsp;
    <span style="background:#00bcd4;"></span> Correct letter (both f&amp;g)&nbsp;&nbsp;
    <span style="background:#ccc;"></span> Other letter&nbsp;&nbsp;
    <span style="background:transparent; border:2px solid #f44336; width:10px; height:10px;"></span> Model's choice
  </div>
</div>

<!-- Entropy Token View -->
<div id="entropyView" class="view">
  <div class="entropy-controls">
    <label>Example:</label>
    <select id="entropySelector" class="entropy-selector" onchange="showEntropySample(this.value)"></select>
    <button class="entropy-nav-btn" onclick="prevEntropySample()">&#9664; Prev</button>
    <button class="entropy-nav-btn" onclick="nextEntropySample()">Next &#9654;</button>
    <span id="entropyRangeInfo" style="font-size:0.85rem; color:#888;"></span>
  </div>
  <div class="entropy-controls" style="margin-top:-8px;">
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="filterValidDiffFinal" onchange="applyEntropyFilters()"> Valid but Different Final
    </label>
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="filterHasInvalidLetter" onchange="applyEntropyFilters()"> Has Invalid Letter (not f/g)
    </label>
    <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
      <input type="checkbox" id="filterNonMonotone" onchange="applyEntropyFilters()"> Non-Monotone Entropy [ENT\u2191]
    </label>
    <span id="entropyFilterInfo" style="font-size:0.85rem; color:#1976d2; font-weight:500;"></span>
  </div>

  <div class="entropy-legend">
    <span style="color:#1976d2; font-weight:bold;">Entropy:</span>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:rgb(0,180,255);"></div>Low</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:rgb(100,220,150);"></div>Med-Low</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:rgb(255,220,100);"></div>Medium</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:rgb(255,150,80);"></div>Med-High</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:rgb(255,80,80);"></div>High</div>
    <div class="entropy-legend-item" style="margin-left:15px;"><div style="width:20px;height:14px;box-shadow:0 0 4px #ff9800;border-radius:3px;background:#ddd;"></div>Decision Point</div>
    <div class="entropy-legend-item" style="margin-left:15px;"><div style="width:20px;height:14px;border-radius:3px;background:#fff;border:1px solid #ddd;position:relative;font-family:monospace;font-size:10px;text-align:center;line-height:14px;">a<span style="color:#f44336;position:absolute;top:-4px;right:1px;font-size:8px;">●</span></div></div><span>Not from f/g</span>
    <div class="entropy-legend-item" style="margin-left:15px;"><span style="font-size:0.78rem;padding:2px 6px;border-radius:4px;background:#fce4ec;color:#c62828;border:1px solid #ef9a9a;font-weight:500;">ENT\u2191</span></div><span>Non-monotone entropy</span>
    <span style="color:#1976d2; font-weight:bold; margin-left:20px;">P(STOP):</span>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:#f44336;height:4px;"></div>&gt;50%</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:#ff9800;height:4px;"></div>10-50%</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:#fdd835;height:4px;"></div>1-10%</div>
    <div class="entropy-legend-item"><div class="entropy-legend-color" style="background:transparent;height:4px;border:1px solid #ddd;"></div>&lt;1%</div>
  </div>

  <div id="entropyContent"></div>
</div>

<!-- Entropy Monotonicity View -->
<div id="monoView" class="view">
  <div style="background:#fff;padding:20px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
    <h2 style="color:#1976d2;font-size:1.1rem;margin-bottom:6px;">Entropy Monotonicity Analysis</h2>
    <p style="color:#666;font-size:0.88rem;margin-bottom:20px;">
      Per-example normalized decision-point entropy (highest step = 1.0).
      Monotone chains (green) have strictly decreasing entropy at every step.
      Non-monotone chains (red) have at least one entropy increase.
    </p>
    <div id="monoSummaryStats" style="margin-bottom:20px;padding:14px;background:#f8f9fa;border-radius:8px;border:1px solid #e0e0e0;font-size:0.9rem;"></div>
    <div id="monoChartsContainer"></div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<script>
Chart.register(ChartDataLabels);

const DATA = {data_json};
const LETTERS = {letters_json};
const ENTROPY_DATA = {entropy_json};
const METRICS = {metrics_json};
const MONO_DATA = {mono_json};

let currentView = 'barChart';
let entropyIndex = 0;

// ═══════════════════════════════════════════════════════
// SUMMARY VIEW
// ═══════════════════════════════════════════════════════
function initSummaryView() {{
  const fmt = (v, pct=false) => {{
    if (v === undefined || v === null) return '—';
    if (pct) return (v * 100).toFixed(2) + '%';
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(4);
  }};

  // Metric tooltip descriptions (HTML with KaTeX delimiters)
  const METRIC_DESC = {{
    'n_way_match': 'Predicted letter sequence \\\\( (\\\\ell_1, \\\\dots, \\\\ell_N) \\\\) matches one of the \\\\( 2^N \\\\) ground-truth traces AND all intermediate vectors match. Each GT trace is a combination of decision functions \\\\( d_i \\\\in \\\\{{f, g\\\\}} \\\\) at every step.',
    'all_blocks_valid': 'All computation blocks in the trace are mathematically correct: \\\\( \\\\ell_i(\\\\mathbf{{v}}_{{i-1}}) = \\\\mathbf{{v}}_i \\\\) for every step \\\\( i \\\\).',
    'parseable': 'Fraction of outputs successfully parsed into operation sequences separated by " ; ".',
    'total_inputs': 'Total number of unique input vectors evaluated.',
    'avg_p_f': 'Average softmax probability for the letter selected by decision function \\\\( f \\\\): \\\\( \\\\bar{{P}}(\\\\ell_f) = \\\\frac{{1}}{{N_{{\\\\text{{steps}}}}}} \\\\sum_i P(\\\\ell_{{f,i}}) \\\\).',
    'avg_p_g': 'Average softmax probability for the letter selected by decision function \\\\( g \\\\): \\\\( \\\\bar{{P}}(\\\\ell_g) = \\\\frac{{1}}{{N_{{\\\\text{{steps}}}}}} \\\\sum_i P(\\\\ell_{{g,i}}) \\\\).',
    'valid_chains': '<b>Valid chain.</b> Requires: (1) correct chain length \\\\( N_{{\\\\text{{pred}}}} = N_{{\\\\text{{expected}}}} \\\\), (2) every predicted letter \\\\( \\\\ell_i \\\\in \\\\{{ \\\\ell_{{f,i}}, \\\\ell_{{g,i}} \\\\}} \\\\), and (3) all blocks mathematically valid. Output may differ from prompt.',
    'num_valid_chains': 'Count of valid chains.',
    'valid_solutions': '<b>Valid solution.</b> Valid chain AND final computed vector matches the OUTPUT vector in the prompt: \\\\( \\\\mathbf{{v}}_N = \\\\mathbf{{v}}_{{\\\\text{{prompt}}}} \\\\).',
    'num_valid_solutions': 'Count of valid solutions.',
    'all_steps_valid': '<b>Lenient.</b> Every predicted step chose an \\\\( f \\\\) or \\\\( g \\\\) letter AND is mathematically valid, but chain length is <b>not</b> checked: \\\\( \\\\forall i: \\\\ell_i \\\\in \\\\{{ \\\\ell_{{f,i}}, \\\\ell_{{g,i}} \\\\}} \\\\wedge \\\\ell_i(\\\\mathbf{{v}}_{{i-1}}) = \\\\mathbf{{v}}_i \\\\).',
    'num_all_steps_valid': 'Count of all-steps-valid predictions.',
    'train_tuple': 'Fraction where the chosen letter tuple \\\\( (\\\\ell_1, \\\\dots, \\\\ell_N) \\\\) was seen during training.',
    'test_tuple': 'Fraction where the chosen letter tuple is from the test set (unseen during training).',
    'neither_tuple': 'Fraction where the chosen letter tuple is neither in train nor test set.',
    'avg_P_stop_wrong': 'Average \\\\( P(\\\\text{{STOP}}) \\\\) when the model is given a prefix ending with a wrong-branch (opposite decision function) step. High values indicate the model detects incorrect branches.',
    'avg_P_stop_correct': 'Average \\\\( P(\\\\text{{STOP}}) \\\\) when the model is given a prefix ending with a correct step. Low values are desired — the model should not want to stop after correct steps.',
    'stop_discrimination': 'Difference \\\\( P(\\\\text{{STOP}}|\\\\text{{wrong}}) - P(\\\\text{{STOP}}|\\\\text{{correct}}) \\\\). Higher is better — indicates the model can distinguish wrong from correct branches.',
    'stop_in_free_gen_rate': 'Fraction of free-generation outputs that contain a STOP token anywhere in the generated sequence.',
    'entropy_monotone': 'Fraction of chains where the Shannon entropy \\\\( H_k = -\\\\sum_a p_k(a) \\\\log p_k(a) \\\\) of the 20-letter decision-point distribution decreases at every step: \\\\( H_{{k+1}} \\\\le H_k + \\\\varepsilon \\\\) for all \\\\( k \\\\), with \\\\( \\\\varepsilon = 0.01 \\\\).',
    'entropy_monotone_acc': 'Valid-solution accuracy among chains with monotonically decreasing decision-point entropy.',
    'entropy_non_monotone_acc': 'Valid-solution accuracy among chains where entropy increases at one or more decision points.',
    'entropy_gap': 'Accuracy gap: monotone accuracy minus non-monotone accuracy. Positive values indicate monotone chains are more likely correct.',
    'entropy_violations': 'Average number of steps where entropy increases (violations of monotonicity) per chain.'
  }};

  // Helper: build a metric row with rich tooltip
  function mrow(label, value, descKey, extraClass) {{
    const cls = extraClass ? ` ${{extraClass}}` : '';
    return `<div class="metric-row"><span class="metric-label">${{label}}<span class="metric-tooltip-data" style="display:none">${{METRIC_DESC[descKey]}}</span></span><span class="metric-value${{cls}}">${{value}}</span></div>`;
  }}

  // Overall metrics
  let overallHtml = mrow('N-way Match', fmt(METRICS.n_way_match, true), 'n_way_match', 'metric-highlight')
    + mrow('All Blocks Valid', fmt(METRICS.all_blocks_valid, true), 'all_blocks_valid')
    + mrow('Parseable', fmt(METRICS.parseable_fraction, true), 'parseable')
    + mrow('Total Inputs', fmt(METRICS.num_unique_inputs), 'total_inputs')
    + mrow('Avg P(f)', fmt(METRICS.avg_prob_f), 'avg_p_f')
    + mrow('Avg P(g)', fmt(METRICS.avg_prob_g), 'avg_p_g');
  document.getElementById('overall-metrics').innerHTML = overallHtml;

  // Valid chain / solution metrics
  let validHtml = mrow('Valid Chains', fmt(METRICS.valid_chain_frac, true), 'valid_chains', 'metric-highlight')
    + mrow('# Valid Chains', fmt(METRICS.num_valid_chains), 'num_valid_chains')
    + mrow('Valid Solutions', fmt(METRICS.valid_solution_frac, true), 'valid_solutions', 'metric-highlight')
    + mrow('# Valid Solutions', fmt(METRICS.num_valid_solutions), 'num_valid_solutions')
    + mrow('All Steps Valid (any length)', fmt(METRICS.all_steps_valid_frac, true), 'all_steps_valid')
    + mrow('# All Steps Valid', fmt(METRICS.num_all_steps_valid), 'num_all_steps_valid');
  document.getElementById('valid-solution-metrics').innerHTML = validHtml;

  // Tuple classification
  let tupleHtml = mrow('Chose Train Tuple', fmt(METRICS.chosen_train_tuple_frac, true), 'train_tuple')
    + mrow('Chose Test Tuple', fmt(METRICS.chosen_test_tuple_frac, true), 'test_tuple')
    + mrow('Chose Neither', fmt(METRICS.chosen_neither_frac, true), 'neither_tuple');
  document.getElementById('tuple-metrics').innerHTML = tupleHtml;

  // Per chain length — discover lengths from metrics keys
  const detectedLens = Object.keys(METRICS).filter(k => k.match(/^n_len\d+$/)).map(k => parseInt(k.replace('n_len', ''))).sort((a,b) => a-b);
  const summaryLengths = detectedLens.length > 0 ? detectedLens : [3, 4, 5];
  let lengthHtml = '';
  summaryLengths.forEach(cl => {{
    const nway = METRICS[`n_way_match_len${{cl}}`];
    const validChain = METRICS[`valid_chain_frac_len${{cl}}`];
    const validSolution = METRICS[`valid_solution_frac_len${{cl}}`];
    const allStepsValid = METRICS[`all_steps_valid_frac_len${{cl}}`];
    const n = METRICS[`n_len${{cl}}`];
    if (n !== undefined) {{
      lengthHtml += `<div style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #e0e0e0;">
        <div style="color:#1976d2;font-weight:bold;margin-bottom:8px;">Length ${{cl}} (n=${{n}})</div>
        ${{mrow('N-way Match', fmt(nway, true), 'n_way_match')}}
        ${{mrow('Valid Chain', fmt(validChain, true), 'valid_chains')}}
        ${{mrow('Valid Solution', fmt(validSolution, true), 'valid_solutions')}}
        ${{mrow('All Steps Valid', fmt(allStepsValid, true), 'all_steps_valid')}}
      </div>`;
    }}
  }});
  document.getElementById('length-metrics').innerHTML = lengthHtml || '<p style="color:#888;">No per-length data</p>';

  // Entropy monotonicity card
  let entMonoHtml = '';
  if (METRICS.entropy_monotone_frac !== undefined) {{
    entMonoHtml += mrow('Monotone Fraction', fmt(METRICS.entropy_monotone_frac, true), 'entropy_monotone', 'metric-highlight');
    entMonoHtml += mrow('# Monotone', fmt(METRICS.num_entropy_monotone), 'entropy_monotone');
    entMonoHtml += mrow('# Non-Monotone', fmt(METRICS.num_entropy_non_monotone), 'entropy_monotone');
    entMonoHtml += mrow('Avg Violations', fmt(METRICS.avg_entropy_violations), 'entropy_violations');
    if (METRICS.entropy_monotone_accuracy !== undefined) {{
      entMonoHtml += mrow('Monotone Accuracy', fmt(METRICS.entropy_monotone_accuracy, true), 'entropy_monotone_acc');
    }}
    if (METRICS.entropy_non_monotone_accuracy !== undefined) {{
      entMonoHtml += mrow('Non-Monotone Accuracy', fmt(METRICS.entropy_non_monotone_accuracy, true), 'entropy_non_monotone_acc');
    }}
    if (METRICS.entropy_monotonicity_gap !== undefined) {{
      const gapClass = METRICS.entropy_monotonicity_gap > 0 ? 'metric-highlight' : 'metric-warning';
      entMonoHtml += mrow('Gap (mono \u2212 non)', (METRICS.entropy_monotonicity_gap > 0 ? '+' : '') + fmt(METRICS.entropy_monotonicity_gap, true), 'entropy_gap', gapClass);
    }}
  }} else {{
    entMonoHtml = '<p style="color:#888;">No entropy monotonicity data</p>';
  }}
  document.getElementById('entropy-mono-metrics').innerHTML = entMonoHtml;

  // STOP metrics card
  let stopHtml = '';
  if (METRICS.avg_P_stop_after_wrong !== undefined) {{
    const disc = METRICS.stop_discrimination;
    const discClass = disc !== undefined && disc > 0.1 ? 'metric-highlight' : (disc !== undefined && disc > 0 ? '' : 'metric-warning');
    stopHtml += mrow('P(STOP|wrong)', fmt(METRICS.avg_P_stop_after_wrong), 'avg_P_stop_wrong', 'metric-highlight')
      + mrow('P(STOP|correct)', fmt(METRICS.avg_P_stop_after_correct), 'avg_P_stop_correct')
      + mrow('Discrimination', fmt(METRICS.stop_discrimination), 'stop_discrimination', discClass)
      + mrow('STOP in Free Gen', fmt(METRICS.stop_in_free_gen_rate, true), 'stop_in_free_gen_rate');

    // Per-step breakdown
    let perStepHtml = '';
    for (let s = 0; s < 10; s++) {{
      const wrongKey = `step${{s}}_P_stop_after_wrong`;
      const correctKey = `step${{s}}_P_stop_after_correct`;
      if (METRICS[wrongKey] !== undefined || METRICS[correctKey] !== undefined) {{
        const w = METRICS[wrongKey] !== undefined ? fmt(METRICS[wrongKey]) : '—';
        const c = METRICS[correctKey] !== undefined ? fmt(METRICS[correctKey]) : '—';
        perStepHtml += `<div class="metric-row"><span class="metric-label">Step ${{s}}</span><span class="metric-value" style="font-size:0.85em;">W:${{w}} C:${{c}}</span></div>`;
      }}
    }}
    if (perStepHtml) {{
      stopHtml += `<div style="margin-top:12px;padding-top:10px;border-top:1px solid #e0e0e0;">
        <div style="color:#666;font-size:0.85em;margin-bottom:6px;">Per-Step P(STOP)</div>
        ${{perStepHtml}}
      </div>`;
    }}
  }} else {{
    stopHtml = '<p style="color:#888;">No STOP metrics available</p>';
  }}
  document.getElementById('stop-metrics').innerHTML = stopHtml;
}}

// Initialize summary on load, then render KaTeX in tooltips
window.addEventListener('load', initSummaryView);

// ═══════════════════════════════════════════════════════
// VIEW SWITCHING
// ═══════════════════════════════════════════════════════
function switchView(viewName) {{
  currentView = viewName;
  document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(viewName + 'View').classList.add('active');
  if (viewName === 'entropy') {{
    initEntropyView();
  }}
  if (viewName === 'mono') {{
    initMonoView();
  }}
}}

// ═══════════════════════════════════════════════════════
// ENTROPY VIEW
// ═══════════════════════════════════════════════════════
let entropyFilteredIndices = [];

// Check if example has any invalid letters (letters not from f/g at decision points)
function hasInvalidLetter(sample) {{
  const tokens = sample.tokens || [];
  const idxFPerStep = sample.idx_f_per_step || [];
  const idxGPerStep = sample.idx_g_per_step || [];

  let decisionPointCount = 0;
  for (let i = 0; i < tokens.length; i++) {{
    const t = tokens[i];
    if (t.is_decision_point) {{
      const stepNum = decisionPointCount;
      decisionPointCount++;
      if (stepNum < idxFPerStep.length) {{
        const idxF = idxFPerStep[stepNum];
        const idxG = idxGPerStep[stepNum];
        const letterIdx = LETTERS.indexOf(t.token.trim());
        if (letterIdx >= 0 && idxF !== null && idxG !== null) {{
          if (letterIdx !== idxF && letterIdx !== idxG) {{
            return true;
          }}
        }}
      }}
    }}
  }}
  return false;
}}

function applyEntropyFilters() {{
  const filterValidDiff = document.getElementById('filterValidDiffFinal').checked;
  const filterInvalid = document.getElementById('filterHasInvalidLetter').checked;
  const filterNonMono = document.getElementById('filterNonMonotone').checked;

  entropyFilteredIndices = [];
  ENTROPY_DATA.forEach((ex, i) => {{
    let include = true;
    if (filterValidDiff && !(ex.is_valid_chain && ex.has_different_final)) {{
      include = false;
    }}
    if (filterInvalid && !hasInvalidLetter(ex)) {{
      include = false;
    }}
    if (filterNonMono && ex.is_entropy_monotone !== false) {{
      include = false;
    }}
    if (include) {{
      entropyFilteredIndices.push(i);
    }}
  }});

  rebuildEntropyDropdown();
}}

function rebuildEntropyDropdown() {{
  const sel = document.getElementById('entropySelector');
  sel.innerHTML = '';

  entropyFilteredIndices.forEach((dataIdx, fi) => {{
    const ex = ENTROPY_DATA[dataIdx];
    const opt = document.createElement('option');
    opt.value = dataIdx;
    let label = '#' + dataIdx + ' — ' + ex.input_str.substring(0, 80);
    if (ex.is_valid_chain && ex.has_different_final) label += ' [DIFF]';
    if (hasInvalidLetter(ex)) label += ' [INV]';
    if (ex.is_entropy_monotone === false) label += ' [ENT\u2191]';
    opt.textContent = label;
    sel.appendChild(opt);
  }});

  const filterInfo = document.getElementById('entropyFilterInfo');
  if (entropyFilteredIndices.length < ENTROPY_DATA.length) {{
    filterInfo.textContent = 'Filtered: ' + entropyFilteredIndices.length + ' / ' + ENTROPY_DATA.length;
  }} else {{
    filterInfo.textContent = '';
  }}

  if (entropyFilteredIndices.length > 0) {{
    entropyIndex = entropyFilteredIndices[0];
    sel.value = entropyIndex;
    showEntropySample(entropyIndex);
  }} else {{
    document.getElementById('entropyContent').innerHTML = '<p style="color:#888;padding:20px;">No examples match the current filters.</p>';
    document.getElementById('entropyRangeInfo').textContent = '0 of 0';
  }}
}}

function initEntropyView() {{
  // Initialize filtered indices to all indices
  if (entropyFilteredIndices.length === 0) {{
    ENTROPY_DATA.forEach((_, i) => entropyFilteredIndices.push(i));
  }}
  rebuildEntropyDropdown();
}}

// ═══════════════════════════════════════════════════════
// ENTROPY MONOTONICITY VIEW
// ═══════════════════════════════════════════════════════
let monoViewInitialized = false;
let monoCharts = [];

function initMonoView() {{
  if (monoViewInitialized) return;
  monoViewInitialized = true;

  // ── Collect per-chain-length, per-monotonicity data ──
  // Uses MONO_DATA (all examples, not html-limited) when available, else falls back to ENTROPY_DATA
  const source = (typeof MONO_DATA !== 'undefined' && MONO_DATA.length > 0) ? MONO_DATA : ENTROPY_DATA;
  const groups = {{}};  // key: "3_true" etc → list of per_step_entropy arrays
  source.forEach(ex => {{
    const ent = ex.per_step_entropy;
    if (!ent || ent.length === 0) return;
    const cl = ex.chain_len;
    const mono = ex.is_entropy_monotone !== false;
    const key = cl + '_' + mono;
    if (!groups[key]) groups[key] = [];
    groups[key].push(ent);
  }});

  // Normalize per-example (max step entropy = 1.0) then average across examples
  function normalizeAndAverage(entropyArrays, numSteps) {{
    if (entropyArrays.length === 0) return new Array(numSteps).fill(0);
    const sums = new Array(numSteps).fill(0);
    const counts = new Array(numSteps).fill(0);
    entropyArrays.forEach(ent => {{
      // Find max entropy in this example (for normalization)
      const vals = ent.filter(h => h !== null && h !== undefined);
      const maxH = vals.length > 0 ? Math.max(...vals) : 1;
      const norm = maxH > 0 ? maxH : 1;
      for (let s = 0; s < numSteps && s < ent.length; s++) {{
        if (ent[s] !== null && ent[s] !== undefined) {{
          sums[s] += ent[s] / norm;
          counts[s]++;
        }}
      }}
    }});
    return sums.map((s, i) => counts[i] > 0 ? s / counts[i] : 0);
  }}

  // Compute standard error for error bars
  function normalizeAndSE(entropyArrays, numSteps) {{
    if (entropyArrays.length < 2) return new Array(numSteps).fill(0);
    const allNorm = entropyArrays.map(ent => {{
      const vals = ent.filter(h => h !== null && h !== undefined);
      const maxH = vals.length > 0 ? Math.max(...vals) : 1;
      const norm = maxH > 0 ? maxH : 1;
      return ent.map(h => (h !== null && h !== undefined) ? h / norm : null);
    }});
    const se = new Array(numSteps).fill(0);
    for (let s = 0; s < numSteps; s++) {{
      const vals = allNorm.map(a => a[s]).filter(v => v !== null);
      if (vals.length < 2) continue;
      const mean = vals.reduce((a,b) => a+b, 0) / vals.length;
      const variance = vals.reduce((a,b) => a + (b - mean) ** 2, 0) / (vals.length - 1);
      se[s] = Math.sqrt(variance / vals.length);
    }}
    return se;
  }}

  // Discover chain lengths dynamically from the data
  const clSet = new Set();
  source.forEach(ex => {{ if (ex.chain_len) clSet.add(ex.chain_len); }});
  const chainLengths = [...clSet].sort((a, b) => a - b);

  const container = document.getElementById('monoChartsContainer');
  container.innerHTML = '';

  // Summary stats
  const totalMono = source.filter(ex => ex.is_entropy_monotone !== false).length;
  const totalNon = source.length - totalMono;
  let statsHtml = `<div style="display:flex;gap:30px;flex-wrap:wrap;"><div><b>Total</b>: ${{source.length}} examples (${{totalMono}} monotone, ${{totalNon}} non-monotone)</div>`;
  chainLengths.forEach(cl => {{
    const monoArr = groups[cl + '_true'] || [];
    const nonArr = groups[cl + '_false'] || [];
    statsHtml += `<div><b>Length ${{cl}}</b>: ${{monoArr.length}} mono, ${{nonArr.length}} non-mono</div>`;
  }});
  statsHtml += '</div>';
  document.getElementById('monoSummaryStats').innerHTML = statsHtml;

  chainLengths.forEach(cl => {{
    const monoArr = groups[cl + '_true'] || [];
    const nonArr = groups[cl + '_false'] || [];
    if (monoArr.length === 0 && nonArr.length === 0) return;

    const monoAvg = normalizeAndAverage(monoArr, cl);
    const nonAvg = normalizeAndAverage(nonArr, cl);
    const monoSE = normalizeAndSE(monoArr, cl);
    const nonSE = normalizeAndSE(nonArr, cl);

    const stepLabels = Array.from({{length: cl}}, (_, i) => 'Step ' + (i + 1));

    // ── Section header ──
    const section = document.createElement('div');
    section.style.cssText = 'margin-bottom:40px;';
    section.innerHTML = `<h3 style="color:#1976d2;font-size:1rem;margin-bottom:14px;border-bottom:1px solid #e0e0e0;padding-bottom:8px;">
      Chain Length ${{cl}} &nbsp;<span style="font-size:0.85rem;color:#666;font-weight:normal;">
      (${{monoArr.length}} monotone, ${{nonArr.length}} non-monotone)</span></h3>`;

    // ── Charts row: bar + line side by side ──
    const row = document.createElement('div');
    row.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:20px;';

    // Bar chart canvas
    const barBox = document.createElement('div');
    barBox.style.cssText = 'background:#f8f9fa;padding:16px;border-radius:8px;border:1px solid #e0e0e0;';
    barBox.innerHTML = '<h4 style="font-size:0.88rem;color:#555;margin-bottom:10px;">Grouped Bar — Normalized Avg Entropy</h4>';
    const barCanvas = document.createElement('canvas');
    barCanvas.style.cssText = 'width:100%;height:300px;';
    barBox.appendChild(barCanvas);
    row.appendChild(barBox);

    // Line chart canvas
    const lineBox = document.createElement('div');
    lineBox.style.cssText = 'background:#f8f9fa;padding:16px;border-radius:8px;border:1px solid #e0e0e0;';
    lineBox.innerHTML = '<h4 style="font-size:0.88rem;color:#555;margin-bottom:10px;">Line — Entropy Trajectory</h4>';
    const lineCanvas = document.createElement('canvas');
    lineCanvas.style.cssText = 'width:100%;height:300px;';
    lineBox.appendChild(lineCanvas);
    row.appendChild(lineBox);

    section.appendChild(row);
    container.appendChild(section);

    // ── Render bar chart ──
    const barChart = new Chart(barCanvas.getContext('2d'), {{
      type: 'bar',
      data: {{
        labels: stepLabels,
        datasets: [
          {{
            label: 'Monotone (n=' + monoArr.length + ')',
            data: monoAvg,
            backgroundColor: 'rgba(76, 175, 80, 0.7)',
            borderColor: '#4caf50',
            borderWidth: 1,
            errorBars: monoSE,
          }},
          {{
            label: 'Non-Monotone (n=' + nonArr.length + ')',
            data: nonAvg,
            backgroundColor: 'rgba(244, 67, 54, 0.7)',
            borderColor: '#f44336',
            borderWidth: 1,
            errorBars: nonSE,
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          datalabels: {{
            display: true,
            anchor: 'end',
            align: 'end',
            font: {{ size: 11 }},
            formatter: v => v.toFixed(3),
            color: '#333',
          }},
          legend: {{ position: 'top', labels: {{ font: {{ size: 12 }} }} }},
          title: {{ display: false }},
        }},
        scales: {{
          y: {{
            beginAtZero: true,
            max: 1.15,
            title: {{ display: true, text: 'Normalized Avg Entropy', font: {{ size: 12 }} }},
          }},
          x: {{
            title: {{ display: true, text: 'Decision Step', font: {{ size: 12 }} }},
          }}
        }}
      }},
      plugins: [{{
        // Custom plugin: draw error bars
        id: 'errorBars',
        afterDraw(chart) {{
          const ctx = chart.ctx;
          chart.data.datasets.forEach((ds, di) => {{
            const meta = chart.getDatasetMeta(di);
            const se = ds.errorBars;
            if (!se) return;
            ctx.save();
            ctx.strokeStyle = ds.borderColor || '#333';
            ctx.lineWidth = 1.5;
            meta.data.forEach((bar, i) => {{
              if (se[i] === 0) return;
              const x = bar.x;
              const yVal = ds.data[i];
              const yTop = chart.scales.y.getPixelForValue(yVal + se[i]);
              const yBot = chart.scales.y.getPixelForValue(yVal - se[i]);
              const capW = 4;
              ctx.beginPath();
              ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
              ctx.moveTo(x - capW, yTop); ctx.lineTo(x + capW, yTop);
              ctx.moveTo(x - capW, yBot); ctx.lineTo(x + capW, yBot);
              ctx.stroke();
            }});
            ctx.restore();
          }});
        }}
      }}]
    }});
    monoCharts.push(barChart);

    // ── Render line chart ──
    const lineChart = new Chart(lineCanvas.getContext('2d'), {{
      type: 'line',
      data: {{
        labels: stepLabels,
        datasets: [
          {{
            label: 'Monotone (n=' + monoArr.length + ')',
            data: monoAvg,
            borderColor: '#4caf50',
            backgroundColor: 'rgba(76, 175, 80, 0.15)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: '#4caf50',
            borderWidth: 2.5,
          }},
          {{
            label: 'Non-Monotone (n=' + nonArr.length + ')',
            data: nonAvg,
            borderColor: '#f44336',
            backgroundColor: 'rgba(244, 67, 54, 0.15)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointBackgroundColor: '#f44336',
            borderWidth: 2.5,
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          datalabels: {{
            display: true,
            anchor: 'end',
            align: 'top',
            font: {{ size: 11 }},
            formatter: v => v.toFixed(3),
            color: '#333',
            offset: 4,
          }},
          legend: {{ position: 'top', labels: {{ font: {{ size: 12 }} }} }},
          title: {{ display: false }},
        }},
        scales: {{
          y: {{
            beginAtZero: true,
            max: 1.15,
            title: {{ display: true, text: 'Normalized Avg Entropy', font: {{ size: 12 }} }},
          }},
          x: {{
            title: {{ display: true, text: 'Decision Step', font: {{ size: 12 }} }},
          }}
        }}
      }}
    }});
    monoCharts.push(lineChart);
  }});
}}

// Shared tooltip positioning: show global tooltip above target, clamped to viewport
function showFixedTooltip(tipEl, anchorRect) {{
  tipEl.style.display = 'block';
  tipEl.style.left = '0px';
  tipEl.style.top = '0px';
  const tr = tipEl.getBoundingClientRect();
  const tipW = tr.width, tipH = tr.height, pad = 8;
  let left = anchorRect.left + anchorRect.width / 2 - tipW / 2;
  let top = anchorRect.top - tipH - 8;
  if (left < pad) left = pad;
  if (left + tipW > window.innerWidth - pad) left = window.innerWidth - pad - tipW;
  if (top < pad) top = anchorRect.bottom + 8;
  tipEl.style.left = left + 'px';
  tipEl.style.top = top + 'px';
}}

// Token tooltips — use global #globalTooltip element at body level
(function() {{
  const gTip = document.getElementById('globalTooltip');
  let activeWrapper = null;
  document.addEventListener('mouseover', function(e) {{
    const wrapper = e.target.closest('.token-wrapper');
    if (!wrapper || wrapper === activeWrapper) return;
    const data = wrapper.getAttribute('data-tip');
    if (!data) return;
    activeWrapper = wrapper;
    gTip.innerHTML = decodeURIComponent(data);
    showFixedTooltip(gTip, wrapper.getBoundingClientRect());
  }});
  document.addEventListener('mouseout', function(e) {{
    const wrapper = e.target.closest('.token-wrapper');
    if (!wrapper || wrapper !== activeWrapper) return;
    const related = e.relatedTarget;
    if (related && wrapper.contains(related)) return;
    gTip.style.display = 'none';
    activeWrapper = null;
  }});
}})();

// Metric tooltips — use global #globalMetricTooltip element at body level
(function() {{
  const gTip = document.getElementById('globalMetricTooltip');
  let activeLabel = null;
  document.addEventListener('mouseover', function(e) {{
    const label = e.target.closest('.metric-label');
    if (!label || label === activeLabel) return;
    const src = label.querySelector('.metric-tooltip-data');
    if (!src) return;
    activeLabel = label;
    gTip.innerHTML = src.innerHTML;
    showFixedTooltip(gTip, label.getBoundingClientRect());
    // Render KaTeX in the tooltip if available
    if (typeof renderMathInElement === 'function') {{
      renderMathInElement(gTip, {{
        delimiters: [{{left: '\\\\(', right: '\\\\)', display: false}}, {{left: '\\\\[', right: '\\\\]', display: true}}],
        throwOnError: false
      }});
    }}
  }});
  document.addEventListener('mouseout', function(e) {{
    const label = e.target.closest('.metric-label');
    if (!label || label !== activeLabel) return;
    const related = e.relatedTarget;
    if (related && label.contains(related)) return;
    gTip.style.display = 'none';
    activeLabel = null;
  }});
}})();

function entropyToColor(entropy, maxEntropy) {{
  const norm = Math.min(entropy / maxEntropy, 1.0);
  let r, g, b;
  if (norm < 0.25) {{
    r = 0; g = Math.floor(180 + 40 * (norm / 0.25)); b = 255;
  }} else if (norm < 0.5) {{
    const t = (norm - 0.25) / 0.25;
    r = Math.floor(100 * t); g = 220; b = Math.floor(255 - 105 * t);
  }} else if (norm < 0.75) {{
    const t = (norm - 0.5) / 0.25;
    r = Math.floor(100 + 155 * t); g = Math.floor(220 - 70 * t); b = Math.floor(150 - 70 * t);
  }} else {{
    const t = (norm - 0.75) / 0.25;
    r = 255; g = Math.floor(150 - 70 * t); b = 80;
  }}
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

function showEntropySample(index) {{
  entropyIndex = parseInt(index);
  const sample = ENTROPY_DATA[entropyIndex];
  const tokens = sample.tokens || [];
  const idxFPerStep = sample.idx_f_per_step || [];
  const idxGPerStep = sample.idx_g_per_step || [];

  document.getElementById('entropySelector').value = index;
  const filteredPos = entropyFilteredIndices.indexOf(entropyIndex);
  document.getElementById('entropyRangeInfo').textContent =
      'Sample ' + (filteredPos + 1) + ' of ' + entropyFilteredIndices.length + ' (#' + entropyIndex + ')';

  const entropies = tokens.map(t => t.entropy);
  // Use example-relative scaling: mean + 2*std as max, with minimum floor
  const meanEntropy = entropies.length > 0 ? entropies.reduce((a,b) => a+b, 0) / entropies.length : 0;
  const variance = entropies.length > 0 ? entropies.reduce((a,b) => a + (b - meanEntropy) ** 2, 0) / entropies.length : 0;
  const stdEntropy = Math.sqrt(variance);
  // Scale: use mean + 2*std as the "high" end, but ensure a minimum range
  const maxEntropy = Math.max(meanEntropy + 2 * stdEntropy, meanEntropy * 3, 0.1);

  // Find decision point indices and map them to steps
  let decisionPointCount = 0;
  const decisionPointToStep = {{}};
  tokens.forEach((t, i) => {{
    if (t.is_decision_point) {{
      decisionPointToStep[i] = decisionPointCount;
      decisionPointCount++;
    }}
  }});

  let tokensHtml = '';
  tokens.forEach((t, i) => {{
    const color = entropyToColor(t.entropy, maxEntropy);
    const displayToken = escapeHtml(t.token);
    const isDecision = t.is_decision_point ? ' decision-point' : '';

    // Check if this is a letter token at a decision point that is NOT from f/g
    let isInvalidLetter = false;
    if (t.is_decision_point) {{
      const stepNum = decisionPointToStep[i];
      if (stepNum !== undefined && stepNum < idxFPerStep.length) {{
        const idxF = idxFPerStep[stepNum];
        const idxG = idxGPerStep[stepNum];
        // Check if token is a letter (a-t)
        const letterIdx = LETTERS.indexOf(t.token.trim());
        if (letterIdx >= 0 && idxF !== null && idxG !== null) {{
          // Token is a letter - check if it's NOT one of the valid f/g choices
          isInvalidLetter = (letterIdx !== idxF && letterIdx !== idxG);
        }}
      }}
    }}
    const invalidClass = isInvalidLetter ? ' invalid-letter' : '';

    let altHtml = '';
    (t.top_k || []).forEach(([altTok, altProb]) => {{
      altHtml += `<div class="alt-token"><span class="alt-token-str">${{escapeHtml(altTok)}}</span><span class="alt-prob">${{(altProb*100).toFixed(1)}}%</span></div>`;
    }});

    const invalidInfo = isInvalidLetter ? '<div class="tooltip-row" style="color:#f44336;"><span class="tooltip-label">⚠</span><span class="tooltip-value">Not from f/g!</span></div>' : '';

    // Relative entropy indicator
    const relativeToAvg = meanEntropy > 0 ? (t.entropy / meanEntropy).toFixed(1) + 'x avg' : '—';
    const relativeColor = t.entropy > meanEntropy * 2 ? '#f44336' : t.entropy > meanEntropy ? '#ff9800' : '#4caf50';

    // P(STOP) bar color
    const pStop = t.p_stop || 0;
    let pStopColor, pStopLabel;
    if (pStop > 0.5) {{ pStopColor = '#f44336'; pStopLabel = 'HIGH'; }}
    else if (pStop > 0.1) {{ pStopColor = '#ff9800'; pStopLabel = 'MED'; }}
    else if (pStop > 0.01) {{ pStopColor = '#fdd835'; pStopLabel = 'LOW'; }}
    else {{ pStopColor = 'transparent'; pStopLabel = ''; }}

    const pStopInfo = pStop > 0.001
      ? `<div class="tooltip-row"><span class="tooltip-label">P(STOP):</span><span class="tooltip-value" style="color:${{pStop > 0.1 ? '#f44336' : '#666'}};">${{(pStop*100).toFixed(2)}}%</span></div>`
      : '';

    const tipContent = `<div class="tooltip-header">Token #${{i+1}}</div>`
      + `<div class="tooltip-row"><span class="tooltip-label">Token:</span><span class="tooltip-value">${{escapeHtml(t.token)}}</span></div>`
      + `<div class="tooltip-row"><span class="tooltip-label">Entropy:</span><span class="tooltip-value">${{t.entropy.toFixed(4)}} <span style="color:${{relativeColor}};font-size:0.85em;">(${{relativeToAvg}})</span></span></div>`
      + `<div class="tooltip-row"><span class="tooltip-label">Prob:</span><span class="tooltip-value">${{(t.probability*100).toFixed(2)}}%</span></div>`
      + pStopInfo
      + invalidInfo
      + `<div class="alternatives"><div class="alternatives-title">Top 5:</div>${{altHtml}}</div>`;

    tokensHtml += `
      <div class="token-wrapper${{isDecision}}" onclick="openSidebar(${{i}})" data-tip="${{encodeURIComponent(tipContent)}}">
        <div class="entropy-bar" style="background:${{color}};"></div>
        <div class="token${{invalidClass}}">${{displayToken}}</div>
        <div class="p-stop-bar" style="background:${{pStopColor}};"></div>
      </div>`;
  }});

  const maxEntropyVal = entropies.length > 0 ? Math.max(...entropies) : 0;
  const minEntropyVal = entropies.length > 0 ? Math.min(...entropies) : 0;

  const content = document.getElementById('entropyContent');
  content.innerHTML = `
    <div class="example-info">
      <div class="example-info-label">Input:</div>
      <div class="example-info-value">${{escapeHtml(sample.input_str)}}</div>
    </div>
    <div class="example-info">
      <div class="example-info-label">Prediction:</div>
      <div class="example-info-value">${{escapeHtml(sample.prediction)}}</div>
    </div>
    <div class="example-info" style="display:flex;gap:20px;flex-wrap:wrap;">
      <div><span class="example-info-label">Avg Entropy:</span> <span style="font-family:monospace;color:#1976d2;">${{meanEntropy.toFixed(4)}}</span></div>
      <div><span class="example-info-label">Min:</span> <span style="font-family:monospace;">${{minEntropyVal.toFixed(4)}}</span></div>
      <div><span class="example-info-label">Max:</span> <span style="font-family:monospace;">${{maxEntropyVal.toFixed(4)}}</span></div>
      <div><span class="example-info-label">Std:</span> <span style="font-family:monospace;">${{stdEntropy.toFixed(4)}}</span></div>
      <div><span class="example-info-label">Color scale max:</span> <span style="font-family:monospace;">${{maxEntropy.toFixed(4)}}</span></div>
    </div>
    ${{sample.per_step_entropy && sample.per_step_entropy.length > 0 ? `
    <div class="example-info" style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
      <div><span class="example-info-label">Decision-Point H:</span>
        <span style="font-family:monospace;color:#1976d2;">[${{sample.per_step_entropy.map(h => h !== null ? h.toFixed(3) : '?').join(' \\u2192 ')}}]</span></div>
      <div><span style="font-size:0.85rem;padding:3px 8px;border-radius:4px;font-weight:600;${{
        sample.is_entropy_monotone === false
          ? 'background:#fce4ec;color:#c62828;border:1px solid #ef9a9a;">ENT\\u2191 (' + sample.entropy_violation_count + ' violation' + (sample.entropy_violation_count !== 1 ? 's' : '') + ')'
          : 'background:#e8f5e9;color:#2e7d32;border:1px solid #4caf50;">Monotone \\u2193'
      }}</span></div>
    </div>` : ''}}
    <div class="token-container">${{tokensHtml}}</div>`;
}}

function prevEntropySample() {{
  const currentPos = entropyFilteredIndices.indexOf(entropyIndex);
  if (currentPos > 0) {{
    entropyIndex = entropyFilteredIndices[currentPos - 1];
    document.getElementById('entropySelector').value = entropyIndex;
    showEntropySample(entropyIndex);
  }}
}}
function nextEntropySample() {{
  const currentPos = entropyFilteredIndices.indexOf(entropyIndex);
  if (currentPos < entropyFilteredIndices.length - 1) {{
    entropyIndex = entropyFilteredIndices[currentPos + 1];
    document.getElementById('entropySelector').value = entropyIndex;
    showEntropySample(entropyIndex);
  }}
}}

function openSidebar(tokenIndex) {{
  const sample = ENTROPY_DATA[entropyIndex];
  const t = sample.tokens[tokenIndex];
  const tokens = sample.tokens || [];

  // Compute example-relative stats for sidebar
  const entropies = tokens.map(tk => tk.entropy);
  const meanEnt = entropies.length > 0 ? entropies.reduce((a,b) => a+b, 0) / entropies.length : 0;
  const variance = entropies.length > 0 ? entropies.reduce((a,b) => a + (b - meanEnt) ** 2, 0) / entropies.length : 0;
  const stdEnt = Math.sqrt(variance);
  const maxEnt = Math.max(meanEnt + 2 * stdEnt, meanEnt * 3, 0.1);
  const color = entropyToColor(t.entropy, maxEnt);

  const relativeToAvg = meanEnt > 0 ? (t.entropy / meanEnt).toFixed(2) : '—';
  const zScore = stdEnt > 0 ? ((t.entropy - meanEnt) / stdEnt).toFixed(2) : '—';

  let altHtml = '';
  (t.top_k || []).forEach(([tok, prob], idx) => {{
    altHtml += `<div class="alt-token"><span style="color:#666;margin-right:8px;">${{idx+1}}.</span><span class="alt-token-str">${{escapeHtml(tok)}}</span><span class="alt-prob">${{(prob*100).toFixed(2)}}%</span></div>`;
  }});

  document.getElementById('sidebar-content').innerHTML = `
    <div class="sidebar-header">Token #${{tokenIndex+1}} Details</div>
    <div class="sidebar-section">
      <div class="sidebar-section-title">Token</div>
      <div style="background:#f8f9fa;padding:12px;border-radius:6px;text-align:center;border:1px solid #e0e0e0;">
        <div style="font-size:1.8em;font-family:monospace;margin-bottom:8px;color:#333;">${{escapeHtml(t.token)}}</div>
        <div style="color:#666;">ID: ${{t.token_id}}</div>
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-section-title">Entropy</div>
      <div style="background:#f8f9fa;padding:12px;border-radius:6px;border:1px solid #e0e0e0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <span style="font-size:1.4em;font-weight:bold;color:#1976d2;">${{t.entropy.toFixed(4)}}</span>
          <div style="width:40px;height:18px;background:${{color}};border-radius:4px;"></div>
        </div>
        <div style="font-size:0.85em;color:#666;">
          <div style="display:flex;justify-content:space-between;"><span>vs. avg:</span><span style="color:#1976d2;">${{relativeToAvg}}x</span></div>
          <div style="display:flex;justify-content:space-between;"><span>z-score:</span><span>${{zScore}}</span></div>
          <div style="display:flex;justify-content:space-between;"><span>example avg:</span><span>${{meanEnt.toFixed(4)}}</span></div>
        </div>
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-section-title">Probability</div>
      <div style="background:#f8f9fa;padding:12px;border-radius:6px;border:1px solid #e0e0e0;">
        <span style="font-size:1.2em;color:#1976d2;">${{(t.probability*100).toFixed(2)}}%</span>
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-section-title">P(STOP)</div>
      <div style="background:#f8f9fa;padding:12px;border-radius:6px;border:1px solid #e0e0e0;">
        ${{(() => {{
          const ps = t.p_stop || 0;
          const barColor = ps > 0.5 ? '#f44336' : ps > 0.1 ? '#ff9800' : ps > 0.01 ? '#fdd835' : '#e0e0e0';
          const textColor = ps > 0.1 ? '#f44336' : ps > 0.01 ? '#ff9800' : '#666';
          return `<div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.2em;color:${{textColor}};font-weight:bold;">${{(ps*100).toFixed(2)}}%</span>
            <div style="flex:1;height:8px;background:#eee;border-radius:4px;overflow:hidden;">
              <div style="width:${{Math.min(ps*100, 100)}}%;height:100%;background:${{barColor}};border-radius:4px;"></div>
            </div>
          </div>`;
        }})()}}
      </div>
    </div>
    <div class="sidebar-section">
      <div class="sidebar-section-title">Top Alternatives</div>
      <div style="background:#f8f9fa;padding:12px;border-radius:6px;border:1px solid #e0e0e0;">${{altHtml}}</div>
    </div>`;

  document.getElementById('token-sidebar').classList.add('open');
  document.querySelectorAll('.token-wrapper').forEach((el, idx) => {{
    el.classList.toggle('selected', idx === tokenIndex);
  }});
}}

function closeSidebar() {{
  document.getElementById('token-sidebar').classList.remove('open');
  document.querySelectorAll('.token-wrapper').forEach(el => el.classList.remove('selected'));
}}

// ═══════════════════════════════════════════════════════
// BAR CHART VIEW (original code)
// ═══════════════════════════════════════════════════════
let filteredIndices = [];
let cursorPos = 0;
let activeCharts = [];

const sel = document.getElementById("exampleSelect");

function isLogScale() {{ return document.getElementById("logScale").checked; }}
function isUnseenOnly() {{ return document.getElementById("unseenOnly").checked; }}
function getNumVisible() {{ return Math.max(1, Math.min(50, parseInt(document.getElementById("numVisible").value) || 3)); }}
function destroyCharts() {{ activeCharts.forEach(c => c.destroy()); activeCharts = []; }}

function rebuildDropdown() {{
  const unseen = isUnseenOnly();
  filteredIndices = [];
  sel.innerHTML = "";
  DATA.forEach((ex, i) => {{
    if (unseen && ex.chosen_in_train !== false) return;
    filteredIndices.push(i);
    const opt = document.createElement("option");
    opt.value = filteredIndices.length - 1;
    const tupleLabel = ex.tuple_str ? "  " + ex.tuple_str + (ex.chosen_in_train === false ? " UNSEEN" : " TRAIN") : "";
    let label = "#" + i + "  vec=[" + ex.vec.join(",") + "]  " + ex.n_steps + "-step" + tupleLabel;
    if (ex.is_entropy_monotone === false) label += " [ENT\u2191]";
    opt.textContent = label;
    sel.appendChild(opt);
  }});
  cursorPos = 0;
  sel.value = 0;
}}

function render() {{
  destroyCharts();
  const useLog = isLogScale();
  const nv = getNumVisible();
  const total = filteredIndices.length;
  const end = Math.min(cursorPos + nv, total);
  document.getElementById("rangeInfo").textContent = total === 0 ? "No matching examples"
      : "Showing " + (cursorPos + 1) + "-" + end + " of " + total + (isUnseenOnly() ? " (unseen tuples)" : "");

  const container = document.getElementById("cardsContainer");
  container.innerHTML = "";

  for (let fi = cursorPos; fi < end; fi++) {{
    const dataIdx = filteredIndices[fi];
    const ex = DATA[dataIdx];
    const card = document.createElement("div");
    card.className = "example-card";

    let title = "<h3>#" + dataIdx + " &nbsp; vec = [" + ex.vec.join(", ") + "]  &nbsp; " + ex.n_steps + "-step";
    if (ex.tuple_str) {{
      if (ex.chosen_in_train === false) {{ title += '<span class="tag tag-unseen">chose ' + ex.tuple_str + ' UNSEEN</span>'; }}
      else if (ex.chosen_in_train === true) {{ title += '<span class="tag tag-seen">chose ' + ex.tuple_str + ' TRAIN</span>'; }}
      else {{ title += '<span class="tag" style="background:#eee;">chose ' + ex.tuple_str + '</span>'; }}
    }}
    if (ex.is_entropy_monotone === false) {{
      const vc = ex.entropy_violation_count || '?';
      title += '<span class="tag tag-ent-up">ENT\u2191 (' + vc + ' violation' + (vc !== 1 ? 's' : '') + ')</span>';
    }}
    if (ex.per_step_entropy && ex.per_step_entropy.length > 0) {{
      const entStr = ex.per_step_entropy.map(h => h !== null ? h.toFixed(2) : '?').join(' \u2192 ');
      title += ' <span style="font-size:0.75rem;color:#888;font-weight:normal;">H=[' + entStr + ']</span>';
    }}
    title += "</h3>";
    card.innerHTML = title;

    const chartsDiv = document.createElement("div");
    chartsDiv.className = "charts";

    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (step.probs) {{
        chartsDiv.appendChild(makeChartBox(step.probs, step.label, step.idx_f, step.idx_g, step.gen_letter, "c_" + fi + "_" + s, useLog));
      }} else {{
        const noData = document.createElement("div");
        noData.className = "chart-box";
        noData.innerHTML = "<h4>" + step.label + "</h4><p style='color:#999;font-size:0.85rem;'>No data</p>";
        chartsDiv.appendChild(noData);
      }}
    }}

    card.appendChild(chartsDiv);
    container.appendChild(card);
  }}
}}

function makeChartBox(probs, title, idxF, idxG, genLetter, canvasId, useLog) {{
  const box = document.createElement("div");
  box.className = "chart-box";
  box.innerHTML = "<h4>" + title + "</h4>";
  const canvas = document.createElement("canvas");
  canvas.id = canvasId;
  box.appendChild(canvas);

  const colors = probs.map((_, j) => {{
    const isF = (idxF !== null && idxF !== undefined && j === idxF);
    const isG = (idxG !== null && idxG !== undefined && j === idxG);
    if (isF && isG) return "#00bcd4";
    if (isF) return "#4caf50";
    if (isG) return "#2196f3";
    return "#ccc";
  }});

  const borders = probs.map((_, j) => (LETTERS[j] === genLetter) ? "#f44336" : "transparent");
  const borderWidths = probs.map((_, j) => (LETTERS[j] === genLetter) ? 3 : 0);
  const displayData = useLog ? probs.map(p => p > 0 ? Math.log10(p) : -6) : probs;
  const yConfig = useLog ? {{ title: {{ display: true, text: "log10 P" }}, grid: {{ color: "#eee" }}, min: -6, max: 0 }}
      : {{ beginAtZero: true, title: {{ display: true, text: "P(letter)" }}, grid: {{ color: "#eee" }} }};

  setTimeout(() => {{
    const chart = new Chart(canvas, {{
      type: "bar",
      data: {{ labels: LETTERS, datasets: [{{ data: displayData, backgroundColor: colors, borderColor: borders, borderWidth: borderWidths }}] }},
      options: {{
        responsive: true, maintainAspectRatio: false, layout: {{ padding: {{ top: 20 }} }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: (ctx) => {{
            const j = ctx.dataIndex; const p = probs[j];
            let label = LETTERS[j] + ": " + p.toFixed(4);
            if (idxF !== null && idxF !== undefined && j === idxF) label += " [f]";
            if (idxG !== null && idxG !== undefined && j === idxG) label += " [g]";
            return label;
          }} }} }},
          datalabels: {{
            anchor: "end", align: "top", font: {{ size: 9 }}, color: "#444",
            display: (ctx) => probs[ctx.dataIndex] >= 0.005,
            formatter: (val, ctx) => {{ const p = probs[ctx.dataIndex]; return p >= 0.01 ? p.toFixed(2) : p.toFixed(3); }}
          }}
        }},
        scales: {{ x: {{ grid: {{ display: false }} }}, y: yConfig }}
      }}
    }});
    activeCharts.push(chart);
  }}, 0);

  return box;
}}

// Navigation
document.getElementById("btnPrev").onclick = () => {{ cursorPos = Math.max(0, cursorPos - getNumVisible()); sel.value = cursorPos; render(); }};
document.getElementById("btnNext").onclick = () => {{ cursorPos = Math.min(Math.max(0, filteredIndices.length - 1), cursorPos + getNumVisible()); sel.value = cursorPos; render(); }};
sel.onchange = () => {{ cursorPos = parseInt(sel.value) || 0; render(); }};
document.getElementById("numVisible").onchange = render;
document.getElementById("logScale").onchange = render;
document.getElementById("unseenOnly").onchange = () => {{ rebuildDropdown(); render(); }};

// Statistics
function computeStats() {{
  const N = DATA.length;
  if (N === 0) return "<p>No data.</p>";

  // Group by chain length
  const byLen = {{}};
  DATA.forEach(ex => {{
    const k = ex.n_steps;
    if (!byLen[k]) byLen[k] = [];
    byLen[k].push(ex);
  }});
  const lengths = Object.keys(byLen).map(Number).sort((a,b) => a - b);

  // Tuple origin
  let trainCount = 0, unseenCount = 0, unknownCount = 0;
  const trainPerLen = {{}}, unseenPerLen = {{}};
  lengths.forEach(l => {{ trainPerLen[l] = 0; unseenPerLen[l] = 0; }});
  DATA.forEach(ex => {{
    if (ex.chosen_in_train === true) {{ trainCount++; trainPerLen[ex.n_steps]++; }}
    else if (ex.chosen_in_train === false) {{ unseenCount++; unseenPerLen[ex.n_steps]++; }}
    else unknownCount++;
  }});

  // Per-step probability stats
  const maxSteps = Math.max(...lengths);
  const probF = {{}}, probG = {{}}, mass = {{}};
  for (let s = 0; s < maxSteps; s++) {{ probF[s] = []; probG[s] = []; mass[s] = []; }}
  DATA.forEach(ex => {{
    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (!step.probs || step.idx_f === null || step.idx_f === undefined) continue;
      const pf = step.probs[step.idx_f];
      const pg = (step.idx_g !== null && step.idx_g !== undefined) ? step.probs[step.idx_g] : 0;
      probF[s].push(pf);
      probG[s].push(pg);
      const m = (step.idx_f === step.idx_g) ? pf : pf + pg;
      mass[s].push(m);
    }}
  }});

  // Per-step: model chose f-letter, g-letter, or other
  const choiceF = {{}}, choiceG = {{}}, choiceOther = {{}};
  for (let s = 0; s < maxSteps; s++) {{ choiceF[s] = 0; choiceG[s] = 0; choiceOther[s] = 0; }}
  DATA.forEach(ex => {{
    for (let s = 0; s < ex.steps.length; s++) {{
      const step = ex.steps[s];
      if (!step.gen_letter || step.idx_f === null || step.idx_f === undefined) continue;
      const gi = LETTERS.indexOf(step.gen_letter);
      if (gi === step.idx_f || gi === step.idx_g) {{
        if (gi === step.idx_f) choiceF[s]++;
        if (gi === step.idx_g) choiceG[s]++;
      }} else {{
        choiceOther[s]++;
      }}
    }}
  }});

  // Top tuples per length
  const tupleCounts = {{}};
  lengths.forEach(l => tupleCounts[l] = {{}});
  DATA.forEach(ex => {{
    if (!ex.tuple_str) return;
    const c = tupleCounts[ex.n_steps];
    c[ex.tuple_str] = (c[ex.tuple_str] || 0) + 1;
  }});

  const mean = arr => arr.length ? (arr.reduce((a,b) => a+b, 0) / arr.length) : NaN;
  const pct = (a, b) => b > 0 ? (100 * a / b).toFixed(1) + "%" : "—";
  const fmt = v => isNaN(v) ? "—" : v.toFixed(4);

  // Build HTML
  let h = "<h2>Dataset Statistics</h2>";

  // Overview
  h += "<h3>Overview</h3><table>";
  h += "<tr><th>Total examples</th><td class='num'>" + N + "</td></tr>";
  lengths.forEach(l => {{
    h += "<tr><th>" + l + "-step</th><td class='num'>" + byLen[l].length
      + " (" + pct(byLen[l].length, N) + ")</td></tr>";
  }});
  h += "</table>";

  // Tuple origin
  h += "<h3>Tuple Origin</h3><table>";
  h += "<tr><th></th><th>Train</th><th>Unseen</th><th>Unknown</th></tr>";
  h += "<tr><th>Overall</th><td class='num'>" + trainCount + " (" + pct(trainCount, N) + ")</td>"
    + "<td class='num'>" + unseenCount + " (" + pct(unseenCount, N) + ")</td>"
    + "<td class='num'>" + unknownCount + "</td></tr>";
  lengths.forEach(l => {{
    const tot = byLen[l].length;
    h += "<tr><th>" + l + "-step</th><td class='num'>" + trainPerLen[l]
      + " (" + pct(trainPerLen[l], tot) + ")</td>"
      + "<td class='num'>" + unseenPerLen[l]
      + " (" + pct(unseenPerLen[l], tot) + ")</td><td></td></tr>";
  }});
  h += "</table>";

  // Per-step probabilities
  h += "<h3>Per-Step Decision Probabilities</h3><table>";
  h += "<tr><th>Step</th><th>Mean P(f)</th><th>Mean P(g)</th><th>Mean mass</th><th>N</th></tr>";
  let allPf = [], allPg = [];
  for (let s = 0; s < maxSteps; s++) {{
    if (probF[s].length === 0) continue;
    allPf = allPf.concat(probF[s]);
    allPg = allPg.concat(probG[s]);
    h += "<tr><td>Step " + (s+1) + "</td>"
      + "<td class='num'>" + fmt(mean(probF[s])) + "</td>"
      + "<td class='num'>" + fmt(mean(probG[s])) + "</td>"
      + "<td class='num'>" + fmt(mean(mass[s])) + "</td>"
      + "<td class='num'>" + probF[s].length + "</td></tr>";
  }}
  if (allPf.length > 0) {{
    h += "<tr style='border-top:1px solid #ccc;'><th>Grand avg</th>"
      + "<td class='num'><b>" + fmt(mean(allPf)) + "</b></td>"
      + "<td class='num'><b>" + fmt(mean(allPg)) + "</b></td>"
      + "<td></td><td></td></tr>";
  }}
  h += "</table>";

  // Per-step model choice
  h += "<h3>Model's Letter Choice</h3><table>";
  h += "<tr><th>Step</th><th>Chose f-letter</th><th>Chose g-letter</th><th>Chose other</th></tr>";
  for (let s = 0; s < maxSteps; s++) {{
    const tot = choiceF[s] + choiceG[s] + choiceOther[s];
    if (tot === 0) continue;
    h += "<tr><td>Step " + (s+1) + "</td>"
      + "<td class='num'>" + choiceF[s] + " (" + pct(choiceF[s], tot) + ")</td>"
      + "<td class='num'>" + choiceG[s] + " (" + pct(choiceG[s], tot) + ")</td>"
      + "<td class='num'>" + choiceOther[s] + " (" + pct(choiceOther[s], tot) + ")</td></tr>";
  }}
  h += "</table>";

  // Top tuples per length
  h += "<h3>Top Tuples</h3>";
  lengths.forEach(l => {{
    const counts = tupleCounts[l];
    const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]).slice(0, 10);
    if (sorted.length === 0) return;
    const tot = byLen[l].length;
    h += "<b>" + l + "-step</b> (top 10):<table>";
    h += "<tr><th>Tuple</th><th>Count</th><th>%</th><th>Origin</th></tr>";
    sorted.forEach(([tup, cnt]) => {{
      const origin = DATA.find(ex => ex.tuple_str === tup && ex.n_steps === l);
      const tag = origin
        ? (origin.chosen_in_train === true ? "TRAIN" : origin.chosen_in_train === false ? "UNSEEN" : "?")
        : "?";
      h += "<tr><td>" + tup + "</td><td class='num'>" + cnt + "</td>"
        + "<td class='num'>" + pct(cnt, tot) + "</td>"
        + "<td>" + tag + "</td></tr>";
    }});
    h += "</table>";
  }});

  return h;
}}

document.getElementById("btnStats").onclick = () => {{
  const panel = document.getElementById("statsPanel");
  const btn = document.getElementById("btnStats");
  if (panel.classList.contains("visible")) {{ panel.classList.remove("visible"); btn.classList.remove("active"); }}
  else {{ if (!panel.innerHTML) panel.innerHTML = computeStats(); panel.classList.add("visible"); btn.classList.add("active"); }}
}};

// Initialize
rebuildDropdown();
render();
</script>
</body>
</html>"""
    return html


# ── Data builders ──

def build_examples_2step(l1_probs, l2_probs, vecs,
                         gen_letter1s=None, gen_letter2s=None,
                         train_pairs=None, pos2_idx_f=None, pos2_idx_g=None):
    """Build example dicts for original 2-step chains."""
    N = len(vecs)
    train_set = set(train_pairs) if train_pairs else set()
    examples = []

    for i in range(N):
        vec = vecs[i].tolist() if hasattr(vecs[i], 'tolist') else list(vecs[i])
        idx_f = decision_func_f(vec)
        idx_g = decision_func_g(vec)

        p1 = l1_probs[i].tolist() if hasattr(l1_probs[i], 'tolist') else list(l1_probs[i])
        p2 = l2_probs[i].tolist() if hasattr(l2_probs[i], 'tolist') else list(l2_probs[i])
        has_p2 = not all(v == 0.0 for v in p2)

        gen_l1 = gen_letter1s[i] if gen_letter1s else None
        gen_l2 = gen_letter2s[i] if gen_letter2s else None

        chosen_in_train = None
        if gen_l1 and gen_l2:
            chosen_in_train = (gen_l1, gen_l2) in train_set

        p2_f = pos2_idx_f[i] if pos2_idx_f else None
        p2_g = pos2_idx_g[i] if pos2_idx_g else None

        steps = [
            {
                "probs": [round(v, 6) for v in p1],
                "idx_f": idx_f,
                "idx_g": idx_g,
                "gen_letter": gen_l1,
                "label": "Step 1 (after [TRACE])",
            },
            {
                "probs": [round(v, 6) for v in p2] if has_p2 else None,
                "idx_f": p2_f,
                "idx_g": p2_g,
                "gen_letter": gen_l2,
                "label": "Step 2 (after ;)",
            },
        ]

        tuple_str = f"({gen_l1},{gen_l2})" if gen_l1 and gen_l2 else None

        examples.append({
            "idx": i,
            "vec": vec,
            "n_steps": 2,
            "steps": steps,
            "gen_letters": [gen_l1, gen_l2],
            "chosen_in_train": chosen_in_train,
            "tuple_str": tuple_str,
        })

    return examples


def build_examples_nstep(per_step_probs, vecs,
                         per_step_idx_f=None, per_step_idx_g=None,
                         gen_letters_all=None,
                         train_tuples_per_length=None):
    """Build example dicts for N-step extended chains.

    Args:
        per_step_probs: list of list of list[float] — [n_examples][n_steps][20]
        vecs: list/array of input vectors
        per_step_idx_f: list of list of int|None — correct f-letter index per step
        per_step_idx_g: list of list of int|None — correct g-letter index per step
        gen_letters_all: list of list of str|None — model's chosen letter per step
        train_tuples_per_length: dict {length: set of tuples} for train/test classification
    """
    N = len(vecs)
    train_sets = {}
    if train_tuples_per_length:
        for length, tuples in train_tuples_per_length.items():
            k = int(length)
            train_sets[k] = set(tuple(t) for t in tuples)

    examples = []
    for i in range(N):
        vec = vecs[i].tolist() if hasattr(vecs[i], 'tolist') else list(vecs[i])
        step_probs = per_step_probs[i]
        n_steps = len(step_probs)

        gen_letters = gen_letters_all[i] if gen_letters_all else [None] * n_steps
        idx_f_list = per_step_idx_f[i] if per_step_idx_f else [None] * n_steps
        idx_g_list = per_step_idx_g[i] if per_step_idx_g else [None] * n_steps

        steps = []
        for s in range(n_steps):
            probs = step_probs[s]
            has_probs = probs is not None and not all(v == 0.0 for v in probs)
            steps.append({
                "probs": [round(v, 6) for v in probs] if has_probs else None,
                "idx_f": idx_f_list[s],
                "idx_g": idx_g_list[s],
                "gen_letter": gen_letters[s] if s < len(gen_letters) else None,
                "label": f"Step {s+1}" + (" (after [TRACE])" if s == 0 else f" (after ;{s})"),
            })

        chosen_tuple = tuple(l for l in gen_letters if l is not None)
        chosen_in_train = None
        if len(chosen_tuple) == n_steps and n_steps in train_sets:
            chosen_in_train = chosen_tuple in train_sets[n_steps]

        tuple_str = "(" + ",".join(l or "?" for l in gen_letters) + ")" if gen_letters else None

        examples.append({
            "idx": i,
            "vec": vec,
            "n_steps": n_steps,
            "steps": steps,
            "gen_letters": list(gen_letters),
            "chosen_in_train": chosen_in_train,
            "tuple_str": tuple_str,
        })

    return examples


# ── Public API ──

def generate_html(l1_probs, l2_probs, vecs, save_path,
                  gen_letter1s=None, gen_letter2s=None,
                  train_pairs=None,
                  pos2_idx_f=None, pos2_idx_g=None):
    """Build and write HTML for 2-step chains (backwards compatible)."""
    examples = build_examples_2step(
        l1_probs, l2_probs, vecs,
        gen_letter1s=gen_letter1s, gen_letter2s=gen_letter2s,
        train_pairs=train_pairs,
        pos2_idx_f=pos2_idx_f, pos2_idx_g=pos2_idx_g,
    )
    html = build_html(examples)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved interactive visualization to {save_path}")


def generate_html_extended(per_step_probs, vecs, save_path,
                           per_step_idx_f=None, per_step_idx_g=None,
                           gen_letters_all=None,
                           train_tuples_per_length=None,
                           token_level_data=None,
                           unique_inputs=None,
                           predictions=None,
                           metrics=None,
                           is_valid_chain=None,
                           is_valid_solution=None,
                           has_different_final=None,
                           is_entropy_monotone=None,
                           entropy_violation_count=None,
                           per_step_entropy=None,
                           mono_data_all=None):
    """Build and write HTML for N-step extended chains."""
    examples = build_examples_nstep(
        per_step_probs, vecs,
        per_step_idx_f=per_step_idx_f,
        per_step_idx_g=per_step_idx_g,
        gen_letters_all=gen_letters_all,
        train_tuples_per_length=train_tuples_per_length,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Use enhanced HTML if token-level data is available
    if token_level_data is not None:
        html = build_html_with_entropy(examples, token_level_data, unique_inputs, predictions, metrics,
                                       per_step_idx_f=per_step_idx_f, per_step_idx_g=per_step_idx_g,
                                       is_valid_chain=is_valid_chain,
                                       is_valid_solution=is_valid_solution, has_different_final=has_different_final,
                                       is_entropy_monotone=is_entropy_monotone,
                                       entropy_violation_count=entropy_violation_count,
                                       per_step_entropy=per_step_entropy,
                                       mono_data_all=mono_data_all)
    else:
        html = build_html(examples)

    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved interactive visualization ({len(examples)} examples) to {save_path}")


# ── Standalone: load from .npz ──

def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML visualization of letter probabilities")
    parser.add_argument("--npz_path", required=True,
                        help="Path to decision_point_probs.npz")
    parser.add_argument("--output", type=str, default="letter_probs.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)

    # Detect format: 2-step (letter1_probs) vs N-step (per_step_probs)
    if "per_step_probs" in data:
        per_step_probs = data["per_step_probs"]
        vecs = data["unique_vecs"]
        per_step_idx_f = data.get("per_step_idx_f")
        per_step_idx_g = data.get("per_step_idx_g")
        gen_letters = data.get("gen_letters_all")
        print(f"Loaded {len(vecs)} examples (N-step format)")
        generate_html_extended(
            per_step_probs.tolist(), vecs, args.output,
            per_step_idx_f=per_step_idx_f.tolist() if per_step_idx_f is not None else None,
            per_step_idx_g=per_step_idx_g.tolist() if per_step_idx_g is not None else None,
            gen_letters_all=gen_letters.tolist() if gen_letters is not None else None,
        )
    else:
        l1 = data["letter1_probs"]
        l2 = data["letter2_probs"]
        vecs = data["unique_vecs"]
        print(f"Loaded {len(vecs)} examples (2-step format)")
        generate_html(l1, l2, vecs, args.output)


if __name__ == "__main__":
    main()
