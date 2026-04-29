"""Merge multiple standalone visualization HTMLs into a single self-contained
HTML with a tab switcher. Each source HTML is embedded as a string in a
<script type="text/html"> block; clicking a tab loads the corresponding
content into a sandboxed iframe via srcdoc.

Usage:
    python scripts/merge_viz_html.py \
        --out outputs/eval_results/comparison_view/comparison.html \
        --label "Pretrained|71/128 (55.5%)" --src outputs/.../pretrained.html \
        --label "GRPO|87/128 (68.0%)"      --src outputs/.../grpo.html \
        --label "Memory align|105/128 (82.0%)" --src outputs/.../memory.html \
        --title "Decision-Chain Letter Probability — 128 ffff val"
"""

import argparse
import re
import sys
from pathlib import Path


def escape_for_script(html: str) -> str:
    return re.sub(r"</(script)", r"<\\/\1", html, flags=re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", action="append", required=True,
                    help="'<button-label>|<score>' per source. Repeat per file.")
    ap.add_argument("--src", action="append", required=True,
                    help="Source HTML file. Repeat per checkpoint.")
    ap.add_argument("--title", default="Visualization Comparison")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    if len(args.label) != len(args.src):
        sys.exit("--label count must equal --src count")

    panels = []
    for i, (label, src) in enumerate(zip(args.label, args.src)):
        name, _, score = label.partition("|")
        html = Path(src).read_text(encoding="utf-8", errors="replace")
        html = escape_for_script(html)
        panels.append({"id": f"panel{i}", "name": name, "score": score,
                       "active": i == 0, "html": html})

    tabs_html = "\n      ".join(
        f'<button class="tab{" active" if p["active"] else ""}" data-target="{p["id"]}">'
        f'{p["name"]}<span class="score">{p["score"]}</span>'
        f'</button>'
        for p in panels
    )
    scripts_html = "\n".join(
        f'<script type="text/html" id="{p["id"]}">{p["html"]}</script>'
        for p in panels
    )

    out_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{args.title}</title>
<style>
  :root {{
    --fg: #1b2330;
    --muted: #5b6878;
    --bg: #f5f7fb;
    --panel: #ffffff;
    --accent: #2563eb;
    --accent-bg: #e6efff;
    --border: #d8dde6;
  }}
  html, body {{ margin: 0; padding: 0; height: 100%; background: var(--bg);
    color: var(--fg); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
  body {{ display: flex; flex-direction: column; }}
  header {{ padding: 12px 22px 8px; background: var(--panel); border-bottom: 1px solid var(--border); }}
  h1 {{ font-size: 16px; margin: 0 0 4px 0; font-weight: 600; }}
  .sub {{ font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }}
  .tabs {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
  .tab {{
    appearance: none;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--fg);
    padding: 7px 14px;
    border-radius: 8px;
    font-size: 13.5px;
    cursor: pointer;
    font-family: inherit;
    transition: background .12s, border-color .12s, color .12s;
  }}
  .tab:hover {{ background: var(--accent-bg); }}
  .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .tab .score {{ font-size: 11.5px; opacity: 0.85; margin-left: 6px; }}
  iframe {{ flex: 1; width: 100%; height: 100%; border: 0; background: var(--panel); }}
  #loading {{ padding: 20px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
  <header>
    <h1>{args.title}</h1>
    <div class="sub">{args.subtitle}</div>
    <div class="tabs">
      {tabs_html}
    </div>
  </header>
  <iframe id="view" sandbox="allow-scripts allow-same-origin" title="Visualization frame"></iframe>
  <div id="loading">Loading…</div>
{scripts_html}
<script>
(function() {{
  const view = document.getElementById('view');
  const loading = document.getElementById('loading');
  const tabs = document.querySelectorAll('.tab');
  let savedIdx = '0';

  function unescapeForBrowser(s) {{
    return s.replace(/<\\\\\\/script/gi, '</script');
  }}

  function readCurrentIdx() {{
    try {{
      const sel = view.contentDocument && view.contentDocument.getElementById('exampleSelect');
      if (sel && sel.value !== '') savedIdx = sel.value;
    }} catch (e) {{ /* ignore */ }}
  }}

  function applyIdxToFrame() {{
    try {{
      const doc = view.contentDocument;
      if (!doc) return;
      const sel = doc.getElementById('exampleSelect');
      if (!sel) return;
      const opts = Array.from(sel.options).map(o => o.value);
      if (opts.includes(savedIdx)) {{
        sel.value = savedIdx;
      }} else if (opts.length) {{
        const target = parseInt(savedIdx, 10);
        let best = opts[0];
        let bestDiff = Infinity;
        for (const v of opts) {{
          const d = Math.abs(parseInt(v, 10) - target);
          if (d < bestDiff) {{ bestDiff = d; best = v; }}
        }}
        sel.value = best;
        savedIdx = best;
      }}
      sel.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }} catch (e) {{ /* cross-origin or not ready */ }}
  }}

  function loadPanel(id) {{
    const tpl = document.getElementById(id);
    if (!tpl) return;
    readCurrentIdx();
    loading.style.display = 'block';
    const raw = unescapeForBrowser(tpl.textContent);
    view.srcdoc = raw;
    view.onload = () => {{
      loading.style.display = 'none';
      // Defer slightly so the inner page's init has run.
      setTimeout(applyIdxToFrame, 30);
    }};
  }}

  tabs.forEach(t => {{
    t.addEventListener('click', () => {{
      tabs.forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      loadPanel(t.dataset.target);
    }});
  }});

  const initial = document.querySelector('.tab.active') || tabs[0];
  if (initial) loadPanel(initial.dataset.target);
}})();
</script>
</body>
</html>
"""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_doc, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
