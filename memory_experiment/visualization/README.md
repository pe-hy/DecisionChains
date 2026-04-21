# Run Visualization

Static HTML table of every experiment in `../outputs/experiments/*.json`.
Zero build step — just vanilla HTML + CSS + JS. **Works under `file://`
with no server** (data is loaded via `<script src="data.js">`, not
`fetch()`, so Chrome's same-origin policy doesn't block it).

## Files

| File | Role |
|---|---|
| `index.html` | The viewer. Open in any browser. |
| `data.js`    | `window.RUNS = [...]` — the data, loaded by `index.html`. |
| `runs.json`  | Same payload in pure JSON — useful for scripting / debugging. |
| `build_data.py` | Regenerates both files from `../outputs/experiments/`. |

## Usage

```bash
# Regenerate after new experiments finish
python visualization/build_data.py

# Open in browser — works from file://, no server needed
xdg-open visualization/index.html          # Linux
open visualization/index.html              # macOS
start visualization\index.html             # Windows
```

If your browser still complains (very strict CORS, extensions, etc.),
fall back to a tiny local server:

```bash
cd visualization && python -m http.server 8000
# then visit http://localhost:8000
```

## Features

- **All 100+ runs** in one sortable table.
- **Metric tooltips** — hover any column header for the definition
  (what `f_selection` means, what `full_f_alignment` counts, etc.).
- **Delta cells** show `baseline → memory (Δ)` with magnitude as a
  shaded bar and sign colored green/red.
- **Filters**: free-text search plus dropdowns for `wandb_group`,
  `layer`, `data_filter`, `ce_mode`, `dp_target`.
- **Click column headers to sort** — defaults to full-val `f_selection`
  descending so the best-aligned runs float to the top.
- **Adding new runs**: just run `build_data.py` and refresh the page.
  No HTML edits.

## Why not serve from W&B?

We log there too for live monitoring, but this local viewer is
self-contained — no login, no network, works offline, trivial to
screenshot the whole table. Good for quick sanity checks and sharing.

## How it's wired together

```
outputs/experiments/*.json   ──►   build_data.py   ──►   data.js ─┐
                                                                  ├──► index.html (opens in browser)
                                                                  ┘
```

`build_data.py` extracts the scalar metrics + config from each run's
JSON (skipping per-step arrays too large for a table), writes
`data.js` as `window.RUNS = [...]`, which `index.html` reads via a
plain `<script>` tag. Because script tags aren't subject to the
file-origin restriction that `fetch()` is, the whole viewer works
from a downloaded folder with no setup.
