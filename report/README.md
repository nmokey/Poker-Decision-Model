# PocketSolver research article

The canonical manuscript is `index.qmd`. This is a local draft; building it does not
publish or change the existing PocketSolver deployment. The HTML, PDF and editable
LaTeX share prose, citations, figure identifiers, captions and result values.

## Build

From the repository root, with the analysis environment and Quarto installed:

```sh
.venv/bin/python scripts/build_report.py
```

Use `--quarto /absolute/path/to/quarto` if it is not on PATH. The renderer also
finds this session's temporary installation under `/tmp/poker-report-tools/`.
Use `--assets-only` to regenerate the data, tables, SVGs and print PDFs without
Quarto. Output is `report/_site/index.html`, `index.pdf`, `index.tex`, and `pocketsolver-latex.zip` (a self-contained LaTeX bundle). `pocketsolver-web.zip` contains the complete web
article for downloading to another computer; extract it and open `index.html`.
The article embeds its verified example snapshot, so its controls also work when
opening `index.html` directly. For a local HTTP preview:

```sh
.venv/bin/python -m http.server 8765 --directory report/_site
```

The build needs Matplotlib (the existing `analysis` extra), Quarto, and a working
LaTeX distribution for PDF. Quarto can install TinyTeX if a local TeX installation
is unavailable. The article loads the live demo in an expanded iframe
automatically in light mode to match the article. Model loading and comparison requests are managed by the embedded
application; recorded figure controls use only the embedded snapshot. Math
typesetting may load MathJax from its CDN.

## Sources and boundaries

Aggregate inputs are the reviewed files in `docs/results/`. The hand explorer joins
`artifacts/holdem_meeting5/published_bert_mini/hands.jsonl` (or the deployment asset
copy) to the BERT Mini fold-1 saved `oof.jsonl`. Obtain/reproduce these artifacts
using the existing project workflows before a full rebuild on a fresh checkout.
The builder selects the lexically first eligible example in each declared category
(preflop success, postflop success, wrong action type); the saved sample contains no wrong-amount examples. It validates
IDs, fold, targets, candidate counts and normalized probabilities. These are
purposefully selected illustrations, not a representative evaluation sample.
Candidate spans are schematic action spans, not claimed tokenizer boundaries.

Generated example data, full build products and temporary Quarto state are ignored.
No raw dataset, model weights, credentials or decision-level exports are added to
version control. Only selected rendered examples appear in the generated report.
`generated/provenance.json` records hashes of all inputs and outputs. Publishing
must carry the complete `_site/` directory, including Quarto libraries, assets,
generated figures, the PDF, and the source bundle.

## Figure contract

| ID | Evidence | Web behavior | Print equivalent |
| --- | --- | --- | --- |
| fig-hand | Saved BERT Mini fold-1 predictions | Choose a recorded example, reveal reference | Same default example and all scores |
| fig-architecture | Implemented joint encoder | Select a candidate to highlight its schematic input span | One complete architecture diagram |
| fig-models | Reviewed development matrix | Interactive SVG points, linked metric readouts, population toggle, exact-value table | Default accuracy/log-loss panels |
| fig-streets | Reviewed per-street matrix | Interactive SVG points with linked readouts and exact-value table | Same panels |
| fig-leduc | Canonical controlled Leduc aggregates | Interactive SVG points with linked readouts and exact-value table | Same panels |
| fig-holdem | Reviewed fixed-validation scaling results | Interactive SVG points with linked readouts and exact-value table | Same four panels |

Keep default states and print figures aligned. Figure captions describe sample,
metric and evaluation scope. Core claims cannot depend on an interaction. Web-only
content lives in format-conditional blocks; print alternatives share the outer
figure ID and caption. Result spans resolve through `filters/results.lua` from
`generated/report-data.json`; edit source data instead of typing duplicate values.

## Editorial work before publication

Member names follow the Members table on the Notion project page (including its
spelling and order). Confirm affiliations, expand related work and source/archive
citations, and review the proposed title and framing. Independent final evaluation,
multiple training seeds, fitted calibration, a controlled LLM comparison, latency
measurements, and payoff evaluation remain future experiments. The article makes
no result claims for these. The existing live application is expanded and embedded
by default; this draft does not alter its layout or serving code.

Browser verification covers the recorded-data controls and responsive layouts. The
external iframe is checked with an intercepted response to avoid issuing live
inference during verification; hosted app availability is not established by that
check. The article leaves the full serving UI inside the existing application.

## Interactive chart implementation

Web charts are inline SVG rendered from the embedded reviewed aggregates, with no
external chart library. A ResizeObserver measures panel widths; CSS stacks panels
below 700px. Hover, tap, focus, or arrow-key navigation selects a condition across
metric panels and exposes exact values in a shared readout. Escape and Clear
selection reset it. Tables expose every value without requiring pointer input.
Model colors and head symbols match the print figures; Hold’em uses log₂ sample
positions. Original SVG fallbacks and PDF/LaTeX figures remain available.

References inspected: [Dust](https://qlabs.sh/research/dust) uses inline SVG with
JavaScript hover/click highlights and readouts; [Cues](https://www.sophielwang.com/cues)
uses custom React/Next.js components, SVG/HTML marks, ResizeObserver and compact
layouts. Browser checks exercised their hover, heatmap, example, cue, model and
token controls. This implementation adopts the interaction pattern within Quarto.
