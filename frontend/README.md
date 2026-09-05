# Revenew console — frontend source

The React console served at `http://127.0.0.1:8000` by `revenew serve`.

**You do not need node to run or demo this project.** The build output is committed to
`revenew/api/static/` and FastAPI serves it directly, so `pip install -e . && revenew serve`
is the whole story — the same reasoning that commits the LLM cassettes rather than requiring
an API key to see a decision. This directory is here so the build is reproducible and
reviewable, not because anyone has to run it.

## If you do want to change it

```bash
cd frontend
npm install
npm run dev      # localhost:5173, proxies /api to a `revenew serve` on :8000
npm run build    # writes revenew/api/static/ — commit the result
```

`npm run dev` expects `revenew serve` already running in another shell; it proxies the API
through so the console hot-reloads against the real database.

## Layout

| Path | What it is |
|---|---|
| `src/routes/Theatre.jsx` | The hero. Plays the run back day by day from `/api/theatre` |
| `src/routes/Decisions.jsx` | Filterable decision list with the full audit trace |
| `src/routes/Learning.jsx` | Learning curve, regret, posterior recovery, learned-vs-truth grid |
| `src/routes/Proof.jsx` | Measured lift, policy compliance, no-action distribution, reproducibility |
| `src/components/charts.jsx` | Every chart, as inline SVG. No charting dependency |
| `src/components/ui.jsx` | Shell, cards, stats, and the shared decision-trace panel |
| `src/lib/util.js` | Hash router, cached API client, formatters, the playback clock |
| `src/styles.css` | The design system from `DESIGN-clickhouse.md`, as CSS custom properties |

## Two decisions worth knowing about

**The theatre animates client-side, not over a stream.** `/api/theatre` returns the whole
timeline — about 140 KB — and the browser walks it. That makes the run scrubbable and
pausable, makes a second view instant, and means a network hiccup mid-demo pauses a local
animation instead of stalling a live stream. See the module docstring in
`revenew/api/theatre.py` for the longer argument.

**Charts are hand-rolled SVG, and colour does almost no work in them.** The design system
allows exactly one accent hue, so identity is carried by position, direct labels, and a
single yellow against a recessive grey baseline. Nothing here needs a categorical palette,
which is also why nothing here can fail a colour-vision check. The green/red status pair is
the one exception, and it always ships with a text label beside it — that pair separates by
only ΔE 7.4 under deuteranopia, so colour alone would not be readable.

## Dependencies

React and ReactDOM. That is the entire runtime dependency list — no charting library, no
router, no CSS framework.
