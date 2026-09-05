"""The demo dashboard: one page, one server, one database connection.

Reads exclusively from `revenew.db`, via `revenew.measure.report.build_report`
-- including `demo_regret_curve` and `demo_posterior_recovery`, which are
DERIVED artifacts a harness run exports into this database once, after the
fact (see harness/regret.py). This module never imports anything from
`harness/` and never opens harness.db; the regret chart it renders is only as
fresh as the last exported run, and the page says so explicitly when the
tables are empty rather than silently showing nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from revenew.api.agent import router as agent_router
from revenew.api.live import router as live_router
from revenew.api.read import router as read_router
from revenew.api.webhooks import get_conn
from revenew.api.webhooks import router as webhooks_router
from revenew.measure.report import build_report, get_decision_trace
from revenew.settings import GROQ_MODEL

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# The built React console. `frontend/` holds the source; this directory holds
# its committed build output, so `pip install -e . && revenew serve` still
# needs no node toolchain -- the same reasoning that puts the recorded LLM
# cassettes in the repo rather than requiring an API key to see a decision.
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONSOLE_INDEX = STATIC_DIR / "index.html"

app = FastAPI(title="Revenew")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhooks_router)
app.include_router(read_router)
app.include_router(live_router)
app.include_router(agent_router)

# Mounted only when a build is present. A checkout that has not run
# `npm run build` still serves the API, the webhook receiver, and the classic
# dashboard; it just falls back to the classic page at `/` instead of 404ing
# on a directory that does not exist.
# Guarded on `assets/` specifically, not just `static/`. StaticFiles RAISES
# from its constructor if the directory is missing, so an interrupted
# `npm run build`, a partial checkout, or an sdist that packaged
# `api/static/*` but not `api/static/assets/*` would make importing this
# module fail outright -- taking down the webhook receiver and the whole read
# API, which is precisely what this guard exists to keep alive when the
# console is unavailable.
if (STATIC_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


def _latest_executed_decision_id(conn: sqlite3.Connection) -> str | None:
    """Default subject for the trace panel: the most recent decision the
    bandit actually chose. `no_action` decisions have no chosen candidate and
    no propensity, so they would render an empty panel."""
    row = conn.execute(
        "SELECT decision_id FROM decisions WHERE status = 'executed' "
        # opportunity_id, not decision_id, breaks a same-`created_at` tie --
        # decision_id is a random uuid4 per run, so tie-breaking on it would
        # make "the most recent decision" pick a DIFFERENT underlying
        # decision (by content) between two runs of the same seed, even
        # though the run's decisions are otherwise identical.
        "ORDER BY created_at DESC, opportunity_id DESC LIMIT 1"
    ).fetchone()
    return row["decision_id"] if row else None


@app.get("/", response_class=HTMLResponse)
def console(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    """The React console, or the classic dashboard if no build is present.

    The console routes entirely in the URL fragment, so this single route is
    the whole SPA entry point -- there is no catch-all rewrite here that could
    shadow a future API path. See `frontend/src/lib/util.js` for why hash
    routing was chosen over the history API."""
    if CONSOLE_INDEX.is_file():
        return FileResponse(CONSOLE_INDEX)
    return classic(request, None, conn)


@app.get("/classic", response_class=HTMLResponse)
def classic(
    request: Request,
    decision_id: str | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
):
    report = build_report(conn)
    # The trace panel is the one place the AI is actually visible: the raw
    # candidates the model proposed, each with the validator's verdict, and
    # which one the bandit picked. `?decision_id=` pins a specific decision;
    # the default is the most recent executed one. Same `get_decision_trace`
    # the CLI and /api/decisions/{id} call -- a third caller, not a third query.
    trace = get_decision_trace(conn, decision_id or _latest_executed_decision_id(conn) or "")
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "lifts": report.lifts,
            "overall": report.overall,
            "no_action_reasons": report.no_action_reasons,
            "candidate_validity": report.candidate_validity,
            "regret_curve": report.regret_curve,
            "regret_curve_all": report.regret_curve_all,
            "learning_curve": report.learning_curve,
            "posterior_recovery": report.posterior_recovery,
            "budget": {"consumed": report.budget_consumed},
            "run_id": report.run_id,
            "trace": trace,
            "llm_model": GROQ_MODEL,
        },
    )
