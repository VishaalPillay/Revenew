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
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from revenew.api.webhooks import get_conn
from revenew.api.webhooks import router as webhooks_router
from revenew.measure.report import build_report

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Revenew")
app.include_router(webhooks_router)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    report = build_report(conn)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "lifts": report.lifts,
            "overall": report.overall,
            "no_action_reasons": report.no_action_reasons,
            "candidate_validity": report.candidate_validity,
            "regret_curve": report.regret_curve,
            "posterior_recovery": report.posterior_recovery,
            "budget": {"consumed": report.budget_consumed},
            "run_id": report.run_id,
        },
    )
