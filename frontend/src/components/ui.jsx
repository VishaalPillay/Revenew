import { ROUTES, dec, n, navigate, pct, rs, titleCase } from '../lib/util.js'

export function Card({ title, aside, children, className = '', note }) {
  return (
    <div className={`card ${className}`}>
      {(title || aside) && (
        <div className="card-head">
          {title && <div className="overline">{title}</div>}
          {aside}
        </div>
      )}
      {children}
      {note && <div className="card-note">{note}</div>}
    </div>
  )
}

export function Stat({ label, value, sub, size = '', tone = '', note, className = '' }) {
  return (
    <div className={`card ${className}`}>
      {label && <div className="overline">{label}</div>}
      <div className={`stat ${size} ${tone}`} style={{ marginTop: 10 }}>
        {value}
      </div>
      {sub && <div className="stat-sub">{sub}</div>}
      {note && <div className="card-note">{note}</div>}
    </div>
  )
}

export function Nav({ path, runId }) {
  return (
    <header className="nav">
      <div className="wrap nav-inner">
        <a className="brand" href="#/theatre">
          <span className="brand-mark">R</span>
          Revenew
        </a>
        <nav className="nav-links">
          {ROUTES.map((r) => (
            <a
              key={r.path}
              className="nav-link"
              href={`#/${r.path}`}
              aria-current={path === r.path ? 'page' : undefined}
            >
              {r.label}
            </a>
          ))}
        </nav>
        <span className="spacer" />
        {runId && <span className="nav-meta">run {runId}</span>}
        <a className="btn btn-ghost btn-sm" href="/classic" title="The original server-rendered dashboard">
          Classic
        </a>
      </div>
    </header>
  )
}

export function Footer({ meta }) {
  return (
    <footer className="footer">
      <div className="wrap">
        Every figure on this console is read from <code>revenew.db</code> through{' '}
        <code>revenew/api/read.py</code> — the same functions <code>revenew report</code> and{' '}
        <code>revenew trace</code> call, so the console, the CLI, and the API cannot disagree.
        {meta?.run_id && (
          <>
            {' '}
            Run <code>{meta.run_id}</code>
            {meta.days ? `, ${meta.days} days` : ''}
            {meta.day_start ? ` (${meta.day_start} → ${meta.day_end})` : ''}.
          </>
        )}
      </div>
    </footer>
  )
}

export function Loading({ what = 'data' }) {
  return (
    <div className="loading">
      <div className="spin" />
      <div>Reading {what}…</div>
    </div>
  )
}

export function ErrorBox({ error }) {
  return (
    <div className="page wrap">
      <div className="err">
        <div className="title-md ink" style={{ marginBottom: 8 }}>
          Could not read the API
        </div>
        <div className="body-sm" style={{ marginBottom: 12 }}>
          {String(error?.message || error)}
        </div>
        {/* Deliberately does NOT suggest `cp demo_snapshot.db revenew.db`.
            That file is gitignored (*.db) and is never present in a fresh
            clone, so on the one surface every failed route lands in, it sent
            anyone following it straight into `cp: cannot stat`. `revenew demo`
            is the instruction that always works. */}
        <div className="body-sm muted">
          The console is served by the same process as the API. If this page loaded but the request
          failed, the database is probably empty — run <code className="mono">revenew demo</code> to
          populate it (~6 min, no API key needed; it replays the committed cassette).
        </div>
      </div>
    </div>
  )
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>
}

/* One decision, end to end. This is the only place the model is visible: the
 * envelope it was bound by, every candidate it proposed, the validator's
 * verdict on each, and which one the bandit drew. Shared by the theatre and
 * the decisions explorer so there is one rendering of a trace, not two. */
export function TracePanel({ trace, compact: isCompact = false }) {
  if (!trace) return <Empty>No decision selected.</Empty>

  const env = trace.envelope || {}
  const chosen = trace.chosen_candidate
  const cands = trace.candidates || []

  return (
    <div>
      <div className="row row-wrap" style={{ marginBottom: 12 }}>
        <span className="pill">{titleCase(trace.segment)}</span>
        <span className="pill mono">{String(trace.decision_id).slice(0, 14)}…</span>
        <span className="pill mono">customer {trace.customer_id}</span>
        {trace.window_id && <span className="pill mono">{trace.window_id}</span>}
        {trace.status === 'executed' ? (
          <span className="pill pill-ok">
            <i className="dot" /> executed
          </span>
        ) : (
          <span className="pill">
            <i className="dot" /> {trace.no_action_reason || trace.status}
          </span>
        )}
      </div>

      <div className="envelope">
        <b>Envelope the model was bound by</b> — max {pct(env.max_discount_pct, 0)} · cap{' '}
        {rs(env.max_absolute_discount)} · cooldown {env.cooldown_days}d ·{' '}
        {env.max_offers_per_customer_per_month}/customer/month · budget left{' '}
        {rs(env.budget_remaining)}
      </div>

      <div>
        {cands.map((c, i) => {
          const cand = c.candidate || {}
          // By index, resolved server-side (measure/report.py). Comparing
          // headlines here starred every candidate sharing a headline —
          // including ones the validator rejected.
          const isChosen = trace.chosen_candidate_index === c.candidate_index
          return (
            <div className={`cand${c.valid ? '' : ' invalid'}`} key={i}>
              <div className="star">{isChosen ? '★' : ''}</div>
              <div>
                <div className="headline">
                  {cand.headline || <span className="dim">(no headline)</span>}
                  {cand.discount_pct ? <span className="depth">{pct(cand.discount_pct, 0)}</span> : null}
                  {cand.discount_amount ? <span className="depth">{rs(cand.discount_amount)}</span> : null}
                </div>
                <div className="famline">{cand.action_family}</div>
              </div>
              <div>
                {c.valid ? (
                  <span className="pill pill-ok">
                    <i className="dot" /> legal
                  </span>
                ) : (
                  <span className="pill pill-bad" title={(c.violations || []).join(', ')}>
                    <i className="dot" /> {(c.violations || []).join(', ') || 'dropped'}
                  </span>
                )}
              </div>
            </div>
          )
        })}
        {!cands.length && <Empty>No candidates recorded for this decision.</Empty>}
      </div>

      {chosen && (
        <div className="card-note">
          <strong>★ The bandit drew {titleCase(trace.action_family)}</strong> — “{chosen.headline}” at
          propensity <strong>{dec(trace.propensity)}</strong>. Thompson sampling over the surviving
          families, not a fixed rule; the propensity is the probability that arm wins a draw, logged
          so an off-policy estimator can reuse this decision later.
          {trace.outcome && (
            <>
              {' '}
              The window closed{' '}
              {trace.outcome.converted ? (
                <strong>converted, {rs(trace.outcome.net_revenue)}</strong>
              ) : (
                <strong>with no conversion</strong>
              )}
              .
            </>
          )}
        </div>
      )}

      {!isCompact && (
        <div className="card-note">
          {n(trace.candidates_generated)} candidates generated, {n(trace.candidates_valid)} survived
          the envelope. The envelope is applied twice from one rule table — injected into the prompt
          and re-checked programmatically after generation — so a model error can only ever produce a
          suboptimal <em>legal</em> action, never an illegal one.
        </div>
      )}
    </div>
  )
}
