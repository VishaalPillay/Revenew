import { LiftBars } from '../components/charts.jsx'
import { Card, Empty, Loading, Stat } from '../components/ui.jsx'
import { compact, dec, n, pct, rs, titleCase, useApi } from '../lib/util.js'

export default function Proof() {
  const { data, error, loading } = useApi('/api/report')
  if (error) throw error
  if (loading) return <Loading what="the measured report" />

  const r = data
  const o = r.overall || {}
  const cv = r.candidate_validity || {}
  const reasons = r.no_action_reasons || []
  const totalWithheld = reasons.reduce((a, c) => a + c.n, 0)
  const significantSegments = (r.lifts || []).filter((l) => l.is_significant).length

  return (
    <div className="page wrap">
      <div className="page-head">
        <h1>Proof</h1>
        <p>
          The claims this project is willing to make, and the ones it deliberately refuses to. Every
          number here is measured against a 20% randomised holdout, not asserted — and the anti-metrics
          matter as much as the metrics.
        </p>
      </div>

      <div className="stack">
        <div className="grid g3">
          <div className="card-yellow">
            <div className="overline">Incremental lift, pooled</div>
            <div className="stat lg" style={{ marginTop: 8 }}>
              {o.lift > 0 ? '+' : ''}
              {dec(o.lift, 1)}
            </div>
            <div className="stat-sub">
              rupees per decision · 95% CI [{dec(o.ci_low, 1)}, {dec(o.ci_high, 1)}] ·{' '}
              {o.is_significant ? 'clear of zero' : 'not yet significant'}
            </div>
            <div className="card-note">
              <strong>The anti-metric:</strong> gross revenue from targeted customers is{' '}
              <em>not</em> reported here, and never will be. Treatment mean {dec(o.mean_treatment, 0)}{' '}
              against control mean {dec(o.mean_control, 0)} — most targeted customers would have
              bought anyway, and reporting the difference is the only honest option.
            </div>
          </div>

          <Stat
            label="LLM policy compliance"
            value={cv.policy_compliance_rate === null ? '—' : pct(cv.policy_compliance_rate, 2)}
            sub={
              <>
                <strong className="ink">{n(cv.policy_violations)}</strong> illegal offers proposed out
                of <strong className="ink">{n(cv.total_generated)}</strong> candidates.
              </>
            }
            note={
              <>
                A further {n(cv.eligibility_blocked)} candidates were dropped because the{' '}
                <em>customer</em> was ineligible — cooldown or monthly cap — and{' '}
                {n(cv.budget_blocked)} because the campaign <em>budget</em> had run dry. Neither is
                a model failure: one is a merchant contact setting, the other is how much money was
                left when the offer was costed. Conflating the first with real violations is what
                made an earlier version of this panel read 3%; conflating the second is what made it
                read 93.7% on a run where the model's illegal-offer count was still exactly zero.
              </>
            }
          />

          <Stat
            label="Budget consumed"
            value={compact(r.budget_consumed)}
            tone="plain"
            sub="Reserved rupees not yet released."
            note={
              <>
                Budget is reserved at decision time and committed by execution, so a crash between
                the two holds the money rather than losing it. Nothing is double-spent and nothing
                vanishes — see <code className="mono">budget_ledger</code>.
              </>
            }
          />
        </div>

        <Card
          title="Incremental lift by segment"
          note={
            <>
              Point estimate with its 95% interval. The only question the chart is built to answer is
              whether an interval clears zero, so zero is the one hard line on it. A segment whose
              whisker crosses zero is marked <span className="mono">n.s.</span> and is not claimed as
              a result.
              <br />
              <br />
              {/* Derived, not asserted. This paragraph used to state
                  "no individual segment reaches significance" as static prose,
                  which would have kept saying so on a run where one did. */}
              <strong>
                {significantSegments === 0
                  ? 'No individual segment reaches significance here'
                  : `${significantSegments} of ${(r.lifts || []).length} segments reach significance here`}
                , and the pooled estimate {o.is_significant ? 'does' : 'does not'}.
              </strong>{' '}
              {significantSegments === 0 && o.is_significant ? (
                <>
                  That is not a contradiction — splitting the same evidence four ways widens every
                  interval, and the pool has the sample size that none of the parts do. The honest
                  reading is that the effect is demonstrable overall and that per-segment
                  differences are not yet separable from noise. Claiming the winning segment on this
                  evidence would be exactly the error the holdout exists to prevent.
                </>
              ) : (
                <>
                  Splitting the same evidence four ways widens every interval, so a segment reading{' '}
                  <span className="mono">n.s.</span> is a statement about sample size, not about the
                  effect being absent.
                </>
              )}
            </>
          }
        >
          <LiftBars lifts={r.lifts} />
        </Card>

        <div className="grid g2">
          <Card
            title="Why the system did nothing"
            note={
              <>
                {n(totalWithheld)} decisions ended in no action. This is a measured distribution, not
                a claim — a system that never declines to act is not being safe, it is being
                unconstrained.
              </>
            }
          >
            {reasons.length ? (
              <table className="t">
                <thead>
                  <tr>
                    <th>Reason</th>
                    <th className="n">Decisions</th>
                    <th className="n">Share</th>
                  </tr>
                </thead>
                <tbody>
                  {reasons.map((x) => (
                    <tr key={x.no_action_reason}>
                      <td>{titleCase(x.no_action_reason)}</td>
                      <td className="n">{n(x.n)}</td>
                      <td className="n dim">{pct(x.n / totalWithheld, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <Empty>No no_action decisions recorded.</Empty>
            )}
          </Card>

          <Card
            title="Reproducibility"
            note={
              <>
                Reproducibility is ranked above availability in this design, and it is{' '}
                <strong>checked, not asserted</strong>. The LLM's candidate offers were generated
                once against a real model and committed as a cassette, so a clone with no API key
                replays the exact run byte-for-byte.
              </>
            }
          >
            <div className="body-sm" style={{ marginBottom: 12 }}>
              Run the replay twice into separate databases and diff the posteriors:
            </div>
            <pre
              className="mono"
              style={{
                background: 'var(--canvas)',
                border: '1px solid var(--hairline)',
                borderRadius: 'var(--r-md)',
                padding: 'var(--s-md)',
                overflowX: 'auto',
                fontSize: 12,
                lineHeight: 1.7,
                color: 'var(--body)',
                margin: 0,
              }}
            >
              <span className="dim">$</span> revenew demo --db a.db --harness-db a_h.db{'\n'}
              <span className="dim">$</span> revenew demo --db b.db --harness-db b_h.db{'\n'}
              <span className="dim"># same seed ⇒ byte-identical posteriors</span>
            </pre>
            <div className="card-note">
              Run <code className="mono">python -m pytest -q</code> for the full suite. The one
              skipped test is the only one that hits the live Razorpay API — it is opt-in, because it
              creates a real payment link every time it runs and a suite should not have side effects
              outside itself.
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
