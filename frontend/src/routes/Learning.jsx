import { LearningBars, LineChart, PosteriorGrid, RecoveryDumbbell } from '../components/charts.jsx'
import { Card, Empty, Loading, Stat } from '../components/ui.jsx'
import { compact, dec, n, pct, useApi } from '../lib/util.js'
import { useState } from 'react'

export default function Learning() {
  const report = useApi('/api/report')
  const theatre = useApi('/api/theatre')
  const [showTruth, setShowTruth] = useState(false)

  if (report.error) throw report.error
  if (report.loading) return <Loading what="the learning record" />

  const r = report.data
  const lc = r.learning_curve || []
  const recovery = r.posterior_recovery || []
  const first = lc[0]
  const last = lc[lc.length - 1]
  const meanErr = recovery.length
    ? recovery.reduce((a, c) => a + c.p_error, 0) / recovery.length
    : null

  const meta = theatre.data?.meta
  const lastFrame = theatre.data?.frames?.[theatre.data.frames.length - 1]
  const totalNoAction = (r.no_action_reasons || []).reduce((a, c) => a + c.n, 0)
  const allCurve = r.regret_curve_all || []
  const allRegret = allCurve.length ? allCurve[allCurve.length - 1].cumulative_regret : 0

  return (
    <div className="page wrap">
      <div className="page-head">
        <h1>Learning</h1>
        <p>
          The bandit is never told which action is best. It only ever sees whether a customer
          converted. These four charts ask, in order: did it improve, did it stop paying for
          mistakes, did it find the <em>true</em> rates, and does its final belief match reality cell
          by cell.
        </p>
      </div>

      <div className="stack">
        <div className="grid g3">
          <div className="card-yellow">
            <div className="overline">Truth-optimal action picked</div>
            <div className="stat lg" style={{ marginTop: 8 }}>
              {first && last ? `${pct(first.optimal_rate, 0)} → ${pct(last.optimal_rate, 0)}` : '—'}
            </div>
            <div className="stat-sub">
              First slice of the run against the last. Chance is 20% across five families — it starts{' '}
              <em>below</em> chance because discount-bearing families are given deliberately
              pessimistic priors.
            </div>
          </div>
          <Stat
            label="Regret per decision"
            value={
              first && last
                ? `${dec(first.regret_per_decision, 0)} → ${dec(last.regret_per_decision, 0)}`
                : '—'
            }
            sub="Rupees left on the table per decision, first slice to last."
          />
          <Stat
            label="Mean posterior error"
            value={meanErr === null ? '—' : dec(meanErr)}
            tone="plain"
            sub={
              <>
                Mean <span className="mono">|p̂ − p*|</span> across all {recovery.length} cells. The
                bandit converged on the truth, not merely on something stable.
              </>
            }
          />
        </div>

        <Card
          title="Did it learn?"
          note={
            <>
              Each bar is a slice of the run; its height is the share of decisions in that slice that
              landed on the action <strong>ground truth says is genuinely best</strong> for that
              segment. Graded in a database the runtime process cannot open. The dip below the chance
              line at the start is the cold-start prior doing its job — refusing to spend margin
              before there is evidence to justify it.
            </>
          }
        >
          <LearningBars buckets={lc} />
        </Card>

        <div className="grid g2">
          <Card
            title="Cumulative regret over the bandit's own decisions"
            note={
              <>
                The {n((r.regret_curve || []).length)}-point curve over the decisions the bandit{' '}
                <em>actually chose</em> — <strong>the flattening slope is the learning</strong>. It
                keeps rising because it is cumulative, but each later decision adds less to it.
                <br />
                <br />
                Measured instead across <em>every</em> decision, including the{' '}
                {n(totalNoAction)} the envelope forced to <code className="mono">no_action</code>{' '}
                before the bandit was ever consulted, the same run totals{' '}
                <strong>{compact(allRegret)}</strong> — but those were not choices, so a curve over
                them measures cooldown policy, not learning. They are deliberately not overlaid
                here: at eight times the scale, plotting both on one axis flattens the curve that
                actually answers the question into a line along the floor.
              </>
            }
          >
            <LineChart
              series={[
                {
                  label: 'Bandit decisions — the learning curve',
                  points: (r.regret_curve || []).map((p) => ({
                    x: p.decision_index,
                    y: p.cumulative_regret,
                  })),
                },
              ]}
              yLabel="₹ regret"
              xLabel="bandit decision #"
              format={compact}
              height={260}
            />
          </Card>

          <Card
            title="Final belief vs ground truth"
            aside={
              <div className="tabs">
                <button className="tab" aria-pressed={!showTruth} onClick={() => setShowTruth(false)}>
                  Learned
                </button>
                <button className="tab" aria-pressed={showTruth} onClick={() => setShowTruth(true)}>
                  Truth
                </button>
              </div>
            }
            note={
              <>
                Toggle between what the bandit <strong>learned</strong> and what is{' '}
                <strong>true</strong>. If the two grids look the same, that is the whole claim of
                this project in one image — and the mean error of {dec(meanErr)} is that sameness as
                a number.
              </>
            }
          >
            {theatre.loading ? (
              <Loading what="the grid" />
            ) : lastFrame && meta ? (
              <PosteriorGrid
                cells={lastFrame.cells}
                cellOrder={meta.cell_order || []}
                segments={meta.segments || []}
                families={meta.families || []}
                truth={theatre.data.truth}
                showTruth={showTruth}
              />
            ) : (
              <Empty>No exported grid.</Empty>
            )}
          </Card>
        </div>

        <Card
          title="Posterior recovery, cell by cell"
          note={
            <>
              One row per <strong>segment × action family</strong>. The hollow marker is the true
              conversion rate; the yellow marker is what the bandit learned. The bar between them is
              the error — and it is the length of those bars, not their position, that the chart is
              built to make readable.
            </>
          }
        >
          <RecoveryDumbbell rows={recovery} />
        </Card>
      </div>
    </div>
  )
}
