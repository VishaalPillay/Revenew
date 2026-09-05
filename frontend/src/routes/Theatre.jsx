import { useEffect, useMemo, useRef, useState } from 'react'
import { LineChart, PosteriorGrid } from '../components/charts.jsx'
import { Card, Empty, Loading, Stat, TracePanel } from '../components/ui.jsx'
import { compact, dec, n, pct, rs, shortDate, titleCase, useApi, usePlayback } from '../lib/util.js'

// Days per second. The default plays a 90-day run in about fifteen seconds —
// long enough to watch the grid fill, short enough to sit inside a five-minute
// demo without anyone reaching for the scrub bar.
const SPEEDS = [3, 6, 12, 24]
const DEFAULT_SPEED = 6
const TICKER_ROWS = 14

export default function Theatre() {
  const { data, error, loading } = useApi('/api/theatre')
  // The compliance card's numbers come from the measured report, not from
  // literals typed into the JSX. `useApi` caches per URL, so this is the same
  // response App.jsx already holds -- not a second round trip.
  const { data: report } = useApi('/api/report')
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [fps, setFps] = useState(DEFAULT_SPEED)
  const [selected, setSelected] = useState(null)
  const [trace, setTrace] = useState(null)
  const started = useRef(false)

  const frames = data?.frames || []
  const meta = data?.meta || {}
  const len = frames.length

  // Start from zero once the payload lands, so the first thing anyone sees is
  // an empty grid filling rather than a finished one.
  useEffect(() => {
    if (len && !started.current) {
      started.current = true
      setFrame(0)
      setPlaying(true)
    }
  }, [len])

  usePlayback({
    length: len,
    playing: playing && len > 0,
    fps,
    onFrame: (advance) =>
      setFrame((f) => {
        const next = f + advance
        if (next >= len - 1) {
          setPlaying(false)
          return len - 1
        }
        return next
      }),
  })

  // Events for the ticker: everything up to the current frame, newest last.
  // Slicing from the tail keeps this O(rows) rather than O(run).
  const events = data?.events || []
  const visible = useMemo(() => {
    if (!events.length) return []
    let hi = events.length
    for (let i = 0; i < events.length; i += 1) {
      if (events[i].f > frame) {
        hi = i
        break
      }
    }
    return events.slice(Math.max(0, hi - TICKER_ROWS), hi)
  }, [events, frame])

  useEffect(() => {
    if (!selected) {
      setTrace(null)
      return undefined
    }
    let live = true
    fetch(`/api/decisions/${selected}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((t) => live && setTrace(t))
      .catch(() => live && setTrace(null))
    return () => {
      live = false
    }
  }, [selected])

  if (loading) return <Loading what="the run timeline" />
  if (error) throw error
  if (!len) {
    return (
      <div className="page wrap">
        <Card title="Agent Theatre">
          <Empty>
            No replay run in this database yet. Run <code className="mono">revenew demo</code>, or
            restore the committed snapshot, then reload.
          </Empty>
        </Card>
      </div>
    )
  }

  const f = frames[Math.min(frame, len - 1)]
  const prev = frames[Math.max(0, frame - 1)]
  const sentToday = f.executed - prev.executed
  const rewardsToday = f.outcomes - prev.outcomes
  const compliance = report?.candidate_validity
  // Only claim a compliance figure once the report actually supplies one.
  // Previously this rendered a hardcoded 100.00% as soon as any candidate
  // existed, which would have kept claiming perfection on a database where
  // the model had genuinely misbehaved.
  const complianceKnown = f.generated > 0 && compliance?.policy_compliance_rate != null

  return (
    <div className="page wrap">
      <div className="page-head">
        <h1>Agent Theatre</h1>
        <p>
          Ninety days of a real replay, played back from <code className="mono">revenew.db</code>.
          Nothing here is simulated: every frame is the belief state rebuilt by replaying the reward
          ledger, exactly as <code className="mono">ledger/replay.py</code> rebuilds it. The grid is
          what the bandit believed on that day; it was never told which action was best.
        </p>
      </div>

      <div className="stack">
        <Transport
          frame={frame}
          len={len}
          day={f.day}
          playing={playing}
          fps={fps}
          onPlay={() => {
            if (frame >= len - 1) setFrame(0)
            setPlaying((p) => !p)
          }}
          onScrub={(v) => {
            setPlaying(false)
            setFrame(v)
          }}
          onSpeed={setFps}
          onRestart={() => {
            setFrame(0)
            setPlaying(true)
          }}
        />

        <div className="grid g4">
          <Stat
            label="Offers sent"
            value={n(f.executed)}
            size="sm"
            sub={
              sentToday > 0 ? (
                <span className="yellow">+{n(sentToday)} today</span>
              ) : (
                <span className="dim">quiet — every customer in cooldown</span>
              )
            }
          />
          <Stat
            label="Rewards returned"
            value={n(f.outcomes)}
            size="sm"
            sub={
              rewardsToday > 0 ? (
                <span className="yellow">+{n(rewardsToday)} today</span>
              ) : (
                <span className="dim">awaiting the 7-day window</span>
              )
            }
          />
          <Stat
            label="Belief error vs truth"
            value={f.mean_error === null ? '—' : dec(f.mean_error)}
            size="sm"
            tone="plain"
            sub={
              <>
                mean <span className="mono">|p̂ − p*|</span> across all 20 cells
              </>
            }
          />
          <Stat
            label="Budget consumed"
            value={compact(f.budget)}
            size="sm"
            tone="plain"
            sub={`${n(f.no_action)} decisions withheld by policy`}
          />
        </div>

        <Narration frame={f} prev={prev} sentToday={sentToday} rewardsToday={rewardsToday} />

        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.55fr) minmax(0, 1fr)' }}>
          <Card
            title="What the bandit believes"
            aside={<span className="caption dim">day {frame + 1} of {len}</span>}
            note={
              <>
                Learned conversion rate <span className="mono">p̂ = α/(α+β)</span> per{' '}
                <strong>segment × action family</strong>, rebuilt from the reward ledger as of this
                day. A <strong>ringed</strong> cell is the family that segment's evidence currently
                favours. Cells start at their cold-start prior — discount-bearing families begin
                deliberately pessimistic at Beta(1, 4) so the system does not burn margin on day one.
              </>
            }
          >
            <PosteriorGrid
              cells={f.cells}
              cellOrder={meta.cell_order || []}
              segments={meta.segments || []}
              families={meta.families || []}
              truth={data.truth}
            />
          </Card>

          <Card
            title="Offers going out"
            aside={<span className="caption dim">{shortDate(f.day)}</span>}
          >
            <div className="ticker">
              {visible.length ? (
                visible.map((e) => (
                  <button
                    key={e.id}
                    className="ticker-row"
                    onClick={() => setSelected(e.id)}
                    style={{
                      background: 'none',
                      border: 'none',
                      borderBottom: '1px solid var(--grid-line)',
                      textAlign: 'left',
                      cursor: 'pointer',
                      width: '100%',
                      color: 'inherit',
                      font: 'inherit',
                    }}
                    title="Open this decision's full trace"
                  >
                    <span className="seg">{e.segment}</span>
                    <span>
                      <span className="offer">{e.headline || '—'}</span>
                      <span className="fam">{e.family}</span>
                    </span>
                    <span className="num caption dim">{dec(e.propensity, 2)}</span>
                  </button>
                ))
              ) : (
                <div className="ticker-empty">
                  No offers on this day.
                  <br />
                  <span className="dim">
                    Cooldown holds every eligible customer until the next monthly window.
                  </span>
                </div>
              )}
            </div>
          </Card>
        </div>

        {trace && (
          <Card
            title="One decision, end to end"
            aside={
              <button className="btn btn-ghost btn-sm" onClick={() => setSelected(null)}>
                Close
              </button>
            }
          >
            <TracePanel trace={trace} />
          </Card>
        )}

        <div className="grid g2">
          <Card
            title="Cumulative regret vs the oracle"
            note={
              <>
                What a perfectly-informed oracle would have earned, minus what the bandit earned,
                totalled over the decisions it actually made. On a day axis this is a{' '}
                <strong>staircase</strong>, because actions arrive in monthly waves — and the step
                heights are the whole story. The first wave cost{' '}
                <span className="mono">{compact(frames[0].bandit_cum_regret)}</span> across{' '}
                {n(frames[0].bandit_decisions)} decisions; every later wave adds far less per
                decision. Graded against ground truth held in a database this process cannot open.
              </>
            }
          >
            <LineChart
              series={[
                {
                  label: 'Cumulative regret (₹)',
                  points: frames.slice(0, frame + 1).map((fr) => ({
                    x: fr.i,
                    y: fr.bandit_cum_regret,
                  })),
                },
              ]}
              yLabel="₹ regret"
              xLabel="day"
              playhead={frame}
              format={compact}
              height={230}
              xDomain={[0, len - 1]}
              yDomain={[0, frames[len - 1].bandit_cum_regret * 1.06]}
            />
          </Card>

          <Card
            title="Distance from the truth"
            note={
              <>
                Mean absolute error between each cell's learned rate and the rate ground truth
                declares. It falls to <strong>{dec(frames[len - 1].mean_error)}</strong> — read from
                the run's own last frame, not typed in here — and the bandit did not merely converge
                on something stable, it converged on the <em>right answer</em>. A sampler can look
                perfectly well-behaved while being confidently wrong; this is the chart that rules
                that out.
              </>
            }
          >
            <LineChart
              series={[
                {
                  label: 'mean |p̂ − p*|',
                  points: frames
                    .slice(0, frame + 1)
                    .filter((fr) => fr.mean_error !== null)
                    .map((fr) => ({ x: fr.i, y: fr.mean_error })),
                },
              ]}
              yLabel="mean |p̂ − p*|"
              xLabel="day"
              playhead={frame}
              format={(v) => dec(v, 2)}
              height={230}
              xDomain={[0, len - 1]}
              yDomain={[0, Math.max(...frames.map((fr) => fr.mean_error || 0)) * 1.1]}
            />
          </Card>
        </div>

        {complianceKnown && (
          <div className="card-yellow">
            <div className="overline">Policy compliance, live</div>
            <div className="row row-wrap" style={{ gap: 'var(--s-xl)', marginTop: 12 }}>
              <div>
                {/* Read from the measured report, never written as a literal.
                    These two figures were hardcoded to "100.00%" and "0", which
                    is the same conflation `v_candidate_compliance` and the
                    CandidateValidity split exist to prevent: a database where
                    the model DID propose an out-of-envelope candidate would
                    still have shown a perfect score, and the one number the
                    safety claim rests on would have been decoration. */}
                <div className="stat">{pct(compliance?.policy_compliance_rate, 2)}</div>
                <div className="stat-sub">
                  {n(f.generated)} candidates composed so far ·{' '}
                  <strong>{n(compliance?.policy_violations)}</strong> illegal offers proposed
                </div>
              </div>
              <div style={{ maxWidth: '46ch' }} className="body-sm">
                The envelope is enforced twice from one rule table — injected into the prompt and
                re-applied programmatically to every candidate returned. A model error can produce a
                suboptimal <em>legal</em> action. It cannot produce an illegal one.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function Transport({ frame, len, day, playing, fps, onPlay, onScrub, onSpeed, onRestart }) {
  const done = frame >= len - 1
  return (
    <div className="transport">
      <button className="btn btn-primary btn-icon" onClick={onPlay} title={playing ? 'Pause' : 'Play'}>
        {playing ? '❚❚' : done ? '↺' : '▶'}
      </button>
      <div style={{ minWidth: 128 }}>
        <div className="num title-md ink">{shortDate(day)}</div>
        <div className="caption dim">
          day {frame + 1} / {len}
        </div>
      </div>
      <input
        className="scrub"
        type="range"
        min={0}
        max={Math.max(0, len - 1)}
        value={frame}
        onChange={(e) => onScrub(Number(e.target.value))}
        aria-label="Scrub through the run"
      />
      <div className="speeds">
        {SPEEDS.map((s) => (
          <button
            key={s}
            className="speed"
            aria-pressed={fps === s}
            onClick={() => onSpeed(s)}
            title={`${s} days per second`}
          >
            {s}d/s
          </button>
        ))}
      </div>
      <button className="btn btn-ghost btn-sm" onClick={onRestart}>
        Restart
      </button>
    </div>
  )
}

/* The caption reads the frame rather than the calendar. Hard-coding "day 31 is
 * the second wave" would be a lie the moment anyone re-runs the replay with a
 * different seed or horizon; deriving it from the deltas keeps the narration
 * true for whatever run is actually loaded. */
function Narration({ frame, prev, sentToday, rewardsToday }) {
  let text
  if (frame.i === 0) {
    text = (
      <>
        <strong>Day one.</strong> Cooldown is empty, so every eligible customer is actionable at
        once — {n(sentToday)} offers go out in a single wave. The grid is still all prior: the
        system has beliefs, but no evidence.
      </>
    )
  } else if (sentToday > 200) {
    text = (
      <>
        <strong>The cooldown lifts.</strong> {n(sentToday)} offers go out today in one burst. This is
        what a one-offer-per-customer-per-month policy looks like from the inside — the system is
        not idle between waves, it is <em>forbidden</em> from acting.
      </>
    )
  } else if (rewardsToday > 200) {
    text = (
      <>
        <strong>Evidence arrives.</strong> {n(rewardsToday)} attribution windows close today. Watch
        the grid: this is the moment belief stops being prior and starts being measurement.
      </>
    )
  } else if (sentToday === 0 && rewardsToday === 0) {
    text = (
      <>
        <strong>A quiet day.</strong> Nothing sent, nothing returned. {n(frame.no_action)} decisions
        so far have been withheld by policy rather than by choice — that number is a merchant
        setting, not a model failure.
      </>
    )
  } else if (rewardsToday > 0) {
    text = (
      <>
        Rewards trickling in — {n(rewardsToday)} attribution{' '}
        {rewardsToday === 1 ? 'window' : 'windows'} closed today. Feedback runs seven days behind the
        action, which is why a thirty-day run ends before its own evidence comes back.
      </>
    )
  } else {
    // Only claim movement when the displayed figures actually differ.
    // "0.031, down from 0.031" is the kind of line that makes a careful
    // viewer stop trusting everything else on the page.
    const moved =
      frame.mean_error !== null &&
      prev.mean_error !== null &&
      dec(frame.mean_error) !== dec(prev.mean_error)
    text = (
      <>
        {n(sentToday)} offers out today. Belief error{' '}
        {moved ? (
          <>
            is now <span className="mono yellow">{dec(frame.mean_error)}</span>, down from{' '}
            <span className="mono">{dec(prev.mean_error)}</span>.
          </>
        ) : (
          <>
            holds at <span className="mono yellow">{dec(frame.mean_error)}</span> — no new evidence
            has landed to move it.
          </>
        )}
      </>
    )
  }
  return (
    <div className="card flat" style={{ padding: '2px 0' }}>
      <div className="body-sm" style={{ color: 'var(--body)' }}>
        {text}
      </div>
    </div>
  )
}
