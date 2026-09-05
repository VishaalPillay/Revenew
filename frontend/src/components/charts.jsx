import { useMemo, useRef, useState } from 'react'
import { compact, dec, heatVar, n, pct, titleCase } from '../lib/util.js'

/* Every chart here is inline SVG on a shared viewBox, scaled by CSS. Hover
 * coordinates are converted through the SVG's own CTM rather than from
 * bounding-box arithmetic, so the crosshair stays exact at any width without
 * a resize observer.
 *
 * Colour does almost no work in this file, by design: the design system
 * forbids a second brand hue, so identity is carried by position, direct
 * labels, and a single yellow against a recessive grey baseline. Nothing here
 * needs a categorical palette, which is also why nothing here can fail a
 * colour-vision check.
 */

const W = 820
const PAD = { t: 14, r: 18, b: 30, l: 52 }

function useSvgPointer(ref) {
  const [pt, setPt] = useState(null)
  const onMove = (e) => {
    const svg = ref.current
    if (!svg) return
    const ctm = svg.getScreenCTM()
    if (!ctm) return
    const p = svg.createSVGPoint()
    p.x = e.clientX
    p.y = e.clientY
    const local = p.matrixTransform(ctm.inverse())
    setPt({ x: local.x, y: local.y, clientX: e.clientX, clientY: e.clientY })
  }
  return { pt, onMove, onLeave: () => setPt(null) }
}

function Tooltip({ at, children }) {
  if (!at) return null
  // Flip across the pointer near the right edge so the tip never runs off.
  const flip = at.clientX > window.innerWidth - 280
  return (
    <div
      className="tip"
      style={{
        left: flip ? at.clientX - 12 : at.clientX + 14,
        top: at.clientY - 12,
        transform: flip ? 'translateX(-100%)' : 'none',
      }}
    >
      {children}
    </div>
  )
}

function ticks(min, max, count = 4) {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min]
  const span = max - min
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const out = []
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v)
  return out
}

/* ------------------------------------------------------------- line chart --
 * One y-axis, always. Two series are allowed only when they share a unit —
 * here both are cumulative rupees of regret — and the second is styled as a
 * recessive dashed baseline so it reads as context, not as a rival subject.
 */
export function LineChart({
  series,
  height = 260,
  yLabel,
  xLabel,
  format = compact,
  playhead = null,
  yFrom = 0,
  // Pin the axes to the full run rather than to the points supplied so far.
  // Without this, a chart being drawn during playback rescales on every frame
  // and the line appears to stretch in place instead of advancing across the
  // plot — the animation reads as noise rather than as progress.
  xDomain = null,
  yDomain = null,
}) {
  const ref = useRef(null)
  const { pt, onMove, onLeave } = useSvgPointer(ref)
  const H = height

  const live = series.filter((s) => s.points && s.points.length > 0)

  const scale = useMemo(() => {
    if (!live.length) return null
    let xMin = Infinity
    let xMax = -Infinity
    let yMax = -Infinity
    let yMin = Infinity
    for (const s of live) {
      for (const p of s.points) {
        if (p.x < xMin) xMin = p.x
        if (p.x > xMax) xMax = p.x
        if (p.y > yMax) yMax = p.y
        if (p.y < yMin) yMin = p.y
      }
    }
    if (yFrom !== null && yFrom !== undefined) yMin = Math.min(yMin, yFrom)
    if (xDomain) [xMin, xMax] = xDomain
    if (yDomain) [yMin, yMax] = yDomain
    if (xMax === xMin) xMax = xMin + 1
    if (yMax === yMin) yMax = yMin + 1
    const x = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r)
    const y = (v) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b)
    return { x, y, xMin, xMax, yMin, yMax }
  }, [live, H, yFrom, xDomain, yDomain])

  if (!scale) return <div className="empty">No data to plot.</div>

  const yTicks = ticks(scale.yMin, scale.yMax, 4)
  const xTicks = ticks(scale.xMin, scale.xMax, 5)

  const path = (points) =>
    points.map((p, i) => `${i ? 'L' : 'M'}${scale.x(p.x).toFixed(2)} ${scale.y(p.y).toFixed(2)}`).join(' ')

  // Nearest point on the primary series to the cursor, for the crosshair.
  let focus = null
  if (pt && live[0]) {
    const pts = live[0].points
    let best = null
    let bestD = Infinity
    for (const p of pts) {
      const d = Math.abs(scale.x(p.x) - pt.x)
      if (d < bestD) {
        bestD = d
        best = p
      }
    }
    if (best && bestD < 60) focus = best
  }

  return (
    <>
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        style={{ height }}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        role="img"
        aria-label={`${yLabel || 'value'} against ${xLabel || 'index'}`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={PAD.l} x2={W - PAD.r} y1={scale.y(t)} y2={scale.y(t)} />
            <text x={PAD.l - 8} y={scale.y(t)} textAnchor="end" dominantBaseline="middle">
              {format(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={scale.x(t)} y={H - PAD.b + 16} textAnchor="middle">
            {compact(t)}
          </text>
        ))}
        <line className="axis-line" x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />

        {playhead !== null && playhead >= scale.xMin && playhead <= scale.xMax && (
          <line
            className="playhead"
            x1={scale.x(playhead)}
            x2={scale.x(playhead)}
            y1={PAD.t}
            y2={H - PAD.b}
          />
        )}

        {/* Baseline series painted first so the subject sits above it. */}
        {live
          .map((s, i) => ({ s, i }))
          .sort((a, b) => (a.s.baseline === b.s.baseline ? 0 : a.s.baseline ? -1 : 1))
          .map(({ s, i }) => (
            <path
              key={i}
              className={`series ${s.baseline ? 'series-baseline' : 'series-primary'}`}
              d={path(s.points)}
            />
          ))}

        {focus && (
          <>
            <line
              className="ref-line"
              x1={scale.x(focus.x)}
              x2={scale.x(focus.x)}
              y1={PAD.t}
              y2={H - PAD.b}
            />
            <circle
              cx={scale.x(focus.x)}
              cy={scale.y(focus.y)}
              r="4.5"
              fill="var(--series-primary)"
              stroke="var(--surface-card)"
              strokeWidth="2"
            />
          </>
        )}

        {/* Left-aligned inside the plot, not over the axis gutter, where it
            would collide with the topmost tick label. */}
        {yLabel && (
          <text x={PAD.l + 4} y={PAD.t + 2} textAnchor="start" fill="var(--muted-soft)">
            {yLabel}
          </text>
        )}
        {xLabel && (
          <text x={W - PAD.r} y={H - 4} textAnchor="end">
            {xLabel}
          </text>
        )}
      </svg>

      {live.length > 1 && (
        <div className="legend">
          {live.map((s, i) => (
            <span className="legend-item" key={i}>
              <span
                className="swatch"
                style={{
                  background: s.baseline ? 'var(--series-baseline)' : 'var(--series-primary)',
                }}
              />
              {s.label}
            </span>
          ))}
        </div>
      )}

      {focus && (
        <Tooltip at={pt}>
          <div>
            <span className="k">{xLabel || 'x'} </span>
            <span className="v">{n(focus.x)}</span>
          </div>
          <div>
            <span className="k">{yLabel || 'y'} </span>
            <span className="v">{format(focus.y)}</span>
          </div>
        </Tooltip>
      )}
    </>
  )
}

/* --------------------------------------------------------- learning bars --
 * "What share of decisions landed on the truth-optimal action" per slice of
 * the run, against the 20% line that five families make chance. The reference
 * line is what turns a bar chart into an argument: without it, 76% is just a
 * number.
 */
export function LearningBars({ buckets, height = 240 }) {
  const ref = useRef(null)
  const { pt, onMove, onLeave } = useSvgPointer(ref)
  const [hoverIdx, setHoverIdx] = useState(null)
  const H = height

  if (!buckets || !buckets.length) return <div className="empty">No exported learning curve.</div>

  const yMax = 1
  const chance = 0.2
  const innerW = W - PAD.l - PAD.r
  const bw = innerW / buckets.length
  const y = (v) => H - PAD.b - (v / yMax) * (H - PAD.t - PAD.b)

  return (
    <>
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        style={{ height }}
        onMouseMove={(e) => {
          onMove(e)
          const svg = ref.current
          const ctm = svg?.getScreenCTM()
          if (!ctm) return
          const p = svg.createSVGPoint()
          p.x = e.clientX
          p.y = e.clientY
          const local = p.matrixTransform(ctm.inverse())
          const idx = Math.floor((local.x - PAD.l) / bw)
          setHoverIdx(idx >= 0 && idx < buckets.length ? idx : null)
        }}
        onMouseLeave={() => {
          onLeave()
          setHoverIdx(null)
        }}
        role="img"
        aria-label="Share of decisions landing on the truth-optimal action, by slice of the run"
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grid-line" x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} />
            <text x={PAD.l - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">
              {pct(t, 0)}
            </text>
          </g>
        ))}

        {buckets.map((b, i) => {
          const h = Math.max(1, H - PAD.b - y(b.optimal_rate))
          const isHot = b.optimal_rate > chance
          return (
            <rect
              key={i}
              x={PAD.l + i * bw + 2}
              y={y(b.optimal_rate)}
              width={Math.max(1, bw - 4)}
              height={h}
              rx="4"
              fill={isHot ? 'var(--primary)' : 'var(--hairline-strong)'}
              opacity={hoverIdx === null || hoverIdx === i ? 1 : 0.55}
            />
          )
        })}

        {/* Chance, drawn over the bars so it is never obscured. The label sits
            at the left, where the early bars are short and cannot collide with
            it — anchored right it lands squarely on the tallest bars. */}
        <line className="ref-line" x1={PAD.l} x2={W - PAD.r} y1={y(chance)} y2={y(chance)} />
        <text x={PAD.l + 6} y={y(chance) - 7} textAnchor="start" fill="var(--muted)">
          chance = 20% (5 families)
        </text>

        <line className="axis-line" x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />
        <text x={W - PAD.r} y={H - 4} textAnchor="end">
          bandit decisions →
        </text>
      </svg>

      {hoverIdx !== null && buckets[hoverIdx] && (
        <Tooltip at={pt}>
          <div>
            <span className="k">after </span>
            <span className="v">{n(buckets[hoverIdx].decision_index)}</span>
            <span className="k"> decisions</span>
          </div>
          <div>
            <span className="k">optimal </span>
            <span className="v">{pct(buckets[hoverIdx].optimal_rate)}</span>
          </div>
          <div>
            <span className="k">regret/decision </span>
            <span className="v">₹{dec(buckets[hoverIdx].regret_per_decision, 1)}</span>
          </div>
        </Tooltip>
      )}
    </>
  )
}

/* ------------------------------------------------------- posterior grid --
 * 4 segments x 5 families of learned conversion rate, on the single-hue
 * sequential ramp. Every cell also prints its own value, so the encoding is
 * never colour alone — which is what lets a heat map stay legible in
 * greyscale, under CVD, and in a compressed demo video.
 */
export function PosteriorGrid({ cells, cellOrder, segments, families, truth, showTruth = false }) {
  const [hover, setHover] = useState(null)
  const [at, setAt] = useState(null)

  const byKey = useMemo(() => {
    const m = new Map()
    cellOrder.forEach(([s, f], i) => m.set(`${s}|${f}`, i))
    return m
  }, [cellOrder])

  // The ramp is scaled to the observed range rather than to [0, 1]: real
  // conversion rates here sit between roughly 0.02 and 0.35, and a ramp
  // stretched to 1.0 would render the entire grid in its bottom two steps.
  const range = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const c of cells) {
      const p = c[0] / (c[0] + c[1])
      if (c[2] > 0) {
        if (p < lo) lo = p
        if (p > hi) hi = p
      }
    }
    if (!isFinite(lo)) return { lo: 0, hi: 1 }
    if (hi - lo < 1e-6) return { lo, hi: lo + 1e-6 }
    return { lo, hi }
  }, [cells])

  // The family the bandit's current belief favours per segment, ringed. This
  // is the "what would it do now" read that a table of numbers does not give.
  const bestPerSegment = useMemo(() => {
    const best = new Map()
    for (const s of segments) {
      let bf = null
      let bv = -Infinity
      for (const f of families) {
        const i = byKey.get(`${s}|${f}`)
        if (i === undefined) continue
        const c = cells[i]
        if (!c || c[2] === 0) continue
        const p = c[0] / (c[0] + c[1])
        if (p > bv) {
          bv = p
          bf = f
        }
      }
      if (bf) best.set(s, bf)
    }
    return best
  }, [cells, segments, families, byKey])

  const truthByKey = useMemo(() => {
    const m = new Map()
    for (const t of truth || []) m.set(`${t.segment}|${t.action_family}`, t.p_true)
    return m
  }, [truth])

  return (
    <>
      <div
        className="heat"
        style={{ gridTemplateColumns: `92px repeat(${families.length}, minmax(0, 1fr))` }}
      >
        <div />
        {families.map((f) => (
          <div className="heat-collabel" key={f}>
            {titleCase(f)}
          </div>
        ))}

        {segments.map((s) => (
          <Row
            key={s}
            s={s}
            families={families}
            cells={cells}
            byKey={byKey}
            range={range}
            best={bestPerSegment.get(s)}
            truthByKey={truthByKey}
            showTruth={showTruth}
            setHover={setHover}
            setAt={setAt}
          />
        ))}
      </div>

      {hover && (
        <Tooltip at={at}>
          <div className="v" style={{ marginBottom: 4 }}>
            {titleCase(hover.segment)} · {titleCase(hover.family)}
          </div>
          <div>
            <span className="k">learned p̂ </span>
            <span className="v">{dec(hover.p)}</span>
          </div>
          {hover.pTrue !== undefined && hover.pTrue !== null && (
            <div>
              <span className="k">true p* </span>
              <span className="v">{dec(hover.pTrue)}</span>
              <span className="k"> · err </span>
              <span className="v">{dec(Math.abs(hover.p - hover.pTrue))}</span>
            </div>
          )}
          <div>
            <span className="k">evidence </span>
            <span className="v">{n(hover.n)}</span>
            <span className="k"> outcomes</span>
          </div>
          <div>
            <span className="k">Beta(</span>
            <span className="v">
              {dec(hover.a, 1)}, {dec(hover.b, 1)}
            </span>
            <span className="k">)</span>
          </div>
        </Tooltip>
      )}
    </>
  )
}

function Row({ s, families, cells, byKey, range, best, truthByKey, showTruth, setHover, setAt }) {
  return (
    <>
      <div className="heat-rowlabel">{titleCase(s)}</div>
      {families.map((f) => {
        const i = byKey.get(`${s}|${f}`)
        const c = i === undefined ? null : cells[i]
        if (!c) return <div className="heat-cell empty" key={f} />
        const [a, b, obs] = c
        const p = a / (a + b)
        const t = (p - range.lo) / (range.hi - range.lo)
        const pTrue = truthByKey.get(`${s}|${f}`)
        const shown = showTruth && pTrue !== undefined ? pTrue : p
        const tShown =
          showTruth && pTrue !== undefined ? (pTrue - range.lo) / (range.hi - range.lo) : t
        return (
          <div
            key={f}
            className={`heat-cell${obs === 0 ? ' empty' : ''}${best === f ? ' best' : ''}`}
            style={{ background: heatVar(tShown, showTruth ? true : obs > 0) }}
            onMouseEnter={(e) => {
              setHover({ segment: s, family: f, p, pTrue, n: obs, a, b })
              setAt({ clientX: e.clientX, clientY: e.clientY })
            }}
            onMouseMove={(e) => setAt({ clientX: e.clientX, clientY: e.clientY })}
            onMouseLeave={() => setHover(null)}
          >
            <span className="v" style={{ color: tShown > 0.55 ? 'var(--on-primary)' : 'var(--body)' }}>
              {obs === 0 && !showTruth ? '·' : dec(shown, 2).replace(/^0/, '')}
            </span>
          </div>
        )
      })}
    </>
  )
}

/* ------------------------------------------------------- recovery dumbbell --
 * Learned rate against true rate, one row per cell. A dumbbell is the right
 * form here because the quantity of interest is the GAP, and a gap is read
 * far more accurately as a length than as two positions on separate bars.
 */
export function RecoveryDumbbell({ rows }) {
  const [hover, setHover] = useState(null)
  const [at, setAt] = useState(null)
  if (!rows || !rows.length) return <div className="empty">No exported recovery data.</div>

  const H = Math.max(220, rows.length * 19 + PAD.t + PAD.b)
  const lo = 0
  const hi = Math.max(...rows.flatMap((r) => [r.p_hat, r.p_true])) * 1.12
  const L = 168
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - PAD.r)
  const rowH = (H - PAD.t - PAD.b) / rows.length

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        style={{ height: H }}
        role="img"
        aria-label="Learned conversion rate against true conversion rate, per cell"
      >
        {ticks(lo, hi, 4).map((t) => (
          <g key={t}>
            <line className="grid-line" x1={x(t)} x2={x(t)} y1={PAD.t} y2={H - PAD.b} />
            <text x={x(t)} y={H - PAD.b + 15} textAnchor="middle">
              {dec(t, 2)}
            </text>
          </g>
        ))}

        {rows.map((r, i) => {
          const cy = PAD.t + rowH * (i + 0.5)
          const a = x(Math.min(r.p_hat, r.p_true))
          const b = x(Math.max(r.p_hat, r.p_true))
          return (
            <g
              key={`${r.segment}-${r.action_family}`}
              onMouseEnter={(e) => {
                setHover(r)
                setAt({ clientX: e.clientX, clientY: e.clientY })
              }}
              onMouseMove={(e) => setAt({ clientX: e.clientX, clientY: e.clientY })}
              onMouseLeave={() => setHover(null)}
            >
              <rect x="0" y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={L - 10} y={cy} textAnchor="end" dominantBaseline="middle">
                {r.segment} · {r.action_family.replace(/_/g, ' ')}
              </text>
              <line x1={a} x2={b} y1={cy} y2={cy} stroke="var(--hairline-strong)" strokeWidth="2" />
              {/* True value: hollow, so it reads as the target. */}
              <circle
                cx={x(r.p_true)}
                cy={cy}
                r="4"
                fill="var(--surface-card)"
                stroke="var(--muted)"
                strokeWidth="1.5"
              />
              {/* Learned value: solid yellow, the subject. */}
              <circle
                cx={x(r.p_hat)}
                cy={cy}
                r="4"
                fill="var(--primary)"
                stroke="var(--surface-card)"
                strokeWidth="1.5"
              />
            </g>
          )
        })}
        <line className="axis-line" x1={L} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} />
      </svg>

      <div className="legend">
        <span className="legend-item">
          <span
            className="swatch"
            style={{ background: 'var(--primary)', height: 9, width: 9, borderRadius: '50%' }}
          />
          learned p̂
        </span>
        <span className="legend-item">
          <span
            className="swatch"
            style={{
              background: 'var(--surface-card)',
              border: '1.5px solid var(--muted)',
              height: 9,
              width: 9,
              borderRadius: '50%',
            }}
          />
          ground truth p*
        </span>
      </div>

      {hover && (
        <Tooltip at={at}>
          <div className="v" style={{ marginBottom: 4 }}>
            {titleCase(hover.segment)} · {titleCase(hover.action_family)}
          </div>
          <div>
            <span className="k">p̂ </span>
            <span className="v">{dec(hover.p_hat)}</span>
            <span className="k"> · p* </span>
            <span className="v">{dec(hover.p_true)}</span>
          </div>
          <div>
            <span className="k">error </span>
            <span className="v">{dec(hover.p_error)}</span>
          </div>
          <div>
            <span className="k">evidence </span>
            <span className="v">{n(hover.n_observed)}</span>
          </div>
        </Tooltip>
      )}
    </>
  )
}

/* ------------------------------------------------------------- lift bars --
 * Incremental lift per segment with its 95% interval. Zero is drawn as a hard
 * axis because the only question the chart answers is "is the interval clear
 * of zero" — significance is shown as the geometry of the whisker, not as a
 * colour a reader has to decode.
 */
export function LiftBars({ lifts }) {
  const [hover, setHover] = useState(null)
  const [at, setAt] = useState(null)
  const rows = (lifts || []).filter((l) => l.n_treatment > 0 || l.n_control > 0)
  if (!rows.length) return <div className="empty">No measured segments yet.</div>

  const H = Math.max(180, rows.length * 46 + PAD.t + PAD.b)
  const lo = Math.min(0, ...rows.map((r) => r.ci_low)) * 1.15
  const hi = Math.max(0, ...rows.map((r) => r.ci_high)) * 1.15
  const L = 96
  const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - PAD.r)
  const rowH = (H - PAD.t - PAD.b) / rows.length

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        style={{ height: H }}
        role="img"
        aria-label="Incremental lift per decision by segment, with 95% confidence intervals"
      >
        {ticks(lo, hi, 5).map((t) => (
          <line key={t} className="grid-line" x1={x(t)} x2={x(t)} y1={PAD.t} y2={H - PAD.b} />
        ))}
        {ticks(lo, hi, 5).map((t) => (
          <text key={`l${t}`} x={x(t)} y={H - PAD.b + 15} textAnchor="middle">
            {compact(t)}
          </text>
        ))}

        {rows.map((r, i) => {
          const cy = PAD.t + rowH * (i + 0.5)
          const sig = r.is_significant
          return (
            <g
              key={r.segment}
              onMouseEnter={(e) => {
                setHover(r)
                setAt({ clientX: e.clientX, clientY: e.clientY })
              }}
              onMouseMove={(e) => setAt({ clientX: e.clientX, clientY: e.clientY })}
              onMouseLeave={() => setHover(null)}
            >
              <rect x="0" y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={L - 12} y={cy} textAnchor="end" dominantBaseline="middle">
                {r.segment}
              </text>
              <line
                x1={x(r.ci_low)}
                x2={x(r.ci_high)}
                y1={cy}
                y2={cy}
                stroke={sig ? 'var(--primary)' : 'var(--hairline-strong)'}
                strokeWidth="2"
                strokeLinecap="round"
                opacity={sig ? 0.85 : 1}
              />
              {[r.ci_low, r.ci_high].map((v, k) => (
                <line
                  key={k}
                  x1={x(v)}
                  x2={x(v)}
                  y1={cy - 5}
                  y2={cy + 5}
                  stroke={sig ? 'var(--primary)' : 'var(--hairline-strong)'}
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              ))}
              <circle
                cx={x(r.lift)}
                cy={cy}
                r="5"
                fill={sig ? 'var(--primary)' : 'var(--muted-soft)'}
                stroke="var(--surface-card)"
                strokeWidth="2"
              />
              <text
                x={x(Math.max(r.ci_high, r.lift)) + 10}
                y={cy}
                dominantBaseline="middle"
                fill={sig ? 'var(--primary)' : 'var(--muted)'}
              >
                {r.lift > 0 ? '+' : ''}
                {dec(r.lift, 1)}
                {sig ? '' : '  n.s.'}
              </text>
            </g>
          )
        })}

        {/* Zero: the only line that matters. */}
        <line x1={x(0)} x2={x(0)} y1={PAD.t} y2={H - PAD.b} stroke="var(--hairline-strong)" strokeWidth="1.5" />
        {/* Below the plot with the other axis labels — above it, the marker
            sits in the first row's whisker track. */}
        <text x={x(0)} y={H - PAD.b + 15} textAnchor="middle" fill="var(--body-strong)">
          0
        </text>
      </svg>

      {hover && (
        <Tooltip at={at}>
          <div className="v" style={{ marginBottom: 4 }}>
            {titleCase(hover.segment)}
          </div>
          <div>
            <span className="k">lift/decision </span>
            <span className="v">₹{dec(hover.lift, 1)}</span>
          </div>
          <div>
            <span className="k">95% CI </span>
            <span className="v">
              [{dec(hover.ci_low, 1)}, {dec(hover.ci_high, 1)}]
            </span>
          </div>
          <div>
            <span className="k">n </span>
            <span className="v">
              {n(hover.n_treatment)} treated · {n(hover.n_control)} control
            </span>
          </div>
          <div className="k">{hover.is_significant ? 'clear of zero' : 'not yet significant'}</div>
        </Tooltip>
      )}
    </>
  )
}
