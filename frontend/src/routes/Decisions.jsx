import { useEffect, useMemo, useState } from 'react'
import { Card, Empty, Loading, TracePanel } from '../components/ui.jsx'
import { dec, n, titleCase, useApi } from '../lib/util.js'

const STATUSES = ['executed', 'no_action', 'pending']

export default function Decisions({ initialId }) {
  const [status, setStatus] = useState('executed')
  const [segment, setSegment] = useState('')
  const [limit, setLimit] = useState(100)
  const [selected, setSelected] = useState(initialId || null)
  const [trace, setTrace] = useState(null)
  const [traceErr, setTraceErr] = useState(null)

  const qs = useMemo(() => {
    const p = new URLSearchParams({ limit: String(limit) })
    if (status) p.set('status', status)
    if (segment) p.set('segment', segment)
    return p.toString()
  }, [status, segment, limit])

  const { data, error, loading } = useApi(`/api/decisions?${qs}`)
  const rows = data?.decisions || []

  // Select the first row automatically so the trace panel is never empty on
  // arrival -- an explorer whose detail pane starts blank reads as broken.
  useEffect(() => {
    if (!selected && rows.length) setSelected(rows[0].decision_id)
  }, [rows, selected])

  useEffect(() => {
    if (!selected) return undefined
    let live = true
    setTraceErr(null)
    fetch(`/api/decisions/${selected}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${r.status} — no such decision`)
        return r.json()
      })
      .then((t) => live && setTrace(t))
      .catch((e) => {
        if (live) {
          setTrace(null)
          setTraceErr(e)
        }
      })
    return () => {
      live = false
    }
  }, [selected])

  if (error) throw error

  return (
    <div className="page wrap">
      <div className="page-head">
        <h1>Decisions</h1>
        <p>
          Every decision the system made, and for each one the complete audit trail: the envelope it
          was bound by, every candidate the model proposed, the validator's verdict on each, which
          one the bandit drew and at what propensity, and what the customer did next. This is the
          same <code className="mono">get_decision_trace</code> that{' '}
          <code className="mono">revenew trace</code> prints.
        </p>
      </div>

      <div className="row row-wrap" style={{ marginBottom: 'var(--s-md)', gap: 'var(--s-md)' }}>
        <div className="tabs">
          {STATUSES.map((s) => (
            <button
              key={s}
              className="tab"
              aria-pressed={status === s}
              onClick={() => {
                setStatus(s)
                setSelected(null)
              }}
            >
              {titleCase(s)}
            </button>
          ))}
        </div>
        <select
          className="input"
          value={segment}
          onChange={(e) => {
            setSegment(e.target.value)
            setSelected(null)
          }}
          aria-label="Filter by segment"
        >
          <option value="">All segments</option>
          {['new', 'active', 'lapsing', 'dormant'].map((s) => (
            <option key={s} value={s}>
              {titleCase(s)}
            </option>
          ))}
        </select>
        <select
          className="input"
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          aria-label="Row limit"
        >
          {[50, 100, 250, 500].map((l) => (
            <option key={l} value={l}>
              {l} rows
            </option>
          ))}
        </select>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.05fr)' }}>
        <Card
          title={`${rows.length} decisions`}
          aside={<span className="caption dim">newest first</span>}
        >
          {loading ? (
            <Loading what="decisions" />
          ) : rows.length ? (
            <div className="table-scroll" style={{ maxHeight: 620, overflowY: 'auto' }}>
              <table className="t">
                <thead>
                  <tr>
                    <th>Segment</th>
                    <th>Action</th>
                    <th className="n">Propensity</th>
                    <th>When</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr
                      key={r.decision_id}
                      className={`clickable${selected === r.decision_id ? ' is-selected' : ''}`}
                      onClick={() => setSelected(r.decision_id)}
                    >
                      <td>{titleCase(r.segment)}</td>
                      <td className="mono" style={{ fontSize: 12 }}>
                        {r.action_family || (
                          <span className="dim">{r.no_action_reason || 'no_action'}</span>
                        )}
                      </td>
                      <td className="n">{r.propensity === null ? '—' : dec(r.propensity, 2)}</td>
                      <td className="mono dim" style={{ fontSize: 11.5 }}>
                        {String(r.created_at).slice(0, 10)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No decisions match these filters.</Empty>
          )}
        </Card>

        <Card title="Full trace">
          {traceErr ? (
            <Empty>{String(traceErr.message)}</Empty>
          ) : trace ? (
            <TracePanel trace={trace} />
          ) : (
            <Loading what="the trace" />
          )}
        </Card>
      </div>
    </div>
  )
}
