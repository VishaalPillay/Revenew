import { useCallback, useEffect, useRef, useState } from 'react'

/* ---------------------------------------------------------------- routing --
 * Hash routing, deliberately. The console is served by the same FastAPI
 * process that serves the webhook receiver and the read API, and history-API
 * routing would need a catch-all rule there that shadows any future route
 * added under a path the SPA also claims. A hash never reaches the server, so
 * deep links and refreshes work with zero server involvement and no rule that
 * can rot.
 */
export const ROUTES = [
  { path: 'theatre', label: 'Theatre' },
  { path: 'decisions', label: 'Decisions' },
  { path: 'learning', label: 'Learning' },
  { path: 'proof', label: 'Proof' },
]

function parseHash() {
  const raw = window.location.hash.replace(/^#\/?/, '')
  const [path, query] = raw.split('?')
  return {
    path: path || 'theatre',
    params: new URLSearchParams(query || ''),
  }
}

export function useRoute() {
  const [route, setRoute] = useState(parseHash)
  useEffect(() => {
    const onChange = () => setRoute(parseHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function navigate(path, params) {
  const q = params ? `?${new URLSearchParams(params)}` : ''
  window.location.hash = `#/${path}${q}`
}

/* -------------------------------------------------------------------- api --
 * Every route reads from `revenew/api/read.py`. Responses are cached per URL
 * for the life of the page: the payloads are immutable snapshots of a
 * finished replay, so refetching them on each navigation would only add
 * latency between tabs during a demo.
 */
const cache = new Map()

export function useApi(url) {
  const [state, setState] = useState(() =>
    cache.has(url)
      ? { data: cache.get(url), error: null, loading: false }
      : { data: null, error: null, loading: true },
  )

  useEffect(() => {
    if (cache.has(url)) {
      setState({ data: cache.get(url), error: null, loading: false })
      return
    }
    let live = true
    setState({ data: null, error: null, loading: true })
    fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`)
        return r.json()
      })
      .then((data) => {
        cache.set(url, data)
        if (live) setState({ data, error: null, loading: false })
      })
      .catch((error) => {
        if (live) setState({ data: null, error, loading: false })
      })
    return () => {
      live = false
    }
  }, [url])

  return state
}

/* ------------------------------------------------------------- formatting --
 * The run is denominated in rupees and the audience is Indian, so large
 * figures use the Indian digit grouping (1,55,701 not 155,701). Getting this
 * wrong is a small thing that reads as carelessness to exactly the people
 * being demoed to.
 */
const inr = new Intl.NumberFormat('en-IN')

export const n = (v) => (v === null || v === undefined ? '—' : inr.format(Math.round(v)))

export const rs = (v) => (v === null || v === undefined ? '—' : `₹${inr.format(Math.round(v))}`)

export function compact(v) {
  if (v === null || v === undefined) return '—'
  const a = Math.abs(v)
  if (a >= 1e7) return `${(v / 1e7).toFixed(a >= 1e8 ? 0 : 1)}Cr`
  if (a >= 1e5) return `${(v / 1e5).toFixed(a >= 1e6 ? 0 : 1)}L`
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`
  return inr.format(Math.round(v))
}

export const pct = (v, digits = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`

export const dec = (v, digits = 3) =>
  v === null || v === undefined ? '—' : v.toFixed(digits)

export const titleCase = (s) =>
  (s || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

export function shortDate(iso) {
  if (!iso) return '—'
  const d = new Date(`${iso}T00:00:00Z`)
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' })
}

/* ------------------------------------------------------------ heat colour --
 * Sequential ramp lookup. Steps come from styles.css so the palette lives in
 * exactly one place; this only decides which step a value lands on.
 * `t` is expected in [0, 1].
 */
const HEAT_STEPS = 8

export function heatVar(t, hasEvidence) {
  if (!hasEvidence) return 'var(--heat-0)'
  const clamped = Math.max(0, Math.min(1, t))
  // Step 0 is reserved for "no evidence", so real values start at step 1 and
  // a genuinely zero rate still reads as data rather than as absence.
  const step = 1 + Math.round(clamped * (HEAT_STEPS - 2))
  return `var(--heat-${step})`
}

/* ------------------------------------------------------------- animation --
 * A frame-rate-independent playback clock. `onTick` is called with the number
 * of frames that should have elapsed, so a slow paint drops frames instead of
 * silently slowing the run down — a 90-day replay stays 90 days long whether
 * the laptop is idle or busy.
 */
export function usePlayback({ length, playing, fps, onFrame }) {
  const raf = useRef(0)
  const acc = useRef(0)
  const last = useRef(0)
  const cb = useRef(onFrame)
  cb.current = onFrame

  useEffect(() => {
    if (!playing || length <= 0) return undefined
    last.current = performance.now()
    acc.current = 0

    const loop = (now) => {
      const dt = now - last.current
      last.current = now
      acc.current += dt
      const step = 1000 / fps
      let advanced = 0
      while (acc.current >= step) {
        acc.current -= step
        advanced += 1
      }
      if (advanced) cb.current(advanced)
      raf.current = requestAnimationFrame(loop)
    }
    raf.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, fps, length])
}

/* Pointer tracking for chart tooltips, shared by every chart so the hover
 * layer behaves identically everywhere. */
export function useHover() {
  const [hover, setHover] = useState(null)
  const clear = useCallback(() => setHover(null), [])
  return { hover, setHover, clear }
}
