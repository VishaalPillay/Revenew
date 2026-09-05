import { Component } from 'react'
import { ErrorBox, Footer, Nav } from './components/ui.jsx'
import Decisions from './routes/Decisions.jsx'
import Learning from './routes/Learning.jsx'
import Proof from './routes/Proof.jsx'
import Theatre from './routes/Theatre.jsx'
import { useApi, useRoute } from './lib/util.js'

/* Routes throw their fetch errors rather than each rendering their own failure
 * state, so every failure -- a missing database, an empty run, a process that
 * did not start -- surfaces through one box with one recovery instruction.
 * During a live demo the worst outcome is a blank screen with no explanation. */
class Boundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidUpdate(prev) {
    if (prev.routeKey !== this.props.routeKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  render() {
    if (this.state.error) return <ErrorBox error={this.state.error} />
    return this.props.children
  }
}

export default function App() {
  const { path, params } = useRoute()
  // Nav shows the run id, and it is the cheapest call on the API -- the
  // heavier /api/theatre payload is cached by the same hook, so the theatre
  // route does not pay for this twice.
  const { data: report } = useApi('/api/report')
  const { data: theatre } = useApi('/api/theatre')

  let view
  if (path === 'decisions') view = <Decisions initialId={params.get('id')} />
  else if (path === 'learning') view = <Learning />
  else if (path === 'proof') view = <Proof />
  else view = <Theatre />

  return (
    <div className="shell">
      <Nav path={path === '' ? 'theatre' : path} runId={report?.run_id} />
      <Boundary routeKey={path}>{view}</Boundary>
      <Footer meta={theatre?.meta || { run_id: report?.run_id }} />
    </div>
  )
}
