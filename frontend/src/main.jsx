import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

// A bare "#" or "#/" lands on the theatre. Setting it explicitly on first load
// means the address bar always names the route, so a link copied off a demo
// machine reopens where the demo was.
if (!window.location.hash || window.location.hash === '#' || window.location.hash === '#/') {
  window.location.hash = '#/theatre'
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
