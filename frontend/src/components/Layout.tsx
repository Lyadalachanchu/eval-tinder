import { useState } from 'react'
import { Link, Outlet } from 'react-router-dom'
import { errorMessage } from '../api'
import { useHealth } from '../hooks/useHealth'
import { defaultApiBase, getApiBaseOverride, getToken, setApiBaseOverride, setToken } from '../settings'
import { Badge } from './Badge'

function SettingsPanel({ onClose }: { onClose: () => void }) {
  const [token, setTokenState] = useState(getToken())
  const [base, setBase] = useState(getApiBaseOverride())
  const [saved, setSaved] = useState(false)

  const save = () => {
    setToken(token)
    setApiBaseOverride(base)
    setSaved(true)
  }

  return (
    <div className="settings-panel" role="dialog" aria-label="Settings">
      <h3>Settings</h3>
      <label className="field">
        <span>Bearer token (optional)</span>
        <input
          type="password"
          value={token}
          onChange={(e) => {
            setTokenState(e.target.value)
            setSaved(false)
          }}
          placeholder="Sent as Authorization: Bearer …"
          autoComplete="off"
        />
        <span className="muted small">Required when the backend is configured with API_TOKEN. Stored only in this browser.</span>
      </label>
      <label className="field">
        <span>API base override (optional)</span>
        <input
          type="text"
          value={base}
          onChange={(e) => {
            setBase(e.target.value)
            setSaved(false)
          }}
          placeholder={defaultApiBase()}
        />
        <span className="muted small">Default: {defaultApiBase()} (dev proxy or VITE_API_BASE).</span>
      </label>
      <div className="row gap">
        <button type="button" className="btn btn-primary" onClick={save}>
          Save
        </button>
        <button type="button" className="btn" onClick={onClose}>
          Close
        </button>
        {saved ? <span className="muted small">Saved. Reload the page to apply everywhere.</span> : null}
      </div>
    </div>
  )
}

export function Layout() {
  const { health, error } = useHealth()
  const [settingsOpen, setSettingsOpen] = useState(false)

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          eval-tinder
        </Link>
        <nav className="topnav">
          <Link to="/">Projects</Link>
        </nav>
        <div className="topbar-right">
          {health?.simulated ? (
            <Badge kind="simulated" title="LLM_PROVIDER=fake: every machine verdict is scripted, not a real model">
              SIMULATED MODEL: predictions come from a scripted fake
            </Badge>
          ) : health ? (
            <span className="muted small">
              {health.llm_provider}
              {health.grader_model ? ` · ${health.grader_model}` : ''}
            </span>
          ) : error ? (
            <span className="muted small" title={errorMessage(error)}>
              backend unreachable
            </span>
          ) : null}
          <button type="button" className="btn btn-small" onClick={() => setSettingsOpen((o) => !o)}>
            Settings
          </button>
        </div>
      </header>
      {settingsOpen ? <SettingsPanel onClose={() => setSettingsOpen(false)} /> : null}
      <main className="content">
        <Outlet />
      </main>
    </div>
  )
}
