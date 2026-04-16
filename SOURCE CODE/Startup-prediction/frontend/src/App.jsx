import { useState, useEffect, useCallback } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:5000'

// ──────────────────────────────────────
// Sub-components
// ──────────────────────────────────────

function HealthBadge({ status }) {
  if (!status) return <div className="health-badge offline">● Checking...</div>
  return (
    <div className={`health-badge ${status.models_loaded ? 'online' : 'error'}`}>
      {status.models_loaded
        ? `● Online  |  ${status.features_count} Features  |  Acc: ${status.model_metadata?.accuracy ?? 'N/A'}%`
        : '● Models Not Loaded'}
      {status.retraining_in_progress && <span className="retrain-pulse"> ↻ Retraining...</span>}
    </div>
  )
}

function ConfidenceGauge({ percent, prediction }) {
  const radius = 70
  const stroke = 10
  const normalizedRadius = radius - stroke / 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (percent / 100) * circumference
  const color = prediction === 'Success' ? '#00f2fe' : '#ff0844'

  return (
    <div className="gauge-wrapper">
      <svg height={radius * 2} width={radius * 2} style={{ transform: 'rotate(-90deg)' }}>
        <circle stroke="#1e293b" fill="transparent" strokeWidth={stroke} r={normalizedRadius} cx={radius} cy={radius} />
        <circle
          stroke={color} fill="transparent" strokeWidth={stroke}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          r={normalizedRadius} cx={radius} cy={radius}
          style={{ transition: 'stroke-dashoffset 1s ease' }}
        />
      </svg>
      <div className="gauge-label" style={{ color }}>
        <span className="gauge-percent">{percent}%</span>
        <span className="gauge-sub">Confidence</span>
      </div>
    </div>
  )
}

function ShapChart({ shap_values, prediction }) {
  if (!shap_values || shap_values.length === 0) return null
  const sorted = [...shap_values].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  const maxVal = Math.max(...sorted.map(s => Math.abs(s.value)), 0.001)

  return (
    <div className="shap-chart">
      <h4>Feature Impact (SHAP)</h4>
      {sorted.map((item) => {
        const pct = (Math.abs(item.value) / maxVal) * 100
        const isPositive = item.value >= 0
        const color = isPositive ? '#00f2fe' : '#ff0844'
        return (
          <div key={item.feature} className="shap-row">
            <span className="shap-label">{item.feature.replace(/_/g, ' ')}</span>
            <div className="shap-bar-bg">
              <div className="shap-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
            </div>
            <span className="shap-value" style={{ color }}>{item.value > 0 ? '+' : ''}{item.value.toFixed(4)}</span>
          </div>
        )
      })}
    </div>
  )
}

function HistoryPanel({ history }) {
  if (!history || history.length === 0)
    return <p className="empty-text">No predictions yet.</p>

  return (
    <div className="history-list">
      {history.map((item) => (
        <div key={item.id} className={`history-item ${item.prediction === 'Success' ? 'hist-success' : 'hist-failure'}`}>
          <span className="hist-badge">{item.prediction}</span>
          <span className="hist-conf">{item.ensemble_confidence}%</span>
          <span className="hist-meta">
            💰 ${Number(item.total_raised_usd).toLocaleString()}  ·
            🔄 {item.total_funding_rounds} rounds  ·
            👥 {item.total_investors} investors  ·
            🕒 {item.timestamp?.slice(0, 10)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ──────────────────────────────────────
// Main App
// ──────────────────────────────────────
function App() {
  const [formData, setFormData] = useState({
    total_funding_rounds: '',
    total_raised_usd: '',
    total_investors: '',
    startup_age: '',
    founder_bio: '',
    category_code: '',
    country_code: ''
  })

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [activeTab, setActiveTab] = useState('predict')
  const [retrainMsg, setRetrainMsg] = useState('')

  const fetchHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`)
      const data = await res.json()
      setHealth(data)
    } catch {
      setHealth(null)
    }
  }, [])

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history?limit=15`)
      const data = await res.json()
      setHistory(data.history || [])
    } catch {
      setHistory([])
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    fetchHistory()
    const interval = setInterval(fetchHealth, 10000)
    return () => clearInterval(interval)
  }, [fetchHealth, fetchHistory])

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData({ ...formData, [name]: value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.details?.join(', ') || data.error || 'Server error')
      setResult(data)
      fetchHistory()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRetrain = async () => {
    setRetrainMsg('Sending retrain request...')
    try {
      const res = await fetch(`${API_BASE}/api/retrain`, { method: 'POST' })
      const data = await res.json()
      setRetrainMsg(data.message)
      setTimeout(fetchHealth, 3000)
    } catch {
      setRetrainMsg('Failed to contact server.')
    }
  }

  return (
    <div className="app-container">
      <div className="dashboard-wrapper">

        {/* Header */}
        <header className="dashboard-header">
          <div>
            <h1>🚀 Startup Success Predictor</h1>
            <p>Ensemble ML · Explainable AI · NLP Sentiment Analysis</p>
          </div>
          <HealthBadge status={health} />
        </header>

        {/* Tabs */}
        <div className="tab-bar">
          {['predict', 'history', 'model'].map(tab => (
            <button key={tab} className={`tab-btn ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
              {tab === 'predict' ? '🔮 Predict' : tab === 'history' ? '📋 History' : '⚙️ Model'}
            </button>
          ))}
        </div>

        {/* ── PREDICT TAB ── */}
        {activeTab === 'predict' && (
          <div className="content-grid">
            {/* Input Panel */}
            <div className="panel form-panel">
              <h2>Enter Startup Metrics</h2>
              <form onSubmit={handleSubmit} className="metrics-form">
                <div className="form-group">
                  <label>Total Funding Rounds</label>
                  <input type="number" name="total_funding_rounds" value={formData.total_funding_rounds} onChange={handleInputChange} required placeholder="e.g. 3" min="0" max="100" />
                </div>
                <div className="form-group">
                  <label>Total Raised (USD)</label>
                  <input type="number" name="total_raised_usd" value={formData.total_raised_usd} onChange={handleInputChange} required placeholder="e.g. 5000000" min="0" />
                </div>
                <div className="form-group">
                  <label>Total Investors</label>
                  <input type="number" name="total_investors" value={formData.total_investors} onChange={handleInputChange} required placeholder="e.g. 4" min="0" />
                </div>
                <div className="form-group">
                  <label>Startup Age (Years)</label>
                  <input type="number" name="startup_age" value={formData.startup_age} onChange={handleInputChange} required placeholder="e.g. 5" min="0" max="200" />
                </div>
                <div className="form-group">
                  <label>Industry / Category <span className="label-opt">(optional)</span></label>
                  <input type="text" name="category_code" value={formData.category_code} onChange={handleInputChange} placeholder="e.g. software, biotech, mobile" />
                </div>
                <div className="form-group">
                  <label>Country Code <span className="label-opt">(optional)</span></label>
                  <input type="text" name="country_code" value={formData.country_code} onChange={handleInputChange} placeholder="e.g. USA, IND, GBR" />
                </div>
                <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                  <label>Founder Bio / Mission Statement <span className="label-opt">(NLP)</span></label>
                  <textarea name="founder_bio" value={formData.founder_bio} onChange={handleInputChange}
                    placeholder="e.g. 'We are aggressively disrupting the market with innovative AI...'" rows="3" className="nlp-textarea" />
                </div>
                <button type="submit" className="predict-btn" disabled={loading} style={{ gridColumn: '1 / -1' }}>
                  {loading ? <span className="spinner" /> : '⚡ Predict Success'}
                </button>
              </form>
            </div>

            {/* Results Panel */}
            <div className="panel results-panel">
              <h2>AI Prediction Engine</h2>

              {!result && !error && !loading && (
                <div className="empty-state">
                  <div className="pulse-circle" />
                  <p>Awaiting data inputs to run prediction protocol...</p>
                </div>
              )}
              {error && <div className="error-box">⚠️ {error}</div>}

              {result && (
                <div className="prediction-results animate-fade-in">
                  <div className={`status-badge ${result.prediction === 'Success' ? 'success' : 'failure'}`}>
                    {result.prediction === 'Success' ? '✅' : '❌'} {result.prediction}
                  </div>

                  {/* Confidence Gauge */}
                  <div style={{ display: 'flex', justifyContent: 'center', margin: '1.5rem 0' }}>
                    <ConfidenceGauge percent={result.ensemble_confidence_percent} prediction={result.prediction} />
                  </div>

                  {/* RF + NN sub-bars */}
                  <div className="stat-card">
                    {[
                      { label: 'Random Forest (Trees)', pct: result.rf_confidence_score_percent },
                      { label: 'Neural Network (Deep Learning)', pct: result.nn_confidence_score_percent }
                    ].map(({ label, pct }) => (
                      <div key={label} style={{ marginTop: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '5px' }}>
                          <span>{label}</span><span>{pct}%</span>
                        </div>
                        <div className="progress-bar-bg">
                          <div className="progress-bar-fill"
                            style={{ width: `${pct}%`, backgroundColor: result.prediction === 'Success' ? '#00f2fe' : '#ff0844' }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* NLP Sentiment */}
                  <div className="xai-box nlp-box" style={{
                    marginTop: '1.5rem',
                    borderColor: result.nlp_sentiment === 'Positive' ? '#00f2fe' : result.nlp_sentiment === 'Negative' ? '#ff0844' : '#94a3b8'
                  }}>
                    <h3>NLP Analysis: <span style={{ color: result.nlp_sentiment === 'Positive' ? '#00f2fe' : result.nlp_sentiment === 'Negative' ? '#ff0844' : '#94a3b8' }}>{result.nlp_sentiment}</span></h3>
                    <p className="xai-text">{result.nlp_impact_text}</p>
                  </div>

                  {/* SHAP Chart */}
                  {result.shap_values && result.shap_values.length > 0 && (
                    <div className="xai-box" style={{ marginTop: '1.5rem' }}>
                      <ShapChart shap_values={result.shap_values} prediction={result.prediction} />
                      <p className="xai-text" style={{ marginTop: '0.5rem' }}>💡 {result.xai_explanation}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── HISTORY TAB ── */}
        {activeTab === 'history' && (
          <div className="panel history-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2>📋 Prediction History</h2>
              <button className="refresh-btn" onClick={fetchHistory}>↻ Refresh</button>
            </div>
            <HistoryPanel history={history} />
          </div>
        )}

        {/* ── MODEL TAB ── */}
        {activeTab === 'model' && (
          <div className="panel model-panel">
            <h2>⚙️ Model Management</h2>
            {health && (
              <div className="model-info-grid">
                <div className="info-card">
                  <div className="info-label">Status</div>
                  <div className="info-value" style={{ color: health.models_loaded ? '#00f2fe' : '#ff0844' }}>
                    {health.models_loaded ? '✅ Loaded' : '❌ Not Loaded'}
                  </div>
                </div>
                <div className="info-card">
                  <div className="info-label">Accuracy</div>
                  <div className="info-value">{health.model_metadata?.accuracy ?? 'N/A'}%</div>
                </div>
                <div className="info-card">
                  <div className="info-label">Trained At</div>
                  <div className="info-value">{health.model_metadata?.trained_at?.slice(0, 10) ?? 'N/A'}</div>
                </div>
                <div className="info-card">
                  <div className="info-label">Features</div>
                  <div className="info-value">{health.features_count ?? 'N/A'}</div>
                </div>
                <div className="info-card">
                  <div className="info-label">SHAP</div>
                  <div className="info-value">{health.shap_available ? '✅ Active' : '❌ Not installed'}</div>
                </div>
                <div className="info-card">
                  <div className="info-label">Retraining</div>
                  <div className="info-value">{health.retraining_in_progress ? '↻ In Progress' : 'Idle'}</div>
                </div>
              </div>
            )}

            <div style={{ marginTop: '1.5rem' }}>
              <h3>Features Used</h3>
              <div className="feature-chips">
                {(health?.features || []).map(f => <span key={f} className="chip">{f}</span>)}
              </div>
            </div>

            <div style={{ marginTop: '2rem' }}>
              <button className="retrain-btn" onClick={handleRetrain} disabled={health?.retraining_in_progress}>
                {health?.retraining_in_progress ? '↻ Retraining...' : '🔄 Trigger Model Retrain'}
              </button>
              {retrainMsg && <p className="retrain-msg">{retrainMsg}</p>}
              <p style={{ color: '#64748b', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                Retrains RF + NN models using latest data from the SQLite database. Runs in background.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App
