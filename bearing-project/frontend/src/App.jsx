import { useState, useRef } from "react";

const API_URL = "http://localhost:8080";

const SEVERITY = {
  Healthy: {
    color: "#00ff9d",
    bg: "rgba(0,255,157,0.07)",
    border: "rgba(0,255,157,0.28)",
    icon: "✦",
    rec: "Bearing is operating normally. Continue standard monitoring schedule.",
  },
  "Light Damage": {
    color: "#ffb800",
    bg: "rgba(255,184,0,0.07)",
    border: "rgba(255,184,0,0.28)",
    icon: "⚠",
    rec: "Early-stage fault detected. Schedule inspection within 2–4 weeks. Increase monitoring frequency.",
  },
  "Heavy Damage": {
    color: "#ff3b5c",
    bg: "rgba(255,59,92,0.07)",
    border: "rgba(255,59,92,0.28)",
    icon: "✖",
    rec: "Critical fault detected. Immediate inspection required. Do not continue operation without assessment.",
  },
};

const LABELS = ["Healthy", "Light Damage", "Heavy Damage"];

const OPERATING_FIELDS = [
  {
    key: "speed_rpm",
    label: "Shaft Speed",
    unit: "RPM",
    options: [900, 1500],
    description: "Rotational speed of drive system",
  },
  {
    key: "torque_Nm",
    label: "Load Torque",
    unit: "Nm",
    options: [0.1, 0.7],
    description: "Load torque in drive train",
  },
  {
    key: "force_N",
    label: "Radial Force",
    unit: "N",
    options: [400, 1000],
    description: "Radial force on test bearing",
  },
];

function ConfidenceBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 4,
          fontSize: 10,
          fontFamily: "'Space Mono', monospace",
          color: "rgba(255,255,255,0.4)",
          letterSpacing: "0.05em",
        }}
      >
        <span>{label.toUpperCase()}</span>
        <span style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div
        style={{
          height: 3,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${(value * 100).toFixed(1)}%`,
            background: color,
            borderRadius: 2,
            transition: "width 0.8s cubic-bezier(0.34,1.56,0.64,1)",
          }}
        />
      </div>
    </div>
  );
}

function OptionButton({ value, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: "9px 0",
        background: active ? "rgba(79,195,247,0.15)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${active ? "rgba(79,195,247,0.5)" : "rgba(255,255,255,0.08)"}`,
        borderRadius: 6,
        color: active ? "#4fc3f7" : "rgba(255,255,255,0.4)",
        fontFamily: "'Space Mono', monospace",
        fontSize: 12,
        cursor: "pointer",
        transition: "all 0.2s",
        fontWeight: active ? "bold" : "normal",
      }}
    >
      {value}
    </button>
  );
}

export default function App() {
  const [values, setValues] = useState({
    speed_rpm: 1500,
    torque_Nm: 0.7,
    force_N: 1000,
  });
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const resultRef = useRef(null);

  const handleFileSelect = (f) => {
    if (f && f.name.endsWith(".mat")) setFile(f);
  };

  const removeFile = () => {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFileSelect(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("speed_rpm", values.speed_rpm);
      form.append("torque_Nm", values.torque_Nm);
      form.append("force_N", values.force_N);

      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const cfg = result ? SEVERITY[result.label] : null;

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#080c14",
        color: "#fff",
        fontFamily: "'DM Sans', sans-serif",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulseRing { 0% { transform: scale(1); opacity: 0.6; } 100% { transform: scale(1.8); opacity: 0; } }
        .predict-btn:hover:not(:disabled) { background: rgba(79,195,247,0.18) !important; box-shadow: 0 0 28px rgba(79,195,247,0.25) !important; transform: translateY(-1px); }
        .predict-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .upload-zone:hover { border-color: rgba(79,195,247,0.5) !important; background: rgba(79,195,247,0.06) !important; }
        .remove-btn:hover { color: #ff3b5c !important; }
      `}</style>

      {/* Background grid */}
      <div
        style={{
          position: "fixed", inset: 0,
          backgroundImage: "linear-gradient(rgba(79,195,247,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(79,195,247,0.03) 1px,transparent 1px)",
          backgroundSize: "40px 40px",
          pointerEvents: "none",
        }}
      />
      {/* Glow blobs */}
      <div style={{ position: "fixed", top: -200, right: -200, width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle,rgba(79,195,247,0.05) 0%,transparent 70%)", pointerEvents: "none" }} />
      <div style={{ position: "fixed", bottom: -300, left: -200, width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle,rgba(100,50,255,0.04) 0%,transparent 70%)", pointerEvents: "none" }} />

      <div style={{ maxWidth: 860, margin: "0 auto", padding: "40px 24px 80px", position: "relative", zIndex: 1 }}>

        {/* Header */}
        <div style={{ marginBottom: 44, animation: "fadeUp 0.6s ease" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <div style={{ width: 34, height: 34, borderRadius: "50%", border: "1px solid rgba(79,195,247,0.4)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, color: "#4fc3f7" }}>◎</div>
            <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, letterSpacing: "0.2em", color: "#4fc3f7", textTransform: "uppercase" }}>
              Paderborn University · Bearing Analysis System
            </span>
          </div>
          <h1 style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: "clamp(44px,8vw,72px)", letterSpacing: "0.04em", lineHeight: 0.95, color: "#fff", marginBottom: 14 }}>
            BEARING<br />
            <span style={{ WebkitTextStroke: "1px rgba(79,195,247,0.6)", color: "transparent" }}>FAULT</span>{" "}DETECTOR
          </h1>
          <p style={{ color: "rgba(255,255,255,0.35)", fontFamily: "'Space Mono',monospace", fontSize: 11, maxWidth: 480, lineHeight: 1.8 }}>
            Upload a .mat measurement file and set the operating conditions. The ML model will classify bearing condition and estimate damage severity.
          </p>
        </div>

        {/* Operating Conditions Card */}
        <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, padding: 28, marginBottom: 16, animation: "fadeUp 0.6s ease 0.1s both" }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "0.18em", color: "rgba(79,195,247,0.7)", textTransform: "uppercase", marginBottom: 20, paddingBottom: 12, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            Operating Conditions
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 20 }}>
            {OPERATING_FIELDS.map((field) => (
              <div key={field.key}>
                <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, letterSpacing: "0.1em", color: "rgba(255,255,255,0.4)", textTransform: "uppercase", marginBottom: 8 }}>
                  {field.label} <span style={{ color: "#4fc3f7" }}>[{field.unit}]</span>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {field.options.map((opt) => (
                    <OptionButton
                      key={opt}
                      value={opt}
                      active={values[field.key] === opt}
                      onClick={() => setValues((prev) => ({ ...prev, [field.key]: opt }))}
                    />
                  ))}
                </div>
                <p style={{ margin: 0, marginTop: 6, fontFamily: "'Space Mono',monospace", fontSize: 9, color: "rgba(255,255,255,0.2)" }}>
                  {field.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* File Upload Card */}
        <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, padding: 28, marginBottom: 20, animation: "fadeUp 0.6s ease 0.2s both" }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "0.18em", color: "rgba(79,195,247,0.7)", textTransform: "uppercase", marginBottom: 20, paddingBottom: 12, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
            Measurement File — PCB Piezotronics 336C04 · Accelerometer Output
          </div>

          {/* Drop zone */}
          <div
            className="upload-zone"
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `1.5px dashed ${dragging ? "rgba(79,195,247,0.55)" : "rgba(79,195,247,0.25)"}`,
              borderRadius: 10,
              padding: "28px 20px",
              textAlign: "center",
              cursor: "pointer",
              transition: "all 0.25s",
              background: dragging ? "rgba(79,195,247,0.06)" : "rgba(79,195,247,0.02)",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mat"
              style={{ display: "none" }}
              onChange={(e) => handleFileSelect(e.target.files[0])}
            />
            <div style={{ width: 40, height: 40, borderRadius: "50%", border: "1px solid rgba(79,195,247,0.3)", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 12px", fontSize: 18, color: "#4fc3f7" }}>↑</div>
            <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 12, color: "rgba(255,255,255,0.6)", marginBottom: 4 }}>
              Drop .mat file here or click to browse
            </div>
            <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: "rgba(255,255,255,0.25)" }}>
              e.g. <span style={{ color: "#4fc3f7" }}>N15_M07_F10_KA01_1.mat</span> · MATLAB format · 64 kHz vibration signal
            </div>
          </div>

          {/* Selected file */}
          {file && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 14px", background: "rgba(79,195,247,0.08)", border: "1px solid rgba(79,195,247,0.3)", borderRadius: 8, marginTop: 12 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#4fc3f7", flexShrink: 0 }} />
              <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 11, color: "#4fc3f7", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {file.name}
              </span>
              <button
                className="remove-btn"
                onClick={removeFile}
                style={{ background: "none", border: "none", color: "rgba(255,59,92,0.6)", cursor: "pointer", fontSize: 14, padding: 0, lineHeight: 1 }}
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* Predict Button */}
        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={loading || !file}
          style={{
            width: "100%", padding: 17,
            background: "rgba(79,195,247,0.08)",
            border: "1px solid rgba(79,195,247,0.3)",
            borderRadius: 12,
            color: "#4fc3f7",
            fontFamily: "'Space Mono',monospace",
            fontSize: 12, letterSpacing: "0.15em", textTransform: "uppercase",
            cursor: "pointer", transition: "all 0.25s",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
            marginBottom: 28,
          }}
        >
          {loading ? (
            <>
              <div style={{ width: 14, height: 14, border: "2px solid rgba(79,195,247,0.3)", borderTop: "2px solid #4fc3f7", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
              Analyzing bearing signal...
            </>
          ) : (
            <>
              <span style={{ fontSize: 15 }}>◈</span>
              Run Fault Detection
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div style={{ padding: 18, background: "rgba(255,59,92,0.08)", border: "1px solid rgba(255,59,92,0.3)", borderRadius: 12, marginBottom: 24, fontFamily: "'Space Mono',monospace", fontSize: 11, color: "#ff3b5c", lineHeight: 1.7, animation: "fadeUp 0.4s ease" }}>
            <strong>⚠ Connection Error</strong><br />
            <span style={{ opacity: 0.7 }}>{error}</span><br />
            <span style={{ opacity: 0.45, fontSize: 10 }}>Make sure FastAPI backend is running at {API_URL}</span>
          </div>
        )}

        {/* Result */}
        {result && (
  <div ref={resultRef} style={{ display: "flex", flexDirection: "column", gap: 16, animation: "fadeUp 0.5s ease" }}>

    {/* Condition card */}
    {(() => {
      const cfg = SEVERITY[result.condition.label] || SEVERITY["Healthy"];
      return (
        <div style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, borderRadius: 14, padding: 28, boxShadow: `0 0 40px ${cfg.color}18` }}>
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "0.18em", color: "rgba(255,255,255,0.4)", textTransform: "uppercase", marginBottom: 16 }}>Bearing Condition</div>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
            <div style={{ width: 52, height: 52, borderRadius: "50%", border: `2px solid ${cfg.color}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, color: cfg.color, background: `${cfg.color}12` }}>{cfg.icon}</div>
            <div>
              <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 34, color: cfg.color, lineHeight: 1 }}>{result.condition.label}</div>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 4 }}>
                Confidence: <span style={{ color: cfg.color }}>{(result.condition.probabilities[result.condition.prediction] * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          {LABELS.map((label, i) => (
            <ConfidenceBar key={label} label={label} value={result.condition.probabilities[i] || 0} color={Object.values(SEVERITY)[i].color} />
          ))}
          <div style={{ marginTop: 16, padding: "12px 14px", background: "rgba(0,0,0,0.22)", borderRadius: 8, borderLeft: `3px solid ${cfg.color}`, fontFamily: "'Space Mono',monospace", fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 1.8 }}>
            <div style={{ color: cfg.color, marginBottom: 4, fontWeight: "bold" }}>▸ Recommendation</div>
            {cfg.rec}
          </div>
        </div>
      );
    })()}

    {/* Location + Damage type row */}
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, padding: 24 }}>
        <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "0.18em", color: "rgba(79,195,247,0.7)", textTransform: "uppercase", marginBottom: 12 }}>Fault Location</div>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 30, color: "#4fc3f7", lineHeight: 1 }}>{result.location.label}</div>
        <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: "rgba(255,255,255,0.25)", marginTop: 6 }}>Inner Race (IR) · Outer Race (OR) · Both · None</div>
      </div>
      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 14, padding: 24 }}>
        <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "0.18em", color: "rgba(79,195,247,0.7)", textTransform: "uppercase", marginBottom: 12 }}>Damage Type</div>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 30, color: "#4fc3f7", lineHeight: 1 }}>{result.damage_type.label}</div>
        <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: "rgba(255,255,255,0.25)", marginTop: 6 }}>EDM · Drilling · Engraving · Fatigue · Plastic Def.</div>
      </div>
    </div>

  </div>
)}

        {/* Footer */}
        <div style={{ marginTop: 56, borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 20, display: "flex", justifyContent: "space-between", fontFamily: "'Space Mono',monospace", fontSize: 9, color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em" }}>
          <span>PADERBORN BEARING DATASET · ML PIPELINE v1.0</span>
          <span>64 kHz · 0.1s WINDOWS · 100% NON-OVERLAP</span>
        </div>
      </div>
    </div>
  );
}