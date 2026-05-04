import { useState, useEffect, useRef } from "react";

const API_URL = "http://localhost:8080";

const SEVERITY_CONFIG = {
  Healthy: {
    color: "#00ff9d",
    bg: "rgba(0,255,157,0.08)",
    border: "rgba(0,255,157,0.3)",
    icon: "✦",
    glow: "0 0 40px rgba(0,255,157,0.25)",
  },
  "Light Damage": {
    color: "#ffb800",
    bg: "rgba(255,184,0,0.08)",
    border: "rgba(255,184,0,0.3)",
    icon: "⚠",
    glow: "0 0 40px rgba(255,184,0,0.25)",
  },
  "Heavy Damage": {
    color: "#ff3b5c",
    bg: "rgba(255,59,92,0.08)",
    border: "rgba(255,59,92,0.3)",
    icon: "✖",
    glow: "0 0 40px rgba(255,59,92,0.25)",
  },
};

const FIELDS = [
  {
    section: "Operating Conditions",
    fields: [
      {
        key: "speed_rpm",
        label: "Shaft Speed",
        unit: "RPM",
        options: [900, 1500],
        type: "select",
        description: "Rotational speed of the bearing shaft",
      },
      {
        key: "torque_Nm",
        label: "Load Torque",
        unit: "Nm",
        options: [0.1, 0.7],
        type: "select",
        description: "Applied torque on the bearing",
      },
      {
        key: "force_N",
        label: "Radial Force",
        unit: "N",
        options: [400, 1000],
        type: "select",
        description: "Radial load applied to the bearing",
      },
    ],
  },
  {
    section: "Vibration Features",
    fields: [
      {
        key: "vib_rms",
        label: "RMS",
        unit: "g",
        type: "number",
        placeholder: "e.g. 0.045",
        description: "Root Mean Square of vibration signal",
        min: 0,
        step: 0.001,
      },
      {
        key: "vib_kurtosis",
        label: "Kurtosis",
        unit: "",
        type: "number",
        placeholder: "e.g. 3.2",
        description: "Sharpness of vibration peaks — key fault indicator",
        min: 0,
        step: 0.1,
      },
      {
        key: "vib_crest_factor",
        label: "Crest Factor",
        unit: "",
        type: "number",
        placeholder: "e.g. 4.5",
        description: "Peak-to-RMS ratio",
        min: 0,
        step: 0.1,
      },
      {
        key: "vib_std",
        label: "Std Deviation",
        unit: "g",
        type: "number",
        placeholder: "e.g. 0.032",
        description: "Signal variability",
        min: 0,
        step: 0.001,
      },
      {
        key: "vib_peak",
        label: "Peak Amplitude",
        unit: "g",
        type: "number",
        placeholder: "e.g. 0.21",
        description: "Maximum absolute signal value",
        min: 0,
        step: 0.001,
      },
      {
        key: "vib_spectral_centroid",
        label: "Spectral Centroid",
        unit: "Hz",
        type: "number",
        placeholder: "e.g. 2400",
        description: "Weighted mean frequency",
        min: 0,
        step: 1,
      },
    ],
  },
  {
    section: "Current Signal Features",
    fields: [
      {
        key: "cur1_rms",
        label: "Phase Current 1 RMS",
        unit: "A",
        type: "number",
        placeholder: "e.g. 1.2",
        description: "RMS of phase current 1",
        min: 0,
        step: 0.01,
      },
      {
        key: "cur1_kurtosis",
        label: "Phase Current 1 Kurtosis",
        unit: "",
        type: "number",
        placeholder: "e.g. 2.8",
        description: "Kurtosis of phase current 1",
        min: 0,
        step: 0.1,
      },
      {
        key: "cur2_rms",
        label: "Phase Current 2 RMS",
        unit: "A",
        type: "number",
        placeholder: "e.g. 1.1",
        description: "RMS of phase current 2",
        min: 0,
        step: 0.01,
      },
    ],
  },
];

function PulseRing({ color, active }) {
  return active ? (
    <div
      style={{
        position: "absolute",
        inset: -16,
        borderRadius: "50%",
        border: `1px solid ${color}`,
        animation: "pulseRing 2s ease-out infinite",
        pointerEvents: "none",
      }}
    />
  ) : null;
}

function ConfidenceBar({ label, value, color }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    setTimeout(() => setWidth(value * 100), 100);
  }, [value]);

  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 5,
          fontSize: 11,
          fontFamily: "'Space Mono', monospace",
          color: "rgba(255,255,255,0.5)",
          letterSpacing: "0.05em",
        }}
      >
        <span>{label.toUpperCase()}</span>
        <span style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div
        style={{
          height: 4,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${width}%`,
            background: color,
            borderRadius: 2,
            transition: "width 0.8s cubic-bezier(0.34,1.56,0.64,1)",
            boxShadow: `0 0 8px ${color}`,
          }}
        />
      </div>
    </div>
  );
}

function InputField({ field, value, onChange }) {
  const [focused, setFocused] = useState(false);

  if (field.type === "select") {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <label
          style={{
            fontSize: 10,
            fontFamily: "'Space Mono', monospace",
            color: "rgba(255,255,255,0.4)",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
          }}
        >
          {field.label}
          {field.unit && (
            <span style={{ color: "#4fc3f7", marginLeft: 4 }}>
              [{field.unit}]
            </span>
          )}
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          {field.options.map((opt) => (
            <button
              key={opt}
              onClick={() => onChange(opt)}
              style={{
                flex: 1,
                padding: "10px 0",
                background:
                  value === opt ? "rgba(79,195,247,0.15)" : "rgba(255,255,255,0.03)",
                border: `1px solid ${value === opt ? "rgba(79,195,247,0.5)" : "rgba(255,255,255,0.08)"}`,
                borderRadius: 6,
                color: value === opt ? "#4fc3f7" : "rgba(255,255,255,0.4)",
                fontFamily: "'Space Mono', monospace",
                fontSize: 13,
                cursor: "pointer",
                transition: "all 0.2s",
                fontWeight: value === opt ? "bold" : "normal",
              }}
            >
              {opt}
            </button>
          ))}
        </div>
        <p
          style={{
            margin: 0,
            fontSize: 10,
            color: "rgba(255,255,255,0.25)",
            fontFamily: "'Space Mono', monospace",
          }}
        >
          {field.description}
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <label
        style={{
          fontSize: 10,
          fontFamily: "'Space Mono', monospace",
          color: focused ? "rgba(79,195,247,0.8)" : "rgba(255,255,255,0.4)",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          transition: "color 0.2s",
        }}
      >
        {field.label}
        {field.unit && (
          <span style={{ color: "#4fc3f7", marginLeft: 4 }}>
            [{field.unit}]
          </span>
        )}
      </label>
      <input
        type="number"
        value={value}
        placeholder={field.placeholder}
        min={field.min}
        step={field.step}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        onChange={(e) => onChange(parseFloat(e.target.value) || "")}
        style={{
          background: focused
            ? "rgba(79,195,247,0.05)"
            : "rgba(255,255,255,0.03)",
          border: `1px solid ${focused ? "rgba(79,195,247,0.4)" : "rgba(255,255,255,0.08)"}`,
          borderRadius: 6,
          padding: "10px 12px",
          color: "#fff",
          fontFamily: "'Space Mono', monospace",
          fontSize: 13,
          outline: "none",
          transition: "all 0.2s",
          width: "100%",
          boxSizing: "border-box",
        }}
      />
      <p
        style={{
          margin: 0,
          fontSize: 10,
          color: "rgba(255,255,255,0.25)",
          fontFamily: "'Space Mono', monospace",
        }}
      >
        {field.description}
      </p>
    </div>
  );
}

export default function App() {
  const [values, setValues] = useState({
    speed_rpm: 1500,
    torque_Nm: 0.7,
    force_N: 1000,
    vib_rms: "",
    vib_kurtosis: "",
    vib_crest_factor: "",
    vib_std: "",
    vib_peak: "",
    vib_spectral_centroid: "",
    cur1_rms: "",
    cur1_kurtosis: "",
    cur2_rms: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState(0);
  const resultRef = useRef(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);

      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const cfg = result ? SEVERITY_CONFIG[result.label] : null;

  const labelMap = ["Healthy", "Light Damage", "Heavy Damage"];

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
      {/* Google Fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Bebas+Neue&display=swap');

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body { background: #080c14; }

        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { opacity: 0.3; }

        @keyframes pulseRing {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(1.8); opacity: 0; }
        }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }

        .section-tab:hover { background: rgba(79,195,247,0.08) !important; }
        .predict-btn:hover:not(:disabled) { 
          background: rgba(79,195,247,0.2) !important; 
          box-shadow: 0 0 30px rgba(79,195,247,0.3) !important;
          transform: translateY(-1px);
        }
        .predict-btn:disabled { opacity: 0.5; cursor: not-allowed; }
      `}</style>

      {/* Background grid */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(79,195,247,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(79,195,247,0.03) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
          pointerEvents: "none",
        }}
      />

      {/* Scanline effect */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: "2px",
          background:
            "linear-gradient(90deg, transparent, rgba(79,195,247,0.15), transparent)",
          animation: "scanline 8s linear infinite",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Glow blobs */}
      <div
        style={{
          position: "fixed",
          top: -200,
          right: -200,
          width: 600,
          height: 600,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(79,195,247,0.04) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "fixed",
          bottom: -300,
          left: -200,
          width: 700,
          height: 700,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(100,50,255,0.04) 0%, transparent 70%)",
          pointerEvents: "none",
        }}
      />

      {/* Main content */}
      <div
        style={{
          maxWidth: 900,
          margin: "0 auto",
          padding: "40px 24px 80px",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: 48, animation: "fadeUp 0.6s ease" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 16,
            }}
          >
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: "50%",
                border: "1px solid rgba(79,195,247,0.4)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 16,
              }}
            >
              ◎
            </div>
            <span
              style={{
                fontFamily: "'Space Mono', monospace",
                fontSize: 11,
                letterSpacing: "0.2em",
                color: "#4fc3f7",
                textTransform: "uppercase",
              }}
            >
              Paderborn University · Bearing Analysis System
            </span>
          </div>
          <h1
            style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: "clamp(48px, 8vw, 80px)",
              letterSpacing: "0.04em",
              lineHeight: 0.95,
              color: "#fff",
              marginBottom: 16,
            }}
          >
            BEARING
            <br />
            <span
              style={{
                WebkitTextStroke: "1px rgba(79,195,247,0.6)",
                color: "transparent",
              }}
            >
              FAULT
            </span>{" "}
            DETECTOR
          </h1>
          <p
            style={{
              color: "rgba(255,255,255,0.35)",
              fontFamily: "'Space Mono', monospace",
              fontSize: 12,
              maxWidth: 480,
              lineHeight: 1.8,
            }}
          >
            Enter sensor readings from your bearing. The ML model will classify
            the bearing condition and estimate damage severity.
          </p>
        </div>

        {/* Section tabs */}
        <div
          style={{
            display: "flex",
            gap: 4,
            marginBottom: 24,
            animation: "fadeUp 0.6s ease 0.1s both",
          }}
        >
          {FIELDS.map((s, i) => (
            <button
              key={i}
              className="section-tab"
              onClick={() => setActiveSection(i)}
              style={{
                flex: 1,
                padding: "10px 8px",
                background:
                  activeSection === i
                    ? "rgba(79,195,247,0.1)"
                    : "rgba(255,255,255,0.02)",
                border: `1px solid ${activeSection === i ? "rgba(79,195,247,0.3)" : "rgba(255,255,255,0.06)"}`,
                borderRadius: 8,
                color:
                  activeSection === i
                    ? "#4fc3f7"
                    : "rgba(255,255,255,0.3)",
                fontFamily: "'Space Mono', monospace",
                fontSize: 10,
                letterSpacing: "0.08em",
                cursor: "pointer",
                transition: "all 0.2s",
                textTransform: "uppercase",
              }}
            >
              {s.section}
            </button>
          ))}
        </div>

        {/* Form card */}
        <div
          style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 16,
            padding: 32,
            marginBottom: 20,
            animation: "fadeUp 0.6s ease 0.2s both",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
              gap: 24,
            }}
          >
            {FIELDS[activeSection].fields.map((field) => (
              <InputField
                key={field.key}
                field={field}
                value={values[field.key]}
                onChange={(val) =>
                  setValues((prev) => ({ ...prev, [field.key]: val }))
                }
              />
            ))}
          </div>
        </div>

        {/* Navigation dots */}
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 8,
            marginBottom: 28,
          }}
        >
          {FIELDS.map((_, i) => (
            <button
              key={i}
              onClick={() => setActiveSection(i)}
              style={{
                width: i === activeSection ? 24 : 8,
                height: 8,
                borderRadius: 4,
                background:
                  i === activeSection
                    ? "#4fc3f7"
                    : "rgba(255,255,255,0.15)",
                border: "none",
                cursor: "pointer",
                transition: "all 0.3s",
                padding: 0,
              }}
            />
          ))}
        </div>

        {/* Predict button */}
        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={loading}
          style={{
            width: "100%",
            padding: "18px",
            background: "rgba(79,195,247,0.08)",
            border: "1px solid rgba(79,195,247,0.3)",
            borderRadius: 12,
            color: "#4fc3f7",
            fontFamily: "'Space Mono', monospace",
            fontSize: 13,
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            cursor: "pointer",
            transition: "all 0.25s",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            marginBottom: 32,
            animation: "fadeUp 0.6s ease 0.3s both",
          }}
        >
          {loading ? (
            <>
              <div
                style={{
                  width: 16,
                  height: 16,
                  border: "2px solid rgba(79,195,247,0.3)",
                  borderTop: "2px solid #4fc3f7",
                  borderRadius: "50%",
                  animation: "spin 0.8s linear infinite",
                }}
              />
              Analyzing bearing signal...
            </>
          ) : (
            <>
              <span style={{ fontSize: 16 }}>◈</span>
              Run Fault Detection
            </>
          )}
        </button>

        {/* Error */}
        {error && (
          <div
            style={{
              padding: 20,
              background: "rgba(255,59,92,0.08)",
              border: "1px solid rgba(255,59,92,0.3)",
              borderRadius: 12,
              marginBottom: 24,
              fontFamily: "'Space Mono', monospace",
              fontSize: 12,
              color: "#ff3b5c",
              animation: "fadeUp 0.4s ease",
            }}
          >
            <strong>⚠ Connection Error</strong>
            <br />
            <span style={{ opacity: 0.7, marginTop: 4, display: "block" }}>
              {error}
            </span>
            <span style={{ opacity: 0.5, fontSize: 10, marginTop: 8, display: "block" }}>
              Make sure your FastAPI backend is running at {API_URL}
            </span>
          </div>
        )}

        {/* Result panel */}
        {result && cfg && (
          <div
            ref={resultRef}
            style={{
              background: cfg.bg,
              border: `1px solid ${cfg.border}`,
              borderRadius: 16,
              padding: 36,
              animation: "fadeUp 0.5s ease",
              boxShadow: cfg.glow,
              position: "relative",
              overflow: "hidden",
            }}
          >
            {/* Result background decoration */}
            <div
              style={{
                position: "absolute",
                top: -60,
                right: -60,
                width: 200,
                height: 200,
                borderRadius: "50%",
                background: `radial-gradient(circle, ${cfg.color}08 0%, transparent 70%)`,
                pointerEvents: "none",
              }}
            />

            <div
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: 24,
                marginBottom: 32,
              }}
            >
              {/* Status icon */}
              <div
                style={{
                  position: "relative",
                  width: 64,
                  height: 64,
                  flexShrink: 0,
                }}
              >
                <div
                  style={{
                    width: 64,
                    height: 64,
                    borderRadius: "50%",
                    border: `2px solid ${cfg.color}`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 24,
                    color: cfg.color,
                    position: "relative",
                    zIndex: 1,
                    background: `${cfg.color}10`,
                  }}
                >
                  {cfg.icon}
                </div>
                <PulseRing color={cfg.color} active={true} />
              </div>

              <div>
                <div
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: 10,
                    letterSpacing: "0.2em",
                    color: "rgba(255,255,255,0.4)",
                    textTransform: "uppercase",
                    marginBottom: 6,
                  }}
                >
                  Diagnosis Result
                </div>
                <div
                  style={{
                    fontFamily: "'Bebas Neue', sans-serif",
                    fontSize: 40,
                    letterSpacing: "0.04em",
                    color: cfg.color,
                    lineHeight: 1,
                    marginBottom: 8,
                  }}
                >
                  {result.label}
                </div>
                <div
                  style={{
                    fontFamily: "'Space Mono', monospace",
                    fontSize: 11,
                    color: "rgba(255,255,255,0.35)",
                  }}
                >
                  Confidence:{" "}
                  <span style={{ color: cfg.color }}>
                    {(
                      result.probabilities[result.prediction] * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
            </div>

            {/* Probability bars */}
            <div
              style={{
                borderTop: "1px solid rgba(255,255,255,0.06)",
                paddingTop: 24,
              }}
            >
              <div
                style={{
                  fontFamily: "'Space Mono', monospace",
                  fontSize: 10,
                  color: "rgba(255,255,255,0.3)",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  marginBottom: 16,
                }}
              >
                Class Probabilities
              </div>
              {labelMap.map((label, i) => (
                <ConfidenceBar
                  key={label}
                  label={label}
                  value={result.probabilities[i] || 0}
                  color={Object.values(SEVERITY_CONFIG)[i].color}
                />
              ))}
            </div>

            {/* Recommendations */}
            <div
              style={{
                marginTop: 24,
                padding: 16,
                background: "rgba(0,0,0,0.2)",
                borderRadius: 8,
                fontFamily: "'Space Mono', monospace",
                fontSize: 11,
                color: "rgba(255,255,255,0.4)",
                lineHeight: 1.8,
                borderLeft: `3px solid ${cfg.color}`,
              }}
            >
              <div style={{ color: cfg.color, marginBottom: 6, fontWeight: "bold" }}>
                ▸ Recommendation
              </div>
              {result.label === "Healthy" &&
                "Bearing is operating normally. Continue standard monitoring schedule."}
              {result.label === "Light Damage" &&
                "Early-stage fault detected. Schedule inspection within 2–4 weeks. Increase monitoring frequency."}
              {result.label === "Heavy Damage" &&
                "Critical fault detected. Immediate inspection required. Do not continue operation without assessment."}
            </div>
          </div>
        )}

        {/* Footer */}
        <div
          style={{
            marginTop: 60,
            borderTop: "1px solid rgba(255,255,255,0.05)",
            paddingTop: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontFamily: "'Space Mono', monospace",
            fontSize: 10,
            color: "rgba(255,255,255,0.2)",
            letterSpacing: "0.08em",
          }}
        >
          <span>PADERBORN BEARING DATASET · ML PIPELINE v1.0</span>
          <span>64 kHz · 0.1s WINDOWS · 50% OVERLAP</span>
        </div>
      </div>
    </div>
  );
}
