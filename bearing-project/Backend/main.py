import io
import numpy as np
import pandas as pd
import joblib
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────
REPO_ID = "shaeel12/Bearing_problem_prediction"


def _download_model_file(filename: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=filename)


def _load_artifact(filename: str):
    return joblib.load(_download_model_file(filename))


model  = _load_artifact("bearing_diagnostic_model_v1.pkl")
scaler = _load_artifact("bearing_scaler.pkl")
pca    = _load_artifact("bearing_pca.pkl")

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLING_RATE = 64_000
WINDOW_SIZE   = 6_400
STEP          = 6_400



SEVERITY_META = {
    # predictions[0] = y1 = master_df['severity']  (0, 1, or 2)
    # However, severity=1 dominates training data (10 bearings vs 6 healthy vs 17 severe),
    # so the model maps to condition semantics: 0=Healthy, 1=Artificial, 2=Real damage.
    # This matches the original intent of "Healthy / Light Damage / Heavy Damage".
    0: {
        "label": "Healthy",
        "description": "No fault detected. Bearing operating normally.",
        "color": "#00ff9d",
        "bg": "rgba(0,255,157,0.07)",
        "border": "rgba(0,255,157,0.28)",
        "icon": "✦",
        "recommendation": "Continue standard monitoring schedule.",
    },
    1: {
        "label": "Light Damage",
        "description": "Artificially-induced fault detected (EDM, drilling, or engraving). Early-stage damage.",
        "color": "#ffb800",
        "bg": "rgba(255,184,0,0.07)",
        "border": "rgba(255,184,0,0.28)",
        "icon": "⚠",
        "recommendation": "Schedule inspection within 2-4 weeks. Increase monitoring frequency.",
    },
    2: {
        "label": "Heavy Damage",
        "description": "Real fatigue or plastic-deformation fault detected. Advanced bearing deterioration.",
        "color": "#ff3b5c",
        "bg": "rgba(255,59,92,0.07)",
        "border": "rgba(255,59,92,0.28)",
        "icon": "✖",
        "recommendation": "Immediate inspection required. Do not continue operation without assessment.",
    },
}

# Location: alphabetical LabelEncoder order from notebook
# le_location.classes_ == ['IR', 'None', 'OR', 'OR+IR']
LOCATION_CLASSES = ["IR", "None", "OR", "OR+IR"]

# Damage type: alphabetical LabelEncoder order from notebook
# le_damage.classes_ == ['Drilling', 'EDM', 'Engraving', 'Fatigue', 'None', 'Plastic Def.']
DAMAGE_CLASSES = ["Drilling", "EDM", "Engraving", "Fatigue", "None", "Plastic Def."]


def _severity_label(sev_int: int) -> dict:
    """Return the full severity metadata for a predicted severity integer."""
    return SEVERITY_META.get(sev_int, {
        "label": f"Unknown (code {sev_int})",
        "description": "Prediction outside known severity range.",
        "color": "#888888",
        "bg": "rgba(128,128,128,0.07)",
        "border": "rgba(128,128,128,0.28)",
        "icon": "?",
        "recommendation": "Check model integrity.",
    })


def _location_label(loc_int: int) -> str:
    """Decode location integer using the same alphabetical LabelEncoder the notebook used."""
    if 0 <= loc_int < len(LOCATION_CLASSES):
        return LOCATION_CLASSES[loc_int]
    return f"Unknown (code {loc_int})"


def _damage_label(dmg_int: int) -> str:
    """Decode damage-type integer using the same alphabetical LabelEncoder the notebook used."""
    if 0 <= dmg_int < len(DAMAGE_CLASSES):
        return DAMAGE_CLASSES[dmg_int]
    return f"Unknown (code {dmg_int})"


# ── /config response ───────────────────────────────────────────────────────────
# Built from the actual class lists so the frontend always stays in sync.

CONFIG = {
    "severity_meta": SEVERITY_META,
    "severity_labels": [m["label"] for m in SEVERITY_META.values()],
    "location_labels": LOCATION_CLASSES,
    "damage_labels": DAMAGE_CLASSES,
    "operating_fields": [
        {
            "key": "speed_rpm",
            "label": "Shaft Speed",
            "unit": "RPM",
            "options": [900, 1500],
            "description": "Rotational speed of drive system",
        },
        {
            "key": "torque_Nm",
            "label": "Load Torque",
            "unit": "Nm",
            "options": [0.1, 0.7],
            "description": "Load torque in drive train",
        },
        {
            "key": "force_N",
            "label": "Radial Force",
            "unit": "N",
            "options": [400, 1000],
            "description": "Radial force on test bearing",
        },
    ],
}

# ── Feature column names (must match scaler fit order from notebook) ───────────
_FEAT_KEYS = [
    "rms", "mean", "std", "kurtosis", "skewness",
    "peak", "peak_to_peak", "crest_factor", "shape_factor", "impulse_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_energy", "dominant_freq",
    "band_energy_0", "band_energy_1", "band_energy_2", "band_energy_3", "band_energy_4",
]

FEATURE_COLUMNS = (
    ["speed_rpm", "torque_Nm", "force_N"] +
    [f"vib_{k}"  for k in _FEAT_KEYS] +   # vibration  (19 features)
    [f"cur1_{k}" for k in _FEAT_KEYS] +   # current 1  (19 features)
    [f"cur2_{k}" for k in _FEAT_KEYS]     # current 2  (19 features)
)

# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_time_features(signal: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(signal ** 2)))
    mean_abs = float(np.mean(np.abs(signal))) + 1e-10
    return {
        "rms":            rms,
        "mean":           float(np.mean(signal)),
        "std":            float(np.std(signal)),
        "kurtosis":       float(kurtosis(signal)),
        "skewness":       float(skew(signal)),
        "peak":           float(np.max(np.abs(signal))),
        "peak_to_peak":   float(np.ptp(signal)),
        "crest_factor":   float(np.max(np.abs(signal)) / (rms + 1e-10)),
        "shape_factor":   float(rms / mean_abs),
        "impulse_factor": float(np.max(np.abs(signal)) / mean_abs),
    }


def extract_freq_features(signal: np.ndarray, fs: int = SAMPLING_RATE, n_bands: int = 5) -> dict:
    N       = len(signal)
    freqs   = np.fft.rfftfreq(N, 1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-10)
    )

    f_psd, Pxx = welch(signal, fs=fs, nperseg=min(512, N))
    band_edges = np.linspace(0, fs / 2, n_bands + 1)
    band_energies = {
        f"band_energy_{i}": float(np.sum(Pxx[(f_psd >= band_edges[i]) & (f_psd < band_edges[i + 1])]))
        for i in range(n_bands)
    }

    features = {
        "spectral_centroid":  float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "spectral_energy":    float(np.sum(fft_mag ** 2)),
        "dominant_freq":      float(freqs[np.argmax(fft_mag)]),
    }
    features.update(band_energies)
    return features


def extract_all_features(signal: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    feats = {}
    feats.update(extract_time_features(signal))
    feats.update(extract_freq_features(signal, fs))
    return feats


# ── .mat loader ────────────────────────────────────────────────────────────────

def load_mat_bytes(file_bytes: bytes) -> dict:
    """Load a Paderborn-format .mat file from raw bytes."""
    mat = loadmat(io.BytesIO(file_bytes), simplify_cells=True)
    data_key = [k for k in mat.keys() if not k.startswith("_")][0]
    data = mat[data_key]
    Y = data["Y"]

    signals = {}
    for channel in Y:
        name = channel["Name"]
        signals[name] = np.array(channel["Data"]).flatten().astype(float)

    return {
        "vibration": signals.get("vibration_1",   np.zeros(256001)),
        "current_1": signals.get("phase_current_1", np.zeros(256001)),
        "current_2": signals.get("phase_current_2", np.zeros(256001)),
        "speed":     signals.get("speed",           np.zeros(16001)),
        "torque":    signals.get("torque",           np.zeros(16001)),
        "force":     signals.get("force",            np.zeros(16001)),
    }


# ── Windowed feature extraction ────────────────────────────────────────────────

def features_from_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply a sliding window to a raw signal, extract features from each window,
    and return the mean feature vector across all windows.
    Mirrors process_single_file() in the notebook exactly.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(len(_FEAT_KEYS), dtype=float)

    windows = [
        signal[start : start + WINDOW_SIZE]
        for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP)
    ]
    if not windows:
        # Edge case: signal shorter than one window
        windows = [signal[:WINDOW_SIZE] if len(signal) >= WINDOW_SIZE else signal]

    # Extract all 19 features for every window, then average across windows.
    # This is identical to what the notebook does during training.
    rows = [list(extract_all_features(w).values()) for w in windows]
    return np.mean(rows, axis=0)   # shape: (19,)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    """
    Return frontend configuration.
    Labels are derived from the actual LabelEncoder class lists used during
    training, not from hard-coded display strings.
    """
    return CONFIG


@app.post("/predict")
async def predict(
    file:      UploadFile = File(...),
    speed_rpm: float      = Form(...),
    torque_Nm: float      = Form(...),
    force_N:   float      = Form(...),
):
    """
    Predict bearing condition from a Paderborn .mat file.

    Pipeline (mirrors paderborn_pipeline.ipynb exactly):
      1. Load .mat → extract vibration, current_1, current_2 channels
      2. Sliding-window feature extraction (19 time+freq features per channel)
      3. Average features across windows  → 60-dim vector (3 op. params + 19×3)
      4. StandardScaler normalisation
      5. PCA (5 components)
      6. MultiOutputClassifier → (severity_int, location_int, damage_int)

    Label decoding:
      - severity   → looked up in SEVERITY_META  (0/1/2, data-driven)
      - location   → decoded via alphabetical LabelEncoder order (LOCATION_CLASSES)
      - damage_type→ decoded via alphabetical LabelEncoder order (DAMAGE_CLASSES)
    """
    try:
        # 1. Read and parse the .mat file
        file_bytes = await file.read()
        signals = load_mat_bytes(file_bytes)
        print(
            f"[DEBUG] Signals loaded: vib={signals['vibration'].shape}, "
            f"cur1={signals['current_1'].shape}, cur2={signals['current_2'].shape}"
        )

        # 2. Extract windowed, averaged features per channel
        vib_feats  = features_from_signal(signals["vibration"])
        cur1_feats = features_from_signal(signals["current_1"])
        cur2_feats = features_from_signal(signals["current_2"])
        print(
            f"[DEBUG] Feature shapes: vib={vib_feats.shape}, "
            f"cur1={cur1_feats.shape}, cur2={cur2_feats.shape}"
        )

        # 3. Build the full 60-dim feature vector
        # ORDER must match process_single_file() in the notebook:
        # vib_* (19) -> cur1_* (19) -> cur2_* (19) -> speed_rpm, torque_Nm, force_N
        # Operating conditions were added AFTER signal features in the dict,
        # so they appear at the END of feature_cols after meta_cols exclusion.
        # The old code put them FIRST which misaligned every column vs the scaler.
        feature_vector = np.concatenate([
            vib_feats,
            cur1_feats,
            cur2_feats,
            [speed_rpm, torque_Nm, force_N],
        ]).reshape(1, -1)
        print(
            f"[DEBUG] Feature vector shape={feature_vector.shape}, "
            f"NaN={np.isnan(feature_vector).any()}, Inf={np.isinf(feature_vector).any()}"
        )

        # 4. StandardScaler normalisation
        feature_scaled = scaler.transform(feature_vector)

        # 5. PCA dimensionality reduction
        feature_pca = pca.transform(feature_scaled)
        print(f"[DEBUG] PCA output shape={feature_pca.shape}, values={feature_pca}")

        # 6. Multi-output prediction
        raw_preds = model.predict(feature_pca)[0]
        print(f"[DEBUG] Raw model predictions: {raw_preds}")

        sev_int = int(raw_preds[0])   # severity index  (0, 1, or 2)
        loc_int = int(raw_preds[1])   # location index  (LabelEncoder order)
        dmg_int = int(raw_preds[2])   # damage-type index (LabelEncoder order)

        # Decode labels using the actual encoding scheme from the notebook
        sev_meta     = _severity_label(sev_int)
        location_str = _location_label(loc_int)
        damage_str   = _damage_label(dmg_int)

        print(
            f"[DEBUG] Decoded — severity: {sev_int} → '{sev_meta['label']}', "
            f"location: {loc_int} → '{location_str}', "
            f"damage: {dmg_int} → '{damage_str}'"
        )

        # Probability estimates for the severity head (estimators_[0])
        if hasattr(model, "estimators_"):
            raw_proba = model.estimators_[0].predict_proba(feature_pca)[0]
            # Align to the fixed 3-class output; pad if the model only saw a subset
            n_model_classes = len(raw_proba)
            if n_model_classes == len(SEVERITY_META):
                probabilities = raw_proba.tolist()
            else:
                # Some classes may be absent from the trained estimator's classes_;
                # build a full-length probability vector from what we have.
                probabilities = [0.0] * len(SEVERITY_META)
                for idx, cls in enumerate(model.estimators_[0].classes_):
                    if int(cls) < len(probabilities):
                        probabilities[int(cls)] = float(raw_proba[idx])
        elif hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_pca)[0].tolist()
        else:
            probabilities = [1.0 if i == sev_int else 0.0 for i in range(len(SEVERITY_META))]

        print(f"[DEBUG] Severity probabilities: {probabilities}")

        return {
            "condition": {
                "prediction":   sev_int,
                "label":        sev_meta["label"],
                "probabilities": probabilities,
            },
            "location": {
                "prediction": loc_int,
                "label":      location_str,
            },
            "damage_type": {
                "prediction": dmg_int,
                "label":      damage_str,
            },
        }

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)