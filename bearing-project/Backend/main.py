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

LABEL_MAP    = {0: "Healthy",  1: "Light Damage",  2: "Heavy Damage"}
LOCATION_MAP = {0: "IR",       1: "None",           2: "OR",          3: "OR+IR"}
DAMAGE_MAP   = {0: "Drilling", 1: "EDM",            2: "Engraving",   3: "Fatigue", 4: "None", 5: "Plastic Def."}


def _flatten_prediction(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.ndim == 2 and prediction.shape[0] == 1:
        return prediction[0]
    return prediction.ravel()


def _get_condition_probabilities(model_obj, features: np.ndarray, default_index: int = 0) -> list[float]:
    estimator = model_obj
    if hasattr(model_obj, "steps"):
        estimator = model_obj.steps[-1][1]

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(features)
        if isinstance(proba, list):
            if not proba:
                return [1.0 if i == default_index else 0.0 for i in range(3)]
            first = np.asarray(proba[0])
            if first.ndim == 2:
                return first[0].tolist()
            return first.ravel().tolist()

        proba = np.asarray(proba)
        if proba.ndim == 2:
            return proba[0].tolist()

    return [1.0 if i == default_index else 0.0 for i in range(3)]

# Feature column names must match what the scaler was fitted with in the notebook
_FEAT_KEYS = [
    "rms", "mean", "std", "kurtosis", "skewness",
    "peak", "peak_to_peak", "crest_factor", "shape_factor", "impulse_factor",
    "spectral_centroid", "spectral_bandwidth", "spectral_energy", "dominant_freq",
    "band_energy_0", "band_energy_1", "band_energy_2", "band_energy_3", "band_energy_4",
]

FEATURE_COLUMNS = (
    ["speed_rpm", "torque_Nm", "force_N"] +
    [f"vib_{k}"  for k in _FEAT_KEYS] +      # vibration (19)
    [f"cur1_{k}" for k in _FEAT_KEYS] +      # current 1  (19)
    [f"cur2_{k}" for k in _FEAT_KEYS]        # current 2  (19)
)

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_time_features(signal: np.ndarray) -> dict:
    return {
        "rms":            float(np.sqrt(np.mean(signal ** 2))),
        "mean":           float(np.mean(signal)),
        "std":            float(np.std(signal)),
        "kurtosis":       float(kurtosis(signal)),
        "skewness":       float(skew(signal)),
        "peak":           float(np.max(np.abs(signal))),
        "peak_to_peak":   float(np.ptp(signal)),
        "crest_factor":   float(np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-10)),
        "shape_factor":   float(np.sqrt(np.mean(signal ** 2)) / (np.mean(np.abs(signal)) + 1e-10)),
        "impulse_factor": float(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10)),
    }


def extract_freq_features(signal: np.ndarray, fs: int = SAMPLING_RATE, n_bands: int = 5) -> dict:
    N       = len(signal)
    freqs   = np.fft.rfftfreq(N, 1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    spectral_centroid  = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
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
    mat = loadmat(io.BytesIO(file_bytes), simplify_cells=True)
    data_key = next(k for k in mat.keys() if not k.startswith("_"))
    data = mat[data_key]

    if isinstance(data, dict):
        Y = data.get("Y", [])
    else:
        Y = getattr(data, "Y", [])

    if Y is None:
        Y = []
    if isinstance(Y, np.ndarray):
        Y = Y.tolist()

    signals = {}
    for ch in Y:
        if not isinstance(ch, dict):
            continue
        name = ch.get("Name")
        data_arr = ch.get("Data")
        if name is None or data_arr is None:
            continue
        signals[name] = np.array(data_arr).flatten().astype(float)

    return {
        "vibration": signals.get("vibration_1",     np.zeros(256_001)),
        "current_1": signals.get("phase_current_1", np.zeros(256_001)),
        "current_2": signals.get("phase_current_2", np.zeros(256_001)),
    }


# ── Windowed feature extraction ────────────────────────────────────────────────
def features_from_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(len(_FEAT_KEYS), dtype=float)

    windows = [
        signal[start: start + WINDOW_SIZE]
        for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP)
    ]
    if not windows:
        windows = [signal[:WINDOW_SIZE] if len(signal) >= WINDOW_SIZE else signal]

    rows = [list(extract_all_features(w).values()) for w in windows]
    return np.mean(rows, axis=0)  # shape: (19,)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    file:      UploadFile = File(...),
    speed_rpm: float      = Form(...),
    torque_Nm: float      = Form(...),
    force_N:   float      = Form(...),
):
    # 1. Read and parse the .mat file
    file_bytes = await file.read()
    signals    = load_mat_bytes(file_bytes)

    # 2. Extract windowed features per channel (19 each → 57 total)
    vib_feats  = features_from_signal(signals["vibration"])
    cur1_feats = features_from_signal(signals["current_1"])
    cur2_feats = features_from_signal(signals["current_2"])

    # 3. Build the 60-feature feature vector
    feature_vector = np.concatenate([
        [speed_rpm, torque_Nm, force_N],
        vib_feats,
        cur1_feats,
        cur2_feats,
    ]).reshape(1, -1)

    # 4. Scale → PCA → predict using raw array input to avoid DataFrame feature name mismatch
    df_scaled = scaler.transform(feature_vector)
    df_pca    = pca.transform(df_scaled)

    predictions = model.predict(df_pca)[0]  # shape: (3,) — one value per target

    cond_pred = int(predictions[0])
    loc_pred  = int(predictions[1])
    dmg_pred  = int(predictions[2])

    # Probabilities for condition (most multi-output models expose proba per estimator)
    if hasattr(model, "estimators_"):
        # MultiOutputClassifier wraps individual estimators
        probabilities = model.estimators_[0].predict_proba(df_pca)[0].tolist()
    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_pca)[0].tolist()
    else:
        probabilities = [1.0 if i == cond_pred else 0.0 for i in range(3)]

    return {
        "condition": {
            "prediction":    cond_pred,
            "label":         LABEL_MAP.get(cond_pred, "Unknown"),
            "probabilities": probabilities,
        },
        "location": {
            "prediction": loc_pred,
            "label":      LOCATION_MAP.get(loc_pred, "Unknown"),
        },
        "damage_type": {
            "prediction": dmg_pred,
            "label":      DAMAGE_MAP.get(dmg_pred, "Unknown"),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)