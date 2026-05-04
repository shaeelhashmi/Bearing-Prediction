from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Missing import added
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
app = FastAPI()

# Fix: Remove the leading space in the filename
REPO_ID = "shaeel12/Bearing_problem_prediction"
FILENAME = "bearing_diagnostic_model_v1.pkl" 

# Download the model
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model = joblib.load(model_path)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def extract_time_features(signal):
    """Extract statistical time-domain features from a signal window."""
    return {
        'rms':          np.sqrt(np.mean(signal ** 2)),
        'mean':         np.mean(signal),
        'std':          np.std(signal),
        'kurtosis':     kurtosis(signal),
        'skewness':     skew(signal),
        'peak':         np.max(np.abs(signal)),
        'peak_to_peak': np.ptp(signal),
        'crest_factor': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-10),
        'shape_factor': np.sqrt(np.mean(signal ** 2)) / (np.mean(np.abs(signal)) + 1e-10),
        'impulse_factor': np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10),
    }

SAMPLING_RATE = 64_000   
def extract_freq_features(signal, fs=SAMPLING_RATE, n_bands=5):
    """Extract frequency-domain features using FFT and PSD."""
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    # Spectral centroid
    spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)

    # Spectral bandwidth
    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-10)
    )

    # PSD energy in frequency bands
    f_psd, Pxx = welch(signal, fs=fs, nperseg=min(512, N))
    band_edges = np.linspace(0, fs / 2, n_bands + 1)
    band_energies = {}
    for i in range(n_bands):
        mask = (f_psd >= band_edges[i]) & (f_psd < band_edges[i + 1])
        band_energies[f'band_energy_{i}'] = np.sum(Pxx[mask])

    features = {
        'spectral_centroid':   spectral_centroid,
        'spectral_bandwidth':  spectral_bandwidth,
        'spectral_energy':     np.sum(fft_mag ** 2),
        'dominant_freq':       freqs[np.argmax(fft_mag)],
    }
    features.update(band_energies)
    return features


def extract_all_features(signal, fs=SAMPLING_RATE):
    """Combine time and frequency features."""
    feats = {}
    feats.update(extract_time_features(signal))
    feats.update(extract_freq_features(signal, fs))
    return feats

@app.post("/predict")
def predict(data: dict):
    # 1. Get raw signals and conditions from the JSON payload
    # This assumes your frontend sends arrays of 6400 samples for signals
    vib_signal = np.array(data['vibration_raw'])
    cur1_signal = np.array(data['current1_raw'])
    cur2_signal = np.array(data['current2_raw'])
    
    # 2. Extract 19 features for EACH channel (Total 57)
    vib_feats = extract_all_features(vib_signal)
    cur1_feats = extract_all_features(cur1_signal)
    cur2_feats = extract_all_features(cur2_signal)
    
    # 3. Create the final 60-feature vector in the EXACT order used in training
    # Order: [Conditions] + [Vibration Stats] + [Current1 Stats] + [Current2 Stats]
    feature_vector = [
        data['speed_rpm'], data['torque_Nm'], data['force_N']
    ]
    feature_vector.extend(list(vib_feats.values()))
    feature_vector.extend(list(cur1_feats.values()))
    feature_vector.extend(list(cur2_feats.values()))

    # 4. Convert to DataFrame and predict
    df = pd.DataFrame([feature_vector])
    prediction = model.predict(df.values)
    return {
        "status": "success",
        "predictions": {
            "fault_type": int(prediction[0]),
            "severity": int(prediction[1]),
            "confidence_score": float(prediction[2])
        }
    }