import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import random
#  SET THIS to your local dataset root folder
# ─────────────────────────────────────────────
DATASET_ROOT = './dataset'   # <-- change this path
SAMPLING_RATE = 64_000              # 64 kHz
WINDOW_SIZE   = 6400                # 0.1 second window
OVERLAP       = 0                # 50% overlap between windows
STEP          = int(WINDOW_SIZE * (1 - OVERLAP))


BEARING_META = {
    # Healthy
    'K001': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    'K002': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    'K003': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    'K004': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    'K005': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    'K006': {'condition': 0, 'label': 'Healthy', 'location': 'None',  'severity': 0, 'damage_type': 'None'},
    # Artificially Damaged
    'KA01': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 1, 'damage_type': 'EDM'},
    'KA03': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 2, 'damage_type': 'EDM'},
    'KA05': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 1, 'damage_type': 'Drilling'},
    'KA06': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 2, 'damage_type': 'Drilling'},
    'KA07': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 1, 'damage_type': 'EDM'},
    'KA08': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 2, 'damage_type': 'Engraving'},
    'KA09': {'condition': 1, 'label': 'Artificial', 'location': 'OR', 'severity': 2, 'damage_type': 'Engraving'},
    'KI01': {'condition': 1, 'label': 'Artificial', 'location': 'IR', 'severity': 1, 'damage_type': 'EDM'},
    'KI03': {'condition': 1, 'label': 'Artificial', 'location': 'IR', 'severity': 2, 'damage_type': 'EDM'},
    'KI05': {'condition': 1, 'label': 'Artificial', 'location': 'IR', 'severity': 1, 'damage_type': 'Drilling'},
    'KI07': {'condition': 1, 'label': 'Artificial', 'location': 'IR', 'severity': 1, 'damage_type': 'EDM'},
    'KI08': {'condition': 1, 'label': 'Artificial', 'location': 'IR', 'severity': 2, 'damage_type': 'Engraving'},
    # Real Damages (most important for car prediction)
    'KA04': {'condition': 2, 'label': 'Real',       'location': 'OR', 'severity': 1, 'damage_type': 'Fatigue'},
    'KA15': {'condition': 2, 'label': 'Real',       'location': 'OR', 'severity': 1, 'damage_type': 'Fatigue'},
    'KA16': {'condition': 2, 'label': 'Real',       'location': 'OR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KA22': {'condition': 2, 'label': 'Real',       'location': 'OR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KA30': {'condition': 2, 'label': 'Real',       'location': 'OR', 'severity': 2, 'damage_type': 'Plastic Def.'},
    'KB23': {'condition': 2, 'label': 'Real',       'location': 'OR+IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KB24': {'condition': 2, 'label': 'Real',       'location': 'OR+IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KB27': {'condition': 2, 'label': 'Real',       'location': 'OR+IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KI04': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 1, 'damage_type': 'Fatigue'},
    'KI14': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 1, 'damage_type': 'Fatigue'},
    'KI15': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KI16': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KI17': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KI18': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 2, 'damage_type': 'Fatigue'},
    'KI21': {'condition': 2, 'label': 'Real',       'location': 'IR', 'severity': 2, 'damage_type': 'Fatigue'},
}

print(f'Total bearing codes defined: {len(BEARING_META)}')
print(f'Window size: {WINDOW_SIZE} samples = {WINDOW_SIZE/SAMPLING_RATE*1000:.0f} ms')
print(f'Step size:   {STEP} samples ({int((1-OVERLAP)*100)}% non-overlap)')
def parse_filename(fname):
    """
    Parse a .mat filename like N15_M07_F10_KA01_1.mat
    Returns a dict with operating condition details.
    """
    base = os.path.splitext(fname)[0]
    parts = base.split('_')
    speed_map  = {'N15': 1500, 'N09': 900}
    torque_map = {'M07': 0.7,  'M01': 0.1}
    force_map  = {'F10': 1000, 'F04': 400}
    return {
        'speed_rpm':   speed_map.get(parts[0], None),
        'torque_Nm':   torque_map.get(parts[1], None),
        'force_N':     force_map.get(parts[2], None),
        'bearing_code': parts[3],
        'meas_index':  int(parts[4]) if len(parts) > 4 else None,
        'filename':    fname
    }


def load_mat_file(filepath):
    """
    Fixed loader based on actual Paderborn .mat file structure.
    Channels are in Y list, accessed by Name field.
    """
    mat = loadmat(filepath, simplify_cells=True)
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    data = mat[data_key]
    Y = data['Y']

    signals = {}
    for channel in Y:
        name = channel['Name']
        signals[name] = np.array(channel['Data']).flatten().astype(float)

    return {
        'vibration': signals.get('vibration_1', np.zeros(256001)),
        'current_1': signals.get('phase_current_1', np.zeros(256001)),
        'current_2': signals.get('phase_current_2', np.zeros(256001)),
        'speed':     signals.get('speed', np.zeros(16001)),
        'torque':    signals.get('torque', np.zeros(16001)),
        'force':     signals.get('force', np.zeros(16001)),
    }

def scan_dataset(root):
    """
    Walk through the dataset root and collect all .mat file paths.
    Returns a list of dicts with path + parsed metadata.
    """
    records = []
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('.mat'):
                meta = parse_filename(fname)
                bearing_code = meta['bearing_code']
                if bearing_code in BEARING_META:
                    meta.update(BEARING_META[bearing_code])
                    meta['filepath'] = os.path.join(folder_path, fname)
                    records.append(meta)
    return records


# Scan and show summary
def get_20_raw_points(root_path, output_name="ui_test_data.csv"):
    # 1. Scan for files
    records = []
    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                if fname.endswith('.mat'):
                    records.append(os.path.join(folder_path, fname))
    
    if not records:
        print("No files found!")
        return

    # 2. Pick a random file and load it
    filepath = random.choice(records)
    fname = os.path.basename(filepath)
    
    mat = loadmat(filepath, simplify_cells=True)
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    Y = mat[data_key]['Y']

    # 3. Pull all available signals (FILTERING OUT METADATA)
    signals = {}
    for channel in Y:
        name = channel['Name']
        data = np.array(channel['Data']).flatten()
        
        # ONLY take channels with more than 100 points to avoid the -15 error
        if len(data) > 100:
            signals[name] = data

    if not signals:
        print(f"No valid sensor data found in {fname}")
        return

    # 4. Find the shortest actual signal to set the random range
    # In Paderborn, Torque/Speed are ~16k points, Vibration is ~256k points.
    # min_length will now likely be 16001.
    min_length = min(len(s) for s in signals.values())
    
    # 5. Extract 20 points
    start_idx = random.randint(0, min_length - 21)
    
    extracted_data = {}
    for name, data in signals.items():
        # We slice every signal at the same starting point
        extracted_data[name] = data[start_idx : start_idx + 20]

    df = pd.DataFrame(extracted_data)

    # 6. Attach Metadata (Answers)
    meta_info = parse_filename(fname)
    bearing_code = meta_info['bearing_code']
    if bearing_code in BEARING_META:
        for key, value in BEARING_META[bearing_code].items():
            df[key] = value
    
    df['bearing_code'] = bearing_code
    df['source_file'] = fname

    # 7. Save
    df.to_csv(output_name, index=False)
    print(f"--- SUCCESS ---")
    print(f"File: {fname}")
    print(f"Extracted {len(signals)} channels: {list(signals.keys())}")

# Run it
get_20_raw_points(DATASET_ROOT)