import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Explicit Brian2 imports
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PoissonGroup,
    run, prefs, ms, Hz
)

# ---- Configuration ---- #
prefs.codegen.target = "numpy"  # fallback to numpy to avoid C++ errors
IMG_SIZE = 28
DATA_PATH = 'data/asl_alphabet_train/asl_alphabet_train'

CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I',        # Skipping 'J' due to motion
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# ---- Data Loading ---- #
def load_data(classes):
    X, y = [], []
    for label in classes:
        class_dir = os.path.join(DATA_PATH, label)
        images = os.listdir(class_dir)[:100]  # reduce samples for speed
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
            img_arr = np.array(img).flatten() / 255.0
            X.append(img_arr)
            y.append(label)
    return np.array(X), np.array(y)

# Load and encode data
X, y = load_data(CLASSES)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# ---- Network Parameters ---- #
n_input = IMG_SIZE * IMG_SIZE
n_output = len(CLASSES)
duration = 100 * ms

# ---- Poisson spike encoding ---- #
def encode_poisson(image):
    rates = image * 100 * Hz  # scaling factor
    return PoissonGroup(n_input, rates=rates)

# ---- Network Definition ---- #
eqs = '''
dv/dt = -v / (10*ms) : 1
'''

# ---- Prediction Loop ---- #
predictions = []

for i in range(len(X_test)):
    # Encode image to spikes
    spike_input = encode_poisson(X_test[i])

    # Define network fresh per sample
    output = NeuronGroup(n_output, eqs, threshold='v > 1', reset='v = 0', method='exact')
    synapses = Synapses(spike_input, output, on_pre='v_post += 0.05')
    synapses.connect(p=0.1)

    spike_mon = SpikeMonitor(output)

    # Run simulation
    run(duration)

    spike_counts = spike_mon.count
    pred = np.argmax(spike_counts)
    predictions.append(pred)

# ---- Accuracy ---- #
acc = accuracy_score(y_test[:len(predictions)], predictions)
print(f"🧠 SNN Accuracy on {len(predictions)} samples: {acc * 100:.2f}%")
