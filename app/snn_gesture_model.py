import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, PoissonGroup,
    run, prefs, ms, Hz, Network
)
prefs.codegen.target = "numpy"

# --- Config ---
IMG_SIZE = 28
DATA_PATH = 'data/asl_alphabet_train/asl_alphabet_train'
CLASSES = ['A', 'B', 'C']
SAMPLES_PER_CLASS = 100
duration = 100 * ms
epochs = 5

# --- Load Data ---
def load_data(classes):
    X, y = [], []
    for label in classes:
        folder = os.path.join(DATA_PATH, label)
        for file in os.listdir(folder)[:SAMPLES_PER_CLASS]:
            img = Image.open(os.path.join(folder, file)).convert('L').resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img).flatten() / 255.0
            X.append(arr)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data(CLASSES)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

n_input = IMG_SIZE * IMG_SIZE
n_output = len(CLASSES)

# --- Create reusable Poisson encoder ---
def get_poisson_input(image):
    return PoissonGroup(n_input, rates=image * 100 * Hz)

# --- LIF neuron model ---
eqs = '''
dv/dt = -v / (10*ms) : 1
'''

# --- Trainable STDP synapses ---
def create_stdp_synapses(input_group, output_group):
    syn = Synapses(input_group, output_group,
        '''
        w : 1
        dpre/dt = -pre / (20*ms) : 1 (event-driven)
        dpost/dt = -post / (20*ms) : 1 (event-driven)
        ''',
        on_pre='''
        v_post += w
        pre = 1
        w = clip(w + post * 0.01, 0, 1)
        ''',
        on_post='''
        post = 1
        w = clip(w + pre * 0.01, 0, 1)
        '''
    )
    syn.connect(p=0.1)
    syn.w = 'rand() * 0.2'
    return syn

# --- Network Definition ---
output_neurons = NeuronGroup(n_output, eqs, threshold='v>1', reset='v=0', method='exact')
spike_mon = SpikeMonitor(output_neurons)

# --- Train over epochs ---
for epoch in range(epochs):
    print(f"🧠 Epoch {epoch+1}/{epochs}")
    for i in range(len(X_train)):
        inp = get_poisson_input(X_train[i])
        stdp = create_stdp_synapses(inp, output_neurons)

        net = Network(inp, output_neurons, stdp, spike_mon)
        spike_mon.count[:] = 0
        net.run(duration)

# --- Inference ---
predictions = []
for i in range(len(X_test)):
    inp = get_poisson_input(X_test[i])
    syn = Synapses(inp, output_neurons, on_pre='v_post += 0.3')
    syn.connect(p=0.1)
    net = Network(inp, output_neurons, syn, spike_mon)
    spike_mon.count[:] = 0
    net.run(duration)

    spike_counts = spike_mon.count
    pred = np.argmax(spike_counts)
    predictions.append(pred)

acc = accuracy_score(y_test, predictions)
print(f"✅ Final STDP SNN Accuracy on {len(predictions)} samples: {acc * 100:.2f}%")
