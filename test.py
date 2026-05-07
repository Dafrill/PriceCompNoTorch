import numpy as np
from utils import load_stock_data, FEATURE_NAMES, N_FEATURES, OUTPUT_DIR
from winTakesAll import (
    KohonenNetwork, SecondLayerNetwork, WinnerTakesAllKohonen,
    ACTIVATION_FUNCTIONS, gaussian_activation
)
import generate_rmd
import os
import time
import json
import csv
import subprocess
import sys

TEST_CONFIGS = [
    # === SERIES 1: impact of layer 1 neuron count (layer 2: 5 neurons) ===
    {
        'name': 'S1_A: L1=10, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 10, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
        'labels_per_neuron': 1,
    },
    {
        'name': 'S1_B: L1=20, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 20, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S1_C: L1=40, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S1_D: L1=60, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 60, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S1_E: L1=80, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 80, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 2: impact of layer 2 neuron count (layer 1: 40 neurons) ===
    {
        'name': 'S2_A: L1=40, L2=0 (1-layer), gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S2_B: L1=40, L2=3, gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 3, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S2_C: L1=40, L2=5, gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S2_D: L1=40, L2=10, gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 10, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S2_E: L1=40, L2=20, gaussian, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 20, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 3: impact of learning rate ===
    {
        'name': 'S3_A: L1=40, L2=5, lr=0.3',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.3, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S3_B: L1=40, L2=5, lr=0.5',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.5, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S3_C: L1=40, L2=5, lr=0.7',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.7, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S3_D: L1=40, L2=5, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 4: impact of decay ===
    {
        'name': 'S4_A: L1=40, L2=5, decay=0.90',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.90, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S4_B: L1=40, L2=5, decay=0.95',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S4_C: L1=40, L2=5, decay=0.98',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.98, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S4_D: L1=40, L2=5, decay=0.99',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.99, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 5: impact of epochs ===
    {
        'name': 'S5_A: L1=40, L2=5, 50 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 50, 'repetitions': 5,
    },
    {
        'name': 'S5_B: L1=40, L2=5, 100 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S5_C: L1=40, L2=5, 200 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 200, 'repetitions': 5,
    },
    {
        'name': 'S5_D: L1=40, L2=5, 500 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 500, 'repetitions': 5,
    },
    # === SERIES 6: different activation functions (4 values) ===
    {
        'name': 'S6_A: L1=40, L2=5, gaussian',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S6_B: L1=40, L2=5, sigmoid',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'sigmoid',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S6_C: L1=40, L2=5, inverse',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'inverse',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S6_D: L1=40, L2=5, inverse_sq',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'inverse_sq',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 7: impact of train ratio (4 values) ===
    {
        'name': 'S7_A: L1=40, L2=5, train=60%',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.60, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S7_B: L1=40, L2=5, train=70%',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.70, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S7_C: L1=40, L2=5, train=80%',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S7_D: L1=40, L2=5, train=90%',
        'n_neurons_l1': 40, 'n_neurons_l2': 5, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.90, 'epochs': 100, 'repetitions': 5,
    },
    # === SERIES 8-12: 1-layer network tests (L2=0) ===
    # S8: impact of neuron count (1-layer)
    {
        'name': 'S8_A: 1-layer, L1=10',
        'n_neurons_l1': 10, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S8_B: 1-layer, L1=20',
        'n_neurons_l1': 20, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S8_C: 1-layer, L1=40',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S8_D: 1-layer, L1=80',
        'n_neurons_l1': 80, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # S9: impact of learning rate (1-layer)
    {
        'name': 'S9_A: 1-layer, lr=0.3',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.3, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S9_B: 1-layer, lr=0.5',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.5, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S9_C: 1-layer, lr=0.7',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.7, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S9_D: 1-layer, lr=0.9',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    # S10: impact of epochs (1-layer)
    {
        'name': 'S10_A: 1-layer, 50 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 50, 'repetitions': 5,
    },
    {
        'name': 'S10_B: 1-layer, 100 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S10_C: 1-layer, 200 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 200, 'repetitions': 5,
    },
    {
        'name': 'S10_D: 1-layer, 500 epochs',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 500, 'repetitions': 5,
    },
    # S11: impact of train ratio (1-layer)
    {
        'name': 'S11_A: 1-layer, train=60%',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.60, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S11_B: 1-layer, train=70%',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.70, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S11_C: 1-layer, train=80%',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
    {
        'name': 'S11_D: 1-layer, train=90%',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.90, 'epochs': 100, 'repetitions': 5,
    },
    # S12: different activation functions (1-layer - only gaussian makes sense for 1-layer)
    {
        'name': 'S12_A: 1-layer, gaussian',
        'n_neurons_l1': 40, 'n_neurons_l2': 0, 'activation': 'gaussian',
        'eta_init': 0.9, 'eta_decay': 0.95, 'train_ratio': 0.80, 'epochs': 100, 'repetitions': 5,
    },
]


class ClassificationKohonen1Layer:
    """One-layer Kohonen: separate network per regression type (cosine similarity)."""

    def __init__(self, n_neurons_l1, input_dim, activation='gaussian',
                 sigma=0.5, slope=5.0, alpha=1.0,
                 eta_init=0.9, eta_decay=0.95, labels_per_neuron=1):
        self.n_neurons_l1 = n_neurons_l1
        self.input_dim = input_dim
        self.activation = activation
        self.eta_init = eta_init
        self.eta_decay = eta_decay
        self.labels_per_neuron = labels_per_neuron

        self.activation_fn, self.activation_kwargs = ACTIVATION_FUNCTIONS.get(
            activation, (gaussian_activation, {'sigma': 0.5}))

        self.networks = {}

    def _transform_cosine(self, cos_sim):
        dist = 1.0 - cos_sim
        return self.activation_fn(dist, **self.activation_kwargs)

    def create_network_for_type(self, reg_type):
        self.networks[reg_type] = {
            'layer1': KohonenNetwork(self.n_neurons_l1, self.input_dim),
        }

    def train(self, train_data, epochs=100):
        """Train separate network for each regression type."""
        data_by_type = {}
        for item in train_data:
            reg_type = item['reg_type']
            if reg_type not in data_by_type:
                data_by_type[reg_type] = []
            data_by_type[reg_type].append(item)

        for reg_type, data_list in data_by_type.items():
            self.create_network_for_type(reg_type)
            net = self.networks[reg_type]['layer1']
            features_array = np.array([item['features'] for item in data_list], dtype=np.float32)
            n_samples = len(features_array)
            batch_size = 32

            for epoch in range(epochs):
                eta = self.eta_init * (self.eta_decay ** epoch)
                perm = np.random.permutation(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_idx = perm[i:i + batch_size]
                    x_batch = features_array[batch_idx]
                    net.train_batch(x_batch, eta)

            print(f"Trained 1-layer {reg_type}: {n_samples} samples")

    def predict(self, x):
        """Predict class by finding the best-matching network (highest cosine similarity)."""
        x = np.array(x, dtype=np.float32)
        best_reg_type = None
        best_score = -1
        best_winner = None

        for reg_type in self.networks:
            layer1 = self.networks[reg_type]['layer1']
            cos_sims = layer1.get_cosine_similarities(x)
            winner = np.argmax(cos_sims)
            score = cos_sims[winner]

            if score > best_score:
                best_score = score
                best_reg_type = reg_type
                best_winner = winner

        return best_winner, best_reg_type

    def evaluate(self, data):
        """Calculate cosine similarity of winning neuron in the winning network."""
        cosine_sims = []

        for item in data:
            x = np.array(item['features'], dtype=np.float32)
            true_reg_type = item['reg_type']

            winner, pred_reg_type = self.predict(x)

            if pred_reg_type is None:
                continue

            # Get cosine similarity of the winner in the winning network
            layer1 = self.networks[pred_reg_type]['layer1']
            cos_sims = layer1.get_cosine_similarities(x)
            score = cos_sims[winner]
            cosine_sims.append(score)

        return {
            'cosine_mean': np.mean(cosine_sims) if cosine_sims else 0,
            'cosine_std': np.std(cosine_sims) if cosine_sims else 0,
        }


class ClassificationKohonen:
    """Two-layer Kohonen WTA: layer1 (cosine) -> activation -> layer2 (class)."""

    def __init__(self, n_neurons_l1, n_neurons_l2, input_dim, activation='gaussian',
                 sigma=0.5, slope=5.0, alpha=1.0,
                 eta_init=0.9, eta_decay=0.95):
        self.n_neurons_l1 = n_neurons_l1
        self.n_neurons_l2 = n_neurons_l2
        self.input_dim = input_dim
        self.activation = activation
        self.eta_init = eta_init
        self.eta_decay = eta_decay

        self.net = WinnerTakesAllKohonen(
            n_neurons_l1, n_neurons_l2, input_dim,
            activation=activation, sigma=sigma, slope=slope, alpha=alpha,
            eta_init=eta_init, eta_decay=eta_decay
        )

    def train(self, train_data, epochs=100):
        """Train the network on mixed data."""
        # Group data by reg_type
        data_by_type = {}
        for item in train_data:
            reg_type = item['reg_type']
            if reg_type not in data_by_type:
                data_by_type[reg_type] = []
            data_by_type[reg_type].append(item)

        # Create networks for each type
        for reg_type in data_by_type:
            self.net.create_network_for_type(reg_type)

        # Train
        self.net.train(data_by_type, epochs=epochs)

    def predict(self, x):
        """Predict class for a new item. Returns (winner1, winner2), predicted_reg_type.
        Uses layer2 matching (distance between activated_score and layer2 weight) for decision."""
        best_reg_type = None
        best_score = -1  # higher is better (closer match in layer2)
        best_winner1 = None
        best_winner2 = None

        for reg_type in self.net.networks:
            winner1, winner2, activated_score = self.net.predict(reg_type, np.array(x, dtype=np.float32))
            if winner1 is None:
                continue

            # Get layer2 weight for the winning neuron
            layer2_net = self.net.networks[reg_type]['layer2']
            weight_winner2 = layer2_net.weights[winner1, winner2]

            # Score based on layer2: higher when activated_score is close to weight
            layer2_score = 1.0 - abs(activated_score - weight_winner2)

            if layer2_score > best_score:
                best_score = layer2_score
                best_reg_type = reg_type
                best_winner1 = winner1
                best_winner2 = winner2

        return (best_winner1, best_winner2), best_reg_type

    def evaluate(self, data):
        """Calculate cosine similarity of winning neuron in the winning network."""
        cosine_sims = []

        for item in data:
            x = np.array(item['features'], dtype=np.float32)
            true_reg_type = item['reg_type']

            (pred_w1, pred_w2), pred_reg_type = self.predict(x)

            if pred_reg_type is None:
                continue

            # Get cosine similarity of the winner in the winning network
            layer1 = self.net.networks[pred_reg_type]['layer1']
            cos_sims = layer1.get_cosine_similarities(x)
            score = cos_sims[pred_w1]
            cosine_sims.append(score)

        return {
            'cosine_mean': np.mean(cosine_sims) if cosine_sims else 0,
            'cosine_std': np.std(cosine_sims) if cosine_sims else 0,
        }


def run_config(config, data, seed, compare_1layer=False):
    """Uruchom jedną konfigurację."""
    np.random.seed(seed)

    data_copy = [dict(item) for item in data]
    np.random.shuffle(data_copy)

    n_train = int(len(data_copy) * config['train_ratio'])
    train_data = data_copy[:n_train]
    test_data = data_copy[n_train:]

    results = {}

    # Zawsze testuj sieć 2-warstwową (lub 1-warstwową gdy L2=0)
    if config['n_neurons_l2'] == 0:
        net2 = ClassificationKohonen1Layer(
            n_neurons_l1=config['n_neurons_l1'],
            input_dim=N_FEATURES,
            activation=config['activation'],
            eta_init=config['eta_init'],
            eta_decay=config['eta_decay'],
        )
    else:
        net2 = ClassificationKohonen(
            n_neurons_l1=config['n_neurons_l1'],
            n_neurons_l2=config['n_neurons_l2'],
            input_dim=N_FEATURES,
            activation=config['activation'],
            eta_init=config['eta_init'],
            eta_decay=config['eta_decay'],
        )

    net2.train(train_data, epochs=config['epochs'])
    train_res2 = net2.evaluate(train_data)
    test_res2 = net2.evaluate(test_data)

    results['2layer'] = {
        'train_cosine_mean': train_res2['cosine_mean'],
        'train_cosine_std': train_res2['cosine_std'],
        'test_cosine_mean': test_res2['cosine_mean'],
        'test_cosine_std': test_res2['cosine_std'],
    }

    if compare_1layer and config['n_neurons_l2'] > 0:
        net1 = ClassificationKohonen1Layer(
            n_neurons_l1=config['n_neurons_l1'],
            input_dim=N_FEATURES,
            activation=config['activation'],
            eta_init=config['eta_init'],
            eta_decay=config['eta_decay'],
        )
        net1.train(train_data, epochs=config['epochs'])
        train_res1 = net1.evaluate(train_data)
        test_res1 = net1.evaluate(test_data)

        results['1layer'] = {
            'train_cosine_mean': train_res1['cosine_mean'],
            'train_cosine_std': train_res1['cosine_std'],
            'test_cosine_mean': test_res1['cosine_mean'],
            'test_cosine_std': test_res1['cosine_std'],
        }

    return results


def run_comparison(data, seed):
    """Porównaj sieć 1-warstwową z 2-warstwową."""
    np.random.seed(seed)

    data_copy = [dict(item) for item in data]
    np.random.shuffle(data_copy)

    train_ratio = 0.80
    n_train = int(len(data_copy) * train_ratio)
    train_data = data_copy[:n_train]
    test_data = data_copy[n_train:]

    config_base = {
        'n_neurons_l1': 40,
        'n_neurons_l2': 5,
        'activation': 'gaussian',
        'eta_init': 0.9,
        'eta_decay': 0.95,
        'epochs': 100,
    }

    # Sieć 2-warstwowa
    net2 = ClassificationKohonen(
        n_neurons_l1=config_base['n_neurons_l1'],
        n_neurons_l2=config_base['n_neurons_l2'],
        input_dim=N_FEATURES,
        activation=config_base['activation'],
        eta_init=config_base['eta_init'],
        eta_decay=config_base['eta_decay'],
    )
    net2.train(train_data, epochs=config_base['epochs'])
    res2_train = net2.evaluate(train_data)
    res2_test = net2.evaluate(test_data)

    # Sieć 1-warstwowa
    net1 = ClassificationKohonen1Layer(
        n_neurons_l1=config_base['n_neurons_l1'],
        input_dim=N_FEATURES,
        activation=config_base['activation'],
        eta_init=config_base['eta_init'],
        eta_decay=config_base['eta_decay'],
    )
    net1.train(train_data, epochs=config_base['epochs'])
    res1_train = net1.evaluate(train_data)
    res1_test = net1.evaluate(test_data)

    return {
        '1layer_train_cosine': res1_train['cosine_mean'],
        '1layer_test_cosine': res1_test['cosine_mean'],
        '2layer_train_cosine': res2_train['cosine_mean'],
        '2layer_test_cosine': res2_test['cosine_mean'],
    }


def main():
    print("Loading stock data...")
    data = load_stock_data()
    print(f"Loaded {len(data)} stocks")
    print(f"Features ({N_FEATURES}): {FEATURE_NAMES}")

    if not data:
        print("ERROR: No data loaded!")
        return

    reg_type_counts = {}
    for item in data:
        rt = item['reg_type']
        reg_type_counts[rt] = reg_type_counts.get(rt, 0) + 1
    print(f"Classes: {reg_type_counts}")

    results = []

    for config in TEST_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 60}")

        # Test 2-layer (or 1-layer if L2=0)
        train_cosines = []
        test_cosines = []

        for rep in range(config['repetitions']):
            seed = 42 + rep * 17
            start = time.time()
            res = run_config(config, data, seed)
            elapsed = time.time() - start

            train_cosines.append(res['2layer']['train_cosine_mean'])
            test_cosines.append(res['2layer']['test_cosine_mean'])

            print(f"  Rep {rep + 1}: train_cosine={res['2layer']['train_cosine_mean']:.4f}, test_cosine={res['2layer']['test_cosine_mean']:.4f} ({elapsed:.1f}s)")

        result_2layer = {
            'config': config['name'],
            'network_type': '1-layer' if config['n_neurons_l2'] == 0 else '2-layer',
            'n_layers': 1 if config['n_neurons_l2'] == 0 else 2,
            'n_neurons_l1': config['n_neurons_l1'],
            'n_neurons_l2': config['n_neurons_l2'],
            'n_neurons': config['n_neurons_l1'] + config['n_neurons_l2'],
            'activation': config['activation'],
            'eta_init': config['eta_init'],
            'eta_decay': config['eta_decay'],
            'epochs': config['epochs'],
            'train_ratio': config['train_ratio'],
            'TrainCosine': np.mean(train_cosines),
            'TrainCosineStd': np.std(train_cosines),
            'TestCosine': np.mean(test_cosines),
            'TestCosineStd': np.std(test_cosines),
        }
        results.append(result_2layer)

        # Skip 1-layer generation for Series 6 (activation functions don't make sense for 1-layer)
        is_series6 = config['name'].startswith('S6_')
        
        if config['n_neurons_l2'] > 0 and not is_series6:
            config_1layer = dict(config)
            # Rebuild name without L2 info for 1-layer
            original_name = config['name']
            # Extract parts: remove L2 info, keep the rest
            name_parts = original_name.split(': ')
            if len(name_parts) == 2:
                prefix = name_parts[0]  # e.g., "S1_E"
                desc = name_parts[1]
                import re
                # Remove L1=X and L2=X from description
                desc_clean = re.sub(r'(L1|L2)=\d+,?\s*', '', desc)
                desc_clean = desc_clean.strip(', ').strip()
                config_1layer['name'] = f"{prefix}: {desc_clean} (1-layer)"
            else:
                config_1layer['name'] = config['name'] + ' (1-layer)'
            config_1layer['n_neurons_l2'] = 0

            train_cosines1 = []
            test_cosines1 = []

            for rep in range(config['repetitions']):
                seed = 42 + rep * 17
                res = run_config(config_1layer, data, seed)
                train_cosines1.append(res['2layer']['train_cosine_mean'])
                test_cosines1.append(res['2layer']['test_cosine_mean'])

            result_1layer = {
                'config': config_1layer['name'],
                'network_type': '1-layer',
                'n_layers': 1,
                'n_neurons_l1': config_1layer['n_neurons_l1'],
                'n_neurons_l2': 0,
                'n_neurons': config_1layer['n_neurons_l1'],
                'activation': config_1layer['activation'],
                'eta_init': config_1layer['eta_init'],
                'eta_decay': config_1layer['eta_decay'],
                'epochs': config_1layer['epochs'],
                'train_ratio': config_1layer['train_ratio'],
                'TrainCosine': np.mean(train_cosines1),
                'TrainCosineStd': np.std(train_cosines1),
                'TestCosine': np.mean(test_cosines1),
                'TestCosineStd': np.std(test_cosines1),
            }
            results.append(result_1layer)

    # === SUMMARY ===
    print(f"\n\n{'=' * 80}")
    print("RESULTS SUMMARY (Distance and Ratio)")
    print(f"{'=' * 80}")

    # Series 1: layer 1 neuron count
    print("\n=== SERIES 1: Number of L1 Neurons (L2=5) ===")
    print(f"{'L1':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S1_'):
            print(f"{r['n_neurons_l1']:>10} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 2: layer 2 neuron count
    print("\n=== SERIES 2: Number of L2 Neurons (L1=40) ===")
    print(f"{'L2':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S2_'):
            print(f"{r['n_neurons_l2']:>10} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 3: learning rate
    print("\n=== SERIES 3: Learning Rate (L1=40, L2=5) ===")
    print(f"{'Eta Init':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S3_'):
            print(f"{r['eta_init']:>10.1f} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 4: decay
    print("\n=== SERIES 4: Learning Rate Decay (L1=40, L2=5) ===")
    print(f"{'Eta Decay':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S4_'):
            print(f"{r['eta_decay']:>10.2f} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 5: epochs
    print("\n=== SERIES 5: Number of Epochs (L1=40, L2=5) ===")
    print(f"{'Epochs':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S5_'):
            print(f"{r['epochs']:>10} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 6: different activation functions
    print("\n=== SERIES 6: Different Activation Functions (L1=40, L2=5) ===")
    print(f"{'Config':>8} | {'L1':>8} | {'L2':>8} | {'Activ.':>10} | {'TestCosine':>11}")
    print("-" * 60)
    for r in results:
        if r['config'].startswith('S6_'):
            name = r['config'].split(': ')[0]
            print(f"{name:>8} | {r['n_neurons_l1']:>8} | {r['n_neurons_l2']:>8} | {r['activation']:>10} | {r['TestCosine']:>11.4f}")

    # Series 7: train ratio
    print("\n=== SERIES 7: Train Ratio (L1=40, L2=5) ===")
    print(f"{'Ratio':>8} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S7_'):
            print(f"{r['train_ratio']:>8.2f} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 8: 1-layer neuron count
    print("\n=== SERIES 8: 1-Layer Neuron Count ===")
    print(f"{'L1':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S8_'):
            print(f"{r['n_neurons_l1']:>10} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 9: 1-layer learning rate
    print("\n=== SERIES 9: 1-Layer Learning Rate ===")
    print(f"{'LR':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S9_'):
            print(f"{r['eta_init']:>10.1f} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 10: 1-layer epochs
    print("\n=== SERIES 10: 1-Layer Number of Epochs ===")
    print(f"{'Epochs':>10} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S10_'):
            print(f"{r['epochs']:>10} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 11: 1-layer train ratio
    print("\n=== SERIES 11: 1-Layer Train Ratio ===")
    print(f"{'Ratio':>8} | {'TrainCosine':>12} | {'TestCosine':>11}")
    print("-" * 40)
    for r in results:
        if r['config'].startswith('S11_'):
            print(f"{r['train_ratio']:>8.2f} | {r['TrainCosine']:>12.4f} | {r['TestCosine']:>11.4f}")

    # Series 12: 1-layer activation (only gaussian)
    print("\n=== SERIES 12: 1-Layer Activation (gaussian only) ===")
    print(f"{'Config':>8} | {'L1':>8} | {'Activ.':>10} | {'TestCosine':>11}")
    print("-" * 60)
    for r in results:
        if r['config'].startswith('S12_'):
            name = r['config'].split(': ')[0]
            print(f"{name:>8} | {r['n_neurons_l1']:>8} | {r['activation']:>10} | {r['TestCosine']:>11.4f}")

    # Best result - highest cosine is best
    best_test = max(results, key=lambda r: r['TestCosine'])
    print(f"\n{'=' * 60}")
    print(f"Best configuration (highest cosine): {best_test['config']}")
    print(f"Test Cosine: {best_test['TestCosine']:.4f} +/- {best_test['TestCosineStd']:.4f}")
    print(f"{'=' * 60}")

    # Zapisz najlepszą konfigurację do JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_config_path = os.path.join(OUTPUT_DIR, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_test, f, indent=2)
    print(f"Saved best config to: {best_config_path}")

    # Dodaj ShortConfig do każdego wyniku
    for r in results:
        r['ShortConfig'] = r['config'].split(': ')[0]

    # Zapisz wyniki do pliku R Markdown
    generate_rmd.save_results_rmd(results, best_test)

    # Porównanie z siecią 1-warstwową
    print("\n=== COMPARISON: 1-Layer vs 2-Layer Network ===")
    comp_seed = 42
    comp_res = run_comparison(data, comp_seed)
    print(f"1-Layer Train Cosine: {comp_res['1layer_train_cosine']:.4f}")
    print(f"1-Layer Test Cosine:  {comp_res['1layer_test_cosine']:.4f}")
    print(f"2-Layer Train Cosine: {comp_res['2layer_train_cosine']:.4f}")
    print(f"2-Layer Test Cosine:  {comp_res['2layer_test_cosine']:.4f}")

    # Generuj PDF z wykresami dla najlepszej konfiguracji
    print("\n=== Generating PDF for best configuration ===")
    try:
        subprocess.run([sys.executable, "view_summary.py"], check=True, cwd="/home/magda/network2")
    except subprocess.CalledProcessError as e:
        print(f"ERROR generating PDF: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
