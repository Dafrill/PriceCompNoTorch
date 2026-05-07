import numpy as np

def gaussian_activation(x, sigma=0.5):
    """Gaussian: exp(-x^2 / (2*sigma^2)), max 1.0 at x=0."""
    return np.exp(-(x ** 2) / (2 * sigma ** 2))


def sigmoid_activation(x, slope=5.0):
    """Sigmoid-like: exp(-slope * |x|), max 1.0 at x=0."""
    return np.exp(-slope * np.abs(x))


def inverse_activation(x, alpha=1.0):
    """Inverse: 1/(1 + alpha*|x|), max 1.0 at x=0."""
    return 1.0 / (1.0 + alpha * np.abs(x))


def inverse_sq_activation(x, alpha=1.0):
    """Inverse squared: 1/(1 + alpha*x^2), max 1.0 at x=0."""
    return 1.0 / (1.0 + alpha * (x ** 2))


ACTIVATION_FUNCTIONS = {
    'gaussian': (gaussian_activation, {'sigma': 0.5}),
    'sigmoid': (sigmoid_activation, {'slope': 5.0}),
    'inverse': (inverse_activation, {'alpha': 1.0}),
    'inverse_sq': (inverse_sq_activation, {'alpha': 1.0}),
}


class KohonenNetwork:
    """First layer - cosine similarity in normalized space."""

    def __init__(self, n_neurons, input_dim):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.weights = np.random.uniform(-0.5, 0.5, size=(n_neurons, input_dim))
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to unit vectors."""
        norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.weights = self.weights / norms

    def get_cosine_similarities(self, x):
        """Return cosine similarity of x with all neurons."""
        is_1d = x.ndim == 1
        if is_1d:
            x = np.expand_dims(x, axis=0)

        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        w_norm = self.weights  # already normalized
        cos_sim = x_norm @ w_norm.T  # shape: (n_samples, n_neurons)

        if is_1d:
            return cos_sim[0]
        return cos_sim

    def get_winner(self, x):
        """Find winner based on maximum cosine similarity."""
        cos_sim = self.get_cosine_similarities(x)
        if x.ndim == 1:
            return np.argmax(cos_sim)
        return np.argmax(cos_sim, axis=1)

    def train_batch(self, x_batch, eta):
        """Update weights in cosine similarity space. Returns winners before update."""
        winners = self.get_winner(x_batch)
        x_norm = x_batch / (np.linalg.norm(x_batch, axis=1, keepdims=True) + 1e-10)

        if isinstance(winners, (int, np.integer)):
            winners_list = [winners]
        else:
            winners_list = list(winners)

        for j, winner in enumerate(winners_list):
            self.weights[winner] += eta * (x_norm[j] - self.weights[winner])
            norm = np.linalg.norm(self.weights[winner])
            if norm > 1e-10:
                self.weights[winner] /= norm

        return winners


class SecondLayerNetwork:
    """Second layer - each layer1 neuron has its own neurons (1D values).
    No activation here - activation is applied in WinnerTakesAllKohonen._transform_cosine().
    """

    def __init__(self, n_neurons_layer1, n_neurons_layer2):
        self.n_neurons_layer1 = n_neurons_layer1
        self.n_neurons_layer2 = n_neurons_layer2
        # Weights: shape (n_neurons_layer1, n_neurons_layer2)
        # Stores values in range [0, 1] (activated distances)
        self.weights = np.random.uniform(0, 1, size=(n_neurons_layer1, n_neurons_layer2))

    def get_winner(self, activated_score, winner1_idx):
        """Find winner2 among layer2 neurons belonging to winner1.
        activated_score: scalar from activation(1 - cos_sim), in range [0, 1]
        Returns index of neuron with weight closest to activated_score.
        """
        # Get layer2 neurons for winner1
        layer2_weights = self.weights[winner1_idx]  # shape: (n_neurons_layer2,)

        # Find neuron with weight closest to activated_score
        distances = np.abs(layer2_weights - activated_score)
        return np.argmin(distances)

    def train_batch(self, activated_score, winner1_idx, winner2_idx, eta):
        """Update the winning layer2 neuron's weight."""
        self.weights[winner1_idx, winner2_idx] += eta * (
            activated_score - self.weights[winner1_idx, winner2_idx]
        )
        # Clip to [0, 1] since activated_score is from activation function
        self.weights[winner1_idx, winner2_idx] = np.clip(
            self.weights[winner1_idx, winner2_idx], 0, 1
        )


class WinnerTakesAllKohonen:
    """Two-layer network: layer1 (cosine) -> activation -> layer2 (class)."""

    def __init__(self, n_neurons_l1, n_neurons_l2, input_dim, activation='gaussian',
                 sigma=0.5, slope=5.0, alpha=1.0, eta_init=0.9, eta_decay=0.95):
        self.n_neurons_l1 = n_neurons_l1
        self.n_neurons_l2 = n_neurons_l2
        self.input_dim = input_dim
        self.activation = activation
        self.sigma = sigma
        self.slope = slope
        self.alpha = alpha
        self.eta_init = eta_init
        self.eta_decay = eta_decay
        self.networks = {}

        # Activation function applied to (1 - cos_sim)
        self.activation_fn, self.activation_kwargs = ACTIVATION_FUNCTIONS.get(
            activation, (gaussian_activation, {'sigma': 0.5}))

    def create_network_for_type(self, reg_type):
        self.networks[reg_type] = {
            'layer1': KohonenNetwork(self.n_neurons_l1, self.input_dim),
            'layer2': SecondLayerNetwork(self.n_neurons_l1, self.n_neurons_l2)
        }

    def _transform_cosine(self, cos_sim):
        """Transform cosine similarity to activated distance: activation(1 - cos_sim)."""
        dist = 1.0 - cos_sim
        return self.activation_fn(dist, **self.activation_kwargs)

    def train(self, data_by_type, epochs=100):
        for reg_type, data_list in data_by_type.items():
            if reg_type not in self.networks:
                continue

            net = self.networks[reg_type]
            layer1 = net['layer1']
            layer2 = net['layer2']
            data_array = [item['features'] for item in data_list]
            features_array = np.array(data_array, dtype=np.float32)
            n_samples = len(features_array)
            batch_size = 32

            for epoch in range(epochs):
                eta = self.eta_init * (self.eta_decay ** epoch)
                perm = np.random.permutation(n_samples)

                for i in range(0, n_samples, batch_size):
                    batch_idx = perm[i:i + batch_size]
                    x_batch = features_array[batch_idx]

                    # Train layer1, get winners (before weight update)
                    winners_l1 = layer1.train_batch(x_batch, eta)

                    # Use the same winners for layer2
                    for j in range(len(x_batch)):
                        winner1 = winners_l1[j] if not isinstance(winners_l1, (int, np.integer)) else winners_l1
                        x = x_batch[j]
                        cos_sim = layer1.get_cosine_similarities(x)[winner1]
                        transformed = self._transform_cosine(cos_sim)

                        # Get winner2 from layer2 (only winner1's neurons)
                        winner2 = layer2.get_winner(transformed, winner1)

                        # Train layer2
                        layer2.train_batch(transformed, winner1, winner2, eta)

            print(f"Trained {reg_type}: {len(data_list)} samples")

    def predict(self, reg_type, x):
        """Return (winner1, winner2, activated_score) where activated_score is from activation(1 - cos_sim)."""
        if reg_type not in self.networks:
            return None, None, None
        net = self.networks[reg_type]
        layer1 = net['layer1']
        layer2 = net['layer2']

        winner1 = layer1.get_winner(x)
        cos_sim = layer1.get_cosine_similarities(x)[winner1]
        activated_score = self._transform_cosine(cos_sim)

        winner2 = layer2.get_winner(activated_score, winner1)

        return winner1, winner2, activated_score
