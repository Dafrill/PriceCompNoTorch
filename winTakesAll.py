
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



FOLDER = "./stocks.cleaned/"

N_POINTS_LAST = 3000
FEATURE_NAMES = ['trend', 'mape', 'rmse', 'q_slope', 'highest_p', 'curvature']
N_FEATURES = len(FEATURE_NAMES)

OUTPUT_DIR = "./results/"


class KohonenNetwork():
    def __init__(self, n_neurons, input_dim):
        super().__init__()
        self.weights = np.random.uniform(-0.5, 0.5, size=(n_neurons, input_dim))
           
    def get_winner(self, x):
        is_1d = x.ndim == 1
        if is_1d:
            x = np.expand_dims(x, axis=0)     
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        w_norm = self.weights / (np.linalg.norm(self.weights, axis=1, keepdims=True) + 1e-10)
        cos_sim = x_norm @ w_norm.T 
        winners = np.argmax(cos_sim, axis=1)
        if is_1d or (winners.ndim == 1 and len(winners) == 1):
            return winners.item() 
        return winners

class WinnerTakesAllKohonen:
    def __init__(self, n_neurons, input_dim):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.networks = {}
    def create_network_for_type(self, reg_type):
        self.networks[reg_type] = KohonenNetwork(self.n_neurons, self.input_dim)
    def train(self, data_by_type, epochs=100):
        for reg_type, data_list in data_by_type.items():
            if reg_type not in self.networks:
                continue
            
            net = self.networks[reg_type]
            data_array = [item['features'] for item in data_list]
            
            features_array = np.array(data_array, dtype=np.float32)
            n_samples = len(features_array)
            batch_size = 32
            
            for epoch in range(epochs):
                eta = 0.9 * (0.95 ** epoch)
                perm = np.random.permutation(n_samples)
                
                for i in range(0, n_samples, batch_size):
                    batch_idx = perm[i:i+batch_size]
                    x_batch = features_array[batch_idx]
                    
                    winners = net.get_winner(x_batch)
                    
                    if isinstance(winners, (int, np.integer)):
                        winners_list = [winners]
                    else:
                        winners_list = list(winners)
                    
                    for j, winner in enumerate(winners_list):
                        net.weights[winner] += eta * (x_batch[j] - net.weights[winner])
                        net.weights[winner] /= (np.linalg.norm(net.weights[winner]) + 1e-10)
                        
            print(f"Trained {reg_type}: {len(data_list)} samples")
    
    def predict(self, reg_type, x):
        if reg_type not in self.networks:
            return None
        return self.networks[reg_type].get_winner(x)



def _plot_worker_kohonen(args):
    filename, folder, features, winner, reg_type, base_folder, n_points, feat_names = args
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import os
    try:
        y = pd.read_csv(os.path.join(base_folder, filename)).iloc[:, 0].values.astype(float)
        y = y[~np.isnan(y) & ~np.isinf(y)]
        y = y[-n_points:]
        n = len(y)
        X = np.arange(n)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y, linewidth=1, color='blue', label='Data')
        
       
        X_train = X[:int(n*0.8)].reshape(-1, 1)
        y_train = y[:int(n*0.8)]
        
        if reg_type == "Linear":
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X.reshape(-1, 1))
        elif reg_type == "Exponential":
            y_safe = np.maximum(y_train, 1e-10)
            y_log = np.log(y_safe)
            model = LinearRegression().fit(X_train, y_log)
            y_pred = np.exp(model.predict(X.reshape(-1, 1)))
        elif reg_type == "Polynomial":
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2)
            X_train_poly = poly.fit_transform(X_train)
            model = LinearRegression().fit(X_train_poly, y_train)
            X_poly = poly.transform(X.reshape(-1, 1))
            y_pred = model.predict(X_poly)
        else:
            y_pred = y
        
        ax.plot(y_pred, linewidth=2, color='red', alpha=0.7, label=f'{reg_type} fit')
        
        title_parts = []
        for i, name in enumerate(feat_names):
            val = features[i] if i < len(features) else 0
            if name == 'curvature':
                title_parts.append(f"{name}:{val:.8f}")
            elif name in ('rmse', 'q_slope', 'trend'):
                title_parts.append(f"{name}:{val:.4f}")
            else:
                title_parts.append(f"{name}:{val:.2f}")
        
        ax.set_title(f"{filename}\nType:{reg_type} | Neuron:N{winner}\n" + " | ".join(title_parts), fontsize=9)
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(folder, filename + ".png"), dpi=80, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"ERROR saving {filename}: {e}")


def save_results(kohonen, data_by_type, rt):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plot_args = []
    csv_rows = []
    for reg_type, data_list in data_by_type.items():
        if reg_type not in kohonen.networks:
            print(f"WARNING: {reg_type} not in kohonen.networks (skipping)")
            continue
        for item in data_list:
            filename = item['filename']
            features = item['features']
            
            # Zamiast torch.tensor
            x = np.array(features, dtype=np.float32)
            
            winner = kohonen.predict(reg_type, x)
            if winner is None:
                continue
            
            folder = os.path.join(OUTPUT_DIR, f"{reg_type}_N{winner}")
            os.makedirs(folder, exist_ok=True)
            
            plot_args.append((filename, folder, features, winner, reg_type, FOLDER, N_POINTS_LAST, FEATURE_NAMES))
            csv_rows.append((filename, reg_type, winner))
    if csv_rows:
        csv_path = os.path.join(OUTPUT_DIR, "stocks.csv")
        with open(csv_path, "w") as f:
            f.write("filename,reg_type,neuron\n")
            for filename, reg_type, winner in csv_rows:
                f.write(f"{filename},{reg_type},{winner}\n")
        print(f"Saved CSV: {csv_path} ({len(csv_rows)} rows)")
    
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        executor.map(_plot_worker_kohonen, plot_args)
