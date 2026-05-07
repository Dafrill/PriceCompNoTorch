from best_regr import *
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import sys

FOLDER = "./stocks.cleaned/"
N_POINTS_LAST = 3000
FEATURE_NAMES = ['trend', 'mape', 'q_slope', 'curvature',
                 'highest_p1', 'highest_p2', 'highest_p3', 'highest_p4', 'highest_p5',
                 'rmse_p1', 'rmse_p2', 'rmse_p3', 'rmse_p4', 'rmse_p5']
N_FEATURES = len(FEATURE_NAMES)
OUTPUT_DIR = "./results/"
CACHE_FILE = "./results/stock_data_cache.pkl"

def _process_file(f, folder, n_points_last):
    try:
        y = pd.read_csv(os.path.join(folder, f)).iloc[:, 0].values.astype(float)
        y = y[~np.isnan(y) & ~np.isinf(y)]
        y = y[-n_points_last:]
        if len(y) < 100:
            return None

        reg_type, features, derivative, score = best_regression(np.arange(len(y)), y)
        if reg_type and features and len(features) == N_FEATURES:
            return {'filename': f, 'reg_type': reg_type, 'features': features}
    except Exception as e:
        print(f"Error loading {f}: {e}")
        return None
    return None

def load_stock_data(n_points_last=N_POINTS_LAST):
    # Check cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} samples from cache")
            return data
        except Exception:
            pass

    # Process files
    files = [f for f in os.listdir(FOLDER) if f.endswith('.csv')]
    data = []

    print(f"Processing {len(files)} files...", file=sys.stderr)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_file, f, FOLDER, n_points_last) for f in files]
        for i, future in enumerate(as_completed(futures)):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(files)}...", file=sys.stderr)
            res = future.result()
            if res is not None:
                data.append(res)

    # Save cache
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples to cache")

    return data

def generate_regression_figure(filename, reg_type, features, folder_data=None, n_points=None, feat_names=None):
    """Generate time series plot with regression line, returns figure."""
    if folder_data is None:
        folder_data = FOLDER
    if n_points is None:
        n_points = N_POINTS_LAST
    if feat_names is None:
        feat_names = FEATURE_NAMES

    try:
        y = pd.read_csv(os.path.join(folder_data, filename)).iloc[:, 0].values.astype(float)
        y = y[~np.isnan(y) & ~np.isinf(y)]
        y = y[-n_points:]
        n = len(y)
        X = np.arange(n)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y, linewidth=1, color='blue', label='Data')

        # Use predict_regression from best_regr to avoid code duplication
        y_pred = predict_regression(X, y, reg_type)

        ax.plot(y_pred, linewidth=2, color='red', alpha=0.7, label=f'{reg_type} fit')

        title_parts = []
        for i, name in enumerate(feat_names):
            val = features[i] if i < len(features) else 0
            if name == 'curvature':
                title_parts.append(f"{name}:{val:.8f}")
            elif name in ('rmse', 'q_slope', 'trend', 'mape') or name.startswith('rmse_'):
                title_parts.append(f"{name}:{val:.4f}")
            else:
                title_parts.append(f"{name}:{val:.2f}")

        ax.set_title(f"{filename}\nType:{reg_type}\n" + " | ".join(title_parts), fontsize=9)
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR generating plot for {filename}: {e}")
        return None
