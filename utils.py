from best_regr import *
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
FOLDER = "./stocks.cleaned/"

N_POINTS_LAST = 3000
FEATURE_NAMES = ['trend', 'mape', 'rmse', 'q_slope', 'highest_p', 'curvature']
N_FEATURES = len(FEATURE_NAMES)


def _process_file(f, folder, n_points_last):
    import pandas as pd
    import numpy as np
    import os
    try:
        y = pd.read_csv(os.path.join(folder, f)).iloc[:, 0].values.astype(float)
        y = y[~np.isnan(y) & ~np.isinf(y)]
        y = y[-n_points_last:]
        if len(y) < 100:
            return None
        
        reg_type, features, derivative, score = best_regression(np.arange(len(y)), y)
        if reg_type and features and len(features) == N_FEATURES:
            return {'filename': f, 'reg_type': reg_type, 'features': features}  # derivative, score - UNUSED
    except Exception as e:
        print(f"Error loading {f}: {e}")
        return None
    return None



def load_stock_data(n_points_last=N_POINTS_LAST):
    files = [f for f in os.listdir(FOLDER) if f.endswith('.csv')]
    data = []
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_process_file, f, FOLDER, n_points_last) for f in files]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                data.append(res)
    
    return data


