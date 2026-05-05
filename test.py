import numpy as np
from utils import load_stock_data
import pandas as pd
from winTakesAll import WinnerTakesAllKohonen, save_results as save_kohonen_results
OUTPUT_DIR = ".results/"
TRAIN_RATIO = 0.80

def main():
    print("Loading stock data...")
    data = load_stock_data()
    print(f"Loaded {len(data)} stocks")
    
    if not data:
        print("ERROR: No data loaded!")
        return
    
    # data = normalize_derivatives(data)  # UNUSED
    
    np.random.seed(42)
    np.random.shuffle(data)
    n_train = int(len(data) * TRAIN_RATIO)
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    data_by_type = {}
    for item in train_data:
        rt = item['reg_type']
        if rt not in data_by_type:
            data_by_type[rt] = []
        data_by_type[rt].append(item)
    
    print("\n=== Kohonen WTA ===")
    kohonen = WinnerTakesAllKohonen(n_neurons=40, input_dim=6)
    for reg_type in data_by_type:
        kohonen.create_network_for_type(reg_type)
        print(f"  {reg_type}: {len(data_by_type[reg_type])} stocks")
    
    kohonen.train(data_by_type, epochs=100)
    
    print("Saving results (test data only)...")
    test_by_type = {}
    for item in test_data:
        rt = item['reg_type']
        if rt not in test_by_type:
            test_by_type[rt] = []
        test_by_type[rt].append(item)
    
    for reg_type, data_list in test_by_type.items():
        save_kohonen_results(kohonen, {reg_type: data_list}, rt=reg_type)
    print(f"Results: {OUTPUT_DIR}")
    
    print("\n=== Done ===")
if __name__ == "__main__":
 main()
