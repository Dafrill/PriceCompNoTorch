import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

from utils import load_stock_data, FEATURE_NAMES, N_FEATURES, FOLDER as FOLDER_DATA, N_POINTS_LAST, generate_regression_figure
from test import ClassificationKohonen, ClassificationKohonen1Layer

RESULTS_DIR = "./results"
BEST_CONFIG_PATH = os.path.join(RESULTS_DIR, "best_config.json")
PDF_PATH = "best_config_summary.pdf"


def main():
    print("Generating PDF for best configuration...")

    # 1. Wczytaj najlepszą konfigurację
    if not os.path.exists(BEST_CONFIG_PATH):
        print(f"ERROR: {BEST_CONFIG_PATH} not found. Run test.py first.")
        return

    with open(BEST_CONFIG_PATH, 'r') as f:
        best_config = json.load(f)

    print(f"Loaded best config: {best_config['config']}")

    # 2. Wczytaj dane
    print("Loading stock data...")
    data = load_stock_data()
    if not data:
        print("ERROR: No data loaded!")
        return

    print(f"Loaded {len(data)} stocks")

    # 3. Trenuj sieć z najlepszymi parametrami (deterministycznie)
    np.random.seed(42)
    data_copy = [dict(item) for item in data]
    np.random.shuffle(data_copy)

    train_ratio = best_config.get('train_ratio', 0.8)
    n_train = int(len(data_copy) * train_ratio)
    train_data = data_copy[:n_train]
    test_data = data_copy[n_train:]

    n_neurons_l2 = best_config['n_neurons_l2']
    n_neurons_l1 = best_config['n_neurons_l1']
    print(f"Training network: L1={best_config['n_neurons_l1']}, L2={n_neurons_l2}, activation={best_config['activation']}...")

    if n_neurons_l2 == 0:
        # Sieć 1-warstwowa
        net = ClassificationKohonen1Layer(
            n_neurons_l1=best_config['n_neurons_l1'],
            input_dim=N_FEATURES,
            activation=best_config['activation'],
            eta_init=best_config.get('eta_init', 0.9),
            eta_decay=best_config.get('eta_decay', 0.95),
        )
    else:
        # Sieć 2-warstwowa
        net = ClassificationKohonen(
            n_neurons_l1=best_config['n_neurons_l1'],
            n_neurons_l2=n_neurons_l2,
            input_dim=N_FEATURES,
            activation=best_config['activation'],
            eta_init=best_config.get('eta_init', 0.9),
            eta_decay=best_config.get('eta_decay', 0.95),
        )

    net.train(train_data, epochs=best_config['epochs'])

    # 4. Generuj wykresy
    print(f"Generating plots for {len(test_data)} test samples...")

    # Grupuj: reg_type -> winner1 -> winner2 -> items
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for item in test_data:
        x = np.array(item['features'], dtype=np.float32)
        true_reg_type = item['reg_type']

        if n_neurons_l2 == 0:
            winner, pred_reg_type = net.predict(x)
            winner1 = winner
            winner2 = None
        else:
            (winner1, winner2), pred_reg_type = net.predict(x)

        groups[true_reg_type][winner1][winner2].append(item)

    with PdfPages(PDF_PATH) as pdf:
        colors = ['#FFE4E8', '#D6E8F5']  # jasnoróżowy, jasnoniebieski
        w1_counter = 0

        for rt in sorted(groups.keys()):
            l1_groups = groups[rt]
            total_samples = sum(
                len(items)
                for w2_dict in l1_groups.values()
                for items in w2_dict.values()
            )
            print(f"  Network: {rt} ({total_samples} samples)")

            # Strona tytułowa sieci
            fig_title, ax_title = plt.subplots(figsize=(10, 6))
            ax_title.text(0.5, 0.5, f'Network: {rt}\n({total_samples} samples)',
                         ha='center', va='center', fontsize=20, fontweight='bold')
            ax_title.axis('off')
            pdf.savefig(fig_title, dpi=100)
            plt.close(fig_title)

            for w1 in sorted(l1_groups.keys()):
                w2_groups = l1_groups[w1]
                w1_samples = sum(len(items) for items in w2_groups.values())

                # Kontener neuronu L1
                fig_l1, ax_l1 = plt.subplots(figsize=(10, 2))
                ax_l1.set_facecolor(colors[w1_counter % len(colors)])
                ax_l1.text(0.5, 0.5, f'Neuron L1 #{w1} ({w1_samples} samples)',
                          ha='center', va='center', fontsize=16, fontweight='bold')
                ax_l1.axis('off')
                pdf.savefig(fig_l1, dpi=100)
                plt.close(fig_l1)
                w1_counter += 1

                # Collect all items for this L1 neuron
                all_items = []
                for w2, items in w2_groups.items():
                    all_items.extend(items)

                # Show max 6 plots per neuron, in 2 rows × 3 columns
                n_show = min(6, len(all_items))
                items_to_show = all_items[:n_show]

                if n_show > 0:
                    # Create grid: 2 rows, 3 columns
                    fig_grid, axes = plt.subplots(2, 3, figsize=(10.5, 5))
                    axes = axes.flatten()

                    for idx, item in enumerate(items_to_show):
                        ax = axes[idx]
                        # Use utils.generate_regression_figure to create the plot
                        fig_small = generate_regression_figure(
                            item['filename'], 
                            item['reg_type'], 
                            item['features'],
                            FOLDER_DATA, 
                            N_POINTS_LAST
                        )

                        # Copy content from small figure to grid subplot
                        if fig_small is not None:
                            small_ax = fig_small.gca()
                            
                            # Copy lines
                            for line in small_ax.get_lines():
                                ax.plot(line.get_xdata(), line.get_ydata(), 
                                       color=line.get_color(), 
                                       linewidth=1,
                                       alpha=line.get_alpha(),
                                       label=line.get_label())

                            # Simplify: only company name as title
                            company_name = os.path.splitext(os.path.basename(item['filename']))[0]
                            ax.set_title(company_name, fontsize=8, fontweight='bold', pad=2)
                            
                            # Copy legend if exists
                            if small_ax.get_legend() is not None:
                                ax.legend(fontsize=6, loc='upper left')
                            
                            ax.tick_params(labelsize=6)
                            ax.grid(True, alpha=0.3)

                            plt.close(fig_small)

                    # Hide unused subplots
                    for idx in range(n_show, 6):
                        axes[idx].axis('off')

                    plt.tight_layout()
                    pdf.savefig(fig_grid, dpi=100)
                    plt.close(fig_grid)

    print(f"\nSaved: {PDF_PATH}")


if __name__ == "__main__":
    main()
