import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_stock_data, N_FEATURES, FEATURE_NAMES

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except ValueError:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass

def prepare_data():
    raw_data = load_stock_data(folder="../stocks.cleaned/")
    if not raw_data:
        raise FileNotFoundError("Brak danych.")
    X = np.array([s['features'] for s in raw_data])
    names = [s['filename'] for s in raw_data]
    reg_types = [s['reg_type'] for s in raw_data]
    print(f"[*] Wczytano {X.shape[0]} spółek, {X.shape[1]} cech.")
    return X, names, reg_types

def load_som_labels(data_dir="../results"):
    """Load SOM cluster labels from best_som_data.json if available."""
    path = os.path.join(os.path.dirname(__file__), data_dir, "best_som_data.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        som = json.load(f)
    labels = {}
    for s in som['stocks']:
        labels[s['filename']] = f"{s['predicted_reg_type']}_L1_{s['winner_l1']}"
    return labels, som['neurons']

def cosine_similarity_to_centroids(X, labels):
    """Mean cosine similarity of each point to its cluster centroid. Matches SOM metric."""
    unique = set(labels) - {-1}
    if len(unique) < 2:
        return 0.0
    sims = []
    for c in unique:
        mask = labels == c
        centroid = X[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-10:
            continue
        centroid = centroid / norm
        for x in X[mask]:
            xn = x / (np.linalg.norm(x) + 1e-10)
            sims.append(float(xn @ centroid))
    return np.mean(sims) if sims else 0.0

def run_regression(X, y, problem_name):
    """Regression problem with 4 methods, each with 4+ parameter values."""
    print(f"\n{'='*70}")
    print(f"REGRESJA – {problem_name}")
    print(f"{'='*70}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
    all_results = []

    methods_config = [
        ("KNN Regressor", 'n_neighbors', [1, 3, 5, 10, 15],
         lambda **kw: KNeighborsRegressor(**kw)),
        ("DT Regressor", 'max_depth', [3, 5, 10, 15, 20],
         lambda **kw: DecisionTreeRegressor(**kw, random_state=42)),
        ("RF Regressor", 'n_estimators', [10, 50, 100, 200, 500],
         lambda **kw: RandomForestRegressor(**kw, max_depth=10, random_state=42)),
        ("SVR", 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'],
         lambda **kw: SVR(**kw, max_iter=10000)),
    ]

    for label, pname, pvalues, builder in methods_config:
        print(f"\n>>> {label} ({pname})")
        for val in pvalues:
            try:
                model = builder(**{pname: val})
                if label == "SVR" and X_train.shape[0] > 500:
                    model.fit(X_train[:500], y_train[:500])
                else:
                    model.fit(X_train, y_train)
                pred = model.predict(X_test)
                pred_orig = scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
                y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
                mae = mean_absolute_error(y_test_orig, pred_orig)
                rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
                r2 = r2_score(y_test_orig, pred_orig)
                all_results.append({'Model': f'{label.split(" ")[0]} {val}', f'{pname}': str(val),
                                    'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)})
                print(f"  {pname}={str(val):12s} -> R2={r2:.4f}, MAE={mae:.2f}")
            except Exception as e:
                print(f"  {pname}={str(val):12s} -> BLAD: {e}")

    df = pd.DataFrame(all_results)
    cols = ['Model', 'MAE', 'RMSE', 'R2']
    print(f"\n--- Tabela zbiorcza: {problem_name} ---")
    print(df[cols].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x='Model', y='R2', data=df, ax=axes[0], hue='Model', palette='viridis', legend=False)
    axes[0].set_title(f'R² – {problem_name}')
    axes[0].tick_params(axis='x', rotation=30)
    sns.barplot(x='Model', y='MAE', data=df, ax=axes[1], hue='Model', palette='rocket', legend=False)
    axes[1].set_title(f'MAE – {problem_name}')
    axes[1].tick_params(axis='x', rotation=30)
    plt.suptitle(f'REGRESJA – {problem_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150)
    plt.close()

def print_method_descriptions():
    print("="*70)
    print("OPIS WYBRANYCH METOD UCZENIA MASZYNOWEGO (nienadzorowane)")
    print("="*70)
    print("""
1. K-Means
   - Grupuje dane w k klastrów na podstawie odległości euklidesowej
   - Parametry: liczba klastrów (k), inicjalizacja

2. DBSCAN
   - Gęstościowa metoda klasterowania, wykrywa outlierów
   - Parametry: promień sąsiedztwa (eps), minimalna liczba punktów (min_samples)

3. Klasterowanie Aglomeracyjne
   - Hierarchiczne łączenie punktów w klastry
   - Parametry: liczba klastrów, rodzaj połączenia (linkage)

4. GMM (Gaussian Mixture Model)
   - Probabilistyczne klasterowanie – każdy klaster to rozkład Gaussa
   - Parametry: liczba komponentów, typ kowariancji

Dodatkowo – zadanie regresji:
5. KNN Regressor – przewidywanie wartości ciągłej na podstawie k sąsiadów
6. Decision Tree Regressor – drzewo regresyjne
7. Random Forest Regressor – las losowy regresyjny
8. SVR – maszyna wektorów nośnych do regresji
""")

if __name__ == '__main__':
    log_file = open('wyniki_um.txt', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)

    print_method_descriptions()
    X_raw, stock_names, reg_types = prepare_data()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # === WCZYTANIE etykiet SOM (jeśli dostępne) ===
    som_data = load_som_labels()
    som_labels = None
    if som_data:
        som_map, _ = som_data
        som_labels = np.array([som_map.get(n, -1) for n in stock_names])
        unique_som = set(som_labels) - {-1}
        print(f"  Załadowano SOM: {len(unique_som)} unikalnych etykiet")
    else:
        print("  Brak danych SOM – pominięto porównanie")

    # === METODY KLASTEROWANIA ===
    all_results = []
    all_clusterings = {}  # name -> labels

    # === 1. KMeans ===
    print("\n" + "="*70)
    print("KLASTEROWANIE NIENADZOROWANE")
    print("="*70)

    print("\n>>> K-Means")
    k_values = [2, 3, 4, 5, 6]
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        cos = cosine_similarity_to_centroids(X, labels)
        all_clusterings[f'KMeans k={k}'] = labels
        r = {'Model': f'KMeans k={k}', 'Param': 'n_clusters', 'Value': str(k),
             'Silhouette': round(sil, 4), 'DBI': round(dbi, 4), 'Cosine': round(cos, 4)}
        if som_labels is not None:
            r['ARI_vs_SOM'] = round(adjusted_rand_score(som_labels, labels), 4)
        all_results.append(r)
        print(f"  k={k} -> Silhouette={sil:.4f}, Cosine={cos:.4f}, DBI={dbi:.4f}")

    # 2. DBSCAN
    print("\n>>> DBSCAN")
    eps_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters >= 2:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            cos = cosine_similarity_to_centroids(X, labels)
        else:
            sil, dbi, cos = -1, -1, 0
        all_clusterings[f'DBSCAN eps={eps}'] = labels
        r = {'Model': f'DBSCAN eps={eps}', 'Param': 'eps', 'Value': str(eps),
             'Silhouette': round(sil, 4), 'DBI': round(dbi, 4), 'Cosine': round(cos, 4),
             'Klastry': n_clusters, 'Noise': n_noise}
        if som_labels is not None and n_clusters >= 2:
            r['ARI_vs_SOM'] = round(adjusted_rand_score(som_labels, labels), 4)
        all_results.append(r)
        noise_str = f", noise={n_noise}" if n_noise > 0 else ""
        print(f"  eps={eps} -> Silhouette={sil:.4f}, Cosine={cos:.4f}, klastry={n_clusters}{noise_str}")

    # 3. Agglomerative
    print("\n>>> Klasterowanie Aglomeracyjne")
    for k in k_values:
        for linkage in ['ward', 'complete', 'average', 'single']:
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = agg.fit_predict(X)
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
                cos = cosine_similarity_to_centroids(X, labels)
                all_clusterings[f'Agglom {linkage} k={k}'] = labels
                r = {'Model': f'Agglom {linkage} k={k}', 'Param': 'linkage', 'Value': f'{linkage},k={k}',
                     'Silhouette': round(sil, 4), 'DBI': round(dbi, 4), 'Cosine': round(cos, 4)}
                if som_labels is not None:
                    r['ARI_vs_SOM'] = round(adjusted_rand_score(som_labels, labels), 4)
                all_results.append(r)
                print(f"  linkage={linkage:8s} k={k} -> Silhouette={sil:.4f}, Cosine={cos:.4f}, DBI={dbi:.4f}")
            except Exception as e:
                print(f"  linkage={linkage:8s} k={k} -> BLAD: {e}")

    # 4. GMM
    print("\n>>> GMM (Gaussian Mixture Model)")
    for k in k_values:
        for cov in ['full', 'diag', 'tied', 'spherical']:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42)
                labels = gmm.fit_predict(X)
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
                cos = cosine_similarity_to_centroids(X, labels)
                all_clusterings[f'GMM {cov} k={k}'] = labels
                r = {'Model': f'GMM {cov} k={k}', 'Param': 'covariance_type', 'Value': f'{cov},k={k}',
                     'Silhouette': round(sil, 4), 'DBI': round(dbi, 4), 'Cosine': round(cos, 4)}
                if som_labels is not None:
                    r['ARI_vs_SOM'] = round(adjusted_rand_score(som_labels, labels), 4)
                all_results.append(r)
                print(f"  cov={cov:8s} k={k} -> Silhouette={sil:.4f}, Cosine={cos:.4f}, DBI={dbi:.4f}")
            except Exception as e:
                print(f"  cov={cov:8s} k={k} -> BLAD: {e}")

    # === TABELA PORÓWNAWCZA ===
    df = pd.DataFrame(all_results)
    cols = ['Model', 'Silhouette', 'Cosine', 'DBI']
    if 'ARI_vs_SOM' in df.columns:
        cols.append('ARI_vs_SOM')

    # Średni cosine SOM dla porównania
    som_cos = None
    som_path = os.path.join(os.path.dirname(__file__), "../results/best_som_data.json")
    if os.path.exists(som_path):
        with open(som_path) as f:
            som_raw = json.load(f)
        som_cos = np.mean([s['winner_cosine'] for s in som_raw['stocks'] if s['winner_cosine'] is not None])

    print(f"\n--- Tabela zbiorcza klasterowania ---")
    if som_cos is not None:
        print(f"SOM (Kohonen)                Cosine={som_cos:.4f}")
    print(df[cols].to_string(index=False))

    if 'ARI_vs_SOM' in df.columns:
        best_ari = df.loc[df['ARI_vs_SOM'].idxmax()]
        print(f"\nNajwyższa zgodność z SOM: {best_ari['Model']} (ARI={best_ari['ARI_vs_SOM']})")

    # === NAJLEPSZE KLASTEROWANIE: WIZUALIZACJA PCA ===
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    valid = df[df['Silhouette'] > 0]
    if len(valid):
        best = valid.loc[valid['Silhouette'].idxmax()]
        best_name = best['Model']
        best_labels = all_clusterings.get(best_name)
        if best_labels is not None:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='tab10', alpha=0.6, s=15)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.title(f'PCA + {best_name} (Silhouette={best["Silhouette"]})')
            plt.savefig('pca_best_clustering.png', dpi=150)
            plt.close()

    # === PORÓWNANIE MIĘDZY METODAMI (ARI) ===
    print("\n" + "="*70)
    print("PORÓWNANIE ZGODNOŚCI MIĘDZY METODAMI (ARI)")
    print("="*70)
    methods = list(all_clusterings.keys())
    # Weź po jednej reprezentatywnej konfiguracji na metodę
    repr_configs = {}
    for r in all_results:
        model_base = r['Model'].rsplit(' ', 1)[0] if 'k=' in r['Model'] or 'eps=' in r['Model'] else r['Model']
        # Pick first occurrence of each base method
        key = r['Model'].split(' ')[0]
        if key not in repr_configs and r['Silhouette'] > 0:
            repr_configs[key] = r['Model']
    repr_names = list(repr_configs.values())

    if len(repr_names) >= 2:
        fig, axes = plt.subplots(len(repr_names), len(repr_names), figsize=(12, 10))
        for i, ni in enumerate(repr_names):
            for j, nj in enumerate(repr_names):
                ax = axes[i, j] if len(repr_names) > 1 else axes
                if i == j:
                    ax.text(0.5, 0.5, ni.split(' ')[0], ha='center', va='center', fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                elif i > j:
                    li = all_clusterings[ni]
                    lj = all_clusterings[nj]
                    ari = adjusted_rand_score(li, lj)
                    nmi = normalized_mutual_info_score(li, lj)
                    ax.text(0.5, 0.5, f'ARI={ari:.3f}\nNMI={nmi:.3f}', ha='center', va='center', fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    ax.axis('off')
        plt.suptitle('Macierz zgodności między metodami (ARI/NMI)', fontsize=12)
        plt.tight_layout()
        plt.savefig('ari_matrix.png', dpi=150)
        plt.close()

    # === PODSUMOWANIE ===
    print(f"\n{'='*50}")
    print("PODSUMOWANIE")
    print(f"{'='*50}")
    print(f"Liczba spółek: {X.shape[0]}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Najlepsza metoda (Silhouette): {best['Model'] if len(valid) else 'brak'} ({best['Silhouette']}, Cosine={best['Cosine']})")
    if 'ARI_vs_SOM' in df.columns:
        print(f"Najwyższa zgodność z SOM: {best_ari['Model']} (ARI={best_ari['ARI_vs_SOM']})")

    # === ZADANIE REGRESJI ===
    # Przewidywanie odległości od centroidu klastra (dla najlepszego klasterowania)
    if len(valid):
        best_reg_labels = all_clusterings[best_name]
        # Dla każdej spółki: odległość euklidesowa od centroidu jej klastra
        reg_target = np.zeros(X.shape[0])
        for c in set(best_reg_labels):
            mask = best_reg_labels == c
            centroid = X[mask].mean(axis=0)
            reg_target[mask] = np.linalg.norm(X[mask] - centroid, axis=1)
        run_regression(X, reg_target, "odległość spółki od centroidu klastra")

    log_file.close()
