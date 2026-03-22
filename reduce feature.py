


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from cdl import CDLFeatureReduction, CDLSampleReduction




def select_anova(X, y, p_thresh=0.05):
    p_vals = np.array([f_oneway(*[X[y==c, j] for c in np.unique(y)])[1]
                       for j in range(X.shape[1])])
    idx = np.where(p_vals < p_thresh)[0]
    if len(idx) == 0:
        idx = np.array([np.argmin(p_vals)])
    return X[:, idx], idx


def select_heatmap(X, threshold=0.80):
    corr    = np.abs(np.corrcoef(X.T))
    n       = X.shape[1]
    to_drop = set()
    for i in range(n):
        if i in to_drop: continue
        for j in range(i+1, n):
            if corr[i, j] >= threshold:
                to_drop.add(j)
    idx = np.array([i for i in range(n) if i not in to_drop])
    return X[:, idx], idx


def select_mutual_info(X, y, quantile=0.5):
    mi  = mutual_info_classif(X, y, random_state=42)
    thr = np.quantile(mi, quantile)
    idx = np.where(mi >= thr)[0]
    if len(idx) == 0:
        idx = np.array([np.argmax(mi)])
    return X[:, idx], idx


# -- Comparison table ---------------------------------------------------------

def compare_methods(X, y):
    n = X.shape[1]
    rows = [{"Method": "Original", "Features": n, "Retained (%)": 100.0, "Threshold": "-"}]

    Xa, _ = select_anova(X, y, p_thresh=0.05)
    rows.append({"Method": "ANOVA", "Features": Xa.shape[1],
                 "Retained (%)": round(100*Xa.shape[1]/n, 1), "Threshold": "p < 0.05"})

    Xh, _ = select_heatmap(X, threshold=0.80)
    rows.append({"Method": "Heatmap", "Features": Xh.shape[1],
                 "Retained (%)": round(100*Xh.shape[1]/n, 1), "Threshold": "|r| >= 0.80"})

    Xm, _ = select_mutual_info(X, y, quantile=0.5)
    rows.append({"Method": "Mutual Info", "Features": Xm.shape[1],
                 "Retained (%)": round(100*Xm.shape[1]/n, 1), "Threshold": "MI >= median"})

    Xc = CDLFeatureReduction(0.8).fit_transform(X)
    rows.append({"Method": "CDL-Features", "Features": Xc.shape[1],
                 "Retained (%)": round(100*Xc.shape[1]/n, 1), "Threshold": "r* = 0.80"})

    df = pd.DataFrame(rows)
    print("\n--- Feature Selection Comparison ---")
    print(df.to_string(index=False))
    return df



def run_cdl_pipeline(X_train, y_train, X_test, save_dir=None):

    n_orig, m_orig = X_train.shape
    print(f"\nOriginal : {n_orig} samples x {m_orig} features")
    cdl_f  = CDLFeatureReduction(threshold=0.8)
    Xtr_f  = cdl_f.fit_transform(X_train)
    Xte_f  = X_test[:, cdl_f.keep_idx_]

    cdl_s  = CDLSampleReduction(delta_q=1.96)
    Xtr_c, ytr_c = cdl_s.fit_transform(Xtr_f, y_train)

    vol_pct = 100 * Xtr_c.size / (n_orig * m_orig)
    print(f"Combined : {Xtr_c.shape[0]} samples x {Xtr_c.shape[1]} features  "
          f"({vol_pct:.1f}% of original volume)")

    if save_dir:
        import os; os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/X_train_cdl.npy", Xtr_c)
        np.save(f"{save_dir}/y_train_cdl.npy", ytr_c)
        np.save(f"{save_dir}/X_test_cdl.npy",  Xte_f)
        print(f"Saved reduced dataset to {save_dir}/")

    return Xtr_c, ytr_c, Xte_f, cdl_f, cdl_s




def plot_figure1(df):
    colors = ['#8B1A1A', '#6BAED6', '#74C476', '#FDD0A2', '#9ECAE1']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Method"], df["Retained (%)"], color=colors, edgecolor='white', width=0.55)
    for bar, val in zip(bars, df["Retained (%)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Features Retained (%)")
    ax.set_title("Figure 1: Feature Reduction Method Comparison")
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()


def plot_figure2(n_orig, m_orig, cdl_f, cdl_s, Xtr_c):
    vol_orig     = n_orig * m_orig
    vol_feat     = n_orig * cdl_f.n_selected
    vol_samp     = cdl_s.n_reduced * m_orig if hasattr(cdl_s, 'n_reduced') else vol_orig
    vol_combined = Xtr_c.size

    methods = ['Original', 'CDL-Features', 'CDL-Samples', 'CDL-Combined']
    pcts    = [100.0,
               round(100*vol_feat/vol_orig, 1),
               round(100*vol_samp/vol_orig, 1),
               round(100*vol_combined/vol_orig, 1)]
    colors  = ['#8B1A1A', '#6BAED6', '#74C476', '#FDD0A2']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, pcts, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Data Volume (%)")
    ax.set_title("Figure 2: CDL Data Volume Comparison")
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()


# -