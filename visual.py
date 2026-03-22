
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_figure1(methods, pcts):
    colors = ['#8B1A1A','#6BAED6','#74C476','#FDD0A2','#9ECAE1']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, pcts, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Features Retained (%)")
    ax.set_title("Figure 1: Feature Reduction Comparison")
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()


def plot_figure2(methods, pcts):
    colors = ['#8B1A1A','#6BAED6','#74C476','#FDD0A2']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, pcts, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Data Volume (%)")
    ax.set_title("Figure 2: CDL Data Volume Comparison")
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.show()


def plot_confusion(y_true, y_pred, n_known=20):
    cm  = confusion_matrix(y_true, y_pred)
    n   = cm.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, ax=axes[0], cmap='Blues', annot=(n<=25), fmt='d', linewidths=0.2)
    axes[0].set_title("Confusion Matrix (Per Subject)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    ki  = list(range(n_known))
    ui  = list(range(n_known, n))
    tp  = cm[np.ix_(ki, ki)].sum(); fn = cm[np.ix_(ki, ui)].sum()
    fp  = cm[np.ix_(ui, ki)].sum(); tn = cm[np.ix_(ui, ui)].sum()
    tot = tp+fn+fp+tn+1e-8
    ann = np.array([[f'{tp}\n({100*tp/tot:.1f}%)', f'{fn}\n({100*fn/tot:.1f}%)'],
                    [f'{fp}\n({100*fp/tot:.1f}%)', f'{tn}\n({100*tn/tot:.1f}%)']])
    sns.heatmap(np.array([[tp,fn],[fp,tn]]), ax=axes[1], annot=ann, fmt='',
                cmap='YlOrRd', xticklabels=['Known','Unknown'],
                yticklabels=['Known','Unknown'], linewidths=1, cbar=False)
    axes[1].set_title("Binary Confusion Matrix")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
    plt.tight_layout(); plt.show()


def plot_roc(models_proba, y_test, n_classes):
    y_bin  = label_binarize(y_test, classes=list(range(n_classes)))
    colors = {'CNN':'red','LSTM':'green','XGBoost':'orange','SVM':'blue'}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in models_proba.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
        ax.plot(fpr, tpr, color=colors.get(name,'gray'), lw=2,
                label=f'{name} (Combined CDL) (AUC={auc:.2f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Figure 5: AUC Results with Combined CDL")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout(); plt.show()


def plot_det(models_proba, y_test, n_classes):
    y_bin  = label_binarize(y_test, classes=list(range(n_classes)))
    colors = {'CNN':'blue','LSTM':'orange','XGBoost':'green','SVM':'red'}
    recall_pts = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in models_proba.items():
        mean_prec = np.zeros(200)
        for c in range(n_classes):
            p, r, _ = precision_recall_curve(y_bin[:, c], proba[:, c])
            mean_prec += np.interp(recall_pts, r[::-1], p[::-1])
        ax.plot(recall_pts, mean_prec/n_classes, color=colors.get(name,'gray'),
                lw=2, label=f'{name} (Combined CDL)')
    ax.plot([0,1],[1,0],'k--',lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Figure 6: DET Curve with Combined CDL")
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout(); plt.show()


def far_frr_table(y_true, y_proba,
                  thresholds=[0.950,0.930,0.910,0.901,0.890,0.880,0.860,0.840]):
    max_prob   = y_proba.max(axis=1)
    pred_class = y_proba.argmax(axis=1)
    rows = []
    for t in thresholds:
        accepted = max_prob >= t
        correct  = pred_class == y_true
        tp = ( correct &  accepted).sum(); fn = ( correct & ~accepted).sum()
        fp = (~correct &  accepted).sum(); tn = (~correct & ~accepted).sum()
        rows.append({"Threshold": t,
                     "FAR (%)": round(fp/(fp+tn+1e-8)*100, 2),
                     "FRR (%)": round(fn/(fn+tp+1e-8)*100, 2)})
    return pd.DataFrame(rows)


