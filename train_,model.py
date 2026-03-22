
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import t as t_dist


DEVICE = torch.device('cuda')


def train_sklearn(model, X_train, y_train):
    t0 = time.time()
    model.fit(X_train, y_train)
    return time.time() - t0


def eval_sklearn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return _metrics(y_test, y_pred)


def train_torch(model, X_train, y_train,
                epochs=50, lr=0.01, batch_size=64, seed=42):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.FloatTensor(X_train).unsqueeze(1)   # (N, 1, features)
    y_t = torch.LongTensor(y_train)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return time.time() - t0


def eval_torch(model, X_test, y_test):
    model.eval()
    X_t = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        y_pred = model(X_t).argmax(1).cpu().numpy()
    return _metrics(y_test, y_pred)


def predict_proba_torch(model, X_test):
    model.eval()
    X_t = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(X_t), dim=1).cpu().numpy()

def _metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }

def hypothesis_test(acc_orig, acc_reduced, alpha=0.05):
    a1, a2 = np.array(acc_orig), np.array(acc_reduced)
    m1, m2 = a1.mean(), a2.mean()
    s1, s2 = a1.std(ddof=1) + 1e-8, a2.std(ddof=1) + 1e-8
    n1, n2 = len(a1), len(a2)

    sp    = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    g     = (m2 - m1) / (sp + 1e-8)
    se    = np.sqrt(s1**2/n1 + s2**2/n2)
    df    = se**4 / (s1**4/n1**2/(n1-1) + s2**4/n2**2/(n2-1))
    p_val = float(2 * t_dist.sf(abs((m2-m1)/se), df))

    return {
        "delta_acc":   round(float(m2 - m1), 4),
        "CI_95":       (round(m2-m1 - 1.96*se, 2), round(m2-m1 + 1.96*se, 2)),
        "hedges_g":    round(float(g), 3),
        "p_value":     round(p_val, 4),
        "H0_accepted": bool(p_val >= alpha),
    }
