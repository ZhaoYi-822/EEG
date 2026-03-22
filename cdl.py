

import numpy as np


class CDLFeatureReduction:
    def __init__(self, threshold=0.8):
        self.threshold   = threshold
        self.keep_idx_   = None
    def fit(self, X):
        corr    = np.abs(np.corrcoef(X.T))
        n       = X.shape[1]
        to_drop = set()
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if corr[i, j] >= self.threshold:
                    to_drop.add(j)
        self.keep_idx_ = [i for i in range(n) if i not in to_drop]
        print(f"[CDL-Features] {n} -> {len(self.keep_idx_)} features "
              f"({100*len(self.keep_idx_)/n:.1f}% retained)")
        return self
    def transform(self, X):
        return X[:, self.keep_idx_]
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CDLSampleReduction:

    def __init__(self, delta_q=1.96, random_state=42):
        self.delta_q      = delta_q
        self.random_state = random_state

    def _n_star(self, X_cls):
        n0      = len(X_cls)
        sigma   = X_cls.std(axis=0) + 1e-8
        diff_sq = (X_cls.mean(axis=0) - X_cls.mean(axis=0))**2 + 1e-8
        n_rk    = np.minimum(n0, (self.delta_q * sigma / diff_sq).astype(int))
        return max(1, min(int(np.floor(n_rk.mean())), n0))

    def fit_transform(self, X, y):
        rng    = np.random.RandomState(self.random_state)
        X_out, y_out = [], []
        for cls in np.unique(y):
            idx    = np.where(y == cls)[0]
            n_star = self._n_star(X[idx])
            chosen = rng.choice(len(idx), size=n_star, replace=False)
            X_out.append(X[idx[chosen]])
            y_out.append(np.full(n_star, cls))
        X_out = np.vstack(X_out)
        y_out = np.concatenate(y_out)
        print(f"[CDL-Sample]   {len(X)} -> {len(X_out)} samples "
              f"({100*len(X_out)/len(X):.1f}% retained)")
        return X_out, y_out

