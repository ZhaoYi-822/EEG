

import numpy as np
from scipy.signal import welch

BANDS = {'delta': (0.5, 4), 'theta': (4, 8),
         'alpha': (8, 13),  'beta':  (13, 30), 'gamma': (30, 45)}


def channel_features(ch, fs=128):
    freqs, psd = welch(ch, fs=fs, nperseg=min(len(ch), fs))
    feats = []

    for lo, hi in BANDS.values():
        idx = (freqs >= lo) & (freqs <= hi)
        bp  = psd[idx]
        feats.append(bp.mean() if bp.size else 0.0)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        feats.append(bp.sum() * df if bp.size else 0.0)
        cs  = np.cumsum(ch - ch.mean())
        H   = np.log(max(cs.max()-cs.min(), 1e-8) / (ch.std()+1e-8)) / np.log(len(ch))
        feats.append(float(H))
        dy  = np.diff(ch)
        mob = np.sqrt(np.var(dy) / (np.var(ch) + 1e-8))
        feats.append(float(mob))
        ddy  = np.diff(dy)
        mob2 = np.sqrt(np.var(ddy) / (np.var(dy) + 1e-8))
        feats.append(float(mob2 / (mob + 1e-8)))
        nd  = np.sum(np.diff(np.sign(np.diff(ch))) != 0)
        N   = len(ch)
        pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4*nd + 1e-8)))
        feats.append(float(pfd))
        feats.append(float(ch.mean()))
        feats.append(float(np.median(ch)))
        feats.append(float(ch.std()))
        diffs = np.abs(np.diff(ch))
        L, a  = diffs.sum(), diffs.mean() + 1e-8
        d     = np.abs(ch - ch[0]).max() + 1e-8
        feats.append(float(np.log10(L/a) / (np.log10(d/a) + 1e-8)))
        trend = np.polyval(np.polyfit(np.arange(N), ch, 1), np.arange(N))
        feats.append(float(np.sqrt(np.mean((ch - trend)**2))))

    return np.array(feats, dtype=np.float32)


def extract_features(eeg, window_size=50, step=5, fs=128):
    rows = []
    for start in range(0, len(eeg) - window_size + 1, step):
        win  = eeg[start: start + window_size]
        feat = np.concatenate([channel_features(win[:, c], fs) for c in range(win.shape[1])])
        rows.append(feat)
    return np.array(rows, dtype=np.float32)

