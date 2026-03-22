import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def preprocess(eeg, fs, mu, sigma):

    eeg = eeg[10:]
    nyq  = fs / 2.0
    b, a = butter(2, [0.5 / nyq, 45.0 / nyq], btype='band')
    eeg  = filtfilt(b, a, eeg, axis=0)
    eeg = eeg[np.all(np.abs(eeg) <= 100, axis=1)]
    win = 50
    if len(eeg) >= win:
        var   = np.array([eeg[i:i+win].var(axis=0).mean() for i in range(len(eeg)-win)])
        valid = np.where(np.abs(var - var.mean()) <= 3 * var.std())[0]
        if len(valid):
            eeg = eeg[valid[0]: valid[-1] + win]
    if mu    is None: mu    = eeg.mean(axis=0)
    if sigma is None: sigma = eeg.std(axis=0) + 1e-8
    eeg = (eeg - mu) / sigma

    return eeg, mu, sigma


if __name__ == "__main__":
    raw=pd.read_csv("Test0_2019.08.18_14.15.17.csv")
    clean, mu, sigma = preprocess(raw)
    print(f"Samples:{clean.shape[0]}, mean={clean.mean()}, std={clean.std()}")