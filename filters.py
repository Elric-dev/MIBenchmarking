from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import mne

class FBFilter(BaseEstimator, TransformerMixin):
    """
    Stateless filter-bank transformer:
      input:  X shape (n_epochs, n_channels, n_times)
      output: list of arrays [Xf_band1, Xf_band2, ...], each like X
    Notes:
      - __init__ stores params *exactly* as given (no casting/copying).
      - Cast/validation happens in transform().
    """
    def __init__(self, filterbank, sfreq=250.0, fb_n_jobs=1):
        self.filterbank = filterbank     # DO NOT modify here
        self.sfreq = sfreq               # keep as passed
        self.fb_n_jobs = fb_n_jobs       # keep as passed

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        # Validate/cast *here*, not in __init__
        if self.filterbank is None:
            raise ValueError("FBFilter.filterbank must be set (list of (fmin,fmax)).")
        fb = list(self.filterbank)  # shallow copy ok at transform-time
        sfreq = float(self.sfreq)
        n_jobs = int(self.fb_n_jobs)

        X = np.asarray(X, dtype=np.float64, order="C")
        out = []
        for (f1, f2) in fb:
            Xf = mne.filter.filter_data(
                X, sfreq, float(f1), float(f2),
                n_jobs=n_jobs, copy=True, verbose=False
            )
            out.append(Xf)
        return out