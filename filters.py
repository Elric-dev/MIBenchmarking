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
    



class CSPBank(BaseEstimator, TransformerMixin):
    """
    One CSP per band on the list produced by FBFilter; concatenates features.
    """
    def __init__(self, n_components=4, log=True, reg="oas"):
        self.n_components = n_components  # keep as passed
        self.log = log
        self.reg = reg

    def fit(self, X_bands, y):
        from mne.decoding import CSP
        n_comp = int(self.n_components)   # cast at fit-time
        log = bool(self.log)
        reg = self.reg

        self.csps_ = []
        for Xf in X_bands:
            csp = CSP(n_components=n_comp, log=log, reg=reg)
            csp.fit(Xf, y)
            self.csps_.append(csp)
        return self

    def transform(self, X_bands):
        import numpy as np
        feats = [csp.transform(Xf) for csp, Xf in zip(self.csps_, X_bands)]
        return np.hstack(feats)
