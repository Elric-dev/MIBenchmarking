
"""
BNCI2014_001 training with shared-preprocessing cache:
- Split FB-CSP into FBFilter (stateless band-pass) + CSPBank (fits CSP), enabling joblib Pipeline cache reuse
- GridSearchCV inner-CV parallelized (n_jobs), with verbose progress and throttled dispatch
- Saves MOABB WithinSession outer-CV results per subject
- Saves inner-CV cv_results_ per subject/pipeline
- Writes per-trial predictions:
    * OUTER test predictions (always)
    * Optional INNER validation predictions for tuned hyper-params (best_params), per outer fold
  Supports incremental append so CSV grows per fold/pipeline

Portable: Linux & macOS (Apple Accelerate / vecLib respected). Headless-safe (Agg backend).
"""

import os, sys, platform, warnings, json, time, glob, gc, pickle, traceback, argparse
from datetime import datetime
warnings.filterwarnings("ignore")

# -------------------- Threading & cache env (portable) --------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# macOS Accelerate/vecLib threads (no-op on Linux)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

JOBLIB_TEMP_FOLDER = os.environ.get("JOBLIB_TEMP_FOLDER", "/tmp/joblib")
SKCACHE_DIR        = os.environ.get("SKCACHE_DIR", "/tmp/skcache")
os.makedirs(JOBLIB_TEMP_FOLDER, exist_ok=True)
os.makedirs(SKCACHE_DIR, exist_ok=True)

print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()} | {platform.machine()}")
print("JOBLIB_TEMP_FOLDER:", JOBLIB_TEMP_FOLDER)
print("SKCACHE_DIR:", SKCACHE_DIR)

# -------------------- Imports --------------------
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless

import mne
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin, clone

try:
    from mord import LogisticIT
    HAVE_MORD = True
except Exception:
    HAVE_MORD = False

RANDOM_STATE = 42

# -------------------- Shared-preprocessing transformers --------------------
class FBFilter(BaseEstimator, TransformerMixin):
    """
    Stateless filter-bank transformer:
      input:  X shape (n_epochs, n_channels, n_times)
      output: list of arrays [Xf_band1, Xf_band2, ...], each like X
    """
    def __init__(self, filterbank, sfreq=250.0, fb_n_jobs=1):
        self.filterbank = list(filterbank)
        self.sfreq = float(sfreq)
        self.fb_n_jobs = int(fb_n_jobs)

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64, order="C")
        out = []
        for (f1, f2) in self.filterbank:
            Xf = mne.filter.filter_data(
                X, self.sfreq, f1, f2,
                n_jobs=self.fb_n_jobs, copy=True, verbose=False
            )
            out.append(Xf)
        return out

class CSPBank(BaseEstimator, TransformerMixin):
    """
    Fits one CSP per band on the list produced by FBFilter, concatenates features.
    """
    def __init__(self, n_components=4, log=True, reg="oas"):
        self.n_components = int(n_components)
        self.log = bool(log)
        self.reg = reg

    def fit(self, X_bands, y):
        self.csps_ = []
        for Xf in X_bands:
            csp = mne.decoding.CSP(n_components=self.n_components, log=self.log, reg=self.reg)
            csp.fit(Xf, y)
            self.csps_.append(csp)
        return self

    def transform(self, X_bands):
        feats = [csp.transform(Xf) for csp, Xf in zip(self.csps_, X_bands)]
        return np.hstack(feats)

# -------------------- Helpers --------------------
PIPELINE_CACHE = joblib.Memory(location=SKCACHE_DIR, verbose=0)

def make_linear_svc(C=1.0, random_state=42, max_iter=10000, tol=1e-3):
    return LinearSVC(C=C, random_state=random_state, max_iter=max_iter, tol=tol, dual=False)

def make_zhang_pipeline(clf="rbf_svm", use_selector=True, fbands=None, k_features=24, fb_n_jobs=1):
    if fbands is None:
        fbands = [(8,12),(10,14),(12,16),(14,20),(20,26),(26,30)]
    steps = [
        ("fb",  FBFilter(filterbank=fbands, sfreq=250.0, fb_n_jobs=fb_n_jobs)),
        ("csp", CSPBank(n_components=4, log=True)),
    ]
    if use_selector:
        steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=k_features)))
    if clf == "rbf_svm":
        steps += [("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=False, random_state=RANDOM_STATE))]
    elif clf == "linear_svm":
        steps += [("scale", StandardScaler()),
                  ("clf", make_linear_svc(C=1.0, random_state=RANDOM_STATE))]
    elif clf == "lda":
        steps += [("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
    elif clf == "gnb":
        from sklearn.naive_bayes import GaussianNB
        steps += [("clf", GaussianNB())]
    else:
        raise ValueError("Unsupported clf")
    return Pipeline(steps, memory=PIPELINE_CACHE)

def make_dev_pipeline(band=(8,30), clf="linear_svm", fb_n_jobs=1):
    steps = [
        ("fb",  FBFilter(filterbank=[band], sfreq=250.0, fb_n_jobs=fb_n_jobs)),
        ("csp", CSPBank(n_components=6, log=True)),
    ]
    if clf == "linear_svm":
        steps += [("scale", StandardScaler()),
                  ("clf", make_linear_svc(C=1.0, random_state=RANDOM_STATE))]
    elif clf == "rbf_svm":
        steps += [("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=False, random_state=RANDOM_STATE))]
    elif clf == "lda":
        steps += [("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
    elif clf == "ordinal":
        if HAVE_MORD:
            steps += [("clf", LogisticIT())]
        else:
            raise RuntimeError("mord not installed; cannot build ordinal model.")
    else:
        raise ValueError("Unsupported clf")
    return Pipeline(steps, memory=PIPELINE_CACHE)

def parse_subjects(spec, avail):
    if spec is None: return list(avail)
    if isinstance(spec, (list, tuple, np.ndarray)):
        subs = [int(s) for s in spec]
    elif isinstance(spec, str):
        subs=[]
        for tok in spec.replace(" ","").split(","):
            if not tok: continue
            if "-" in tok:
                a,b = map(int, tok.split("-")); subs += list(range(min(a,b), max(a,b)+1))
            else:
                subs.append(int(tok))
        seen=set(); subs=[s for s in subs if not (s in seen or seen.add(s))]
    else:
        raise ValueError("SUBJECTS must be None, list/tuple, or '1-3,5' string.")
    aset=set(avail); return [s for s in subs if s in aset]

def normalize_label(z):
    return z.strip().lower() if isinstance(z, str) else z

def encode_labels(y, meta, force_zero_based=True):
    y_norm = [normalize_label(v) for v in y]
    if isinstance(meta, dict) and meta.get("classes"):
        classes = [normalize_label(c) for c in meta["classes"]]
    else:
        classes = list(pd.unique(pd.Series(y_norm)))
        try: classes = sorted(classes)
        except Exception: pass
    mapping = {lab: i for i, lab in enumerate(classes)}
    y_enc = np.asarray([mapping[v] for v in y_norm], dtype=int)
    if not force_zero_based:
        y_enc = y_enc + 1
    return y_enc, mapping

def flatten_params(params):
    if not params: return {}
    flat = {}
    for k, v in params.items():
        key = f"param_{k}"
        try:
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[key] = v
            else:
                flat[key] = json.dumps(v)
        except Exception:
            flat[key] = str(v)
    return flat

def save_model_any(obj, out_dir, base_name):
    out = {"ok": False, "path": None, "format": None, "error": None}
    try:
        p_joblib = os.path.join(out_dir, base_name + ".joblib")
        joblib.dump(obj, p_joblib)
        out.update({"ok": True, "path": p_joblib, "format": "joblib"})
        return out
    except Exception as e:
        err1 = f"joblib: {type(e).__name__}: {e}"
    try:
        p_pkl = os.path.join(out_dir, base_name + ".pkl")
        with open(p_pkl, "wb") as f:
            pickle.dump(obj, f)
        out.update({"ok": True, "path": p_pkl, "format": "pickle"})
        return out
    except Exception as e:
        err2 = f"pickle: {type(e).__name__}: {e}"
        out["error"] = f"{err1} | {err2}"
        return out

def _normalize_subject_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.isna().mean() > 0.5:
        s_num = pd.to_numeric(s.str.extract(r"(\d+)$")[0], errors="coerce")
    return s_num.astype("Int64")

def _prep_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pipeline" not in out.columns and "Pipeline" in out.columns:
        out = out.rename(columns={"Pipeline": "pipeline"})
    if "subject" in out.columns:
        out["subject_key"] = _normalize_subject_series(out["subject"])
    else:
        out["subject_key"] = pd.Series([pd.NA]*len(out), dtype="Int64")
    if "pipeline" in out.columns:
        out["pipeline_key"] = out["pipeline"].astype(str)
    else:
        out["pipeline_key"] = ""
    return out

def safe_merge_params_into_results(res_sub: pd.DataFrame, params_df: pd.DataFrame) -> pd.DataFrame:
    left  = _prep_for_merge(res_sub)
    right = _prep_for_merge(params_df)
    drop_cols = [c for c in ["subject","pipeline"] if c in right.columns]
    right = right.drop(columns=drop_cols)
    merged = left.merge(right, on=["subject_key","pipeline_key"], how="left", validate="m:1")
    merged = merged.rename(columns={"subject_key":"subject", "pipeline_key":"pipeline"})
    return merged

# -------------------- CV builders --------------------
CPU_COUNT = os.cpu_count() or 8
DEFAULT_GRID_JOBS = str(min( max(1, CPU_COUNT // 2), CPU_COUNT))
GRIDSEARCH_NJOBS = int(os.environ.get("GRIDSEARCH_NJOBS", DEFAULT_GRID_JOBS))
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
DEV_BANDS = [[(7,13)],[(8,14)],[(9,15)],[(12,18)],[(14,22)],[(16,26)],[(8,30)]]

def zhang_grid(clf="rbf_svm", use_selector=True):
    base = make_zhang_pipeline(clf=clf, use_selector=use_selector, k_features=24, fb_n_jobs=1)
    if clf == "rbf_svm":
        grid = {"select__k":[12,24,36], "csp__n_components":[2,4,6],
                "clf__C":[0.5,1.0,2.0], "clf__gamma":["scale",0.05,0.1]}
    elif clf == "linear_svm":
        grid = {"select__k":[12,24,36], "csp__n_components":[2,4,6],
                "clf__C":[0.5,1.0,2.0]}
    elif clf in ["lda","gnb"]:
        grid = {"select__k":[12,24,36], "csp__n_components":[2,4,6]}
    else:
        raise ValueError("Unsupported clf")
    if not use_selector:
        grid = {k:v for k,v in grid.items() if not k.startswith("select__")}
    return GridSearchCV(
        base, grid, scoring="balanced_accuracy", cv=inner_cv,
        n_jobs=GRIDSEARCH_NJOBS, refit=True, verbose=2, pre_dispatch="2*n_jobs"
    )

def dev_grid(clf="linear_svm"):
    base = make_dev_pipeline(band=(8,30), clf=clf, fb_n_jobs=1)
    if clf == "linear_svm":
        grid = {"fb__filterbank":DEV_BANDS, "csp__n_components":[4,6,8], "clf__C":[0.25,1.0,4.0]}
    elif clf == "rbf_svm":
        grid = {"fb__filterbank":DEV_BANDS, "csp__n_components":[4,6,8], "clf__C":[0.5,2.0,8.0], "clf__gamma":["scale",0.05,0.1]}
    elif clf == "lda":
        grid = {"fb__filterbank":DEV_BANDS, "csp__n_components":[4,6,8]}
    elif clf == "ordinal":
        if not HAVE_MORD: raise RuntimeError("mord not installed")
        grid = {"fb__filterbank":DEV_BANDS, "csp__n_components":[4,6,8]}
    else:
        raise ValueError("Unsupported clf")
    return GridSearchCV(
        base, grid, scoring="balanced_accuracy", cv=inner_cv,
        n_jobs=GRIDSEARCH_NJOBS, refit=True, verbose=2, pre_dispatch="2*n_jobs"
    )

def assert_grid_matches_pipeline(name, gscv):
    est = gscv.estimator
    params = est.get_params(deep=True)
    for key in gscv.param_grid:
        base = key.split("__", 1)[0]
        if base not in params:
            raise ValueError(f"[{name}] param '{key}' refers to missing step '{base}'. Steps: {[n for n,_ in est.steps]}")

# -------------------- Prediction logging --------------------
def predict_scores(est, X):
    out = {}
    y_pred = est.predict(X)
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        for k in range(proba.shape[1]):
            out[f"proba_{k}"] = proba[:, k]
    elif hasattr(est, "decision_function"):
        dec = est.decision_function(X)
        dec = np.atleast_2d(dec)
        if dec.ndim == 1:
            out["decision"] = dec
        else:
            for k in range(dec.shape[1]):
                out[f"decision_{k}"] = dec[:, k]
    return y_pred, out

def write_trial_predictions_for_subject(
    subj, dataset, paradigm, pipelines_dict, out_dir,
    outer_splits=5, save_inner_val_preds=True, incremental=True
):
    """
    OUTER CV: always logs test predictions (cv_level='outer').
    INNER CV: optionally logs validation predictions for tuned hyper-params (cv_level='inner').
    If incremental=True, appends per outer fold so the CSV grows during the run.
    """
    from sklearn.model_selection import StratifiedKFold
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
    y_enc, label_map = encode_labels(y, meta, force_zero_based=True)

    id_cols = {}
    if isinstance(meta, dict):
        for key in ("session", "run", "event_id", "file_id", "tmin", "tmax"):
            try:
                if key in meta and len(meta[key]) == len(y_enc):
                    id_cols[key] = meta[key]
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bnci2014_001_subject_{int(subj):02d}_trial_predictions.csv")
    wrote_header = os.path.exists(out_path) and os.path.getsize(out_path) > 0

    for pname, est in pipelines_dict.items():
        fold_id = 0
        for train_idx, test_idx in outer_cv.split(X, y_enc):
            fold_id += 1
            est_fold = clone(est)
            est_fold.fit(X[train_idx], y_enc[train_idx])
            est_infer = getattr(est_fold, "best_estimator_", est_fold)

            # OUTER test predictions
            y_pred_out, score_out = predict_scores(est_infer, X[test_idx])
            rows_outer = []
            for i_rel, i_abs in enumerate(test_idx):
                row = {
                    "cv_level": "outer",
                    "subject": int(subj),
                    "pipeline": pname,
                    "outer_fold": fold_id,
                    "trial_index": int(i_abs),
                    "y_true": int(y_enc[i_abs]),
                    "y_pred": int(y_pred_out[i_rel]),
                    "label_mapping_json": json.dumps(label_map),
                }
                for k, v in score_out.items():
                    row[k] = float(v[i_rel])
                for k, arr in id_cols.items():
                    row[k] = arr[i_abs]
                rows_outer.append(row)

            # INNER validation predictions for tuned params (optional)
            rows_inner = []
            if save_inner_val_preds:
                inner = getattr(est_fold, "cv", StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE))
                X_tr, y_tr = X[train_idx], y_enc[train_idx]
                best_params = getattr(est_fold, "best_params_", {}) or {}
                base_est = getattr(est_fold, "estimator", None)
                if base_est is None:
                    base_est = clone(est_infer)
                else:
                    base_est = clone(base_est)
                    if best_params:
                        base_est.set_params(**best_params)

                inner_fold_id = 0
                for in_tr_idx, in_val_idx in inner.split(X_tr, y_tr):
                    inner_fold_id += 1
                    est_inner = clone(base_est)
                    est_inner.fit(X_tr[in_tr_idx], y_tr[in_tr_idx])
                    y_pred_in, score_in = predict_scores(est_inner, X_tr[in_val_idx])

                    for i_rel, i_abs in enumerate(in_val_idx):
                        abs_idx = int(train_idx[i_abs])  # map back to global index
                        row = {
                            "cv_level": "inner",
                            "subject": int(subj),
                            "pipeline": pname,
                            "outer_fold": fold_id,
                            "inner_fold": inner_fold_id,
                            "trial_index": abs_idx,
                            "y_true": int(y_tr[i_abs]),
                            "y_pred": int(y_pred_in[i_rel]),
                            "label_mapping_json": json.dumps(label_map),
                            "best_params_json": json.dumps(best_params),
                        }
                        for k, v in score_in.items():
                            row[k] = float(v[i_rel])
                        for k, arr in id_cols.items():
                            row[k] = arr[abs_idx]
                        rows_inner.append(row)

            # write incrementally per fold
            if incremental:
                for chunk in (rows_outer, rows_inner):
                    if not chunk: continue
                    df_chunk = pd.DataFrame(chunk)
                    df_chunk.to_csv(out_path, index=False, mode="a", header=not wrote_header)
                    wrote_header = True
                print(f"[{datetime.now().strftime('%H:%M:%S')}] appended "
                      f"{len(rows_outer)+len(rows_inner)} rows → {out_path}")
            else:
                # collect to return at end (not used in this script)
                pass

    return out_path

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train BNCI2014_001 pipelines (shared-preprocessing cache)")
    parser.add_argument("--subjects", type=str, default="1-9", help="e.g. '1-3,5,7' or '1-9' or None for all")
    parser.add_argument("--fmin", type=float, default=7.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--eval_njobs", type=int, default=int(os.environ.get("EVAL_NJOBS", "2")))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save_models", action="store_true", default=True)
    parser.add_argument("--save_only_best", action="store_true", default=True)
    parser.add_argument("--write_predictions", action="store_true", default=True)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--save_inner_val_preds", action="store_true", default=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_bnci2014_001")
    args = parser.parse_args()

    CHECKPOINT_DIR = args.checkpoint_dir
    MODELS_DIR = os.path.join(CHECKPOINT_DIR, "models")
    PRED_DIR   = os.path.join(CHECKPOINT_DIR, "predictions")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    def ck_path(subj): return os.path.join(CHECKPOINT_DIR, f"bnci2014_001_subject_{int(subj):02d}.csv")

    # Build pipelines (all share the same Memory cache)
    pipelines = {
        "Zhang_FBCSP_MI_RBFSVM_CV": zhang_grid("rbf_svm", use_selector=True),
        "Zhang_FBCSP_MI_LinSVM_CV": zhang_grid("linear_svm", use_selector=True),
        "Zhang_FBCSP_MI_LDA_CV":    zhang_grid("lda", use_selector=True),
        "Zhang_FBCSP_MI_GNB_CV":    zhang_grid("gnb", use_selector=True),
        "Zhang_FBCSP_noFS_RBFSVM_CV": zhang_grid("rbf_svm", use_selector=False),
        "Zhang_FBCSP_noFS_LinSVM_CV": zhang_grid("linear_svm", use_selector=False),
        "Dev_CSP_LinSVM_CV": dev_grid("linear_svm"),
        "Dev_CSP_RBFSVM_CV": dev_grid("rbf_svm"),
        "Dev_CSP_LDA_CV":    dev_grid("lda"),
    }
    if HAVE_MORD:
        pipelines["Dev_CSP_Ordinal_CV"] = dev_grid("ordinal")

    for name, g in pipelines.items():
        assert_grid_matches_pipeline(name, g)

    # Subjects
    AVAILABLE_SUBJECTS = list(BNCI2014_001().subject_list)
    subjects = parse_subjects(args.subjects, AVAILABLE_SUBJECTS)
    if not subjects:
        raise RuntimeError("No valid subjects to run.")
    print("Subjects:", subjects)
    print(f"GRIDSEARCH_NJOBS={GRIDSEARCH_NJOBS} | EVAL_NJOBS={args.eval_njobs}")

    # Loop subjects
    start_all = time.perf_counter()
    for subj in subjects:
        # Skip if checkpoint exists (still write predictions if missing)
        ck = ck_path(subj)
        if args.resume and os.path.exists(ck):
            try:
                tmp = pd.read_csv(ck)
                if not tmp.empty:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SKIP subject {subj} (checkpoint exists)")
                    if args.write_predictions:
                        pred_csv = os.path.join(PRED_DIR, f"bnci2014_001_subject_{int(subj):02d}_trial_predictions.csv")
                        if not os.path.exists(pred_csv):
                            dataset = BNCI2014_001(); dataset.subject_list = [subj]
                            paradigm = MotorImagery(n_classes=4, fmin=args.fmin, fmax=args.fmax)
                            _ = write_trial_predictions_for_subject(
                                subj, dataset, paradigm, pipelines, PRED_DIR,
                                outer_splits=args.outer_splits, save_inner_val_preds=args.save_inner_val_preds, incremental=True
                            )
                    continue
            except Exception:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: unreadable checkpoint; re-running subject {subj}")

        dataset = BNCI2014_001(); dataset.subject_list = [subj]
        paradigm = MotorImagery(n_classes=4, fmin=args.fmin, fmax=args.fmax)
        evaluation = WithinSessionEvaluation(
            paradigm=paradigm, datasets=[dataset], overwrite=True,
            n_jobs=args.eval_njobs, hdf5_path=os.path.join(CHECKPOINT_DIR, "models")
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] START eval subject {subj} …")
        t0 = time.perf_counter()
        try:
            res_sub = evaluation.process(pipelines)  # MOABB results DF (outer CV)
            if 'subject' not in res_sub.columns:
                res_sub['subject'] = int(subj)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] FAIL eval subject {subj}: {type(e).__name__}: {e}")
            del evaluation, dataset; gc.collect()
            continue

        # Fit final models on all data + save inner CV grids + models
        param_rows = []
        if args.save_models:
            try:
                X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
                y_fit, label_map = encode_labels(y, meta, force_zero_based=True)

                for name, est in pipelines.items():
                    est_fit = clone(est)
                    fit_error = None
                    saved = {"ok": False, "path": None, "format": None, "error": None}
                    best_params = None
                    est_obj = None
                    pipe_dir = os.path.join(MODELS_DIR := os.path.join(CHECKPOINT_DIR, "models"), name, f"subject-{int(subj):02d}")
                    os.makedirs(pipe_dir, exist_ok=True)

                    try:
                        est_fit.fit(X, y_fit)
                        if hasattr(est_fit, "cv_results_"):
                            try:
                                cv_df = pd.DataFrame(est_fit.cv_results_)
                                cv_df.to_csv(os.path.join(pipe_dir, "inner_cv_results.csv"), index=False)
                            except Exception as e_cv:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: inner_cv_results save failed ({name}, subj {subj}): {e_cv}")

                        if hasattr(est_fit, "best_params_"):
                            best_params = est_fit.best_params_
                        if hasattr(est_fit, "best_estimator_") and args.save_only_best:
                            est_obj = est_fit.best_estimator_
                        else:
                            est_obj = est_fit

                        saved = save_model_any(est_obj, pipe_dir, "model")

                    except Exception as e_fit:
                        fit_error = f"{type(e_fit).__name__}: {e_fit}"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: fit failed (subject {subj}, pipeline {name}): {fit_error}")
                        print(traceback.format_exc(limit=2))

                    row = {
                        "subject": int(subj),
                        "pipeline": str(name),
                        "label_mapping_json": json.dumps(label_map),
                        "model_saved": saved["ok"],
                        "model_path": saved["path"],
                        "model_format": saved["format"],
                        "fit_error": fit_error,
                    }
                    row.update(flatten_params(best_params))
                    param_rows.append(row)

            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: data fetch/save block failed for subject {subj}: {type(e).__name__}: {e}")

        # Merge params into MOABB results
        if param_rows:
            params_df = pd.DataFrame(param_rows)
            res_sub = safe_merge_params_into_results(res_sub, params_df)

        # Write per-subject MOABB CSV
        ck = ck_path(subj)
        res_sub.to_csv(ck, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE eval subject {subj} in {(time.perf_counter()-t0)/60:.1f} min → {ck} (rows={len(res_sub)})")

        # Per-trial predictions (incremental append)
        if args.write_predictions:
            try:
                _ = write_trial_predictions_for_subject(
                    subj, dataset, paradigm, pipelines, os.path.join(CHECKPOINT_DIR, "predictions"),
                    outer_splits=args.outer_splits, save_inner_val_preds=args.save_inner_val_preds, incremental=True
                )
            except Exception as e_pred:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARN: per-trial predictions failed for subject {subj}: {e_pred}")

        del evaluation, dataset
        gc.collect()

    # Merge subjects (MOABB results)
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "bnci2014_001_subject_*.csv")))
    if files:
        try:
            merged = pd.concat([pd.read_csv(fp) for fp in files if os.path.getsize(fp) > 0], ignore_index=True)
            merged.to_csv(os.path.join(CHECKPOINT_DIR, "bnci2014_001_merged_results.csv"), index=False)
            print(f"Merged {len(files)} subject files → bnci2014_001_merged_results.csv (rows={len(merged)})")
        except Exception as e:
            print("Merge failed:", e)

    # Merge per-trial predictions
    pred_dir = os.path.join(CHECKPOINT_DIR, "predictions")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "bnci2014_001_subject_*_trial_predictions.csv")))
    if pred_files:
        try:
            pred_merged = pd.concat([pd.read_csv(fp) for fp in pred_files if os.path.getsize(fp) > 0], ignore_index=True)
            pred_merged.to_csv(os.path.join(pred_dir, "bnci2014_001_all_trial_predictions.csv"), index=False)
            print(f"Merged {len(pred_files)} prediction files → bnci2014_001_all_trial_predictions.csv (rows={len(pred_merged)})")
        except Exception as e:
            print("Prediction merge failed:", e)

    print(f"ALL DONE in {(time.perf_counter()-start_all)/60:.1f} min")
    print("Checkpoints:", CHECKPOINT_DIR)
    print("Models dir:", os.path.join(CHECKPOINT_DIR, "models"))
    print("Predictions dir:", pred_dir)

if __name__ == "__main__":
    # late imports used by helpers
    import numpy as np
    import pandas as pd
    main()

