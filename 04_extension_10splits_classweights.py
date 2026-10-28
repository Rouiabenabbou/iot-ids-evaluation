# Step 3: extension of the main experiment (same protocol as step 2).
#   (a) 10 splits instead of 5 (seeds 0-9; seeds 0-4 reproduce step 2 exactly for DT, RF, XGB).
#   (b) Class-weighted versions of DT, RF and XGB ("_bal": weights inversely proportional to the
#       class counts in the training rows, sklearn "balanced" rule). Tests whether the rankings on
#       the rare families survive the most common fix for class imbalance.
#   (c) Cell 10: corrected resampled t-test (Nadeau & Bengio 2003) across the 10 splits for the
#       model pairs of interest, with Holm correction for the number of tests.
#   LR is not re-run (5 seeds in step 2 already; it is the slowest model and clearly the worst).
#
# Original step 2 header:
# Minority-family reliability of LR, DT, RF and XGBoost on CICIoT2023
# (Kaggle copy himadri07/ciciot2023), 8 classes = Benign + 7 attack families.
#
# Protocol
#   1. Pool train + validation + test, then remove exact duplicate feature vectors (keep one copy).
#      Vectors that carry more than one label are dropped entirely and counted.
#   2. For each of N_SPLITS seeds: a fresh stratified 70/15/15 split of the de-duplicated pool.
#      Train and test therefore share no feature vector (checked by an assert).
#   3. Training rows: cap of TRAIN_CAP made by subsampling only DDoS, DoS and Mirai (uniformly at
#      random within each); Benign, Recon, Spoofing, BruteForce and Web are kept whole.
#      Tuning rows: cap of VAL_CAP drawn from the validation part, same rule.
#   4. Each model has a small fixed grid. The grid point with the best validation macro-F1 is kept.
#      The test part of that split is predicted exactly once, by that kept model.
#   5. Per family: precision, recall, F1 with 95% bootstrap CIs (1,000 resamples of the test set).
#      Across seeds: mean, SD, min, max. Model pairs: per-seed difference with a paired bootstrap CI,
#      and whether the sign of the difference is the same in every seed.
#   Timing is recorded for description only (shared Kaggle CPUs are noisy).
#
# Run it as a Kaggle CPU notebook, one "# %%" block per cell. For the full run use
# "Save Version" -> "Save & Run All (Commit)" so it keeps running with the browser closed.
# Each finished (seed, model) is saved to OUT/runs/ and skipped if the cell is run again.
# Outputs: /kaggle/working/step3/

# %% Cell 1 - settings, paths, family mapping (same rules as the SQ1 and step 1 scripts)
import os, re, json, time, glob, platform, warnings
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

ROOT, OUT = "/kaggle/input", "/kaggle/working/step3"
SEEDS = list(range(10))          # one split per seed (step 2 used 0-4)
MODELS = ["DT", "RF", "XGB", "DT_bal", "RF_bal", "XGB_bal"]   # _bal = class-weighted
TRAIN_CAP = 1_000_000           # training rows per split (stratified)
VAL_CAP = 300_000               # validation rows used for model selection (stratified)
SUBSAMPLED = ["DDoS", "DoS", "Mirai"]   # only these high-volume families are subsampled by the caps;
                                # all other families (Benign, Recon, Spoofing, BruteForce, Web) are kept
                                # whole, so their relative proportions stay as in the data. (Version 1
                                # kept only families under 10% of the cap whole; that shrank Benign 5.8x
                                # relative to Recon/Spoofing and inflated the benign false-positive rate.)
N_BOOT = 1000
LR_MAX_ITER = 2000              # run 1 used 500: LR did not converge in any seed
CHUNK = 500_000
NCPU = os.cpu_count()

GRIDS = {
    "LR":  [{"C": c} for c in (0.1, 1.0, 10.0)],
    "DT":  [{"max_depth": d} for d in (10, 20, None)],
    "RF":  [{"max_depth": d} for d in (20, None)],
    "XGB": [{"max_depth": d} for d in (6, 10)],
}
for _m in ("DT", "RF", "XGB"):
    GRIDS[f"{_m}_bal"] = GRIDS[_m]          # same grid, only the class weights differ

def balanced_weights(ytr):
    """sklearn 'balanced' rule: n_samples / (n_classes * count of the row's class)."""
    counts = np.bincount(ytr, minlength=K).astype(np.float64)
    present = counts > 0
    w_class = np.zeros(K)
    w_class[present] = len(ytr) / (present.sum() * counts[present])
    return w_class[ytr]

def make_model(name, p, seed):
    if name == "LR":   # scaler is fitted on the training rows only, inside the pipeline
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=p["C"], max_iter=LR_MAX_ITER, random_state=seed))
    cw = "balanced" if name.endswith("_bal") else None
    if name in ("DT", "DT_bal"):
        return DecisionTreeClassifier(max_depth=p["max_depth"], class_weight=cw, random_state=seed)
    if name in ("RF", "RF_bal"):
        return RandomForestClassifier(n_estimators=100, max_depth=p["max_depth"], class_weight=cw,
                                      max_features="sqrt", n_jobs=NCPU, random_state=seed)
    raise ValueError(name)

XGB_BASE = dict(objective="multi:softprob", num_class=8, tree_method="hist", learning_rate=0.2,
                subsample=0.8, colsample_bytree=0.8, max_bin=256, eval_metric="mlogloss",
                nthread=NCPU)

for d in ("runs", "confusion"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

FILES = {}
for d, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(".csv"):
            FILES[os.path.splitext(f)[0].lower()] = os.path.join(d, f)
assert {"train", "test", "validation"} <= set(FILES), FILES

head = pd.read_csv(FILES["train"], nrows=1000)
text_cols = head.select_dtypes(exclude="number").columns.tolist()
assert len(text_cols) == 1, text_cols
LABEL = text_cols[0]
FEATURES = [c for c in head.columns if c != LABEL]
assert len(FEATURES) == 46, len(FEATURES)

RULES = [(r"^benign", "Benign"), (r"^ddos", "DDoS"), (r"^dos", "DoS"), (r"^mirai", "Mirai"),
         (r"^recon|vulnerabilityscan", "Recon"), (r"spoof", "Spoofing"),
         (r"bruteforce", "BruteForce"),
         (r"sqlinjection|commandinjection|xss|uploading|backdoor|browserhijack", "Web")]
FAMILIES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
K = len(FAMILIES)
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
def family(label):
    s = str(label).strip().lower()
    for p, fam in RULES:
        if re.search(p, s):
            return fam
    raise ValueError(f"Unmapped label: {label}")

print(f"python {platform.python_version()} | sklearn {sklearn.__version__} | "
      f"xgboost {xgb.__version__} | pandas {pd.__version__} | CPU cores {NCPU}")

# %% Cell 2 - read all three files in chunks into one pool (float32 features, int8 family id)
Xs, ys, hs = [], [], []
dtypes = {c: "float64" for c in FEATURES}
dtypes[LABEL] = "string"
label_map = {}
t0 = time.time()
for name in ("train", "validation", "test"):
    for ch in pd.read_csv(FILES[name], dtype=dtypes, chunksize=CHUNK):
        hs.append(pd.util.hash_pandas_object(ch[FEATURES], index=False).to_numpy())  # exact values
        Xs.append(ch[FEATURES].to_numpy(dtype=np.float32))
        for lab in ch[LABEL].unique():
            label_map.setdefault(lab, family(lab))
        ys.append(ch[LABEL].map(lambda l: FAM_ID[label_map[l]]).to_numpy(dtype=np.int8))
X, y, h = np.concatenate(Xs), np.concatenate(ys), np.concatenate(hs)
del Xs, ys, hs
print(f"pooled {len(y):,} rows in {time.time() - t0:.0f} s | {X.nbytes / 1e9:.2f} GB of features")
print("labels -> family:", json.dumps(dict(sorted(label_map.items())), indent=0))

# %% Cell 3 - remove exact duplicate feature vectors; drop vectors with conflicting labels
order = np.argsort(h, kind="stable")
hs_, ys_ = h[order], y[order]
starts = np.flatnonzero(np.r_[True, hs_[1:] != hs_[:-1]])
conflict = np.minimum.reduceat(ys_, starts) != np.maximum.reduceat(ys_, starts)
keep_idx = np.sort(order[starts[~conflict]])           # first copy of each clean vector
rows_before = np.bincount(y, minlength=K)
n_conflict_rows = int(np.add.reduceat(np.ones(len(hs_), np.int64), starts)[conflict].sum())
X, y, h = X[keep_idx], y[keep_idx], h[keep_idx]
del order, hs_, ys_, starts, keep_idx
rows_after = np.bincount(y, minlength=K)

dedup = pd.DataFrame({"rows_before": rows_before, "rows_after": rows_after}, index=FAMILIES)
dedup["removed_%"] = 100 * (1 - dedup["rows_after"] / dedup["rows_before"])
print(dedup.to_string())
print(f"total {rows_before.sum():,} -> {rows_after.sum():,} rows | "
      f"vectors with conflicting labels: {int(conflict.sum()):,} ({n_conflict_rows:,} rows dropped)")
dedup.to_csv(f"{OUT}/dedup_by_family.csv")
dedup_info = {"rows_before": int(rows_before.sum()), "rows_after": int(rows_after.sum()),
              "conflicting_vectors": int(conflict.sum()), "conflicting_rows": n_conflict_rows}

# %% Cell 4 - helpers: stratified split and cap, metrics from a confusion matrix, bootstrap
def cap(idx, n_max, rng):
    """Subsample idx to about n_max rows: families in SUBSAMPLED share one sampling fraction,
    every other family is kept whole."""
    if len(idx) <= n_max:
        return idx
    yy = y[idx]
    counts = np.bincount(yy, minlength=K)
    small = ~np.isin(np.arange(K), [FAM_ID[f] for f in SUBSAMPLED])   # kept whole
    frac = min(1.0, (n_max - counts[small].sum()) / max(1, counts[~small].sum()))
    assert frac > 0, "cap smaller than the families that are kept whole"
    parts = [idx[yy == k] if small[k] else
             rng.choice(idx[yy == k], int(round(counts[k] * frac)), replace=False)
             for k in range(K) if counts[k] > 0]
    return np.sort(np.concatenate(parts))

def make_split(seed):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y))
    tr, rest = train_test_split(all_idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    tr_c, va_c = cap(tr, TRAIN_CAP, rng), cap(va, VAL_CAP, rng)
    # no feature vector may appear in both training and test rows (float64 hash)
    assert not np.isin(h[te], h[tr]).any(), "train/test overlap after de-duplication"
    return tr_c, va_c, np.sort(te)

def metrics_from_cm(cm):
    """cm[..., true, pred] -> precision, recall, f1 per family (0 when undefined, as sklearn)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1

def cm_of(yt, yp):
    return np.bincount(yt.astype(np.int64) * K + yp, minlength=K * K).reshape(K, K)

def macro_f1(yt, yp):
    return float(metrics_from_cm(cm_of(yt, yp))[2].mean())

def bootstrap_ci(cm, rng):
    """Resampling n test rows with replacement = a multinomial draw over the confusion-matrix cells."""
    n = cm.sum()
    draws = rng.multinomial(n, cm.ravel() / n, size=N_BOOT).reshape(N_BOOT, K, K)
    p, r, f = metrics_from_cm(draws)
    out = {}
    for nm, v in (("precision", p), ("recall", r), ("f1", f)):
        out[f"{nm}_lo"], out[f"{nm}_hi"] = np.percentile(v, [2.5, 97.5], axis=0)
    out["macro_f1_lo"], out["macro_f1_hi"] = np.percentile(f.mean(axis=1), [2.5, 97.5])
    return out

# %% Cell 5 - one function per model: tune on validation, then predict test once
def fit_select(name, seed, Xtr, ytr, Xva, yva):
    tried = []
    best = None
    for p in GRIDS[name]:
        t0 = time.time()
        if name in ("XGB", "XGB_bal"):
            params = dict(XGB_BASE, max_depth=p["max_depth"], seed=seed)
            wtr = balanced_weights(ytr) if name == "XGB_bal" else None
            dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, max_bin=256)
            dva = xgb.DMatrix(Xva, label=yva)
            m = xgb.train(params, dtr, num_boost_round=300, evals=[(dva, "val")],
                          early_stopping_rounds=20, verbose_eval=False)
            fit_s = time.time() - t0
            pv = m.predict(dva, iteration_range=(0, m.best_iteration + 1)).argmax(1)
            extra = {"best_iteration": int(m.best_iteration)}
            del dtr
        else:
            m = make_model(name, p, seed)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                m.fit(Xtr, ytr)
            fit_s = time.time() - t0
            pv = m.predict(Xva)
            extra = {"converged": not any(issubclass(x.category, ConvergenceWarning) for x in w)} \
                if name == "LR" else {}
        score = macro_f1(yva, pv)
        tried.append({"params": p, "val_macro_f1": score, "fit_seconds": round(fit_s, 1), **extra})
        print(f"    {name} {p} val macro-F1 {score:.4f} ({fit_s:.0f} s)")
        if best is None or score > best[0]:
            best = (score, p, m, fit_s, extra)
        else:
            del m
    return best, tried

def predict(name, m, Xte):
    t0 = time.time()
    if name in ("XGB", "XGB_bal"):
        pred = m.predict(xgb.DMatrix(Xte), iteration_range=(0, m.best_iteration + 1)).argmax(1)
    else:
        pred = m.predict(Xte)
    return pred.astype(np.int8), time.time() - t0

# %% Cell 6 - the main loop (long: run with "Save & Run All (Commit)" for all seeds)
for seed in SEEDS:
    tr, va, te = make_split(seed)
    np.save(f"{OUT}/runs/seed{seed}_test_index.npy", te)
    print(f"seed {seed}: train {len(tr):,} | val (tuning) {len(va):,} | test {len(te):,} | "
          f"test rows per family {dict(zip(FAMILIES, np.bincount(y[te], minlength=K).tolist()))}")
    print(f"  train rows per family {dict(zip(FAMILIES, np.bincount(y[tr], minlength=K).tolist()))}")
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    for name in MODELS:
        tag = f"{OUT}/runs/seed{seed}_{name}"
        if os.path.exists(f"{tag}.json"):
            print(f"  {name}: already done, skipped")
            continue
        (val_score, p, m, fit_s, extra), tried = fit_select(name, seed, Xtr, ytr, Xva, yva)
        pred, pred_s = predict(name, m, Xte)
        del m
        np.save(f"{tag}_pred.npy", pred)
        cm = cm_of(yte, pred)
        pd.DataFrame(cm, index=FAMILIES, columns=FAMILIES).to_csv(
            f"{OUT}/confusion/seed{seed}_{name}.csv")
        rec = {"seed": seed, "model": name, "chosen_params": p, "val_macro_f1": val_score,
               "grid": tried, "fit_seconds_chosen": round(fit_s, 1),
               "predict_seconds_test": round(pred_s, 2),
               "latency_us_per_row": round(1e6 * pred_s / len(yte), 3),
               "n_train": int(len(tr)), "n_val_tuning": int(len(va)), "n_test": int(len(te)),
               "test_macro_f1": float(metrics_from_cm(cm)[2].mean()),
               "test_benign_fpr": float(1 - cm[FAM_ID["Benign"], FAM_ID["Benign"]]
                                        / cm[FAM_ID["Benign"]].sum()), **extra}
        json.dump(rec, open(f"{tag}.json", "w"), indent=1, default=str)
        print(f"  {name}: chose {p} | test macro-F1 {rec['test_macro_f1']:.4f} | "
              f"benign FPR {rec['test_benign_fpr']:.5f}")
    del Xtr, ytr, Xva, yva, Xte, yte

# %% Cell 7 - per-family results with bootstrap CIs, and the summary across seeds
rng = np.random.default_rng(12345)
rows, runs = [], []
for jf in sorted(glob.glob(f"{OUT}/runs/seed*_*.json")):
    r = json.load(open(jf))
    seed, name = r["seed"], r["model"]
    te = np.load(f"{OUT}/runs/seed{seed}_test_index.npy")
    pred = np.load(f"{OUT}/runs/seed{seed}_{name}_pred.npy")
    cm = cm_of(y[te], pred)
    p, rc, f = metrics_from_cm(cm)
    ci = bootstrap_ci(cm, rng)
    r["test_macro_f1_lo"], r["test_macro_f1_hi"] = ci["macro_f1_lo"], ci["macro_f1_hi"]
    runs.append({k: v for k, v in r.items() if k != "grid"})
    for k in range(K):
        rows.append({"seed": seed, "model": name, "family": FAMILIES[k], "n_test": int(cm[k].sum()),
                     "precision": p[k], "recall": rc[k], "f1": f[k],
                     **{c: ci[c][k] for c in ci if not c.startswith("macro")}})
per_family = pd.DataFrame(rows)
runs = pd.DataFrame(runs)
per_family.to_csv(f"{OUT}/per_family_by_seed.csv", index=False)
runs.to_csv(f"{OUT}/runs_summary.csv", index=False)

agg = per_family.groupby(["family", "model"])[["n_test", "precision", "recall", "f1"]].agg(
    ["mean", "std", "min", "max"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.drop(columns=["n_test_std", "n_test_min", "n_test_max"])
agg["n_seeds"] = per_family.groupby(["family", "model"]).size()
agg.to_csv(f"{OUT}/per_family_across_seeds.csv")

overall = runs.groupby("model")[["test_macro_f1", "test_benign_fpr", "fit_seconds_chosen",
                                 "latency_us_per_row"]].agg(["mean", "std"])
overall.columns = [f"{a}_{b}" for a, b in overall.columns]
overall.to_csv(f"{OUT}/overall_across_seeds.csv")
pd.set_option("display.width", 220)
print(overall.round(4).to_string())
minority = ["Web", "BruteForce", "Recon"]
print(agg.loc[minority, ["n_test_mean", "recall_mean", "recall_std", "f1_mean", "f1_std",
                         "n_seeds"]].round(4).to_string())

# %% Cell 8 - model pairs: per-seed differences with paired bootstrap CIs; sign consistency
def paired_diff(yt, pa, pb, rng):
    """Joint resample of (true, pred A, pred B) so both models see the same bootstrap test sets."""
    t = np.bincount(yt.astype(np.int64) * K * K + pa.astype(np.int64) * K + pb,
                    minlength=K ** 3)
    n = t.sum()
    d = rng.multinomial(n, t / n, size=N_BOOT).reshape(N_BOOT, K, K, K)
    fa = metrics_from_cm(d.sum(axis=3))   # (B, true, predA)
    fb = metrics_from_cm(d.sum(axis=2))   # (B, true, predB)
    pa_, pb_ = metrics_from_cm(t.reshape(K, K, K).sum(2)), metrics_from_cm(t.reshape(K, K, K).sum(1))
    out = {}
    for i, nm in ((1, "recall"), (2, "f1")):
        diff = fa[i] - fb[i]
        out[nm] = (pa_[i] - pb_[i], *np.percentile(diff, [2.5, 97.5], axis=0))
    md = fa[2].mean(1) - fb[2].mean(1)
    out["macro_f1"] = (pa_[2].mean() - pb_[2].mean(), *np.percentile(md, [2.5, 97.5]))
    return out

done = {(r.seed, r.model) for r in runs.itertuples()}
pair_rows = []
for seed in sorted({s for s, _ in done}):
    te = np.load(f"{OUT}/runs/seed{seed}_test_index.npy")
    preds = {m: np.load(f"{OUT}/runs/seed{seed}_{m}_pred.npy") for m in MODELS if (seed, m) in done}
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = paired_diff(y[te], preds[a], preds[b], rng)
            for metric in ("recall", "f1"):
                est, lo, hi = res[metric]
                for k in range(K):
                    pair_rows.append({"seed": seed, "model_a": a, "model_b": b, "metric": metric,
                                      "family": FAMILIES[k], "diff_a_minus_b": est[k],
                                      "ci_lo": lo[k], "ci_hi": hi[k]})
            est, lo, hi = res["macro_f1"]
            pair_rows.append({"seed": seed, "model_a": a, "model_b": b, "metric": "macro_f1",
                              "family": "ALL", "diff_a_minus_b": est, "ci_lo": lo, "ci_hi": hi})
pairs = pd.DataFrame(pair_rows)
pairs["ci_excludes_0"] = (pairs["ci_lo"] > 0) | (pairs["ci_hi"] < 0)
pairs.to_csv(f"{OUT}/pairwise_by_seed.csv", index=False)

g = pairs.groupby(["metric", "family", "model_a", "model_b"])
pair_sum = pd.DataFrame({
    "n_seeds": g.size(),
    "mean_diff": g["diff_a_minus_b"].mean(), "sd_diff": g["diff_a_minus_b"].std(),
    "seeds_a_better": g["diff_a_minus_b"].apply(lambda s: int((s > 0).sum())),
    "seeds_b_better": g["diff_a_minus_b"].apply(lambda s: int((s < 0).sum())),
    "seeds_ci_excludes_0": g["ci_excludes_0"].sum(),
})
# "consistent" = same sign of the difference in every seed (ties count as not consistent)
pair_sum["consistent_sign"] = ((pair_sum["seeds_a_better"] == pair_sum["n_seeds"]) |
                               (pair_sum["seeds_b_better"] == pair_sum["n_seeds"]))
pair_sum.to_csv(f"{OUT}/pairwise_across_seeds.csv")
print(pair_sum.loc[(slice(None), ["Web", "BruteForce", "Recon", "ALL"]), :].round(4).to_string())

# %% Cell 9 - save the configuration so the run can be reported and repeated
json.dump({"dataset": "Kaggle himadri07/ciciot2023 (train+validation+test pooled)",
           "label_column": LABEL, "n_features": len(FEATURES), "families": FAMILIES,
           "label_to_family": label_map, "dedup": dedup_info, "seeds": SEEDS, "models": MODELS,
           "split": "stratified 70/15/15 per seed (sklearn train_test_split)",
           "train_cap": TRAIN_CAP, "val_cap": VAL_CAP, "subsampled_families": SUBSAMPLED, "cap_rule_version": 2,
           "grids": GRIDS, "xgb_base": XGB_BASE, "xgb_rounds": 300, "xgb_early_stopping": 20,
           "rf_n_estimators": 100, "lr_max_iter": LR_MAX_ITER, "selection": "best validation macro-F1",
           "class_weighting": "_bal models: sklearn 'balanced' rule on training rows (XGB via sample weights)",
           "bootstrap": f"{N_BOOT} multinomial resamples of the test confusion matrix, 95% percentile",
           "python": platform.python_version(), "sklearn": sklearn.__version__,
           "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "cpu_cores": NCPU}, open(f"{OUT}/config.json", "w"), indent=1, default=str)
print("Saved:", sorted(os.listdir(OUT)))

# %% Cell 10 - corrected resampled t-test across splits (Nadeau & Bengio 2003) + Holm correction
# The plain t-test over splits is too optimistic because the splits overlap. The corrected version
# multiplies the variance by (1/J + n_test/n_train), with J = number of splits.
from scipy import stats
n_tr = float(runs["n_train"].mean()); n_te = float(runs["n_test"].mean())
# Pairs decided in advance: the three tree models, each model vs its class-weighted version,
# and DT vs the class-weighted ensembles. Tested on F1 for Web, BruteForce, Recon and on macro-F1.
PAIRS_OF_INTEREST = [("DT", "RF"), ("DT", "XGB"), ("RF", "XGB"), ("DT", "DT_bal"), ("RF", "RF_bal"),
                     ("XGB", "XGB_bal"), ("DT", "RF_bal"), ("DT", "XGB_bal")]
tests = []
for (metric, fam, a, b), grp in pairs.groupby(["metric", "family", "model_a", "model_b"]):
    if (a, b) not in PAIRS_OF_INTEREST or metric == "recall":
        continue
    if fam not in ("Web", "BruteForce", "Recon", "ALL"):
        continue
    d = grp.sort_values("seed")["diff_a_minus_b"].to_numpy()
    J = len(d)
    if J < 3:
        continue
    var = d.var(ddof=1) * (1.0 / J + n_te / n_tr)
    if var > 0:
        t = d.mean() / np.sqrt(var); p = 2 * stats.t.sf(abs(t), df=J - 1)
    else:                       # identical predictions in every split
        t, p = (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    tests.append({"metric": metric, "family": fam, "model_a": a, "model_b": b, "n_splits": J,
                  "mean_diff": d.mean(), "t_corrected": t, "p_corrected": p})
tests = pd.DataFrame(tests)
# Holm correction over all tests in this table (8 pairs x 4 metrics = one family of tests)
ps = tests["p_corrected"].to_numpy(); order = np.argsort(ps); m = len(ps)
adj = np.maximum.accumulate((m - np.arange(m)) * ps[order]).clip(max=1.0)
tests["p_holm"] = np.nan
tests.loc[tests.index[order], "p_holm"] = adj
tests["significant_holm_0.05"] = tests["p_holm"] < 0.05
tests.to_csv(f"{OUT}/corrected_ttests.csv", index=False)
print(tests.sort_values(["family", "p_holm"]).round(4).to_string(index=False))
