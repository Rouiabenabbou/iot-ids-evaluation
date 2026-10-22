# Step 1: one XGBoost run on the supplied split, to measure (not just bound) how much
# test rows that also occur in train ("seen") inflate per-family results.
# CICIoT2023 Kaggle copy (himadri07/ciciot2023). 8 classes: Benign + 7 attack families.
# Paste each "# %%" block into its own Kaggle CPU notebook cell. Outputs: /kaggle/working/step1/
# Nothing here touches hyperparameters on the test set: early stopping uses validation only.

# %% Cell 1 - settings, paths, family mapping (same rules as the SQ1 script)
import os, re, json, time, platform
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

SEED = 42
TRAIN_CAP = None        # None = use all training rows. If the run is too slow or runs out of
                        # memory, set e.g. 1_500_000 (stratified, small families kept whole).
CHUNK = 500_000
ROOT, OUT = "/kaggle/input", "/kaggle/working/step1"
os.makedirs(OUT, exist_ok=True)

FILES = {}
for d, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(".csv"):
            FILES[os.path.splitext(f)[0].lower()] = os.path.join(d, f)
assert {"train", "test", "validation"} <= set(FILES), FILES

head = pd.read_csv(FILES["train"], nrows=1000)
LABEL = head.select_dtypes(exclude="number").columns[0]
FEATURES = [c for c in head.columns if c != LABEL]
assert len(FEATURES) == 46, len(FEATURES)

RULES = [(r"^benign", "Benign"), (r"^ddos", "DDoS"), (r"^dos", "DoS"), (r"^mirai", "Mirai"),
         (r"^recon|vulnerabilityscan", "Recon"), (r"spoof", "Spoofing"),
         (r"bruteforce", "BruteForce"),
         (r"sqlinjection|commandinjection|xss|uploading|backdoor|browserhijack", "Web")]
FAMILIES = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web"]
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
def family(label):
    s = str(label).strip().lower()
    for p, fam in RULES:
        if re.search(p, s):
            return fam
    raise ValueError(f"Unmapped label: {label}")

print(f"xgboost {xgb.__version__} | pandas {pd.__version__} | CPU cores {os.cpu_count()}")

# %% Cell 2 - load each split as float32 in chunks, keep a 64-bit hash of every feature row
def load(path):
    Xs, ys, hs = [], [], []
    dtypes = {c: "float64" for c in FEATURES}
    dtypes[LABEL] = "string"
    for ch in pd.read_csv(path, dtype=dtypes, chunksize=CHUNK):
        hs.append(pd.util.hash_pandas_object(ch[FEATURES], index=False).to_numpy())  # exact values
        Xs.append(ch[FEATURES].to_numpy(dtype=np.float32))
        ys.append(ch[LABEL].map(lambda l: FAM_ID[family(l)]).to_numpy(dtype=np.int8))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(hs)

t0 = time.time()
Xtr, ytr, htr = load(FILES["train"])
Xva, yva, hva = load(FILES["validation"])
Xte, yte, hte = load(FILES["test"])
print(f"loaded in {time.time() - t0:.0f} s | train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
print("train rows per family:", dict(zip(FAMILIES, np.bincount(ytr, minlength=8))))

# %% Cell 3 - optional stratified cap on training rows (all rows of small families kept)
if TRAIN_CAP is not None and len(ytr) > TRAIN_CAP:
    rng = np.random.default_rng(SEED)
    counts = np.bincount(ytr, minlength=8)
    small = counts < 100_000
    budget = TRAIN_CAP - counts[small].sum()
    frac = budget / counts[~small].sum()
    keep = np.zeros(len(ytr), dtype=bool)
    for k in range(8):
        idx = np.flatnonzero(ytr == k)
        keep[idx if small[k] else rng.choice(idx, int(round(len(idx) * frac)), replace=False)] = True
    Xtr, ytr, htr = Xtr[keep], ytr[keep], htr[keep]
    print(f"capped training set: {len(ytr):,} rows")

# "Seen" is defined against the training rows actually used
train_hashes = np.unique(htr)
seen_te = np.isin(hte, train_hashes)
print(f"test rows with an exact feature copy in the training rows used: "
      f"{seen_te.sum():,} of {len(yte):,} ({100 * seen_te.mean():.2f}%)")

# %% Cell 4 - train XGBoost (fixed settings; early stopping on validation only)
params = dict(objective="multi:softprob", num_class=8, tree_method="hist", max_depth=8,
              learning_rate=0.2, subsample=0.8, colsample_bytree=0.8, max_bin=256,
              eval_metric="mlogloss", seed=SEED, nthread=os.cpu_count())
dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=256)
dval = xgb.DMatrix(Xva, label=yva)
t0 = time.time()
booster = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval, "val")],
                    early_stopping_rounds=20, verbose_eval=25)
train_s = time.time() - t0
print(f"training time {train_s:.0f} s | best iteration {booster.best_iteration}")

# %% Cell 5 - predict on test once; per-family metrics on all, seen and unseen test rows
dtest = xgb.DMatrix(Xte)
t0 = time.time()
pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1)).argmax(axis=1)
pred_s = time.time() - t0

def per_family(mask, name):
    p, r, f, n = precision_recall_fscore_support(yte[mask], pred[mask], labels=range(8),
                                                 zero_division=0)
    return pd.DataFrame({f"{name}_n": n, f"{name}_precision": p, f"{name}_recall": r,
                         f"{name}_f1": f}, index=FAMILIES)

res = pd.concat([per_family(np.ones(len(yte), bool), "all"),
                 per_family(seen_te, "seen"),
                 per_family(~seen_te, "unseen")], axis=1)
for part in ("seen", "unseen"):  # no rows in a subset -> metrics undefined, not 0
    res.loc[res[f"{part}_n"] == 0, [f"{part}_precision", f"{part}_recall", f"{part}_f1"]] = np.nan
res["recall_seen_minus_unseen"] = res["seen_recall"] - res["unseen_recall"]
pd.set_option("display.width", 220)
print(res.round(4).to_string())

summary = {
    "macro_f1_all": f1_score(yte, pred, average="macro"),
    "macro_f1_unseen_only": f1_score(yte[~seen_te], pred[~seen_te], average="macro"),
    "weighted_f1_all": f1_score(yte, pred, average="weighted"),
    "benign_fpr_all": float(((pred != FAM_ID["Benign"]) & (yte == FAM_ID["Benign"])).sum()
                            / (yte == FAM_ID["Benign"]).sum()),
    "test_seen_share": float(seen_te.mean()),
    "train_rows_used": int(len(ytr)), "train_seconds": round(train_s, 1),
    "predict_seconds_test": round(pred_s, 2),
    "latency_us_per_row": round(1e6 * pred_s / len(yte), 3),
    "best_iteration": int(booster.best_iteration),
    "params": params, "train_cap": TRAIN_CAP, "seed": SEED,
    "xgboost": xgb.__version__, "python": platform.python_version(), "cpu_cores": os.cpu_count(),
}
for k in ("macro_f1_all", "macro_f1_unseen_only", "weighted_f1_all", "benign_fpr_all"):
    print(f"{k}: {summary[k]:.4f}")

res.to_csv(f"{OUT}/per_family_seen_unseen.csv")
pd.DataFrame(confusion_matrix(yte, pred, labels=range(8)), index=FAMILIES,
             columns=FAMILIES).to_csv(f"{OUT}/confusion_matrix_test.csv")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print("Saved:", os.listdir(OUT))
