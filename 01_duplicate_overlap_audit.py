# SQ1 gate: where does the train-test duplicate overlap fall, per label and per attack family?
# CICIoT2023 Kaggle copy (himadri07/ciciot2023). No model training. Reads each file in chunks.
# Paste each "# %%" block into its own Kaggle notebook cell. Outputs go to /kaggle/working/sq1/.

# %% Cell 1 - paths, label column and family mapping
import os, re, json
import numpy as np
import pandas as pd

ROOT = "/kaggle/input"
OUT = "/kaggle/working/sq1"
os.makedirs(OUT, exist_ok=True)
FILES = {}
for d, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(".csv"):
            FILES[os.path.splitext(f)[0].lower()] = os.path.join(d, f)
assert {"train", "test", "validation"} <= set(FILES), f"Expected train/test/validation CSVs, found {FILES}"

head = pd.read_csv(FILES["train"], nrows=1000)
text_cols = head.select_dtypes(exclude="number").columns.tolist()
assert len(text_cols) == 1, f"Expected exactly one text (label) column, found {text_cols}; set LABEL by hand"
LABEL = text_cols[0]
FEATURES = [c for c in head.columns if c != LABEL]
print("Label column:", LABEL, "| features:", len(FEATURES))

# 8-class grouping following the 7 attack families in Neto et al. (2023) + benign.
# Rules match on label text; Cell 3 prints the mapping so you can CHECK every label by eye.
RULES = [
    (r"^benign", "Benign"),
    (r"^ddos", "DDoS"),
    (r"^dos", "DoS"),
    (r"^mirai", "Mirai"),
    (r"^recon|vulnerabilityscan", "Recon"),
    (r"spoof", "Spoofing"),
    (r"bruteforce", "BruteForce"),
    (r"sqlinjection|commandinjection|xss|uploading|backdoor|browserhijack", "Web"),
]
def family(label):
    s = str(label).strip().lower()
    for pattern, fam in RULES:
        if re.search(pattern, s):
            return fam
    return "UNMAPPED"

# %% Cell 2 - hash every training row (features only); about 8 bytes per row in memory
CHUNK = 500_000
dtypes = {c: "float64" for c in FEATURES}
dtypes[LABEL] = "string"

def stream(path):
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=CHUNK):
        h = pd.util.hash_pandas_object(chunk[FEATURES], index=False).to_numpy()
        yield h, chunk[LABEL].to_numpy(dtype=object)

train_h, train_y = [], []
for h, y in stream(FILES["train"]):
    train_h.append(h)
    train_y.append(pd.Series(y).astype("category"))
train_h = np.concatenate(train_h)
train_unique = np.unique(train_h)
print(f"train rows {len(train_h):,} | distinct feature vectors {len(train_unique):,}")

# Within-train duplicate share per label (rows beyond the first copy of a vector)
tr = pd.DataFrame({"h": train_h, "label": pd.concat(train_y, ignore_index=True).astype(str)})
del train_y
dup_tr = tr.groupby("label").agg(train_rows=("h", "size"), train_distinct=("h", "nunique"))
dup_tr["train_within_dup_%"] = 100 * (1 - dup_tr["train_distinct"] / dup_tr["train_rows"])
del tr

# %% Cell 3 - for test and validation: share of rows per label whose features also occur in train
results = []
for split in ("test", "validation"):
    parts = []
    for h, y in stream(FILES[split]):
        seen = np.isin(h, train_unique)
        parts.append(pd.DataFrame({"label": y, "seen": seen}))
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby("label")["seen"].agg(rows="size", rows_with_train_copy="sum")
    g.columns = [f"{split}_rows", f"{split}_rows_with_train_copy"]
    g[f"{split}_overlap_%"] = 100 * g[f"{split}_rows_with_train_copy"] / g[f"{split}_rows"]
    results.append(g)
    print(f"{split}: {int(df['seen'].sum()):,} of {len(df):,} rows "
          f"({100 * df['seen'].mean():.2f}%) have an exact feature copy in train")

by_label = pd.concat([dup_tr] + results, axis=1).fillna(0)
by_label.insert(0, "family", [family(l) for l in by_label.index])
by_label = by_label.sort_values(["family", "test_rows"], ascending=[True, False])
pd.set_option("display.width", 200)
print(by_label.round(2).to_string())
unmapped = by_label.index[by_label["family"] == "UNMAPPED"].tolist()
print("\nUNMAPPED labels (add a rule in Cell 1 if any):", unmapped)
by_label.to_csv(f"{OUT}/overlap_by_label.csv")

# %% Cell 4 - the same per family (8 classes)
count_cols = [c for c in by_label.columns if c.endswith("_rows") or c.endswith("_copy") or c == "train_distinct"]
fam = by_label.groupby("family")[count_cols].sum()
fam["train_within_dup_%"] = 100 * (1 - fam["train_distinct"] / fam["train_rows"])
for split in ("test", "validation"):
    fam[f"{split}_overlap_%"] = 100 * fam[f"{split}_rows_with_train_copy"] / fam[f"{split}_rows"]
fam = fam.sort_values("test_rows")
print(fam.round(2).to_string())
fam.to_csv(f"{OUT}/overlap_by_family.csv")
json.dump({"label_column": LABEL, "n_features": len(FEATURES), "rules": RULES,
           "unmapped": unmapped}, open(f"{OUT}/sq1_config.json", "w"), indent=1)
print("Saved:", os.listdir(OUT))
