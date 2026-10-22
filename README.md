# Rare-family reliability on CICIoT2023

Code for the paper "How Reliable Are Rare-Family Results? A Repeated-Split Evaluation of Standard Classifiers for IoT Intrusion Detection on CICIoT2023".

The study compares Logistic Regression, Decision Tree, Random Forest and XGBoost on the 8-class family task of CICIoT2023. It removes exact duplicate rows, repeats the experiment over 5 stratified 70/15/15 splits, and reports per-family precision, recall and F1 with bootstrap confidence intervals and paired model comparisons.

## Data

Kaggle copy of CICIoT2023: https://www.kaggle.com/datasets/himadri07/ciciot2023 (7,845,673 rows, 46 features, 34 labels).
Original dataset: Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment", Sensors 23(13):5941, 2023, https://doi.org/10.3390/s23135941.
The data is not included in this repository.

## Scripts (run in this order)

| Script | What it does |
|---|---|
| `code/01_duplicate_overlap_audit.py` | Counts test rows that have an exact copy in the training file, per family |
| `code/02_xgb_seen_vs_unseen.py` | One XGBoost on the supplied split; scores test rows with and without a training copy separately |
| `code/03_main_experiment.py` | Main experiment: de-duplication, 5 splits, 4 models, bootstrap CIs, paired comparisons |

Each script is written as a Kaggle notebook, one `# %%` block per cell. Add the dataset as notebook input and run on a CPU session. For the full main experiment use "Save Version → Save & Run All (Commit)". Outputs are written to `/kaggle/working/`.

## Environment used

Kaggle CPU notebook, 4 cores, Python 3.13.15. Package versions are in `requirements.txt`.

## Licence

[to be chosen, e.g. DPU]
