# DoubleMuon2016G Trigger Emulation (CMS Open Data) — End-to-End ML Pipeline

An end-to-end, **CERN-style** data engineering + ML project that builds an explainable “trigger emulator” on real CMS Run‑2 collision data: ROOT NanoAOD → certified filtering → feature engineering → Parquet dataset → leakage-safe multi-label modeling → explainability + drift monitoring.

> **Status:** Completed Phases 1–4 (EDA → Parquet builder → baseline modeling → event-level multi-label robustness package).  
> **Primary objective:** Predict whether selected CMS HLT muon triggers fired in an event using offline-reconstructed observables, with rigorous evaluation and stability checks.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset (CERN Open Data)](#dataset-cern-open-data)
- [Kaggle Assets](#kaggle-assets)
- [Repository Structure](#repository-structure)
- [Methods (Phases 1–4)](#methods-phases-1–4)
- [Dataset Schema (Parquet)](#dataset-schema-parquet)
- [Modeling & Evaluation](#modeling--evaluation)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Limitations & Next Steps](#limitations--next-steps)
- [Citations & Disclaimer](#citations--disclaimer)

---

## Project Overview

### Why this project?
CERN experiments record proton-proton collisions at extremely high rates. The **High-Level Trigger (HLT)** selects which events to keep for physics analyses. This project demonstrates:
- **IT/Data engineering:** converting large ROOT datasets into compact, analysis-ready Parquet
- **Mathematics/ML:** interpretable binary classification + robust metrics for imbalanced labels
- **Robustness/operations thinking:** drift checks across run segments, calibration, thresholding

### Core deliverable
A reproducible pipeline that produces:
1. A certified Parquet dataset with physics-driven features and HLT labels
2. Leakage-safe training and evaluation (grouped by collision event)
3. Explainability (feature importance + SHAP)
4. Drift/stability monitoring plots
5. Saved models and metrics artifacts

---

## Dataset (CERN Open Data)

### Primary dataset (real collision data)
- **CERN Open Data record:** https://opendata.cern.ch/record/30522  
- **Dataset name:** `/DoubleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD`  
- **DOI:** 10.7483/OPENDATA.CMS.ZQS3.LGLP  
- **Collision energy:** 13 TeV pp (Run‑2)  
- **Run range:** 278820–280385  
- **Dataset characteristics:** 45,235,604 events, 29 files, 39.3 GiB  
- **Key note:** CERN Open Data requires applying the **validated run/lumi JSON** (“full validation” or “muons only”) for physics-quality analyses.

From the record:
- NanoAOD is ROOT tree format and can be analyzed with common ROOT/Python tools
- Events are selected by the presence of **at least two energetic muons**
- Links are provided for validated runs and HLT configuration

---

## Kaggle Assets

### 1) Local ROOT files (converted subset for fast iteration)
- Kaggle dataset: https://www.kaggle.com/datasets/katakuricharlotte/doublemuon2016g-rootfiles  
- Contents: `root_converted/` directory with 5 ROOT files:
  - `doublemuon2016g_0.root` (2.16 GB)
  - `doublemuon2016g_1.root` (2.28 GB)
  - `doublemuon2016g_2.root` (2.27 GB)
  - `doublemuon2016g_3.root` (1.02 GB)
  - `doublemuon2016g_4.root` (2.16 GB)
- Total: ~9.89 GB

> These files are used to run EDA and the Parquet-building pipeline inside Kaggle without needing to stream via XRootD.

### 2) Parquet dataset for ML (feature table + certification JSON)
- Kaggle dataset: https://www.kaggle.com/datasets/katakuricharlotte/parquet-triggeremu  
- Contains:
  - `parquet_dimuon/` Parquet shard(s)
  - `validated_runs_muons_2016.txt` (JSON mapping of certified run/lumi ranges)

---

## Repository Structure

Recommended structure for GitHub:



---

## Methods (Phases 1–4)

### Phase 1 — EDA (ROOT → understanding)
- Opened local ROOT files with `uproot`
- Inspected `Events` tree and branch inventory (including many `HLT_*` branches)
- Checked:
  - event keys: `run`, `lumi`, `event`
  - muon-related observables
  - dimuon invariant mass sanity check (OS dimuon pairing)
  - trigger inventory + fired fractions for candidate labels

**Outcome:** Verified the dataset is usable for muon/dimuon ML tasks, and selected target HLT labels.

### Phase 2 — Feature engineering + Parquet build
- Applied **validated run/lumi filtering** using `validated_runs_muons_2016.txt`
- Constructed **opposite-sign dimuon pairs**
- Built a dimuon-level table with compact, explainable features
- Added event-level HLT labels to each row
- Wrote Parquet shards to `parquet_dimuon/`

**Outcome:** Produced an ML-ready dataset that can be trained quickly and shared as a Kaggle dataset.

### Phase 3 — Baseline model + explainability + drift
- Trained LightGBM baseline on the Parquet table
- Avoided leakage by splitting using `GroupKFold` grouped by `event_id`
- Reported ROC-AUC and Average Precision (AP)
- Produced:
  - feature importances
  - SHAP summary (optional)
  - drift plots vs run bins

**Outcome:** Baseline performance validated and engineering workflow established.

### Phase 4 — Event-level evaluation + multi-label + robustness
- Converted pair-level predictions to **event-level**:
  - event score = `max(score)` across pairs in an event
  - event label = `max(y)` (safe since HLT label is event-level)
- Trained **one model per HLT label** using the same event-group folds
- Added:
  - calibration curves + Brier score
  - threshold selection
  - drift monitoring
- Saved artifacts:
  - `metrics_per_label.csv`
  - `event_metrics_oof.csv`
  - `thresholds.csv`
  - per-label models (`.joblib`)
  - plots (`.png`)
  - `config.json`

---

## Dataset Schema (Parquet)

The Parquet table is **dimuon-pair level** (one row per OS dimuon candidate).  
Key columns:

### Event keys
- `run` — run number
- `lumi` — luminosity block
- `event` — event number
- `event_id` — constructed as `run:lumi:event` for grouped evaluation (Phase 3/4)

### Features (used in Phase 3/4)
- `m_mumu` — invariant mass of OS muon pair
- `pt_mumu` — dimuon transverse momentum
- `eta_mumu` — dimuon pseudorapidity
- `dR_mumu` — ΔR separation between muons
- `PV_npvs` — primary vertex multiplicity (pileup proxy)
- `MET_pt` — missing transverse momentum (context)

### Labels (HLT triggers)
Trained labels in this project:
- `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ`
- `HLT_Mu17_Mu8`
- `HLT_Mu17_Mu8_DZ`

> Note: Labels are **event-level** triggers repeated on each dimuon row of the event. This is why grouped splitting and event-level aggregation are mandatory.

---

## Modeling & Evaluation

### Leakage prevention
Because multiple rows belong to the same collision event, random row splits would leak information.  
This project uses `GroupKFold` with groups = `event_id` to ensure **no event appears in both train and validation**.

### Metrics
- **ROC-AUC**: general ranking quality
- **Average Precision (AP)**: preferred for imbalanced labels
- **Brier score**: calibration / probability quality

### Event-level scoring (Phase 4)
- Pair → event aggregation:
  - event score = max pair score
  - event label = max pair label (safe for repeated event label)

---

## Results

### Dataset size used in Phase 4 run
- Pair-level rows: **1,378,545**
- Unique events: **1,171,442**
- Feature columns used for modeling: 6 (physics + context)
- Labels: 3

### Label prevalence (approx, from project outputs)
- `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ`: ~45%
- `HLT_Mu17_Mu8`: ~2.3%
- `HLT_Mu17_Mu8_DZ`: ~2.0%

### Event-level OOF metrics (Phase 4)
From `event_metrics_oof.csv`:
- `HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ`:
  - ROC-AUC ≈ **0.854**
  - AP ≈ **0.821**
- `HLT_Mu17_Mu8`:
  - ROC-AUC ≈ **0.642**
  - AP ≈ **0.041**
- `HLT_Mu17_Mu8_DZ`:
  - ROC-AUC ≈ **0.656**
  - AP ≈ **0.037**

**Interpretation:**
- The high-statistics “TrkIsoVVL DZ” label is modeled well with compact, explainable dimuon/event features.
- The rarer labels are more challenging (low AP is expected at ~2% positives), and benefit from more trigger-proximal features and/or tighter operating-point selection.

### Artifacts produced
Saved to `/kaggle/working/phase4_artifacts/`:
- `metrics_per_label.csv`, `event_metrics_oof.csv`, `thresholds.csv`, `config.json`
- Models: `phase4_artifacts/models/*.joblib`
- Plots: `phase4_artifacts/plots/*.png` (ROC/PR/Calibration/Drift/SHAP)

---

## Reproducibility

### Environment (Kaggle)
Primary Python stack used across notebooks:
- `uproot`, `awkward`, `vector` for ROOT/NanoAOD reading and physics-vector operations
- `pandas`, `pyarrow` for Parquet + tabular processing
- `lightgbm`, `scikit-learn` for modeling and metrics
- `shap` for explainability
- `matplotlib`, `seaborn` for plotting

### How to reproduce (high level)
1. Run Phase 1 EDA on the ROOT subset dataset  
2. Run Phase 2 builder to produce Parquet + validated JSON outputs  
3. Run Phase 3 baseline modeling (grouped evaluation)  
4. Run Phase 4 multi-label robustness pipeline (event-level metrics, plots, artifacts export)

---

## Limitations & Next Steps

### Current limitations
- Training features are primarily dimuon-level aggregates; some trigger paths may require more detailed muon-level features (leading/subleading muon pT, isolation, ID flags).
- Thresholding strategy for rare labels should use rate/acceptance constraints or precision targets (more trigger-like), not only recall targets.

### Recommended next improvements
- Build an **event-level feature table** (top-2 muons + event context) to align perfectly with event-level HLT labels.
- Add rate/efficiency tradeoff curves and fixed acceptance operating points.
- Extend to additional HLT paths listed in the CERN Open Data record.
- Run the pipeline on a larger fraction of the 29-file dataset (or stream via XRootD) for closer-to-production scale.

---

## Citations & Disclaimer

### Citation
If you use these data, cite:
- CMS Collaboration (2024). DoubleMuon primary dataset in NANOAOD format from RunG of 2016. CERN Open Data Portal. DOI: **10.7483/OPENDATA.CMS.ZQS3.LGLP**  
Record: https://opendata.cern.ch/record/30522

### License / disclaimer
- CERN Open Data Portal releases these data under **CC0** and notes that neither CMS nor CERN endorses any work produced using the data.  
- This repository and Kaggle artifacts are for educational/research demonstration and are not official CMS/CERN software.

---

## Links
- CERN Open Data (DoubleMuon Run2016G NanoAOD): https://opendata.cern.ch/record/30522  
- Kaggle ROOT subset: https://www.kaggle.com/datasets/katakuricharlotte/doublemuon2016g-rootfiles  
- Kaggle Parquet dataset: https://www.kaggle.com/datasets/katakuricharlotte/parquet-triggeremu
