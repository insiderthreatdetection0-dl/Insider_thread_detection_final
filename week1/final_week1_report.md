# Week 1: Exploratory Data Analysis, Canonical Data Split, and Baseline Modeling

**Project:** Hierarchical Multi-Agent Insider Threat Detection System  
**Dataset:** Carnegie Mellon SEI CERT Insider Threat Dataset v4.2 (`dayr4.2.csv`)  
**Status:** Authoritative Milestone Report

---

## 1. Executive Summary

Week 1 established the foundational data processing and baseline evaluation for the insider threat detection project:
1. **Dataset Ingestion & EDA:** Verified integrity, structure, and extreme class imbalance of the CERT r4.2 dataset (`dayr4.2.csv`).
2. **Canonical 60/20/20 User-Level Stratified Split:** Formulated the leakage-free user-level split protocol across 1,000 unique users, strictly stratifying by malicious user status.
3. **Flat Baseline Classifier:** Implemented and evaluated a traditional flat Random Forest classifier on clean behavioral features to establish the baseline performance benchmark that subsequent hierarchical multi-agent architectures must beat.

---

## 2. Dataset Overview & Behavioral Modalities

The Carnegie Mellon SEI CERT r4.2 dataset consists of daily aggregated behavioral metrics representing 1,000 corporate employees over a ~1.5 year observation window (330,452 total user-days).

### 2.1 Class Distribution & Threat Scenarios

The dataset exhibits extreme class imbalance:
- **Normal Class (0):** 329,486 user-days (99.708%)
- **Malicious Insider Scenarios (1–3):** 966 user-days (0.292%) across **70 malicious users**
  - **Scenario 1 (WikiLeaks Uploader):** 85 user-days across 30 users. Characterized by anomalous HTTP uploads and cloud storage access after hours.
  - **Scenario 2 (Job-Hunter Data Exfiltrator):** 861 user-days across 30 users. Characterized by job search activity, mass email forwards, and removable media/file downloads.
  - **Scenario 3 (Keylogger Sysadmin):** 20 user-days across 10 users. Characterized by unauthorized admin login and software installation via USB.

### 2.2 Feature Engineering & Extraction
Features span five distinct categories:
- **Logon / Authentication:** Logon count, logout count, off-hours logons, weekend logons, multi-machine access.
- **Email Communications:** Internal/external sent count, attachment sizes, recipient counts, off-hours emailing.
- **File & Removable Device Activity:** USB connect/disconnect events, file read/write/copy counts, non-standard extension operations.
- **HTTP / Web Browsing:** Request counts, external downloads/uploads, job site visits, cloud storage interactions.
- **Psychometric OCEAN Scores:** Static Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism scores.

---

## 3. Official 60/20/20 User-Level Stratified Split

To eliminate data leakage, partitioning is performed strictly at the **user level** rather than the record level. Since user behavior exhibits temporal correlation across days, mixing records from the same user across splits would introduce severe optimistic bias.

### 3.1 Split Methodology (`week1_official_split.ipynb`)
1. User-level malicious labels: `user_labels = df.groupby('user')['insider'].max().astype(int)`.
2. Initial split (60% Train / 40% Temp) with `random_state=42`, stratified on `user_labels`.
3. Equal split of Temp (50% / 50%) → 20% Validation / 20% Test, stratified on remaining user labels.

### 3.2 Partition Breakdown

| Partition | Users | Malicious Users | Total User-Days | Positive (Malicious) User-Days |
|---|:---:|:---:|:---:|:---:|
| **Train (60%)** | 600 | 42 (70.0%) | 185,744 | 564 |
| **Validation (20%)** | 200 | 14 (70.0%) | 61,076 | 201 |
| **Held-Out Test (20%)** | 200 | 14 (70.0%) | 61,203 | 201 |
| **Total (100%)** | **1,000** | **70 (7.0%)** | **308,023\*** | **966** |

*\*Note: 308,023 user-days represent the clean preprocessed active modeling window.*

User sets are saved to `data/user_splits/` (`train_users.npy`, `val_users.npy`, `test_users.npy`) and are identically preserved in the percentile dataset (`data/user_split_formlp/`).

---

## 4. Flat Random Forest Baseline

### 4.1 Model Specification (`week1_random_forest_binary.ipynb`)
- **Classifier:** `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)`
- **Input Features:** Clean numeric features (`data/processed/features_clean.parquet`) + static OCEAN scores.
- **Training Protocol:** Trained strictly on the 600 Train users (185,744 rows).
- **Threshold Calibration:** Tuned on the 200 Validation users via Youden's J statistic ($J = \text{TPR} - \text{FPR}$) to determine optimal decision threshold ($\tau = 0.05$).
- **Evaluation:** Evaluated exactly once on the 200 held-out Test users (61,203 rows).

### 4.2 Baseline Results on Held-Out Test Set

| Metric | Official 60/20/20 Held-Out Test | Legacy 80/20 Split (Historical Reference) |
|---|:---:|:---:|
| **Recall (TPR)** | **0.6418 (129 / 201)** | 0.1875 |
| **Precision** | **0.2074 (129 / 622)** | 0.8571 |
| **F1 Score** | **0.3135** | 0.3077 |
| **False Positive Rate (FPR)** | **0.0081 (493 / 61,002)** | 0.0001 |
| **ROC-AUC** | **0.9416** | 0.9639 |
| **PR-AUC** | **0.3541** | 0.3850 |

---

## 5. Key Findings & Motivation for Hierarchical Multi-Agent Architecture

1. **Class Imbalance & Flat Feature Limitations:** While the Random Forest baseline achieved ~64% recall on the held-out test split, it missed 72 malicious days (35.8% false negative rate) and flagged nearly 500 normal user-days as false alarms.
2. **Loss of Temporal & Modality Specificity:** Treating all behavioral features in a single flat vector obscures subtle, domain-specific anomalies (e.g., subtle logon shifts or single-day thumb-drive transfers).
3. **Requirement for Modality-Specific Deep Agents:** These baseline findings directly motivated the Week 2 research direction: developing independent deep neural agents per modality (Logon, Email, File/Device, HTTP) with specialized balancing and feature representations.

---

## 6. Artifact & Notebook Inventory

- `week1/week1_eda.ipynb` — Comprehensive Exploratory Data Analysis, distribution checks, and feature preparation.
- `week1/week1_official_split.ipynb` — Deterministic generation of the official 60/20/20 user-level split (`data/user_splits/*.npy`).
- `week1/week1_random_forest_binary.ipynb` — Canonical Random Forest baseline training, validation calibration, and held-out test evaluation.
- `week1/week1_random_forest_fixed.ipynb` / `week1_random_forest_model_only.ipynb` — Historical exploratory baseline variations.
- `week1/final_week1_report.md` — Authoritative Milestone 1 Report.
