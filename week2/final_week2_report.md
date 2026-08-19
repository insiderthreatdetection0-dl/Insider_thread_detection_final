# Week 2: Deep Learning Behavioral Modality Agents & Architecture Selection

**Project:** Hierarchical Multi-Agent Insider Threat Detection System  
**Dataset:** Carnegie Mellon SEI CERT Insider Threat Dataset v4.2  
**Status:** Authoritative Milestone Report

---

## 1. Executive Summary

Week 2 focused on developing independent, deep learning-based behavioral agents for each of the four core modalities: **Logon**, **Email**, **File/Device**, and **HTTP**.

We explored multiple model families and balancing strategies across two major experimental phases:
1. **Unsupervised Reconstruction Baselines:** Standard Autoencoders (AE), Sparse Autoencoders (SAE), and Variational Autoencoders (VAE).
2. **Temporal Recurrent Architectures:** Bidirectional LSTM (BiLSTM) and Unidirectional LSTM (UniLSTM) utilizing 7-day sliding windows.
3. **Imbalance Mitigation & Failure Analysis:** Identified temporal dilution and oversampling memorization pitfalls; resolved through binary target reformulation and structural RandomUnderSampling.
4. **Feedforward Neural Networks on Percentile Features:** Modality Feedforward Neural Networks (`ModalityFFNN`) trained on 30-day temporal percentile representations.
5. **Final Tier-1 Agent Selection:** Selected four specialized MLP agents (HTTP 6:1, File/Device 4:1, Email 4:1, Logon 4:1) as the canonical Tier-1 layer.

---

## 2. Experimental Phase 1: Unsupervised & Recurrent Experiments

### 2.1 Early Failure Modes & Lessons Learned (`week2/experiments/`)

| Iteration | Modality / Setup | Strategy | Result | Root Cause |
|---|---|---|---|---|
| **Logon Exp 1** | UniLSTM (7-day window) | Class-Weighted Cross-Entropy | Recall = 0% on threat classes | Severe temporal dilution: 1 anomalous day buried in 6 normal days. |
| **Logon Exp 2** | UniLSTM (7-day window) | `WeightedRandomSampler` | Recall = 33% (Scenario 1), 0% (Scenario 2), high FPR | Re-sampling normal days repeatedly caused high false alarm rates. |
| **Logon Exp 3** | UniLSTM (7-day window) | Capacity increase (hidden 64 $\rightarrow$ 128) | No recall improvement | Model capacity was not the bottleneck; temporal window representation was. |
| **Email Exp 1** | UniLSTM (7-day window) | Raw Imbalanced Training | 99.6% Accuracy, F1 = 0 everywhere | "Lazy model" problem: predicting all 0s minimizes loss. |
| **Email Exp 2** | UniLSTM (7-day window) | `WeightedRandomSampler` | Predicted Normal for 100% test insiders | Memorization of synthetic oversampled sequences. |

### 2.2 Key Architectural Solutions
1. **Binary Target Reformulation:** Collapsed multi-class insider scenarios into a single binary anomaly signal ($y \in \{0, 1\}$).
2. **Structural Random Under-Sampling (`RandomUnderSampler`):** Under-sampled majority normal days at controlled ratios ($1:1$, $4:1$, $6:1$) after feature extraction, eliminating the lazy model failure mode.
3. **Numerically Stable Loss:** Standardized on `BCEWithLogitsLoss` with sigmoid probability outputs for downstream calibration.

### 2.3 Temporal Recurrent Models (BiLSTM & UniLSTM) (`week2/bilstm/`, `week2/unilstm/`)
- **Setup:** 7-day sliding window sequences ($X \in \mathbb{R}^{7 \times d}$), stride = 1.
- **BiLSTM:** `bidirectional=True`, hidden_dim=64, forward+backward concatenation $\rightarrow \text{Linear}(128, 1)$.
- **UniLSTM:** Forward LSTM, hidden_dim=64 $\rightarrow \text{Linear}(64, 1)$.

### 2.4 Unsupervised Reconstruction Baselines (`week2/autoencoder/`)
- **Architectures Evaluated:** Autoencoder, Sparse Autoencoder (KL divergence penalty on activations), and Variational Autoencoder (VAE with ELBO loss).
- **Hypothesis:** Normal employee behavior exhibits tight clustering; malicious days produce elevated reconstruction error.
- **Findings:** While reconstruction error separated coarse outliers, it lacked specificity for stealthy exfiltration (Scenario 2) and produced high false positive rates (>15% FPR at 80% recall). Reconstruction error alone proved insufficient as a primary detection mechanism.

---


## 3. Experimental Phase 2: Solutions & Percentile MLP Architecture

### 3.1 Percentile-30 Feature Representation & `ModalityFFNN` (`week2/mlp/`)
Instead of raw daily aggregates or rigid sliding windows, features were transformed into 30-day rolling percentile representations, capturing how an employee's activity on day $t$ deviates from their historical baseline distribution.

**`ModalityFFNN` Architecture:**
```text
Input (Modality Features + 5 OCEAN Scores)
   │
   ▼
Linear(input_dim, 64) ──► ReLU ──► Dropout(0.2)
   │
   ▼
Linear(64, 32) ──► ReLU
   │
   ▼
Linear(32, 1)  (Logits)
```

- **Optimizer:** Adam ($\text{lr} = 0.001$), Batch Size = 128, 20 Epochs, Seed = 42.
- **Preprocessing:** `StandardScaler` fitted strictly on the 60% Train user split.

---

## 4. Final Selected Tier-1 Agent Configuration

Comparing all architectures on validation performance and downstream stability, the four specialized `ModalityFFNN` MLP agents emerged as superior across all modalities:

| Modality Channel | Architecture | Feature Set | Sampling Ratio | Input Dim | Validation Role |
|---|:---:|:---:|:---:|:---:|---|
| **HTTP** | `ModalityFFNN` | Percentile-30 | **6:1** | 204 | Primary high-precision detector for web upload & job browsing. |
| **File / Device** | `ModalityFFNN` | Percentile-30 | **4:1** | 215 | Critical detector for USB exfiltration & file operations. |
| **Email** | `ModalityFFNN` | Percentile-30 | **4:1** | 47 | Exfiltration via attachment & external mail communications. |
| **Logon** | `ModalityFFNN` | Percentile-30 | **4:1** | 20 | Authentication anomaly & off-hours logon detection. |

---

## 5. Artifact & Notebook Inventory

- `week2/mlp/mlp_agents_percentile_1to1.ipynb` — MLP agent training with 1:1 undersampling.
- `week2/mlp/mlp_agents_percentile_4to1.ipynb` — Canonical MLP agent training with 4:1 undersampling.
- `week2/mlp/mlp_agents_percentile_6to1.ipynb` — Canonical HTTP MLP agent training with 6:1 undersampling.
- `week2/bilstm/bilstm_agents.ipynb` / `bilstm_agents_imbalanced_4_to_1.ipynb` / `bilstm_agents_improved.ipynb` — BiLSTM temporal experiments.
- `week2/unilstm/unilstm_agents.ipynb` / `logon_unilstm_agent.ipynb` — Historical UniLSTM temporal experiments.
- `week2/autoencoder/autoencoder_agents.ipynb` / `sparse_autoencoder_agents.ipynb` / `vae_agents.ipynb` — Unsupervised reconstruction baselines.
- `week2/experiments/logon_experiment_1..3.ipynb` / `email_experiment_1..2.ipynb` — Early balancing and architecture iterations.
- `week2/final_week2_report.md` — Authoritative Milestone 2 Report.
