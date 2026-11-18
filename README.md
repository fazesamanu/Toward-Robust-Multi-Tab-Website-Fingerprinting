# Toward Robust Multi-Tab Website Fingerprinting (ARES / Trans-WF)

Complete PyTorch implementation and experimental reproduction of **"Toward Robust Multi-Tab Website Fingerprinting"** by Xinhao Deng, Xiyuan Zhao, Qilei Yin, Zhuotao Liu, Qi Li, Mingwei Xu, Ke Xu, and Jianping Wu (*IEEE Transactions on Networking*, Vol. 34, pp. 3656–3671, 2026).

---

## 1. Architecture Overview

**ARES** formulates multi-tab Website Fingerprinting (WF) as a **multi-label classification problem** without requiring prior knowledge of the number of tabs opened by the client. It uses a single parameter-shared **Trans-WF** backbone equipped with $N$ site-specific linear heads (One-vs-All binary cross-entropy loss), comprising three core modules:

1. **Multi-Level Traffic Aggregation (`MTAF`)** ([`WFlib/tools/data_processor.py`](WFlib/tools/data_processor.py)):
   - Partitions an obfuscated traffic trace into fixed-length sub-segments of uniform time interval $t = 20\text{ ms}$ (up to $d = 8000$ segments).
   - Separately extracts **8 bidirectional features** per segment for outgoing ($+1$) and incoming ($-1$) traffic:
     - **Packet-level features**: Packet count (2 channels) and inter-packet interval statistics (2 channels).
     - **Burst-level features**: Burst count (2 channels) and average burst size (2 channels).
2. **CNN-Based Local Profiling (`LocalProfiling`)** ([`WFlib/models/ARES.py`](WFlib/models/ARES.py)):
   - Consists of $L = 4$ residual 1D convolutional blocks (`Conv1d` with kernel size $7$, `BatchNorm1d`, `ReLU`, skip connection, `MaxPool1d` with pool size $8$ and stride $4$, and `Dropout(0.1)`), projecting 8-channel MTAF segments into $256$-dimensional translation-invariant local pattern embeddings.
3. **Website Identification (`TopM_MHSA`)** ([`WFlib/models/ARES.py`](WFlib/models/ARES.py)):
   - Stacks $n = 4$ Transformer encoder layers equipped with **Multi-Head Top-$m$ Self-Attention** ($m = 20$) and `DropPath` regularization:
     $$\mathrm{Attention}^{\mathrm{Top-}m}(Q, K, V) = \mathrm{softmax}\!\left(\Gamma\!\left(\frac{QK^T}{\sqrt{d}}\right)\right)V$$
     where $\Gamma(\cdot)$ retains only the top-$m$ largest attention weights per query row and masks out noisy segments caused by concurrent tabs and WF defenses.

---

## 2. Repository Structure

```text
├── WFlib/
│   ├── models/
│   │   ├── ARES.py          # ARES (Trans-WF) model, Top-m MHSA, LocalProfiling, ablations & incremental head expansion
│   │   ├── TMWF.py          # Transformer-based Multi-Tab WF baseline (CCS '23)
│   │   ├── BAPM.py          # Block Attention Profiling Model baseline (ACSAC '21)
│   │   ├── MultiTabRF.py    # Robust Fingerprinting (RF) adapted for multi-tab WF (USENIX Security '23)
│   │   ├── VarCNN.py        # Dilated ResNet dual-encoder baseline (PETS '19)
│   │   ├── TikTok.py        # Directional timing CNN baseline (PETS '20)
│   │   ├── DF.py            # Deep Fingerprinting CNN baseline (CCS '18)
│   │   └── NetCLR.py        # Contrastive learning WF baseline (CCS '23)
│   └── tools/
│       ├── data_processor.py # Vectorized MTAF, TAF, TAM, DT, DT2, DIR feature extraction
│       ├── defenses.py       # Random, WTF-PAD, Front, RegulaTor simulators, overlap mixer & Mutual Info analysis
│       ├── evaluator.py      # Multi-label metrics: AUC, P@K, MAP@K, AUC_M, AUC_N, PR curves
│       └── model_utils.py    # Training & evaluation loops
├── exp/                     # Standard WFlib CLI scripts (train.py, test.py, dataset_process/)
├── scripts/                 # Shell scripts & Zenodo dataset downloader (download_zenodo_datasets.py)
├── reproduce_paper.py       # Master end-to-end GPU reproduction script for all tables & figures
├── plot_paper_figures.py    # Publication-quality figure generator (Figures 3, 7, 8, 9, 10)
└── results/
    ├── reproduction_results.json
    └── figures/             # Generated figures (fig3, fig7, fig8, fig9, fig10)
```

---

## 3. Quick Start

### Run Complete Reproduction Suite (Tables II–IX & Figures 3, 7–10)
```bash
python reproduce_paper.py
```

### Download Full Official Datasets from Zenodo (Optional, ~1.1–2.1 GB per archive)
```bash
python scripts/download_zenodo_datasets.py --dataset Closed_2tab --out_dir ./datasets
```
