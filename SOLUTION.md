# Solution: Hallucination Detection via Hidden State Probing

## Reproducibility Instructions

### Environment
- Python 3.12
- CUDA-capable GPU (tested on single GPU)
- Dependencies: `torch`, `transformers`, `sklearn`, `numpy`, `pandas`, `tqdm`

### Commands
Same as baseline
```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

### Output
- `results.json` — evaluation metrics (5-fold cross-validation)
- `predictions.csv` — 100 predictions for test set

---

## Final Solution Description

### Modified Components

| File | Changes |
|------|---------|
| `aggregation.py` | Complete rewrite: multi-layer last-token + geometric features |
| `probe.py` | Replaced MLP with multi-classifier auto-selection |
| `splitting.py` | 5-fold stratified CV with validation split |

### Approach

**Feature Extraction (aggregation.py):**
- Extract last-token hidden state from layers 23, 24, 25 (3 × 896 = 2688 dims)
- Compute 171 geometric features across all 25 layers:
  - Layer norms (25)
  - Inter-layer cosine similarities (24)
  - Norm ratios between consecutive layers (24)
  - L2 distances between consecutive layers (24)
  - Skip-connection cosines to first layer (24)
  - Skip-connection cosines to last layer (24)
  - Cosine between last-token and mean-of-tokens at 5 key layers (5)
  - Token norm statistics (mean, std, range) at 5 key layers (15)
  - Sequence length + summary statistics (6)
- Total: **2859 features** per sample

**Classification (probe.py):**
- StandardScaler was switched to a PCA (100 components) for dimensionality reduction
- Auto-selection from 8 classifier candidates via 5-fold CV on AUROC:
  - LogisticRegression (C in {0.01, 0.1, 1.0, 10.0}, balanced class weights)
  - SVM with RBF kernel (C in {1.0, 10.0}, balanced)
  - GradientBoosting (50 trees depth=3, 100 trees depth=2)
- Best classifier is retrained on full training set
- Threshold tuned on validation set to maximize F1

**Evaluation (splitting.py):**
- 5-fold Stratified KFold (shuffle, seed=42)
- 18% of train+val used as validation per fold

### Why These Choices

1. **Last-token from multiple layers** --- captures information at different abstraction levels; hallucinated responses may show distinct layer-wise patterns
2. **Geometric features** --- low-dimensional but highly informative; capture how representations evolve across layers (hallucinations may cause different "trajectories")
3. **PCA reduction** --- essential with 689 samples and 2859 features to prevent overfitting
4. **Multi-classifier selection** --- different folds may benefit from different inductive biases; SVM/GB can capture non-linearities that LogReg misses
5. **No neural network** --- 450 training samples is far too few for MLP to generalize reliably

---

## Experiments and Failed Attempts

| Attempt | Test AUROC | Why Failed/Discarded |
|---------|-----------|---------------------|
| MLP (2 hidden layers, dropout) | 63.7% | Severe overfitting on 450 samples (train AUROC ~1.0) |
| PCA→64 + small MLP | 68.2% | Still overfitting, unstable across folds |
| Mean-pooling + last-token (7168 features) | 66.1% | Too many features, overfitting returned |
| 4 layers + diff + mean (5604 features) | 68.9% | More features ≠ better; LogReg failed to converge |
| Ensemble LogReg + MLP | 66.1% | MLP component dragged ensemble down ||
| LogRegCV alone (2859 features) | 69.3% | Good baseline but single model limits ceiling |

### Key Takeaways
- Adding more raw hidden dimensions beyond 3 layers hurts (curse of dimensionality)
- Geometric features (norms, cosines, distances) are the most informative per-dimension
- Classical ML (LogReg, SVM, GB) vastly outperforms neural probes at this sample size
- Auto-selection of classifier per fold adds robustness (+2.4% AUROC over fixed LogReg)

---

## Final Metric

**Test AUROC: 71.74%** (averaged over 5 folds, up from 63.7% baseline MLP)

## prediction.csv

Link to a prediction.csv file [here](https://disk.yandex.ru/d/MdbVefATUMnDBw) 
