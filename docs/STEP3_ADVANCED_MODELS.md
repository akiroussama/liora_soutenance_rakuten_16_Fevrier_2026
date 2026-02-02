# Step 3: Advanced Models - Results Summary

**Date**: January 2026
**Project**: Rakuten Product Classification
**Team**: DataScientest OCT25 BMLE

---

## Executive Summary

Step 3 explored advanced modeling approaches beyond baseline MLPs. The key finding is that **multimodal fusion (images + text)** dramatically improves classification accuracy.

| Approach | Best Val Accuracy | vs Baseline |
|----------|-------------------|-------------|
| Step 1 Baseline (MLP) | 60.36% | - |
| Step 2 Optimized (MLP) | 61.38% | +1.02% |
| Step 3 Ensemble (Soft Voting) | 63.68% | +5.5% |
| **Step 3 Multimodal (MLP)** | **75.75%** | **+25.5%** |

---

## 1. Gradient Boosting Models

### Approach
Applied XGBoost and LightGBM directly on EfficientNet-B0 features (1280 dimensions).

### Results

| Model | Train Acc | Val Acc | Val F1 | Training Time |
|-------|-----------|---------|--------|---------------|
| XGBoost | 91.78% | 59.14% | 58.04% | ~25 min |
| LightGBM | 99.47% | 60.22% | 59.26% | ~12 min |

### Key Findings
- Both models **underperform** compared to MLP baseline
- Severe overfitting (high train accuracy, lower validation)
- Gradient Boosting is less suited for high-dimensional dense CNN features
- Neural networks better capture relationships in embedding spaces

### Configuration
```python
# XGBoost
n_estimators=100, max_depth=6, learning_rate=0.1

# LightGBM
n_estimators=100, max_depth=8, learning_rate=0.1, num_leaves=63
```

---

## 2. Ensemble Models

### Approach
Combined MLP (optimized), XGBoost, and LightGBM using voting and averaging strategies.

### Results

| Method | Val Accuracy | Val F1 | vs Baseline |
|--------|-------------|--------|-------------|
| MLP (individual) | 61.38% | 60.91% | +1.02% |
| XGBoost (individual) | 59.14% | 58.04% | -1.22% |
| LightGBM (individual) | 60.22% | 59.26% | -0.15% |
| Hard Voting | 61.63% | 60.86% | +1.27% |
| **Soft Voting** | **63.68%** | **62.85%** | **+5.5%** |
| Weighted Voting | 63.67% | 62.83% | +5.5% |
| MLP-Heavy (0.6/0.2/0.2) | 63.33% | 62.63% | +4.9% |

### Key Findings
- **Soft voting** (average of probabilities) achieves best performance
- Even underperforming models contribute through ensemble diversity
- Improvement of +3.3% over best individual model (MLP)
- Hard voting provides smaller improvement than soft voting

---

## 3. Multimodal Fusion (Image + Text)

### Approach
Combined EfficientNet visual features with TF-IDF text features from product descriptions.

### Feature Extraction
- **Image**: EfficientNet-B0 (1280 dimensions, pre-extracted)
- **Text**: TF-IDF + SVD reduction (3000 vocab → 200 dimensions)
- **Combined**: Concatenation (1480 total features)

### Results

| Method | Val Accuracy | Val F1 | vs Baseline |
|--------|-------------|--------|-------------|
| MLP Image Only | 61.65% | 61.21% | +1.29% |
| **MLP Text Only** | **72.20%** | 71.74% | +11.84% |
| **MLP Multimodal** | **75.75%** | **75.59%** | **+25.5%** |
| Late Fusion (0.7 img + 0.3 txt) | 71.48% | 70.97% | +11.12% |

### Key Findings
- **TEXT FEATURES ALONE OUTPERFORM IMAGES** (72.20% vs 61.65%)
- Product descriptions contain rich semantic information
- Multimodal fusion is synergistic (75.75% > 72.20% + 61.65% combined)
- Early fusion (concatenation) outperforms late fusion (probability averaging)

### Text Processing
```python
TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=3, max_df=0.95,
    sublinear_tf=True
)
# SVD: 3000 → 200 dimensions (45% variance explained)
```

---

## Summary of All Results

| Rank | Model | Val Acc | Improvement |
|------|-------|---------|-------------|
| 1 | MLP Multimodal | **75.75%** | +25.5% |
| 2 | MLP Text Only | 72.20% | +19.6% |
| 3 | Late Fusion | 71.48% | +18.4% |
| 4 | Soft Voting Ensemble | 63.68% | +5.5% |
| 5 | MLP Optimized | 61.38% | +1.7% |
| 6 | MLP Baseline | 60.36% | - |
| 7 | LightGBM | 60.22% | -0.2% |
| 8 | XGBoost | 59.14% | -2.0% |

---

## Recommendations

1. **Use Multimodal Approach**: The combination of image and text features provides the best results (75.75% accuracy).

2. **Prioritize Text Features**: Text descriptions are surprisingly more informative than CNN features for product classification.

3. **Consider Text-Only Baseline**: For production, a text-only model (72.20%) may be sufficient and computationally cheaper.

4. **Ensemble as Fallback**: When text data is unavailable, ensemble of image-based models provides +5.5% improvement.

5. **Avoid Pure GB Models**: XGBoost/LightGBM alone don't improve over MLP on CNN features.

---

## Files Created

- `gradient_boosting.py` - XGBoost/LightGBM training
- `text_features.py` - TF-IDF and multimodal extractors
- `ensemble_models.py` - Ensemble building utilities
- `run_step3_quick.py` - Fast GB testing
- `run_step3_ensemble.py` - Ensemble experiments
- `run_step3_multimodal.py` - Multimodal fusion

## Models Saved

- `xgboost_quick.joblib` - XGBoost model
- `lightgbm_quick.joblib` - LightGBM model
- `mlp_multimodal.joblib` - Best multimodal model

---

## Next Steps

1. Fine-tune multimodal architecture (different text representations)
2. Explore Sentence Transformers for better text embeddings
3. Try attention-based fusion mechanisms
4. Create final evaluation on test set
5. Prepare presentation materials
