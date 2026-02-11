# Rakuten Product Classifier

Multimodal e-commerce product classification into 27 categories using text (TF-IDF + LinearSVC) and image (Voting System: DINOv3 + EfficientNet + XGBoost) pipelines, fused via late fusion (~94% accuracy).

**Team**: Johan Frachon, Liviu Andronic, Hery M. Ralaimanantsoa, Oussama Akir
**Program**: Machine Learning Engineer — DataScientest (Oct 2025)

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd OCT25_BMLE_RAKUTEN
pip install -r requirements.txt
```

### 2. Download Models from Google Drive

Models are too large for Git (~2.5 GB total). Download them from:

**[Google Drive — Models](https://drive.google.com/drive/folders/1faJtjk_fsNNw4DCiPiz67-lYivyGjnKB?usp=sharing)**

Place the following files in the `models/` directory:

| File | Size | Description |
|------|------|-------------|
| `M1_IMAGE_DeepLearning_DINOv3.pth` | 1.2 GB | DINOv3 ViT-Large vision transformer (w=4/7 in voting) |
| `M2_IMAGE_Classic_XGBoost.json` | ~50 MB | XGBoost classifier on ResNet50 features (w=1/7) |
| `M2_IMAGE_XGBoost_Encoder.pkl` | <1 KB | Label encoder for XGBoost class mapping |
| `M3_IMAGE_Classic_EfficientNetB0.pth` | 16 MB | EfficientNet-B0 fine-tuned classifier (w=2/7) |
| `text_classifier.joblib` | 32 MB | TF-IDF + LinearSVC text pipeline |
| `category_mapping.json` | 4 KB | 27-category code-to-name mapping |

### 3. Launch the App

```bash
streamlit run src/streamlit/app.py
```

The app runs in **demo mode** (mock data) if CSV datasets are not present, and in **real mode** if models are downloaded. Both modes are fully functional.

---

## Project Structure

```
.
├── models/                     # Trained model weights (download from Drive)
├── notebooks/                  # Jupyter notebooks (exploration, training, benchmarks)
├── reports/                    # Final deliverables (HTML report + presentation)
├── src/
│   ├── features/               # Image preprocessing for inference
│   ├── models/                 # VotingPredictor — ensemble inference engine
│   └── streamlit/              # Streamlit multi-page application
│       ├── app.py              # Home page & app entry point
│       ├── config.py           # All paths, weights, and app settings
│       ├── pages/              # 5 interactive pages
│       ├── utils/              # Business logic (classifiers, data loading, UI)
│       ├── assets/             # CSS + model comparison charts
│       └── tests/              # Unit, integration, ML, and security tests
├── requirements.txt            # Python dependencies
└── archive/                    # Archived files (not tracked by git)
```

---

## File Reference

### `models/` — Trained Weights

| File | Role |
|------|------|
| `M1_IMAGE_DeepLearning_DINOv3.pth` | DINOv3 ViT-Large — best single image model (91.4%) |
| `M2_IMAGE_Classic_XGBoost.json` | XGBoost on ResNet50 features — diversity in voting (76.5%) |
| `M2_IMAGE_XGBoost_Encoder.pkl` | LabelEncoder to map XGBoost indices back to Rakuten codes |
| `M3_IMAGE_Classic_EfficientNetB0.pth` | EfficientNet-B0 — lightweight CNN for ensemble (~75%) |
| `text_classifier.joblib` | Sklearn Pipeline: TF-IDF FeatureUnion + LinearSVC (83%) |
| `category_mapping.json` | Maps 27 Rakuten product type codes to human-readable names |

### `src/features/` — Feature Engineering

| File | Role |
|------|------|
| `build_features.py` | `preprocess_image()` — resize, normalize, and prepare tensors for DINOv3 (518px) or standard models (224px) |

### `src/models/` — Inference Engine

| File | Role |
|------|------|
| `predict_model.py` | `VotingPredictor` — loads all 3 image models, runs inference, applies probability sharpening (p^3) on XGBoost, weighted vote (4:2:1), returns top-5 predictions |

### `src/streamlit/` — Application

| File | Role |
|------|------|
| `app.py` | Home page — project overview, key metrics, navigation |
| `config.py` | Single source of truth: paths, fusion weights (60/40), UI theme |
| **Pages** | |
| `pages/1_Donnees.py` | Dataset exploration — class distribution, text stats, sample products |
| `pages/2_Preprocessing.py` | Pipeline visualization — text cleaning + image transformation steps |
| `pages/3_Modeles.py` | Model architecture — voting system diagram, benchmark tables |
| `pages/4_Demo.py` | Live classification — text, image, or fusion with adjustable weights |
| `pages/5_Conclusions.py` | Results summary — business impact, limitations, perspectives |
| **Utils** | |
| `utils/real_classifier.py` | `MultimodalClassifier` — production classifier using real models (text + image + fusion) |
| `utils/mock_classifier.py` | `DemoClassifier` — deterministic mock for UI development and testing |
| `utils/model_interface.py` | `BaseClassifier` — abstract interface all classifiers implement |
| `utils/data_loader.py` | Loads CSVs from `data/raw/`, falls back to realistic mock statistics |
| `utils/category_mapping.py` | 27-category mapping with codes, names, and emojis |
| `utils/preprocessing.py` | Text cleaning pipeline — HTML removal, normalization, language detection |
| `utils/image_utils.py` | Image loading, validation, resizing with padding, ResNet preprocessing |
| `utils/ui_utils.py` | CSS injection helper |
| **Assets** | |
| `assets/style.css` | Custom Streamlit theme |
| `assets/comparaison profil modele.png` | Radar chart comparing model profiles |
| `assets/calibre.png` | XGBoost probability calibration plot |
| `assets/matrice.png` | Inter-model correlation matrix |

### `notebooks/` — Research & Training

| File | Role |
|------|------|
| `01_Exploration_*.ipynb` | Data exploration and visualization |
| `02_Preprocessing_*.ipynb` | Text and image preprocessing experiments |
| `03_*.ipynb` | Model benchmarks and cross-validation |
| `M1_IMAGE_Dino.ipynb` | DINOv3 training and evaluation |
| `M2_IMAGE_XGBoost.ipynb` | XGBoost on ResNet50 features |
| `M3_IMAGE_EfficientNet.ipynb` | EfficientNet-B0 fine-tuning |
| `M4_IMAGE_*_Overfit.ipynb` | Overfitting case study (ResNet "Phoenix") |
| `M5_IMAGE_Voting_Final.ipynb` | Voting system assembly and calibration |
| `05_Benchmark_Final.ipynb` | Final benchmark across all models |

### `reports/` — Deliverables

| File | Role |
|------|------|
| `RAPPORT_FINAL_RAKUTEN.html` | Full technical report (open in browser, print to PDF) |
| `PRESENTATION_RAKUTEN_SOUTENANCE.html` | 16-slide presentation (arrow keys to navigate) |
| `RAPPORT_SOUTENANCE_RAKUTEN.md` | Report outline in Markdown |
| `figures/` | Charts and screenshots used in the report |

---

## Architecture

```
Product Input (text + image)
         │
    ┌────┴────┐
    ▼         ▼
  TEXT       IMAGE
    │         │
 TF-IDF    ┌──┼──────────┐
    │       ▼  ▼          ▼
 LinearSVC  DINOv3  EfficientNet  XGBoost
  (83%)     (w=4/7)  (w=2/7)     (w=1/7)
    │         └──────┬──────┘
    │            VOTING (92%)
    │                │
    └───── LATE FUSION (60% image / 40% text) ─── ~94%
```

---

## Key Results

| Pipeline | Model | Accuracy |
|----------|-------|----------|
| Text | LinearSVC + TF-IDF | 83% |
| Image | Voting (DINOv3 + EfficientNet + XGBoost) | 92% |
| **Fusion** | **Late Fusion (60/40)** | **~94%** |

---

## Cloud Deployment (Hugging Face Spaces)

The app auto-downloads models from Hugging Face Hub at startup. This section explains how to set it up.

### Step 1 — Upload Models to Hugging Face Hub

```bash
# Install the CLI
pip install huggingface_hub

# Login (creates token at https://huggingface.co/settings/tokens)
huggingface-cli login

# Create a model repository and upload
huggingface-cli repo create rakuten-models --type model
huggingface-cli upload oussama-akir/rakuten-models models/ --repo-type model
```

This uploads all 6 files from `models/` to `huggingface.co/oussama-akir/rakuten-models`.

### Step 2 — Create a Hugging Face Space

```bash
# Create the Space with Streamlit SDK
huggingface-cli repo create rakuten-classifier --type space --space-sdk streamlit
```

Or create it via the UI at [huggingface.co/new-space](https://huggingface.co/new-space):
- **SDK**: Streamlit
- **Hardware**: CPU Basic (free, 16 GB RAM) — sufficient for all models

### Step 3 — Push Code to the Space

```bash
# Clone the Space repo
git clone https://huggingface.co/spaces/oussama-akir/rakuten-classifier
cd rakuten-classifier

# Copy app files (from the project repo)
cp -r /path/to/OCT25_BMLE_RAKUTEN/src/ .
cp -r /path/to/OCT25_BMLE_RAKUTEN/.streamlit/ .
cp /path/to/OCT25_BMLE_RAKUTEN/requirements.txt .
cp /path/to/OCT25_BMLE_RAKUTEN/models/category_mapping.json models/
```

Create a `README.md` in the Space repo with this frontmatter:

```yaml
---
title: Rakuten Product Classifier
emoji: 🛒
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.0
app_file: src/streamlit/app.py
pinned: false
---
```

### Step 4 — Configure Secrets

In the Space settings (Settings > Variables and secrets), add:

| Key | Value |
|-----|-------|
| `huggingface.repo_id` | `oussama-akir/rakuten-models` |
| `huggingface.token` | *(only if repo is private)* |

Or for local development, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in values.

### Step 5 — Deploy

```bash
git add . && git commit -m "Initial deploy" && git push
```

The Space will build automatically. On first visit, it downloads models (~2.5 GB, takes ~2 min), then caches them for subsequent visits.

### Hardware Options

| Tier | RAM | Cost | Use Case |
|------|-----|------|----------|
| CPU Basic (free) | 16 GB | $0 | Sufficient for all models |
| CPU Upgrade | 32 GB | $0.03/hr | Faster inference |
| T4 Small | 16 GB + GPU | $0.60/hr | Real-time demo |

---

## License

MIT
