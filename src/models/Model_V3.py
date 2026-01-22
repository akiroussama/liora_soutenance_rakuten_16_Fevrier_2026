"""
Model_Search_Simple
- Vectorizes ONCE (word + char TF-IDF)
- Runs quick hyperparam search with FEW fits:
    - RandomizedSearchCV with small n_iter
    - cv=2
"""

from pathlib import Path
from time import time
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[2]
TRANSL_PATH = ROOT / "data" / "processed" / "Rak_train_translations.csv"
Y_PATH = ROOT / "data" / "raw" / "Y_train_CVw08PX.csv"  


# =========================
# Load translations + labels
# =========================
if not TRANSL_PATH.exists():
    raise FileNotFoundError(f"❌ Introuvable: {TRANSL_PATH}")
if not Y_PATH.exists():
    raise FileNotFoundError(f"❌ Introuvable: {Y_PATH}")

df = pd.read_csv(TRANSL_PATH, index_col=0)
if "product_txt_transl" not in df.columns:
    raise ValueError("❌ Rak_train_translations.csv doit contenir 'product_txt_transl'")

df["product_txt"] = (
    df["product_txt_transl"]
    .fillna("")
    .astype(str)
    .replace({"nan": "", "None": ""})
)
df = df[df["product_txt"].str.strip().ne("")]

y = pd.read_csv(Y_PATH, index_col=0).iloc[:, 0]
df["prdtypecode"] = y.reindex(df.index)

missing = df["prdtypecode"].isna().sum()
if missing > 0:
    raise ValueError(f"❌ {missing} labels manquants après alignement (index mismatch).")

df["prdtypecode"] = df["prdtypecode"].astype(int)

X = df["product_txt"]
y = df["prdtypecode"]

print(f"✅ Dataset: {len(df)} lignes")


# =========================
# Split once
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Vectorize ONCE (best stable defaults)
# =========================
word_vec = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=120_000,
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=160_000,
)

feats = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec),
])

print("⚙️ Vectorizing (fit once)...")
X_train_vec = feats.fit_transform(X_train)
X_val_vec = feats.transform(X_val)
print("✅ Vectorization done")


# =========================
# Quick model search (few fits)
# =========================
def quick_search(name, estimator, param_distributions, n_iter=8):
    
    print("RUN_ID:", time.time())

    """
    cv=2 => few fits
    n_iter small => few fits
    Total fits ~ n_iter * cv
    """
    print(f"\n🔎 Searching: {name} (n_iter={n_iter}, cv=2)")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=2,
        verbose=0,
        n_jobs=1,
        random_state=42,
        refit=True,  # refit only the best on all train folds
    )
    search.fit(X_train_vec, y_train)

    best = search.best_estimator_
    print(f"✅ Best params for {name}: {search.best_params_}")
    y_pred = best.predict(X_val_vec)
    print(f"\n📊 Report (VAL) - {name}\n")
    print(classification_report(y_val, y_pred, digits=2))
    return search.best_params_, search.best_score_


# 1) LinearSVC ( texte)
svc = LinearSVC()

svc_params = {
    "C": [0.5],
    
}

# 3) SGDClassifier (rapide)
sgd = SGDClassifier(
    max_iter=2000,
    tol=1e-3,
    random_state=42,
)
sgd_params = {
    "loss": ["hinge", "log_loss"],          # SVM-like / logreg-like
    "alpha": [1e-5, 1e-4, 1e-3],            # regularization
}

results = []
results.append(("LinearSVC",) + quick_search("LinearSVC", svc, svc_params, n_iter=4))

results.append(("SGDClassifier",) + quick_search("SGDClassifier", sgd, sgd_params, n_iter=6))

print("\n🏁 Summary (best CV f1_weighted):")
for name, params, score in results:
    print(f"- {name}: best_cv_f1w={score:.4f} | params={params}")
