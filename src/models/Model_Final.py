"""
Model_Final_Prod (FULL)
- Train final: FeatureUnion(TF-IDF word+char) + LinearSVC(C=0.5)
- Fit on ALL training data (using Rak_train_translations.csv -> product_txt_transl)
- Save pipeline to .joblib
- Predict test (X_test_update.csv) using designation + description
- Export submission.csv

Run:
    python Model_Final_Prod.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[2]

# Train text (translations)
TRANSL_PATH = ROOT / "data" / "processed" / "Rak_train_translations.csv"

# Train labels
Y_PATH = ROOT / "data" / "raw" / "Y_train_CVw08PX.csv"

# Test file (raw)
X_TEST_PATH = ROOT / "data" / "raw" / "X_test_update.csv"

# Output artifacts
OUT_DIR = ROOT / "models" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "tfidf_wordchar_linearsvc_C0p5.joblib"
SUBMISSION_PATH = OUT_DIR / "submission.csv"


# =========================
# Load train data
# =========================
if not TRANSL_PATH.exists():
    raise FileNotFoundError(f"❌ Introuvable: {TRANSL_PATH}")
if not Y_PATH.exists():
    raise FileNotFoundError(f"❌ Introuvable: {Y_PATH}")

df = pd.read_csv(TRANSL_PATH, index_col=0)

TEXT_COL = "product_txt_transl"
if TEXT_COL not in df.columns:
    raise ValueError(f"❌ {TRANSL_PATH.name} doit contenir '{TEXT_COL}'")

df["product_txt"] = (
    df[TEXT_COL]
    .fillna("")
    .astype(str)
    .replace({"nan": "", "None": ""})
)
df = df[df["product_txt"].str.strip().ne("")].copy()

# -------------------------
# Load labels (ROBUST ALIGN)
# -------------------------
y = pd.read_csv(Y_PATH, index_col=0).iloc[:, 0]

# Normalize index types (string/int mismatch is common)
df_idx = pd.to_numeric(df.index, errors="coerce")
y_idx = pd.to_numeric(y.index, errors="coerce")

if df_idx.notna().all() and y_idx.notna().all():
    df.index = df_idx.astype(int)
    y.index = y_idx.astype(int)

df["prdtypecode"] = y.reindex(df.index)
missing = df["prdtypecode"].isna().sum()

# Fallback: if ALL missing, align by row order (only if same length)
if missing == len(df):
    if len(df) != len(y):
        raise ValueError(
            f"❌ Index mismatch + length mismatch: len(df)={len(df)} vs len(y)={len(y)}"
        )
    df["prdtypecode"] = y.to_numpy()
    missing = df["prdtypecode"].isna().sum()

if missing > 0:
    raise ValueError(f"❌ {missing} labels manquants après alignement.")

X_train = df["product_txt"]
y_train = df["prdtypecode"].astype(int)

print(f"✅ Train dataset: {len(df)} lignes")


# =========================
# Build FINAL pipeline
# =========================
word_vec = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=120_000,
    dtype=np.float32,
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=160_000,
    dtype=np.float32,
)

feats = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec),
])

clf = LinearSVC(C=0.5)

model = Pipeline([
    ("feats", feats),
    ("clf", clf),
])


# =========================
# Train on ALL train
# =========================
print("🏋️ Training final model on ALL train data...")
t0 = time.time()
model.fit(X_train, y_train)
print(f"✅ Training done in {time.time() - t0:.1f}s")

joblib.dump(model, MODEL_PATH)
print(f"💾 Saved model to: {MODEL_PATH}")


# =========================
# Predict test & export submission
# =========================
if X_TEST_PATH.exists():
    print(f"📦 Found test file: {X_TEST_PATH.name} -> predicting...")

    df_test = pd.read_csv(X_TEST_PATH, index_col=0)
    print("🧾 Test columns:", list(df_test.columns))

    # Build test text from designation + description
    if "designation" not in df_test.columns or "description" not in df_test.columns:
        raise ValueError("❌ Test doit contenir les colonnes 'designation' et 'description'.")

    df_test["designation"] = df_test["designation"].fillna("").astype(str).replace({"nan": "", "None": ""})
    df_test["description"] = df_test["description"].fillna("").astype(str).replace({"nan": "", "None": ""})

    test_text = (df_test["designation"] + " " + df_test["description"]).str.strip()

    y_pred = model.predict(test_text)

    sub = pd.DataFrame({"prdtypecode": y_pred}, index=df_test.index)
    sub.to_csv(SUBMISSION_PATH)

    print(f"✅ Submission saved to: {SUBMISSION_PATH}")
    print("✅ Submission shape:", sub.shape)
else:
    print(f"ℹ️ No test file found at: {X_TEST_PATH}")
    print("   -> Skipping submission export.")



