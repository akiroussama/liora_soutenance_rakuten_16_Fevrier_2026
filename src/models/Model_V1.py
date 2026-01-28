from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Choisis la source de texte :
# 1) "translations" => data/processed/Rak_train_translations.csv (texte = product_txt_transl)
# 2) "train_ready"  => data/processed/train_ready.csv (texte = product_txt ou product_txt_transl + label déjà dedans)
DATA_SOURCE = "translations"   # <-- change en "train_ready" si tu veux


def load_data(root: Path, source: str):
    """
    Charge X (texte) et y (labels) selon la source.
    - translations: lit le texte traduit et lit y depuis data/raw
    - train_ready: lit un fichier déjà prêt contenant X et y
    """
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"

    if source == "translations":
        x_path = processed / "Rak_train_translations.csv"
        y_path = raw / "Y_train_CVw08PX.csv"

        df_x = pd.read_csv(x_path, index_col=0)

        # colonne texte attendue
        if "product_txt_transl" not in df_x.columns:
            raise ValueError("Rak_train_translations.csv doit contenir la colonne 'product_txt_transl'.")

        X = df_x["product_txt_transl"].fillna("").astype(str)

        # labels
        df_y = pd.read_csv(y_path, index_col=0)
        # parfois y est une seule colonne sans nom, donc on gère les 2 cas
        if "prdtypecode" in df_y.columns:
            y = df_y["prdtypecode"]
        else:
            y = df_y.iloc[:, 0]
            y.name = "prdtypecode"

        # alignement index (important)
        y = y.reindex(df_x.index)

        return X, y

    elif source == "train_ready":
        ready_path = processed / "train_ready.csv"
        df = pd.read_csv(ready_path, index_col=0)

        # accepte product_txt ou product_txt_transl
        text_col = None
        for c in ["product_txt_transl", "product_txt"]:
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise ValueError("train_ready.csv doit contenir 'product_txt' ou 'product_txt_transl'.")

        if "prdtypecode" not in df.columns:
            raise ValueError("train_ready.csv doit contenir la colonne 'prdtypecode'.")

        X = df[text_col].fillna("").astype(str)
        y = df["prdtypecode"]

        return X, y

    else:
        raise ValueError("DATA_SOURCE doit être 'translations' ou 'train_ready'.")


def main():
    # Racine du projet : OCT25_BMLE_RAKUTEN/
    ROOT = Path(__file__).resolve().parents[2]

    # 1) Load
    X, y = load_data(ROOT, DATA_SOURCE)

    # sécurité anti-NaN
    X = X.fillna("").astype(str)
    y = y.dropna()

    # 2) Split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3) Pipeline TF-IDF + LinearSVC (baseline NLP solide)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=120_000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LinearSVC())
    ])

    # 4) Train
    pipe.fit(X_tr, y_tr)

    # 5) Eval
    y_pred = pipe.predict(X_va)

    f1 = f1_score(y_va, y_pred, average="macro")
    print("\n=== Model_V1 (LinearSVC) ===")
    print("F1 macro:", f1)

    print("\n=== Classification report ===")
    print(classification_report(y_va, y_pred))


if __name__ == "__main__":
    main()

