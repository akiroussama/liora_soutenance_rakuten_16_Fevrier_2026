from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# "translations" = Rak_train_translations.csv (texte=product_txt_transl) + y depuis raw
# "train_ready"  = train_ready.csv (texte + prdtypecode dedans)
DATA_SOURCE = "translations"

# Si tu veux tester aussi LogisticRegression
RUN_LOGREG = True


def load_data(root: Path, source: str):
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"

    if source == "translations":
        x_path = processed / "Rak_train_translations.csv"
        y_path = raw / "Y_train_CVw08PX.csv"

        df_x = pd.read_csv(x_path, index_col=0)
        if "product_txt_transl" not in df_x.columns:
            raise ValueError("Rak_train_translations.csv doit contenir la colonne 'product_txt_transl'.")

        X = df_x["product_txt_transl"].fillna("").astype(str)

        df_y = pd.read_csv(y_path, index_col=0)
        if "prdtypecode" in df_y.columns:
            y = df_y["prdtypecode"]
        else:
            y = df_y.iloc[:, 0]
            y.name = "prdtypecode"

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
            raise ValueError("train_ready.csv doit contenir 'prdtypecode'.")

        X = df[text_col].fillna("").astype(str)
        y = df["prdtypecode"]
        return X, y

    else:
        raise ValueError("DATA_SOURCE doit être 'translations' ou 'train_ready'.")


def run_grid(name: str, pipe: Pipeline, param_grid: dict, X_tr, y_tr, X_va, y_va):
    print(f"\n=== GRID SEARCH: {name} ===")

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    gs.fit(X_tr, y_tr)

    print("\n✅ Best params:")
    print(gs.best_params_)
    print("✅ Best CV f1_macro:", gs.best_score_)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_va)

    f1 = f1_score(y_va, y_pred, average="macro")
    print("\n✅ Validation F1 macro:", f1)
    print("\n=== Classification report (VAL) ===")
    print(classification_report(y_va, y_pred))

    return best_model, f1


def main():
    ROOT = Path(__file__).resolve().parents[2]
    X, y = load_data(ROOT, DATA_SOURCE)

    # sécurité NaN
    X = X.fillna("").astype(str)
    y = y.dropna()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # =========================
    # 1) LinearSVC Grid
    # =========================
    pipe_svc = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    grid_svc = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_df": [0.9, 0.95],
        "tfidf__max_features": [80_000, 120_000],
        "tfidf__sublinear_tf": [True],
        "clf__C": [0.5, 1.0, 2.0],
    }

    best_models = []
    best_pipe_svc, best_f1_svc = run_grid(
        "LinearSVC", pipe_svc, grid_svc, X_tr, y_tr, X_va, y_va
    )
    best_models.append(("LinearSVC", best_f1_svc, best_pipe_svc))

    # =========================
    # 2) LogisticRegression Grid (option)
    # =========================
    if RUN_LOGREG:
        pipe_lr = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
        ])

        grid_lr = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [2, 5],
            "tfidf__max_df": [0.9, 0.95],
            "tfidf__max_features": [80_000, 120_000],
            "tfidf__sublinear_tf": [True],
            "clf__C": [0.5, 1.0, 2.0],
            "clf__class_weight": [None, "balanced"],
        }

        best_pipe_lr, best_f1_lr = run_grid(
            "LogisticRegression", pipe_lr, grid_lr, X_tr, y_tr, X_va, y_va
        )
        best_models.append(("LogisticRegression", best_f1_lr, best_pipe_lr))

    # =========================
    # BEST OVERALL
    # =========================
    best_name, best_f1, best_est = sorted(best_models, key=lambda x: x[1], reverse=True)[0]
    print("\n============================")
    print("🏆 BEST OVERALL MODEL")
    print("Model:", best_name)
    print("F1 macro (VAL):", best_f1)
    print("============================")


if __name__ == "__main__":
    main()
