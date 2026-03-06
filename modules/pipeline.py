import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from modules.tfidf_features import build_tfidf_features, save_features_npy
from modules.bert_embed import EmbedConfig, get_or_build_embeddings
from modules.train_classical import train_eval_logreg, pretty_print_result
import os
def build_features(cfg, train_texts, test_texts):

    if cfg.feature_method == "tfidf":

        N_TRAIN = cfg.n_train_demo
        N_TEST = cfg.n_test_demo

        cfg.tfidf_dir.mkdir(parents=True, exist_ok=True)

        X_train, X_test, _ = build_tfidf_features(
            train_texts[:N_TRAIN],
            test_texts[:N_TEST],
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
        )

        train_path, test_path = save_features_npy(
            X_train,
            X_test,
            feature_dir=str(cfg.tfidf_dir),
            train_name=cfg.tfidf_train_name,
            test_name=cfg.tfidf_test_name,
        )

        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)

        print("Saved:", train_path)
        print("Saved:", test_path)

        print("Files exist?", os.path.exists(train_path), os.path.exists(test_path))

        return X_train, X_test, N_TRAIN, N_TEST



    elif cfg.feature_method == "sbert":
        N_TRAIN = cfg.n_train_emb_demo
        N_TEST = cfg.n_test_emb_demo

        cfg.bert_dir.mkdir(parents=True, exist_ok=True)

        emb_cfg = EmbedConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            normalize=True,
            device=None
        )

        emb_train, emb_test, p_train, p_test = get_or_build_embeddings(
            train_texts[:N_TRAIN],
            test_texts[:N_TEST],
            feature_dir=str(cfg.bert_dir),
            train_name=cfg.bert_train_name,
            test_name=cfg.bert_test_name,
            cfg=emb_cfg,
            rebuild=False
        )

        print("SBERT embeddings saved:")
        print(p_train, p_test)

        return emb_train, emb_test, N_TRAIN, N_TEST


    else:
        raise ValueError("Unknown feature_method")

def run_evaluation(
    X_train,
    y_train,
    X_test,
    y_test,
    method_name="Model"
):
    y_train = y_train[:len(X_train)]
    y_test = y_test[:len(X_test)]
    result = train_eval_logreg(X_train, y_train, X_test, y_test)

    metrics = pretty_print_result(result)
    metrics["method"] = method_name

    df = pd.DataFrame([metrics])
    display(df)

    # confusion matrix

    plt.title(f"Confusion Matrix - {method_name}")
    plt.show()

    return result, df