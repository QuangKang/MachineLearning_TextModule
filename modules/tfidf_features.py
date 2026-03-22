import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_features(
    train_texts,
    test_texts,
    *,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
):
    """Build TF-IDF features for train/test and return dense float32 arrays."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    X_train_sparse = vectorizer.fit_transform(train_texts)
    X_test_sparse = vectorizer.transform(test_texts)

    # Chuyển về float32 để chống tràn RAM
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)

    return X_train, X_test, vectorizer


def save_features_npy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    feature_dir: str = "features",
    train_name: str = "tfidf_train.npy",
    test_name: str = "tfidf_test.npy",
):
    """Save train/test feature arrays to .npy files and return file paths."""

    os.makedirs(feature_dir, exist_ok=True)
    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)

    np.save(train_path, X_train)
    np.save(test_path, X_test)
    print(f"Đã lưu thành công:\n- {train_path}\n- {test_path}")

    return train_path, test_path


def load_features_npy(
    *,
    feature_dir: str = "features",
    train_name: str = "tfidf_train.npy",
    test_name: str = "tfidf_test.npy",
):
    """Load train/test feature arrays from .npy files."""

    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)
    return np.load(train_path), np.load(test_path)
