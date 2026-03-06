import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFExtractor:
    def __init__(self, max_features=None, ngram_range=(1, 1)):
        # Khởi tạo TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.feature_names = None

    def fit_transform(self, corpus):
        """Huấn luyện vectorizer và biến đổi corpus thành ma trận TF-IDF."""
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Chuyển đổi ma trận thưa (sparse matrix) sang mảng numpy đặc (dense array) để dễ lưu trữ
        return tfidf_matrix.toarray()

    def transform(self, corpus):
        """Biến đổi tập dữ liệu mới (test set) dựa trên vectorizer đã huấn luyện."""
        return self.vectorizer.transform(corpus).toarray()

    def save_features(self, features_array, filename="tfidf_features.npy"):
        """Lưu mảng đặc trưng thành file .npy theo yêu cầu của bài tập."""
        np.save(filename, features_array)
        print(f"Đã lưu ma trận đặc trưng vào {filename}")
# modules/tfidf_features.py

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
    """
    Tạo đặc trưng TF-IDF cho train/test.
    - Fit vectorizer trên train_texts
    - Transform train/test
    - Trả về X_train, X_test dạng numpy float32 (dense) + vectorizer
    """

    # 1) Khởi tạo TF-IDF vectorizer với các tham số cấu hình
    vectorizer = TfidfVectorizer(
        lowercase=True,          # đưa về chữ thường
        ngram_range=ngram_range, # (1,2) = unigram + bigram
        max_features=max_features,  # giới hạn số đặc trưng để tránh nặng RAM
        min_df=min_df,           # bỏ từ xuất hiện quá ít (lọc nhiễu)
    )

    # 2) Fit trên train và biến train thành ma trận đặc trưng (sparse)
    X_train_sparse = vectorizer.fit_transform(train_texts)

    # 3) Transform test bằng vectorizer đã fit (sparse)
    X_test_sparse = vectorizer.transform(test_texts)

    # 4) Chuyển sparse -> dense numpy để lưu .npy (Pylance-friendly)
    #    Lưu ý: dense có thể nặng, nên giới hạn max_features.
    X_train = np.asarray(X_train_sparse.todense(), dtype=np.float32)
    X_test = np.asarray(X_test_sparse.todense(), dtype=np.float32)

    return X_train, X_test, vectorizer


def save_features_npy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    feature_dir: str = "features",
    train_name: str = "tfidf_train.npy",
    test_name: str = "tfidf_test.npy",
):
    """
    Lưu features ra .npy trong thư mục features/
    Trả về đường dẫn file đã lưu.
    """

    # 1) Tạo thư mục nếu chưa có
    os.makedirs(feature_dir, exist_ok=True)

    # 2) Ghép đường dẫn file
    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)

    # 3) Lưu numpy arrays
    np.save(train_path, X_train)
    np.save(test_path, X_test)

    return train_path, test_path


def load_features_npy(
    *,
    feature_dir: str = "features",
    train_name: str = "tfidf_train.npy",
    test_name: str = "tfidf_test.npy",
):
    """
    Load lại features từ .npy (dùng cho TV3 train/eval).
    """

    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)

    X_train = np.load(train_path)
    X_test = np.load(test_path)

    return X_train, X_test
