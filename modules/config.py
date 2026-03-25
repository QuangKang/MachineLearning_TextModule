from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Dataset
    dataset_name: str = "ag_news"
    text_column: str = "text"
    label_column: str = "label"
    seed: int = 42
    test_size: float = 0.2

    # demo subset sizes
    n_train_demo: int = 5000
    n_test_demo: int = 2000
    n_train_emb_demo: int = 2000
    n_test_emb_demo: int = 500

    # Preprocessing
    use_preprocessing: bool = True
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = True

    # Mode (immutable)
    mode: str = "demo"  # demo | full

    # Feature extraction
    feature_method: str = "sbert"  # tfidf | sbert

    # TF-IDF params
    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    tfidf_train_name: str = "tfidf_train.npy"
    tfidf_test_name: str = "tfidf_test.npy"

    # SBERT params
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int = 32
    sbert_normalize: bool = True
    use_sbert_preprocessing: bool = False
    bert_train_name: str = "bert_train.npy"
    bert_test_name: str = "bert_test.npy"

    # Model
    model_type: str = "logistic_regression"  # naive_bayes | logistic_regression | svm
    model_list: tuple[str, ...] = ("logistic_regression", "svm", "naive_bayes")
    primary_metric: str = "f1_weighted"
    C: float = 1.0
    max_iter: int = 200

    # Training
    epochs: int = 5
    learning_rate: float = 2e-5

    # ---------- Demo / Full helper properties ----------
    @property
    def n_train(self) -> int | None:
        return self.n_train_demo if self.mode == "demo" else None

    @property
    def n_test(self) -> int | None:
        return self.n_test_demo if self.mode == "demo" else None

    @property
    def n_train_emb(self) -> int | None:
        return self.n_train_emb_demo if self.mode == "demo" else None

    @property
    def n_test_emb(self) -> int | None:
        return self.n_test_emb_demo if self.mode == "demo" else None

    # ---------- Paths (same as before) ----------
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"
    @property
    def feature_dir(self) -> Path:
        return self.project_root / "features"
    @property
    def tfidf_dir(self) -> Path:
        return self.feature_dir / "tfidf"
    @property
    def bert_dir(self) -> Path:
        return self.feature_dir / "bert"
    @property
    def model_dir(self) -> Path:
        return self.project_root / "models"
    @property
    def result_dir(self) -> Path:
        return self.project_root / "results"
    @property
    def figure_dir(self) -> Path:
        return self.result_dir / "figures"
    @property
    def table_dir(self) -> Path:
        return self.result_dir / "tables"
    @property
    def log_dir(self) -> Path:
        return self.result_dir / "logs"

    # Feature paths
    @property
    def tfidf_train_path(self) -> Path:
        return self.tfidf_dir / self.tfidf_train_name
    @property
    def tfidf_test_path(self) -> Path:
        return self.tfidf_dir / self.tfidf_test_name
    @property
    def bert_train_path(self) -> Path:
        return self.bert_dir / self.bert_train_name
    @property
    def bert_test_path(self) -> Path:
        return self.bert_dir / self.bert_test_name