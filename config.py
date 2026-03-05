# modules/config.py

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

    # demo subset 
    n_train_demo: int = 5000
    n_test_demo: int = 2000

    n_train_emb_demo: int = 2000
    n_test_emb_demo: int = 500

    # Preprocessing

    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lemmatize: bool = False


    # Feature extraction

    feature_method: str = "sbert"   # tfidf | sbert


    # TF-IDF params

    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2

    tfidf_train_name: str = "tfidf_train.npy"
    tfidf_test_name: str = "tfidf_test.npy"


    # BERT params

    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sbert_batch_size: int = 32
    sbert_normalize: bool = True

    bert_train_name: str = "bert_train.npy"
    bert_test_name: str = "bert_test.npy"


    # Model

    model_type: str = "logistic_regression"  # naive_bayes | logistic_regression | svm

    C: float = 1.0


    # Training

    epochs: int = 5
    learning_rate: float = 2e-5


    # Paths

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent

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