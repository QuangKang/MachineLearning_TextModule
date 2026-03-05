# modules/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Dataset
    dataset_name: str = "ag_news"

    # Reproducibility
    seed: int = 42

    # Week 1 (demo/subset sizes)
    n_train_demo: int = 5000
    n_test_demo: int = 2000

    # TF-IDF params
    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2

    # Output filenames
    tfidf_train_name: str = "tfidf_train.npy"
    tfidf_test_name: str = "tfidf_test.npy"

    @property
    def project_root(self) -> Path:
        # modules/config.py nằm trong <root>/modules/config.py
        return Path(__file__).resolve().parents[1]

    @property
    def feature_dir(self) -> Path:
        return self.project_root / "features"

    @property
    def tfidf_train_path(self) -> Path:
        return self.feature_dir / self.tfidf_train_name

    @property
    def tfidf_test_path(self) -> Path:
        return self.feature_dir / self.tfidf_test_name