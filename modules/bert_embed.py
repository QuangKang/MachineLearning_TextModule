# modules/bert_embed.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Any, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    device: Optional[str] = None  # None = auto


def _lazy_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing dependency: sentence-transformers. Install with:\n"
            "pip install sentence-transformers"
        ) from e


def build_sbert_embeddings(
    train_texts: List[str],
    test_texts: List[str],
    *,
    cfg: EmbedConfig = EmbedConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode texts using SBERT -> embeddings (numpy float32).

    Returns:
        emb_train: (n_train, dim)
        emb_test:  (n_test, dim)
    """
    SentenceTransformer = _lazy_import_sentence_transformers()

    model = SentenceTransformer(cfg.model_name, device=cfg.device)

    emb_train = model.encode(
        train_texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=cfg.normalize,
    )
    emb_test = model.encode(
        test_texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=cfg.normalize,
    )

    emb_train = np.asarray(emb_train, dtype=np.float32)
    emb_test = np.asarray(emb_test, dtype=np.float32)
    return emb_train, emb_test


def save_embeddings_npy(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    *,
    feature_dir: str = "features",
    train_name: str = "bert_train.npy",
    test_name: str = "bert_test.npy",
) -> Tuple[str, str]:
    os.makedirs(feature_dir, exist_ok=True)
    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)

    np.save(train_path, emb_train)
    np.save(test_path, emb_test)
    return train_path, test_path


def load_embeddings_npy(
    *,
    feature_dir: str = "features",
    train_name: str = "bert_train.npy",
    test_name: str = "bert_test.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)
    return np.load(train_path), np.load(test_path)


def get_or_build_embeddings(
    train_texts: List[str],
    test_texts: List[str],
    *,
    feature_dir: str = "features",
    train_name: str = "bert_train.npy",
    test_name: str = "bert_test.npy",
    cfg: EmbedConfig = EmbedConfig(),
    rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    If embeddings exist (and rebuild=False) -> load.
    Else -> build + save.
    """
    train_path = os.path.join(feature_dir, train_name)
    test_path = os.path.join(feature_dir, test_name)

    if (not rebuild) and os.path.exists(train_path) and os.path.exists(test_path):
        emb_train, emb_test = load_embeddings_npy(
            feature_dir=feature_dir, train_name=train_name, test_name=test_name
        )
        return emb_train, emb_test, train_path, test_path

    emb_train, emb_test = build_sbert_embeddings(train_texts, test_texts, cfg=cfg)
    p1, p2 = save_embeddings_npy(
        emb_train, emb_test,
        feature_dir=feature_dir, train_name=train_name, test_name=test_name
    )
    return emb_train, emb_test, p1, p2