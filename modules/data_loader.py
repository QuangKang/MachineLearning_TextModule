# modules/data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    train_size: int
    test_size: int
    text_field: str = "text"
    label_field: str = "label"
    num_classes: int | None = None


def load_data(
    dataset_name: str = "ag_news",
    *,
    text_field: str = "text",
    label_field: str = "label",
) -> Tuple[List[str], List[Any], List[str], List[Any], DatasetInfo]:
    """Load a text classification dataset from Hugging Face datasets.

    Returns:
        train_texts, train_labels, test_texts, test_labels, info
    """
    ds = load_dataset(dataset_name)

    train = ds["train"]
    test = ds["test"]

    train_texts = train[text_field]
    train_labels = train[label_field]
    test_texts = test[text_field]
    test_labels = test[label_field]

    # Best-effort num_classes
    num_classes = None
    try:
        # HuggingFace ClassLabel feature if present
        feat = train.features.get(label_field)
        if feat is not None and hasattr(feat, "num_classes"):
            num_classes = int(feat.num_classes)
        else:
            num_classes = len(set(train_labels))
    except Exception:
        pass

    info = DatasetInfo(
        name=dataset_name,
        train_size=len(train_texts),
        test_size=len(test_texts),
        text_field=text_field,
        label_field=label_field,
        num_classes=num_classes,
    )
    return train_texts, train_labels, test_texts, test_labels, info
