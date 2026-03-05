# modules/train_classical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


@dataclass
class EvalResult:
    accuracy: float
    f1_weighted: float


def train_eval_logreg(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    *,
    max_iter: int = 200,
) -> EvalResult:
    """
    Train + Eval Logistic Regression trên features (TF-IDF dense).
    Trả về Accuracy và F1-weighted.
    """

    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1w = float(f1_score(y_test, y_pred, average="weighted"))

    return EvalResult(accuracy=acc, f1_weighted=f1w)


def pretty_print_result(result: EvalResult) -> Dict[str, Any]:
    """
    Trả dict để in bảng/ghi log dễ hơn.
    """
    return {"accuracy": result.accuracy, "f1_weighted": result.f1_weighted}