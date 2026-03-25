"""
BERT embedding benchmark script.

Runs SBERT embedding extraction on multiple dataset sizes,
trains baseline classifiers, and saves a benchmark report.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def find_project_root(start: Path = Path.cwd()) -> Path:
    """Find project root by looking for modules/ and requirements.txt."""
    for p in [start] + list(start.parents):
        if (p / "modules").exists() and (p / "requirements.txt").exists():
            return p
    raise RuntimeError("Cannot find project root with modules/ and requirements.txt")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from modules.bert_embed import EmbedConfig, build_sbert_embeddings
from modules.config import Config
from modules.data_loader import load_data
from modules.train_classical import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SBERT benchmark on selected dataset scales.")
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["5k_2k", "20k_2k"],
        help="Benchmark scales to run. Choices: 5k_2k, 20k_2k",
    )
    return parser.parse_args()


def main(scales: list[str] | None = None) -> None:
    print("\n" + "=" * 80)
    print(" " * 20 + "BERT EMBEDDING BENCHMARK")
    print("=" * 80)

    cfg = Config()
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Dataset: {cfg.dataset_name}")

    print("\n" + "-" * 80)
    print("1. Loading dataset...")
    print("-" * 80)

    train_texts, train_labels, test_texts, test_labels, info = load_data(cfg.dataset_name)
    print("Full dataset loaded")
    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,} samples")
    print(f"  Classes: {info.num_classes}")

    all_bench_sizes = {
        "5k_2k": {"train": 5000, "test": 2000},
        "20k_2k": {"train": 20000, "test": 2000},
    }

    requested_scales = scales if scales is not None else ["5k_2k", "20k_2k"]
    bench_sizes = {k: all_bench_sizes[k] for k in requested_scales if k in all_bench_sizes}
    if not bench_sizes:
        raise ValueError("No valid benchmark scales selected. Use 5k_2k and/or 20k_2k.")

    datasets: dict[str, dict[str, Any]] = {}
    for name, size_cfg in bench_sizes.items():
        n_train = min(size_cfg["train"], len(train_texts))
        n_test = min(size_cfg["test"], len(test_texts))
        datasets[name] = {
            "train_texts": train_texts[:n_train],
            "train_labels": np.array(train_labels[:n_train]),
            "test_texts": test_texts[:n_test],
            "test_labels": np.array(test_labels[:n_test]),
        }
        print(f"  {name}: {n_train:,} train + {n_test:,} test")

    print("\n" + "-" * 80)
    print("2. Generating SBERT embeddings...")
    print("-" * 80)

    embed_cfg = EmbedConfig(
        model_name=cfg.sbert_model_name,
        batch_size=cfg.sbert_batch_size,
        normalize=cfg.sbert_normalize,
    )

    embeddings_results: dict[str, dict[str, Any]] = {}
    timing_results = defaultdict(dict)

    for bench_name, dataset in datasets.items():
        print(f"\n  Processing {bench_name}...")
        x_train_texts = dataset["train_texts"]
        x_test_texts = dataset["test_texts"]

        start = time.time()
        emb_train, emb_test = build_sbert_embeddings(x_train_texts, x_test_texts, cfg=embed_cfg)
        embed_time = time.time() - start

        embeddings_results[bench_name] = {
            "train": emb_train,
            "test": emb_test,
            "train_labels": dataset["train_labels"],
            "test_labels": dataset["test_labels"],
        }

        timing_results[bench_name]["embed_time"] = embed_time
        timing_results[bench_name]["n_samples"] = len(x_train_texts) + len(x_test_texts)
        timing_results[bench_name]["time_per_sample"] = embed_time / timing_results[bench_name]["n_samples"]

        print(f"    done in {embed_time:.2f}s")
        print(f"    embedding shape: {emb_train.shape}")

    print("\n" + "-" * 80)
    print("3. Saving embeddings to .npy files...")
    print("-" * 80)

    for bench_name, emb_data in embeddings_results.items():
        bench_dir = cfg.bert_dir / bench_name
        bench_dir.mkdir(parents=True, exist_ok=True)

        np.save(bench_dir / "bert_train.npy", emb_data["train"].astype(np.float32))
        np.save(bench_dir / "bert_test.npy", emb_data["test"].astype(np.float32))
        np.save(bench_dir / "labels_train.npy", emb_data["train_labels"])
        np.save(bench_dir / "labels_test.npy", emb_data["test_labels"])

        print(f"  saved: {bench_dir}")

    print("\n" + "-" * 80)
    print("4. Training baseline models (logistic_regression, svm, naive_bayes)...")
    print("-" * 80)

    model_configs: dict[str, dict[str, Any]] = {
        "logistic_regression": {"C": 1.0, "max_iter": 300},
        "svm": {"C": 1.0, "max_iter": 300},
        "naive_bayes": {"alpha": 1.0},
    }

    eval_results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for bench_name, emb_data in embeddings_results.items():
        print(f"\n  Training on {bench_name}...")

        x_train = emb_data["train"]
        y_train = emb_data["train_labels"]
        x_test = emb_data["test"]
        y_test = emb_data["test_labels"]

        for model_type, model_kwargs in model_configs.items():
            x_train_fit = x_train
            x_test_fit = x_test

            # MultinomialNB requires non-negative features.
            if model_type == "naive_bayes":
                min_value = float(min(x_train.min(), x_test.min()))
                if min_value < 0:
                    offset = -min_value
                    x_train_fit = x_train + offset
                    x_test_fit = x_test + offset

            start = time.time()
            model = get_model(model_type, **model_kwargs)
            model.fit(x_train_fit, y_train)
            train_time = time.time() - start

            y_pred = model.predict(x_test_fit)
            metrics = {
                "train_time": float(train_time),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }

            eval_results[bench_name][model_type] = metrics
            print(
                f"    {model_type}: train={metrics['train_time']:.2f}s | "
                f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f}"
            )

    print("\n" + "=" * 80)
    print("5. Timing analysis and full-dataset estimation")
    print("=" * 80)

    print("\nBenchmark results:")
    print("-" * 80)
    for bench_name in eval_results:
        timing = timing_results[bench_name]
        print(f"\n{bench_name}:")
        print(f"  Samples processed: {timing['n_samples']:,}")
        print(
            f"  Embedding time: {timing['embed_time']:.2f}s "
            f"({timing['time_per_sample'] * 1000:.3f} ms/sample)"
        )

        for model_type, result in eval_results[bench_name].items():
            print(
                f"  {model_type:>20}: train={result['train_time']:.2f}s | "
                f"acc={result['accuracy']:.4f} | f1={result['f1']:.4f}"
            )

    full_train_size = 120000
    full_test_size = 7600
    full_total = full_train_size + full_test_size

    avg_time_per_sample = np.mean([timing_results[name]["time_per_sample"] for name in timing_results])
    estimated_embed_time = avg_time_per_sample * full_total

    train_time_ratios = []
    for bench_name in eval_results:
        n_train = embeddings_results[bench_name]["train"].shape[0]
        for model_type in eval_results[bench_name]:
            train_time = eval_results[bench_name][model_type]["train_time"]
            train_time_ratios.append(train_time / (n_train / 1000))

    estimated_train_time = float(np.mean(train_time_ratios)) * (full_train_size / 1000)
    total_time = estimated_embed_time + estimated_train_time

    print("\n" + "-" * 80)
    print("Extrapolation to full dataset (120k train + 7.6k test)")
    print("-" * 80)
    print(f"Average embedding time per sample: {avg_time_per_sample * 1000:.3f} ms")
    print(f"Estimated full embedding time: {estimated_embed_time / 60:.2f} min")
    print(f"Estimated full training time: {estimated_train_time / 60:.2f} min")
    print(f"Estimated total time: {total_time / 60:.2f} min ({total_time / 3600:.2f} hours)")

    print("\n" + "-" * 80)
    print("6. Saving report...")
    print("-" * 80)

    report_rows = []
    for bench_name in eval_results:
        for model_type, result in eval_results[bench_name].items():
            report_rows.append(
                {
                    "Dataset": bench_name,
                    "Model": model_type,
                    "Train Size": embeddings_results[bench_name]["train"].shape[0],
                    "Test Size": embeddings_results[bench_name]["test"].shape[0],
                    "Embedding Dim": embeddings_results[bench_name]["train"].shape[1],
                    "Embed Time (s)": timing_results[bench_name]["embed_time"],
                    "Train Time (s)": result["train_time"],
                    "Accuracy": f"{result['accuracy']:.4f}",
                    "Precision": f"{result['precision']:.4f}",
                    "Recall": f"{result['recall']:.4f}",
                    "F1-Score": f"{result['f1']:.4f}",
                }
            )

    df = pd.DataFrame(report_rows)
    cfg.table_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.table_dir / "bert_benchmark_results.csv"
    df.to_csv(report_path, index=False)

    print(f"Report saved to: {report_path}")
    print("\n" + df.to_string(index=False))

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    main(scales=cli_args.scales)
