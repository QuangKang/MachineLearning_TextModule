"""
Copyright (c) 2026 ML Text Module - BERT Embedding Benchmark Script
Run BERT embeddings benchmark on different dataset sizes and estimate
full dataset processing time.

Usage:
    python bert_benchmark.py
    
Output:
    - Embeddings saved to features/bert/5k_2k/ and features/bert/20k_2k/
    - Timing estimates printed to console
    - Performance metrics saved to results/tables/
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def find_project_root(start=Path.cwd()):
    """Find project root by looking for modules and requirements.txt"""
    for p in [start] + list(start.parents):
        if (p / "modules").exists() and (p / "requirements.txt").exists():
            return p
    raise RuntimeError("Cannot find project root with modules/ and requirements.txt")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import Config
from modules.data_loader import load_data
from modules.bert_embed import build_sbert_embeddings, EmbedConfig
from modules.train_classical import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    """Main benchmark function"""
    print("\n" + "="*80)
    print(" " * 20 + "BERT EMBEDDING BENCHMARK")
    print("="*80)
    
    # Configuration
    cfg = Config()
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Dataset: {cfg.dataset_name}")
    
    # ===== Load full dataset =====
    print("\n" + "-"*80)
    print("1. Loading dataset...")
    print("-"*80)
    
    train_texts, train_labels, test_texts, test_labels, info = load_data("ag_news")
    print(f"✓ Full dataset loaded:")
    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Test:  {len(test_texts):,} samples")
    print(f"  Classes: {info.num_classes}")
    
    # Define benchmark sizes
    BENCH_SIZES = {
        "5k/2k": {"train": 5000, "test": 2000},
        "20k/2k": {"train": 20000, "test": 2000}
    }
    
    datasets = {}
    for name, size_dict in BENCH_SIZES.items():
        n_train = min(size_dict["train"], len(train_texts))
        n_test = min(size_dict["test"], len(test_texts))
        
        datasets[name] = {
            "train_texts": train_texts[:n_train],
            "train_labels": np.array(train_labels[:n_train]),
            "test_texts": test_texts[:n_test],
            "test_labels": np.array(test_labels[:n_test])
        }
        print(f"\n✓ {name}: {n_train:,} train + {n_test:,} test")
    
    # ===== BERT Embedding =====
    print("\n" + "-"*80)
    print("2. Generating BERT embeddings...")
    print("-"*80)
    
    embed_cfg = EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        normalize=True
    )
    
    embeddings_results = {}
    timing_results = defaultdict(dict)
    
    for bench_name, dataset in datasets.items():
        print(f"\n  Processing {bench_name}...")
        
        train_texts = dataset["train_texts"]
        test_texts = dataset["test_texts"]
        
        start = time.time()
        emb_train, emb_test = build_sbert_embeddings(
            train_texts,
            test_texts,
            cfg=embed_cfg
        )
        embed_time = time.time() - start
        
        embeddings_results[bench_name] = {
            "train": emb_train,
            "test": emb_test,
            "train_labels": dataset["train_labels"],
            "test_labels": dataset["test_labels"]
        }
        
        timing_results[bench_name]["embed_time"] = embed_time
        timing_results[bench_name]["n_samples"] = len(train_texts) + len(test_texts)
        timing_results[bench_name]["time_per_sample"] = embed_time / timing_results[bench_name]["n_samples"]
        
        print(f"    ✓ Done in {embed_time:.2f}s ({emb_train.shape[0]:,}+{emb_test.shape[0]:,} samples)")
        print(f"    ✓ Embedding shape: {emb_train.shape}")
    
    # ===== Save embeddings =====
    print("\n" + "-"*80)
    print("3. Saving embeddings to .npy files...")
    print("-"*80)
    
    for bench_name, emb_data in embeddings_results.items():
        bench_dir = cfg.bert_dir / bench_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = bench_dir / "bert_train.npy"
        test_path = bench_dir / "bert_test.npy"
        labels_path = bench_dir / "labels_train.npy"
        labels_test_path = bench_dir / "labels_test.npy"
        
        np.save(train_path, emb_data["train"].astype(np.float32))
        np.save(test_path, emb_data["test"].astype(np.float32))
        np.save(labels_path, emb_data["train_labels"])
        np.save(labels_test_path, emb_data["test_labels"])
        
        print(f"  ✓ {bench_name}: saved to {bench_dir}")
    
    # ===== Train baseline models =====
    print("\n" + "-"*80)
    print("4. Training Logistic Regression baselines...")
    print("-"*80)
    
    eval_results = {}
    
    for bench_name, emb_data in embeddings_results.items():
        print(f"\n  Training on {bench_name}...")
        
        X_train = emb_data["train"]
        y_train = emb_data["train_labels"]
        X_test = emb_data["test"]
        y_test = emb_data["test_labels"]
        
        start = time.time()
        model = get_model("logistic_regression", C=1.0, max_iter=300)
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "train_time": train_time,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        
        eval_results[bench_name] = metrics
        
        print(f"    ✓ Training: {train_time:.2f}s")
        print(f"    ✓ Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    
    # ===== Time estimation =====
    print("\n" + "="*80)
    print("5. TIMING ANALYSIS & FULL DATASET ESTIMATION")
    print("="*80)
    
    print("\n📊 Benchmark Results:")
    print("-" * 80)
    
    for bench_name in eval_results:
        timing = timing_results[bench_name]
        result = eval_results[bench_name]
        
        print(f"\n{bench_name}:")
        print(f"  Samples processed: {timing['n_samples']:,}")
        print(f"  Embedding time: {timing['embed_time']:.2f}s ({timing['time_per_sample']*1000:.3f} ms/sample)")
        print(f"  Training time: {result['train_time']:.2f}s")
        print(f"  Accuracy: {result['accuracy']:.4f} | F1-Score: {result['f1']:.4f}")
    
    # Extrapolate
    print("\n" + "-"*80)
    print("📈 Extrapolation to FULL Dataset (120k train + 7.6k test):")
    print("-" * 80)
    
    avg_time_per_sample = np.mean([timing_results[name]['time_per_sample'] 
                                   for name in timing_results.keys()])
    
    FULL_TRAIN_SIZE = 120000
    FULL_TEST_SIZE = 7600
    FULL_TOTAL = FULL_TRAIN_SIZE + FULL_TEST_SIZE
    
    estimated_embed_time = avg_time_per_sample * FULL_TOTAL
    
    print(f"\nAverage time per sample: {avg_time_per_sample*1000:.3f} ms")
    print(f"Full dataset size: {FULL_TOTAL:,} samples")
    print(f"\n📌 Estimated FULL embedding time: {estimated_embed_time:.1f}s = {estimated_embed_time/60:.1f} min = {estimated_embed_time/3600:.2f} hours")
    
    avg_train_time_ratio = np.mean([eval_results[name]['train_time'] / (embeddings_results[name]['train'].shape[0]/1000) 
                                     for name in eval_results])
    estimated_train_time = avg_train_time_ratio * (FULL_TRAIN_SIZE/1000)
    
    print(f"📌 Estimated FULL training time: {estimated_train_time:.1f}s = {estimated_train_time/60:.2f} min")
    
    total_time = estimated_embed_time + estimated_train_time
    print(f"\n✅ TOTAL estimated time: {total_time:.1f}s = {total_time/60:.1f} min = {total_time/3600:.2f} hours")
    
    # ===== Save report =====
    print("\n" + "-"*80)
    print("6. Saving report...")
    print("-" * 80)
    
    report_data = []
    for bench_name in eval_results:
        report_data.append({
            "Dataset": bench_name,
            "Train Size": embeddings_results[bench_name]['train'].shape[0],
            "Test Size": embeddings_results[bench_name]['test'].shape[0],
            "Embedding Dim": embeddings_results[bench_name]['train'].shape[1],
            "Embed Time (s)": timing_results[bench_name]['embed_time'],
            "Train Time (s)": eval_results[bench_name]['train_time'],
            "Accuracy": f"{eval_results[bench_name]['accuracy']:.4f}",
            "Precision": f"{eval_results[bench_name]['precision']:.4f}",
            "Recall": f"{eval_results[bench_name]['recall']:.4f}",
            "F1-Score": f"{eval_results[bench_name]['f1']:.4f}"
        })
    
    df = pd.DataFrame(report_data)
    
    cfg.table_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.table_dir / "bert_benchmark_results.csv"
    df.to_csv(report_path, index=False)
    
    print(f"\n✓ Report saved to: {report_path}")
    print(f"\n{df.to_string(index=False)}")
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    print(f"""
Next steps for Week 3:
  1. Process full 120k training dataset (estimated {total_time/3600:.1f} hours)
  2. Train SVM and Naive Bayes for comparison
  3. Fine-tune hyperparameters
  4. Generate final performance report
""")


if __name__ == "__main__":
    main()
