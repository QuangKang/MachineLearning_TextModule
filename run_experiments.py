import time
import numpy as np
from modules.data_loader import load_data
from modules.text_preprocess import TextCleaner
from modules.tfidf_features import build_tfidf_features, save_features_npy
from modules.train_classical import train_eval
from modules.metrics import print_result

def clean_large_corpus(cleaner, corpus, batch_print=20000):
    cleaned = []
    total = len(corpus)
    for i, text in enumerate(corpus):
        cleaned.append(cleaner.clean_text(text))
        if (i + 1) % batch_print == 0:
            print(f"  -> Đã làm sạch {i + 1}/{total} văn bản...")
    return cleaned

def main():
    print("=== BƯỚC 1: TẢI TOÀN BỘ DỮ LIỆU ===")
    train_texts, y_train, test_texts, y_test, info = load_data("ag_news")
    
    # Ép kiểu nhãn về định dạng NumPy Array để tương thích với Scikit-learn GridSearchCV
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Đã tải xong! Tập Train gốc: {info.train_size} mẫu, Tập Test gốc: {info.test_size} mẫu.\n")

    print("=== BƯỚC 2: TIỀN XỬ LÝ TOÀN BỘ DỮ LIỆU ===")
    start_time = time.time()
    cleaner = TextCleaner()
    
    print("Đang làm sạch tập Train (120,000 mẫu)...")
    train_clean = clean_large_corpus(cleaner, train_texts, batch_print=20000)
    
    print("Đang làm sạch tập Test (7,600 mẫu)...")
    test_clean = clean_large_corpus(cleaner, test_texts, batch_print=2000)
    
    print(f"Hoàn thành làm sạch trong {time.time() - start_time:.2f} giây.\n")

    print("=== BƯỚC 3: HUẤN LUYỆN & TÌM CẤU HÌNH TỐT NHẤT ===")
    
    configs = [
        {"name": "Cấu hình 1 (Unigram, 1k features)", "max_features": 1000, "ngram_range": (1, 1)},
        {"name": "Cấu hình 2 (Unigram+Bigram, 3k features)", "max_features": 3000, "ngram_range": (1, 2)},
        {"name": "Cấu hình 3 (Unigram+Bigram, 5k features)", "max_features": 5000, "ngram_range": (1, 2)},
    ]

    # Các class của tập ag_news để cung cấp cho hàm train_eval vẽ Confusion Matrix
    ag_news_classes = ["World", "Sports", "Business", "Sci/Tech"]
    
    best_f1 = 0.0
    best_config_name = ""
    best_X_train = None
    best_X_test = None

    for cfg in configs:
        print(f"\n>> Đang chạy {cfg['name']}...")
        
        X_train, X_test, _ = build_tfidf_features(
            train_clean, test_clean,
            max_features=cfg['max_features'],
            ngram_range=cfg['ngram_range'],
            min_df=3
        )
        
        # Gọi hàm train_eval theo ĐÚNG thiết kế mới của đồng đội
        result = train_eval(
            model_type="logistic_regression",
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            class_names=ag_news_classes,
            save="results"
        )
        
        metrics_dict = print_result(result)
        print(f"   Accuracy: {metrics_dict['accuracy']:.4f} | F1-Score: {metrics_dict['f1_weighted']:.4f}")

        if metrics_dict['f1_weighted'] > best_f1:
            best_f1 = metrics_dict['f1_weighted']
            best_config_name = cfg['name']
            best_X_train = X_train
            best_X_test = X_test

    print("\n=== BƯỚC 4: LƯU ĐẶC TRƯNG TỐT NHẤT ===")
    print(f"Cấu hình chiến thắng: {best_config_name} với F1={best_f1:.4f}")
    
    save_features_npy(
        best_X_train, best_X_test, 
        feature_dir="features", 
        train_name="tfidf_train_best.npy", 
        test_name="tfidf_test_best.npy"
    )

if __name__ == "__main__":
    main()
