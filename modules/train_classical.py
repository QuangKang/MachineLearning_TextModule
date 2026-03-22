# modules/train_classical.py
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from modules.metrics import calculate_metrics, EvalResult, print_result, plot_confusion_matrix

def get_model(model_type: str, C: float = 1.0, alpha: float = 1.0, max_iter: int = 500) -> BaseEstimator:
    """
    Factory pattern để người làm Pipeline/Config dễ dàng đổi mô hình.
    """
    if model_type == "logistic_regression":
        return LogisticRegression(C=C, max_iter=max_iter)
    elif model_type == "naive_bayes":
        return MultinomialNB(alpha=alpha)
    elif model_type == "svm":
        return LinearSVC(C=C, max_iter=max_iter, dual=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
def train_eval(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    save: str="results"
) -> EvalResult:
    """
    Hàm thực hiện Tuning nhẹ, dự đoán và xuất Confusion Matrix.
    """
    print(f"\n--- Đang Tuning mô hình: {model_type.upper()} ---")

    # 1. Cấu hình các tham số cần thử (Tuning nhẹ)
    if model_type == "logistic_regression" or model_type == "svm":
        model = get_model(model_type)
        param_grid = {'C':[0.1, 1.0, 10.0]} # Thử 3 giá trị C
    elif model_type == "naive_bayes":
        model = get_model(model_type)
        param_grid = {'alpha':[0.1, 0.5, 1.0]} # Thử 3 giá trị alpha

    # 2. Tìm kiếm tham số tốt nhất bằng GridSearchCV (ưu tiên F1-weighted)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Tham số tốt nhất tìm được: {grid_search.best_params_}")

    # 3. Dự đoán trên tập Test
    y_pred = best_model.predict(X_test)

    # 4. Tính toán Metrics
    result = calculate_metrics(y_test, y_pred)

    # 5. Vẽ và lưu Confusion Matrix
    save_path = os.path.join(save, f"cm_{model_type}.png")
    plot_confusion_matrix(y_test, y_pred, class_names, save_path, title=f"Confusion Matrix - {model_type.upper()}")

    return result
    
# --- PHẦN TEST VỚI DUMMY FEATURES ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("--- Bắt đầu test khung Train/Tuning/Eval ---")
    
    # Tạo data giả (4 classes)
    X_dummy, y_dummy = make_classification(n_samples=500, n_features=100, n_classes=4, n_informative=10, random_state=42)
    X_dummy = np.abs(X_dummy) # Ép số dương cho Naive Bayes
    X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)
    
    # Tên của 4 class giả định
    dummy_classes = ["Class 0", "Class 1", "Class 2", "Class 3"]

    # Chạy thử quy trình cho Logistic Regression
    res_lr = train_eval("logistic_regression", X_train, y_train, X_test, y_test, dummy_classes)
    print("Kết quả LR:", print_result(res_lr))

    # Chạy thử quy trình cho Naive Bayes
    res_nb = train_eval("naive_bayes", X_train, y_train, X_test, y_test, dummy_classes)
    print("Kết quả NB:", print_result(res_nb))