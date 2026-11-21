import json
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Đảm bảo import được preprocess dù chạy từ thư mục gốc hay thư mục src
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import build_preprocessing_pipeline, load_and_split_data


DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def train_and_tune():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nạp và chia dữ liệu (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = load_and_split_data(
        str(DATA_PATH), test_size=0.2, random_state=42
    )
    print(f"Tập huấn luyện: {X_train.shape[0]} mẫu | Tập kiểm thử: {X_test.shape[0]} mẫu")

    # Lưu tập kiểm thử để evaluate.py sử dụng độc lập
    joblib.dump((X_test, y_test), MODELS_DIR / "test_data.joblib")

    # Tính trọng số xử lý mất cân bằng cho XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)
    print(f"Tỷ lệ mất cân bằng (Negative / Positive): {scale_pos_weight:.2f}")

    # 2. Định nghĩa 4 mô hình và không gian siêu tham số
    models_config = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "params": {
                "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
            "n_iter": 10,
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 8, 12, 16, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4, 6],
                "classifier__max_features": ["sqrt", "log2"],
            },
            "n_iter": 15,
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "classifier__reg_lambda": [1.0, 2.0, 5.0],
            },
            "n_iter": 20,
        },
        "MLP_NeuralNet": {
            "estimator": MLPClassifier(
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
            "params": {
                "classifier__hidden_layer_sizes": [(64,), (128, 64), (64, 32), (32, 16)],
                "classifier__activation": ["relu", "tanh"],
                "classifier__alpha": [0.0001, 0.001, 0.01, 0.1],
                "classifier__learning_rate_init": [0.0005, 0.001, 0.005],
            },
            "n_iter": 12,
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = []
    best_overall_score = -1.0
    best_overall_name = ""

    # 3. Huấn luyện & Tối ưu tham số qua Cross-Validation
    for name, cfg in models_config.items():
        print("\n" + "-" * 60)
        print(f"Đang huấn luyện và tối ưu tham số: {name} ...")

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                ("classifier", cfg["estimator"]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=cfg["params"],
            n_iter=cfg["n_iter"],
            scoring="roc_auc",
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        best_cv_auc = search.best_score_
        best_params = {
            k.replace("classifier__", ""): v for k, v in search.best_params_.items()
        }

        print(f"-> Best CV ROC-AUC ({name}): {best_cv_auc:.4f}")
        print(f"-> Best Params: {best_params}")

        # Lưu từng mô hình
        model_file = MODELS_DIR / f"{name}.joblib"
        joblib.dump(best_model, model_file)

        cv_summary.append(
            {
                "Model": name,
                "CV_ROC_AUC": round(best_cv_auc, 4),
                "Best_Params": json.dumps(best_params),
            }
        )

        if best_cv_auc > best_overall_score:
            best_overall_score = best_cv_auc
            best_overall_name = name
            joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    cv_df = pd.DataFrame(cv_summary).sort_values(by="CV_ROC_AUC", ascending=False)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT HUẤN LUYỆN! Mô hình tốt nhất trên CV: {best_overall_name} (ROC-AUC = {best_overall_score:.4f})")
    print(f"Kết quả CV đã lưu tại: {RESULTS_DIR / 'cv_results.csv'}")


if __name__ == "__main__":
    train_and_tune()