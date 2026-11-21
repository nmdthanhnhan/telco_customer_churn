from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parent))
import preprocess  # noqa: F401 - Cần thiết để joblib unpickle TelcoFeatureEngineer


MODELS_DIR = Path("models")
FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")
MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost", "MLP_NeuralNet"]


def get_feature_names(pipeline) -> list:
    """Trích xuất tên đặc trưng sau bước ColumnTransformer."""
    col_transformer = pipeline.named_steps["preprocessor"].named_steps["column_transformer"]
    raw_names = col_transformer.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def evaluate_all():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    X_test, y_test = joblib.load(MODELS_DIR / "test_data.joblib")

    metrics_list = []
    predictions_store = {}

    # 1. Tính toán đầy đủ các độ đo trên tập Test
    for name in MODEL_NAMES:
        model = joblib.load(MODELS_DIR / f"{name}.joblib")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        predictions_store[name] = {"y_pred": y_pred, "y_prob": y_prob, "model": model}

        metrics_list.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision (Yes)": round(precision_score(y_test, y_pred), 4),
                "Recall (Yes)": round(recall_score(y_test, y_pred), 4),
                "F1-Score (Yes)": round(f1_score(y_test, y_pred), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
            }
        )

    metrics_df = pd.DataFrame(metrics_list).sort_values(by="ROC-AUC", ascending=False)
    metrics_df.to_csv(RESULTS_DIR / "test_metrics_comparison.csv", index=False)

    print("\n" + "=" * 75)
    print("BẢNG SO SÁNH KẾT QUẢ THỰC NGHIỆM TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("=" * 75)
    print(metrics_df.to_string(index=False))

    # 2. Vẽ Ma trận nhầm lẫn (Confusion Matrix) cho 4 mô hình
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name in zip(axes.flatten(), MODEL_NAMES):
        cm = confusion_matrix(y_test, predictions_store[name]["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix - {name}", fontweight="bold")
        ax.set_xlabel("Nhãn dự đoán (Predicted)")
        ax.set_ylabel("Nhãn thực tế (Actual)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_confusion_matrices.png", dpi=300)
    plt.close()

    # 3. Vẽ đường cong ROC và Precision-Recall
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name in MODEL_NAMES:
        y_prob = predictions_store[name]["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_val:.4f})")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_val = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_val:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
    ax1.set_title("Đường cong ROC (ROC Curve)", fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right")

    baseline_pr = y_test.mean()
    ax2.axhline(baseline_pr, color="k", linestyle="--", lw=1, label=f"Baseline ({baseline_pr:.2f})")
    ax2.set_title("Đường cong Precision-Recall", fontweight="bold")
    ax2.set_xlabel("Recall (Độ phủ lớp Churn)")
    ax2.set_ylabel("Precision (Độ chính xác lớp Churn)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_roc_and_pr_curves.png", dpi=300)
    plt.close()

    # 4. Phân tích điều chỉnh ngưỡng quyết định (Threshold Tuning) cho mô hình tốt nhất
    best_model_name = metrics_df.iloc[0]["Model"]
    best_prob = predictions_store[best_model_name]["y_prob"]

    thresholds = np.arange(0.1, 0.9, 0.02)
    prec_scores, rec_scores, f1_scores = [], [], []
    for t in thresholds:
        preds_t = (best_prob >= t).astype(int)
        prec_scores.append(precision_score(y_test, preds_t, zero_division=0))
        rec_scores.append(recall_score(y_test, preds_t, zero_division=0))
        f1_scores.append(f1_score(y_test, preds_t, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    optimal_thresh = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, prec_scores, label="Precision (Yes)", lw=2)
    ax.plot(thresholds, rec_scores, label="Recall (Yes)", lw=2)
    ax.plot(thresholds, f1_scores, label="F1-Score (Yes)", lw=2.5, color="#d95f02")
    ax.axvline(
        optimal_thresh,
        color="gray",
        linestyle="--",
        label=f"Ngưỡng F1 tối ưu = {optimal_thresh:.2f} (F1 = {f1_scores[best_idx]:.4f})",
    )
    ax.set_title(f"Ảnh hưởng của ngưỡng phân loại ({best_model_name})", fontweight="bold")
    ax.set_xlabel("Ngưỡng xác suất (Decision Threshold)")
    ax.set_ylabel("Giá trị độ đo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_threshold_tuning.png", dpi=300)
    plt.close()

    # 5. Giải thích mô hình bằng SHAP (hoặc Feature Importance dự phòng)
    best_pipeline = predictions_store[best_model_name]["model"]
    feature_names = get_feature_names(best_pipeline)
    X_test_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)
    classifier = best_pipeline.named_steps["classifier"]

    try:
        import shap

        if best_model_name in ["XGBoost", "RandomForest"]:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.Explainer(classifier, X_test_transformed)
            shap_values = explainer(X_test_transformed).values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_test_transformed,
            feature_names=feature_names,
            max_display=12,
            show=False,
        )
        plt.title(f"SHAP Feature Impact - {best_model_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "04_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("-> Đã xuất biểu đồ giải thích SHAP thành công.")
    except Exception as e:
        print(f"-> Không thể chạy SHAP ({e}), chuyển sang vẽ Feature Importance chuẩn...")
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = None

        if importances is not None:
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values(by="Importance", ascending=False)
                .head(12)
            )
            fig, ax = plt.subplots(figsize=(9, 5.5))
            sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
            ax.set_title(f"Top 12 đặc trưng quan trọng nhất ({best_model_name})", fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "04_feature_importance.png", dpi=300)
            plt.close()

    print(f"\nToàn bộ biểu đồ đánh giá đã lưu tại: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    evaluate_all()