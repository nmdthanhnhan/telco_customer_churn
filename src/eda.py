from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = Path("results/eda")


def run_eda():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    df = pd.read_csv(DATA_PATH)
    print("=" * 60)
    print("1. THÔNG TIN TỔNG QUAN TẬP DỮ LIỆU")
    print("=" * 60)
    print(f"Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"Số dòng trùng lặp: {df.duplicated().sum()}")

    # Kiểm tra giá trị khuyết ẩn trong TotalCharges
    total_charges_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = total_charges_num.isna().sum()
    print(f"Số giá trị khuyết ẩn (' ') trong cột TotalCharges: {missing_tc}")
    print(
        df.loc[total_charges_num.isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
    )

    print("\n" + "=" * 60)
    print("2. PHÂN BỐ BIẾN MỤC TIÊU (CHURN)")
    print("=" * 60)
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    for label in churn_counts.index:
        print(f"Churn = {label}: {churn_counts[label]} mẫu ({churn_pct[label]:.2f}%)")

    # Biểu đồ 1: Phân bố lớp mục tiêu
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = sns.countplot(
        data=df,
        x="Churn",
        hue="Churn",
        palette=["#2b5c8f", "#d95f02"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Phân bố biến mục tiêu (Churn)", fontweight="bold")
    ax.set_xlabel("Trạng thái rời bỏ dịch vụ (Churn)")
    ax.set_ylabel("Số lượng khách hàng")
    for container in bars.containers:
        ax.bar_label(
            container,
            labels=[f"{int(v)} ({v/len(df)*100:.1f}%)" for v in container.datavalues],
            padding=3,
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_churn_distribution.png", dpi=300)
    plt.close()

    # Biểu đồ 2: Phân bố các biến số theo nhóm Churn
    df["TotalCharges_num"] = total_charges_num.fillna(0.0)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges_num"]
    titles = ["Thời gian gắn bó (tenure - tháng)", "Cước hàng tháng (MonthlyCharges)", "Tổng cước (TotalCharges)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, col, title in zip(axes, num_cols, titles):
        sns.kdeplot(
            data=df,
            x=col,
            hue="Churn",
            fill=True,
            common_norm=False,
            palette=["#2b5c8f", "#d95f02"],
            alpha=0.45,
            ax=ax,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Mật độ xác suất")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_numerical_distributions.png", dpi=300)
    plt.close()

    # Biểu đồ 3: Tỷ lệ Churn theo các biến phân loại quan trọng
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), cat_cols):
        prop_df = (
            df.groupby(col)["Churn"]
            .value_counts(normalize=True)
            .mul(100)
            .rename("Percentage")
            .reset_index()
        )
        sns.barplot(
            data=prop_df[prop_df["Churn"] == "Yes"],
            x=col,
            y="Percentage",
            hue=col,
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Tỷ lệ Churn (%) theo {col}", fontweight="bold")
        ax.set_ylabel("Tỷ lệ rời bỏ (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 60)
        ax.tick_params(axis="x", rotation=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_categorical_churn_rates.png", dpi=300)
    plt.close()

    # Biểu đồ 4: Ma trận tương quan giữa các biến số
    corr_df = df[["tenure", "MonthlyCharges", "TotalCharges_num"]].copy()
    corr_df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Ma trận tương quan các biến định lượng", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"\nĐã lưu toàn bộ biểu đồ EDA tại thư mục: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()