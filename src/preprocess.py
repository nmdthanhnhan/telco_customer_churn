import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "AvgMonthlySpend",
    "ChargeRatio",
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Làm sạch dữ liệu thô và tạo các đặc trưng nghiệp vụ cho bài toán Telco Churn."""

    def __init__(self):
        self.internet_service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Loại bỏ cột định danh nếu còn tồn tại
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # 2. Xử lý TotalCharges: chuyển sang kiểu số và điền 0 cho 11 khách hàng mới (tenure = 0)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # 3. Chuẩn hóa SeniorCitizen về dạng phân loại 'Yes'/'No' để mã hóa nhất quán
        if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
            df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

        # 4. Giảm dư thừa nhãn: gộp 'No internet service' và 'No phone service' thành 'No'
        for col in self.internet_service_cols:
            if col in df.columns:
                df[col] = df[col].replace({"No internet service": "No"})

        if "MultipleLines" in df.columns:
            df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

        # 5. Feature Engineering
        # Tổng số dịch vụ cộng thêm khách hàng đang sử dụng (0 -> 6)
        df["TotalServices"] = (df[self.internet_service_cols] == "Yes").sum(axis=1)

        # Chi tiêu trung bình thực tế mỗi tháng
        safe_tenure = np.where(df["tenure"] == 0, 1, df["tenure"])
        df["AvgMonthlySpend"] = df["TotalCharges"] / safe_tenure

        # Tỷ lệ cước tích lũy thực tế so với mức cước kỳ vọng (tenure * MonthlyCharges)
        expected_total = df["tenure"] * df["MonthlyCharges"]
        df["ChargeRatio"] = np.where(
            df["tenure"] == 0,
            1.0,
            df["TotalCharges"] / (expected_total + 1e-5),
        )

        return df[NUMERICAL_COLS + CATEGORICAL_COLS]


def build_preprocessing_pipeline() -> Pipeline:
    """Xây dựng pipeline tiền xử lý hoàn chỉnh (Feature Engineering + Scaling + Encoding)."""
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
    return pipeline


def load_and_split_data(
    csv_path: str, test_size: float = 0.2, random_state: int = 42
):
    """Đọc dữ liệu gốc và chia tập Train/Test theo tỷ lệ phân tầng (Stratified Split)."""
    df = pd.read_csv(csv_path)

    # Mã hóa biến mục tiêu Churn: Yes -> 1, No -> 0
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test