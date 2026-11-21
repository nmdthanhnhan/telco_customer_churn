import os
import pandas as pd

def load_telco_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại '{filepath}'. "
            "Hãy tải 'WA_Fn-UseC_-Telco-Customer-Churn.csv' từ Kaggle và đặt vào thư mục data/."
        )
    
    df = pd.read_csv(filepath)
    
    # Loại bỏ cột ID không có ý nghĩa khái quát hóa
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # Chuyển TotalCharges sang dạng số; các ô chứa khoảng trắng " " thành NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # SeniorCitizen đang ở dạng 0/1, chuyển sang chuỗi để xử lý đồng nhất với các biến phân loại khác
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    # Mã hóa biến mục tiêu: Yes -> 1 (Rời bỏ), No -> 0 (Ở lại)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [col for col in X.columns if col not in num_cols]
    
    return X, y, num_cols, cat_cols