# Bài tập lớn Học máy (CO3117) — Dự đoán Khách hàng Rời bỏ Dịch vụ (Telco Customer Churn)

## 1. Giới thiệu bài toán
Dự án xây dựng quy trình học máy hoàn chỉnh (*end-to-end Machine Learning pipeline*) nhằm dự đoán khả năng khách hàng ngừng sử dụng dịch vụ viễn thông dựa trên dữ liệu nhân khẩu học, gói dịch vụ đăng ký và lịch sử thanh toán.

- **Tập dữ liệu:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7.043 mẫu, 21 thuộc tính).
- **Bài toán:** Phân loại nhị phân (*Binary Classification*) trên dữ liệu mất cân bằng (`Churn = Yes` chiếm 26,54%).

## 2. Cấu trúc thư mục

2312435/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── preprocess.py    # Pipeline làm sạch, Feature Engineering & Encoding
│   ├── eda.py           # Phân tích khám phá dữ liệu & xuất biểu đồ EDA
│   ├── train.py         # Huấn luyện & tối ưu tham số 4 mô hình (Stratified 5-Fold CV)
│   └── evaluate.py      # Đánh giá trên tập Test, vẽ ROC/PR Curve, Threshold & SHAP
├── models/              # Lưu trữ mô hình tốt nhất (.joblib) sau huấn luyện
├── results/             # Bảng kết quả (.csv) và biểu đồ độ phân giải cao (.png)
├── report/              # Báo cáo tổng kết (PDF)
├── slides/              
├── Dockerfile           
├── requirements.txt     
└── README.md            


## 3. Hướng dẫn cài đặt môi trường

- **Tạo và kích hoạt môi trường ảo**
python -m venv venv
source venv/bin/activate      # Trên Linux/macOS
venv\Scripts\activate         # Trên Windows

- **Cài đặt thư viện**
pip install -r requirements.txt

## 4. Hướng dẫn chạy thực nghiệm từng bước

**Bước 1 — Phân tích khám phá dữ liệu (EDA):**

python src/eda.py

Kết quả: In thống kê dữ liệu khuyết ẩn và xuất 4 biểu đồ phân tích vào results/eda/.

**Bước 2 — Huấn luyện và tối ưu tham số mô hình:**

python src/train.py

Kết quả: Thực hiện RandomizedSearchCV (5-Fold Stratified) cho LogisticRegression, RandomForest, XGBoost và MLP_NeuralNet; lưu trọng số tại models/ và kết quả CV tại results/cv_results.csv.

**Bước 3 — Đánh giá và giải thích mô hình trên tập Test:**

python src/evaluate.py

Kết quả: Xuất bảng so sánh 6 độ đo (results/test_metrics_comparison.csv) cùng các biểu đồ Confusion Matrix, ROC/PR Curve, Threshold Tuning và SHAP Summary vào results/figures/.

**Chạy bằng Docker**

docker build -t telco-churn-ml .
docker run --rm -v "$(pwd)/results:/app/results" -v "$(pwd)/models:/app/models" telco-churn-ml