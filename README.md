# Phân tích dữ liệu và xây dựng các mô hình học máy với Python

Project về việc xây dựng một số mô hình học máy có giám sát (supervised learning) để phân tích và dự đoán giá. Các mô hình được xây dựng trên các bộ dữ liệu trên Kaggle, tập trung vào việc xử lý dữ liệu, so sánh hiệu suất mô hình và tối ưu hóa tham số.

## Mô hình và bài toán đã thực hiện

### Linear Regression
- **Bài toán**: Dự đoán giá xe bán lại (resale car price)
- **Mô tả**: Sử dụng mô hình hồi quy tuyến tính đơn biến và đa biến để dự đoán giá xe dựa trên các yếu tố như: năm sản xuất, loại nhiên liệu, hộp số, số km đã đi (mileage),...
- **Kết quả**: Mô hình giúp xác định các yếu tố ảnh hưởng lớn nhất đến giá xe và đưa ra định giá chính xác hơn cho từng loại xe.

### Decision Tree & Random Forest
- **Bài toán**: Dự đoán giá phòng nghỉ ở Singapore
- **Mô tả**:
  - Áp dụng Decision Tree & Random Forest để dự đoán giá thuê phòng Airbnb dựa trên các đặc trưng như: loại phòng, vị trí (latitude/longitude), số ngày khả dụng, v.v.
  - Thực hiện huấn luyện mô hình qua nhiều giai đoạn: trước & sau xử lý outlier, tối ưu tham số bằng Grid Search và Random Search.
- **Kết quả**: So sánh độ chính xác và sai số giữa các mô hình, Random Forest cho kết quả tốt nhất sau khi tối ưu.
## Kỹ thuật sử dụng

- Tiền xử lý dữ liệu: xử lý missing values, outliers, encoding, scaling
- Đánh giá mô hình: RMSE, R², confusion matrix, accuracy, precision, recall
- Tối ưu mô hình: GridSearchCV, RandomizedSearchCV
- Cross-validation: sử dụng kỹ thuật k-Fold

## Công cụ và thư viện

- Python: `scikit-learn`, `pandas`, `matplotlib`, `seaborn`
