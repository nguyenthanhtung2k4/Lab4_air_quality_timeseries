# Beijing Multi-Site Air Quality — Classification + Regression + Time Series (ARIMA)

Phân tích dữ liệu chất lượng không khí **Beijing Multi-Site Air Quality (12 stations)** để xây dựng một pipeline hoàn chỉnh gồm:

- **Phân lớp mức độ ô nhiễm (AQI level)**: tạo nhãn từ **PM2.5 rolling 24h**, nhưng **KHÔNG dùng PM2.5** trong tập đặc trưng đầu vào (tránh leakage).
- **Hồi quy (Regression)**: dự đoán **PM2.5 tương lai** theo horizon (ví dụ t+1, t+24…).
- **Chuỗi thời gian (Time Series)**: phân tích đặc điểm dữ liệu time series “đúng bài giảng” và dự báo **chỉ dùng ARIMA** (statsmodels).

Project triển khai theo pipeline notebook → module hoá trong `src/` → tự động chạy bằng **Papermill** để phục vụ giảng dạy & demo ra quyết định chọn mô hình.

---


## 
