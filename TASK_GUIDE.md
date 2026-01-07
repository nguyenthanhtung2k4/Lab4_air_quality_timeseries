LAB4 - PHÂN TÍCH CHẤT LƯỢNG KHÔNG KHÍ BẮCKINH

## 🎯 MỤC TIÊU TỔNG QUAN

Xây dựng 3 mô hình dự đoán chất lượng không khí từ dữ liệu 12 trạm đo tại Bắc Kinh:
- ✅ **Phân loại** mức độ ô nhiễm (Good, Moderate, Unhealthy...)
- ✅ **Hồi quy** dự đoán giá trị PM2.5 tương lai
- ✅ **Chuỗi thời gian** dự báo bằng ARIMA

---

## 📚 PHẦN 1: CHUẨN BỊ MÔI TRƯỜNG

### [ ] Bước 1.1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Packages chính:**
- `pandas`, `numpy` - Xử lý dữ liệu
- `scikit-learn` - Machine Learning
- `statsmodels` - ARIMA
- `papermill` - Tự động chạy notebooks

### [ ] Bước 1.2: Tải dữ liệu
**Option 1:** Tự động từ UCI (khuyến nghị)
```python
USE_UCIMLREPO = True
```

**Option 2:** Tải thủ công
- Download: [Beijing Multi-Site Air Quality Dataset](https://archive.ics.uci.edu/dataset/501)
- Đặt vào: `data/raw/PRSA2017_Data_20130301-20170228.zip`

### [ ] Bước 1.3: Kiểm tra cấu trúc thư mục
```
Lab4_air_quality_timeseries/
├── data/
│   ├── raw/              ← Đặt file ZIP ở đây
│   └── processed/        ← Kết quả sẽ lưu ở đây
├── notebooks/            ← Jupyter notebooks
├── src/                  ← Code modules
└── run_papermill.py      ← Script chạy tự động
```

---

## 📊 PHẦN 2: PREPROCESSING & EDA (Khám Phá Dữ Liệu)

### [ ] Bước 2.1: Chạy notebook preprocessing
```bash
jupyter notebook notebooks/preprocessing_and_eda.ipynb
```

### [ ] Bước 2.2: Hiểu các bước xử lý

#### 🔹 Load dữ liệu
- 12 trạm đo, mỗi trạm có file CSV riêng
- Gộp tất cả thành 1 DataFrame

#### 🔹 Làm sạch dữ liệu
```python
# Thay thế missing values
"NA", "N/A", "null" → np.nan

# Tạo cột datetime
datetime = year-month-day-hour

# Chuyển đổi kiểu dữ liệu
PM2.5, PM10, SO2... → float64
station, wd → object
```

#### 🔹 Tạo features thời gian
```python
hour_sin = sin(2π × hour / 24)    # Chu kỳ giờ
hour_cos = cos(2π × hour / 24)    
dow = day_of_week                  # Thứ trong tuần
month = month                      # Tháng
is_weekend = (dow >= 5)            # Cuối tuần
```

#### 🔹 Tạo lag features
**Thông số:** `LAG_HOURS = [1, 3, 24]`

**Ví dụ với PM10:**
- `PM10_lag1` = PM10 của 1 giờ trước
- `PM10_lag3` = PM10 của 3 giờ trước
- `PM10_lag24` = PM10 của 24 giờ trước

**Áp dụng cho:** PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM

### [ ] Bước 2.3: Kiểm tra output
**File:** `data/processed/cleaned.parquet`

**Cột quan trọng:**
- `datetime` - Thời gian
- `station` - Tên trạm (12 trạm)
- `PM2.5` - Bụi mịn (target chính)
- `pm25_24h` - Trung bình 24h của PM2.5
- `aqi_class` - Phân loại AQI (Good, Moderate...)
- Các lag features (`*_lag1`, `*_lag3`, `*_lag24`)

---

## 🏷️ PHẦN 3: CLASSIFICATION (Phân Loại Mức Độ Ô Nhiễm)

### [ ] Bước 3.1: Hiểu bài toán

**Input:** Các features thời tiết + lag features (KHÔNG bao gồm PM2.5 trực tiếp)
**Output:** 6 lớp AQI

| Lớp | PM2.5 Range (µg/m³) | Ý Nghĩa |
|-----|---------------------|---------|
| Good | 0.0 – 9.0 | Tốt |
| Moderate | 9.1 – 35.4 | Trung bình |
| Unhealthy for Sensitive Groups | 35.5 – 55.4 | Không tốt cho nhóm nhạy cảm |
| Unhealthy | 55.5 – 125.4 | Không tốt |
| Very Unhealthy | 125.5 – 225.4 | Rất không tốt |
| Hazardous | 225.5+ | Nguy hiểm |

### [ ] Bước 3.2: Tạo label AQI
```python
# Tính trung bình 24h của PM2.5
pm25_24h = rolling(window=24, min_periods=18).mean()

# Phân loại theo breakpoints
aqi_class = pm25_to_aqi_class(pm25_24h)
```

### [ ] Bước 3.3: Chia train/test
**Thông số:** `CUTOFF = "2017-01-01"`

```python
train = data[datetime < 2017-01-01]  # ~3.5 năm
test  = data[datetime >= 2017-01-01] # ~2 tháng
```

⚠️ **Lưu ý:** Không shuffle! (Time series data)

### [ ] Bước 3.4: Train model
**Model:** `HistGradientBoostingClassifier`

**Thông số:**
```python
max_depth = 6          # Độ sâu cây
learning_rate = 0.08   # Tốc độ học
max_iter = 250         # Số vòng lặp
random_state = 42      # Seed
```

**Features loại bỏ (tránh leakage):**
- ❌ `PM2.5` (biến gốc)
- ❌ `pm25_24h` (biến trung gian)
- ❌ `datetime` (không dùng trực tiếp)

### [ ] Bước 3.5: Chạy notebook
```bash
jupyter notebook notebooks/classification_modelling.ipynb
```

### [ ] Bước 3.6: Đánh giá kết quả
**File output:**
- `data/processed/metrics.json` - Accuracy, F1-score, confusion matrix
- `data/processed/predictions_sample.csv` - Mẫu dự đoán

**Metrics quan trọng:**
- `accuracy` - Độ chính xác tổng thể
- `f1_macro` - F1-score trung bình các lớp
- `confusion_matrix` - Ma trận nhầm lẫn 6×6

---

## 📈 PHẦN 4: REGRESSION (Hồi Quy Dự Đoán PM2.5)

### [ ] Bước 4.1: Hiểu bài toán

**Input:** Các features + lag features (BAO GỒM PM2.5 lag)
**Output:** Giá trị PM2.5 tương lai (số thực)

**Khác biệt với Classification:**
- ✅ Được dùng `PM2.5_lag1`, `PM2.5_lag3`, `PM2.5_lag24`
- ✅ Output là số liên tục, không phải lớp
- ✅ Có thông số `HORIZON` (dự đoán bao xa)

### [ ] Bước 4.2: Hiểu HORIZON

**Thông số:** `HORIZON = 1`

**Công thức:**
```python
y(t) = PM2.5(t + HORIZON)
```

**Ví dụ:**
- `HORIZON = 1`: Dự đoán PM2.5 sau 1 giờ
- `HORIZON = 6`: Dự đoán PM2.5 sau 6 giờ
- `HORIZON = 24`: Dự đoán PM2.5 sau 1 ngày

### [ ] Bước 4.3: Tạo dataset regression
```python
# Thêm lag features (bao gồm PM2.5)
LAG_HOURS = [1, 3, 24]
cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", 
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# Tạo target
y = PM2.5.shift(-HORIZON)
```

### [ ] Bước 4.4: Train model
**Model:** `HistGradientBoostingRegressor` (tương tự classifier)

### [ ] Bước 4.5: Chạy notebook
```bash
jupyter notebook notebooks/regression_modelling.ipynb
```

### [ ] Bước 4.6: Đánh giá kết quả
**File output:**
- `data/processed/regressor.joblib` - Model đã train
- `data/processed/regression_metrics.json` - MAE, RMSE, R²
- `data/processed/regression_predictions_sample.csv` - Mẫu dự đoán

**Metrics quan trọng:**
- `MAE` (Mean Absolute Error) - Sai số trung bình
- `RMSE` (Root Mean Squared Error) - Sai số bình phương
- `R²` - Hệ số xác định (càng gần 1 càng tốt)

---

## ⏱️ PHẦN 5: TIME SERIES ARIMA (Dự Báo Chuỗi Thời Gian)

### [ ] Bước 5.1: Hiểu bài toán

**Khác biệt với Regression:**
- ❌ KHÔNG dùng features khác (chỉ dùng chính chuỗi PM2.5)
- ❌ KHÔNG dùng lag features thủ công
- ✅ Dự báo "thuần túy" dựa trên pattern của chuỗi

**ARIMA = AutoRegressive Integrated Moving Average**

### [ ] Bước 5.2: Hiểu ARIMA(p, d, q)

| Tham Số | Ý Nghĩa | Ví Dụ |
|---------|---------|-------|
| **p** | AR order - Số lag của chính chuỗi | `p=2`: dùng `y(t-1)` và `y(t-2)` |
| **d** | Differencing - Số lần lấy sai phân | `d=1`: `y'(t) = y(t) - y(t-1)` |
| **q** | MA order - Số lag của sai số | `q=1`: dùng sai số `e(t-1)` |

### [ ] Bước 5.3: Chuẩn bị dữ liệu

#### 🔹 Chọn 1 trạm
**Thông số:** `STATION = "Aotizhongxin"`

**12 trạm có sẵn:**
- Aotizhongxin, Changping, Dingling, Dongsi
- Guanyuan, Gucheng, Huairou, Nongzhanguan
- Shunyi, Tiantan, Wanliu, Wanshouxigong

#### 🔹 Tạo chuỗi hourly
```python
# Lọc 1 trạm
df_station = df[df.station == "Aotizhongxin"]

# Set datetime index
series = df_station.set_index("datetime")["PM2.5"]

# Resample theo giờ
series = series.resample("H").mean()

# Điền missing
series = series.interpolate(method="time")
```

### [ ] Bước 5.4: Kiểm tra tính dừng (Stationarity)

#### 🔹 ADF Test (Augmented Dickey-Fuller)
```python
# H0: Chuỗi có unit root (không dừng)
# H1: Chuỗi dừng

if p_value < 0.05:
    print("Chuỗi dừng ✅")
else:
    print("Cần differencing ❌")
```

#### 🔹 KPSS Test
```python
# H0: Chuỗi dừng
# H1: Chuỗi không dừng

if p_value > 0.05:
    print("Chuỗi dừng ✅")
else:
    print("Cần differencing ❌")
```

### [ ] Bước 5.5: Grid Search ARIMA

**Thông số:**
```python
P_MAX = 3  # Thử p = 0, 1, 2, 3
D_MAX = 2  # Thử d = 0, 1, 2
Q_MAX = 3  # Thử q = 0, 1, 2, 3
IC = "aic" # Information Criterion (aic hoặc bic)
```

**Tổng số models thử:** `4 × 3 × 4 = 48 models`

**Chọn model tốt nhất:**
```python
best_model = model với AIC thấp nhất
```

### [ ] Bước 5.6: Chia train/test
```python
train = series[datetime < 2017-01-01]
test  = series[datetime >= 2017-01-01]
```

### [ ] Bước 5.7: Fit & Forecast
```python
# Fit ARIMA trên train
model = ARIMA(train, order=(p, d, q))
fitted = model.fit()

# Dự báo length(test) bước
forecast = fitted.forecast(steps=len(test))
```

### [ ] Bước 5.8: Chạy notebook
```bash
jupyter notebook notebooks/arima_forecasting.ipynb
```

### [ ] Bước 5.9: Đánh giá kết quả
**File output:**
- `data/processed/arima_pm25_diagnostics.json` - Thông tin model
- `data/processed/arima_pm25_forecast.csv` - Dự báo

**Nội dung diagnostics:**
- `best_order` - (p, d, q) tốt nhất
- `aic`, `bic` - Information criteria
- `adf_pvalue`, `kpss_pvalue` - Stationarity tests
- `seasonal_strength` - Độ mạnh tính mùa

---

## 🚀 PHẦN 6: CHẠY TỰ ĐỘNG TOÀN BỘ PIPELINE

### [ ] Bước 6.1: Tạo Jupyter kernel
```bash
# Tạo virtual environment
python -m venv beijing_env

# Activate
beijing_env\Scripts\activate  # Windows
source beijing_env/bin/activate  # Linux/Mac

# Cài packages
pip install -r requirements.txt

# Tạo kernel
pip install ipykernel
python -m ipykernel install --user --name=beijing_env
```

### [ ] Bước 6.2: Chạy Papermill
```bash
python run_papermill.py
```

**Pipeline sẽ chạy theo thứ tự:**
1. ✅ Preprocessing & EDA
2. ✅ Feature Preparation
3. ✅ Classification Modelling
4. ✅ Regression Modelling
5. ✅ ARIMA Forecasting

**Thời gian ước tính:** 10-30 phút (tùy máy)

### [ ] Bước 6.3: Kiểm tra kết quả
**Notebooks đã chạy:** `notebooks/runs/*_run.ipynb`

**Data outputs:** `data/processed/`
```
✅ cleaned.parquet
✅ dataset_for_clf.parquet
✅ dataset_for_regression.parquet
✅ metrics.json
✅ predictions_sample.csv
✅ regressor.joblib
✅ regression_metrics.json
✅ regression_predictions_sample.csv
✅ arima_pm25_diagnostics.json
✅ arima_pm25_forecast.csv
```

---

## 🎨 PHẦN 7: VISUALIZATION & ANALYSIS

### [ ] Bước 7.1: Vẽ biểu đồ Classification
```python
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics
with open("data/processed/metrics.json") as f:
    metrics = json.load(f)

# Confusion Matrix
cm = metrics["confusion_matrix"]
labels = metrics["labels"]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", 
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - AQI Classification")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
```

### [ ] Bước 7.2: Vẽ biểu đồ Regression
```python
import pandas as pd

# Load predictions
preds = pd.read_csv("data/processed/regression_predictions_sample.csv")

plt.figure(figsize=(14, 6))
plt.plot(preds["datetime"], preds["y_true"], label="Actual", alpha=0.7)
plt.plot(preds["datetime"], preds["y_pred"], label="Predicted", alpha=0.7)
plt.title("PM2.5 Regression - Actual vs Predicted")
plt.xlabel("Datetime")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### [ ] Bước 7.3: Vẽ biểu đồ ARIMA
```python
# Load forecast
forecast = pd.read_csv("data/processed/arima_pm25_forecast.csv")

plt.figure(figsize=(14, 6))
plt.plot(forecast["datetime"], forecast["actual"], label="Actual", alpha=0.7)
plt.plot(forecast["datetime"], forecast["forecast"], label="Forecast", alpha=0.7)
plt.fill_between(forecast["datetime"], 
                 forecast["lower_ci"], forecast["upper_ci"],
                 alpha=0.2, label="95% CI")
plt.title("ARIMA Forecast - PM2.5")
plt.xlabel("Datetime")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🔧 PHẦN 8: TUNING & EXPERIMENTS

### [ ] Thử nghiệm 1: Thay đổi LAG_HOURS
```python
# Trong run_papermill.py, thử:
LAG_HOURS = [1, 3, 6, 12, 24, 48]  # Thêm lag 6h, 12h, 48h
```

**Kỳ vọng:** Accuracy/R² tăng nhưng training time tăng

### [ ] Thử nghiệm 2: Thay đổi HORIZON
```python
# Dự đoán xa hơn
HORIZON = 6   # 6 giờ
HORIZON = 12  # 12 giờ
HORIZON = 24  # 1 ngày
```

**Kỳ vọng:** HORIZON càng lớn, sai số càng cao

### [ ] Thử nghiệm 3: Thay đổi CUTOFF
```python
# Thử các mốc khác
CUTOFF = "2016-01-01"  # Test set lớn hơn
CUTOFF = "2017-02-01"  # Test set nhỏ hơn
```

### [ ] Thử nghiệm 4: Thay đổi ARIMA parameters
```python
P_MAX = 5  # Thử nhiều AR lags hơn
Q_MAX = 5  # Thử nhiều MA lags hơn
IC = "bic" # Thử BIC thay vì AIC
```

**Lưu ý:** Grid search sẽ lâu hơn!

### [ ] Thử nghiệm 5: Thử các trạm khác
```python
# Trong arima_forecasting
STATION = "Dongsi"
STATION = "Tiantan"
# ... thử cả 12 trạm
```

**So sánh:** Trạm nào có forecast tốt nhất?

---

## 📊 PHẦN 9: SO SÁNH 3 PHƯƠNG PHÁP

### [ ] Bước 9.1: Tạo bảng so sánh

| Tiêu Chí | Classification | Regression | ARIMA |
|----------|---------------|------------|-------|
| **Input** | Features + Lags (không PM2.5) | Features + Lags (có PM2.5) | Chỉ chuỗi PM2.5 |
| **Output** | 6 lớp AQI | Giá trị PM2.5 | Giá trị PM2.5 |
| **Độ phức tạp** | Trung bình | Trung bình | Thấp |
| **Khả năng giải thích** | Cao | Cao | Thấp |
| **Dự đoán ngắn hạn** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dự đoán dài hạn** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Cần features khác** | ✅ | ✅ | ❌ |

### [ ] Bước 9.2: Khi nào dùng phương pháp nào?

**Classification:**
- ✅ Cần cảnh báo mức độ ô nhiễm (Good/Bad)
- ✅ Có nhiều features thời tiết
- ✅ Không cần giá trị chính xác

**Regression:**
- ✅ Cần giá trị PM2.5 cụ thể
- ✅ Có nhiều features thời tiết
- ✅ Dự đoán ngắn-trung hạn (1-24h)

**ARIMA:**
- ✅ Chỉ có dữ liệu PM2.5 (không có features khác)
- ✅ Dự đoán rất ngắn hạn (1-6h)
- ✅ Cần hiểu pattern chuỗi thời gian

---

## ✅ CHECKLIST HOÀN THÀNH

### Cơ Bản
- [ ] Cài đặt môi trường thành công
- [ ] Chạy được preprocessing notebook
- [ ] Chạy được classification notebook
- [ ] Chạy được regression notebook
- [ ] Chạy được ARIMA notebook
- [ ] Hiểu được ý nghĩa các thông số

### Nâng Cao
- [ ] Chạy được toàn bộ pipeline bằng Papermill
- [ ] Vẽ được biểu đồ visualization
- [ ] Thử nghiệm thay đổi LAG_HOURS
- [ ] Thử nghiệm thay đổi HORIZON
- [ ] So sánh kết quả các trạm khác nhau
- [ ] Viết được báo cáo phân tích

### Chuyên Sâu
- [ ] Hiểu được cách tránh data leakage
- [ ] Hiểu được stationarity tests (ADF, KPSS)
- [ ] Tune được hyperparameters
- [ ] So sánh được 3 phương pháp
- [ ] Đề xuất được cải tiến

---

## 🆘 TROUBLESHOOTING

### Lỗi 1: Không tìm thấy kernel "beijing_env"
```bash
# Giải pháp:
python -m ipykernel install --user --name=beijing_env
```

### Lỗi 2: Missing data file
```bash
# Kiểm tra:
ls data/raw/PRSA2017_Data_20130301-20170228.zip

# Nếu không có, set:
USE_UCIMLREPO = True
```

### Lỗi 3: ARIMA không converge
```python
# Giảm P_MAX, Q_MAX
P_MAX = 2
Q_MAX = 2
```

### Lỗi 4: Out of memory
```python
# Giảm LAG_HOURS
LAG_HOURS = [1, 24]  # Chỉ dùng 2 lags
```

---

## 📚 TÀI LIỆU THAM KHẢO

1. **Dataset:** [UCI ML Repository #501](https://archive.ics.uci.edu/dataset/501)
2. **ARIMA:** [Statsmodels Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
3. **HistGradientBoosting:** [Scikit-learn Docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
4. **AQI Breakpoints:** [EPA Air Quality Index](https://www.airnow.gov/aqi/aqi-basics/)

---

