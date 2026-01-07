import os
import papermill as pm # type: ignore

# Chỉ chạy ARIMA forecasting (nhanh hơn)
os.makedirs("notebooks/runs", exist_ok=True)

KERNEL = "beijing_env"

print("🚀 Bắt đầu chạy ARIMA forecasting...")
print("=" * 60)
print("Tham số thí nghiệm:")
print("  - Trạm: Dongsi (thay đổi từ Aotizhongxin)")
print("  - P_MAX: 2, Q_MAX: 2, D_MAX: 2 (nhanh hơn)")
print("  - IC: BIC (thay đổi từ AIC)")
print("=" * 60)

# --- Time-series forecasting with ARIMA only ---
pm.execute_notebook(
    "notebooks/arima_forecasting.ipynb",
    "notebooks/runs/arima_forecasting_run.ipynb",
    parameters=dict(
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        STATION="Dongsi",  # Thay đổi trạm để thí nghiệm
        VALUE_COL="PM2.5",
        CUTOFF="2017-01-01",
        P_MAX=2,  # Giảm xuống để chạy nhanh hơn (9 models thay vì 48)
        Q_MAX=2,  # Giảm xuống để chạy nhanh hơn
        D_MAX=2,
        IC="bic",  # Thử BIC thay vì AIC
        ARTIFACTS_PREFIX="arima_dongsi_experiment",  # Đổi tên output
    ),
    language="python",
    kernel_name=KERNEL,
)

print("\n✅ Đã chạy xong ARIMA forecasting!")
print("\n📊 Kết quả được lưu tại:")
print("  - data/processed/arima_dongsi_experiment_diagnostics.json")
print("  - data/processed/arima_dongsi_experiment_forecast.csv")
