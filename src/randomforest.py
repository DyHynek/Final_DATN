import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tên các thư mục chứa dữ liệu
time_domain_dir = 'Time_domain'
nonlinear_domain_dir = 'nonlinear_domain'
wavelet_dir = 'Wavelet_Transform'

# Hàm để đọc tất cả các file CSV từ một thư mục và kết hợp chúng thành một DataFrame
def load_and_combine_all(directory, prefix, drop_cols):
    all_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    df_list = []
    for file in all_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df_dropped = df.drop(columns=drop_cols, errors='ignore')
        df_list.append(df_dropped)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df
    return None

# Đọc và kết hợp dữ liệu cho từng miền
df_thoi_gian_all = load_and_combine_all(time_domain_dir, 'heart_rate_results', ['Start Time (s)', 'End Time (s)'])
df_phi_tuyen_all = load_and_combine_all(nonlinear_domain_dir, 'nonlinear_results', ['Time start', 'Time end'])
df_wavelet_all = load_and_combine_all(wavelet_dir, 'wavelet_result', ['Start Time (s)', 'End Time (s)'])

# Hàm để huấn luyện và đánh giá mô hình trên một DataFrame
def train_evaluate_domain(df, domain_name, model_filename):
    if df is not None:
        X = df.drop('Label', axis=1, errors='ignore')
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nĐánh giá mô hình cho {domain_name}:")
        print("Độ chính xác:", accuracy)
        print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))
        print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred))


         # Lưu mô hình
        joblib.dump(model, model_filename)
        print(f"✅ Đã lưu mô hình {domain_name} vào '{model_filename}'")
        return accuracy
    else:
        print(f"\nKhông có dữ liệu cho {domain_name}.")
        return None

# Huấn luyện và đánh giá cho từng miền
accuracy_thoi_gian = train_evaluate_domain(df_thoi_gian_all, "miền thời gian", "model_time_domain.pkl")
accuracy_phi_tuyen = train_evaluate_domain(df_phi_tuyen_all, "đặc trưng phi tuyến", "model_nonlinear.pkl")
accuracy_wavelet = train_evaluate_domain(df_wavelet_all, "biến đổi wavelet", "model_wavelet.pkl")

# In ra độ chính xác của từng miền
print("\nĐộ chính xác của từng miền:")
if accuracy_thoi_gian is not None:
    print(f"Miền thời gian: {accuracy_thoi_gian:.4f}")
if accuracy_phi_tuyen is not None:
    print(f"Miền phi tuyến: {accuracy_phi_tuyen:.4f}")
if accuracy_wavelet is not None:
    print(f"Biến đổi Wavelet: {accuracy_wavelet:.4f}")