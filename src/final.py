import serial
import threading
import time
import csv
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import pywt
import joblib
from scipy.signal import find_peaks
from sklearn.feature_selection import mutual_info_regression
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kurtosis



SERIAL_PORT = 'COM3'
SERIAL_BAUD = 115200
CSV_FILE_NAME = "data.csv"

TIME_FEATURE_FILE_NAME = "heart_rate_results.csv"
WAVELET_FEATURE_FILE_NAME = "wavelet_results.csv"
NONLINEAR_FEATURE_FILE_NAME = "nonlinear_results.csv"

MODEL_TIME = "model_time_domain.pkl"
MODEL_NONLINEAR = "model_nonlinear.pkl"
MODEL_WAVELET = "model_wavelet.pkl"

OUTPUT_FILE = "final_prediction.csv"

acc_time = 0.9437
acc_wavelet = 0.6905
acc_nonlinear = 0.8783

# Định nghĩa múi giờ Việt Nam
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
def get_vietnam_time():
    return datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")

# Ghi dữ liệu từ Serial vào CSV
def serial_reader():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
        except serial.SerialException as e:
            print(f"Lỗi kết nối Serial: {e}")
            return

        with open(CSV_FILE_NAME, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Time (real)", "IR Value raw", "IR Value filtered", "Time (s)"])

            print("🟢 Bắt đầu thu thập dữ liệu (bỏ 5 giây đầu, ghi 60 giây / nghỉ 10 giây)... Nhấn Ctrl+C để thoát.")
            try:
                total_start = time.time()
                print("⏳ Bỏ qua 5 giây đầu để ổn định tín hiệu...")
                
                # Đợi 5 giây đầu không ghi dữ liệu
                while time.time() - total_start < 5:
                    if ser.in_waiting:
                        ser.readline()  # Đọc bỏ để không backlog

                print("📡 Bắt đầu ghi dữ liệu trong 60 giây...")
                record_start = time.time()
                while time.time() - record_start < 60:
                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8').strip()
                        try:
                            ir_raw, ir_filtered = map(int, line.split(','))
                            if ir_raw < 50000:
                                print("Đặt tay vào cảm biến!")
                        except ValueError:
                            continue
                        elapsed_time = time.time() - record_start
                        csv_writer.writerow([get_vietnam_time(), ir_raw, ir_filtered, elapsed_time])
                        csv_file.flush()

                print("⏸️ Nghỉ 10 giây...")
                time.sleep(10)

            except KeyboardInterrupt:
                print("⛔ Dừng ghi dữ liệu.")
            finally:
                ser.close()


# Phân tích dữ liệu định kỳ mỗi 10 giây
# Phân tích dữ liệu sau 60s và lặp lại mỗi 10s
def feature_time_extractor():
    while True:
        print("⏳ [Time] Đang đợi 60 giây để thu đủ dữ liệu...")
        time.sleep(65)  # Đợi 60 giây cho quá trình đo

        try:
            df = pd.read_csv(CSV_FILE_NAME)

            # Kiểm tra cột cần thiết
            if {"Time (s)", "IR Value filtered"}.issubset(df.columns):
                # Loại bỏ trùng và sắp xếp
                df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
                time_arr = np.array(df["Time (s)"])
                ir_signal = np.array(df["IR Value filtered"])
                

                # if len(time_arr) < 2:
                #     print("⚠️ Dữ liệu quá ít, bỏ qua lần này.")
                #     continue

                # Tìm đỉnh và đáy
                dt = np.mean(np.diff(time_arr))  # sampling interval
                peaks, _ = find_peaks(ir_signal, height=np.mean(ir_signal), distance=0.5 / dt)
                valleys, _ = find_peaks(-ir_signal, height=-np.mean(ir_signal), distance=0.3 / dt)

                # Rising time (valley -> peak)
                rising_times = []
                for v in valleys:
                    future_peaks = peaks[peaks > v]
                    if len(future_peaks) == 0:
                        continue
                    p = future_peaks[0]
                    rising_times.append(time_arr[p] - time_arr[v])

                # Tính đặc trưng
                if len(peaks) >= 2:
                    peak_times = time_arr[peaks]
                    intervals = np.diff(peak_times)
                    avg_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                else:
                    avg_interval = std_interval = np.nan

                if len(peaks) > 0 and len(valleys) > 0:
                    amplitude = np.mean(ir_signal[peaks]) - np.mean(ir_signal[valleys])
                else:
                    amplitude = np.nan

                avg_rising_time = np.mean(rising_times) if rising_times else np.nan
                num_beats = len(peaks)
                bpm = (num_beats / 60) * 60  # do đã đo đúng 60s

                # Ghi kết quả
        
                result = pd.DataFrame([[bpm, avg_interval, std_interval, amplitude, avg_rising_time]],
                                      columns=["BPM", "AVG Interval (s)", "STD Interval (s)", "Amplitude", "Time V2P (s)"])
                result.to_csv(TIME_FEATURE_FILE_NAME, index=False)
                print(f"✅ Đã lưu các đặc trưng vào '{TIME_FEATURE_FILE_NAME}'")
                time.sleep(10)

        except Exception as e:
            print(f"❌ Lỗi xử lý feature: {e}")


def wavelet_feature_extractor():
    
    while True:
        print("⏳ [Wavelet] Đang đợi 60 giây trước khi bắt đầu tính đặc trưng biến đổi wavelet...")
        time.sleep(65)

        
        try:
            df = pd.read_csv(CSV_FILE_NAME)

            if {"Time (s)", "IR Value filtered"}.issubset(df.columns):
                df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
                time_arr = np.array(df["Time (s)"])
                ir_signal = np.array(df["IR Value filtered"])
                result = []
                
           

                
                # Biến đổi wavelet mức 4 với coif5
                coeffs = pywt.wavedec(ir_signal, wavelet='coif5', level=4)
                A4, D4, D3, D2, D1 = coeffs  # Giải nén hệ số (ngược lại với thứ tự trả về)

                # Tính kurtosis
                kurt_D1 = kurtosis(D1)
                kurt_D2 = kurtosis(D2)
                kurt_D3 = kurtosis(D3)
                kurt_A4 = kurtosis(A4)

                result.append([kurt_D1, kurt_D2, kurt_D3, kurt_A4])
                result_df = pd.DataFrame(result, columns=["Kurtosis D1", "Kurtosis D2", "Kurtosis D3", "Kurtosis A4"])
                result_df.to_csv(WAVELET_FEATURE_FILE_NAME, index=False)
                print("✅ Đã lưu các đặc trưng vào 'wavelet_results.csv'")
                time.sleep(10)
        except Exception as e:
            print(f"❌ [Wavelet] Lỗi xử lý Wavelet: {e}")

    #time.sleep(10)  # Chờ 10 giây trước khi xử lý lại

def calculate_fnn(signal, delay, max_dim, Rtol=10.0, Atol=None):
    if Atol is None:
        Atol = 2.0 * np.std(signal)

    N = len(signal)
    fnn_percentages = []

    for d in range(1, max_dim + 1):
        M = N - (d + 1) * delay
        if M <= 0:
            break

        embedded_d = np.array([signal[i:i + d * delay:delay] for i in range(M)])
        embedded_d1 = np.array([signal[i:i + (d + 1) * delay:delay] for i in range(M)])

        false_nearest = 0
        for i in range(M):
            dists = np.linalg.norm(embedded_d - embedded_d[i], axis=1)
            dists[i] = np.inf  # bỏ chính nó
            nearest_idx = np.argmin(dists)

            dist_d = dists[nearest_idx]
            dist_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[nearest_idx])

            if dist_d == 0:
                continue

            if dist_d1 / dist_d > Rtol or abs(signal[i + d * delay] - signal[nearest_idx + d * delay]) > Atol:
                false_nearest += 1

        fnn_percentages.append(false_nearest / M * 100)
    return fnn_percentages

def estimate_delay_ami(signal, max_lag):
    mi_values = []
    for lag in range(1, max_lag + 1):
        X = signal[:-lag].reshape(-1,1)
        Y = signal[lag:]
        mi = mutual_info_regression(X, Y, discrete_features=False)
        mi_values.append(mi[0])
    optimal_lag = np.argmin(mi_values) + 1 if mi_values else 1
    ami_value = mi_values[optimal_lag - 1]
    return optimal_lag, ami_value

def plot_fnn(fnn_vals):
    # Tìm minimum cục bộ đầu tiên nếu có
    for i in range(1, len(fnn_vals) - 1):
        if fnn_vals[i] < fnn_vals[i - 1] and fnn_vals[i] < fnn_vals[i + 1]:
            return fnn_vals[i]
    # fallback: chọn giá trị nhỏ hơn 1% đầu tiên, nếu có
    for i, val in enumerate(fnn_vals):
        if val < 1:
            return val
    return fnn_vals[0]

def reconstruct_phase_space(signal, delay, embedding_dimension):
    n_points = len(signal) - (embedding_dimension - 1) * delay
    return np.array([signal[i:i + embedding_dimension * delay:delay] for i in range(n_points)])

def calculate_lyapunov_exponent(phase_space, k=10):
    nbrs = NearestNeighbors(n_neighbors=2).fit(phase_space)
    distances, indices = nbrs.kneighbors(phase_space)
    divergence = []
    for i in range(1, len(phase_space)):
        d0 = distances[i, 1]
        if i + k < len(phase_space) and indices[i, 1] + k < len(phase_space):
            d1 = norm(phase_space[i + k] - phase_space[indices[i, 1] + k])
            if d0 > 0 and d1 > 0:
                divergence.append(np.log(d1 / d0))
    return np.mean(divergence) / k if divergence else 0

def calculate_attractor_reconstruction_error(phase_space, delay):
    N = len(phase_space)
    if N < 2:
        return np.nan
    indices = NearestNeighbors(n_neighbors=2).fit(phase_space).kneighbors(phase_space, return_distance=False)
    errors = []
    for i in range(N - delay):
        neighbor_idx = indices[i, 1]
        if neighbor_idx + delay < N:
            original_next = phase_space[i + delay]
            predicted_next = phase_space[neighbor_idx + delay]
            errors.append(norm(original_next - predicted_next))
    return np.mean(errors) if errors else np.nan

def calculate_dfa(signal, scales=None):
    if scales is None:
        scales = [10, 20, 40, 80, 100]
    flucts = []
    for scale in scales:
        segments = len(signal) // scale
        reshaped = np.reshape(signal[:segments * scale], (segments, scale))
        F = []
        for segment in reshaped:
            x = np.arange(scale)
            trend = np.polyfit(x, segment, 1)
            detrended = segment - np.polyval(trend, x)
            F.append(np.sqrt(np.mean(detrended ** 2)))
        flucts.append(np.mean(F))
    log_scales = np.log(scales)
    log_flucts = np.log(flucts)
    coeffs = np.polyfit(log_scales, log_flucts, 1)
    return coeffs[0]  # Slope = DFA exponent

def run_nonlinear_analysis():
    
    while True:
        print("⏳ [Nonlinear] Đang đợi 60 giây trước khi bắt đầu tính đặc trưng phi tuyến...")
        time.sleep(65)
        
        try:
            df = pd.read_csv(CSV_FILE_NAME)
            if "Time (s)" in df.columns and "IR Value filtered" in df.columns:
                df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
                times = np.array(df["Time (s)"])
                ir_signal = np.array(df["IR Value filtered"])
               


                nonlinear_results = []


                optimal_delay, amivalue = estimate_delay_ami(ir_signal, max_lag=50)
                fnn_vals = calculate_fnn(ir_signal, delay=optimal_delay, max_dim=10)
                fnn = plot_fnn(fnn_vals)
                embedding_dimension = next((d + 1 for d, val in enumerate(fnn_vals) if val < 1), 3)
                phase_space = reconstruct_phase_space(ir_signal, optimal_delay, embedding_dimension)
                lyap = calculate_lyapunov_exponent(phase_space)
                reconstruction_error = calculate_attractor_reconstruction_error(phase_space, delay=optimal_delay)
                dfa = calculate_dfa(ir_signal)
                nonlinear_results.append([amivalue, fnn, lyap, reconstruction_error, dfa])
                nonlinear_df = pd.DataFrame(nonlinear_results, columns=["AMI", "FNN", "Lyapunov", "ARE", "DFA"])
                nonlinear_df.to_csv(NONLINEAR_FEATURE_FILE_NAME, index=False)
                print("✅ Đã lưu các đặc trưng vào 'nonlinear_results.csv'")
                time.sleep(10)
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file '{CSV_FILE_NAME}'")
        except Exception as e:
            print(f"⚠️ Lỗi không mong muốn: {e}")

def predic_with_model():
    while True:
        print("⏳ [Prediction] Đang đợi 60 giây trước khi chạy dự đoán...")
        time.sleep(70)
        try: 
            df_time = pd.read_csv(TIME_FEATURE_FILE_NAME)
            df_wavelet = pd.read_csv(WAVELET_FEATURE_FILE_NAME)
            df_nonlinear = pd.read_csv(NONLINEAR_FEATURE_FILE_NAME)
            
            if df_time.empty or df_wavelet.empty or df_nonlinear.empty:
                print("⚠️ Một trong các file đặc trưng bị rỗng, bỏ qua lần này.")
                continue

            model_time = joblib.load(MODEL_TIME)
            model_wavelet = joblib.load(MODEL_WAVELET)
            model_nonlinear = joblib.load(MODEL_NONLINEAR)

            pred_time = model_time.predict(df_time)[0]
            pred_wavelet = model_wavelet.predict(df_wavelet)[0]
            pred_nonlinear = model_nonlinear.predict(df_nonlinear)[0]

            weight_sum = pred_time*acc_time + pred_nonlinear*acc_nonlinear + pred_wavelet*acc_wavelet
            threshold = (acc_wavelet + acc_nonlinear + acc_time)/2

            final_pred = (weight_sum >= threshold).astype(int)

            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([get_vietnam_time(), pred_time, pred_wavelet, pred_nonlinear, final_pred])

            print(f"✅ [Prediction] Dự đoán: Time = {pred_time:.2f}, Wavelet = {pred_wavelet:.2f}, Nonlinear = {pred_nonlinear:.2f}, Final={final_pred:.2f}")
            time.sleep(5)


        except Exception as e:
            print(f"❌ [Prediction] Lỗi dự đoán: {e}")



# Khởi chạy song song
t1 = threading.Thread(target=serial_reader)
t2 = threading.Thread(target=feature_time_extractor)
t3 = threading.Thread(target=wavelet_feature_extractor)
t4 = threading.Thread(target=run_nonlinear_analysis)
t5 = threading.Thread(target=predic_with_model)


t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
