import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 🔹 1. Đọc file CSV
file_path = "data/data_04042025.csv"
df = pd.read_csv(file_path)

# 🔹 2. Kiểm tra và xử lý dữ liệu
if "IR Value filtered" in df.columns and "Time (s)" in df.columns and "Label" in df.columns:
    df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
    time = np.array(df["Time (s)"])
    ir_signal = np.array(df["IR Value filtered"])
    label = np.array(df["Label"])

    # 🔹 3. Cấu hình cửa sổ trượt
    window_size = 10  # giây
    step_size = 1     # giây
    start_time = time[0]
    end_time = time[-1]
    result = []

    # 🔹 4. Duyệt qua từng cửa sổ
    for current_start in np.arange(start_time, end_time - window_size, step_size):
        current_end = current_start + window_size

        # Lọc tín hiệu trong cửa sổ
        mask = (time >= current_start) & (time <= current_end)
        time_window = time[mask]
        ir_signal_window = ir_signal[mask]
        label_window = label[mask]
        # Gán nhãn phổ biến nhất trong cửa sổ
        if len(label_window) > 0:
            label_counts = pd.Series(label_window).value_counts()
            dominant_label = label_counts.idxmax()
        else:
            dominant_label = "unknown"

        N = len(ir_signal_window)
        if N < 2:
            continue  # Bỏ qua nếu cửa sổ quá nhỏ

        # 🔹 5. Tính FFT
        dt = np.mean(np.diff(time_window))  # Chu kỳ lấy mẫu
        freqs = np.fft.rfftfreq(N, d=dt)
        fft_result = np.fft.rfft(ir_signal_window)

        amplitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        power = amplitude ** 2  # 🔹 Phổ công suất

        # 🔹 6. Tìm các đỉnh trong phổ pha
        peaks_all, _ = find_peaks(phase)
        peaks = peaks_all[phase[peaks_all] > 1] 
        troughs_all, _ = find_peaks(-phase)
        troughs = troughs_all[phase[troughs_all] < -1]
        peak_freqs = freqs[peaks]
        peak_phases = phase[peaks]
        peak_mean = np.mean(peak_phases)
        peak_std = np.std(peak_phases)

        trough_freqs = freqs[troughs]
        trough_phases = phase[troughs]
        trough_mean = np.mean(trough_phases)
        trough_std = np.std(trough_phases)

        max_amplitude = np.max(amplitude)       # 🔸 Biên độ cực đại
        total_power = np.sum(power)             # 🔸 Tổng năng lượng



        result.append([current_start, current_end, peak_mean, peak_std, trough_mean, trough_std, max_amplitude, total_power, dominant_label])
    #     # 🔹 7. Vẽ đồ thị
        # plt.figure(figsize=(12, 8))

        # Tín hiệu IR theo thời gian
        # plt.subplot(4, 1, 1)
        # plt.plot(time_window, ir_signal_window, label="IR Signal", color="b")
        # plt.title(f"IR Signal ({current_start:.1f}s - {current_end:.1f}s)")
        # plt.xlabel("Time (s)")
        # plt.ylabel("IR Value")
        # plt.grid()
        # plt.legend()

        # # Phổ biên độ
        # plt.subplot(4, 1, 2)
        # plt.plot(freqs, amplitude, label="Amplitude", color="g")
        # plt.title("Amplitude Spectrum")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Amplitude")
        # plt.grid()
        # plt.legend()

        # # 🔹 Phổ pha với đỉnh và đáy
        # plt.subplot(4, 1, 3)
        # plt.plot(freqs, phase, label="Phase", color="r")
        # plt.plot(peak_freqs, peak_phases, 'go', label="Peaks")      # Đỉnh: chấm xanh
        # plt.plot(trough_freqs, trough_phases, 'bo', label="Troughs") # Đáy: chấm xanh dương
        # plt.title("Phase Spectrum with Peaks and Troughs")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Phase (radians)")
        # plt.grid()
        # plt.legend()


        # # Phổ công suất
        # plt.subplot(4, 1, 4)
        # plt.plot(freqs, power, label="Power Spectrum", color="m")
        # plt.title("Power Spectrum")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power")
        # plt.grid()
        # plt.legend()

        # plt.tight_layout()
        # plt.show()


    fft_df = pd.DataFrame(result, columns=["Start Time (s)", "End Time (s)", "AVG Peak", "STD Peak", "AVG Trough", "STD Trough", "MAX Amplitude", "Total Power", "Label"])
    fft_df.to_csv("FFT_result.csv", index=False)



    print("✅ Đã lưu các đặc trưng vào 'FFT_result.csv'")
else:
    print("⚠️ Không tìm thấy cột 'IR Value filtered' hoặc 'Time (s)' trong file CSV.")
