import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from scipy.signal import detrend
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn.feature_selection import mutual_info_regression

# === Improved AMI Estimation ===
def estimate_delay_ami(signal, max_lag):
    mi_values = []
    for lag in range(1, max_lag + 1):
        X = signal[:-lag].reshape(-1, 1)
        Y = signal[lag:]
        mi = mutual_info_regression(X, Y, discrete_features=False)
        mi_values.append(mi[0])

    optimal_lag = np.argmin(mi_values) + 1 if mi_values else 1
    ami_value = mi_values[optimal_lag - 1]
    return optimal_lag, ami_value

# === FNN Estimation ===
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


def plot_fnn(fnn_vals):
    # Tìm minimum cục bộ đầu tiên nếu có
    for i in range(1, len(fnn_vals) - 1):
        if fnn_vals[i] < fnn_vals[i - 1] and fnn_vals[i] < fnn_vals[i + 1]:
            return fnn_vals[i]
    # fallback: chọn giá trị nhỏ hơn 1% đầu tiên, nếu có
    for i, val in enumerate(fnn_vals):
        if val < 1:
            return val
    return fnn_vals[0]  # fallback nếu không có giá trị phù hợp


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

def plot_attractor_3d(attractor_3d, title="3D Attractor"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(attractor_3d[:, 0], attractor_3d[:, 1], attractor_3d[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()

def projection_on_plane(points_3d, normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    projected_points_2d = []
    for point in points_3d:
        point = np.asarray(point).flatten()
        projection = point - np.dot(point, normal_vector) * normal_vector
        projected_points_2d.append(projection[:2])
    return np.array(projected_points_2d)

def plot_attractor_2d_projection(attractor_2d, title="2D Projection", xlabel="v'", ylabel="w'"):
    plt.figure(figsize=(8, 6))
    plt.plot(attractor_2d[:, 0], attractor_2d[:, 1], lw=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_density_2d(attractor_2d, title="2D Density Plot", xlabel="v'", ylabel="w'"):
    x = attractor_2d[:, 0]
    y = attractor_2d[:, 1]
    k = gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.viridis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    file_path = "data/data_Cong_08052025.csv"
    try:
        df = pd.read_csv(file_path)
        if "Time (s)" in df.columns and "IR Value filtered" in df.columns:
            df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
            time = np.array(df["Time (s)"])
            ir_signal = np.array(df["IR Value filtered"])
            fs = 1 / np.mean(np.diff(time))
            label = np.array(df["Label"])

            window_size = int(60 * fs)
            stride = int(1 * fs)
            nonlinear_results = []

            for i in range(0, len(ir_signal) - window_size + 1, stride):
                windowed_signal = ir_signal[i:i + window_size]
                windowed_time = time[i:i + window_size]
                time_start = time[i]
                time_end = time[i + window_size - 1]
                window_label = label[i:i + window_size]
                label_counter = Counter(window_label)
                most_commom_label, count = label_counter.most_common(1)[0]

                if len(windowed_signal) < 60:
                    continue
                try:
                    optimal_delay, amivalue = estimate_delay_ami(windowed_signal, max_lag=50)
                    fnn_vals = calculate_fnn(windowed_signal, delay=optimal_delay, max_dim=10)
                    fnn = plot_fnn(fnn_vals)
                    embedding_dimension = next((d + 1 for d, val in enumerate(fnn_vals) if val < 1), 3)

                    # attractor_3d = reconstruct_phase_space(windowed_signal, optimal_delay, 3)

                    # plt.figure(figsize=(8, 4))
                    # plt.plot(windowed_time, windowed_signal)
                    # plt.title(f"PPG Signal ({time[i]:.2f}s)")
                    # plt.xlabel("Time (s)")
                    # plt.ylabel("PPG")
                    # plt.show()

                    # plot_attractor_3d(attractor_3d, title=f"3D Attractor ({time[i]:.2f}s)")

                    # normal_vector = np.array([1, 1, 1])
                    # attractor_2d = projection_on_plane(attractor_3d, normal_vector)

                    # plot_attractor_2d_projection(attractor_2d, title=f"2D Attractor ({time[i]:.2f}s)")
                    # plot_density_2d(attractor_2d, title=f"2D Density ({time[i]:.2f}s)")

                    phase_space = reconstruct_phase_space(windowed_signal, optimal_delay, embedding_dimension)
                    lyap = calculate_lyapunov_exponent(phase_space)
                    reconstruction_error = calculate_attractor_reconstruction_error(phase_space, delay=optimal_delay)
                    dfa = calculate_dfa(windowed_signal)
                    nonlinear_results.append([time_start, time_end, amivalue, fnn, lyap, reconstruction_error, dfa, most_commom_label])
                    nonlinear_df = pd.DataFrame(nonlinear_results, columns=["Time start", "Time end", "AMI", "FNN", "Lyapunov", "ARE", "DFA", "Label"])
                    nonlinear_df.to_csv("nonlinear_results.csv", index=False)
                
                except Exception as e:
                    print(f"Lỗi tại {time[i]:.2f}s: {e}")

            
            print("✅ Đã lưu các đặc trưng vào 'nonlinear_results.csv'")
        else:
            print("⚠️ Cột dữ liệu không hợp lệ!")
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file '{file_path}'")
    except Exception as e:
        print(f"⚠️ Lỗi không mong muốn: {e}")