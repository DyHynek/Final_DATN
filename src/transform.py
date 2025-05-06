import numpy as np
import pandas as pd

# Đọc dữ liệu
file_path = "data/data_04042025.csv"
df = pd.read_csv(file_path)

# Kiểm tra cột và xử lý
if "Time (s)" in df.columns and "IR Value raw" in df.columns:
    df = df.drop_duplicates(subset=["Time (s)"]).sort_values(by=["Time (s)"])
    time = np.array(df["Time (s)"])
    ir_raw = np.array(df["IR Value raw"], dtype=np.int16)

    # Bộ lọc DC
    def dc_filter(signal):
        ir_filtered = []
        prev_w = 0
        for x in signal:
            prev_w += (((int(x) << 15) - prev_w) >> 4)
            y = prev_w >> 15
            ir_filtered.append(x - y)
        return np.array(ir_filtered)

    ir_filtered = dc_filter(ir_raw)
    df["IR Value filtered"] = ir_filtered

    # Lưu kết quả vào file CSV
    df.to_csv("filtered_output.csv", index=False)
