import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque
from datetime import datetime
import pytz
import time
import csv
import numpy as np

# Cấu hình
SERIAL_PORT = 'COM3'
SERIAL_BAUD = 115200
CSV_FILE_NAME = "data.csv"
GRAPH_TITLE = "Real-time MAX30102 IR Signal Plot"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
Y_RANGE = (85000, 90000)
X_RANGE_SECONDS = 20
UPDATE_INTERVAL_MS = 10
DATA_BUFFER_SIZE = 2000
CSV_WRITE_INTERVAL_SECONDS = 0

# Kết nối Serial
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD)
except serial.SerialException as e:
    print(f"Lỗi kết nối Serial: {e}")
    exit()

# Mở file CSV để ghi dữ liệu
csv_file = open(CSV_FILE_NAME, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time (real)", "IR Value raw", "IR Value filtered", "Time (s)"])

# Khởi tạo ứng dụng và cửa sổ
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title=GRAPH_TITLE)
win.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

# Định nghĩa múi giờ Việt Nam
vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')

def get_vietnam_time():
    return datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")

# Tạo đồ thị
plot = win.addPlot(title="IR Signal vs Time (seconds)")
plot.setLabel('bottom', 'Time (seconds)')
plot.setLabel('left', 'IR Value')
curve_goc = plot.plot(pen='r', name='Gốc')
curve_loc = plot.plot(pen='b', name='Lọc') # Thêm đường vẽ cho tín hiệu lọc
plot.addLegend()
plot.setYRange(*Y_RANGE)

# Buffer dữ liệu
data_goc = deque([0] * DATA_BUFFER_SIZE, maxlen=DATA_BUFFER_SIZE)
data_loc = deque([0] * DATA_BUFFER_SIZE, maxlen=DATA_BUFFER_SIZE)
time_data = deque([0] * DATA_BUFFER_SIZE, maxlen=DATA_BUFFER_SIZE)

# Buffer dữ liệu miền tần số
frequency_data = deque([0] * DATA_BUFFER_SIZE, maxlen=DATA_BUFFER_SIZE)
amplitude_data = deque([0] * DATA_BUFFER_SIZE, maxlen=DATA_BUFFER_SIZE)

# Bắt đầu đếm thời gian
start_time = time.time()
last_csv_write_time = start_time

# Hàm cập nhật đồ thị
def update():
    global data_goc, data_loc, time_data, curve_goc, curve_loc, start_time, last_csv_write_time
    try:
        while ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            try:
                ir_value_goc, ir_value_loc = map(int, line.split(','))
            except ValueError:
                continue

            current_time = time.time() - start_time
            real_time = get_vietnam_time()
            data_goc.append(ir_value_goc)
            data_loc.append(ir_value_loc)
            time_data.append(current_time)

            # Ghi vào file CSV theo lô
            #if current_time - last_csv_write_time >= CSV_WRITE_INTERVAL_SECONDS:
            csv_writer.writerow([real_time, ir_value_goc, ir_value_loc, current_time])
            csv_file.flush()
            last_csv_write_time = current_time

        # Cập nhật đồ thị với trục X là thời gian
        curve_goc.setData(list(time_data), list(data_goc))
        curve_loc.setData(list(time_data), list(data_loc))
        plot.setXRange(max(time_data) - X_RANGE_SECONDS, max(time_data))

    except Exception as e:
        print(f"Lỗi: {e}")

# Cập nhật đồ thị theo khoảng thời gian
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL_MS)

# Xử lý khi đóng cửa sổ
def closeEvent(event):
    csv_file.close()
    ser.close()
    event.accept()

win.closeEvent = closeEvent

# Hiển thị cửa sổ đồ thị
win.show()
app.exec_()