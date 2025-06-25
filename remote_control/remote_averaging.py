import win32com.client
import numpy as np
import matplotlib.pyplot as plt
import time

# オシロと接続
scope = win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") # same for every scope
scope.MakeConnection("IP:192.168.0.100") #this is just an example

def get_waveform():
    scope.WriteString("COMM_HEADER OFF", 1)
    scope.WriteString("WAVEFORM_SETUP SP,0,NP,0,FP,0", 1)
    scope.WriteString("C1:WF? DAT1", 1)

    # data acquisition datapoints config
    raw = scope.ReadBinary(500000)  # 50kS (10µs/200ps)

    # バイナリデータを float に変換（1点 = 4バイト float）
    y = np.frombuffer(raw, dtype=np.float32)
    return y

def estimate_baseline(y, method, window=200): #window = n of samples used for baseline correction
    if method == 'mean':
        return np.mean(y)
    elif method == 'rolling':
        return np.mean(y[:window])  # 最初の200点でベースライン
    else:
        return 0  # 無補正

# 初期化
avg_waveform = None
count = 0

plt.ion()  # インタラクティブモードON

while True:
    try:
        y = get_waveform()
        y_corr = y - estimate_baseline(y, method='rolling')

        if avg_waveform is None:
            avg_waveform = y_corr.copy()
            count = 1
        else:
            count += 1
            avg_waveform += (y_corr - avg_waveform) / count

        if count % 100 == 0:
            plt.clf()
            plt.plot(avg_waveform)
            plt.title(f"Running average (n={count})")
            plt.xlabel("Point")
            plt.ylabel("Voltage (V)")
            plt.pause(0.01)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
