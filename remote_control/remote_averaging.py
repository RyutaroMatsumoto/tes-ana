import win32com.client #pywin32 library needs to be installed
import numpy as np
import matplotlib.pyplot as plt
import time

# オシロと接続
scope = win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") # same for every scope
scope.MakeConnection("IP:192.168.1.177") #this is just an example

def get_waveform():
    #scope.WriteString("COMM_FORMAT DEF9,WORD,BIN") #designate format
    # scope.WriteString("COMM_HEADER OFF", 1)
    # scope.WriteString("WAVEFORM_SETUP SP,0,NP,0,FP,0", 1)
    scope.SetupWaveformTransfer(0, 0, 0)
    scope.WriteString("C2:WF? DAT1", 1)

    # data acquisition datapoints config
    raw = scope.ReadBinary(400)  # unit: byte, 4byte = 1 sample
    print(f"Raw length: {len(raw)}") #debug

    # バイナリデータを float に変換（1点 = 4バイト float）
    raw = scope.ReadBinary(100000)
    raw = raw[16:]
    raw = raw[:len(raw) - len(raw) % 4]

    y = np.frombuffer(raw, dtype=np.float32)
    
    print("NaN含む？", np.isnan(y).any())
    print("inf含む？", np.isinf(y).any())
    print("値の範囲:", np.nanmin(y), np.nanmax(y))

    # scope.WriteString("C2:YMULT?", 1)
    # ymult = float(scope.ReadString(80))
    # scope.WriteString("C2:YOFF?", 1)
    # yoff = float(scope.ReadString(80))
    # scope.WriteString("C2:YZERO?", 1)
    # yzero = float(scope.ReadString(80))

    # # int16 で読み直す
    # y_raw = np.frombuffer(raw, dtype=np.int16)
    # y = (y_raw - yoff) * ymult + yzero

    return y

def estimate_baseline(y, method, window): #window = n of samples used for baseline correction
    if method == 'mean':
        return np.mean(y)
    elif method == 'rolling':
        return np.mean(y[:window])  # 最初のwindow点でベースライン補正
    else:
        return 0  # 無補正

# 初期化
avg_waveform = None
count = 0
count_max = 400000
bl_window = 200
plt.ion()  # インタラクティブモードON

while count < count_max:
    try:
        y = get_waveform()
        y_corr = y - estimate_baseline(y, method='rolling', window=bl_window)

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
scope.Disconnect()