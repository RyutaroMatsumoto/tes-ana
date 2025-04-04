import numpy as np
import matplotlib.pyplot as plt

# ファイルパスの設定
folder_path = input("Put your folder path: ")
file_paths = [
    f"{folder_path}/C2wave.npy",
    f"{folder_path}/C4wave.npy"
]

# 時間間隔 dt の設定
dt = 5.0e-11  # 50 ps

# プロットの設定
plt.figure(figsize=(10, 5))

# 色とラベルの設定
colors = ['b', 'r']  # 青と赤の色指定
labels = ['C2 Data', 'C4 Data']  # ラベル指定

# 各ファイルから50番目の行を読み込む
for file_path, color, label in zip(file_paths, colors, labels):
    # データを読み込む
    data = np.load(file_path)
    
    # 50番目の行を抽出（0-indexedで49）
    wave_data = data[55, :]
    
    # 時間データを生成
    time_data = np.arange(len(wave_data)) * dt
    
    # プロット
    plt.plot(time_data, wave_data, marker='', linestyle='-', color=color, label=label)

plt.title('Waveform Data Comparison at Row 50')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()
