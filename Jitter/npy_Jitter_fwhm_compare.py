import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# NumPy配列から最初のx値を取得する関数
def get_first_x_above_threshold(data, y_threshold):
    x_values = np.arange(len(data))  # 1次元配列のインデックスを時間の代わりに使用
    y_values = data
    above_threshold = y_values > y_threshold
    if any(above_threshold):
        first_index = np.where(above_threshold)[0][0]
        return x_values[first_index]
    return None
# NumPy配列から最初のx値を取得する関数
def get_first_x_below_threshold(data, y_threshold):
    x_values = np.arange(len(data))  # 1次元配列のインデックスを時間の代わりに使用
    y_values = data
    above_threshold = y_values < y_threshold
    if any(above_threshold):
        first_index = np.where(above_threshold)[0][0]
        return x_values[first_index]
    return None
####################################

# ファイルパス
data_dir = input("Put your INput folder pass: ")
output_dir = input("Put your OUTput folder pass: ")
# C2wave.npyとC3wave.npyを読み込む
c2_data = np.load(f"{data_dir}/C2wave.npy")
c3_data = np.load(f"{data_dir}/C3wave.npy")

#閾値の設定
ch2_threshold = -0.35
ch3_threshold = 0.6
# 時間間隔 dt を定義
dt = 5.0e-11

# xの数値の差を計算し、遅延時間を求める
differences = []
for c2_wave, c3_wave in zip(c2_data, c3_data):
    c2_x_index = get_first_x_above_threshold(c2_wave,ch2_threshold)
    c3_x_index = get_first_x_above_threshold(c3_wave,ch3_threshold)
    if c2_x_index is not None and c3_x_index is not None:
        time_difference = (c3_x_index - c2_x_index) * dt
        differences.append(time_difference)


# 'db'の値を保存
diffrences = np.array(differences)
np.save(os.path.join(output_dir, 'jitter.npy'),differences)

# フィッティングパラメータを計算し保存
param = norm.fit(diffrences)
np.save(os.path.join(output_dir, 'fit_params.npy'), param)
# 統計値の計算
mean_value = np.mean(differences)
median_value = np.median(differences)
std_value = np.std(differences)

# プロットの準備
fig, ax1 = plt.subplots()

# ヒストグラムをプロット
color = 'tab:blue'
ax1.set_xlabel('Time Difference (seconds)')
ax1.set_ylabel('Frequency', color=color)
hist, bins, _ = ax1.hist(differences, bins=50, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 正規分布にフィット
param = norm.fit(differences)
x = np.linspace(min(differences), max(differences), 100)
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1]) * sum(hist) * np.diff(bins[:2])

# 右側の軸を追加
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Fit Probability', color=color)
ax2.plot(x, pdf_fitted, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# FWHMを計算
fwhm = 2 * np.sqrt(2 * np.log(2)) * param[1]

plt.title('Histogram of Time Differences with Gaussian Fit')
plt.annotate(f'FWHM: {fwhm:.2e}', xy=(0.7, 0.80), xycoords='axes fraction')
plt.annotate(f'Mean: {mean_value:.2e}', xy=(0.7, 0.95), xycoords='axes fraction')
plt.annotate(f'Median: {median_value:.2e}', xy=(0.7, 0.90), xycoords='axes fraction')
plt.annotate(f'Variance: {std_value:.2e}', xy=(0.7, 0.85), xycoords='axes fraction')
plt.show()
