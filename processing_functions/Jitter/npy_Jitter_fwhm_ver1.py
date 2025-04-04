import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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


def get_first_peak_above_threshold(data, y_threshold):
    # 閾値を超えたインデックスを取得
    above_threshold = data > y_threshold
    if any(above_threshold):
        start_index = np.where(above_threshold)[0][0]  # 最初に閾値を超えた位置

        # 極大値（ピーク）を探す
        for i in range(start_index, len(data) - 1):
            if data[i] > data[i + 1]:  # 減少に転じる点を検出
                return data[i]  # 極大値の y 値を返す
    return None  # 極大値が見つからない場合


def get_first_peak_below_threshold(data, y_threshold):
    # 閾値を超えたインデックスを取得
    below_threshold = data > y_threshold
    if any(below_threshold):
        start_index = np.where(below_threshold)[0][0]  # 最初に閾値を超えた位置

        # 極大値（ピーク）を探す
        for i in range(start_index, len(data) - 1):
            if data[i] < data[i + 1]:  # 増加に転じる点を検出
                return data[i]  # 極小値の y 値を返す
    return None  # 極小値が見つからない場合
####################################

# ファイルパス
data_dir = input("Put your folder pass: ")

# C2wave.npyとC3wave.npyを読み込む
c2_data = np.load(f"{data_dir}/C2wave.npy")
c3_data = np.load(f"{data_dir}/C3wave.npy")

# 時間間隔 dt を定義
dt = 5.0e-11

# xの数値の差を計算し、遅延時間を求める
differences = []
for c2_wave, c3_wave in zip(c2_data, c3_data):
    c2_x_index = get_first_x_above_threshold(c2_wave,-0.37)
    c3_x_index = get_first_x_above_threshold(c3_wave,2.2)
    if c2_x_index is not None and c3_x_index is not None:
        time_difference = (c3_x_index - c2_x_index) * dt
        differences.append(time_difference)

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
