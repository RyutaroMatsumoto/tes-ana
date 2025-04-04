import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

# ファイルを読み込んでデータを抽出する関数
def read_first_x_above_threshold(file_path, y_threshold, comparison):
    with open(file_path, 'r') as file:
        lines = file.readlines()[6:]  # 最初の6行を無視
        for line in lines:
            if line.strip():
                line = line.lstrip(', ')
                values = line.split(',')
                if len(values) == 2:
                    try:
                        x_value = float(values[0])
                        y_value = float(values[1])
                        if comparison(y_value, y_threshold):
                            return x_value
                    except ValueError:
                        pass  # 無効なデータは無視
    return None

# 比較条件を定義する関数
def is_greater(y_value, threshold):
    return y_value > threshold

# ファイルが存在するディレクトリのパス
data_dir = '/Users/ryutaro_matsumoto/Desktop/Reaserch/20240812_HEMT/20240725gitterascii'

# xの数値の差を計算する
differences = []

for i in range(1, 1001):
    c1_file = os.path.join(data_dir, f'C1--wave--{i:05d}.txt')
    c2_file = os.path.join(data_dir, f'C2--wave--{i:05d}.txt')

    c1_x_value = read_first_x_above_threshold(c1_file, -0.05, is_greater)
    c2_x_value = read_first_x_above_threshold(c2_file, 0.12, is_greater)

    if c1_x_value is not None and c2_x_value is not None:
        differences.append(c2_x_value - c1_x_value)


# 統計値の計算
mean_value = np.mean(differences)
median_value = np.median(differences)
std_value = np.std(differences)

# プロットの準備
fig, ax1 = plt.subplots()

# ヒストグラムをプロット
color = 'tab:blue'
ax1.set_xlabel('Difference (C2_x - C1_x)')
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

plt.title('Histogram of Differences with Gaussian Fit')
plt.xlabel('Difference (C2_x - C1_x)')
plt.annotate(f'FWHM: {fwhm:.2e}', xy=(0.7, 0.80), xycoords='axes fraction')
plt.annotate(f'Mean: {mean_value:.2e}', xy=(0.7, 0.95), xycoords='axes fraction')
plt.annotate(f'Median: {median_value:.2e}', xy=(0.7, 0.90), xycoords='axes fraction')
plt.annotate(f'Variance: {std_value:.2e}', xy=(0.7, 0.85), xycoords='axes fraction')
plt.show()
