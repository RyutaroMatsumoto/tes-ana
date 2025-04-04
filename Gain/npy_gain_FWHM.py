import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import os
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
    below_threshold = data < y_threshold
    if any(below_threshold):
        start_index = np.where(below_threshold)[0][0]  # 最初に閾値を超えた位置

        # 極大値（ピーク）を探す
        for i in range(start_index, len(data) - 1):
            if data[i] < data[i + 1]:  # 増加に転じる点を検出
                return data[i]  # 極小値の y 値を返す
    return None  # 極小値が見つからない場合
####################################

# ファイルパス
data_dir = input("Put your INput folder pass: ")
output_dir = input("Put your OUTput folder pass: ")
# フォルダが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 閾値
ch2_threshold = 0.194
ch3_threshold = 0.267
# ベースライン
ch2_baseline = 0.245
ch3_baseline = 0.43

# C2wave.npyとC3wave.npyを読み込む
c2_data = np.load(f"{data_dir}/C4wave.npy")
c3_data = np.load(f"{data_dir}/C3wave.npy")

# ピークの値からゲインを求める
db = []
for c2_wave, c3_wave in zip(c2_data, c3_data):
    c2_peak_index = get_first_peak_below_threshold(c2_wave,ch2_threshold)
    c3_peak_index = get_first_peak_below_threshold(c3_wave,ch3_threshold)
    if c2_peak_index is not None and c3_peak_index is not None:
        gain = abs((c3_peak_index - ch3_baseline)/(c2_peak_index - ch2_baseline))
        db_gain= 20* math.log10(gain)
        db.append(db_gain)

# 'db'の値を保存
db = np.array(db)
np.save(os.path.join(output_dir, 'db.npy'), db)

# フィッティングパラメータを計算し保存
param = norm.fit(db)
np.save(os.path.join(output_dir, 'fit_params.npy'), param)
# 統計値の計算
mean_value = np.mean(db)
median_value = np.median(db)
std_value = np.std(db)

# プロットの準備
fig, ax1 = plt.subplots()

# ヒストグラムをプロット
color = 'tab:blue'
ax1.set_xlabel('Gain(dB)')
ax1.set_ylabel('Frequency', color=color)
hist, bins, _ = ax1.hist(db, bins=50, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 正規分布にフィット
param = norm.fit(db)
x = np.linspace(min(db), max(db), 100)
pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1]) * sum(hist) * np.diff(bins[:2])

# 右側の軸を追加
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Fit Probability', color=color)
ax2.plot(x, pdf_fitted, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# FWHMを計算
fwhm = 2 * np.sqrt(2 * np.log(2)) * param[1]

plt.title('Histogram of Gain(dB) with Gaussian Fit')
plt.annotate(f'FWHM: {fwhm:.2e}', xy=(0.7, 0.80), xycoords='axes fraction')
plt.annotate(f'Mean: {mean_value:.2e}', xy=(0.7, 0.95), xycoords='axes fraction')
plt.annotate(f'Median: {median_value:.2e}', xy=(0.7, 0.90), xycoords='axes fraction')
plt.annotate(f'Variance: {std_value:.2e}', xy=(0.7, 0.85), xycoords='axes fraction')
plt.show()
