import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

folder= input("Input Upper folder pass: ")
# データフォルダのリスト（フォルダ1～3のパスを指定）
data_dirs = [
    f"{folder}/BFP",
    f"{folder}/CEL",
    f"{folder}/SKY"
]

# ヒストグラムとフィッティング曲線を重ねて表示
colors = ['blue', 'green', 'orange']
labels = ['BFP', 'CEL', 'SKY']

plt.figure()

for data_dir, color, label in zip(data_dirs, colors, labels):
    # 'db.npy' を読み込み
    db = np.load(os.path.join(data_dir, 'db.npy'))

    # ヒストグラムを計算
    hist, bins = np.histogram(db, bins=50)

    # ヒストグラムをプロット
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, hist, width=bins[1]-bins[0], color=color, alpha=0.3, label=label)

    # フィッティングパラメータを読み込み
    param = np.load(os.path.join(data_dir, 'fit_params.npy'))

    # フィッティング曲線をプロット
    x = np.linspace(min(db), max(db), 100)
    pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1]) * len(db) * (bins[1]-bins[0])
    plt.plot(x, pdf_fitted, color=color, linewidth=2)

# グラフの装飾
plt.xlabel('Gain (dB)')
plt.ylabel('Frequency')
plt.legend()

# プロットを表示
plt.show()
