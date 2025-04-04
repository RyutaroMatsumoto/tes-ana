import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
# ユーザーからフォルダパスを入力
folder = input("Input Upper folder path: ")

# データフォルダのリスト（フォルダ1～3のパスを指定）
data_dirs = [
    os.path.join(folder, "3K"),
    os.path.join(folder, "Room"),
    #os.path.join(folder, "SKY")
]

# ヒストグラムの色とラベル
colors = ['blue', 'green', 'orange']
labels = ['3K', 'Room-temp', #'SKY'
          ]

# プロットの設定
fig, ax1 = plt.subplots(figsize=(10, 6))

# 二重の縦軸を作成（右側に確率密度用）
ax2 = ax1.twinx()

# 凡例用のリスト（ヒストグラムのみ）
hist_handles = []
hist_labels = []

# フィッティング曲線の最大値を保持する変数
max_pdf = 0

# テキストの垂直位置の初期値（上から下へ）
text_y_position = 0.95

for data_dir, color, label in zip(data_dirs, colors, labels):
    # 'Jitter.npy' を読み込み
    jitter_path = os.path.join(data_dir, 'Jitter.npy')
    if not os.path.exists(jitter_path):
        print(f"Warning: {jitter_path} does not exist.")
        continue
    jitter = np.load(jitter_path)
    
    # フィッティングパラメータを読み込み
    fit_params_path = os.path.join(data_dir, 'fit_params.npy')
    if not os.path.exists(fit_params_path):
        print(f"Warning: {fit_params_path} does not exist.")
        continue
    param = np.load(fit_params_path)
    mu, sigma = param[0], param[1]
    
    # FWHMを計算
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    
    # ヒストグラムをプロット（カウント表示）
    counts, bins, patches = ax1.hist(jitter, bins=50, color=color, alpha=0.6, edgecolor='black')
    
    # ヒストグラムのハンドルを保存（最初のパッチのみ）
    hist_handles.append(patches[0])
    
    # ヒストグラムのラベルにFWHMを追加
    hist_labels.append(f"{label}, FWHM: {fwhm:.2e} s")
    
    # フィッティング曲線をプロット
    x = np.linspace(min(jitter), max(jitter), 1000)
    pdf_fitted = norm.pdf(x, loc=mu, scale=sigma)
    ax2.plot(x, pdf_fitted, color=color, linewidth=2)  # 凡例には追加しない
    
    # フィッティング曲線の最大値を更新
    current_max_pdf = pdf_fitted.max()
    if current_max_pdf > max_pdf:
        max_pdf = current_max_pdf

# グラフの装飾（左軸）
ax1.set_xlabel('Rise-time [s]')
ax1.set_ylabel('Frequency [N]')
# グラフの装飾（右軸）
ax2.set_ylabel('Probability Density')
# ax2.set_ylim(0, 1)  # これを削除またはコメントアウト

# 必要に応じて、動的にy軸の範囲を設定
ax2.set_ylim(0, max_pdf * 1.1)  # 最大値の10%上まで表示

# 凡例の設定（ヒストグラムのみ、FWHMを含む）
ax1.legend(hist_handles, hist_labels, loc='upper left')

# グラフを表示
plt.tight_layout()
plt.show()
