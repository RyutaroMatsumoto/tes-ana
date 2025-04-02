import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ユーザーからフォルダパスを入力
folder = input("Input Upper folder path: ")

# データフォルダのリスト（フォルダ1～3のパスを指定）
data_dirs = [
    os.path.join(folder, "BFP"),
    os.path.join(folder, "CEL-2"),
    #os.path.join(folder, "SKY")
]

# ヒストグラムの色とラベル
colors = ['blue', 'green', #'orange'
          ]
labels = ['BFP', 'CEL', #'SKY'
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

for data_dir, color, label in zip(data_dirs, colors, labels):
    # 'db.npy' を読み込み
    db_path = os.path.join(data_dir, 'db.npy')
    if not os.path.exists(db_path):
        print(f"Warning: {db_path} does not exist.")
        continue
    db = np.load(db_path)
    
    # フィッティングパラメータを読み込み
    fit_params_path = os.path.join(data_dir, 'fit_params.npy')
    if not os.path.exists(fit_params_path):
        print(f"Warning: {fit_params_path} does not exist.")
        continue
    param = np.load(fit_params_path)
    mu, sigma = param[0], param[1]
    
    # FWHMを計算
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    
    # ヒストグラムをプロット（カウント表示、枠線なし）
    counts, bins, patches = ax1.hist(db, bins=50, color=color, alpha=0.6, edgecolor='none')
    
    # ヒストグラムのハンドルを保存（最初のパッチのみ）
    hist_handles.append(patches[0])
    
    # ヒストグラムのラベルにFWHMを追加
    hist_labels.append(f"{label}, FWHM: {fwhm:.2e} dB")
    
    # フィッティング曲線をプロット（凡例には追加しない）
    x = np.linspace(db.min(), db.max(), 1000)
    pdf_fitted = norm.pdf(x, loc=mu, scale=sigma)
    ax2.plot(x, pdf_fitted,color=color, linewidth=2)
    
    # フィッティング曲線の最大値を更新
    current_max_pdf = pdf_fitted.max()
    if current_max_pdf > max_pdf:
        max_pdf = current_max_pdf

# グラフの装飾（左軸）
ax1.set_xlabel('Gain (dB)')
ax1.set_ylabel('Frequency')

# グラフの装飾（右軸）
ax2.set_ylabel('Probability Density')
# 確率密度の範囲を動的に設定
ax2.set_ylim(0, max_pdf * 1.1)

# 凡例の設定（ヒストグラムのみ、FWHMを含む）
ax1.legend(hist_handles, hist_labels, loc='upper left')

# グリッドの追加（オプション）
ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

# レイアウトの自動調整
plt.tight_layout()

# プロットを表示
plt.show()
