import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"       #tes01 for local directory, tes01_link for SSD directory
plt_dir = BASE_DIR/"generated_data"/"pyplt"/ "simulation"
plt_dir.mkdir(parents=True, exist_ok=True)
# データの読み込み
# タブ区切りのテキストファイルを読み込む
data = pd.read_csv(BASE_DIR/"teststand_metadata" /"hardware"/ "tes"/ "p03"/"simulation.txt", delimiter='\t')

# 時間とVinの列を取得
time = data.iloc[:, 0]  # 1列目（時間）
vin = data.iloc[:, 1]   # 2列目（Vin）

# 時間をマイクロ秒に変換（秒 * 10^6）
time_ns = time * 1e9
v_µv = vin * 1e6
# プロット
plt.figure(figsize=(10, 6))
plt.plot(time_ns, v_µv, label='Vin')

# 軸ラベルとタイトルの設定
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (µV)')


# 凡例の表示
plt.legend()

# グリッドの表示
plt.grid(True)

# プロットの表示
plt.tight_layout()
plt.savefig(f"{plt_dir}/simulation_vin.png")
logging.info(f"Saved plot to {plt_dir} ")
plt.show()
