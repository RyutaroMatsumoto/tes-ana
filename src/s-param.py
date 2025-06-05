import skrf as rf
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "amps"         #tes01 for local directory, tes01_link for SSD directory


# S2Pファイルの読み込み
ntwk = rf.Network(BASE_DIR/"LNF"/"LNF_LNC_Low_band.s2p") 

# ネットワークの情報表示
print(ntwk)

# Sパラメータの周波数プロット（dB単位）
ntwk.plot_s_db()  # S11, S21, S12, S22 全部を表示

plt.title("S-Parameters (dB)")
plt.show()
