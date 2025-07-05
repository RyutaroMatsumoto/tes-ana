import win32com.client, numpy as np, matplotlib.pyplot as plt, time, os
from typing import Dict, Optional


RUN_ID = "r004"
MAX_SHOTS = 5000                # how many repetition for averaging
IP        = "192.168.1.177"   # ←実機 IP
CHANNELS   = ["C1","C2","C4"]              # 取得チャネル 複数指定可能
NPOINTS   = 100000             # 最大サンプル（ポイント数）上限を高くしておけば、オシロの表示領域に一致する
BL_WIN    = 2000              # ベースライン平均点
SLEEP_SEC = 0.0              # error handling. base :0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "..", "tes01", "generated_data",
                         "average", "p07", RUN_ID)
os.makedirs(SAVE_DIR, exist_ok=True)

dso = win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1")
dso.MakeConnection(f"IP:{IP}")
dso.SetupWaveformTransfer(0, 0, 0)         # 先頭・全点・全セグメント


avg: Dict[str, Optional[np.ndarray]] = {ch: None for ch in CHANNELS}
count = {ch: 0    for ch in CHANNELS}
plt.ion()

while True:
    done_all = True
    for ch in CHANNELS:
        if count[ch] >= MAX_SHOTS:   # 上限達していたらスキップ
            continue
        done_all = False

        # 波形取得
        wf = dso.GetScaledWaveform(ch, NPOINTS, 0)
        y  = np.asarray(wf, dtype=np.float32)
        if y.size == 0 or np.isnan(y).any() or np.isinf(y).any():
            time.sleep(0.05);  continue

        # ベースライン補正
        y -= y[:BL_WIN].mean()

        # ランニング平均
        count[ch] += 1
        avg[ch] = y if avg[ch] is None else avg[ch] + (y - avg[ch]) / count[ch]

    # 描画（100 ショットごと）
    if all(c % 100 == 0 or c == 0 for c in count.values()):
        plt.clf()
        for ch in CHANNELS:
            if avg[ch] is not None:
                plt.plot(avg[ch], label=f"{ch}  n={count[ch]}")  # type: ignore
        ttl = f"Long Avg  {RUN_ID}   " + " / ".join(f"{ch}:{count[ch]}" for ch in CHANNELS)
        plt.title(ttl)          
        plt.legend(); plt.pause(0.01)

    if done_all:       # 全チャネル終了でループ脱出
        break

# ----------- 保存 -----------
for ch in CHANNELS:
    if avg[ch] is not None:
        np.save(os.path.join(SAVE_DIR, f"{ch}_avg.npy"), avg[ch])  # type: ignore
        print(f"{ch} 平均波形を保存 → {SAVE_DIR}")

dso.Disconnect()
print("計測完了")