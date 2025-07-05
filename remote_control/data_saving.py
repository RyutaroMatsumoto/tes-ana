"""
LeCroy ActiveDSO から “生波形をそのまま保存” するだけの最小スクリプト
  – 平均もプロットも一切しない
  – RUN_ID, MAX_SHOTS などはベタ書きで調整
  – 処理時間を計測して最後に表示
"""

import os, time, numpy as np, win32com.client, logging

# ===== ユーザ設定（必要なら数字や ID を書き換えてください） =====
RUN_ID     = "r002"           # r00x 形式のラン ID
MAX_SHOTS  = 3000             # 各チャネルの取得回数（ショット数）
CHANNELS   = ["C1", "C2"]     # 保存対象チャネル
IP         = "192.168.1.177"  # オシロ IP
NPOINTS       = 20000            # 1 波形あたりの最大ポイント
# 保存先：スクリプトと相対。 …/tes01/generated_data/raw/p07/r00x/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(SCRIPT_DIR, "..", "..", "tes01",
                          "generated_data", "raw", "p07", RUN_ID)
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== オシロ初期化 =====
dso = win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1")
dso.MakeConnection(f"IP:{IP}")
dso.SetupWaveformTransfer(0, 0, 0)

# ===== 時間計測開始 =====
t0 = time.perf_counter()
logging.info("Started data aquisition")
# ===== メインループ =====
for shot in range(1, MAX_SHOTS + 1):
    for ch in CHANNELS:
        try:
            wf = dso.GetScaledWaveform(ch, NPOINTS, 0)
            y  = np.asarray(wf, dtype=np.float32)

            # 無効波形はスキップ
            # if y.size == 0 or np.isnan(y).any() or np.isinf(y).any():
                # print(f"{ch} shot {shot}: invalid, skipped")
                # continue

            # ファイル名例: C1_shot0001.npy
            fn = os.path.join(SAVE_DIR, f"{ch}_shot{shot:04d}.npy")
            np.save(fn, y)

        except Exception as e:
            print(f"{ch} shot {shot}: {e}")
            if dso.ErrorFlag:
                print("DSO-Err:", dso.ErrorString)

    if shot % 100 == 0:
        print(f"=== {shot}/{MAX_SHOTS} shots 完了 ===")

# ===== 時間計測終了 =====
t1 = time.perf_counter()
elapsed = t1 - t0
rate    = (MAX_SHOTS * len(CHANNELS)) / elapsed
print(f"\n保存完了: {elapsed:.2f} s  ({rate:.1f} 波形/秒)")

dso.Disconnect()
