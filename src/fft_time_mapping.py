import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def main():
    # JSON ファイルのパス
    json_file_path = Path("/Users/ltd/Documents/workspace/TES/tes01/teststand_metadata/par/fft/p01/r006/fft_metadata_p01_r006.json")

    # メタデータのディレクトリ構造から pXX/rYYY を取得
    relative_path = json_file_path.parts[-3:-1]  # ["p01", "r006"]
    save_dir = Path("/Users/ltd/Documents/workspace/TES/tes01/generated_data/pyplt/noise").joinpath(*relative_path)
    save_dir.mkdir(parents=True, exist_ok=True)  # ディレクトリを作成

    # JSON データを読み込む
    with json_file_path.open("r") as f:
        data = json.load(f)

    # threads, batch, time のデータを抽出
    threads = []
    batches = []
    times = []

    for entry in data:
        threads.append(entry["threads"])
        batches.append(entry["batch"])
        times.append(entry["time"])

    # ユニークな threads と batches を取得
    unique_threads = sorted(set(threads))
    unique_batches = sorted(set(batches))

    # 2D 平面を作成
    X, Y = np.meshgrid(unique_threads, unique_batches)

    # Z 軸 (time) の値を格納する配列を作成
    Z = np.zeros_like(X, dtype=np.float64)

    # Z 配列に time の値を埋め込む
    for t, b, time in zip(threads, batches, times):
        x_idx = unique_threads.index(t)
        y_idx = unique_batches.index(b)
        Z[y_idx, x_idx] = time

    # 3D プロットを作成
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # サーフェスプロット
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    # ラベルを設定
    ax.set_xlabel("Threads")
    ax.set_ylabel("Batch")
    ax.set_zlabel("Time (s)")
    ax.set_title("Time vs Threads and Batch")

    # カラーバーを追加
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # 各軸からの視点で保存
    views = {
        "default": (30, 170),  # デフォルトの視点
        "x_axis": (0, 0),     # X軸方向から
        "y_axis": (0, 90),    # Y軸方向から
        "z_axis": (90, 0)     # Z軸方向から
    }

    for view_name, (elev, azim) in views.items():
        ax.view_init(elev=elev, azim=azim)
        save_path = save_dir / f"fft_time_mapping_{view_name}.png"
        plt.savefig(save_path)
        print(f"Saved plot from {view_name} view to {save_path}")

if __name__ == "__main__":
    main()