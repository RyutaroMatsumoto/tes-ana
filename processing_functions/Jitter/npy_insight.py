import numpy as np

# .npy ファイルのパスを指定
file_path = '/Users/ryutaro_matsumoto/Desktop/Reaserch/20240930_SKY_CE/20241002CE_noise_npy/C3wave.npy'

# ファイルを読み込む
data = np.load(file_path)

# データの形状と型を表示
print("Shape of the data:", data.shape)
print("Data type:", data.dtype)

# データの最初の5要素を表示（データが多次元の場合は調整が必要）
print("First 5 elements:", data[:5])
