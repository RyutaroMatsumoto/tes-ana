import numpy as np
import matplotlib.pyplot as plt
import tes_analysis_tools
import random
# `.npy` ファイルからデータを読み込む
npy_file_path = input("Paste the correct Pass for data:")
data_array = np.load(npy_file_path,allow_pickle=True)

# モード選択のための入力
mode = input("Enter '5' for random 5 samples or 'run' to process all data: ")

freqs = []
sdv = []

# モードによって処理を分岐
if mode == "5":
    # データの行数からランダムに5つのインデックスを選択
    selected_indices = random.sample(range(data_array.shape[0]), 5)
    
    # 選択した行ごとに処理
    for index in selected_indices:
        wave_data = data_array[index]
        scaled_wave_data = wave_data 
        # tes_analysis_toolsを使った分析
        results = tes_analysis_tools.spectrum_density(wave=scaled_wave_data, dt=5.0e-11, Rf=0, Mf=0, Min=0, showplot=False, verbose=True)
        
        # 結果を保存
        freqs.append(results[0])
        scaled_sdv = results[1]
        sdv.append(scaled_sdv)

elif mode == "run":
    # 全行に対して処理
    for wave_data in data_array:
        # データを10^100倍する
        scaled_wave_data = wave_data 
        # tes_analysis_toolsを使った分析
        results = tes_analysis_tools.spectrum_density(wave=scaled_wave_data, dt=5.0e-11, Rf=0, Mf=0, Min=0, showplot=False, verbose=True)
        
        # 結果を保存
        freqs.append(results[0])
        scaled_sdv = results[1]
        sdv.append(scaled_sdv)

else:
    print("Invalid input. Please enter 'test' or 'entire'.")

# スペクトル密度データを配列に変換
if sdv:
    sdv_data = np.array(sdv)
    # 各周波数でのスペクトル密度の平均を計算
    mean_sdv = np.mean(sdv_data, axis=0)
    # 結果をプロット
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.plot(freqs[0], mean_sdv, color='red', linewidth=0.5)
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mean Spectrum density [V / √Hz]")
    plt.show()