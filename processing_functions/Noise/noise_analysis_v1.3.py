import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
from scipy.optimize import curve_fit
import tes_analysis_tools

def get_freq_SD_V(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_lines = lines[6:]  # 最初の6行を無視
    x_list = []
    for line in data_lines:
        if line.strip():  # 空行を無視
            line = line.lstrip(', ')
            values = line.split(',')
            if len(values) == 2:
                try:
                    x_value = float(values[0])
                    x_list.append(x_value)
                except ValueError:
                    print(f"Invalid data: {values}")

    x_data = np.array(x_list)
    results = tes_analysis_tools.spectrum_density(wave=x_data, dt=1e-10, Rf=0, Mf=0, Min=0, showplot=False, verbose=True)
    return results[0], results[1]

# ファイルが存在するディレクトリのパス
data_dir = '/Users/ryutaro_matsumoto/Desktop/HEMT/20240725noiseascii'
file_names = [f'C2--noise--{i:05d}.txt' for i in range(1, 801)]

# 800個のファイルからランダムに30個を選択
selected_files = random.sample(file_names, 100)

freqs = []
sdv = []

for file_name in selected_files:
    c2_file = os.path.join(data_dir, file_name)
    Results = get_freq_SD_V(c2_file)
    freqs.append(Results[0])
    sdv.append(Results[1])

sdv_data = np.array(sdv)
mean_sdv = np.mean(sdv_data, axis=0)

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
