#! -*-coding:utf-8 -*-

# ===============================
# TES Analysis Tools
# Author      : Yuki Mitsuya
# Last update : 2024-07-24
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import math, json, logging
from scipy.optimize import curve_fit
import numba
from numba import jit, objmode
from numpy.fft import fft, fftfreq
import accelerate_fft as afft
from scipy.fft import next_fast_len
import time
from pathlib import Path
# 物理定数
kB = 1.380E-23   # ボルツマン定数 [J/K]


# ==============================================
# ジョンソンノイズ
# ==============================================
# 引数
# T : 温度 [K]
# R : 抵抗値 [Ohm]
# f : 周波数 [Hz]

# 返り値
# dI : [A/√Hz]
# dV : [V/√Hz]

# メモ
# TESの抵抗によって生じるノイズ.
# dV = √(4kBTR)
# dI = √(4kBT/R)
# ジョンソンノイズのノイズ密度は本来は周波数によらないが、
# w << 1/τ0 の領域では電熱フィードバックによって抑制され,
# w >> 1/τeff の領域では本来の周波数に戻るような形になる.

def calc_johnson_noise(T, R, f, tau_0, tau_eff, L0):
    
    # ノイズ電流密度
    dI2 = (4*kB*T/R) * (1/(1 + L0))**2 * (1 + f**2 * tau_0**2)  / (1 + f**2 * tau_eff**2)
    dI  = np.sqrt(dI2)
    
    dV2 = dI2*R
    dV  = np.sqrt(dV2)
    
    return dI


# ==============================================
# フォノンノイズ
# ==============================================
# 引数
# T : 温度 [K]
# G : 熱伝導度 [W/K]
# f : 周波数 [Hz]

# 返り値
# dI : [A/√Hz]
# dV : [V/√Hz]


# メモ
# 熱浴への有限な熱伝導度によって生じるノイズ

def calc_phonon_noise(T, G, f, Vb, tau_eff, L0):
   
    dI2 = 4*kB*G*(T**2) * responsivity_pow(L0, tau_eff, f, Vb)
    dI  = np.sqrt(dI2)
    
    return dI


# ==============================================
# Responsivity SI^2
# ==============================================
# 引数
# L0 : ループゲイン
# tau_eff : 実効時定数 [s]
# f  : 周波数 [Hz]
# Vb : バイアス電圧 [V]

# 返り値
# SI^2

def responsivity_pow(L0, tau_eff, f, Vb):
    SI2 = (1./Vb)**2 * (L0/(L0 + 1))**2 * (1/(1 + f**2 * tau_eff**2))
    return SI2

# ==============================================
# パルス波形
# ==============================================

def pulse_shape(x, x0, a, tau1, tau2):

    #ind = np.where(x <= x0)
    #maxphind = np.max(ind)
    
    rise  = a * (1.0 - np.exp(-(x - x0)/tau1) )
    decay = a * np.exp( -(x - x0)/tau2 )
    #pulse = np.hstack( [rise[:maxphind], decay[maxphind:] ])
    #return pulse
    
    return rise
    
    '''
    if x <= x0 :
        return a * (1.0 - np.exp(-(x - x0)/tau1) )
    if x > x0 :
        return a * np.exp( -(x - x0)/tau2 )
    '''

# ==============================================
# パルス波形にフィット
# ==============================================

def fit_pulse(pulse, dt, verbose, savefig, dir_output):
    
    dp = pulse.size
    
    # パルス最大波高位置を求め, 立ち下がり開始位置を探す
    x0 = np.argmax(pulse)
    
    x = np.linspace(0, dp, dp)
    t = x * dt
   
    # フィッティング
    param, cov = curve_fit(pulse_shape, x, pulse)
       
    x0_opt = param[0]   # 最大波高位置
    a_opt = param[1]    # スケール
    tau1_opt = param[2] # 立ち上がり時定数1
    tau2_opt = param[3] # 立ち下がり時定数1

    if verbose:
        print("Rise time constant [s] : ", tau1_opt * dt)
        print("Decay time constant [s] : ", tau2_opt * dt)


    if savefig:
        
        ymax = np.max(pulse) * 1.2
        ymin = -0.005

        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        plt.plot(t, pulse,  color=[0.0, 1.0, 0.0], marker='.')
        plt.plot(t, pulse_shape(x, *param), color='red')
        
        plt.ylabel("Pulse height [a.u.]")
        plt.xlabel("Time (x dt) [s]")

        #plt.xticks(x_all, time)
        #plt.xticks(time)
        
        plt.ylim([ymin, ymax])
        plt.grid()
        plt.savefig(dir_output + "fit_pulse.png")
        #plt.show()

    return tau1_opt, tau2_opt



# ==============================================
# 指数関数的減衰関数
# ==============================================
# 引数
# x : 
# a : スケール
# b : 減衰時定数

# 返り値
# t = 0 からの指数関数減衰後の値

def exp_decay(x, a, b):
    return a * np.exp(-x/b)


# ==============================================
# 立ち下がりフィッティング
# ==============================================
# 引数
# pulse[dp] : 波形データ1個, データ点数dp
# dt        : サンプリング時間 [s]

# 返り値
# tau : 立ち下がり時定数 [s]
    
def fit_pulse_decay(pulse, dt, verbose, savefig, dir_output):
    
    dp = pulse.size
    
    # パルス最大波高位置を求め, 立ち下がり開始位置を探す
    ind = np.argmax(pulse)
    ind = int( ind * 1.05 )
    
    if verbose:
        print("Decay from : ", ind, " (index)")

    x = np.linspace(0, dp - ind, dp - ind) 
    y = pulse[ind : dp]
   
    # 立ち下がり位置から最後までをフィッティング
    param, cov = curve_fit(exp_decay, x, y)
       
    a_opt = param[0]  # スケール
    b_opt = param[1]  # 減衰時定数

    # 時間のスケールに直す
    tau = b_opt * dt

    if verbose:
        print("Decay time constant [s] : ", tau)

    # フィッティングしていない側のプロット用
    x_neg = np.linspace(-ind, -1, ind)
    
    if savefig:
        
        ymax = np.max(pulse) * 1.2
        ymin = -0.005

        #x_all = np.linspace(0, dp-1, dp)
        #time = np.linspace(0, dp * dt, dp)
        
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(x_neg, pulse[0 : ind],  color=[0.0, 1.0, 0.0], marker='.')
        plt.plot(x,     pulse[ind : dp], color=[0.0, 1.0, 0.0], marker='.')
        plt.plot(x_neg, exp_decay(x_neg, a_opt, b_opt), color='red')
        plt.plot(x,     exp_decay(x, a_opt, b_opt),     color='red')
        
        plt.ylabel("Pulse height [a.u.]")
        plt.xlabel("Time (x dt) [s]")

        #plt.xticks(x_all, time)
        #plt.xticks(time)
        
        plt.ylim([ymin, ymax])
        plt.grid()
        plt.savefig(dir_output + "fit_pulse_decay.png")
        #plt.show()

    return tau
        

# ==============================================
# ベースラインオフセット補正
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個, 1波形あたりdp点
# dpbl         : ベースライン計算に使う立ち上がり前のデータ点数

# 返り値
# pulse_corr[n, dp] : 補正後の波形データn個, 1波形あたりdp点

def correct_baseline(pulse, dpbl):
    
    n  = pulse.shape[0]
    dp = pulse.shape[1]
    pulse_corr = np.zeros([n, dp])
    
    for i in range(0, n):
        pulse_corr[i, :] = pulse[i, :] - np.average(pulse[i, 0:dpbl])

    return pulse_corr

# ==============================================
# 波高値で選択的に平均波形作成
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個, 1波形あたりdp点
# phmin        : 平均波形を作るパルスの選択, 波高値の下限
# phmax        : 平均波形を作るパルスの選択, 波高値の上限
# timin        : 波高値を測定する時間indexの下限
# timax        : 波高値を測定する時間indexの上限

# 返り値
# pulse_avg[dp] : 平均波形, データ点数dp点

def make_average_pulse(pulse, phmin, phmax, timin, timax, normalize, verbose, showplot):

    n  = pulse.shape[0]
    dp = pulse.shape[1]
    ph = np.zeros(n)
    
    # 波高値測定
    for i in range(0, n):
        ph[i] = np.max((pulse[i, timin:timax]))

    pulse_avg = np.zeros(dp)
    cnt = 0

    for i in range(0, n):

        if phmin < ph[i] and ph[i] < phmax:
            if normalize:
                pulse_avg = pulse_avg + pulse[i, :] / ph[i]
            else :
                pulse_avg = pulse_avg + pulse[i, :]
            cnt = cnt + 1

    pulse_avg = pulse_avg / cnt
                    
    if verbose:
        print("number of selected pulses = ", cnt)        

    if showplot:
        plt.plot(pulse_avg)
        plt.title("averaged pulse")
        #plt.xlabel("time")
        #plt.ylabel("pulse height")
        plt.grid()
        plt.show()

    return pulse_avg


# ==============================================
# 単純波高値スペクトル
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個,  1波形あたりdp点
# timin        : 波高値を求める時間レンジ（index）の下限
# timax        : 波高値を求める時間レンジ（index）の上限

# 返り値
# ph_array[n] : n波形の波高値
# hist_data[nbins, 2] : 1列目bin幅中心値, 2列目ヒストグラムカウント値

def simple_ph_spectrum(pulse, timin, timax, showplot, verbose):

    n = pulse.shape[0]
    ph_array = np.zeros(n)
    
    for i in range(0, n):
        ph_array[i] = np.max( pulse[i, timin:timax] )

    if verbose:
        print("max pulse height = ", np.max(ph_array))
        print("min pulse height = ", np.min(ph_array))
    
    if showplot:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(ph_array, bins=256)
        plt.savefig("./hist_simple_ph.png")
        plt.show()
        
        # 対数スケール
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(ph_array, bins=256)
        plt.yscale('log')
        plt.savefig("./hist_simple_ph_log.png")
        plt.show()
        

    # ヒストグラム作成
    hist, bins = np.histogram(ph_array, bins=256)
    bins_center = np.zeros(hist.size)
    bins_center = (bins[0:bins.size - 1] + bins[1:bins.size])/2
    
    hist_data = np.zeros([bins_center.size, 2])
    hist_data[:, 0] = bins_center
    hist_data[:, 1] = hist
    
    return ph_array, hist_data

# ==============================================
# 波形整形 -> 波高値スペクトル
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個,  1波形あたりdp点
# timin        : 波高値を求める時間レンジ（index）の下限
# timax        : 波高値を求める時間レンジ（index）の上限
# dt           : サンプリング時間 [sec]
# cr           : CR整形時定数 [sec]
# rc           : RC整形時定数 [sec]

# 返り値
# ph_array[n] : n波形の波高値
# hist_data[nbins, 2] : 1列目bin幅中心値, 2列目ヒストグラムカウント値

def shaping_ph_spectrum(pulse, timin, timax, dt, cr, rc, showplot, verbose):
    print("Shaping spectrum started ...")
    
    n = pulse.shape[0]
    ph_array = np.zeros(n)
    
    for i in range(0, n):
        
        if i % 1000 == 0 and verbose:
            print(100.0 * i / n, "%  (", i ,"pulses processed )")

        pls = cr_diff(pulse[i, :], cr, dt)
        pls = rc_int(pls, rc, dt)
        ph_array[i] = np.max( pls[timin:timax] )
        
        if showplot and i % 1000 == 0:
            plt.plot(pls)
            plt.show()

    if verbose:
        print("max pulse height = ", np.max(ph_array))
        print("min pulse height = ", np.min(ph_array))
    
    if showplot:
        plt.hist(ph_array, bins=256)
        #plt.savefig("./hist_simple_ph.png")
        plt.show()

    # ヒストグラム作成
    hist, bins = np.histogram(ph_array, bins=256)
    bins_center = np.zeros(hist.size)
    bins_center = (bins[0:bins.size - 1] + bins[1:bins.size])/2
    
    hist_data = np.zeros([bins_center.size, 2])
    hist_data[:, 0] = bins_center
    hist_data[:, 1] = hist
    
    print("Shaping spectrum done.")
        
    return ph_array, hist_data

# ==============================================
# 最適フィルタ（周波数領域）
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個,  1波形あたりdp点
# model[dp]    : モデルパルス1個, 1波形あたりdp点
# noise[m, dp] : ノイズデータm個, 1波形あたりdp点
# dt           : サンプリング時間 [sec]
# maxfreq      : 最適フィルタを掛ける周波数領域上限 [Hz]
# showplot     : bool値. プロット表示.
# verbose      : bool値. verbose表示.

# 返り値
# ph_array[n] : n波形の波高値
# hist_data[nbins, 2] : 1列目bin幅中心値, 2列目ヒストグラムカウント値

def optimal_filter_freq(pulse, model, noise, dt, maxfreq, showplot, verbose):
    logging.info("Optimal filtering started ...")
    
    n  = pulse.shape[0]      # num of pulses
    dp = pulse.shape[1]      # data points
    ph_array = np.zeros(n)   # pulse height

    # ノイズのフーリエ変換の二乗 |N(f)|^2
    m = noise.shape[0]
    for i in range(0, m):
        if i == 0:
            X_noise_2 = np.abs(np.fft.fft(noise[i, :])) ** 2
        else :
            X_noise_2 = X_noise_2 + np.abs(np.fft.fft(noise[i, :]))**2

    X_noise_2 = X_noise_2 / m
    if verbose :
        logging.info(f"|N(f)|^2 = {X_noise_2}")

    # モデルパルスのフーリエ変換の二乗
    X_model = np.fft.fft(model)
    X_model_2 = np.abs(X_model)**2
    if verbose :
        logging.info(f"|M(f)|^2 = {X_model_2}")

    # 最適フィルタに利用する周波数上限の指定 (Hz)
    # FFTの結果は
    # freq[0]: 0Hz(DC), freq[1 ~ 半分まで]: 正の周波数, freq[半分 ~ 最後まで]: 負の周波数
    freq = np.fft.fftfreq(dp, dt)
    ind = np.where(freq[0:int(dp/2)] < maxfreq)
    maxfreq_ind = np.max(ind)
    if verbose :
        logging.info(f"max frequency = {maxfreq}")
        logging.info(f"max frequency index ={maxfreq_ind}")

    # 最小二乗法の解析式により波高値Aを求める.
    # A = ∫DM*/|N|^2df / ∫|M|^2/|N|^2df
    # 式の例の参考: 宇宙研平社氏修論 式2.107
        
    # 分母の計算
    tmp2 = X_model_2 / X_noise_2
    sum2 = np.sum(tmp2[1 : maxfreq_ind])   # DC除外して指定の上限周波数まで積分.
    #sum2 = np.sum(tmp2[0 : maxfreq_ind])  # DCから指定の上限周波数まで積分. 
    #sum2 = np.sum(tmp2)                   # 周波数指定関係なく全領域. 負の周波数も.

    # 全パルスについて
    for i in range(0, n):
        if i % 1000 == 0 and verbose:
            logging.info(f"{100.0 * i / n} % pulses processed ")
        
        pls = pulse[i, :]
        X_pls = np.fft.fft(pls)
    
        # 分子
        tmp1 = (np.real(X_pls)*np.real(X_model) + np.imag(X_pls)*np.imag(X_model)) / X_noise_2
        sum1 = np.sum(tmp1[1 : maxfreq_ind])   # DC除外して指定の上限周波数まで積分.
        #sum1 = np.sum(tmp1[0 : maxfreq_ind])  # DCから指定の上限周波数まで積分. 
        #sum1 = np.sum(tmp1)                   # 周波数指定関係なく全領域.  負の周波数も.
    
        # 波高値
        A = sum1 / sum2
        ph_array[i] = A

        # フィッティング精度の確認        
        if i % 1000 == 0 :
            plt.plot(pls)
            plt.plot(model*A)
            plt.show()
            

    if showplot:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(ph_array, bins=256)
        plt.savefig("./hist_optimal-filter.png")
        plt.show()

        # 対数スケール
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(ph_array, bins=256)
        plt.yscale('log')
        plt.show()

    
    # ヒストグラム作成
    hist, bins = np.histogram(ph_array, bins=256)
    bins_center = np.zeros(hist.size)
    bins_center = (bins[0:bins.size - 1] + bins[1:bins.size])/2
    
    hist_data = np.zeros([bins_center.size, 2])
    hist_data[:, 0] = bins_center
    hist_data[:, 1] = hist

    logging.info("Optimal filtering done.")
    
    return ph_array, hist_data



# ==============================================
# スペクトル密度 V/√Hz, A/√Hz の算出 (Batch processing version)
# ==============================================
# 引数
# waves[n, dp] : n波形データ. データ点数は偶数とすること (例 1024点).
# dt           : サンプリング時間 (sec)

def calculate_spectrum_density_batch_padded(waves: np.ndarray, dt: float):
    """
    waves : (n_waveforms, n_samples)  2‑D 配列にしておくこと
    dt    : サンプリング間隔 [s]
    """
    n_waveforms, n_samples = waves.shape
    # --- 1. FFT 長を決める --------------------------------------------------
    # vDSP(=Accelerate) は「2 の冪」長しか受け付けない
    n_fft = 1 << (n_samples - 1).bit_length()   # 次の 2^k へ切り上げ
    # ↓ 他バックエンド向けに 5‑smooth 長を取りたいときはこっち
    # n_fft = next_fast_len(n_samples, real=True)  # :contentReference[oaicite:1]{index=1}

    # --- 2. ゼロパディング --------------------------------------------------
    if n_fft != n_samples:
        pad = n_fft - n_samples
        waves = np.pad(waves, ((0, 0), (0, pad)), mode="constant")

    # --- 3. rFFT ------------------------------------------------------------
    # Accelerate の rfft は NumPy と異なる「パック形式」で返るので unpack が必須
    X_packed = afft.rfft(waves, axis=1)
    X = afft.unpack(X_packed) / 2.0            # ← /2 で NumPy と同スケール

    # --- 4. PSD → SD --------------------------------------------------------
    T   = n_fft * dt          # パディング後の総時間
    df  = 1.0 / T
    PS  = np.abs(X) ** 2 / (n_fft ** 2)
    PS[:, 1:] *= 2.0          # 片側スペクトルなので 2 倍
    PSD = PS / df
    SD  = np.sqrt(PSD)

    # --- 5. 周波数軸 ---------------------------------------------------------
    freq = np.fft.rfftfreq(n_fft, dt)  # afft.rfftfreq でも OK

    return freq, SD

# Optimized calculation function with batch processing+ afft + padding
def calculate_spectrum_density_batch(waves, dt):
    """
    Process multiple waveforms at once using vectorized operations.
    
    Parameters:
    -----------
    waves : ndarray, shape (n_waveforms, n_samples)
        Array of waveforms to process
    dt : float
        Sampling time interval
        
    Returns:
    --------
    freq : ndarray
        Frequency array (same for all waveforms)
    SD_V : ndarray, shape (n_waveforms, n_freq)
        Spectrum density for each waveform
    """
    n_waveforms, n_samples = waves.shape
    T = n_samples * dt
    df = 1.0 / T
    
    # Perform batch FFT on all waveforms at once
    X = afft.rfft(waves, axis=1)  # Use rfft for real input (faster)
    n_freq = X.shape[1]
    
    # Calculate absolute values (magnitude)
    X_abs = np.abs(X)
    
    # Calculate power spectrum
    PS = X_abs**2 / (n_samples**2)
    
    # Double values for single-sided spectrum (except DC component)
    PS[:, 1:] = PS[:, 1:] * 2.0
    
    # Calculate power spectral density
    PSD = PS / df
    
    # Calculate spectrum density
    SD_V = np.sqrt(PSD)
    
    # Calculate frequency array (same for all waveforms)
    freq = afft.rfftfreq(n_samples, dt)
    
    return freq, SD_V

# ==============================================
# スペクトル密度 V/√Hz, A/√Hz の算出 (Non-Numba version)
# ==============================================
# 引数
# wave[dp] : 1波形データ. データ点数は偶数とすること (例 1024点).
# dt       : サンプリング時間 (sec)

# Optimized calculation function without Numba
def calculate_spectrum_density_no_numba(x, dt):
    n = x.size
    T = n * dt
    df = 1.0 / T
    
    # FFT
    X = np.fft.fft(x)
    N = X.size
    
    # |X(k)| [V]
    half_N = N // 2
    X_abs = np.abs(X[0:half_N])
    
    # パワースペクトル PS = |X|^2 [V^2]
    PS = X_abs**2 / N**2
    
    # 片側スペクトルのため、値を2倍しておく(DC成分以外)
    PS[1:half_N] = PS[1:half_N] * 2.0
    
    # パワースペクトル密度 [V^2/Hz]
    PSD = PS / df
    
    # PSDの平方根をとる -> スペクトル密度 [V/√Hz]
    SD_V = np.sqrt(PSD)
    
    # Calculate frequency array
    freq = np.fft.fftfreq(n, dt)
    freq_ = freq[0:half_N]  # 周波数範囲:片側
    
    return freq_, SD_V

# ==============================================
# 引数
# wave[dp] : 1波形データ. データ点数は偶数とすること (例 1024点).
# dt       : サンプリング時間 (sec)

# Optimized core calculation function with Numba JIT compilation
@numba.jit(nopython=True, cache=True)
def _calculate_spectrum_density(x, dt):
    n = x.size
    T = n * dt
    df = 1.0 / T
    
    # Use objmode for FFT operations that aren't supported in nopython mode
    with objmode(X='complex128[:]'):
        X = fft(x)
    
    N = X.size
    half_N = N // 2
    
    # Calculate absolute values
    X_abs = np.zeros(half_N)
    for i in range(half_N):
        X_abs[i] = abs(X[i])
    
    # パワースペクトル PS = |X|^2 [V^2]
    PS = np.zeros(half_N)
    for i in range(half_N):
        PS[i] = X_abs[i]**2 / (N**2)
    
    # 片側スペクトルのため、値を2倍しておく(DC成分以外)
    for i in range(1, half_N):
        PS[i] = PS[i] * 2.0
    
    # パワースペクトル密度 [V^2/Hz]
    PSD = np.zeros(half_N)
    for i in range(half_N):
        PSD[i] = PS[i] / df
    
    # PSDの平方根をとる -> スペクトル密度 [V/√Hz]
    SD_V = np.zeros(half_N)
    for i in range(half_N):
        SD_V[i] = np.sqrt(PSD[i])
    
    return SD_V

def spectrum_density(wave, dt, Rf, Mf, Min, showplot, verbose):
    x = wave       # 1波形
    n = x.size     # sampling data points
    T = n * dt     # total sampling time (sec)
    fs = 1.0 / dt  # sampling freq (Hz)
    df = 1.0 / T   # 周波数分解能 (Hz)

    # Calculate frequency array once (outside the optimized function)
    freq = fftfreq(n, dt)
    freq_ = freq[0:n//2]  # 周波数範囲:片側
    
    if verbose:
        print("---------- Parameters ----------")
        print("n = ", n)
        print("dt = ", dt)
        print("T = ", T)
        print("fs = ", fs)
        print("df = ", df)
    
    # Call the optimized calculation function
    SD_V = _calculate_spectrum_density(x, dt)
    
    # Only create plots if showplot is True
    if showplot:
        time = np.linspace(0, T, n)
        
        # Wave plot
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        plt.plot(time, x, color='red', linewidth=0.5)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [V]")
        plt.grid()
        plt.savefig("./Noise.png")
        plt.show()
        
        # Spectrum density plot
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        plt.plot(freq_, SD_V, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Spectrum density [V / √Hz]")
        plt.show()
    
    return freq_, SD_V
        
    # -----------------------------------
    # SQUID params
    #Min = 1/26.132
    #Mf  = 1/33.49
    #Rf  = 100.0E+3 # ohm
    
    # -----------------------------------
    # Voltage -> Current
    
    '''
    SD_I = (SD_V / Rf) * (Mf / Min)

    if showplot == True:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(freq_, SD_I, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [$Hz$]")
        plt.ylabel("Spectrum Density [$A / \sqrt{Hz}$]")
        #plt.savefig("./SD_A.png")
        plt.show()

    return freq_, SD_I, SD_V
    '''

# ==============================================
# パルスデータを整形する
# ==============================================
# 引数
# pulse[n, dp] : 波形データn個,  1波形あたりdp点
# dt           : サンプリング時間 [sec]
# cr           : CR整形時定数 [sec]
# rc           : RC整形時定数 [sec]

# 返り値
# pulse_shaped[n, dp] : 整形された波形データn個, 1波形あたりdp点

def shaping(pulse, dt, cr, rc, verbose):
    
    if verbose == True:
        print("Shaping ... ")
    
    n  = pulse.shape[0]      # num of pulses
    dp = pulse.shape[1]      # num of datapoints

    pulse_shaped = np.zeros([n, dp])
    
    for i in range(n):
        pls = cr_diff(pulse[i], cr, dt)
        pls = rc_int(pls, rc, dt)
        pls = rc_int(pls, rc, dt)
        pls = rc_int(pls, rc, dt)
        pls = rc_int(pls, rc, dt)
        pulse_shaped[i, :] = pls
        
    return pulse_shaped

# ==============================================
# CR微分回路フィルタ
# ==============================================
def cr_diff(x, cr, dt):

    length = x.size
    y = np.zeros(length)

    tau = dt/cr
    a = 1./(1.+tau)

    # 2021-03-03
    # i = 0の場合のロジックを変更. 
    
    for i in range(length):
        if i > 0:
            y[i] = a*y[i-1] + a*(x[i]-x[i-1])
        else:
            #y[i] = a*y0 + a*(x[i]-x0)
            y[i] = y[0]

    return y

# ==============================================
# RC積分回路フィルタ
# ==============================================
# y[k] = a*y[k-1] + b*x[k]
def rc_int(x, rc, dt):

    length = x.size    
    y = np.zeros(length)

    tau = dt/rc
    a = 1./(1.+tau)
    b = tau/(1.+tau)

    # 2021-03-03
    # i = 0の場合のロジックを変更. 

    for i in range(length):
        if i > 0:
            y[i] = a*y[i-1] + b*x[i]
        else:
            #y[i] = a*y0     + b*x[i]
            y[i] = y[0]

    return y

    
# ==============================================
# 波形の面積 積分
# ==============================================
def pulse_integral(pulse, timin, timax, showplot, verbose):

    n  = pulse.shape[0]      # num of pulses
    dp = pulse.shape[1]      # data points
    area_array = np.zeros(n) # pulse integral
    
    for i in range(n):
        if i % 1000 == 0 and verbose:
            print(100.0 * i / n, "%  (", i ,"pulses processed )")
            
        pls = pulse[i, :]
        
        area = np.sum(pls[timin:timax])
        area_array[i] = area
        
        if i % 1000 == 0 and verbose:
            print(area)

    if showplot:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(area_array, bins=256)
        plt.savefig("./hist_integral.png")
        plt.show()
        
        # 対数スケール
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.grid()
        plt.ylabel("Counts")
        plt.xlabel("Pulse height [V]")
        plt.hist(area_array, bins=256)
        plt.yscale('log')
        plt.savefig("./hist_integral_log.png")
        plt.show()
        
        
    return area_array






