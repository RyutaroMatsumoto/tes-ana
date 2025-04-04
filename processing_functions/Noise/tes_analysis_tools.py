#! -*-coding:utf-8 -*-

# ===============================
# TES Analysis Tools
# Author      : Yuki Mitsuya
# Last update : 2024-07-24
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

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
        ph[i] = np.max(pulse[i, timin:timax])

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
    print("Optimal filtering started ...")
        
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
        print("|N(f)|^2 = ", X_noise_2)

    # モデルパルスのフーリエ変換の二乗
    X_model = np.fft.fft(model)
    X_model_2 = np.abs(X_model)**2
    if verbose :
        print("|M(f)|^2 = ", X_model_2)

    # 最適フィルタに利用する周波数上限の指定 (Hz)
    # FFTの結果は
    # freq[0]: 0Hz(DC), freq[1 ~ 半分まで]: 正の周波数, freq[半分 ~ 最後まで]: 負の周波数
    freq = np.fft.fftfreq(dp, dt)
    ind = np.where(freq[0:int(dp/2)] < maxfreq)
    maxfreq_ind = np.max(ind)
    if verbose :
        print("max frequency = ", maxfreq)
        print("max frequency index = ", maxfreq_ind)

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
            print(100.0 * i / n, "%  (", i ,"pulses processed )")
        
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
        plt.savefig("./hist_optimal-filter_log.png")
        plt.show()

    
    # ヒストグラム作成
    hist, bins = np.histogram(ph_array, bins=256)
    bins_center = np.zeros(hist.size)
    bins_center = (bins[0:bins.size - 1] + bins[1:bins.size])/2
    
    hist_data = np.zeros([bins_center.size, 2])
    hist_data[:, 0] = bins_center
    hist_data[:, 1] = hist

    print("Optimal filtering done.")
    
    return ph_array, hist_data


# ==============================================
# スペクトル密度 V/√Hz, A/√Hz の算出
# ==============================================
# 引数
# wave[dp] : 1波形データ. データ点数は偶数とすること (例 1024点).
# dt       : サンプリング時間 (sec)

def spectrum_density(wave, dt, Rf, Mf, Min, showplot, verbose):
    
    x  = wave       # 1波形
    n  = x.size     # sampling data points
    T  = n*dt       # total sampling time (sec)
    fs = 1./dt      # sampling freq (Hz)
    df = 1./T       # 周波数分解能 (Hz)

    time = np.linspace(0, T, n)
    
    if verbose == True:
        print("---------- Parameters ----------")
        print("n = ", n)
        print("dt = ", dt)
        print("T = ", T)
        print("fs = ", fs)
        print("df = ", df)
        
    # wave plot
    if showplot == True:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        plt.plot(time, x, color='red', linewidth=0.5)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [V]")
        plt.grid()
        plt.savefig("./Noise.png")
        plt.show()

    # FFT
    X = np.fft.fft(x) # 離散フーリエ変換
    
    N = X.size
    freq = np.fft.fftfreq(n, dt)  # frequency (Hz)

    if verbose == True:
        print("N = ", N)

    if verbose == True:
        print("---------- Check Perseval's theorem ----------")
        print("Σ|x|^2       = ", np.sum(x**2) )
        print("1/N * Σ|X|^2 = ", np.sum(np.abs(X)**2) / N )
    

    # -----------------------------------
    # 直流成分
    if verbose == True:

        print("---------- FFT Frequency component ----------")
        print("*** DC component ***")
        print("[k = 0]: freq = ", freq[0], ", F[k] = ", X[0])
        print("")

    # フーリエ変換成分（複素数）, 正負の周波数で複素共役になっている.
    if verbose == True:
        
        # 正の周波数 k = 1 ~ N/2
        print("*** Positive freq ***")
        print("[k = 1]:       freq = ", freq[1], ", F[k] = ", X[1])
        print("[k = 2]:       freq = ", freq[2], ", F[k] = ", X[2])
        print("[k = 3]:       freq = ", freq[3], ", F[k] = ", X[3])
        print("...")
        print("[k = N/2 - 2]: freq = ", freq[int(N/2) - 2], ", F[k] = ", X[int(N/2) - 2])
        print("[k = N/2 - 1]: freq = ", freq[int(N/2) - 1], ", F[k] = ", X[int(N/2) - 1])
        print("")
        
        print("*** Nyquist freq ***")
        print("[k = N/2] (nyquist freq) : ", freq[int(N/2)], ", F[k] = ", X[int(N/2)])
        print("")

        # 負の周波数 k = N/2 + 1 ~ N
        print("*** Negative freq ***")
        print("[k = N/2 + 1]: freq = ", freq[int(N/2) + 1], ", F[k] = ", X[int(N/2) + 1])
        print("[k = N/2 + 2]: freq = ", freq[int(N/2) + 2], ", F[k] = ", X[int(N/2) + 2])
        print("...")
        print("[k = N - 3]:   freq = ", freq[N-3], ", F[k] = ", X[N-3])
        print("[k = N - 2]:   freq = ", freq[N-2], ", F[k] = ", X[N-2])
        print("[k = N - 1]:   freq = ", freq[N-1], ", F[k] = ", X[N-1])
        

    # -----------------------------------
    # |X(k)| [V]
    freq_ = freq[0 : int(N/2)]  # 周波数範囲:片側
    X_abs = np.abs(X[0 : int(N/2)])

    
    if showplot == True:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(freq_, X_abs, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [$Hz$]")
        plt.ylabel("|F($\omega$)|")
        #plt.savefig("./F.png")
        plt.show()

    
    # -----------------------------------
    # パワースペクトル PS = |X|^2 [V^2]
    # 和がパワーになるように、Nの二乗で規格化
    PS = X_abs**2 / N**2
    
    # (2022/01/14 上記の規格化について説明追加）
    # どうして上記の規格化をするのかの説明をします.
    # まず, 信号のパワーは, 時間領域信号の二乗平均値です.
    # したがって, 離散時間信号では、(1/N) * Σ|x|^2 ということになります.
    # また, パーセバルの等式は以下となります.
    # Σ|x|^2 = (1/N) * Σ|X|^2
    # ここで両辺に1/Nを掛けると,
    # (1/N) * Σ|x|^2 = (1/N^2) * Σ|X|^2
    # 左辺は先程述べたように, 時間領域信号の二乗平均値、すなわちパワーを表します.
    # これが右辺と等しくなるのですが, その右辺はフーリエ変換の二乗和をNの二乗で割った値になっています.
    # これが, 上記コードでN^2で割っている理由になります.
    
    # 片側スペクトルのため、値を2倍しておく(DC成分以外)
    PS[1 : int(N/2)] = PS[1 : int(N/2)] * 2.0

    if showplot == True:    
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(freq_, PS, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [$Hz$]")
        plt.ylabel("Power Spectrum [$V^{2}$]")
        #plt.savefig("./PS.png")
        plt.show()
    
    # -----------------------------------
    # パワースペクトル密度 [V^2/Hz]
    PSD = PS/df

    if showplot == True:    
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(freq_, PSD, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [$Hz$]")
        plt.ylabel("Power Spectrum Density [$V^{2} / Hz$]")
        plt.savefig("./PSD.png")
        plt.show()


    # -----------------------------------
    # PSDの平方根をとる -> スペクトル密度 [V/√Hz]
    SD_V = np.sqrt(PSD)

    if showplot == True:
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family']= 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']

        plt.plot(freq_, SD_V, color='red', linewidth=0.5)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [$Hz$]")
        plt.ylabel("Spectrum density [$V / \sqrt{Hz}$]")
        #plt.savefig("./SD_V.png")
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






