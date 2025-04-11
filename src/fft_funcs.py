"""
Original is tes_analysis_tools.py
Copied and modified fft functions for faster processing.
    ca

"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import numba
from numba import jit, objmode
from numpy.fft import fft, fftfreq
import accelerate_fft as afft
from scipy.fft import next_fast_len
import time
# 物理定数
kB = 1.380E-23   # ボルツマン定数 [J/K]




def calc_sd_pocketfft(waves: np.ndarray, dt: float):
    waves = np.ascontiguousarray(np.atleast_2d(waves), dtype=np.float64)
    n_waveforms, n_samples = waves.shape

    # ❷ 5‑smooth 長へゼロパディング  ← vDSP の 2^k より ~25 % 小さい
    n_fft = fft.next_fast_len(n_samples, real=True)      # :contentReference[oaicite:1]{index=1}
    X = fft.rfft(waves, n=n_fft, axis=1, workers=os.cpu_count())  # :contentReference[oaicite:2]{index=2}

    # ❸ スペクトル密度へ変換（スケーリングは以前と同じ）
    T  = n_fft * dt
    df = 1.0 / T
    PS = (np.abs(X) ** 2) / (n_fft ** 2)
    PS[:, 1:] *= 2.0
    SD = np.sqrt(PS / df)

    freq = fft.rfftfreq(n_fft, dt)
    return freq, SD


def calc_sd_batch_padded(waves: np.ndarray, dt: float):
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
def calc_sd_batch(waves, dt):
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


# Optimized core calculation function with Numba JIT compilation
@numba.jit(nopython=True, cache=True)
def _calc_sd_numba(x, dt):
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

def spectrum_density(wave, dt):
    x = wave       # 1波形
    n = x.size     # sampling data points
    T = n * dt     # total sampling time (sec)
    fs = 1.0 / dt  # sampling freq (Hz)
    df = 1.0 / T   # 周波数分解能 (Hz)

    # Calculate frequency array once (outside the optimized function)
    freq = fftfreq(n, dt)
    freq_ = freq[0:n//2]  # 周波数範囲:片側
    
    # Call the optimized calculation function
    SD_V = _calc_sd_numba(x, dt)
    
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

