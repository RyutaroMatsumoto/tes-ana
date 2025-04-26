"""
Original is tes_analysis_tools.py
Copied and modified fft functions for faster processing.
1. waveforms are aggrageted into array for batch processing, which is faster than processing every vector.
2. waveforms are zero padded for faster fft.
    "fivesmt_fft(waves: np.ndarray, dt: float): 5 smoothing zero padding for faster fft process, valid for PC and Mac.
    "pwrtwo_afft(waves: np.ndarray, dt: float): 2^n zero padding for afft library, valid for M series Mac.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import numba
from numba import jit, objmode
import accelerate_fft as afft
import pyfftw
from scipy.fft import next_fast_len,rfft,rfftfreq
import time
import os
from pathlib import Path
# 物理定数
kB = 1.380E-23   # ボルツマン定数 [J/K]


def fivesmt_pyfftw(path: Path, dt: float, threads:int, Batch:int):
    """
    Optimized 5-smooth zero padding for faster FFT processing without using afft.
    This is the main function that replaces the original implementation.
    
    Parameters:
    -----------
    waves : ndarray, shape (n_waveforms, n_samples) or (n_samples,)
        Array of waveforms to process
    dt : float
        Sampling time interval
        
    Returns:
    --------
    freq : ndarray
        Frequency array
    SD : ndarray
        Spectral density for each waveform
    """
    start_time = time.time()
    # Cashe FFTW plans for reprocess  
    pyfftw.interfaces.cache.enable()
    #load only needed data
    mmap = np.load(path,mmap_mode="r")
    n_wave, n_orig = mmap.shape     #n_waves: some lines from ndarray, n_orig: original sample points
    # Find optimal FFT length
    n_fft = _find_optimal_fft_length(n_orig)
    
    # Calculate time and frequency step
    T = n_fft * dt
    df = 1.0 / T
    n_freq = n_fft//2 + 1
    mean_psd = np.zeros(n_freq, np.float64)
    count=0
    BATCH = Batch                                # まとめる本数 (要: 104 % 8 == 0)
    buf_f32 = pyfftw.empty_aligned((BATCH, n_fft),  dtype="float32")
    out_c64 = pyfftw.empty_aligned((BATCH, n_freq), dtype="complex64")

    fftw_obj = pyfftw.FFTW(buf_f32, out_c64,
                       direction='FFTW_FORWARD',
                       flags=('FFTW_MEASURE',),
                       axes=(1,),
                       threads=threads)
    
    for i in range(0, n_wave, BATCH):
        chunk = mmap[i:i+BATCH]            
        n_this = chunk.shape[0]

        buf_f32[:n_this, :n_orig] = chunk
        buf_f32[:n_this, n_orig:] = 0.0

        fftw_obj()                           
        psd = (np.abs(out_c64[:n_this])**2).astype(np.float32) / n_fft**2
        psd[:,1:] *= 2.0                 
        mean_psd += psd.sum(0)
        count    += n_this

    mean_psd /= count
    sd = np.sqrt(mean_psd/df)
    freq = rfftfreq(n_fft, dt)               
    
    execution_time = time.time() - start_time
    
    return freq, sd, execution_time


def fivesmt_scupyrfft(path: Path, dt: float, threads:int, Batch:int):
    """
    Optimized 5-smooth zero padding for faster FFT processing without using afft.
    This is the main function that replaces the original implementation.
    
    Parameters:
    -----------
    waves : ndarray, shape (n_waveforms, n_samples) or (n_samples,)
        Array of waveforms to process
    dt : float
        Sampling time interval
        
    Returns:
    --------
    freq : ndarray
        Frequency array
    SD : ndarray
        Spectral density for each waveform
    """
    start_time = time.time()

    #load only needed data
    mmap = np.load(path,mmap_mode="r")
    n_wave, n_orig = mmap.shape     #n_waves: some lines from ndarray, n_orig: original sample points
    # Find optimal FFT length
    n_fft = _find_optimal_fft_length(n_orig)
    freq   = rfftfreq(n_fft, dt)
    # Calculate time and frequency step
    T = n_fft * dt
    df = 1.0 / T
    n_freq = n_fft//2 + 1
    mean_psd = np.zeros(n_freq, np.float64)
    processed=0 #initialize
    BATCH = Batch                           
    
    # ---------- バッチで回す ----------
    for start in range(0, n_wave, BATCH):
        end   = min(start + BATCH, n_wave)
        chunk = mmap[start:end]                      # ★ memmap スライス ― RAM 未展開

        # SciPy rfft は 2D 入力をそのまま batched FFT してくれる
        fx = rfft(chunk, n=n_fft, axis=1, workers=threads, overwrite_x=False)

        psd = (np.abs(fx)**2).astype(np.float32) / n_fft**2   # shape (b,n_freq)
        psd[:, 1:] *= 2.0                                     # 片側補正

        mean_psd  += psd.sum(axis=0)                          # バッチを合算
        processed += psd.shape[0]

    mean_psd /= processed
    sd        = np.sqrt(mean_psd / df)
                
        
    execution_time = time.time() - start_time
    
    return freq, sd, execution_time




def pwrtwo_fft(waves: np.ndarray, dt: float):
    """
    waves : (n_waveforms, n_samples)  2‑D 配列にしておくこと
    dt    : サンプリング間隔 [s]
    """
    afft.set_nthreads(os.cpu_count())
    n_waveforms, n_samples = waves.shape
    # --- 1. FFT 長を決める --------------------------------------------------
    # vDSP(=Accelerate) は「2 の冪」長しか受け付けない
    n_fft = 1 << (n_samples - 1).bit_length()   # 次の 2^k へ切り上げ
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

def _find_optimal_fft_length(n):
    """
    Find an optimal FFT length for performance, similar to next_fast_len.
    Returns a length that is a multiple of small primes (2, 3, 5, 7).
    """
    # Use scipy's next_fast_len if available
    try:
        return next_fast_len(n, real=True)
    except:
        # Fallback implementation if next_fast_len is not available
        if n <= 16:
            return n
        
        # Try to find a number that's a product of 2, 3, 5, 7
        best = n
        best_cost = float('inf')
        
        # Try lengths up to 2*n
        for target in range(n, 2*n + 1):
            # Factorize into small primes
            factors = []
            temp = target
            for p in [2, 3, 5, 7]:
                while temp % p == 0:
                    factors.append(p)
                    temp //= p
            
            # If temp is 1, then target is a product of 2, 3, 5, and 7 only
            if temp == 1:
                # Calculate a "cost" based on the factors
                # Fewer factors and smaller factors are better
                cost = sum(factors)
                if cost < best_cost:
                    best = target
                    best_cost = cost
        
        # If no optimal length found, return the next power of 2
        if best == n:
            return 1 << (n - 1).bit_length()
        
        return best

