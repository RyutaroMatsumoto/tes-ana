from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import scipy.signal as sig
import scipy.special as sp
from sklearn.metrics import roc_curve, auc
from src.fft_funcs import fivesmt_scupyrfft
from scipy.signal import iirnotch, sosfilt, sosfreqz

###############################################################################
# Utility helpers
###############################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

###############################################################################
# AVERAGING PREPROCESS
###############################################################################

def _average_groups(arr: np.ndarray, n: int) -> np.ndarray:
    """
    arr.shape = (N_waveforms, N_points)
    n: averaging_num
    → N_waveforms // n 個のグループに分け、各グループを平均して返す。
    """
    if n <= 1:
        return arr.copy()
    m = arr.shape[0] // n
    if m == 0:
        raise ValueError("averaging_num is larger than the number of waveforms.")
    # 先頭から m*n 行だけを reshape して平均
    trimmed = arr[: m * n].reshape(m, n, -1)
    return trimmed.mean(axis=1)


def prepare_averaged_datasets(
    sig_raw: np.ndarray,
    noi_raw: np.ndarray,
    averaging_num: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    信号（sig_raw）とノイズ（noi_raw）をそれぞれ averaging_num 枚ごとに平均し、
    signal_diff (= averaged_signal - 全ノイズ平均) と
    averaged_noise_waveforms を返す。

    戻り値:
      signal_diff: shape = (N_sig_groups, N_points)
      noi_ave:     shape = (N_noi_groups, N_points)
    """
    sig_ave = _average_groups(sig_raw, averaging_num)
    noi_ave = _average_groups(noi_raw, averaging_num)
    # 各グループごとにランダムなnoi_aveの行を選ぶ
    num_groups = sig_ave.shape[0]
    random_indices = np.random.randint(0, noi_ave.shape[0], size=num_groups)
    random_noi = noi_ave[random_indices]  # shape: (N_groups, N_points)
    # 全ノイズの“グローバル平均”を計算（1×N_points）
    noise_global = noi_raw.mean(axis=0, keepdims=True)
    # 各グループ化された averaged_signal から“全ノイズ平均”を引く ->何も引かない
    signal_diff = sig_ave
    return signal_diff, noi_ave


###############################################################################
# 1.  POWER‑SPECTRAL‑DENSITY ESTIMATION
###############################################################################

def compute_median_psd_mmap(path: Path, dt: float, threads: int = 4, batch: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Return frequency vector *f* [Hz] and *mean* amplitude spectral density [V/sqrt(Hz)]
    using custom FFT logic (batched with optimized 5-smooth padding).
    """
    logger.info("Computing PSD using optimized FFT…")
    freq, sd, execution_time = fivesmt_scupyrfft(path, dt, threads, batch)
    logger.info("PSD computed in %.2f sec", execution_time)
    return freq, sd**2  # Convert ASD to PSD

###############################################################################
# 2.  WHITENING  (frequency‑domain pre‑whitening)
###############################################################################

def whiten_trace(trace: np.ndarray, noise_psd: np.ndarray) -> np.ndarray:
    """Whiten *trace* using external PSD (same shape as FFT output)."""
    # Get the length of the noise PSD
    n_freq = len(noise_psd)
    # Calculate the corresponding FFT length (n_freq = n_fft//2 + 1)
    n_fft = 2 * (n_freq - 1)
    
    # Compute FFT with the same length as used for noise_psd
    T = np.fft.rfft(trace, n=n_fft)
    asd = np.sqrt(noise_psd)
    eps = 1e-24
    
    # Now T and asd should have the same shape
    W = T / (asd + eps)
    whitened = np.fft.irfft(W)
    return whitened[: len(trace)]

###############################################################################
# 3.  ANALYTIC BAND‑PASS (20 MHz – 1 GHz)
###############################################################################

def design_bandpass(fs: float, f_low: float = 20e6, f_high: float = 1e9, order: int = 4):
    nyq = 0.5 * fs
    sos = sig.iirfilter(
        N=order,
        Wn=[f_low / nyq, f_high / nyq],
        btype="band",
        ftype="butter",
        output="sos",
    )
    return sos

def bandpass_filter(trace: np.ndarray, sos) -> np.ndarray:
    return sig.sosfiltfilt(sos, trace)

###############################################################################
# 3.a Notch (50MHz)
###############################################################################

fs   = 2_000_000_000       # 2 GS/s
f0   = 50_000_000          # 50 MHz
Q    = 30                  # バンド幅 ≈ f0/Q ≈ 1.7 MHz
#Q 値 を上げすぎると位相遅延が増え過ぎるので注意（目安 Q≲50）。
#複数帯を同時に抜く場合は sos を縦に連結していくだけ。
# 2次 IIR ノッチをセカンドオーダセクションで生成
b, a = iirnotch(w0=f0/(fs/2), Q=Q)
sos  = np.array([[b[0], b[1], b[2], 1, a[1], a[2]]])

def notch_50MHz(x):
    return sosfilt(sos, x)
###############################################################################
# 4.  MATCHED FILTER
###############################################################################

def build_matched_template(template: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(template)
    if norm == 0:
        raise ValueError("Template has zero energy.")
    return template / norm

def matched_filter(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    return sig.fftconvolve(signal, template[::-1], mode="same")

###############################################################################
# 5.  SCORE (MF peak)  ––– helper
###############################################################################
def _score_single(
    trace: np.ndarray,
    noise_psd: np.ndarray,
    sos,
    template_rev: np.ndarray,
) -> float:
    w = whiten_trace(trace, noise_psd)
    bp = bandpass_filter(w, sos)
    out = matched_filter(bp, template_rev)
    return float(np.max(np.abs(out)))


def _score_wrapper(trace: np.ndarray, noise_psd: np.ndarray, sos, template_rev: np.ndarray) -> float:
    """Wrapper function for _score_single that can be pickled for multiprocessing."""
    return _score_single(trace, noise_psd, sos, template_rev)


###############################################################################
# 6.  HIGH-LEVEL  ROC  EVALUATION
###############################################################################
def evaluate_roc(
    # training inputs for PSD & template
    sig_train: np.ndarray,
    noise_long_path: Path,
    # test inputs for ROC
    sig_test: np.ndarray,
    noise_test: np.ndarray,
    dt: float,
    fft_threads: int = 4,
    fft_batch: int = 100,
    n_workers: int = os.cpu_count() or 4,
) -> Dict[str, Any]:
    """
    Build whitening & matched filter from `sig_train` & `noise_long_path`,
    then evaluate ROC on `sig_test` & `noise_test` only.
    """
    fs = 1.0 / dt
    # 0) t domain cut -> done before input
    # 1) PSD from long noise
    f_noise, P_noise = compute_median_psd_mmap(
        noise_long_path, dt, threads=fft_threads, batch=fft_batch
    )

    # 2) design band‐pass filter
    sos = design_bandpass(fs)
    # 3) notch
    # sos_bp        = design_bandpass(fs)
    # sos_notch50   = sig.tf2sos(*sig.iirnotch(50e6/(fs/2), Q=30))
    # sos           = np.vstack([sos_bp, sos_notch50])  
    # 4) build matched template (mean of sig_train → whiten → BP → normalize → reverse)
    template = np.mean(sig_train, axis=0)
    tpl_w = whiten_trace(template, P_noise)
    tpl_bp = bandpass_filter(tpl_w, sos)
    
    tpl_rev = build_matched_template(tpl_bp)[::-1]

    # 6) score test traces in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        # Create a partial function that can be pickled
        score_func = partial(_score_wrapper, noise_psd=P_noise, sos=sos, template_rev=tpl_rev)
        
        sig_scores = np.array(
            list(ex.map(score_func, sig_test)),
            dtype=float
        )
        noi_scores = np.array(
            list(ex.map(score_func, noise_test)),
            dtype=float
        )

    # 5) compute ROC on test set
    y_true = np.concatenate([np.ones_like(sig_scores), np.zeros_like(noi_scores)])
    y_score = np.concatenate([sig_scores, noi_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
        "sig_scores": sig_scores,
        "noi_scores": noi_scores,
        "f_noise": f_noise,
        "P_noise": P_noise,
    }