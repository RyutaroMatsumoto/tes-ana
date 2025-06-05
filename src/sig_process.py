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
import scipy.signal as sig
import scipy.special as sp
from sklearn.metrics import roc_curve, auc
from src.fft_funcs import fivesmt_scupyrfft

###############################################################################
# Utility helpers
###############################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def design_bandpass(fs: float, f_low: float = 20e6, f_high: float = 700e6, order: int = 4):
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

    # 1) PSD from long noise
    f_noise, P_noise = compute_median_psd_mmap(
        noise_long_path, dt, threads=fft_threads, batch=fft_batch
    )

    # 2) design band‐pass filter
    sos = design_bandpass(fs)

    # 3) build matched template (mean of sig_train → whiten → BP → normalize → reverse)
    template = np.mean(sig_train, axis=0)
    tpl_w = whiten_trace(template, P_noise)
    tpl_bp = bandpass_filter(tpl_w, sos)
    tpl_rev = build_matched_template(tpl_bp)[::-1]

    # 4) score test traces in parallel
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