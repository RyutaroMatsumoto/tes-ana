#import necessary libs
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Dict, List,Any
import scipy.signal as sig
from src.tes_analysis_tools import highres_psd
from src.sig_process import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

#cut all of inputs into t_range
def process_signal(p_id1:int,r_id1:int, c_id1:int, p_id2: int, r_id2:int, c_id2: int, t_range: List, base_dir: Path)->None:
    """
    Purpose:
        Process signal data from two sources, plot mean waveforms within t_range, and save cut waveforms.
            -Deployed
        Analize signal and noise to detect signal from single waveform
            -Yet to be deployed
            Pipeline of the signal processing
            1. PSD analysis(upto 2GHz)
            2. Band-path filter(HPF for under 20MHz(1/f), LPF for over 1GHz(reflection block) )
            3. Notch (50MHz)
            4. Whitening (20MHz-1GHz)
            5. Matched filter around signal detection
            6. Draw ROC

    Data:
        sampling rate: 2GHz(500ps/pt)
        sample length: 50µs(100k sample)
        sample number: 10k for signal, 3k for noise
    Measurement Property:
        -Voltage amplification under 3K-
            signal detected from averaged pulse
            want to detect from single pulse
            switching noise interference starts just after signal.
        averaged signal amplitude: 0.015V
        raw noise amplitude: 0.2V
        rise time (from average): 5-6ns
        signal time range: 15ns (as we just need rise time to detect signal)
        rise time jitter: 1,2 samples (0.5-1ns: wild guess, but mostly true)
    environment:
        Apple M3 
        24GB RAM
            -need to use mmap
    Args:
        p_id1, r_id1, c_id1: IDs for first data source (signal)
        p_id2, r_id2, c_id2: IDs for second data source (noise)
        t_range: List containing [start_time, end_time] in microseconds
        base_dir: Base directory path
    """
    #data input paths
    dataname1 = f'p{p_id1}_r{r_id1}_C{c_id1}'
    dataname2 = f'p{p_id2}_r{r_id2}_C{c_id2}'
    dataname = f'p{p_id1}_r{r_id1}_C{c_id1} and p{p_id2}_r{r_id2}_C{c_id2}'
    metadata_path1 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id1}" / f"r{r_id1}" / f"lecroy_metadata_p{p_id1}_r{r_id1}.json"
    metadata_path2 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id2}" / f"r{r_id2}" / f"lecroy_metadata_p{p_id2}_r{r_id2}.json"
    #load metadata(time interval)
    with open(metadata_path1, 'r') as f:
        metadata1 = json.load(f)
    dt1 = metadata1[f"C{c_id1}--00000"]['time_resolution']['dt']
    with open(metadata_path2, 'r') as f:
        metadata2 = json.load(f)
    dt2 = metadata2[f"C{c_id2}--00000"]['time_resolution']['dt']

    # t-domain raw data files: file1 for signal, file2 for noise
    raw_file1 = base_dir / "generated_data" / "raw" / f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}" /f"C{c_id1}--Trace.npy"
    raw_file2 = base_dir / "generated_data" / "raw" / f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}" /f"C{c_id2}--Trace.npy"

    # Format t_range for filename (in microseconds)
    t_range_str = f"{t_range[0]}-{t_range[1]}_μs"
    
    #data output paths
    par_dir1 = base_dir / "generated_data" / "raw" / f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}" / f"C{c_id1}--Trace--{t_range_str}.npy"
    par_dir2 = base_dir / "generated_data" / "raw" / f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}" / f"C{c_id2}--Trace--{t_range_str}.npy"
    par_dir3 = base_dir / "generated_data" / "pypar" / "params" /f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}"
    plt_dir = base_dir / "generated_data" / "pyplt" /"roc" /f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}"
    # Create output directories if they don't exist
    par_dir1.parent.mkdir(parents=True, exist_ok=True)
    par_dir2.parent.mkdir(parents=True, exist_ok=True)
    par_dir3.parent.mkdir(parents=True, exist_ok=True)
    plt_dir.parent.mkdir(parents=True, exist_ok=True)
    #load
    data1 = np.load(raw_file1)
    mean1 = np.mean(data1, axis = 0)
    t1 = np.arange(len(mean1)) * dt1
    
    data2 = np.load(raw_file2)
    mean2 = np.mean(data2, axis = 0)
    t2 = np.arange(len(mean2)) * dt2
  
    t1_start_idx = int(t_range[0] * 1e-6 / dt1)
    t1_end_idx = int(t_range[1] * 1e-6 / dt1)
    t2_start_idx = int(t_range[0] * 1e-6 / dt2)
    t2_end_idx = int(t_range[1] * 1e-6 / dt2)
    
    # Add debug logging
    logging.info(f"Time range in microseconds: {t_range}")
    logging.info(f"Time range in seconds: {[t * 1e-6 for t in t_range]}")
    logging.info(f"dt1: {dt1}, dt2: {dt2}")
    logging.info(f"Indices for data1: {t1_start_idx} to {t1_end_idx}")
    logging.info(f"Indices for data2: {t2_start_idx} to {t2_end_idx}")
    
    # Ensure indices are within bounds
    t1_start_idx = max(0, t1_start_idx)
    t1_end_idx = min(len(t1), t1_end_idx)
    t2_start_idx = max(0, t2_start_idx)
    t2_end_idx = min(len(t2), t2_end_idx)
    
   
    
    # Plot only the t_range portion of the mean waveforms
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Plot only the t_range portion
    t1_cut = t1[t1_start_idx:t1_end_idx]
    mean1_cut = mean1[t1_start_idx:t1_end_idx]
    data1_cut = data1[t1_start_idx:t1_end_idx, :]
    t2_cut = t2[t2_start_idx:t2_end_idx]
    mean2_cut = mean2[t2_start_idx:t2_end_idx]
    data2_cut = data1[t1_start_idx:t1_end_idx, :]
    data2_cut_path = base_dir / "generated_data" / "raw" / f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}" /f"C{c_id2}--time-cut--Trace.npy"
    np.save(data2_cut_path, data2_cut)

    plt.plot(t1_cut, mean1_cut, color='red', linewidth=0.5, label=dataname1)
    plt.plot(t2_cut, mean2_cut, color='blue', linewidth=0.5, label=dataname2)

    plt.grid(which='major')
    plt.grid(which='minor')
    plt.xlabel("Time [s]")
    plt.ylabel("Mean amplitude [V]")
    plt.legend()
    plt.title(f"Mean Wave Comparison ({t_range[0]}-{t_range[1]} μs) - {dataname}")

    # Save plot
    plot_dir = base_dir / "generated_data" / "pyplt" / "compare"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / f"mean_wave_comparison_{dataname}(t_range: {t_range[0]}-{t_range[1]} μs).png"
    plt.savefig(plot_file)
    logging.info(f"Saved plot to {plot_file}")
    
    # Show Plot
    logging.info("Creating plot")
    #plt.show()


    # 目標 AUC
    auc_target = 1.0
    # 平均化サイクル：1→5→10→...→50
    sweep = [1] + list(range(5, 101, 5))
    auc_list: List[Tuple[int, float]] = []

    for averaging_num in sweep:
        logging.info(f"--- Averaging num = {averaging_num} ---")
        try:
            sig_diff, noi_ave = prepare_averaged_datasets(data1_cut, data2_cut, averaging_num)
        except ValueError as e:
            logging.warning(f"Skipping averaging_num={averaging_num}: {e}")
            continue

        # 50:50 で学習/テスト分割 (signal 側のみ)   #need to be modified based on data
        n_sig = sig_diff.shape[0]
        n_train = int(0.5 * n_sig)
        sig_train = sig_diff[:n_train]
        sig_test = sig_diff[n_train:]
        # noise を signal_test と同数だけ使う
        noi_test = noi_ave[: len(sig_test)]

        # ROC 評価
        res = evaluate_roc(sig_train, data2_cut_path, sig_test, noi_test, dt=dt1)
        auc = res["auc"]
        auc_list.append((averaging_num, auc))

        # 個別 ROC プロットを保存
        plt.figure(figsize=(5, 4))
        plt.semilogx(res["fpr"], res["tpr"], ".-")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC (avg={averaging_num})  AUC={auc:.3f}")
        plt.grid(True)
        roc_save = plt_dir / f"roc_avg{averaging_num}.png"
        plt.savefig(roc_save, dpi=300)
        plt.close()
        logging.info(f"Saved ROC → {roc_save}")

        if auc >= auc_target:
            logging.info(f"AUC target {auc_target} reached at averaging_num={averaging_num}")
            break

    # AUC vs Averaging_Num のトレンドを保存
    if auc_list:
        nums, aucs = zip(*auc_list)
        plt.figure(figsize=(6, 4))
        plt.plot(nums, aucs, marker="o")
        plt.xlabel("Averaging Num")
        plt.ylabel("AUC")
        plt.title("AUC vs Averaging Num")
        plt.grid(True)
        auc_plot_path = plt_dir / "auc_vs_averaging.png"
        plt.savefig(auc_plot_path, dpi=300)
        plt.close()
        logger.info(f"AUC-vs-averaging プロットを保存 → {auc_plot_path}")

