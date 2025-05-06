"""
correct baseline 
invert signal
optimal filter
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import sys
import logging
import os
import json
import src.trap_filter as trap_filter
from src.tes_analysis_tools import correct_baseline,make_average_pulse,optimal_filter_freq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def process_squid(period1:int,run1:int,channel1:int,period2:int,run2:int,channel2:int, maxfreq, phmin, phmax, timin, timax, normalize, showplot, verbose, base_dir:Path)->None:
    # create path for pulse and noise
    pulse = np.load(base_dir / "generated_data" / "raw" /f"p{period1}"/ f"r{run1}" / f"C{channel1}" / f"C{channel1}--Trace.npy")    #signal waveform
    noise = np.load(base_dir / "generated_data" / "raw" /f"p{period2}"/ f"r{run2}" / f"C{channel2}" / f"C{channel2}--Trace.npy")    #noise waveform
    #model = np.load(base_dir / "generated_data" / "pypar" / "wave" / f"p{period1}"/ f"r{run1}"/ f"mean_wave_C{channel1}.npz").get("mean_wave") # average signal waveform 
    #model(average wave) should be got by 02_t_domain_analysis.py using show_ave = True, before using this code.
    metadata_path1 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{period1}" / f"r{run1}" / f"lecroy_metadata_p{period1}_r{run1}.json"
    #load metadata(time interval)
    with open(metadata_path1, 'r') as f:
        metadata1 = json.load(f)
        dt = metadata1[f"C{channel1}--00000"]['time_resolution']['dt']
    
    plt_dir = base_dir / "generated_data" / "pyplt" /"optimal"/f"p{period1}"/ f"r{run1}" / f"C{channel1}"
    # Create directory if it doesn't exist
    plt_dir.mkdir(parents=True, exist_ok=True)
    
    #average the waveform
    average = make_average_pulse(-pulse, phmin, phmax, timin, timax, normalize, verbose, showplot)
    #correct base line for mean waveform
    #corrected_ave = correct_baseline(average, timin)
    #Optimal Filter
    ph_array, histdata =optimal_filter_freq(pulse, average, noise, dt, maxfreq, showplot, verbose)
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.grid()
    plt.ylabel("Counts")
    plt.xlabel("Pulse height [V]")
    plt.hist(ph_array, bins=256)
    plt.savefig(f"{plt_dir}/hist_optimal-filter_p{period1}_r{run1}.png")
    logging.info(f"hist_optimal-filter_log.png saved to {plt_dir}")
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
    plt.savefig(f"{plt_dir}/hist_optimal-filter_log_p{period1}_r{run1}.png")
    logging.info(f"hist_optimal-filter_log.png saved to {plt_dir}")


