import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from src.tes_analysis_tools import fit_pulse
from src.trap_filter import trap_filter
import sys
import os
import logging
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, Dict, List,Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

domain= Literal["time", "freq"]
def process_compare(p_id1:int,r_id1:int, c_id1:int, p_id2: int, r_id2:int, c_id2: int, domain:domain,fitting:bool, fitting_samples:int, p_init:List, traces:List[str], base_dir: Path)->None:
    plt_dir = base_dir / "generated_data" / "pyplt" / "compare"
    # Create output directories if they don't exist
    plt_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directories created: plt_dir={plt_dir}")
    dataname1 = f'p{p_id1}_r{r_id1}_C{c_id1}'
    dataname2 = f'p{p_id2}_r{r_id2}_C{c_id2}'
    dataname = f'p{p_id1}_r{r_id1}_C{c_id1} and p{p_id2}_r{r_id2}_C{c_id2}'
    if domain == "freq":
        # Try spectrum_data_CX.npz first, then noise_data_CX.npz if not found
        raw_file1 = base_dir / "generated_data" / "pypar" /"noise"/ f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}"/f"spectrum_data_C{c_id1}.npz"
        raw_file2 = base_dir / "generated_data" / "pypar" /"noise"/ f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}"/f"spectrum_data_C{c_id2}.npz"
        
        # File name variation handling - check if files exist
        if not raw_file1.exists():
            raw_file1 = base_dir / "generated_data" / "pypar" /"noise"/ f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}"/f"noise_data_C{c_id1}.npz"
        if not raw_file1.exists():
            logging.warning("No .npz files found in %s", raw_file1.parent)
        
        if not raw_file2.exists():
            raw_file2 = base_dir / "generated_data" / "pypar" /"noise"/ f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}"/f"noise_data_C{c_id2}.npz"
        if not raw_file2.exists():
            logging.warning("No .npz files found in %s", raw_file2.parent)
        logging.info(f"Input paths: raw_dir={raw_file1} and {raw_file2}")

        #load
        data1 = np.load(raw_file1)
        freq1 = data1.get('frequencies')
        mean_sdv1 = data1.get('mean_spectrum_density')
        
        data2 = np.load(raw_file2)
        freq2 = data2.get('frequencies')
        mean_sdv2 = data2.get('mean_spectrum_density')

        #ovelaid plot in freq domain 
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.plot(freq1, mean_sdv1, color='red', linewidth=0.5, label = dataname1)
        plt.plot(freq2, mean_sdv2, color='blue', linewidth=0.5, label= dataname2)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Mean Spectrum density [V / √Hz]")
        plt.legend()
        plt.title(f"Spectrum Comparison - {dataname}")
    
        # Save plot to plt_dir
        logging.info("Creating plot")
        plot_file = plt_dir / f"spectrum_comparison_{dataname}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {plot_file}")

    if domain == "time":
        metadata_path1 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id1}" / f"r{r_id1}" / f"lecroy_metadata_p{p_id1}_r{r_id1}.json"
        metadata_path2 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id2}" / f"r{r_id2}" / f"lecroy_metadata_p{p_id2}_r{r_id2}.json"
        #load metadata(time interval)
        with open(metadata_path1, 'r') as f:
            metadata1 = json.load(f)
        dt1 = metadata1[f"C{c_id1}--00000"]['time_resolution']['dt']
        with open(metadata_path2, 'r') as f:
            metadata2 = json.load(f)
        dt2 = metadata2[f"C{c_id2}--00000"]['time_resolution']['dt']

        # Try spectrum_data_CX.npz first, then noise_data_CX.npz if not found
        raw_file1 = base_dir / "generated_data" / "pypar" /"wave"/ f"p{p_id1}" / f"r{r_id1}" /f"mean_wave_C{c_id1}.npz"
        raw_file2 = base_dir / "generated_data" / "pypar" /"wave"/ f"p{p_id2}" / f"r{r_id2}" /f"mean_wave_C{c_id2}.npz"
        
        # File name variation handling - check if files exist
        if not raw_file1.exists():
            raw_file1 = base_dir / "generated_data" / "pypar" /"wave"/ f"p{p_id1}" / f"r{r_id1}" / f"C{c_id1}"/f"mean_wave_C{c_id1}.npz"
        if not raw_file1.exists():
            logging.warning("No .npz files found in %s", raw_file1.parent)
        
        if not raw_file2.exists():
            raw_file2 = base_dir / "generated_data" / "pypar" /"wave"/ f"p{p_id2}" / f"r{r_id2}" / f"C{c_id2}"/f"mean_wave_C{c_id2}.npz"
        if not raw_file2.exists():
            logging.warning("No .npz files found in %s", raw_file2.parent)
        logging.info(f"Input paths: raw_dir={raw_file1} and {raw_file2}")

        #load
        data1 = np.load(raw_file1)
        mean1 = data1.get('mean_wave')
        t1 = np.arange(len(mean1)) * dt1
        
        data2 = np.load(raw_file2)
        mean2 = data2.get('mean_wave')
        t2 = np.arange(len(mean2)) * dt2
        #ovelaid plot in freq domain 
        fig = plt.figure(figsize=(9, 6))
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        filename_suffix = ""

        if "channels1" in traces:
            plt.plot(t1, mean1, color='red', linewidth=0.5, label = dataname1)
            filename_suffix += "_with_" + "1"
        if "channels2" in traces:
            plt.plot(t2, mean2, color='blue', linewidth=0.5, label= dataname2)
            filename_suffix += "_with_" + "2"
        if "differential" in traces:
            plt.plot(t2, mean1-mean2, color='green', linewidth=0.5, label= dataname1+"-"+dataname2)
            filename_suffix += "_with_" + "diff"

        # ファイル名を生成 (例: "mean_wave_comparison" + filename_suffix + ".png")
        filename = f"mean_wave_comparison_{dataname}" + "(" + filename_suffix + ")" + ".png"

        # ファイルを保存 (plt.savefig() を使用)
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.xlabel("Time [s]")
        plt.ylabel("Mean amplitude [V]")
        plt.legend()
        plt.title(f"mean wave Comparison - {dataname}")
    
        # Save plot to plt_dir
        logging.info("Creating plot")
        plot_file = plt_dir / filename
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {plot_file}")
        #plt.show()

        #Fitting
        if fitting:
            logging.info("Fitting pulse...")
            fit_pulse(mean1-mean2, dt1, plt_dir, dataname, 1e-6, fitting_samples, p_init, True, True)
        
        
# %%
