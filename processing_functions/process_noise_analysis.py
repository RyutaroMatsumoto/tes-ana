import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import logging
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal, Dict, List,Any
import accelerate_fft as afft
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import fft_funcs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define Mode type as a Literal with allowed values
Mode = Literal["quick", "full"]
Padding = Literal["5_smt_pyfftw","5_smt_scupyrfft","pwr_2"]

def process_noise(p_id: str, r_id: str, c_id: str, base_dir: Path, padding:Padding, threads:int, Batch:int, reprocess=True) -> None:
    logging.info(f"Starting noise processing for p{p_id}_r{r_id}_C{c_id} with padding: {Padding}")
    try:
        #input
        raw_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        raw_file = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}" / f"C{c_id}--Trace.npy"
        metadata_path = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
        metadata_time_path = base_dir / "teststand_metadata" / "par" /"fft" / f"p{p_id}" / f"r{r_id}" / f"fft_metadata_p{p_id}_r{r_id}.json"
        logging.info(f"Input paths: raw_dir={raw_dir}, metadata_path={metadata_path}")
        
        #output
        plt_dir = base_dir / "generated_data" / "pyplt" / "noise" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        par_dir = base_dir / "generated_data" / "pypar" / "noise" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        # Create output directories if they don't exist
        plt_dir.mkdir(parents=True, exist_ok=True)
        par_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directories created: plt_dir={plt_dir}, par_dir={par_dir}")
        
        #load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        dt = metadata['C1--00000']['time_resolution']['dt']
        #FFT process
        logging.info(f"Running analysis mode: {padding}")
        if padding == "5_smt_pyfftw": #5 smothing
            freq, sdv, time = fft_funcs.fivesmt_pyfftw(raw_file, dt, threads,Batch)
        if padding == "5_smt_scupyrfft": #5 smothing
            freq, sdv, time = fft_funcs.fivesmt_scupyrfft(raw_file, dt, threads, Batch)
        elif padding == "pwr_2": #afft
            freq, sdv, time = fft_funcs.pwrtwo_fft(raw_file, dt, threads, Batch)
        else:
            logging.error(f"Invalid mode: {padding}. Please use '5_smt_pyfftw', '5_smt_scupyfft' or 'pwr_2'.")

        if sdv is not None:
            mean_sdv = sdv  #already calculated mean
            # 結果をプロット
            fig = plt.figure(figsize=(9, 6))
            plt.rcParams['font.size'] = 14
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.plot(freq, mean_sdv, color='red', linewidth=0.5)
            plt.grid(which='major')
            plt.grid(which='minor')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Mean Spectrum density [V / √Hz]")
            plt.title(f"Spectrum Analysis - p{p_id}_r{r_id}_C{c_id}")
        
            # Save plot to plt_dir
            logging.info("Creating plot")
            plot_file = plt_dir / f"spectrum_C{c_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {plot_file}")
            
            # Save data to par_dir
            logging.info("Saving data")
            data_file = par_dir / f"spectrum_data_C{c_id}.npz"
            np.savez(data_file,
                    frequencies=freq,
                    mean_spectrum_density=mean_sdv,
                    all_spectrum_density=sdv)
            logging.info(f"Data saved to {data_file}")
            
            # Show plot if in interactive mode
            plt.show()
            plt.close()  # Close the plot to avoid memory issues
            logging.info("Processing completed successfully")
        else:
            logging.error("No spectrum density data was collected. Check previous errors.")

        # save time information on metadata_time_path
        logging.info("Processing metadata...")
        meta = {
            "threads":threads,
            "batch":Batch,
            "time":time
        } 
        append_metadata(meta, metadata_time_path)
        logging.info(f'time data saved to {metadata_time_path}')
    except Exception as e:
        logging.error(f"Error in process_noise: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

def append_metadata(meta_dict: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # 既存のメタデータを読み込む
    if dest.exists():
        with dest.open("r", encoding="utf-8") as fh:
            try:
                existing_meta = json.load(fh)
                if isinstance(existing_meta, list):
                    existing_meta.append(meta_dict)
                elif isinstance(existing_meta, dict):
                    existing_meta = [existing_meta, meta_dict]
                else:
                    existing_meta = [meta_dict]
            except json.JSONDecodeError:
                existing_meta = [meta_dict]
    else:
        existing_meta = [meta_dict]
    
    # 更新されたメタデータを保存
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(existing_meta, fh, indent=2, ensure_ascii=False)
    logging.info("Metadata updated → %s (%d entries)", dest, len(existing_meta))