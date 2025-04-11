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
from typing import Literal, Dict, List
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

# Define a function to process a single waveform
def process_waveform(wave_data, dt):
    #choose 
    results = fft_funcs.calc_sd_batch(
        wave_data[np.newaxis,:],
        dt)
    return results[0], results[1]

# Function to process a waveform at a specific index
def process_index(index, data_array, t_interval):
    wave_data = data_array[index]
    freq, spectrum = process_waveform(wave_data, t_interval)
    return index, freq, spectrum

def process_noise(p_id: str, r_id: str, c_id: str, base_dir: Path,fcpu:int, mode: Mode, reprocess=True) -> None:
    logging.info(f"Starting noise processing for p{p_id}_r{r_id}_C{c_id} with mode: {mode}")
    try:
        #input
        raw_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        raw_file = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}" / f"C{c_id}--Trace.npy"
        metadata_path = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
        
        logging.info(f"Input paths: raw_dir={raw_dir}, metadata_path={metadata_path}")
        
        #output
        plt_dir = base_dir / "generated_data" / "pyplt" / "noise" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        par_dir = base_dir / "generated_data" / "pypar" / "noise" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}"
        # Create output directories if they don't exist
        plt_dir.mkdir(parents=True, exist_ok=True)
        par_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directories created: plt_dir={plt_dir}, par_dir={par_dir}")
        
     
        #load array data from raw_dir
        data_array = np.load(raw_file, allow_pickle=True)
        logging.info(f"Data loaded successfully. Shape: {data_array.shape}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        t_interval = metadata['C1--00000']['time_resolution']['dt']
        freqs = []
        sdv = []
        
        # Determine the number of CPU cores to use (leave how many cores free)
        num_cores = max(1, multiprocessing.cpu_count() - fcpu)
        logging.info(f"Using {num_cores} CPU cores for parallel processing")

        # Quick or Full analysis
        if mode == "quick":
            logging.info(f"Running quick analysis mode (5 random samples)")
            # Randomly choose 5 data from data_array
            if data_array.shape[0] < 5:
                logging.warning(f"Data array has fewer than 5 samples ({data_array.shape[0]}). Using all available samples.")
                selected_indices = range(data_array.shape[0])
            else:
                selected_indices = random.sample(range(data_array.shape[0]), 5)
            
            logging.info(f"Selected indices for analysis: {selected_indices}")
            # Process waveforms in parallel
            
            # Process waveforms in parallel
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(process_index, index, data_array, t_interval) for index in selected_indices]
                
                for i, future in enumerate(futures):
                    index, freq, spectrum = future.result()
                    logging.info(f"Processing sample {i+1}/5 (index {index}) completed")
                    freqs.append(freq)
                    sdv.append(spectrum)
                    logging.info(f"Sample {i+1} processed successfully. Frequency array length: {len(freq)}")

        elif mode == "full":
            # logging.info(f"Running full analysis mode (all {data_array.shape[0]} samples)")
            # # Process waveforms in parallel
            
            # # Process all waveforms in parallel using chunks
            # total_samples = data_array.shape[0]
            # chunk_size = max(1, total_samples // (num_cores * 2))  # Process in chunks for better load balancing
            
            # for chunk_start in range(0, total_samples, chunk_size):
            #     chunk_end = min(chunk_start + chunk_size, total_samples)
            #     chunk_indices = list(range(chunk_start, chunk_end))
                
            #     logging.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_samples + chunk_size - 1)//chunk_size} (samples {chunk_start+1}-{chunk_end})")
                
            #     with ProcessPoolExecutor(max_workers=num_cores) as executor:
            #         futures = [executor.submit(process_index, i, data_array, t_interval) for i in chunk_indices]
                    
            #         for future in futures:
            #             index, freq, spectrum = future.result()
            #             freqs.append(freq)
            #             sdv.append(spectrum)
                        
            #             if index % 20 == 0:  # Log every 20th sample
            #                 logging.info(f"Sample {index+1} processed successfully. Frequency array length: {len(freq)}")
            
            #afft用　
            n_waveforms = data_array.shape[0]
            logging.info(f"Running full analysis mode (all {n_waveforms} samples)")
            # ---- 1. vDSP スレッドを全コアに設定 -----------------------------------
            afft.set_nthreads(os.cpu_count())

            # ---- 2. バッチ FFT を一発で回す ---------------------------------------
            freq, sdv_ = fft_funcs.calc_sd_batch_padded(data_array, t_interval)

            # ---- 3. 必要ならここで結果をリストに格納 ------------------------------
            # （旧コード互換のために list へコピー）
            freqs  = [freq] * n_waveforms
            sdv = [sdv_[i] for i in range(n_waveforms)]

            logging.info("All samples processed in a single batch")
            
        else:
            logging.error(f"Invalid mode: {mode}. Please use 'quick' or 'full'.")

        # スペクトル密度データを配列に変換
        logging.info(f"Processing collected data. Number of samples processed: {len(sdv)}")
        if sdv:
            logging.info("Converting spectrum density data to array")
            sdv_data = np.array(sdv)
            logging.info(f"Spectrum density data array shape: {sdv_data.shape}")
            
            # 各周波数でのスペクトル密度の平均を計算
            mean_sdv = np.mean(sdv_data, axis=0)
            logging.info(f"Mean spectrum density calculated. Length: {len(mean_sdv)}")
            # 結果をプロット
            fig = plt.figure(figsize=(9, 6))
            plt.rcParams['font.size'] = 14
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial']

            plt.plot(freqs[0], mean_sdv, color='red', linewidth=0.5)
            plt.grid(which='major')
            plt.grid(which='minor')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Mean Spectrum density [V / √Hz]")
            plt.title(f"Noise Spectrum Analysis - p{p_id}_r{r_id}_C{c_id}")
        
            # Save plot to plt_dir
            logging.info("Creating plot")
            plot_file = plt_dir / f"noise_spectrum_C{c_id}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {plot_file}")
            
            # Save data to par_dir
            logging.info("Saving data")
            data_file = par_dir / f"noise_data_C{c_id}.npz"
            np.savez(data_file,
                    frequencies=freqs[0],
                    mean_spectrum_density=mean_sdv,
                    all_spectrum_density=sdv_data)
            logging.info(f"Data saved to {data_file}")
            
            # Show plot if in interactive mode
            plt.close()  # Close the plot to avoid memory issues
            logging.info("Processing completed successfully")
        else:
            logging.error("No spectrum density data was collected. Check previous errors.")

    except Exception as e:
        logging.error(f"Error in process_noise: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())