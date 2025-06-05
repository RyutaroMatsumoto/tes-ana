"""
----------------------

Directory layout:
    npy traces  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ_trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/noise/pXX/rYYY/CZ
    params dir  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes three layers:
    1. **Dynamic orchestration wrapper** -> 03_f_domain_analysis.py
    2. **Static processing functions** -> process_noise.py
    3. **src fft functions** -> fft_funcs.py

------------------------
fft config optimization
04/26/2025
for P01,r006 (50MS*104samples) Scupyrfft was fastest with M3 iMac.
threads= 6, batch = 10 
57s for fft process
------------------------

Author: Ryutaro Matsumoto - 2025-04-09
"""

from pathlib import Path
import sys
import os
import logging
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_noise_analysis import process_noise, Mode, Padding

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"


# edit here
period = "05"
run = "001"
channels = ["2"]  # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT

#analysis Parameters 
padding_mode: Padding = "5_smt_scupyrfft" # Either of ["5_smt_pyfftw","5_smt_scupyrfft","pwr_2"]
CPU_THREADS =os.cpu_count()                    # NUM of CPUs used for FFT
Batch = 16                       # Num of lines which will be batched together
Reprocess_noise = True


if __name__ == "__main__":
    for channel in channels:
        logging.info(f"Processing noise analysis for C{channel} with threads:{CPU_THREADS} and Batch:{Batch}")
        process_noise(
            p_id=period,
            r_id=run,
            c_id=channel,
            base_dir=BASE_DIR,
            padding=padding_mode,
            threads=CPU_THREADS,
            Batch=Batch,
            reprocess=Reprocess_noise
        )


    #perform noise analysis for each channel
    # for threads_run in range(1, CPU_THREADS + 1):
    #     for Batch_run in range(1, Batch + 1):
    #         for channel in channels:
    #             logging.info(f"Processing noise analysis for C{channel} with threads:{threads_run} and Batch:{Batch_run}")
    #             process_noise(
    #                 p_id=period,
    #                 r_id=run,
    #                 c_id=channel,
    #                 base_dir=BASE_DIR,
    #                 padding=padding_mode,
    #                 threads=threads_run,
    #                 Batch=Batch_run,
    #                 reprocess=Reprocess_noise
    #             )
