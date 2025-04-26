"""
----------------------

Directory layout:
    npy traces  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ_trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/noise/pXX/rYYY/CZ
    params dir  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes two layers:
    1. **Static processing functions** -> process_noise.py
    2. **Dynamic orchestration wrapper** -> 02_noise_analysis.py

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
period = "01"
run = "006"
channels = ["1"]  # Channel number, add "Cn" if needed.
# analysis_mode: Mode = "full"  # "quick" or "full"
padding_mode: Padding = "5_smt_pyfftw" # Either of ["5_smt_pyfftw","5_smt_scupyrfft","pwr_2"]
CPU_THREADS =os.cpu_count()                    # NUM of CPUs used for FFT
Batch = 8                       # Num of lines which will be batched together
Reprocess_noise = True

if __name__ == "__main__":
    #perform noise analysis for each channel
    for channel in channels:
        logging.info(f"Processing noise analysis for C{channel}")
        process_noise(
            p_id=period,
            r_id=run,
            c_id=channel,
            base_dir=BASE_DIR,
            padding = padding_mode,
            threads = CPU_THREADS,
            Batch = Batch,
            reprocess=Reprocess_noise
        )
