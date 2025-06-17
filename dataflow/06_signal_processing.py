"""
Comparison of different samples on frequency or time domain 
----------------------

Directory layout:
    npz data(freq domain)  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ/noise_data_CZ.npz
    npy data(time domain)  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ--Trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/compare
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes three layers:
    1. **Dynamic orchestration wrapper** -> 05_squid_analysis.py
    2. **Static processing functions** -> src/tes_analysis_tools.py (by Yuki Mitsuya)
------------------------

Author: Ryutaro Matsumoto - 2025-04-30
"""
from pathlib import Path
import sys
import os
import logging, json
import numpy as np
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_signal import process_signal

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"

# edit here
period1 = "05"          ##pulse##
run1 = "001"
channel1 = "2"          # Channel number  #For after P03,C1 is SQUID and C2 is HBT

period2 = "05"          ##noise##
run2 = "002"
channel2 = "2"

t_range=[4.5, 5.8]      #signal time range (µs) minimum : 0.5ns/pt = 0.0005µs/pt



if __name__ == "__main__":
    logging.info(f"Processing SN analysis for Pulse:p{period1}/r{run1}/C{channel1} and Noise: p{period2}/r{run2}/C{channel2} ")
    process_signal(
        p_id1 = period1, 
        r_id1 = run1, 
        c_id1 = channel1,  
        p_id2 = period2, 
        r_id2 = run2, 
        c_id2 = channel2,  
        t_range = t_range,
        base_dir=BASE_DIR
    )
    