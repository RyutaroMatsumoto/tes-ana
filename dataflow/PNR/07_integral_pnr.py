"""
PNR analysis using integral of SQUID signals instead of optimal filter.
(Using Energy for classification, in terms of physics)
----------------------

Directory layout:
    npy data(time domain)  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ--Trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/integral
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes three layers:
    1. **Dynamic orchestration wrapper** -> 07_squid_analysis.py
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
from processing_functions.process_optfit import process_optfit

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tes01"

# edit here
#pulse
period1 = "06"
run1 = "001"
channel1 = "1"  # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT
#noise
period2 = "06"
run2 = "002"
channel2 = "1"

#Options
Verbose = True #log

if __name__ == "__main__":
    logging.info(f"Processing SQUID analysis for Pulse:p{period1}/r{run1}/C{channel1} and Noise: p{period2}/r{run2}/C{channel2} ")
    process_integral(
        period1 = period1, 
        run1 = run1, 
        channel1 = channel1,  
        period2 = period2, 
        run2 = run2, 
        channel2 = channel2,  
        cnr = cnr,
        maxfreq = Max_freq, 
        phmin = phmin,
        phmax = phmax,
        timin = timin,
        timax = timax,
        normalize = normalize,
        verbose = Verbose,
        base_dir=BASE_DIR
    )
