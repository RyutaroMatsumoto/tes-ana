"""
Comparison of different samples on frequency or time domain 
----------------------

Directory layout:
    npz data(freq domain)  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ/noise_data_CZ.npz
    npy data(time domain)  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ--Trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/compare
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes three layers:
    1. **Dynamic orchestration wrapper** -> 04_compare_analysis.py
    2. **Static processing functions** -> process_compare.py
------------------------

Author: Ryutaro Matsumoto - 2025-04-30
"""
from pathlib import Path
import sys
import os
import logging
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_compare import process_compare, domain

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"


# edit here
period1 = "03"
run1 = "002"
channels1 = "1"  # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT

period2 = "03"
run2 = "004"
channels2 = "1"  # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT

domain = "freq"    # "time" or "freq" domain, either domain you want to compare

if __name__ == "__main__":
    logging.info(f"Processing compare analysis for {period1}/{run1}/{channels1} and {period2}{run2}{channels2}")
    process_compare(
        p_id1=period1,
        r_id1=run1,
        c_id1=channels1,
        p_id2=period2,
        r_id2=run2,
        c_id2=channels2,
        base_dir=BASE_DIR,
        domain = domain
    )
