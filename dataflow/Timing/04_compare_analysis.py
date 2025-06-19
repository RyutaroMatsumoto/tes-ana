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
period1 = "06"
run1 = "005"
channels1 = "2"             # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT

period2 = "06"
run2 = "006"
channels2 = "2"             # Channel number, add "Cn" if needed.  #For P03,C1 is SQUID and C2 is HBT

domain = "time"             # "time" or "freq" domain, either domain you want to compare
fitting = False              #True for fitting only when domain = time.
fitting_samples = 500       #Sampling points used for fitting around amplitude max point.
p_init = [11.5, 35, 0.015]  #initial value for fitting. (rise time(0-100%), fall time(0-100%), amplitude) params in ns.
traces = [
    #"channels1",
    #"channels2",
    "differential"    #choose which trace want to show in the fig
]
notch = False            #notch 
notch_range = [         #notch range [Hz, Q-value] Q-value(sharpness of notching) should be <100
    (50e6,10),
    (45.5e6,20),
    (125e6,20),
    (200e6,10)
    ]   
if __name__ == "__main__":
    logging.info(f"Processing compare analysis for p{period1}/r{run1}/C{channels1} and p{period2}/r{run2}/C{channels2}")
    process_compare(
        p_id1=period1,
        r_id1=run1,
        c_id1=channels1,
        p_id2=period2,
        r_id2=run2,
        c_id2=channels2,
        base_dir=BASE_DIR,
        domain = domain,
        fitting = fitting,
        fitting_samples = fitting_samples,
        p_init = p_init,
        traces = traces,
        notch = notch,
        notch_range = notch_range
    )
