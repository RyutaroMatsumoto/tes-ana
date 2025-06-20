"""
----------------------
Directory layout:
    npy traces  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ_trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/noise/pXX/rYYY/CZ
    params dir  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes two layers:
    1. **Static processing functions** -> process_wave.py
    2. **Dynamic orchestration wrapper** -> 02_t_domain_analysis.py

Author: Ryutaro Matsumoto - 2025-04-09
Updated: added pulse fit and time constant calculation function -2025-05-07
"""

from pathlib import Path
import sys
import os
import logging
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_wave_analysis import process_wave

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"

# edit here
period = "06"
run = "010"
channels = ["1",
            "2"
             #,"4"
            ]              # Channel number, add "Cn" if needed. For P03, C1=SQUID, C2=HBT, C4= Timing Trigger
row_index=134
show_single_wave = False
show_single_10 =False      #compare trap-on & trap-off for 10 single waves
show_sample_ave = True      #Either show_single or show_sample_ave should be True!
trap=False
t_range=[0,10]                  #graph display time range in µs
reprocess = False            #Must be true for the first time, false for just plot




if __name__ == "__main__":
    # perform noise analysis for each channel
    logging.info(f"Processing wave analysis for C{', '.join(channels)}")
    process_wave(
        p_id=period,
        r_id=run,
        c_ids=channels,
        base_dir=BASE_DIR,
        row_index=row_index,
        show_single_wave=show_single_wave,
        show_single_10 =show_single_10,
        show_sample_avg=show_sample_ave,
        trap=trap,
        t_range=t_range,
        reprocess=reprocess
    )
