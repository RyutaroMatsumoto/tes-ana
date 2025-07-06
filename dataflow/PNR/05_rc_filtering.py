"""
----------------------
Directory layout:
    npy traces  : ../../tes01/generated_data/raw/pXX/rYYY/CZ/CZ_trace.npy
    plot dir    : ../../tes01/generated_data/pyplt/noise/pXX/rYYY/CZ
    params dir  : ../../tes01/generated_data/pypar/noise/pXX/rYYY/CZ
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes two layers:
    1. **Static processing functions** -> process_wave.py
    2. **Dynamic orchestration wrapper** -> 05_rc_filtering.py

Author: Ryutaro Matsumoto - 2025-04-09
Updated: added pulse fit and time constant calculation function -2025-05-07
"""

from pathlib import Path
import sys
import os
import logging
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from processing_functions.process_rcfilt import process_rcfilt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tes01"

# edit here
period = "07"
run = "009"
channels = ["1",
            #"2"
            # ,"4"
            ]                # Channel number, add "Cn" if needed. For P03, C1=SQUID, C2=HBT, C4= Timing Trigger
row_index=1005
rc = 1.0e-6                  # tau for LP filtering
t_range=[0,20]               #graph display time range in µs
reprocess = True             #Must be true for the first time, false for just plot




if __name__ == "__main__":
    # perform noise analysis for each channel
    logging.info(f"Processing rcfilt for C{', '.join(channels)}")
    process_rcfilt(
        p_id=period,
        r_id=run,
        c_ids=channels,
        base_dir=BASE_DIR,
        rc=rc,
        t_range=t_range,
        reprocess=reprocess
    )
