"""
----------------------
Utility script to convert Teledyne-LeCroy *.trc binary waveform files into

* one *.npy file containing the raw waveform samples for each trace file
* a single JSON file holding the metadata for the whole run

Directory layout:
    Raw traces  : ../../tes01/generated_data/raw_trc/pXX/rYYY/CZ--Trace--00000.trc
    Waveform npy: ../../tes01/generated_data/raw/pXX/rYYY/CZ--Trace.npy
    Metadata    : ../../tes01/teststand_metadata/hardware/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY_CZ.json

The script exposes two layers:
    1. **Static processing functions** -> process_trc2npy.py
    2. **Dynamic orchestration wrapper** -> 01_trc2npy.py

Author: Ryutaro Matsumoto - 2025-04-09
Updated: added baseline correction function 2025-05-06 

"""

from pathlib import Path
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_trc2npy import process_trc2npy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01_link"         #tes01 for local directory, tes01_link for SSD directory

# edit here
period = "01"
run = "004"
REPROCESS_WAVEFORM = True          #True for reprocess, False for skip process for waveforms
REPROCESS_METADATA = True          #True for reprocess, False for skip process for metadata
flush = 10                         #Frequency of writing enforcement for RAM clear (set 0 for inside SSD data:tes01)
threads = 0                        #Threads used for loading (0 is suitable for NVMeSSD or RAM-Disk, if slow, use 2 -> 4)
dpbl = 1000                        #Data points used for Baseline correction must ends before the signal


if __name__ == "__main__":
    process_trc2npy(
        p_id=period,
        r_id=run,
        base_dir=BASE_DIR,
        reprocess_waveform=REPROCESS_WAVEFORM,
        reprocess_metadata=REPROCESS_METADATA,
        flush_interval=flush,
        max_workers=threads,
        dpbl = dpbl
    )
