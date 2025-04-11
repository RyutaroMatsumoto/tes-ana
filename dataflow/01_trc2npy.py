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

# edit here
period = "01"
run = "007"
REPROCESS_WAVEFORM = True          #True for reprocess, False for skip process for waveforms
REPROCESS_METADATA = True          #True for reprocess, False for skip process for metadata

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"

if __name__ == "__main__":
    process_trc2npy(
        p_id=period,
        r_id=run,
        base_dir=BASE_DIR,
        reprocess_waveform=REPROCESS_WAVEFORM,
        reprocess_metadata=REPROCESS_METADATA,
    )
