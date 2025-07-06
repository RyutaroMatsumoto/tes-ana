from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import time
from src.tes_analysis_tools import correct_baseline, rc_int
import matplotlib.pyplot as plt
from processing_functions.lecroy import LecroyBinaryWaveform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
def process_rcfilt(p_id: str, r_id: str, c_ids: list, base_dir: Path, rc, t_range, reprocess) -> None:
    """
    load npy file 
    rc filtering 
    save rcfilt npy file
    """
    #load general data
    raw_dir = base_dir/"generated_data"/"raw"/f"p{p_id}"/f"r{r_id}"
    meta_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata does not exist: {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    dt = metadata[f"C{c_ids[0]}--00000"]['time_resolution']['dt']

    c_dirs = []
    for item in raw_dir.iterdir():
        if item.is_dir() and item.name.startswith('C') and item.name[1:].isdigit():
            c_id = item.name[1:]
            if c_id in c_ids:
                c_dirs.append(item)
    
    if not c_dirs:
        logging.warning(f"No directories matching C{{{','.join(c_ids)}}} found in {raw_dir}")
        return
    # 保存先ディレクトリの設定
    plt_dir = base_dir / "generated_data" / "pyplt" / "rcfilt" / f"p{p_id}" / f"r{r_id}"
    plt_dir.mkdir(parents=True, exist_ok=True)
    par_dir = base_dir / "generated_data" / "pypar" / "rcfilt" / f"p{p_id}" / f"r{r_id}"
    par_dir.mkdir(parents=True, exist_ok=True)

    #rcfilt
    for c_dir in c_dirs:
        #load 
        logging.info("data loading...")
        c_id = c_dir.name[1:]
        data_path = raw_dir/f"C{c_id}"/f"C{c_id}--Trace.npy"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {data_path}")
        data = np.load(data_path)

        #rcfilt
        rcfilt_data = np.zeros_like(data)
        for row_index in range(data.shape[0]):
            wave_raw = data[row_index,:]
            wave_rcfilt = rc_int(wave_raw, rc, dt)
            rcfilt_data[row_index, :] = wave_rcfilt

        #plot
        plt.figure(figsize=(10, 5))
        dataname = f"rcfilt_waveform_p{p_id}_r{r_id}_s1_C{c_id}"
        time_data = np.arange(len(rcfilt_data[0])) * dt
        t_range_seconds = [t_range[0] * 1e-6, t_range[1] * 1e-6]

        if c_id == '1':
            color = 'r'  # c1は赤
        elif c_id == '2':
            color = 'b'  # c2は青
        else:
            color = 'g'  
        plt.plot(time_data, rcfilt_data[0], marker='', linestyle='-', color=color, label=f'C{c_id} Data')
        plt.title(f'{dataname} (t_range: {t_range[0]}-{t_range[1]} μs)')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid(True)
        # 表示範囲を設定
        if max(time_data) > t_range_seconds[1]:
            plt.xlim(t_range_seconds)
            
        else:
            logging.warning(f"Time range {t_range} μs exceeds available data range (0-{max(time_data)*1e6:.1f} μs) for C{c_id}")
            logging.info(f"Plotted sample data from C{c_id}")

        plt.show()

        #save
        out_path = par_dir/("C" + c_id + "--Trace.npy")
        save_waveform(rcfilt_data,out_path)
        logging.info(f"saved data to {out_path}")




def save_waveform(array: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, array)
    logging.debug("Saved waveform → %s", dest)
