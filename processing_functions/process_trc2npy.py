import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np

from processing_functions.lecroy import LecroyBinaryWaveform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_trc_file_wave(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    logging.debug("Parsing %s", path)
    trc = LecroyBinaryWaveform(str(path))
    
    waveform = np.asarray(trc.wave_array_1, dtype=np.float32)
    
    return waveform

def parse_trc_file_meta(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    logging.debug("Parsing %s", path)
    trc = LecroyBinaryWaveform(str(path))
    
    waveform = np.asarray(trc.wave_array_1, dtype=np.float32)
    
    metadata = {
        #"path": trc._inputfilename,
        "overview":{
            "instrument": trc.INSTRUMENT_NAME,
            "trigger_time": str(trc.TRIG_TIME),
            "channel": trc.WAVE_SOURCE,
            "coupling": str(trc.VERT_COUPLING),
            "sweep": str(trc.RECORD_TYPE),
            "samples": waveform.size,
            
        },
        "time_resolution":{
            "dt": float(trc.HORIZ_INTERVAL),
            "t_offset": float(trc.HORIZ_OFFSET),
            "t_uncertainty": float(trc.HORIZ_UNCERTAINTY),
            "sparsing_factor": int(trc.SPARSING_FACTOR),
            "t_unit": trc.HORUNIT,
        },
        "wave_resolution":{
            "v_over_bit": float(trc.VERTICAL_GAIN),
            "v_offset": float(trc.VERTICAL_OFFSET),
            "total_bits": int(trc.NOMINAL_BITS),
            "comm_type": int(trc.COMM_TYPE),
            "sweeps_per_acq": int(trc.SWEEPS_PER_ACQ),
            "v_unit": trc.VERTUNIT,
        },
        "GUI":{
            "fixed_vertical_gain": str(trc.FIXED_VERT_GAIN),
            "time_base": str(trc.TIMEBASE),
            "bandwidth_limit": str(trc.BANDWIDTH_LIMIT),
            "acq_vert_offset": float(trc.ACQ_VERT_OFFSET),
            "probe_att": float(trc.PROBE_ATT),
        },
        "other_information":{
            "acq_duration": float(trc.ACQ_DURATION),
            "vertical_vernier": float(trc.VERTICAL_VERNIER),
        }
    }
    return metadata
    
def save_waveform(array: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, array)
    logging.debug("Saved waveform → %s", dest)

def append_metadata(meta_dict: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, indent=2, ensure_ascii=False)
    logging.info("Metadata written → %s (%d channels)", dest, len(meta_dict))

def process_trc2npy(p_id: str, r_id: str, base_dir: Path, reprocess_waveform=True, reprocess_metadata=True) -> None:
    raw_dir = base_dir / "generated_data" / "raw_trc" / f"p{p_id}" / f"r{r_id}"
    out_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}"
    meta_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    trc_files = sorted(raw_dir.glob("C*--Trace--*.trc"))
    if not trc_files:
        trc_files = sorted(raw_dir.glob("C*--wave--*.trc"))
    if not trc_files:
        logging.warning("No .trc files found in %s", raw_dir)
        return
    
    channel_files = {}
    for f in trc_files:
        c_id = f.name[1]
        channel_files.setdefault(c_id, []).append(f)
    
    if reprocess_waveform:
        logging.info("Processing waveforms...")
        
        # Process each channel separately
        for c_id, files in channel_files.items():
            logging.info(f"Processing {len(files)} files for channel C{c_id}")
            
            # Directly process .trc files into memory without saving individual .npy files
            data_list = []
            for file_path in files:
                waveform = parse_trc_file_wave(file_path)
                data_list.append(waveform)
            
            # Combine all data into a single array
            data_array = np.array(data_list)
            logging.info(f"Processed {len(data_list)} files for channel C{c_id}, shape: {data_array.shape}, dtype: {data_array.dtype}")
            
            # Create directory if it doesn't exist
            (out_dir / f"C{c_id}").mkdir(parents=True, exist_ok=True)
            logging.info("Saving npy file ...")
            save_waveform(data_array, out_dir / f"C{c_id}" / f"C{c_id}--Trace.npy")
            logging.info(f"Saved npy file to {out_dir / f'C{c_id}' / f'C{c_id}--Trace.npy'}")
    if reprocess_metadata:
        logging.info("Processing metadata...")
        meta_all = {}
        for c_id, files in channel_files.items():
            first_file = next((f for f in files if "--00000" in f.name), files[0])
            meta = parse_trc_file_meta(first_file)
            meta_all["C" + c_id + "--00000"] = meta

        append_metadata(meta_all, meta_path)

    logging.info("✓ Run p%s r%s completed", p_id, r_id)
