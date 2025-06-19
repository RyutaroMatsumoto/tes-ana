from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import time
from src.tes_analysis_tools import correct_baseline

from processing_functions.lecroy import LecroyBinaryWaveform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

def parse_trc_file_wave(path: Path, dpbl: int) -> np.ndarray:
    logging.debug("Parsing %s", path)
    trc = LecroyBinaryWaveform(str(path))
    waveform = np.asarray(trc.wave_array_1, dtype=np.float32)
    dp = waveform.shape[0]
    logging.debug("Correcting baseline with %d samples, where total samples: %d", dpbl, dp)
    wave_corr = np.zeros(dp)

    wave_corr = waveform - np.average(waveform[0:dpbl])

    return wave_corr


def parse_trc_file_meta(path: Path) -> Dict[str, Any]:
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

def _write_channel_memmap(files: List[Path], out_dir: Path, flush_interval: int, max_workers: int, dpbl: int) -> None:
    if not files:
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    # Extract channel ID from the first file to create consistent output filename
    channel_id = files[0].name.split("--")[0]  # e.g., "C1", "C2", etc.
    out_path = out_dir / (channel_id + "--Trace.npy")
    # Extract channel ID from the first file to create consistent output filename
    channel_id = files[0].name.split("--")[0]  # e.g., "C1", "C2", etc.
    out_path = out_dir / (channel_id + "--Trace.npy")

    # --- Determine dtype & trace length from the first file ---
    first_wave = parse_trc_file_wave(files[0],dpbl)
    n_samples = first_wave.size
    dtype = first_wave.dtype

    # --- Pre‑allocate memory‑mapped output array ---
    mmap = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(len(files), n_samples)
    )
    mmap[0] = first_wave  # store the first trace we already loaded

    LOGGER.info(
        "Allocated memmap %s, shape=(%d, %d), dtype=%s",
        out_path, len(files), n_samples, dtype,
    )
    # Progress tracking
    start_time = time.perf_counter()
    total_traces = len(files)

    def _log_progress(done_idx: int) -> None:
        """Log progress every time we flush or finish."""
        elapsed_min = (time.perf_counter() - start_time) / 60.0
        LOGGER.info(
            "Processed %d / %d traces → %s (%.2f min elapsed)",
            done_idx + 1,  # +1 because idx is 0‑based
            total_traces,
            out_path.name,
            elapsed_min,
        )

    # --- Define loader helper for threaded mode ---
    def _loader(idx_path: Tuple[int, Path]) -> Tuple[int, np.ndarray]:
    def _loader(idx_path: Tuple[int, Path]) -> Tuple[int, np.ndarray]:
        idx, p = idx_path
        return idx, parse_trc_file_wave(p, dpbl)

    # --- Choose sequential vs threaded reading ---
    if max_workers > 0:
        LOGGER.info("Starting threaded prefetch with %d workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_loader, (i, p)): i for i, p in enumerate(files[1:], 1)}
            for f in as_completed(futures):
                i, wave = f.result()
                if wave.size != n_samples:
                    raise ValueError(
                        f"Sample length mismatch in {files[i].name}: "
                        f"{wave.size} vs {n_samples}"
                    )
                mmap[i] = wave
                if flush_interval and i % flush_interval == 0:
                    mmap.flush()
                    _log_progress(i)
    else:
        # Plain sequential loop (fast for SSD/NVMe)
        for i, p in enumerate(files[1:], 1):
            wave = parse_trc_file_wave(p, dpbl)
            if wave.size != n_samples:
                raise ValueError(
                    f"Sample length mismatch in {p.name}: {wave.size} vs {n_samples}"
                )
            mmap[i] = wave
            if flush_interval and i % flush_interval == 0:
                mmap.flush()
                _log_progress(i)

    mmap.flush()
    del mmap  # Close file handle
    LOGGER.info("Finished writing %s (%d traces)", out_path, len(files))    

def process_trc2npy(p_id: str, r_id: str, base_dir: Path, reprocess_waveform:bool , reprocess_metadata: bool, flush_interval: int, max_workers: int, dpbl: int, channels: Optional[List[str]] = None) -> None:
    # Find all directories matching the pattern r{r_id} and r{r_id}-n
    raw_base_dir = base_dir / "generated_data" / "raw_trc" / f"p{p_id}"
    out_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}"
    meta_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    
    # Collect all directories that match the pattern
    raw_dirs = []
    
    # First check the main directory r{r_id}
    main_raw_dir = raw_base_dir / f"r{r_id}"
    if main_raw_dir.is_dir():
        raw_dirs.append(main_raw_dir)
    
    # Then check for additional directories r{r_id}-n (n=2,3,4,...)
    n = 2
    while True:
        additional_dir = raw_base_dir / f"r{r_id}-{n}"
        if additional_dir.is_dir():
            raw_dirs.append(additional_dir)
            n += 1
        else:
            break
    
    if not raw_dirs:
        raise FileNotFoundError(f"No raw directories found matching pattern r{r_id} or r{r_id}-n in {raw_base_dir}")
    
    LOGGER.info("Found %d directories to process: %s", len(raw_dirs), [d.name for d in raw_dirs])
    
    # Collect all trc files from all directories with continuous numbering
    all_trc_files = []
    for raw_dir in raw_dirs:
        trc_files = sorted(raw_dir.glob("C*--Trace--*.trc"))
        if not trc_files:
            trc_files = sorted(raw_dir.glob("C*--wave--*.trc"))
        if trc_files:
            all_trc_files.extend(trc_files)
    
    if not all_trc_files:
        logging.warning("No .trc files found in any of the directories: %s", [str(d) for d in raw_dirs])
    
    # Collect all directories that match the pattern
    raw_dirs = []
    
    # First check the main directory r{r_id}
    main_raw_dir = raw_base_dir / f"r{r_id}"
    if main_raw_dir.is_dir():
        raw_dirs.append(main_raw_dir)
    
    # Then check for additional directories r{r_id}-n (n=2,3,4,...)
    n = 2
    while True:
        additional_dir = raw_base_dir / f"r{r_id}-{n}"
        if additional_dir.is_dir():
            raw_dirs.append(additional_dir)
            n += 1
        else:
            break
    
    if not raw_dirs:
        raise FileNotFoundError(f"No raw directories found matching pattern r{r_id} or r{r_id}-n in {raw_base_dir}")
    
    LOGGER.info("Found %d directories to process: %s", len(raw_dirs), [d.name for d in raw_dirs])
    
    # Collect all trc files from all directories with continuous numbering
    all_trc_files = []
    for raw_dir in raw_dirs:
        trc_files = sorted(raw_dir.glob("C*--Trace--*.trc"))
        if not trc_files:
            trc_files = sorted(raw_dir.glob("C*--wave--*.trc"))
        if trc_files:
            all_trc_files.extend(trc_files)
    
    if not all_trc_files:
        logging.warning("No .trc files found in any of the directories: %s", [str(d) for d in raw_dirs])
        return
    
    # Log which channels will be processed
    if channels is not None:
        LOGGER.info("Processing only specified channels: %s", channels)
    else:
        LOGGER.info("Processing all available channels")
    
    # Group files by channel and renumber them continuously
    channel_files: Dict[str, List[Path]] = {}
    for f in all_trc_files:
        c_id = f.name[1]
        # Only include files for specified channels (if channels parameter is provided)
        if channels is None or c_id in channels:
            channel_files.setdefault(c_id, []).append(f)
    
    # Sort files within each channel and renumber them continuously
    for c_id in channel_files:
        # Sort files by directory order and then by original file number
        def sort_key(file_path):
            # Extract directory index (0 for main r{r_id}, 1 for r{r_id}-2, etc.)
            dir_name = file_path.parent.name
            if dir_name == f"r{r_id}":
                dir_idx = 0
            else:
                # Extract n from r{r_id}-n
                dir_idx = int(dir_name.split('-')[-1]) - 1
            
            # Extract original file number
            file_parts = file_path.name.split('--')
            if len(file_parts) >= 3:
                file_num = int(file_parts[2].split('.')[0])
            else:
                file_num = 0
            
            return (dir_idx, file_num)
        
        channel_files[c_id] = sorted(channel_files[c_id], key=sort_key)
        LOGGER.info("Channel C%s: %d files collected from %d directories",
                   c_id, len(channel_files[c_id]), len(raw_dirs))
    
    # Check if we have any files to process after filtering
    if not channel_files:
        if channels is not None:
            logging.warning("No files found for specified channels %s in directories: %s",
                          channels, [str(d) for d in raw_dirs])
        else:
            logging.warning("No channel files found in directories: %s", [str(d) for d in raw_dirs])
        return
    
    if reprocess_waveform:
        logging.info("Processing waveforms...")
        # Process each channel separately
        for c_id, files in channel_files.items():
          LOGGER.info("→ Channel C%s: %d traces", c_id, len(files))
          _write_channel_memmap(
                files, out_dir / f"C{c_id}", flush_interval, max_workers, dpbl
            )
            
    if reprocess_metadata:
        logging.info("Processing metadata...")
        meta_all = {}
        for c_id, files in channel_files.items():
            first_file = next((f for f in files if "--00000" in f.name), files[0])
            meta = parse_trc_file_meta(first_file)
            meta_all["C" + c_id + "--00000"] = meta

        append_metadata(meta_all, meta_path)

    logging.info("✓ Run p%s r%s completed", p_id, r_id)

