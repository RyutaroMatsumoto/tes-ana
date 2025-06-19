from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import json
import logging
import mmap
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import time
import gc
import psutil
from contextlib import contextmanager
from src.tes_analysis_tools import correct_baseline

from processing_functions.lecroy import LecroyBinaryWaveform

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

def json_serialize_numpy(obj):
    """JSON serializer for numpy data types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class MemoryMonitor:
    """Monitor system memory usage and provide warnings"""
    
    def __init__(self, warning_threshold: float = 0.85):
        self.warning_threshold = warning_threshold
        self.process = psutil.Process()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        system_mem = psutil.virtual_memory()
        process_mem = self.process.memory_info()
        
        return {
            'system_used_gb': system_mem.used / (1024**3),
            'system_available_gb': system_mem.available / (1024**3),
            'system_percent': system_mem.percent / 100.0,
            'process_rss_gb': process_mem.rss / (1024**3),
            'process_vms_gb': process_mem.vms / (1024**3)
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        mem_info = self.get_memory_info()
        return mem_info['system_percent'] > self.warning_threshold
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status"""
        mem_info = self.get_memory_info()
        LOGGER.info(
            f"Memory Status {context}: System {mem_info['system_used_gb']:.1f}GB/"
            f"{mem_info['system_available_gb']:.1f}GB ({mem_info['system_percent']:.1%}), "
            f"Process RSS: {mem_info['process_rss_gb']:.1f}GB"
        )

@contextmanager
def mmap_file(filepath: Path, mode: str = 'r'):
    """Context manager for memory-mapped file access"""
    try:
        with open(filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                yield mm
    except Exception as e:
        LOGGER.error(f"Failed to mmap file {filepath}: {e}")
        raise

class OptimizedLecroyReader:
    """Optimized TRC file reader using memory mapping and caching"""
    
    def __init__(self, filepath: Path, use_mmap: bool = True):
        self.filepath = filepath
        self.use_mmap = use_mmap
        self._header_cache = None
        self._metadata_cache = None
    
    def _get_header_info(self) -> Dict[str, Any]:
        """Get header information with caching"""
        if self._header_cache is not None:
            return self._header_cache
        
        # Use standard lecroy reader for header parsing (small overhead)
        trc = LecroyBinaryWaveform(str(self.filepath))
        
        self._header_cache = {
            'wave_array_size': trc._WAVE_ARRAY_1_SIZE,
            'payload_offset': trc._payload_offset,
            'comm_type': trc.COMM_TYPE,
            'vertical_gain': trc.VERTICAL_GAIN,
            'vertical_offset': trc.VERTICAL_OFFSET,
            'samples': trc._WAVE_ARRAY_1_SIZE if trc.COMM_TYPE == 0 else trc._WAVE_ARRAY_1_SIZE // 2,
            'hifirst': trc.hifirst,
            'metadata': trc.metadata
        }
        
        return self._header_cache
    
    def read_waveform_mmap(self, dpbl: int) -> np.ndarray:
        """Read waveform data using memory mapping for better performance"""
        header = self._get_header_info()
        
        if self.use_mmap and self.filepath.stat().st_size > 50 * 1024 * 1024:  # Use mmap for files > 50MB
            return self._read_waveform_mmap_impl(header, dpbl)
        else:
            return self._read_waveform_standard(header, dpbl)
    
    def _read_waveform_mmap_impl(self, header: Dict[str, Any], dpbl: int) -> np.ndarray:
        """Implementation using memory mapping"""
        with mmap_file(self.filepath) as mm:
            # Determine data type and format
            if header['comm_type'] == 0:
                dtype = np.int8
                fmt = '>i1' if header['hifirst'] else '<i1'
            else:
                dtype = np.int16
                fmt = '>i2' if header['hifirst'] else '<i2'
            
            # Read raw data from memory map
            offset = header['payload_offset']
            nbytes = header['wave_array_size']
            
            # Create numpy array from memory map slice
            raw_data = np.frombuffer(mm[offset:offset + nbytes], dtype=fmt)
            
            # Convert to voltage values
            waveform = (header['vertical_gain'] * raw_data.astype(np.float32) - 
                       header['vertical_offset'])
            
            # Apply baseline correction
            if dpbl > 0 and dpbl < len(waveform):
                baseline = np.mean(waveform[:dpbl])
                waveform = waveform - baseline
            
            return waveform
    
    def _read_waveform_standard(self, header: Dict[str, Any], dpbl: int) -> np.ndarray:
        """Fallback to standard reading method"""
        trc = LecroyBinaryWaveform(str(self.filepath))
        waveform = np.asarray(trc.wave_array_1, dtype=np.float32)
        
        if dpbl > 0 and dpbl < len(waveform):
            baseline = np.mean(waveform[:dpbl])
            waveform = waveform - baseline
        
        return waveform
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata with caching"""
        if self._metadata_cache is not None:
            return self._metadata_cache
        
        header = self._get_header_info()
        trc_meta = header['metadata']
        
        # Convert to the expected format
        self._metadata_cache = {
            "overview": {
                "instrument": trc_meta.get('INSTRUMENT_NAME', ''),
                "trigger_time": str(trc_meta.get('TRIG_TIME', '')),
                "channel": trc_meta.get('WAVE_SOURCE', ''),
                "coupling": str(trc_meta.get('VERT_COUPLING', '')),
                "sweep": str(trc_meta.get('RECORD_TYPE', '')),
                "samples": header['samples'],
            },
            "time_resolution": {
                "dt": float(trc_meta.get('HORIZ_INTERVAL', 0)),
                "t_offset": float(trc_meta.get('HORIZ_OFFSET', 0)),
                "t_uncertainty": float(trc_meta.get('HORIZ_UNCERTAINTY', 0)),
                "sparsing_factor": int(trc_meta.get('SPARSING_FACTOR', 1)),
                "t_unit": trc_meta.get('HORUNIT', ''),
            },
            "wave_resolution": {
                "v_over_bit": float(trc_meta.get('VERTICAL_GAIN', 0)),
                "v_offset": float(trc_meta.get('VERTICAL_OFFSET', 0)),
                "total_bits": int(trc_meta.get('NOMINAL_BITS', 8)),
                "comm_type": int(trc_meta.get('COMM_TYPE', 0)),
                "sweeps_per_acq": int(trc_meta.get('SWEEPS_PER_ACQ', 1)),
                "v_unit": trc_meta.get('VERTUNIT', ''),
            },
            "GUI": {
                "fixed_vertical_gain": str(trc_meta.get('FIXED_VERT_GAIN', '')),
                "time_base": str(trc_meta.get('TIMEBASE', '')),
                "bandwidth_limit": str(trc_meta.get('BANDWIDTH_LIMIT', '')),
                "acq_vert_offset": float(trc_meta.get('ACQ_VERT_OFFSET', 0)),
                "probe_att": float(trc_meta.get('PROBE_ATT', 1)),
            },
            "other_information": {
                "acq_duration": float(trc_meta.get('ACQ_DURATION', 0)),
                "vertical_vernier": float(trc_meta.get('VERTICAL_VERNIER', 1)),
            }
        }
        
        return self._metadata_cache

def parse_trc_file_wave_optimized(path: Path, dpbl: int, use_mmap: bool = True) -> np.ndarray:
    """Optimized TRC file parsing with memory mapping"""
    LOGGER.debug("Parsing %s with mmap=%s", path, use_mmap)
    
    reader = OptimizedLecroyReader(path, use_mmap=use_mmap)
    return reader.read_waveform_mmap(dpbl)

def parse_trc_file_meta_optimized(path: Path) -> Dict[str, Any]:
    """Optimized metadata parsing with caching"""
    LOGGER.debug("Parsing metadata %s", path)
    
    reader = OptimizedLecroyReader(path, use_mmap=False)  # Metadata doesn't need mmap
    return reader.get_metadata()

def save_waveform_chunked(array: np.ndarray, dest: Path, chunk_size: int = 1000) -> None:
    """Save waveform with chunked writing for large arrays"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if array.ndim == 1 or array.shape[0] <= chunk_size:
        # Small array, save normally
        np.save(dest, array)
    else:
        # Large array, use memory-mapped saving
        mmap_array = np.lib.format.open_memmap(
            dest, mode='w+', dtype=array.dtype, shape=array.shape
        )
        
        # Copy in chunks to avoid memory pressure
        for i in range(0, array.shape[0], chunk_size):
            end_idx = min(i + chunk_size, array.shape[0])
            mmap_array[i:end_idx] = array[i:end_idx]
            
            if i % (chunk_size * 10) == 0:  # Flush every 10 chunks
                mmap_array.flush()
        
        mmap_array.flush()
        del mmap_array
    
    LOGGER.debug("Saved waveform → %s", dest)

def _write_channel_memmap_optimized(
    files: List[Path], 
    out_dir: Path, 
    flush_interval: int, 
    max_workers: int, 
    dpbl: int,
    memory_monitor: MemoryMonitor,
    use_mmap: bool = True,
    batch_size: int = 100
) -> None:
    """Optimized memory-mapped writing with better memory management"""
    
    if not files:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    channel_id = files[0].name.split("--")[0]
    out_path = out_dir / (channel_id + "--Trace.npy")

    # Determine dtype & trace length from the first file
    memory_monitor.log_memory_status("before first file read")
    first_wave = parse_trc_file_wave_optimized(files[0], dpbl, use_mmap=use_mmap)
    n_samples = first_wave.size
    dtype = first_wave.dtype
    
    # Calculate memory requirements
    total_size_gb = (len(files) * n_samples * np.dtype(dtype).itemsize) / (1024**3)
    LOGGER.info(f"Estimated output size: {total_size_gb:.2f} GB for {len(files)} traces")
    
    # Check if we have enough memory
    mem_info = memory_monitor.get_memory_info()
    if total_size_gb > mem_info['system_available_gb'] * 0.8:
        LOGGER.warning(f"Large dataset ({total_size_gb:.1f}GB) may cause memory pressure. "
                      f"Available: {mem_info['system_available_gb']:.1f}GB")

    # Pre-allocate memory-mapped output array
    mmap_array = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype, shape=(len(files), n_samples)
    )
    mmap_array[0] = first_wave
    del first_wave  # Free memory immediately
    gc.collect()

    LOGGER.info(
        "Allocated memmap %s, shape=(%d, %d), dtype=%s, size=%.2f GB",
        out_path, len(files), n_samples, dtype, total_size_gb
    )

    # Progress tracking
    start_time = time.perf_counter()
    total_traces = len(files)
    processed_count = 1  # Already processed first file

    def _log_progress(done_idx: int) -> None:
        elapsed_min = (time.perf_counter() - start_time) / 60.0
        rate = done_idx / elapsed_min if elapsed_min > 0 else 0
        eta_min = (total_traces - done_idx) / rate if rate > 0 else 0
        
        LOGGER.info(
            "Processed %d / %d traces → %s (%.2f min elapsed, %.1f traces/min, ETA: %.1f min)",
            done_idx + 1, total_traces, out_path.name, elapsed_min, rate, eta_min
        )
        memory_monitor.log_memory_status(f"after {done_idx + 1} traces")

    # Batch processing function
    def process_batch(batch_files: List[Tuple[int, Path]]) -> List[Tuple[int, np.ndarray]]:
        """Process a batch of files"""
        results = []
        for idx, path in batch_files:
            try:
                wave = parse_trc_file_wave_optimized(path, dpbl, use_mmap=use_mmap)
                if wave.size != n_samples:
                    raise ValueError(f"Sample length mismatch in {path.name}: {wave.size} vs {n_samples}")
                results.append((idx, wave))
            except Exception as e:
                LOGGER.error(f"Error processing {path}: {e}")
                raise
        return results

    # Choose processing strategy based on system resources and file count
    remaining_files = list(enumerate(files[1:], 1))
    
    if max_workers > 0 and len(remaining_files) > batch_size:
        # Threaded processing with batching
        LOGGER.info(f"Starting batched threaded processing: {max_workers} workers, batch size {batch_size}")
        
        # Process in batches to control memory usage
        for batch_start in range(0, len(remaining_files), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_files))
            batch = remaining_files[batch_start:batch_end]
            
            # Check memory pressure before each batch
            if memory_monitor.check_memory_pressure():
                LOGGER.warning("Memory pressure detected, forcing garbage collection")
                gc.collect()
                mmap_array.flush()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch processing jobs
                batch_chunks = [batch[i:i+max_workers] for i in range(0, len(batch), max_workers)]
                
                for chunk in batch_chunks:
                    future = executor.submit(process_batch, chunk)
                    results = future.result()
                    
                    # Store results
                    for idx, wave in results:
                        mmap_array[idx] = wave
                        processed_count += 1
                        
                        if flush_interval and processed_count % flush_interval == 0:
                            mmap_array.flush()
                            _log_progress(processed_count - 1)
                            
                            # Force garbage collection periodically
                            if processed_count % (flush_interval * 5) == 0:
                                gc.collect()
    else:
        # Sequential processing for smaller datasets or when threading is disabled
        LOGGER.info("Using sequential processing")
        
        for i, path in remaining_files:
            wave = parse_trc_file_wave_optimized(path, dpbl, use_mmap=use_mmap)
            if wave.size != n_samples:
                raise ValueError(f"Sample length mismatch in {path.name}: {wave.size} vs {n_samples}")
            
            mmap_array[i] = wave
            processed_count += 1
            
            if flush_interval and processed_count % flush_interval == 0:
                mmap_array.flush()
                _log_progress(processed_count - 1)
                
                # Check memory and clean up periodically
                if processed_count % (flush_interval * 2) == 0:
                    if memory_monitor.check_memory_pressure():
                        gc.collect()

    # Final flush and cleanup
    mmap_array.flush()
    memory_monitor.log_memory_status("before final cleanup")
    del mmap_array
    gc.collect()
    
    LOGGER.info("✓ Finished writing %s (%d traces)", out_path, len(files))

def process_trc2npy_optimized(
    p_id: str,
    r_id: str,
    base_dir: Path,
    reprocess_waveform: bool,
    reprocess_metadata: bool,
    flush_interval: int,
    max_workers: int,
    dpbl: int,
    use_mmap: bool = True,
    batch_size: int = 100,
    memory_warning_threshold: float = 0.85,
    channels: Optional[List[str]] = None
) -> None:
    """
    Optimized TRC to NPY conversion with advanced memory management
    
    Args:
        p_id: Period ID
        r_id: Run ID
        base_dir: Base directory path
        reprocess_waveform: Whether to reprocess waveform data
        reprocess_metadata: Whether to reprocess metadata
        flush_interval: How often to flush data to disk
        max_workers: Number of worker threads (0 for sequential)
        dpbl: Data points for baseline correction
        use_mmap: Whether to use memory mapping for TRC files
        batch_size: Batch size for processing (controls memory usage)
        memory_warning_threshold: Memory usage threshold for warnings (0.0-1.0)
        channels: List of channel IDs to process (e.g., ["1", "2"]). If None, all channels are processed.
    """
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(memory_warning_threshold)
    memory_monitor.log_memory_status("at start")
    
    # Set up paths
    raw_base_dir = base_dir / "generated_data" / "raw_trc" / f"p{p_id}"
    out_dir = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}"
    meta_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    
    # Collect directories
    raw_dirs = []
    main_raw_dir = raw_base_dir / f"r{r_id}"
    if main_raw_dir.is_dir():
        raw_dirs.append(main_raw_dir)
    
    # Check for additional directories
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
    
    # Collect all TRC files
    all_trc_files = []
    for raw_dir in raw_dirs:
        trc_files = sorted(raw_dir.glob("C*--Trace--*.trc"))
        if not trc_files:
            trc_files = sorted(raw_dir.glob("C*--wave--*.trc"))
        if trc_files:
            all_trc_files.extend(trc_files)
    
    if not all_trc_files:
        LOGGER.warning("No .trc files found in any of the directories: %s", [str(d) for d in raw_dirs])
        return
    
    # Log which channels will be processed
    if channels is not None:
        LOGGER.info("Processing only specified channels: %s", channels)
    else:
        LOGGER.info("Processing all available channels")
    
    # Calculate total data size estimate
    if all_trc_files:
        sample_file_size = all_trc_files[0].stat().st_size
        total_input_size_gb = (len(all_trc_files) * sample_file_size) / (1024**3)
        LOGGER.info(f"Estimated total input size: {total_input_size_gb:.2f} GB ({len(all_trc_files)} files)")
    
    # Group files by channel and apply channel filtering
    channel_files: Dict[str, List[Path]] = {}
    for f in all_trc_files:
        c_id = f.name[1]
        # Only include files for specified channels (if channels parameter is provided)
        if channels is None or c_id in channels:
            channel_files.setdefault(c_id, []).append(f)
    
    # Sort files within each channel
    for c_id in channel_files:
        def sort_key(file_path):
            dir_name = file_path.parent.name
            if dir_name == f"r{r_id}":
                dir_idx = 0
            else:
                dir_idx = int(dir_name.split('-')[-1]) - 1
            
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
            LOGGER.warning("No files found for specified channels %s in directories: %s",
                          channels, [str(d) for d in raw_dirs])
        else:
            LOGGER.warning("No channel files found in directories: %s", [str(d) for d in raw_dirs])
        return
    
    # Process waveforms
    if reprocess_waveform:
        LOGGER.info("Processing waveforms with optimized pipeline...")
        
        for c_id, files in channel_files.items():
            LOGGER.info("→ Channel C%s: %d traces", c_id, len(files))
            
            # Adjust batch size based on available memory and file count
            adaptive_batch_size = min(batch_size, max(10, int(memory_monitor.get_memory_info()['system_available_gb'] * 10)))
            
            _write_channel_memmap_optimized(
                files, 
                out_dir / f"C{c_id}", 
                flush_interval, 
                max_workers, 
                dpbl,
                memory_monitor,
                use_mmap=use_mmap,
                batch_size=adaptive_batch_size
            )
            
            # Force cleanup between channels
            gc.collect()
            memory_monitor.log_memory_status(f"after channel C{c_id}")
    
    # Process metadata
    if reprocess_metadata:
        LOGGER.info("Processing metadata...")
        meta_all = {}
        for c_id, files in channel_files.items():
            first_file = next((f for f in files if "--00000" in f.name), files[0])
            meta = parse_trc_file_meta_optimized(first_file)
            meta_all["C" + c_id + "--00000"] = meta
        
        # Save metadata with JSON serialization fix
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta_all, fh, indent=2, ensure_ascii=False, default=json_serialize_numpy)
        LOGGER.info("Metadata written → %s (%d channels)", meta_path, len(meta_all))
    
    memory_monitor.log_memory_status("at completion")
    LOGGER.info("✓ Run p%s r%s completed successfully", p_id, r_id)