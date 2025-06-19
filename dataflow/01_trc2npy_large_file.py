"""
----------------------
OPTIMIZED Utility script to convert Teledyne-LeCroy *.trc binary waveform files

PERFORMANCE OPTIMIZATIONS FOR 100GB+ FILES:
* Memory-mapped file I/O for efficient large file handling
* Adaptive batch processing to control memory usage
* System memory monitoring and pressure detection
* Chunked writing for large arrays
* Optimized threading with controlled concurrency
* Garbage collection management
* Progress tracking with ETA estimation

SYSTEM REQUIREMENTS:
* Windows 10+ with 64GB RAM
* Fast storage (NVMe SSD recommended)
* Python 3.8+ with numpy, psutil

Author: Ryutaro Matsumoto - 2025-04-09
Optimized: Performance enhancements for large datasets - 2025-06-17

"""

from pathlib import Path
import sys
import os
import logging
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_functions.process_trc2npy_optimized import process_trc2npy_optimized

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_system_info():
    """Log system information for optimization reference"""
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory()
    
    logging.info("=== SYSTEM INFORMATION ===")
    logging.info(f"CPU: {cpu_count} physical cores, {cpu_count_logical} logical cores")
    logging.info(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    logging.info(f"Memory usage: {memory.percent:.1f}%")
    
    # Check disk space for output directory
    try:
        disk_usage = psutil.disk_usage(str(BASE_DIR))
        logging.info(f"Disk space: {disk_usage.free / (1024**3):.1f} GB free of {disk_usage.total / (1024**3):.1f} GB total")
    except:
        logging.warning("Could not determine disk space")
    
    logging.info("========================")

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent / "tes01"

# ============================================================================
# OPTIMIZATION PARAMETERS - ADJUST FOR YOUR SYSTEM
# ============================================================================

# Data processing parameters
period = "06"
run = "009"
channels = [    # Added channels selection function for large-sized data. Only selected channels will be converted.
    #"1",       # If none, all the channels will be converted.
    "2"
    #,"4"
]
REPROCESS_WAVEFORM = True
REPROCESS_METADATA = True
dpbl = 2000  # Data points for baseline correction

# Performance tuning parameters
FLUSH_INTERVAL = 50        # Increased for better performance with large files
MAX_WORKERS = 12            # Reduced from 12 to prevent memory pressure
USE_MMAP = True            # Enable memory mapping for large files
BATCH_SIZE = 50            # Batch size for processing (controls memory usage)
MEMORY_WARNING_THRESHOLD = 0.80  # Warning threshold for memory usage (80%)

# ============================================================================
# ADAPTIVE CONFIGURATION BASED ON SYSTEM SPECS
# ============================================================================

def get_optimized_config():
    """Automatically configure parameters based on system specs"""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True) or 4  # Default to 4 if None
    
    # Memory-based adjustments
    memory_gb = memory.total / (1024**3)
    
    if memory_gb >= 64:  # High-memory system (like yours)
        max_workers = min(cpu_count - 2, 10)  # Leave some cores for system
        batch_size = 100
        flush_interval = 100
        memory_threshold = 0.85
    elif memory_gb >= 32:  # Medium-memory system
        max_workers = min(cpu_count - 1, 6)
        batch_size = 50
        flush_interval = 50
        memory_threshold = 0.80
    else:  # Low-memory system
        max_workers = min(cpu_count, 4)
        batch_size = 25
        flush_interval = 25
        memory_threshold = 0.75
    
    # CPU-based adjustments for Xeon Silver 4110 (8 cores, 16 threads)
    if cpu_count >= 16:  # High-end CPU
        max_workers = min(max_workers, 8)  # Don't over-thread
    
    return {
        'max_workers': max_workers,
        'batch_size': batch_size,
        'flush_interval': flush_interval,
        'memory_threshold': memory_threshold
    }

if __name__ == "__main__":
    # Log system information
    log_system_info()
    
    # Get optimized configuration
    config = get_optimized_config()
    
    logging.info("=== OPTIMIZATION CONFIGURATION ===")
    logging.info(f"Max workers: {config['max_workers']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Flush interval: {config['flush_interval']}")
    logging.info(f"Memory threshold: {config['memory_threshold']:.1%}")
    logging.info(f"Memory mapping: {USE_MMAP}")
    logging.info("=================================")
    
    # Override with adaptive configuration
    max_workers = config['max_workers']
    batch_size = config['batch_size']
    flush_interval = config['flush_interval']
    memory_threshold = config['memory_threshold']
    
    # Manual overrides (uncomment to use fixed values)
    # max_workers = MAX_WORKERS
    # batch_size = BATCH_SIZE
    # flush_interval = FLUSH_INTERVAL
    # memory_threshold = MEMORY_WARNING_THRESHOLD
    
    try:
        process_trc2npy_optimized(
            p_id=period,
            r_id=run,
            channels=channels,
            base_dir=BASE_DIR,
            reprocess_waveform=REPROCESS_WAVEFORM,
            reprocess_metadata=REPROCESS_METADATA,
            flush_interval=flush_interval,
            max_workers=max_workers,
            dpbl=dpbl,
            use_mmap=USE_MMAP,
            batch_size=batch_size,
            memory_warning_threshold=memory_threshold
        )
        
        logging.info("✓ Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise
    
    finally:
        # Final system status
        memory = psutil.virtual_memory()
        logging.info(f"Final memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB used)")