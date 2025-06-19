"""
Benchmark script to compare original vs optimized TRC to NPY conversion performance

This script helps you:
1. Test both original and optimized implementations
2. Compare performance metrics
3. Validate memory usage
4. Ensure output consistency

Usage:
    python benchmark_trc2npy.py --test-size small|medium|large
"""

import argparse
import time
import psutil
import logging
from pathlib import Path
import sys
import os
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from processing_functions.process_trc2npy import process_trc2npy
from processing_functions.process_trc2npy_optimized import process_trc2npy_optimized

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class PerformanceBenchmark:
    """Performance benchmarking utility"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = {}
    
    def start_benchmark(self, name: str) -> Dict[str, Any]:
        """Start a benchmark measurement"""
        initial_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'name': name,
            'start_time': time.perf_counter(),
            'start_memory_rss': initial_memory.rss,
            'start_memory_vms': initial_memory.vms,
            'start_system_memory': system_memory.used,
            'peak_memory_rss': initial_memory.rss,
            'peak_system_memory': system_memory.used
        }
    
    def update_peak_memory(self, benchmark: Dict[str, Any]):
        """Update peak memory usage"""
        current_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        benchmark['peak_memory_rss'] = max(benchmark['peak_memory_rss'], current_memory.rss)
        benchmark['peak_system_memory'] = max(benchmark['peak_system_memory'], system_memory.used)
    
    def end_benchmark(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """End benchmark and calculate results"""
        end_time = time.perf_counter()
        final_memory = self.process.memory_info()
        
        # Update final peak memory
        self.update_peak_memory(benchmark)
        
        # Calculate results
        results = {
            'name': benchmark['name'],
            'duration_seconds': end_time - benchmark['start_time'],
            'duration_minutes': (end_time - benchmark['start_time']) / 60.0,
            'memory_used_mb': (benchmark['peak_memory_rss'] - benchmark['start_memory_rss']) / (1024 * 1024),
            'peak_memory_mb': benchmark['peak_memory_rss'] / (1024 * 1024),
            'system_memory_used_mb': (benchmark['peak_system_memory'] - benchmark['start_system_memory']) / (1024 * 1024),
            'final_memory_mb': final_memory.rss / (1024 * 1024)
        }
        
        self.results[benchmark['name']] = results
        return results
    
    def print_results(self):
        """Print benchmark results"""
        if not self.results:
            logging.info("No benchmark results to display")
            return
        
        logging.info("=" * 80)
        logging.info("BENCHMARK RESULTS")
        logging.info("=" * 80)
        
        for name, result in self.results.items():
            logging.info(f"\n{name.upper()}:")
            logging.info(f"  Duration: {result['duration_minutes']:.2f} minutes ({result['duration_seconds']:.1f} seconds)")
            logging.info(f"  Memory Used: {result['memory_used_mb']:.1f} MB")
            logging.info(f"  Peak Memory: {result['peak_memory_mb']:.1f} MB")
            logging.info(f"  System Memory Used: {result['system_memory_used_mb']:.1f} MB")
            logging.info(f"  Final Memory: {result['final_memory_mb']:.1f} MB")
        
        # Compare if we have both results
        if len(self.results) >= 2:
            results_list = list(self.results.values())
            original = next((r for r in results_list if 'original' in r['name'].lower()), None)
            optimized = next((r for r in results_list if 'optimized' in r['name'].lower()), None)
            
            if original and optimized:
                logging.info(f"\nCOMPARISON:")
                speed_improvement = original['duration_seconds'] / optimized['duration_seconds']
                memory_improvement = original['peak_memory_mb'] / optimized['peak_memory_mb']
                
                logging.info(f"  Speed Improvement: {speed_improvement:.2f}x faster")
                logging.info(f"  Memory Improvement: {memory_improvement:.2f}x less memory")
                logging.info(f"  Time Saved: {original['duration_minutes'] - optimized['duration_minutes']:.2f} minutes")
        
        logging.info("=" * 80)

def validate_outputs(original_dir: Path, optimized_dir: Path) -> bool:
    """Validate that both implementations produce the same output"""
    logging.info("Validating output consistency...")
    
    # Find NPY files in both directories
    original_files = list(original_dir.rglob("*.npy"))
    optimized_files = list(optimized_dir.rglob("*.npy"))
    
    if len(original_files) != len(optimized_files):
        logging.error(f"File count mismatch: {len(original_files)} vs {len(optimized_files)}")
        return False
    
    # Compare each file
    for orig_file in original_files:
        # Find corresponding optimized file
        rel_path = orig_file.relative_to(original_dir)
        opt_file = optimized_dir / rel_path
        
        if not opt_file.exists():
            logging.error(f"Missing optimized file: {opt_file}")
            return False
        
        # Load and compare arrays
        try:
            orig_data = np.load(orig_file)
            opt_data = np.load(opt_file)
            
            if orig_data.shape != opt_data.shape:
                logging.error(f"Shape mismatch in {rel_path}: {orig_data.shape} vs {opt_data.shape}")
                return False
            
            # Check if arrays are close (allowing for small numerical differences)
            if not np.allclose(orig_data, opt_data, rtol=1e-6, atol=1e-8):
                max_diff = np.max(np.abs(orig_data - opt_data))
                logging.error(f"Data mismatch in {rel_path}: max difference = {max_diff}")
                return False
            
            logging.debug(f"✓ {rel_path} - shapes match, data consistent")
            
        except Exception as e:
            logging.error(f"Error comparing {rel_path}: {e}")
            return False
    
    logging.info("✓ All outputs are consistent between implementations")
    return True

def run_benchmark(test_size: str = "small"):
    """Run the benchmark comparison"""
    
    # Configuration based on test size
    configs = {
        "small": {
            "period": "06",
            "run": "004",  # Smaller run for testing
            "description": "Small test (few files)"
        },
        "medium": {
            "period": "06", 
            "run": "003",  # Medium run
            "description": "Medium test (moderate files)"
        },
        "large": {
            "period": "06",
            "run": "005",  # Your large run
            "description": "Large test (many files)"
        }
    }
    
    if test_size not in configs:
        raise ValueError(f"Invalid test size: {test_size}. Choose from: {list(configs.keys())}")
    
    config = configs[test_size]
    base_dir = Path(__file__).resolve().parent.parent / "tes01"
    
    logging.info(f"Starting benchmark: {config['description']}")
    logging.info(f"Period: {config['period']}, Run: {config['run']}")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Test parameters
    common_params = {
        'p_id': config['period'],
        'r_id': config['run'],
        'base_dir': base_dir,
        'reprocess_waveform': True,
        'reprocess_metadata': True,
        'dpbl': 80
    }
    
    # Create separate output directories for comparison
    original_output = base_dir / "generated_data" / "raw_original" / f"p{config['period']}" / f"r{config['run']}"
    optimized_output = base_dir / "generated_data" / "raw_optimized" / f"p{config['period']}" / f"r{config['run']}"
    
    try:
        # Benchmark original implementation
        logging.info("\n" + "="*50)
        logging.info("TESTING ORIGINAL IMPLEMENTATION")
        logging.info("="*50)
        
        # Modify output directory for original
        original_params = common_params.copy()
        original_base = base_dir.parent / "tes01"
        original_params['base_dir'] = original_base
        
        bench_original = benchmark.start_benchmark("Original Implementation")
        
        try:
            process_trc2npy(
                flush_interval=10,
                max_workers=4,  # Conservative settings for original
                **original_params
            )
        except Exception as e:
            logging.error(f"Original implementation failed: {e}")
            return
        
        result_original = benchmark.end_benchmark(bench_original)
        
        # Benchmark optimized implementation  
        logging.info("\n" + "="*50)
        logging.info("TESTING OPTIMIZED IMPLEMENTATION")
        logging.info("="*50)
        
        # Modify output directory for optimized
        optimized_params = common_params.copy()
        optimized_base = base_dir.parent / "tes01_optimized_test"
        optimized_params['base_dir'] = optimized_base
        
        bench_optimized = benchmark.start_benchmark("Optimized Implementation")
        
        try:
            process_trc2npy_optimized(
                flush_interval=50,
                max_workers=8,
                use_mmap=True,
                batch_size=100,
                memory_warning_threshold=0.80,
                **optimized_params
            )
        except Exception as e:
            logging.error(f"Optimized implementation failed: {e}")
            return
        
        result_optimized = benchmark.end_benchmark(bench_optimized)
        
        # Print results
        benchmark.print_results()
        
        # Validate outputs if both succeeded
        original_out_dir = original_base / "generated_data" / "raw" / f"p{config['period']}" / f"r{config['run']}"
        optimized_out_dir = optimized_base / "generated_data" / "raw" / f"p{config['period']}" / f"r{config['run']}"
        
        if original_out_dir.exists() and optimized_out_dir.exists():
            validate_outputs(original_out_dir, optimized_out_dir)
        
    except KeyboardInterrupt:
        logging.info("Benchmark interrupted by user")
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Benchmark TRC to NPY conversion performance")
    parser.add_argument(
        "--test-size", 
        choices=["small", "medium", "large"], 
        default="small",
        help="Size of test dataset (default: small)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log system information
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True)
    
    logging.info("=" * 60)
    logging.info("SYSTEM INFORMATION")
    logging.info("=" * 60)
    logging.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {cpu_count} logical")
    logging.info(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    logging.info(f"Memory Usage: {memory.percent:.1f}%")
    
    try:
        run_benchmark(args.test_size)
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()