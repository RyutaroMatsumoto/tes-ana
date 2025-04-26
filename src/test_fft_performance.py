#!/usr/bin/env python
"""
Test script to compare the performance of different FFT implementations.
This script loads the same data that's used in the actual noise analysis
and compares the performance of all FFT implementations.
"""

import numpy as np
import time
import argparse
import json
import logging
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.fft_funcs import (
    compare_fft_implementations,
    fivesmt_fft_original,
    fivesmt_fft_optimized_f32,
    fivesmt_fft_optimized_f64,
    fivesmt_fft_numba_f32,
    fivesmt_fft_numba_f64,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def test_with_real_data(p_id, r_id, c_id, base_dir, n_runs=5):
    """
    Test FFT performance with real data from the specified dataset.
    
    Parameters:
    -----------
    p_id : str
        Period ID
    r_id : str
        Run ID
    c_id : str
        Channel ID
    base_dir : Path
        Base directory for the data
    n_runs : int
        Number of runs for each implementation
    """
    # Construct paths
    raw_file = base_dir / "generated_data" / "raw" / f"p{p_id}" / f"r{r_id}" / f"C{c_id}" / f"C{c_id}--Trace.npy"
    metadata_path = base_dir / "teststand_metadata" / "hardware" / "scope" / f"p{p_id}" / f"r{r_id}" / f"lecroy_metadata_p{p_id}_r{r_id}.json"
    
    logging.info(f"Loading data from {raw_file}")
    logging.info(f"Loading metadata from {metadata_path}")
    
    # Load data and metadata
    try:
        data_array = np.load(raw_file, allow_pickle=True)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        dt = metadata['C1--00000']['time_resolution']['dt']
        
        logging.info(f"Data loaded successfully. Shape: {data_array.shape}")
        logging.info(f"Time resolution (dt): {dt}")
        
        # Run performance tests
        print(f"\nPerformance test with real data: p{p_id}_r{r_id}_C{c_id}")
        print(f"Data shape: {data_array.shape}, dt: {dt}")
        
        results = compare_fft_implementations(data_array, dt, n_runs)
        
        # Print results
        print("\nPerformance Results (average execution time in seconds):")
        print("-" * 80)
        print(f"{'Implementation':<20} {'Mean Time (s)':<15} {'Std Dev (s)':<15} {'Speedup':<10}")
        print("-" * 80)
        
        # Calculate speedup relative to original implementation
        original_mean = results['original']['mean']
        
        for name, data in results.items():
            speedup = original_mean / data['mean'] if data['mean'] > 0 else float('inf')
            print(f"{name:<20} {data['mean']:<15.6f} {data['std']:<15.6f} {speedup:<10.2f}x")
        
        # Find the fastest implementation
        fastest = min(results.items(), key=lambda x: x[1]['mean'])
        print("\nFastest implementation:", fastest[0])
        print(f"Average execution time: {fastest[1]['mean']:.6f} seconds")
        print(f"Speedup over original: {original_mean / fastest[1]['mean']:.2f}x")
        
        # Recommend the best implementation
        print("\nRecommendation:")
        if fastest[0] == 'numba_f32':
            print("The Numba float32 implementation is fastest. This is now the default implementation.")
        elif fastest[0] == 'numba_f64':
            print("The Numba float64 implementation is fastest. Consider updating the default implementation.")
        elif fastest[0] == 'optimized_f32':
            print("The optimized float32 implementation is fastest. Consider updating the default implementation.")
        elif fastest[0] == 'optimized_f64':
            print("The optimized float64 implementation is fastest. Consider updating the default implementation.")
        else:
            print("The original implementation is fastest. No changes needed.")
        
        return results
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description='Test FFT performance with real data')
    parser.add_argument('--period', type=str, default="01",
                        help='Period ID (default: 01)')
    parser.add_argument('--run', type=str, default="007",
                        help='Run ID (default: 004)')
    parser.add_argument('--channel', type=str, default="1",
                        help='Channel ID (default: 1)')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of runs for each implementation (default: 5)')
    
    args = parser.parse_args()
    
    # Set base directory
    base_dir = Path(__file__).resolve().parent.parent / "tes01"
    
    # Run the performance test with real data
    test_with_real_data(
        p_id=args.period,
        r_id=args.run,
        c_id=args.channel,
        base_dir=base_dir,
        n_runs=args.n_runs
    )


if __name__ == "__main__":
    main()