"""
correct baseline 
invert signal
optimal filter
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import sys
import logging
import os
import json
import src.trap_filter as trap_filter
from src.gaussian_fitting import gaussian_fit
from src.tes_analysis_tools import correct_baseline,make_average_pulse,optimal_filter_freq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
def make_integral_pulse(pulse: np.ndarray, dt: float, verbose: bool = False) -> np.ndarray:
    """
    Calculate the integral of pulse data using trapezoidal integration.
    
    This function computes the integral for each pulse sample by integrating
    over the time domain using the trapezoidal rule for numerical integration.
    
    Args:
        pulse (np.ndarray): 2D array of pulse data with shape (n_samples, n_datapoints)
                           where n_samples is the number of pulse traces and
                           n_datapoints is the number of time points per trace
        dt (float): Time interval between consecutive data points in seconds
        verbose (bool, optional): Enable verbose logging. Defaults to False.
    
    Returns:
        np.ndarray: 1D array of integrated values for each pulse sample
        
    Raises:
        ValueError: If pulse array is empty, has wrong dimensions, or dt is invalid
        TypeError: If inputs are not of expected types
        
    Example:
        >>> pulse_data = np.random.randn(100, 1000)  # 100 pulses, 1000 points each
        >>> dt = 1e-9  # 1 nanosecond sampling
        >>> integrals = make_integral_pulse(pulse_data, dt, verbose=True)
        >>> print(f"Computed {len(integrals)} pulse integrals")
    """
    # Input validation
    if not isinstance(pulse, np.ndarray):
        raise TypeError(f"pulse must be numpy array, got {type(pulse)}")
    
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError(f"dt must be positive number, got {dt}")
    
    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be boolean, got {type(verbose)}")
    
    if pulse.size == 0:
        raise ValueError("pulse array cannot be empty")
    
    # Handle both 1D and 2D arrays
    if pulse.ndim == 1:
        # Single pulse trace - reshape to 2D for consistent processing
        pulse = pulse.reshape(1, -1)
        if verbose:
            logging.info(f"Reshaped 1D pulse array to shape {pulse.shape}")
    elif pulse.ndim != 2:
        raise ValueError(f"pulse must be 1D or 2D array, got {pulse.ndim}D")
    
    n_samples, n_datapoints = pulse.shape
    
    if verbose:
        logging.info(f"Processing {n_samples} pulse samples with {n_datapoints} data points each")
        logging.info(f"Time resolution: {dt:.2e} seconds")
        logging.info(f"Total time span per pulse: {(n_datapoints-1) * dt:.2e} seconds")
    
    # Vectorized trapezoidal integration
    # Using numpy's trapz function which is optimized and handles edge cases
    try:
        # Integrate along the time axis (axis=1) for each pulse
        integral = np.trapezoid(pulse, dx=dt, axis=1)
        
        if verbose:
            logging.info(f"Integration completed successfully")
            logging.info(f"Integral statistics - Mean: {np.mean(integral):.2e}, "
                        f"Std: {np.std(integral):.2e}, "
                        f"Min: {np.min(integral):.2e}, "
                        f"Max: {np.max(integral):.2e}")
        
        return integral
        
    except Exception as e:
        logging.error(f"Error during integration: {str(e)}")
        raise RuntimeError(f"Integration failed: {str(e)}") from e


def process_integral(period1:int,run1:int,channel1:int,period2:int,run2:int,channel2:int, verbose, base_dir:Path)->None:
    DEBUG = True
    # create path for pulse and noise
    pulse = np.load(base_dir / "generated_data" / "raw" /f"p{period1}"/ f"r{run1}" / f"C{channel1}" / f"C{channel1}--Trace.npy")    #signal waveform
    noise = np.load(base_dir / "generated_data" / "raw" /f"p{period2}"/ f"r{run2}" / f"C{channel2}" / f"C{channel2}--Trace.npy")    #noise waveform
    metadata_path1 = base_dir / "teststand_metadata" / "hardware" /"scope" / f"p{period1}" / f"r{run1}" / f"lecroy_metadata_p{period1}_r{run1}.json"
    #load metadata(time interval)
    with open(metadata_path1, 'r') as f:
        metadata1 = json.load(f)
        dt = metadata1[f"C{channel1}--00000"]['time_resolution']['dt']
    
    plt_dir = base_dir / "generated_data" / "pyplt" /"optimal"/f"p{period1}"/ f"r{run1}" / f"C{channel1}"
    # Create directory if it doesn't exist
    plt_dir.mkdir(parents=True, exist_ok=True)

    par_dir = base_dir / "generated_data" / "pypar" /"optimal"/f"p{period1}"/ f"r{run1}" / f"C{channel1}"
    # Create directory if it doesn't exist
    par_dir.mkdir(parents=True, exist_ok=True)
    
    #integral for each pulse
    integral = make_integral_pulse(pulse, dt, verbose)
    #plot histgram 
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family']= 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.grid()
    plt.ylabel("Counts")
    plt.xlabel("Energy [a.u]")          #as integral of V should be proportional to deposited energy
    plt.hist(integral, bins=256)
    plt.plot(gaussian_fit(integral))
    plt.savefig(f"{plt_dir}/hist_integral_PNR_p{period1}_r{run1}.png")
    logging.info(f"hist_integral_PNR_log.png saved to {plt_dir}")
    #save hist data
    np.save(par_dir / f"integral_p{period1}_r{run1}.npy", integral)

    # # 対数スケール
    # fig = plt.figure(figsize=(9, 6))
    # plt.rcParams['font.size'] = 14
    # plt.rcParams['font.family']= 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial']
    # plt.grid()
    # plt.ylabel("Counts")
    # plt.xlabel("Pulse height [a.u]")
    # plt.hist(ph_array, bins=256)
    # plt.yscale('log')
    # plt.savefig(f"{plt_dir}/hist_optimal-filter_log_p{period1}_r{run1}.png")
    # logging.info(f"hist_optimal-filter_log.png saved to {plt_dir}")


