import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import logging

def gaussian_fit(data, num_gauss=5):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import matplotlib.cm as cm

    def multi_gaussian(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp = params[i]
            cen = params[i + 1]
            wid = params[i + 2]
            # Add small epsilon to avoid division by zero
            y += amp * np.exp(-((x - cen) ** 2) / (2 * (wid ** 2 + 1e-10)))
        return y

    # ヒストグラム
    counts, bins = np.histogram(data, bins=300)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Remove zero counts to avoid fitting issues
    non_zero_mask = counts > 0
    bin_centers = bin_centers[non_zero_mask]
    counts = counts[non_zero_mask]
    
    if len(bin_centers) == 0:
        logging.warning("No non-zero counts found in histogram")
        # Create a simple plot with the original data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=300, alpha=0.5, label="Data")
        ax.set_xlabel("Pulse height [a.u]")
        ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return plt

    # Better initial parameter estimation
    data_range = np.max(data) - np.min(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    for i in range(num_gauss):
        # Amplitude: start with max counts divided by number of gaussians
        amp_init = np.max(counts) / (i + 1)
        p0.append(amp_init)
        bounds_lower.append(0)  # Amplitude must be positive
        bounds_upper.append(np.max(counts) * 2)  # Upper bound for amplitude
        
        # Center: distribute across data range
        center_init = data_mean + (i - num_gauss/2) * data_std / 2
        p0.append(center_init)
        bounds_lower.append(np.min(data) - data_range)  # Allow some extrapolation
        bounds_upper.append(np.max(data) + data_range)
        
        # Width: start with reasonable fraction of data std
        width_init = np.maximum(data_std / (num_gauss + 1), data_range / 100)
        p0.append(width_init)
        bounds_lower.append(data_range / 1000)  # Minimum width
        bounds_upper.append(data_range)  # Maximum width
    
    bounds = (bounds_lower, bounds_upper)

    # フィット with improved parameters
    try:
        popt, pcov = curve_fit(
            multi_gaussian,
            bin_centers,
            counts,
            p0=p0,
            bounds=bounds,
            maxfev=10000,  # Increase maximum function evaluations
            method='trf',  # Trust Region Reflective algorithm, good for bounded problems
            ftol=1e-8,     # Function tolerance
            xtol=1e-8      # Parameter tolerance
        )
        fit_success = True
        logging.info("Gaussian fitting converged successfully")
    except Exception as e:
        logging.warning(f"Gaussian fitting failed: {e}")
        logging.info("Attempting simplified fitting with fewer Gaussians")
        
        # Fallback: try with fewer Gaussians
        simplified_num_gauss = min(3, num_gauss)
        p0_simple = []
        bounds_lower_simple = []
        bounds_upper_simple = []
        
        for i in range(simplified_num_gauss):
            amp_init = np.max(counts) / (i + 1)
            p0_simple.append(amp_init)
            bounds_lower_simple.append(0)
            bounds_upper_simple.append(np.max(counts) * 2)
            
            center_init = data_mean + (i - simplified_num_gauss/2) * data_std / 2
            p0_simple.append(center_init)
            bounds_lower_simple.append(np.min(data) - data_range)
            bounds_upper_simple.append(np.max(data) + data_range)
            
            width_init = np.maximum(data_std / (simplified_num_gauss + 1), data_range / 100)
            p0_simple.append(width_init)
            bounds_lower_simple.append(data_range / 1000)
            bounds_upper_simple.append(data_range)
        
        bounds_simple = (bounds_lower_simple, bounds_upper_simple)
        
        try:
            popt, pcov = curve_fit(
                multi_gaussian,
                bin_centers,
                counts,
                p0=p0_simple,
                bounds=bounds_simple,
                maxfev=10000,
                method='trf'
            )
            num_gauss = simplified_num_gauss  # Update for plotting
            fit_success = True
            logging.info(f"Simplified fitting with {simplified_num_gauss} Gaussians succeeded")
        except Exception as e2:
            logging.error(f"Even simplified fitting failed: {e2}")
            # Use original parameters for plotting, but no fit curve
            popt = p0_simple
            fit_success = False

    # プロット用
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
    y_fit = multi_gaussian(x_fit, *popt)

    # プロット作成
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data, bins=300, alpha=0.5, label="Data")
    ax.plot(x_fit, y_fit, color='black', linewidth=2, label="Total fit")

    # 各成分
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, num_gauss))
    for i in range(num_gauss):
        amp = popt[3 * i]
        cen = popt[3 * i + 1]
        wid = popt[3 * i + 2]
        y = amp * np.exp(-((x_fit - cen) ** 2) / (2 * wid ** 2))
        ax.plot(x_fit, y, linestyle='--', color=colors[i], label=f"n = {i}")

    ax.set_xlabel("Pulse height [a.u]")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return plt
