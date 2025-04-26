import numpy as np

def trap_filter(wave_data, dt:float, rt=1e-8, ft=5e-9):
    """
    Trapezoidal filter function
    
    Args:
        wave_data: input raw waveform (numpy array)
        rt: rise time in seconds
        ft: flat top time in seconds
        
    Returns:
        numpy array containing the filtered waveform
    """
    # Convert input to numpy array if it's not already
    wave_data = np.asarray(wave_data)
    
    # Get the sampling time from metadata or use a default value
    # This should be adjusted based on your actual data
    dt = dt # Assuming 1 ns sampling time, adjust as needed
    
    # Calculate filter parameters in samples
    k = int(rt / dt)  # Rise time in samples
    l = int(ft / dt)  # Flat top time in samples
    
    if k < 1:
        k = 1
    if l < 1:
        l = 1
    
    # Initialize output array
    n = len(wave_data)
    filtered = np.zeros(n)
    
    # Implement trapezoidal filter algorithm
    # This is a recursive algorithm that efficiently computes the trapezoidal filter
    
    # First difference (d_k[n] = x[n] - x[n-k])
    d_k = np.zeros(n)
    for i in range(k, n):
        d_k[i] = wave_data[i] - wave_data[i-k]
    
    # Second difference (d_kl[n] = d_k[n] - d_k[n-l])
    d_kl = np.zeros(n)
    for i in range(l, n):
        d_kl[i] = d_k[i] - d_k[i-l]
    
    # Integration (recursive sum)
    filtered[0] = d_kl[0]
    for i in range(1, n):
        filtered[i] = filtered[i-1] + d_kl[i]
    
    # Normalize the output
    # The normalization factor depends on the specific application
    # Here we normalize by k to maintain the pulse height
    if k > 0:
        filtered = filtered / k
    
    return filtered