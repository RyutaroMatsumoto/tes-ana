import numpy as np, timeit, pyfftw, scipy.fft as sf

n   = 50_000_000
sig = np.random.rand(n).astype('float32')

# SciPy pocketfft (Accelerate/NEON) 1-thread
print("SciPy 1T :",
      timeit.timeit(lambda: sf.rfft(sig, workers=1), number=3))

# SciPy 1T + 4 workers
print("SciPy 8T :",
      timeit.timeit(lambda: sf.rfft(sig, workers=8), number=3))

# pyFFTW 1-thread
ff1 = pyfftw.builders.rfft(sig, overwrite_input=False,
                           threads=1, planner_effort='FFTW_ESTIMATE')
print("pyFFTW 1T:", timeit.timeit(ff1, number=3))

# pyFFTW 4-thread
ff4 = pyfftw.builders.rfft(sig, overwrite_input=False,
                           threads=8, planner_effort='FFTW_ESTIMATE')
print("pyFFTW 8T:", timeit.timeit(ff4, number=3))
