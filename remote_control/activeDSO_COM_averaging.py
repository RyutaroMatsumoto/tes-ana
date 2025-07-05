import win32com.client
import numpy as np 
import matplotlib.pyplot as plt

scope = win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") # same for every scope
scope.MakeConnection("IP:192.168.1.177") #this is just an example

acq = scope.Acquisition

ave_waveform = None
count = 0

plt.ion()

while True:
    acq.start()
    acq.WaitForEnd()

    result = acq.C1.Out.Result
    data = np.array(result.DataArray)

    baseline = np.mean(data[:200])
    y_corr = data - baseline

    if ave_waveform is None:
        ave_waveform = y_corr.copy()
        count = 1
    else:
        count += 1
        ave_waveform += (y_corr - ave_waveform)/count
    
    if count % 100 == 0:
        plt.clf()
        plt.plot(ave_waveform)
        plt.title(f"Running Average: n = {count}")
        plt.pause(0.01)