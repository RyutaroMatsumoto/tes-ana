import numpy as np

n = 1000
directory = "/Users/ryutaro_matsumoto/Desktop/Reaserch/Analysis_codes/lecroy_data_converter/03_asc_to_npy/signal/"
k=0
# 最初の1データだけ読んで形式を把握.
filename = directory + "C1--wave--{:05d}".format(0) + ".txt"
data = np.genfromtxt(filename, delimiter=',', skip_header=7, filling_values=np.nan)
dp   = data.shape[0]
time = data[:, 0]
dt   = time[1] - time[0]

pulse = np.zeros([n, dp])
noise = np.zeros([n, dp])

# pulse data 

for i in range(0, n):
    print("i = ", i)
    filename = directory + "C1--wave--{:05d}".format(i) + ".txt"
    data = np.genfromtxt(filename, delimiter=',', skip_header=7, filling_values=np.nan)
    pulse[i, :] = data[:, 1]

np.save(directory + "pulse_C1.npy", pulse)

for i in range(0, n):
    print("i = ", i)
    filename = directory + "C2--wave--{:05d}".format(i) + ".txt"
    data = np.genfromtxt(filename, delimiter=',', skip_header=7, filling_values=np.nan)
    pulse[i, :] = data[:, 1]

np.save(directory + "pulse_C2.npy", pulse)


# noise data
for i in range(0, n):
    print("i = ", i)
    filename = directory + "C1--noise--{:05d}".format(i) + ".txt"
    data = np.genfromtxt(filename, delimiter=',', skip_header=7, filling_values=np.nan)
    noise[i, :] = data[:, 1]

np.save(directory + "noise.npy", noise)
