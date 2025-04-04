import numpy as np

n = 137
directory = "../file_csv/"

# 最初の1データだけ読んで, サンプリング時間dtを取得.
filename = directory + "C3--wave--{:05d}".format(0) + ".csv"
data = np.loadtxt(filename, dtype='float', delimiter=',', skiprows=0)
time = data[:, 0]
dt   = time[1] - time[0]
f = open('dt.txt', 'w')
f.write( str(dt) )
f.close()

# 波形ファイルによっては, データサイズが違っていることがある.
# そのため, data.shape[0]の値は使わないほうがよい.
# たとえば1kSで取得するとだいたい1002ポイントになるが, たまに1001ポイントの波形もある.
# data.shape[0]は目安に使う.
#dp   = data.shape[0]
dp = 1000

pulse = np.zeros([n, dp])
noise = np.zeros([n, dp])

# ch3 wave data 
for i in range(0, n):
    print("i = ", i)
    filename = directory + "C3--wave--{:05d}".format(i) + ".csv"
    data = np.loadtxt(filename, delimiter=',', skiprows=0)
    pulse[i, :] = data[0:dp, 1]

np.save("./C3wave.npy", pulse)

# ch4 wave data 
for i in range(0, n):
    print("i = ", i)
    filename = directory + "C4--wave--{:05d}".format(i) + ".csv"
    data = np.loadtxt(filename, delimiter=',', skiprows=0)
    pulse[i, :] = data[0:dp, 1]

np.save("./C4wave.npy", pulse)
