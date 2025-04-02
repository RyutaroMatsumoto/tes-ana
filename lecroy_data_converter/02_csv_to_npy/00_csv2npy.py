import numpy as np
import os

directory = "./csv/"
file_prefix = "C2--wave--"
file_suffix = ".csv"

# ディレクトリ内の特定のファイル形式の数を数える
##n = len([name for name in os.listdir(directory) if name.startswith(file_prefix) and name.endswith(file_suffix)])

n = int(input("Data file number="))

# 最初の1データだけ読んで、サンプリング時間dtとデータポイント数dpを取得。
filename = directory + f"{file_prefix}{0:05d}{file_suffix}"
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
print('Reference Number of data points='+str(data.shape[0]))
dp = int(input('Set appropriate Number of points'))

pulse = np.zeros([n, dp])
noise = np.zeros([n, dp])
# ch3 wave data 
for i in range(0, n):
    print("i = ", i)
    filename = directory + "C2--wave--{:05d}".format(i) + ".csv"
    data = np.loadtxt(filename, delimiter=',', skiprows=0)
    pulse[i, :] = data[0:dp, 1]

np.save("./C2wave.npy", pulse)

# # ch2 wave data 
# for i in range(0, n):
#     print("i = ", i)
#     filename = directory + "C2--wave--{:05d}".format(i) + ".csv"
#     data = np.loadtxt(filename, delimiter=',', skiprows=0)
#     pulse[i, :] = data[0:dp, 1]

# np.save("./C2wave.npy", pulse)

# ch4 wave data 
for i in range(0, n):
    print("i = ", i)
    filename = directory + "C4--wave--{:05d}".format(i) + ".csv"
    data = np.loadtxt(filename, delimiter=',', skiprows=0)
    pulse[i, :] = data[0:dp, 1]

np.save("./C4wave.npy", pulse)