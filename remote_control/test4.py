import win32com.client #import the pywin32 library
import numpy as np
import matplotlib.pyplot as plt
import time


scope=win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") #creates instance of the ActiveDSO control
scope.MakeConnection("IP:192.168.1.177") #connects to the oscilloscope, substitute your IP address
scope.WriteString("vbs? 'Dim dArray : dArray = app.acquisition.C1.Out.Result.Dataarray(1, -1, 0, 1) : return = join(dArray, Chr(44)) '",1)
done = False
data = ''
while not done:
   response = scope.ReadString(10000)
   data += response
   if len(response) < 8192:
     done = True
print(len(data))
plt.plot(data)
plt.title("test4")
plt.xlabel("Point")
plt.ylabel("Voltage (V)")
scope.Disconnect() #Disconnects from the oscilloscope