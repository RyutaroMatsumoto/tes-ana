import win32com.client #imports the pywin32 library

scope=win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") #creates instance of the ActiveDSO control
scope.MakeConnection("IP:192.168.1.177") #Connects to the oscilloscope. Substitute your IP address
scope.WriteString("C2:VDIV .02",1) #Remote Command to set C1 volt/div setting to 20 mV.
scope.Disconnect() #Disconnects from the oscilloscope