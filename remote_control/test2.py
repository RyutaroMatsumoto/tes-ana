import win32com.client #import the pywin32 library
scope=win32com.client.Dispatch("LeCroy.ActiveDSOCtrl.1") #creates instance of the ActiveDSO control
scope.MakeConnection("IP:192.168.1.177") #Connects to the oscilloscope. Substitute your IP address
scope.WriteString("VBS app.Measure.ShowMeasure = true",1) #Automation command to show measurement table
scope.WriteString("""VBS 'app.Measure.P1.ParamEngine="Mean" ' """,1) #Automation command to change P1 to Mean
scope.Disconnect() #Disconnects from the oscilloscope
