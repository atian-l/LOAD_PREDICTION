Set WshShell = CreateObject("WScript.Shell")
batPath = "E:\01\python\SdPproject\load_prediction\auto_sync.bat"
WshShell.Run """" & batPath & """", 0, False
Set WshShell = Nothing