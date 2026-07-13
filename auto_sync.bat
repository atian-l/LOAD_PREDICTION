@echo off
REM Wrapper for Windows Task Scheduler: run auto_sync.sh via Git Bash, log output.
REM Prepend Git bin dirs so bash/git/env are found in the task's minimal PATH.
REM Invoke script as a bash arg (not -c "./...") so the shebang is not re-resolved.
set "PATH=E:\Git\Git\usr\bin;E:\Git\Git\mingw64\bin;%PATH%"
"E:\Git\Git\usr\bin\bash.exe" "E:\01\python\SdPproject\load_prediction\auto_sync.sh" >> "E:\01\python\SdPproject\load_prediction\auto_sync.log" 2>&1
