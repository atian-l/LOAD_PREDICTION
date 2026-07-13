@echo off
REM Wrapper for Windows Task Scheduler: run auto_sync.sh via Git Bash, log output.
REM The task runs with a minimal PATH, so prepend Git's bin dirs (bash/git/env).
REM cd to repo and pass the script as a bash arg so its shebang is NOT re-resolved
REM (the old "bash -c ./auto_sync.sh" form failed: /usr/bin/env could not find bash
REM in the task's minimal PATH, so auto-sync silently did nothing for 4 days).
set "PATH=E:\Git\Git\usr\bin;E:\Git\Git\mingw64\bin;%PATH%"
cd /d E:\01\python\SdPproject\load_prediction
"E:\Git\Git\usr\bin\bash.exe" "./auto_sync.sh" >> auto_sync.log 2>&1
