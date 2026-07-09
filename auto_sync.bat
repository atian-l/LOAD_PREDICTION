@echo off
REM Wrapper for Windows Task Scheduler: run auto_sync.sh via Git Bash, log output.
REM Runs unattended under the current user; uses cached Git Credential Manager token.
cd /d E:\01\python\SdPproject\load_prediction
"E:\Git\Git\usr\bin\bash.exe" -c "./auto_sync.sh >> auto_sync.log 2>&1"
