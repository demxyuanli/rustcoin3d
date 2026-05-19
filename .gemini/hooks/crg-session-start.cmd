@echo off
setlocal EnableDelayedExpansion
for /f "delims=" %%i in ('code-review-graph status --repo "D:/source/repos/rustcoin3d" 2^>^&1') do set "CRG_MSG=%%i" & goto :done
:done
if not defined CRG_MSG set "CRG_MSG="
powershell -NoProfile -ExecutionPolicy Bypass -Command "$m=$env:CRG_MSG; Write-Output (@{systemMessage=$m;suppressOutput=$true} | ConvertTo-Json -Compress)"
exit /b 0
