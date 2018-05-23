@echo off
for /f "delims=" %%a in ('dir /a-d /b /s') do (
pushd %%~dpa
echo >"%%~nxa%%~za.txt"
popd
)