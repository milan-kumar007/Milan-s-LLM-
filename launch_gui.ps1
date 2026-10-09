$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Launching Custom GPT Desktop GUI on NVIDIA GPU..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
$GuiScript = Join-Path $ScriptDir "scripts\gui_app.py"

if (Test-Path $PythonExe) {
    & $PythonExe $GuiScript
} else {
    Write-Host "[ERROR] Virtual environment not found at .venv!" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
