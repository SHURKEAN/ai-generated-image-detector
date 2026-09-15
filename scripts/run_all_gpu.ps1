param(
    [string]$Python = "",
    [switch]$SkipHybrid
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
$PortableHome = Join-Path $Root ".python-3.10.11"
$VenvConfig = Join-Path $Root ".venv-tuned\pyvenv.cfg"
$VenvPython = Join-Path $Root ".venv-tuned\Scripts\python.exe"

# A copied venv records its original absolute Python home. Repair that generated
# path when the self-contained portable runtime is present beside the venv.
if ((Test-Path -LiteralPath $PortableHome) -and (Test-Path -LiteralPath $VenvConfig)) {
    $Config = Get-Content -LiteralPath $VenvConfig
    $UpdatedConfig = $Config -replace '^home\s*=.*$', "home = $PortableHome"
    if (($Config -join "`n") -ne ($UpdatedConfig -join "`n")) {
        Set-Content -LiteralPath $VenvConfig -Value $UpdatedConfig -Encoding ASCII
    }
}

$Candidates = @()
if ($Python) {
    $Candidates += $Python
}
if ($env:TDK_PYTHON) {
    $Candidates += $env:TDK_PYTHON
}
$Candidates += $VenvPython

$PathPython = Get-Command python -ErrorAction SilentlyContinue
if ($PathPython) {
    $Candidates += $PathPython.Source
}

$ResolvedPython = $null
foreach ($Candidate in $Candidates | Select-Object -Unique) {
    try {
        if (-not (Test-Path -LiteralPath $Candidate)) {
            continue
        }
        & $Candidate -c "import sys; print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $ResolvedPython = $Candidate
            break
        }
    }
    catch {
        continue
    }
}

if (-not $ResolvedPython) {
    throw "No working Python was found. Pass -Python, set TDK_PYTHON, or restore .venv-tuned."
}

$DriverArgs = @((Join-Path $ScriptDir "run_all_gpu.py"))
if ($SkipHybrid) {
    $DriverArgs += "--skip-hybrid"
}

Write-Host "Python:" $ResolvedPython
Write-Host "Workspace:" $Root
& $ResolvedPython @DriverArgs
if ($LASTEXITCODE -ne 0) {
    throw "Notebook driver failed with exit code $LASTEXITCODE."
}
