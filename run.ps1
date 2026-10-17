param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MikirinArgs
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    "python",
    "py"
)

$selectedPython = $null
$pythonPrefix = @()
foreach ($candidate in $pythonCandidates) {
    try {
        $candidatePrefix = @()
        if ($candidate -eq "py") { $candidatePrefix = @("-3") }
        & $candidate @candidatePrefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = $candidate
            $pythonPrefix = $candidatePrefix
            break
        }
    } catch {
        continue
    }
}

if (-not $selectedPython) {
    throw "Python 3.12+ was not found. Create .venv or install Python 3.12+ first."
}
if (-not $MikirinArgs) { $MikirinArgs = @("tui") }

$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    & $selectedPython @pythonPrefix -m mikirin --root $projectRoot @MikirinArgs
    $resultCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $savedPythonPath
}
exit $resultCode
