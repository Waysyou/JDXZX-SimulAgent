param(
    [ValidateSet('cpu', 'cuda128')][string]$TorchBackend = 'cpu'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$pythonProbe = Join-Path $root 'scripts\check_python.py'
$env:PIP_CACHE_DIR = Join-Path $root '.cache\pip'
$env:TEMP = Join-Path $root '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR,$env:TEMP | Out-Null

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += @{ exe='py'; prefix=@('-3.13') } }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += @{ exe='python'; prefix=@() } }
    $selected = $null
    foreach ($candidate in $candidates) {
        try { $probe = & $candidate.exe @($candidate.prefix) $pythonProbe 2>$null }
        catch { continue }
        if ($LASTEXITCODE -eq 0 -and ($probe -join '').Trim() -eq '3.13-64') { $selected = $candidate; break }
    }
    if ($null -eq $selected) { throw 'Install Python 3.13 x64, reopen PowerShell, then retry. No administrator rights are required.' }
    Invoke-CheckedPython -Executable $selected.exe -Arguments (@($selected.prefix) + @('-m','venv',$venv))
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) { throw 'Virtual environment creation did not complete.' }
Invoke-CheckedPython -Executable $venvPython -Arguments @($pythonProbe)
Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','install','--upgrade','pip')
$torchIndex = if ($TorchBackend -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
Write-Host "Installing PyTorch from $torchIndex. This may download large files."
Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','install','-r',(Join-Path $root 'requirements-torch.txt'),'--index-url',$torchIndex)
Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','install','-r',(Join-Path $root 'requirements.txt'))
Invoke-CheckedPython -Executable $venvPython -Arguments @('-m','pip','check')
Write-Host 'Dependencies installed. Prepare authorized model files using docs/MODELS.md.'
Write-Host 'Check resources: .\.venv\Scripts\python.exe -X utf8 run_local.py --check'
