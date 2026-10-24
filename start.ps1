param([switch]$Check, [switch]$UiOnly, [switch]$NoBrowser, [int]$Port = 0)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Run setup_runtime.ps1 first. This project uses its own .venv, not a parent workspace.' }
$launchArgs = @('-X','utf8',(Join-Path $root 'run_local.py'))
if ($Check) { $launchArgs += '--check' }
if ($UiOnly) { $launchArgs += '--ui-only' }
if ($NoBrowser) { $launchArgs += '--no-browser' }
if ($Port -ne 0) { $launchArgs += @('--port', [string]$Port) }
Push-Location $root
try { & $python @launchArgs; $exitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $exitCode
