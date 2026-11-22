# One-command health check for Windows PowerShell.
# Run:  powershell -ExecutionPolicy Bypass -File .\check.ps1
# Uses the offline agent modes, so any GROQ_API_KEY in your session is set aside and restored.
$savedKey = $env:GROQ_API_KEY
$env:GROQ_API_KEY = $null
function Step($title, [scriptblock]$cmd) {
    Write-Host "== $title =="
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title"; $env:GROQ_API_KEY = $savedKey; exit 1 }
}
Step "1/4 unit tests"                      { python -m pytest -q }
Step "2/4 train on synthetic data"         { python -m supportlens.train --n 1500 --epochs 6 --out artifacts_check }
$env:SUPPORTLENS_ARTIFACTS = "artifacts_check"
Step "3/4 LangChain agent (offline)"       { python -m supportlens.agent "my parcel is stuck at the courier hub" }
Step "4/4 CrewAI agent (offline fallback)" { python -m supportlens.crew_agent "my parcel is stuck at the courier hub" }
Remove-Item Env:SUPPORTLENS_ARTIFACTS
Remove-Item artifacts_check -Recurse -Force
$env:GROQ_API_KEY = $savedKey
Write-Host "ALL CHECKS PASSED"
