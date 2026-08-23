<#
commit.ps1
Run this from your project folder every time you want to save + push
new work (a new feature tested, a score change, a rule added, etc).

USAGE:
    .\commit.ps1 "Add Zoom audio-band rule: 0.8300 -> 0.8349 nested-CV"

If you don't pass a message, it'll prompt you for one.
#>

param(
    [string]$Message
)

if (-not $Message) {
    $Message = Read-Host "Commit message (describe what changed + score delta)"
}

if (-not $Message) {
    Write-Host "No commit message given, aborting." -ForegroundColor Red
    exit 1
}

Write-Host "== Staging all changes ==" -ForegroundColor Cyan
git add -A

Write-Host "== Checking for staged changes ==" -ForegroundColor Cyan
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit — working tree matches last commit." -ForegroundColor Yellow
    exit 0
}

Write-Host "Files being committed:" -ForegroundColor Cyan
$staged | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

Write-Host "== Committing ==" -ForegroundColor Cyan
git commit -m "$Message"

Write-Host "== Pushing to GitHub ==" -ForegroundColor Cyan
git push

Write-Host ""
Write-Host "Done. Pushed: `"$Message`"" -ForegroundColor Green
