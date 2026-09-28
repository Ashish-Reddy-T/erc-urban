# Create an isolated git worktree for feature work, with its own venv.
#
# Usage:
#   scripts\new-worktree.ps1 -Branch <branch-name> [-Base origin/main] [-SkipPip]
#
# What it does:
#   1. Creates .worktrees\<branch-name> as a new linked worktree, branching
#      off -Base (creates the branch if it does not exist yet).
#   2. Copies earth-rovers-sdk\.env into the new worktree if the source repo
#      has one, so the SDK server has credentials without touching the
#      tracked file in this checkout.
#   3. Creates a Python venv at .venv inside the new worktree and installs
#      requirements.txt (-SkipPip to skip).
#
# It does not download model checkpoints or pre-cache DINOv2/SegFormer -
# run setup.sh inside the new worktree for that when you actually need it.

param(
    [Parameter(Mandatory = $true)]
    [string]$Branch,

    [string]$Base = "origin/main",

    [switch]$SkipPip,

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $repoRoot

$wtDir = Join-Path ".worktrees" $Branch

if (Test-Path $wtDir) {
    Write-Error "error: $wtDir already exists"
    exit 1
}

New-Item -ItemType Directory -Force -Path ".worktrees" | Out-Null

$branchExists = git show-ref --verify --quiet "refs/heads/$Branch"
if ($LASTEXITCODE -eq 0) {
    Write-Host "==> branch $Branch already exists, reusing it"
    git worktree add $wtDir $Branch
}
else {
    Write-Host "==> creating branch $Branch from $Base"
    git worktree add $wtDir -b $Branch $Base
}

$srcEnv = Join-Path "earth-rovers-sdk" ".env"
$dstEnv = Join-Path $wtDir "earth-rovers-sdk\.env"
if ((Test-Path $srcEnv) -and -not (Test-Path $dstEnv)) {
    Write-Host "==> copying earth-rovers-sdk\.env into new worktree"
    New-Item -ItemType Directory -Force -Path (Join-Path $wtDir "earth-rovers-sdk") | Out-Null
    Copy-Item $srcEnv $dstEnv
}

if (-not $SkipPip) {
    $venvDir = Join-Path $wtDir ".venv"
    Write-Host "==> creating venv at $venvDir"
    & $Python -m venv $venvDir
    $venvPip = Join-Path $venvDir "Scripts\pip.exe"
    & $venvPip install -q -U pip
    & $venvPip install -q -r (Join-Path $wtDir "requirements.txt")
}
else {
    Write-Host "==> -SkipPip set, not creating a venv"
}

Write-Host ""
Write-Host "Worktree ready: $wtDir"
Write-Host "Next:"
Write-Host "  cd $wtDir"
if (-not $SkipPip) {
    Write-Host "  .venv\Scripts\Activate.ps1"
}
Write-Host "  ./setup.sh            # only if you need checkpoints / model caches"
