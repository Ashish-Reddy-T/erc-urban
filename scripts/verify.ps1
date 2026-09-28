# The single entry point for "is this change done": runs both offline
# pytest suites (test/ and erc_urban_test/tests, the same ones documented
# under Tests in the README), then a mirror-drift check, then
# git diff --check. Exits nonzero if anything fails.
#
# Mirror-drift check: erc_urban and erc_urban_test are two forks of the
# same stack and are expected to diverge (erc_urban_test carries sandbox
# tools like local_run.sh and its own tests/, and files like run.py differ
# by design). But when a file exists at the same relative path under both
# trees (e.g. modules/baseline.py, sdk_client.py), a real change to one
# side that has the same code on the other side is supposed to be mirrored
# there too. This check compares the current diff (commits since BaseRef,
# plus any staged/unstaged changes) against origin/main by default and
# fails if a shared-path file changed on one side but not the other.
# Files that only exist on one side are not mirrors and are skipped.
# Runs outside a git checkout, or against a BaseRef that doesn't exist
# locally, print "skip" instead of failing (the pytest suites still run).
#
# Usage:
#   pwsh scripts/verify.ps1
#   pwsh scripts/verify.ps1 -BaseRef main
#   pwsh scripts/verify.ps1 -PythonExe C:\path\to\python.exe   # e.g. the rover_env conda python
param(
    [string]$BaseRef = "origin/main",
    [string]$PythonExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    $fail = $false
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

    if ($PythonExe) {
        if (-not (Test-Path $PythonExe)) { throw "PythonExe not found: $PythonExe" }
        $py = $PythonExe
    } else {
        $python = Get-Command python3 -ErrorAction SilentlyContinue
        if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
        if (-not $python) { throw "no python3/python on PATH (pass -PythonExe)" }
        $py = $python.Source
    }

    Write-Host "== test/"
    & $py -m pytest -q -p no:cacheprovider test --ignore=test/test_live_sdk.py
    if ($LASTEXITCODE -ne 0) { $fail = $true }

    Write-Host "== erc_urban_test/tests"
    & $py -m pytest -q -p no:cacheprovider erc_urban_test/tests
    if ($LASTEXITCODE -ne 0) { $fail = $true }

    Write-Host "== mirror-drift (erc_urban vs erc_urban_test)"
    $git = Get-Command git -ErrorAction SilentlyContinue
    $inGit = $false
    if ($git) {
        & git rev-parse --is-inside-work-tree *> $null
        $inGit = ($LASTEXITCODE -eq 0)
    }
    if (-not $inGit) {
        Write-Host "skip: not a git checkout"
    } else {
        & git rev-parse --verify -q $BaseRef *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "skip: base ref '$BaseRef' not found (pass -BaseRef <ref>)"
        } else {
            $committed = & git diff --name-only "$BaseRef...HEAD" -- erc_urban erc_urban_test
            $unstaged  = & git diff --name-only -- erc_urban erc_urban_test
            $staged    = & git diff --cached --name-only -- erc_urban erc_urban_test
            # @() each one first: a single-line git output is a scalar string,
            # and "a" + "b" concatenates instead of building a list.
            $changed = @(@($committed) + @($unstaged) + @($staged) | Where-Object { $_ } | Sort-Object -Unique)
            $drift = $false
            foreach ($path in $changed) {
                if ($path -like "erc_urban/*") {
                    $other = "erc_urban_test/" + $path.Substring("erc_urban/".Length)
                } elseif ($path -like "erc_urban_test/*") {
                    $other = "erc_urban/" + $path.Substring("erc_urban_test/".Length)
                } else {
                    continue
                }
                if (-not (Test-Path $other)) { continue }
                if ($changed -notcontains $other) {
                    Write-Host "drift: $path changed but $other did not"
                    $drift = $true
                }
            }
            if ($drift) {
                $fail = $true
            } else {
                Write-Host "ok: no one-sided changes to mirrored files"
            }
        }
    }

    Write-Host "== git diff --check (whitespace)"
    if ($inGit) {
        & git diff --check
        if ($LASTEXITCODE -ne 0) { $fail = $true }
        & git diff --cached --check
        if ($LASTEXITCODE -ne 0) { $fail = $true }
        # Committed changes on this branch, which the two checks above never see.
        & git rev-parse --verify -q $BaseRef *> $null
        if ($LASTEXITCODE -eq 0) {
            & git diff --check "$BaseRef...HEAD"
            if ($LASTEXITCODE -ne 0) { $fail = $true }
        } else {
            Write-Host "skip committed check: base ref '$BaseRef' not found (pass -BaseRef <ref>)"
        }
    } else {
        Write-Host "skip: not a git checkout"
    }

    if ($fail) {
        Write-Host "VERIFY FAILED"
        exit 1
    }
    Write-Host "VERIFY OK"
} finally {
    Pop-Location
}
