param(
    [Parameter(Mandatory = $true)][string]$Dir
)

$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $here $Dir

if (-not (Test-Path $target)) { throw "no such dir: $target" }
Push-Location $target

$flags = @("-interaction=nonstopmode", "-halt-on-error", "-enable-installer")

Write-Host "=== pass 1: $Dir ==="
& pdflatex @flags main.tex | Out-Null
Write-Host "exit=$LASTEXITCODE"

if (Test-Path main.aux) {
    Write-Host "=== bibtex: $Dir ==="
    & bibtex main | Out-Null
    Write-Host "exit=$LASTEXITCODE"
}

foreach ($i in 2..3) {
    Write-Host "=== pass $i: $Dir ==="
    & pdflatex @flags main.tex | Out-Null
    Write-Host "exit=$LASTEXITCODE"
}

Pop-Location

$pdf = Join-Path $target "main.pdf"
if (Test-Path $pdf) {
    $size = (Get-Item $pdf).Length
    Write-Host "PDF OK: $pdf ($size bytes)"
} else {
    Write-Host "PDF MISSING"
}
