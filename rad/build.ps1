# Компајлира рад.  Покретање:  .\build.ps1
# (latexmk се не користи јер захтева Perl, који није инсталиран)

Set-Location $PSScriptRoot

Write-Host "[1/4] xelatex ..." -ForegroundColor Cyan
xelatex -interaction=nonstopmode rad.tex | Out-Null

Write-Host "[2/4] biber ..." -ForegroundColor Cyan
biber rad | Out-Null

Write-Host "[3/4] xelatex ..." -ForegroundColor Cyan
xelatex -interaction=nonstopmode rad.tex | Out-Null

Write-Host "[4/4] xelatex ..." -ForegroundColor Cyan
xelatex -interaction=nonstopmode rad.tex | Out-Null

$errors = @(Select-String -Path rad.log -Pattern '^!' -ErrorAction SilentlyContinue)
$pages  = @(Select-String -Path rad.log -Pattern 'Output written' -ErrorAction SilentlyContinue | ForEach-Object { $_.Line.Trim() }) | Select-Object -Last 1

# Помоћне датотеке се бришу; PDF и .log остају.
Remove-Item -ErrorAction SilentlyContinue *.aux,*.bbl,*.bcf,*.blg,*.out,*.toc,*.run.xml,pass*.txt,biber.txt

if ($errors.Count -gt 0) {
    Write-Host "`nГРЕШКЕ:" -ForegroundColor Red
    $errors | Select-Object -First 15
} else {
    Write-Host "`nГотово. $pages" -ForegroundColor Green
}
