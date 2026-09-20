<#
.SYNOPSIS
    Pokreće ceo lanac bez nadzora: panel v3 -> prognoze -> izveštaj -> dijagnostika.

.DESCRIPTION
    Četiri koraka, redom, sa zaustavljanjem na prvoj grešci:

      1. build_panel      28 simbola x 36 dizajna -> data/processed/v3/panel.parquet
      2. run_parallel     --models fast (16 modela, 246 kombinacija) -> results/run_v3
      3. run_report       svi artefakti nivoa 1-4  -> results/run_v3/report
      4. run_diagnostics  H5 placebo + učestalost ponovne ocene -> isti report dir

    NASTAVLJIV: svaki korak ima marker završenosti i preskače se ako je gotov,
    pa se skripta posle pada može pokrenuti ponovo bez gubitka posla. Korak 2 je
    nastavljiv i iznutra (šardovi preskaču tabele koje već postoje na disku), pa
    se pokreće i kad je delimično gotov.

    Bez interaktivnih pitanja. Svaki korak ima svoj log pod logs/ sa vremenskom
    oznakom, plus zajednički logs/run_all.log; sve se istovremeno ispisuje i na
    ekran.

.NOTES
    Tee je BAJTOVNI, ne linijski: run_parallel.py ispisuje napredak preko "\r"
    bez novog reda, pa bi linijski pipeline zadržao ceo ispis do kraja koraka.
    Zato se izlaz preusmerava u fajl kroz cmd (>"log" 2>&1), a skripta taj fajl
    prati i prepisuje nove bajtove na konzolu i u zajednički log.
#>

[CmdletBinding()]
param(
    [string]$RunDir = "results/run_v3",
    [string]$Panel  = "data/processed/v3/panel.parquet",
    [int]$Shards    = 18,
    [int]$Workers   = 8
)

Set-StrictMode -Version 2.0
# Native izlazni kod se čita iz $LASTEXITCODE, nikad iz $? -- u Windows
# PowerShell-u 5.1 "2>&1" nad native programom pakuje svaki red stderr-a u
# ErrorRecord i obara $? i kad je program vratio 0.
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'
$ConfirmPreference     = 'None'
$env:PYTHONIOENCODING  = 'utf-8'
$env:PYTHONUNBUFFERED  = '1'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Stamp       = Get-Date -Format 'yyyyMMdd-HHmmss'
$CombinedLog = Join-Path $LogDir 'run_all.log'

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Combined  = New-Object System.IO.StreamWriter($CombinedLog, $true, $Utf8NoBom)
$Combined.AutoFlush = $true

# 28 simbola iz config.UNIVERSE (MT i NVDA su isključeni, vidi EXCLUDED_SYMBOLS)
$Symbols = @(
    'AAPL','MSFT','INTC','AMD','BAC','AMZN','MU','C','JPM','WFC','CSCO','XOM',
    'ORCL','QCOM','PFE','T','NFLX','MS','AMAT','WMT','CVX','DIS','VZ','F',
    'ACN','CB','SU','AEM'
)

# 10 modela zavisnih samo od mreže x 3 dizajna + 6 osetljivih na dizajn x 36
$ExpectedCombinations = 246

function Write-Both {
    param([string]$Text)
    [Console]::Out.Write($Text)
    $script:Combined.Write($Text)
}

function Write-Banner {
    param([string]$Text)
    Write-Both ("`n" + ('=' * 78) + "`n$Text`n" + ('=' * 78) + "`n")
}

# Prepiši nove bajtove iz fajla na konzolu i u zajednički log. Bajtovno, da bi
# se sačuvao "\r" napredak koji ispisuje run_parallel.py.
function Copy-NewBytes {
    param([string]$Path, [ref]$Position)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $stream = $null
    try {
        $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
                                         [System.IO.FileAccess]::Read,
                                         [System.IO.FileShare]::ReadWrite)
        if ($stream.Length -le $Position.Value) { return }
        $count = [int]($stream.Length - $Position.Value)
        $stream.Position = $Position.Value
        $buffer = New-Object byte[] $count
        $read = $stream.Read($buffer, 0, $count)
        $Position.Value += $read
        Write-Both ([System.Text.Encoding]::UTF8.GetString($buffer, 0, $read))
    } catch {
        # fajl je trenutno zaključan -- probaj ponovo u sledećem krugu
    } finally {
        if ($null -ne $stream) { $stream.Dispose() }
    }
}

function Invoke-Step {
    <#  Pokreće jednu komandu kroz cmd radi objedinjenog preusmeravanja
        (stdout i stderr u isti fajl), prati taj fajl i vraća izlazni kod. #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Slug,
        [Parameter(Mandatory = $true)][string]$CommandLine
    )
    $stepLog = Join-Path $LogDir "$Stamp-$Slug.log"
    Write-Banner "KORAK: $Name`nlog  : $stepLog`nkoman: $CommandLine"
    $started = Get-Date

    $inner = "$CommandLine > `"$stepLog`" 2>&1"
    $proc = Start-Process -FilePath $env:ComSpec -ArgumentList '/c', $inner `
                          -NoNewWindow -PassThru -WorkingDirectory $Root
    # Pristup .Handle kešira handle procesa. Bez toga .ExitCode ostaje PRAZAN
    # i posle WaitForExit(), pa bi provera pada koraka bila neispravna.
    $null = $proc.Handle
    $pos = 0
    while (-not $proc.HasExited) {
        Copy-NewBytes -Path $stepLog -Position ([ref]$pos)
        Start-Sleep -Milliseconds 700
    }
    $proc.WaitForExit()
    Copy-NewBytes -Path $stepLog -Position ([ref]$pos)

    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) { $exitCode = 1 }   # nepoznat ishod tretiraj kao pad
    $elapsed = (Get-Date) - $started
    $script:Timings += [pscustomobject]@{
        Step = $Name; Elapsed = $elapsed; Log = $stepLog; Skipped = $false
    }
    Write-Both ("`n-- ${Name}: izlazni kod $exitCode, trajanje {0:hh\:mm\:ss}`n" -f $elapsed)
    return $exitCode
}

function Stop-WithFailure {
    param([string]$Name, [int]$Code)
    $entry = @($script:Timings | Where-Object { $_.Step -eq $Name }) | Select-Object -Last 1
    $log = '(bez loga)'
    if ($null -ne $entry) { $log = $entry.Log }
    Write-Banner "PAO KORAK: $Name (izlazni kod $Code)`nlog: $log`n`nPopravi uzrok pa pokreni istu komandu ponovo -- zavrseni koraci se preskacu."
    $script:Combined.Dispose()
    exit $Code
}

function Skip-Step {
    param([string]$Name, [string]$Why)
    Write-Both "`n-- PRESKACEM $Name -- $Why`n"
    $script:Timings += [pscustomobject]@{
        Step = $Name; Elapsed = [TimeSpan]::Zero; Log = '(preskoceno)'; Skipped = $true
    }
}

function Count-Files {
    param([string]$Path, [string]$Filter)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return @(Get-ChildItem -LiteralPath $Path -Filter $Filter -File -ErrorAction SilentlyContinue).Count
}

$Timings = @()
$OverallStart = Get-Date

Write-Both "`n"
Write-Banner "run_all.ps1  pokrenut $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`nrepo : $Root`nlog  : $CombinedLog"

if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Both "GRESKA: 'uv' nije na PATH-u. Otvori ljusku u kojoj 'uv --version' radi.`n"
    $Combined.Dispose()
    exit 127
}

$ReportDir = Join-Path $RunDir 'report'
$TablesDir = Join-Path $RunDir 'tables'

# ── 1. panel ─────────────────────────────────────────────────────────────
$panelDir = Split-Path -Parent $Panel
if ($panelDir -and -not (Test-Path $panelDir)) {
    New-Item -ItemType Directory -Path $panelDir -Force | Out-Null
}
if (Test-Path -LiteralPath $Panel) {
    Skip-Step 'panel' "$Panel vec postoji"
} else {
    $cmd = "uv run python -m projekat.build_panel --symbols $($Symbols -join ' ')" +
           " --start 2010-01-01 --end 2025-12-31" +
           " --grids 1m 5m 15m --estimators bpv medrv trbpv" +
           " --jump-tests naive bns lm lm_fdr --out `"$Panel`""
    $code = Invoke-Step -Name 'panel' -Slug 'panel' -CommandLine $cmd
    # build_panel vraca 1 ako je BILO KOJI simbol pao: panel je tada nepotpun,
    # pa lanac mora da stane umesto da se modeluje na 27 simbola.
    if ($code -ne 0) { Stop-WithFailure 'panel' $code }
}

# ── 2. prognoze ──────────────────────────────────────────────────────────
$done = Count-Files -Path $TablesDir -Filter '*.parquet'
if ($done -ge $ExpectedCombinations) {
    Skip-Step 'prognoze' "$done/$ExpectedCombinations tabela vec na disku"
} else {
    if ($done -gt 0) { Write-Both "`n-- nastavljam prognoze: $done/$ExpectedCombinations tabela vec postoji`n" }
    $cmd = "uv run python scripts/run_parallel.py --run-dir `"$RunDir`"" +
           " --panel `"$Panel`" --models fast --shards $Shards"
    $code = Invoke-Step -Name 'prognoze' -Slug 'prognoze' -CommandLine $cmd
    # run_parallel vraca 1 samo ako je neki SARD pao; pojedinacne kombinacije
    # koje padnu zavrse u failures.jsonl i prijavljuju se u rezimeu ispod.
    if ($code -ne 0) { Stop-WithFailure 'prognoze' $code }
}

# ── 3. izveštaj ──────────────────────────────────────────────────────────
$reportDone = (Test-Path (Join-Path $ReportDir 'main_result_table.csv')) -and
              (Test-Path (Join-Path $ReportDir 'jump_characterization.json'))
if ($reportDone) {
    Skip-Step 'izvestaj' 'main_result_table.csv i jump_characterization.json vec postoje'
} else {
    $cmd = "uv run python scripts/run_report.py --run-dir `"$RunDir`" --panel `"$Panel`""
    $code = Invoke-Step -Name 'izvestaj' -Slug 'izvestaj' -CommandLine $cmd
    if ($code -ne 0) { Stop-WithFailure 'izvestaj' $code }
}

# ── 4. dijagnostika ──────────────────────────────────────────────────────
if (Test-Path (Join-Path $ReportDir 'h5_placebo.csv')) {
    Skip-Step 'dijagnostika' 'h5_placebo.csv vec postoji'
} else {
    $cmd = "uv run python scripts/run_diagnostics.py --panel `"$Panel`"" +
           " --out `"$ReportDir`" --workers $Workers"
    $code = Invoke-Step -Name 'dijagnostika' -Slug 'dijagnostika' -CommandLine $cmd
    if ($code -ne 0) { Stop-WithFailure 'dijagnostika' $code }
}

# ── rezime ───────────────────────────────────────────────────────────────
Write-Banner 'REZIME'

$panelRows = 'n/a'
if (Test-Path -LiteralPath $Panel) {
    $py = "import pyarrow.parquet as pq;" +
          "m=pq.ParquetFile(r'$Panel').metadata;print(f'{m.num_rows:,} redova, {m.num_columns} kolona')"
    $panelRows = (& uv run python -c $py 2>$null | Select-Object -Last 1)
    if (-not $panelRows) { $panelRows = 'n/a (citanje nije uspelo)' }
}

$tables   = Count-Files -Path $TablesDir -Filter '*.parquet'
$failFile = Join-Path $RunDir 'failures.jsonl'
$failures = 0
if (Test-Path -LiteralPath $failFile) {
    $failures = @(Get-Content -LiteralPath $failFile -ErrorAction SilentlyContinue |
                  Where-Object { $_.Trim() -ne '' }).Count
}
$artefacts = 0
if (Test-Path -LiteralPath $ReportDir) {
    $artefacts = @(Get-ChildItem -LiteralPath $ReportDir -File -ErrorAction SilentlyContinue).Count
}

Write-Both "panel            : $panelRows`n"
Write-Both "                   $Panel`n"
Write-Both "kombinacije      : $tables / $ExpectedCombinations tabela`n"
Write-Both "pale kombinacije : $failures"
if ($failures -gt 0) { Write-Both "   <-- vidi $failFile" }
Write-Both "`n"
Write-Both "artefakti        : $artefacts fajlova u $ReportDir`n"
Write-Both "`nvreme po koraku:`n"
foreach ($t in $Timings) {
    $mark = ''
    if ($t.Skipped) { $mark = '  (preskoceno)' }
    Write-Both ("  {0,-14} {1:hh\:mm\:ss}{2}`n" -f $t.Step, $t.Elapsed, $mark)
}
Write-Both ("  {0,-14} {1:hh\:mm\:ss}`n" -f 'UKUPNO', ((Get-Date) - $OverallStart))

$contam = Join-Path $RunDir 'ttm_contamination_check.json'
if (Test-Path -LiteralPath $contam) {
    Write-Both "`nkontaminacija TTM-a: $contam`n"
}

Write-Banner "GOTOVO  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`nzajednicki log: $CombinedLog"
$Combined.Dispose()
exit 0
