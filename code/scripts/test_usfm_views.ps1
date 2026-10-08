param(
    [ValidateSet('SAX', 'LAX', 'Both')]
    [string]$View = 'SAX',
    [ValidateRange(1, 60)]
    [int]$Fps = 8,
    [ValidateRange(1, 1000)]
    [int]$Stride = 1
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonUsfm = Join-Path $projectRoot '.venv-usfm\Scripts\python.exe'
$pythonData = Join-Path $projectRoot '.venv\Scripts\python.exe'

function Invoke-CheckedPython {
    param(
        [string]$Interpreter,
        [string[]]$Arguments,
        [string]$Step
    )
    & $Interpreter @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Test-View {
    param([string]$Name)

    $lowerName = $Name.ToLowerInvariant()
    $dataset = "outputs/usfm_${lowerName}_only"
    $training = "outputs/usfm_${lowerName}_partial_8"
    $testing = "outputs/usfm_${lowerName}_partial_8_test"
    $checkpointDirectory = Join-Path $training 'outputs'

    if (-not (Test-Path (Join-Path $dataset 'export.json'))) {
        throw "Dataset $dataset is not complete. Run prepare_usfm_dataset.py first."
    }
    if (Test-Path (Join-Path $dataset '_INCOMPLETE')) {
        throw "Dataset $dataset is marked _INCOMPLETE. Re-export it first."
    }
    $best = Get-ChildItem -Path $checkpointDirectory -Filter 'best*.pth' -File -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $best) {
        throw "No best*.pth checkpoint found in $checkpointDirectory"
    }

    Write-Host "[$Name] checkpoint: $($best.FullName)"
    Invoke-CheckedPython -Interpreter $pythonUsfm -Step "$Name test inference" -Arguments @(
        'code/scripts/run_usfm.py',
        '--mode', 'test',
        '--dataset', $dataset,
        '--resume', $best.FullName,
        '--output', $testing,
        '--batch-size', '1',
        '--execute'
    )

    $predictionRoot = Join-Path $testing 'outputs'
    $predictionRun = Get-ChildItem -Path $predictionRoot -Directory -Filter 'best_test_dice*' -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $predictionRun) {
        throw "No best_test_dice* prediction directory found in $predictionRoot"
    }
    $predictions = Join-Path $predictionRun.FullName 'mask_pre'
    if (-not (Test-Path $predictions)) {
        throw "Prediction masks not found: $predictions"
    }

    $scores = Join-Path $testing 'scores'
    $gifs = Join-Path $testing 'gifs'
    Invoke-CheckedPython -Interpreter $pythonData -Step "$Name scoring" -Arguments @(
        'code/scripts/eval_usfm_predictions.py',
        '--dataset', $dataset,
        '--predictions', $predictions,
        '--output', $scores
    )
    Invoke-CheckedPython -Interpreter $pythonData -Step "$Name GIF export" -Arguments @(
        'code/scripts/export_usfm_prediction_gifs.py',
        '--dataset', $dataset,
        '--predictions', $predictions,
        '--output', $gifs,
        '--fps', [string]$Fps,
        '--stride', [string]$Stride
    )

    Write-Host "[$Name] masks: $predictions"
    Write-Host "[$Name] scores: $(Join-Path $scores 'summary.json')"
    Write-Host "[$Name] GIFs: $gifs"
    Get-Content -Path (Join-Path $scores 'summary.json') -Encoding UTF8
}

Push-Location $projectRoot
try {
    if (-not (Test-Path $pythonUsfm) -or -not (Test-Path $pythonData)) {
        throw 'Missing .venv-usfm or .venv Python interpreter.'
    }
    $env:MPLCONFIGDIR = (Resolve-Path 'outputs').Path
    $env:NO_ALBUMENTATIONS_UPDATE = '1'
    $views = if ($View -eq 'Both') { @('SAX', 'LAX') } else { @($View) }
    foreach ($name in $views) {
        Test-View -Name $name
    }
}
finally {
    Pop-Location
}
