#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$models = @(
    @{
        Name = "hand_landmarker.task"
        Url  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    }
)

foreach ($model in $models) {
    $name = $model.Name
    $url  = $model.Url

    if (Test-Path -LiteralPath $name) {
        Write-Host "✓ $name already present, skipping"
        continue
    }

    Write-Host "↓ Downloading $name"
    $tmp = "$name.tmp"
    try {
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
        Move-Item -LiteralPath $tmp -Destination $name -Force
        Write-Host "✓ $name"
    }
    catch {
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
        throw
    }
}

Write-Host "Done."
