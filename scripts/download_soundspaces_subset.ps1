$ErrorActionPreference = 'Stop'
$workspace = 'C:\DL-project sem V'
$manifest = Get-Content -LiteralPath (Join-Path $workspace 'data\raw\_manifests\soundspaces-selected.json') -Raw | ConvertFrom-Json
$destination = Join-Path $workspace 'data\raw\SoundSpaces_Replica\binaural_rirs\replica'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($item in $manifest) {
    $filename = $item.scene + '.tar.gz'
    $target = Join-Path $destination $filename
    if ((Test-Path -LiteralPath $target) -and (Get-Item -LiteralPath $target).Length -eq [long]$item.bytes) {
        Write-Output "already complete: $filename"
        continue
    }
    $url = 'https://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs/replica/' + $filename
    & curl.exe -L --fail --show-error --silent --retry 6 --retry-delay 5 --continue-at - $url --output $target
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $filename" }
    $actual = (Get-Item -LiteralPath $target).Length
    if ($actual -ne [long]$item.bytes) { throw "size mismatch for $filename : $actual" }
    Write-Output "verified: $filename ($actual bytes)"
}
