param([Parameter(Mandatory=$true)][string]$Filename)
$ErrorActionPreference = 'Stop'
$root = 'C:\DL-project sem V'
$manifest = Get-Content -LiteralPath (Join-Path $root 'data\raw\_manifests\raf-furnished.json') -Raw | ConvertFrom-Json
$item = $manifest | Where-Object { (Split-Path -Path $_.key -Leaf) -eq $Filename } | Select-Object -First 1
if (-not $item) { throw "Unknown RAF part: $Filename" }
$target = Join-Path (Join-Path $root 'data\raw\RAF\FurnishedRoom\rir_parts') $Filename
$url = 'https://s3.amazonaws.com/fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/' + $item.key
$stalls = 0
while ($true) {
    $before = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
    if ($before -eq [long]$item.size) { Write-Output "verified: $Filename ($before bytes)"; break }
    if ($before -gt [long]$item.size) { throw "oversized part: $Filename" }
    & curl.exe -L --fail --show-error --silent --max-time 120 --continue-at - $url --output $target
    $actual = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
    if ($actual -eq [long]$item.size) { Write-Output "verified: $Filename ($actual bytes)"; break }
    if ($actual -le $before) { $stalls++ } else { $stalls = 0 }
    if ($stalls -ge 6) { throw "RAF part repeatedly stalled: $Filename" }
    Write-Output "resuming: $Filename ($actual / $($item.size) bytes)"
    Start-Sleep -Seconds 2
}
