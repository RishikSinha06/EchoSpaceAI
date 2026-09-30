$ErrorActionPreference = 'Stop'
$workspace = 'C:\DL-project sem V'
$manifest = Get-Content -LiteralPath (Join-Path $workspace 'data\raw\_manifests\raf-furnished.json') -Raw | ConvertFrom-Json
$destination = Join-Path $workspace 'data\raw\RAF\FurnishedRoom\rir_parts'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($item in $manifest) {
    $filename = Split-Path -Path $item.key -Leaf
    $target = Join-Path $destination $filename
    $url = 'https://s3.amazonaws.com/fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/' + $item.key
    $stalledAttempts = 0
    while ($true) {
        $before = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
        if ($before -eq [long]$item.size) { Write-Output "verified: $filename ($before bytes)"; break }
        if ($before -gt [long]$item.size) { throw "oversized RAF part: $filename ($before bytes)" }
        & curl.exe -L --fail --show-error --silent --max-time 120 --continue-at - $url --output $target
        $actual = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
        if ($actual -eq [long]$item.size) { Write-Output "verified: $filename ($actual bytes)"; break }
        if ($actual -gt [long]$item.size) { throw "oversized RAF part: $filename ($actual bytes)" }
        if ($actual -le $before) { $stalledAttempts++ } else { $stalledAttempts = 0 }
        if ($stalledAttempts -ge 6) { throw "RAF transfer stalled repeatedly at $filename : $actual bytes" }
        Write-Output "resuming: $filename ($actual / $($item.size) bytes)"
        Start-Sleep -Seconds 2
    }
}
