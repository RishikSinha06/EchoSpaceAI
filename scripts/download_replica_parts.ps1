param([ValidateSet('b','c','d','e')][string]$LastPart = 'c')
$ErrorActionPreference = 'Stop'
$destination = 'C:\DL-project sem V\data\raw\SoundSpaces_Replica\replica_geometry_parts'
foreach ($letter in @('a','b','c','d','e')) {
    if ([int][char]$letter -gt [int][char]$LastPart) { break }
    $filename = "replica_v1_0.tar.gz.parta$letter"
    $target = Join-Path $destination $filename
    $url = 'https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/' + $filename
    $stalls = 0
    while ($true) {
        $before = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
        if ($before -eq 2000000000) { Write-Output "verified: $filename"; break }
        if ($before -gt 2000000000) { throw "oversized part: $filename" }
        & curl.exe -L --fail --show-error --silent --max-time 120 --continue-at - $url --output $target
        $actual = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
        if ($actual -eq 2000000000) { Write-Output "verified: $filename"; break }
        if ($actual -le $before) { $stalls++ } else { $stalls = 0 }
        if ($stalls -ge 6) { throw "Replica part repeatedly stalled: $filename" }
        Write-Output "resuming: $filename ($actual / 2000000000 bytes)"
        Start-Sleep -Seconds 2
    }
}
