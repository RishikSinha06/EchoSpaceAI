param([string]$Scene = 'apartment_0', [long]$ExpectedBytes = 14298640606)
$ErrorActionPreference = 'Stop'
$target = Join-Path 'C:\DL-project sem V\data\raw\SoundSpaces_Replica\binaural_rirs\replica' ($Scene + '.tar.gz')
$url = 'https://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs/replica/' + $Scene + '.tar.gz'
$stalls = 0
while ($true) {
    $before = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
    if ($before -eq $ExpectedBytes) { Write-Output "verified: $Scene ($before bytes)"; break }
    if ($before -gt $ExpectedBytes) { throw "oversized archive: $Scene" }
    & curl.exe -L --fail --show-error --silent --max-time 120 --continue-at - $url --output $target
    $actual = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
    if ($actual -eq $ExpectedBytes) { Write-Output "verified: $Scene ($actual bytes)"; break }
    if ($actual -le $before) { $stalls++ } else { $stalls = 0 }
    if ($stalls -ge 6) { throw "SoundSpaces transfer stalled repeatedly: $Scene" }
    Write-Output "resuming: $Scene ($actual / $ExpectedBytes bytes)"
    Start-Sleep -Seconds 2
}
