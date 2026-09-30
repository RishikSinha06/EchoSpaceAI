$ErrorActionPreference = 'Stop'
$workspace = 'C:\DL-project sem V'
$manifest = Get-Content -LiteralPath (Join-Path $workspace 'data\raw\_manifests\raf-furnished.json') -Raw | ConvertFrom-Json
$destination = Join-Path $workspace 'data\raw\RAF\FurnishedRoom\rir_parts'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$pending = @($manifest | Where-Object {
    $target = Join-Path $destination (Split-Path -Path $_.key -Leaf)
    -not (Test-Path -LiteralPath $target) -or (Get-Item -LiteralPath $target).Length -ne [long]$_.size
})
if ($pending.Count -eq 0) { Write-Output 'All RAF parts complete.'; exit 0 }
$curlArgs = @('--parallel', '--parallel-immediate', '--parallel-max', '3')
foreach ($item in $pending) {
    $filename = Split-Path -Path $item.key -Leaf
    $target = Join-Path $destination $filename
    $url = 'https://s3.amazonaws.com/fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/' + $item.key
    $curlArgs += @('-L', '--fail', '--show-error', '--silent', '--retry', '20', '--retry-delay', '5', '--speed-limit', '1024', '--speed-time', '45', '--continue-at', '-', '--url', $url, '--output', $target, '--next')
}
$curlArgs = $curlArgs[0..($curlArgs.Count - 2)]
& curl.exe @curlArgs
if ($LASTEXITCODE -ne 0) { throw "parallel curl failed with code $LASTEXITCODE" }
foreach ($item in $manifest) {
    $filename = Split-Path -Path $item.key -Leaf
    $target = Join-Path $destination $filename
    if (-not (Test-Path -LiteralPath $target)) { throw "missing RAF part: $filename" }
    $actual = (Get-Item -LiteralPath $target).Length
    if ($actual -ne [long]$item.size) { throw "size mismatch for $filename : $actual" }
    Write-Output "verified: $filename ($actual bytes)"
}
