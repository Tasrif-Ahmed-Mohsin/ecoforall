$now = Get-Date
Write-Output "Now: $now"
$dir = "e:\research\project\data\features\horizon_5y_v2"
if (Test-Path $dir) {
    Get-ChildItem -Path $dir | ForEach-Object {
        Write-Output ("{0,-35} {1,12:N0}  {2}" -f $_.Name, $_.Length, $_.LastWriteTime)
    }
} else {
    Write-Output "(directory not yet created)"
}
Write-Output "----- log file size -----"
$log = "e:\research\project\data\features\_phase8_v2_h5.log"
if (Test-Path $log) {
    $info = Get-Item $log
    Write-Output ("log: {0:N0} bytes" -f $info.Length)
} else {
    Write-Output "(no log file)"
}