$log = "e:\research\project\data\features\_phase8_v2_h5.log"
$m = "e:\research\project\data\features\horizon_5y_v2\metrics.json"
if (Test-Path $m) { Write-Output "DONE" } else { Write-Output "RUNNING" }
Write-Output "----- log tail -----"
if (Test-Path $log) { Get-Content $log -Tail 60 }
else { Write-Output "(no log file yet)" }