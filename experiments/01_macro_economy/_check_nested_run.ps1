$ErrorActionPreference = 'Continue'

# Find the active python process running _panel_backtest.py
$proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like '*python*' -and $_.StartTime -gt (Get-Date).AddHours(-2)
} | Sort-Object StartTime -Descending | Select-Object -First 1

if ($null -eq $proc) {
    Write-Host 'No active python process found.'
} else {
    $cpuMin = [math]::Round($proc.CPU / 60.0, 1)
    $memMB  = [math]::Round($proc.WorkingSet / 1MB, 1)
    $uptime = (Get-Date) - $proc.StartTime
    Write-Host ("PID {0}  CPU={1} min  Mem={2} MB  Uptime={3}" -f $proc.Id, $cpuMin, $memMB, $uptime.ToString())
}

Write-Host '---- stdout (last 50 lines) ----'
$stdoutPath = 'E:\project_gmd\data\features\_nested_cv_run.stdout.log'
if (Test-Path $stdoutPath) {
    $lines = Get-Content $stdoutPath
    Write-Host ("stdout line count: {0}" -f $lines.Count)
    $lines | Select-Object -Last 50 | ForEach-Object { Write-Host $_ }
} else { Write-Host 'stdout log missing.' }
Write-Host '---- stderr ----'
$errPath = 'E:\project_gmd\data\features\_nested_cv_run.stderr.log'
if (Test-Path $errPath) {
    $sz = (Get-Item $errPath).Length
    Write-Host ("stderr size: {0} bytes" -f $sz)
    if ($sz -gt 0) { Get-Content $errPath -Tail 30 | ForEach-Object { Write-Host $_ } }
} else { Write-Host 'stderr log missing.' }
Write-Host '---- output artifacts ----'
foreach ($f in @('walk_forward_cv.csv','walk_forward_cv_summary.json','walk_forward_cv_nested_params.json')) {
    $p = Join-Path 'E:\project_gmd\data\features' $f
    if (Test-Path $p) {
        $len = (Get-Item $p).Length
        $mtime = (Get-Item $p).LastWriteTime
        Write-Host ("{0}  size={1}  mtime={2}" -f $f, $len, $mtime)
    } else {
        Write-Host ("{0}  (not yet written)" -f $f)
    }
}