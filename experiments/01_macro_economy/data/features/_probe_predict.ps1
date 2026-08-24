$env:PYTHONPATH = 'E:\project_gmd;E:\project_gmd\scripts'
Set-Location E:\project_gmd
$body = '{"iso3":"USA","year":2023,"horizon":5,"ranked":true,"k":5,"min_overlap":60}'
try {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/stream' `
        -Method POST -Body $body -ContentType 'application/json' -UseBasicParsing -TimeoutSec 60
  Write-Output ('STATUS=' + $r.StatusCode)
  Write-Output ($r.Content.Substring(0, [Math]::Min(2000, $r.Content.Length)))
} catch {
  Write-Output ('ERR: ' + $_.Exception.Message)
}