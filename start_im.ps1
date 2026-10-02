[CmdletBinding()]
param(
    [string]$Provider = "deepseek",
    [string]$ModelId = "deepseek-chat"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $repoRoot "env.ps1"

if (Test-Path -LiteralPath $envFile) {
    . $envFile
}

# 无论从哪个目录启动，都让 Python 找到 src 下的 ai / im / coding_agent。
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONIOENCODING = "utf-8"

$required = @("FEISHU_APP_ID", "FEISHU_APP_SECRET")
if ($Provider -eq "deepseek") {
    $required += "DEEPSEEK_API_KEY"
}
foreach ($name in $required) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing environment variable: $name. Put it in env.ps1 or set it before starting."
    }
}

Push-Location $repoRoot
try {
    & python -m im `
        --transport longconn `
        --provider $Provider `
        --model-id $ModelId `
        --feishu-app-id $env:FEISHU_APP_ID `
        --feishu-app-secret $env:FEISHU_APP_SECRET
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
