#Requires -Version 5.1
<#
.SYNOPSIS
    在 Windows 10 (2004+) / Windows 11 上准备 SO-Agent 基础安全沙箱。
.DESCRIPTION
    检测 Docker/Podman 与镜像，执行隔离自检。-Install 会通过 winget 安装
    Docker Desktop（需要用户确认并可能重启）。脚本不会修改已有 .env 值。
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/sandbox_setup.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/sandbox_setup.ps1 -Install -WriteEnv
#>
param(
    [switch]$Install,
    [switch]$WriteEnv,
    [switch]$SkipPull,
    [string]$Image = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = (Join-Path $RepoRoot "src") + [IO.Path]::PathSeparator + $env:PYTHONPATH
$Python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { $null }
if (-not $Python) {
    Write-Error "未找到 Python，请先安装 Python 3.10+ 并加入 PATH"
    exit 1
}

function Test-Runtime([string]$Exe) {
    if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return $false }
    & $Exe info *> $null
    return ($LASTEXITCODE -eq 0)
}

$Runtime = $null
foreach ($candidate in @("docker", "podman")) {
    if (Test-Runtime $candidate) { $Runtime = $candidate; break }
}

if (-not $Runtime) {
    Write-Host "未检测到可用的 Docker/Podman。" -ForegroundColor Yellow
    Write-Host "Windows 10 需要 2004+ 并启用 WSL2；Windows 11 默认支持。"
    if ($Install) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Write-Host "未找到 winget，请从 Microsoft Store 安装 App Installer 后重试。" -ForegroundColor Red
            exit 2
        }
        Write-Host "即将安装 Docker Desktop（会弹出安装程序，可能需要重启）。"
        winget install --id Docker.DockerDesktop -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Docker Desktop 安装失败；也可安装 RedHat.Podman-Desktop。" -ForegroundColor Red
            exit 2
        }
        Write-Host "安装完成。请重启并启动 Docker Desktop，然后重新运行本脚本。" -ForegroundColor Yellow
        exit 2
    }
    Write-Host "安装命令：winget install --id Docker.DockerDesktop -e"
    Write-Host "或：      winget install --id RedHat.Podman-Desktop -e"
    Write-Host "安装并启动后重新运行：powershell -ExecutionPolicy Bypass -File scripts/sandbox_setup.ps1"
    Write-Host "在此之前项目按配置回退到本地验证沙箱（不是安全边界）。"
    exit 2
}

Write-Host "检测到容器运行时：$Runtime"
$setupArgs = @("-m", "so_agent.sandbox_cli", "setup")
if ($Image) { $setupArgs += @("--image", $Image) }
if (-not $SkipPull) { $setupArgs += "--pull-image" }
if ($WriteEnv) { $setupArgs += @("--write-env", (Join-Path $RepoRoot ".env")) }
& $Python @setupArgs
$setupCode = $LASTEXITCODE
if ($setupCode -ne 0) {
    Write-Host "沙箱配置未完成（状态码 $setupCode）。" -ForegroundColor Yellow
    exit $setupCode
}

& $Python -m so_agent.sandbox_cli selftest
exit $LASTEXITCODE
