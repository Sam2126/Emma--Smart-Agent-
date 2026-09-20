# Self-Improving Browser Agent - One-Click Startup Script
# Run from the project root: .\start_agent.ps1
#
# Chrome handling: the agent uses its OWN Chrome window and profile
# (~/.self_improving_agent/chrome_profile), opened when a task first needs the
# browser. Your Chrome and its profiles are never launched, copied or closed.
#
# Usage:
#   .\start_agent.ps1          start the agent here — or, if one is already
#                              running (desktop app), watch its live log
#   .\start_agent.ps1 -Force   stop the agent running elsewhere (and its desktop
#                              app) and run it in this console instead
#
# Stopping: Ctrl+C always stops the agent completely, wake word included —
# also when this window is only watching an agent the desktop app started.

param(
    [switch]$Force
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Self-Improving Browser Agent Startup  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$backendDir = Join-Path $PSScriptRoot "backend"
$venv = Join-Path $backendDir ".venv\Scripts\python.exe"

# Step 0: An agent may already be running, usually started by the desktop app.
# Starting a second one would kill it (freeing the ports below) — the two
# launchers did exactly that to each other on 2026-09-15. Without -Force, show
# that agent's live log here instead, which is what a console is wanted for.
$running = $null
try { $running = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5 } catch {}

# Stop the agent everywhere: the desktop app first (so it cannot start the agent
# again), then the backend itself. Found 2026-09-17: stopping the project left
# the agent listening for the wake word in the background.
function Stop-AgentEverywhere {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", 8767)
        $bytes = [System.Text.Encoding]::ASCII.GetBytes("quit")
        $client.GetStream().Write($bytes, 0, $bytes.Length)
        $client.Close()
        Write-Host "  Asked the desktop app to quit." -ForegroundColor DarkGray
    } catch {}
    try { Invoke-RestMethod -Uri "http://127.0.0.1:8000/app/shutdown" -Method Post -TimeoutSec 5 | Out-Null } catch {}
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try { Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null } catch { return $true }
    }
    return $false
}

if ($null -ne $running -and $null -ne $running.wake_listener) {
    if ($Force) {
        Write-Host "The agent is running elsewhere. Stopping it and its desktop app (-Force)..." -ForegroundColor Yellow
        $stopped = Stop-AgentEverywhere
        if (-not $stopped) {
            Write-Host "  It is still running. Quit it from the tray icon, then run this again." -ForegroundColor Red
            exit 1
        }
        Write-Host "  Stopped. Starting a fresh one in this console." -ForegroundColor Green
        Write-Host ""
    } else {
        $wake = if ($running.wake_listener.running) { "listening for '" + $running.wake_listener.wake_word + "'" } else { "wake word off" }
        Write-Host "The agent is already running ($wake) - started by the desktop app or another window." -ForegroundColor Green
        Write-Host "  Showing its live log below." -ForegroundColor DarkGray
        Write-Host "  Ctrl+C or Q: STOP the agent completely (desktop app and wake word too)." -ForegroundColor Yellow
        Write-Host "  D: stop watching and leave it running (quit it later from the tray icon)." -ForegroundColor DarkGray
        Write-Host "  To run it in THIS console instead:  .\start_agent.ps1 -Force" -ForegroundColor DarkGray
        Write-Host ""
        $log = Join-Path $PSScriptRoot "backend\data\logs\backend.log"
        $interactive = $true
        try { $null = [Console]::KeyAvailable } catch { $interactive = $false }
        if (-not $interactive) {
            if (Test-Path $log) { Get-Content -Path $log -Tail 30 }
            exit 0
        }
        $reader = $null
        [Console]::TreatControlCAsInput = $true
        try {
            if (Test-Path $log) {
                Get-Content -Path $log -Tail 30
                $stream = [System.IO.File]::Open($log, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
                $null = $stream.Seek(0, [System.IO.SeekOrigin]::End)
                $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
            } else {
                Write-Host "  No log file yet at $log." -ForegroundColor DarkGray
            }
            $lastCheck = Get-Date
            while ($true) {
                if ($null -ne $reader) {
                    while ($null -ne ($line = $reader.ReadLine())) { Write-Host $line }
                }
                while ([Console]::KeyAvailable) {
                    $key = [Console]::ReadKey($true)
                    $ctrlC = ($key.Key -eq [ConsoleKey]::C) -and (($key.Modifiers -band [ConsoleModifiers]::Control) -ne 0)
                    if ($ctrlC -or $key.Key -eq [ConsoleKey]::Q) {
                        Write-Host ""
                        Write-Host "Stopping the agent..." -ForegroundColor Yellow
                        if (Stop-AgentEverywhere) {
                            Write-Host "  Stopped. Nothing is listening any more." -ForegroundColor Green
                        } else {
                            Write-Host "  It did not stop. Quit it from the tray icon." -ForegroundColor Red
                        }
                        exit 0
                    }
                    if ($key.Key -eq [ConsoleKey]::D) {
                        Write-Host "Stopped watching. The agent keeps running: quit it from the tray icon." -ForegroundColor DarkGray
                        exit 0
                    }
                }
                if (((Get-Date) - $lastCheck).TotalSeconds -ge 5) {
                    $lastCheck = Get-Date
                    try { Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 | Out-Null } catch {
                        Write-Host "The agent has stopped." -ForegroundColor Green
                        exit 0
                    }
                }
                Start-Sleep -Milliseconds 300
            }
        } finally {
            [Console]::TreatControlCAsInput = $false
            if ($null -ne $reader) { $reader.Close() }
        }
    }
}

# Step 1: Free backend ports 8000/8765 (kill only stale backend processes)
Write-Host "[1/2] Checking backend ports (8000, 8765)..." -ForegroundColor Yellow
foreach ($checkPort in @(8000, 8765)) {
    $portProcs = netstat -ano 2>$null | Select-String "LISTENING" | Select-String ":$checkPort " | ForEach-Object { ($_ -split '\s+')[-1] } | Sort-Object -Unique
    foreach ($p in $portProcs) {
        if ($p -match '^\d+$' -and [int]$p -ne 0) {
            Stop-Process -Id ([int]$p) -Force -ErrorAction SilentlyContinue
            Write-Host "  Freed port $checkPort (pid=$p)" -ForegroundColor DarkGray
        }
    }
}

# Kill only stale agent Chrome processes (never your own Chrome)
Get-Process -Name chrome -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
    if ($cmdline -and $cmdline -like "*.self_improving_agent*") {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  Cleaned stale agent Chrome (pid=$($_.Id))" -ForegroundColor DarkGray
    }
}
Write-Host "  Backend ports ready." -ForegroundColor Green

# Helper: check if port 9222 is open
function Test-CdpPort {
    try {
        $t = New-Object System.Net.Sockets.TcpClient
        $a = $t.BeginConnect("127.0.0.1", 9222, $null, $null)
        $ok = $a.AsyncWaitHandle.WaitOne(400, $false)
        if ($ok -and $t.Connected) { $t.EndConnect($a); $t.Close(); return $true }
        $t.Close()
    } catch {}
    return $false
}

# Helper: wait up to N seconds for port 9222
function Wait-CdpPort([int]$maxSec) {
    for ($i = 0; $i -lt ($maxSec * 2); $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-CdpPort) { return $true }
    }
    return $false
}

# Step 2: Ensure Chrome is reachable on debug port 9222
Write-Host "[2/2] Checking Chrome connection..." -ForegroundColor Yellow

if (Test-CdpPort) {
    Write-Host "  [OK] A Chrome window already has debug port 9222 open. The agent will work in a new tab there." -ForegroundColor Green

} else {
    # Per user preference the backend starts browser-free; the agent's own
    # Chrome window comes up the moment a task needs the browser.
    Write-Host "  [INFO] The agent's Chrome window opens when a task first needs the browser." -ForegroundColor Cyan
    Write-Host "  It has its own profile: your Chrome profiles stay open and signed in." -ForegroundColor Cyan
    Write-Host "  To sign in to sites for the agent first, run start_chrome_debugging.bat." -ForegroundColor Cyan
}

# Step 3: Start backend
Write-Host ""
Write-Host "Starting backend..." -ForegroundColor Cyan
Write-Host "  API:       http://localhost:8000" -ForegroundColor Green
Write-Host "  WebSocket: ws://localhost:8765" -ForegroundColor Green
Write-Host "  Wake Word: Say 'Emma' to activate (hands-free), 'done' to finish" -ForegroundColor Magenta
Write-Host "  Press Ctrl+C to stop the agent completely (wake word included)." -ForegroundColor DarkGray
Write-Host ""

Set-Location $backendDir
& $venv -m app.main
