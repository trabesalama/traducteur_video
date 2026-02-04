<#
Script: activate_venv.ps1
But: run this script in PowerShell to automatically find and activate a .venv in the current folder or any parent folder.
Usage: .\activate_venv.ps1
#>

function Find-AndActivateVenv {
    $current = Resolve-Path -Path .
    while ($current) {
        $candidate = Join-Path $current '.venv\Scripts\Activate.ps1'
        if (Test-Path $candidate) {
            Write-Host "Found venv activate script: $candidate" -ForegroundColor Green
            try {
                # Bypass ExecutionPolicy just for this dot-source call if needed
                & $candidate
                Write-Host "Activated .venv in: $current" -ForegroundColor Green
            } catch {
                Write-Warning "Activation failed: $_"
            }
            return $true
        }
        $parent = Split-Path $current -Parent
        if (-not $parent -or $parent -eq $current) { break }
        $current = $parent
    }
    return $false
}

# Execute when the script is invoked
if (-not (Find-AndActivateVenv)) {
    Write-Host "No .venv found in current or parent folders." -ForegroundColor Yellow
}
