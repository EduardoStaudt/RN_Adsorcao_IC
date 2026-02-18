# Baixa o dataset do Google Drive e coloca em data/processed/adsorption/
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\scripts\download_dataset_adsorption.ps1

$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OUTDIR = Join-Path $ROOT "data\processed\adsorption"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null


$URL = "https://drive.google.com/file/d/1N7GK1NneEwtyptZESoI1yD2Ek2RQOV0j/view?usp=drive_link"

$OUTFILE = Join-Path $OUTDIR "dataset_FULL.npz"
Write-Host "[INFO] Baixando para: $OUTFILE"
Invoke-WebRequest -Uri $URL -OutFile $OUTFILE
Write-Host "[OK] Download concluído."

