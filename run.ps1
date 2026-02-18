param(
    [switch]$BuildData,
    [switch]$Train,
    [switch]$Validate,
    [switch]$GUI
)

if ($BuildData) { python src/adsorption_nn/dataset_build.py }
if ($Train)     { python src/adsorption_nn/train.py }
if ($Validate)  { python src/adsorption_nn/validate.py }
if ($GUI)       { python src/adsorption_nn/gui_flet.py }
