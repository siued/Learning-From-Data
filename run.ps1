param (
    [string]$PythonFile,
    [string]$OutputDir
)

# Check if a Python file is provided
if (-not $PythonFile) {
    Write-Host "Usage: .\$($MyInvocation.MyCommand) <python_file> [output_dir]"
    exit 1
}

$BaseDir = Split-Path -Path $PythonFile
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Set output directory: use second argument if provided, otherwise create a timestamped directory
$OutDir = Join-Path -Path $BaseDir -ChildPath (if ($OutputDir) { $OutputDir } else { "output_$Timestamp" })
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$OutFile = Join-Path -Path $OutDir -ChildPath "output.txt"
$PythonFileCopy = Join-Path -Path $OutDir -ChildPath ("{0}_copy.py" -f [System.IO.Path]::GetFileNameWithoutExtension($PythonFile))

# Run the Python file, save output to file and print it to stdout
python $PythonFile 2>&1 | Tee-Object -FilePath $OutFile

# Copy the Python file to the output directory
Copy-Item -Path $PythonFile -Destination $PythonFileCopy
