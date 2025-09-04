# create_environment.ps1
$ErrorActionPreference = "Stop" # Exit immediately if a command exits with a non-zero status.

$ENV_NAME = "ml-project"
$PYTHON_VERSION = "3.9" # Recommended for broader compatibility with ML libs

Write-Host "Starting environment setup for '$ENV_NAME'..."

# --- 1. Conda Installation Check ---
# This check is mostly for robustness; Conda should already be installed and in PATH.
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda command not found. Please ensure Miniconda is installed and added to your system's PATH."
    Write-Host "You might need to restart PowerShell after installing Miniconda."
    exit 1
}

# --- 2. Conda Environment Creation ---
Write-Host "Checking for existing environment '$ENV_NAME'..."
$conda_envs = conda env list | Out-String
if ($conda_envs -match $ENV_NAME) {
    Write-Host "Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
}

Write-Host "Creating new Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# --- 3. Install Dependencies ---
Write-Host "Installing dependencies from dev_requirements.txt into '$ENV_NAME'..."
$dev_req_path = "dev_requirements.txt"
if (Test-Path $dev_req_path) {
    # Use 'conda run' to execute pip within the specified environment
    conda run -n $ENV_NAME pip install -r $dev_req_path
    Write-Host "Downloading NLTK data..."
    conda run -n $ENV_NAME python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
} else {
    Write-Host "Error: dev_requirements.txt not found. Please ensure it exists in the current directory."
    exit 1
}

# --- 4. Jupyter Kernel Setup ---
Write-Host "Setting up Jupyter kernel for '$ENV_NAME'..."
conda run -n $ENV_NAME python -m ipykernel install --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

Write-Host "Environment setup complete for '$ENV_NAME'."
Write-Host "To activate this environment, run: conda activate $ENV_NAME"
Write-Host "To start Jupyter Notebook, run: jupyter notebook"