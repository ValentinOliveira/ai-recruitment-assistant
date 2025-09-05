# AI Recruitment Assistant - Project Structure Setup
# This script creates the complete directory structure and essential files

Write-Host "🚀 Setting up AI Recruitment Assistant Project Structure..." -ForegroundColor Cyan

# Create main directory structure
$directories = @(
    "src",
    "src/models",
    "src/training",
    "src/data",
    "src/api",
    "src/agents",
    "src/utils",
    "data/raw",
    "data/processed",
    "data/training",
    "data/validation",
    "configs",
    "scripts",
    "notebooks",
    "tests",
    "tests/unit",
    "tests/integration",
    "docker",
    "docs",
    "docs/learning-notes",
    "docs/api",
    "deployment",
    "deployment/aws",
    "deployment/local",
    "monitoring",
    "logs",
    ".github",
    ".github/workflows",
    "models",
    "models/checkpoints",
    "models/fine-tuned",
    "models/quantized"
)

foreach ($dir in $directories) {
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
    Write-Host "✅ Created directory: $dir" -ForegroundColor Green
}

Write-Host ""
Write-Host "📁 Project structure created successfully!" -ForegroundColor Green
Write-Host "Next: Run the WSL setup script to configure your deep learning environment" -ForegroundColor Yellow
