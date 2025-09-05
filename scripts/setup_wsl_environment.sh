#!/bin/bash

# =============================================================================
# AI Recruitment Assistant - WSL Deep Learning Environment Setup
# =============================================================================
# This script sets up a complete deep learning environment in WSL2 Ubuntu
# Optimized for fine-tuning Llama 3.1 8B on RTX 4060 with 8GB VRAM
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running in WSL
check_wsl() {
    if grep -qi microsoft /proc/version; then
        log "✅ Running in WSL environment"
    else
        log_error "❌ This script must be run in WSL"
        exit 1
    fi
}

# System information
show_system_info() {
    log "📊 System Information:"
    echo "OS: $(lsb_release -d | cut -f2)"
    echo "Kernel: $(uname -r)"
    echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    echo "CPU: $(nproc) cores"
    echo ""
}

# Update system packages
update_system() {
    log "📦 Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y curl wget git build-essential software-properties-common
}

# Install NVIDIA drivers and CUDA
install_cuda() {
    log "🚀 Installing NVIDIA CUDA Toolkit..."
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        log_info "CUDA already installed: $(nvcc --version | grep 'release' | awk '{print $5,$6}')"
        return
    fi
    
    # Remove any existing CUDA installations
    sudo apt remove --purge -y cuda* nvidia-cuda-toolkit
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    
    # Install CUDA 12.6
    sudo apt install -y cuda-toolkit-12-6
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Source the bashrc
    source ~/.bashrc
    
    log "✅ CUDA installation completed"
}

# Install Miniconda
install_miniconda() {
    log "🐍 Installing Miniconda..."
    
    if command -v conda &> /dev/null; then
        log_info "Conda already installed"
        return
    fi
    
    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    
    log "✅ Miniconda installation completed"
}

# Create and setup Python environment
setup_python_environment() {
    log "🐍 Setting up Python environment for AI/ML..."
    
    # Ensure conda is available
    source $HOME/miniconda3/bin/activate
    
    # Create environment
    conda create -n recruitment-assistant python=3.11 -y
    source activate recruitment-assistant
    
    log "✅ Python 3.11 environment created"
}

# Install PyTorch with CUDA support
install_pytorch() {
    log "🔥 Installing PyTorch with CUDA support..."
    
    source $HOME/miniconda3/bin/activate recruitment-assistant
    
    # Install PyTorch with CUDA 12.1 support (compatible with RTX 4060)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install additional ML libraries
    pip install transformers accelerate peft datasets bitsandbytes
    pip install wandb tensorboard matplotlib seaborn pandas numpy scipy
    pip install fastapi uvicorn pydantic python-multipart
    pip install pytest pytest-cov black isort flake8 mypy
    pip install jupyter jupyterlab ipywidgets
    
    log "✅ PyTorch and ML libraries installed"
}

# Install development tools
install_dev_tools() {
    log "🛠️ Installing development tools..."
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Install additional tools
    sudo apt install -y htop tree jq unzip zip
    
    # Install GitHub CLI
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update && sudo apt install gh -y
    
    log "✅ Development tools installed"
}

# Setup SSH server for remote access
setup_ssh() {
    log "🔐 Setting up SSH server..."
    
    sudo apt install -y openssh-server
    
    # Configure SSH
    sudo sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
    sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
    
    # Start SSH service
    sudo service ssh start
    
    # Add to startup
    echo 'sudo service ssh start' >> ~/.bashrc
    
    log "✅ SSH server configured on port 2222"
}

# Create project configuration files
create_config_files() {
    log "📄 Creating configuration files..."
    
    # Go to project root
    cd /mnt/c/Users/$(whoami)/engineer-blog/ai-recruitment-assistant
    
    # Create environment.yml
    cat > environment.yml << 'EOF'
name: recruitment-assistant
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pip
  - pip:
    - torch>=2.1.0
    - torchvision>=0.16.0
    - torchaudio>=2.1.0
    - transformers>=4.35.0
    - accelerate>=0.24.0
    - peft>=0.6.0
    - datasets>=2.14.0
    - bitsandbytes>=0.41.0
    - wandb>=0.16.0
    - tensorboard>=2.15.0
    - fastapi>=0.104.0
    - uvicorn>=0.24.0
    - pydantic>=2.5.0
    - python-multipart>=0.0.6
    - pytest>=7.4.0
    - pytest-cov>=4.1.0
    - black>=23.9.0
    - isort>=5.12.0
    - flake8>=6.1.0
    - mypy>=1.6.0
    - jupyter>=1.0.0
    - jupyterlab>=4.0.0
    - ipywidgets>=8.1.0
    - matplotlib>=3.8.0
    - seaborn>=0.13.0
    - pandas>=2.1.0
    - numpy>=1.25.0
    - scipy>=1.11.0
EOF

    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
recruitment-assistant/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints
*.ipynb

# Model files and checkpoints
models/checkpoints/*.pt
models/checkpoints/*.bin
models/checkpoints/*.safetensors
models/fine-tuned/*.pt
models/fine-tuned/*.bin
models/fine-tuned/*.safetensors
models/quantized/*.pt
models/quantized/*.bin
models/quantized/*.safetensors

# Data files
data/raw/*.csv
data/raw/*.json
data/raw/*.parquet
data/processed/*.csv
data/processed/*.json
data/processed/*.parquet
data/training/*.csv
data/training/*.json
data/training/*.parquet
data/validation/*.csv
data/validation/*.json
data/validation/*.parquet

# Logs and monitoring
logs/*.log
logs/*.txt
wandb/
tensorboard_logs/
.wandb/

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# AWS
.aws/

# Secrets
.env.local
.env.production
config/secrets.yaml
*.key
*.pem
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
# Core ML dependencies
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.35.0
accelerate>=0.24.0
peft>=0.6.0
datasets>=2.14.0
bitsandbytes>=0.41.0

# Monitoring and logging
wandb>=0.16.0
tensorboard>=2.15.0

# API and web
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Development and testing
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.9.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.6.0

# Jupyter and analysis
jupyter>=1.0.0
jupyterlab>=4.0.0
ipywidgets>=8.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.1.0
numpy>=1.25.0
scipy>=1.11.0

# Additional utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.6.0
typer>=0.9.0
EOF

    log "✅ Configuration files created"
}

# Test GPU access
test_gpu() {
    log "🧪 Testing GPU access..."
    
    source $HOME/miniconda3/bin/activate recruitment-assistant
    
    python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("❌ CUDA not available!")
EOF

    log "✅ GPU test completed"
}

# Create learning documentation
create_learning_docs() {
    log "📚 Creating learning documentation..."
    
    cd /mnt/c/Users/$(whoami)/engineer-blog/ai-recruitment-assistant
    
    # Create deep learning concepts documentation
    cat > docs/learning-notes/01-deep-learning-concepts.md << 'EOF'
# Deep Learning Concepts for Fine-tuning

## Large Language Models (LLMs)

### What is Llama 3.1 8B?
- **Llama**: Large Language Model Meta AI developed by Meta
- **3.1**: Version with improved capabilities and context length
- **8B**: 8 billion parameters - optimal for single GPU fine-tuning

### Model Architecture
- **Transformer**: Attention-based neural network architecture
- **Parameters**: Learned weights that define the model behavior
- **Context Length**: Maximum input sequence length (128K tokens for Llama 3.1)

## Fine-tuning Techniques

### LoRA (Low-Rank Adaptation)
- **Purpose**: Parameter-efficient fine-tuning
- **How it works**: Adds small trainable matrices to frozen base model
- **Benefits**: 
  - Uses ~0.1% of original parameters
  - Reduces memory requirements by 3x
  - Faster training and inference

### QLoRA (Quantized LoRA)
- **Quantization**: Reduces precision from 32-bit to 4-bit
- **Memory saving**: Up to 75% reduction in GPU memory
- **Performance**: Minimal accuracy loss with significant efficiency gains

## GPU Memory Management

### Your RTX 4060 Setup
- **VRAM**: 8GB GDDR6
- **Compute Capability**: 8.9
- **Tensor Cores**: 4th Gen RT Cores for AI acceleration

### Memory Optimization
- **Model Loading**: Use 4-bit quantization
- **Batch Size**: Optimize for 8GB VRAM (~2-4 samples)
- **Gradient Accumulation**: Simulate larger batches
- **Gradient Checkpointing**: Trade compute for memory

## Training Process

### Data Preparation
1. **Collection**: Gather recruitment-specific conversations
2. **Cleaning**: Remove PII and standardize format
3. **Formatting**: Convert to instruction-following format
4. **Splitting**: Train/validation/test splits

### Training Loop
1. **Forward Pass**: Model predictions
2. **Loss Calculation**: Compare predictions to targets
3. **Backward Pass**: Calculate gradients
4. **Parameter Update**: Adjust weights using optimizer

### Monitoring
- **Loss**: Training and validation loss curves
- **Perplexity**: How "surprised" the model is by data
- **GPU Utilization**: Ensure efficient hardware usage
- **Memory Usage**: Monitor VRAM consumption

## Key Metrics

### Training Metrics
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Controls update step size
- **Gradient Norm**: Indicates training stability

### Quality Metrics
- **BLEU Score**: Text generation quality
- **Perplexity**: Model confidence in predictions
- **Human Evaluation**: Professional tone and accuracy
EOF

    # Create MLOps documentation
    cat > docs/learning-notes/02-mlops-concepts.md << 'EOF'
# MLOps Concepts and Implementation

## What is MLOps?

MLOps (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows. It encompasses the entire ML lifecycle from development to production deployment.

### Core Components

#### 1. Experiment Tracking
- **Purpose**: Track model versions, hyperparameters, and results
- **Tools**: Weights & Biases (W&B), TensorBoard, MLflow
- **Benefits**: Reproducibility, comparison, and collaboration

#### 2. Model Registry
- **Purpose**: Centralized storage for trained models
- **Features**: Versioning, metadata, staging, and production promotion
- **Implementation**: W&B Artifacts, MLflow Registry

#### 3. Continuous Integration/Continuous Deployment (CI/CD)
- **Purpose**: Automated testing, building, and deployment
- **Tools**: GitHub Actions, Jenkins, GitLab CI
- **Benefits**: Faster iterations, reduced errors, consistency

#### 4. Model Monitoring
- **Purpose**: Track model performance in production
- **Metrics**: Accuracy drift, data drift, latency, throughput
- **Tools**: CloudWatch, Prometheus, custom dashboards

## Our MLOps Pipeline

### Development Phase
1. **Data Versioning**: Track dataset changes with DVC
2. **Experiment Tracking**: Log all training runs with W&B
3. **Code Quality**: Linting, testing, and documentation
4. **Model Validation**: Automated testing of model outputs

### Deployment Phase
1. **Model Packaging**: Docker containers with dependencies
2. **Infrastructure**: AWS EC2, Lambda, or Bedrock deployment
3. **API Development**: FastAPI endpoints for inference
4. **Load Testing**: Ensure production readiness

### Production Phase
1. **Health Monitoring**: API uptime and response times
2. **Performance Tracking**: Model accuracy and drift detection
3. **Automated Retraining**: Trigger retraining on performance degradation
4. **Rollback Capability**: Quick reversion to previous model versions

## Tools and Technologies

### Experiment Tracking (W&B)
```python
import wandb

# Initialize experiment
wandb.init(
    project="recruitment-assistant",
    config={
        "learning_rate": 2e-4,
        "batch_size": 4,
        "epochs": 3
    }
)

# Log metrics during training
wandb.log({
    "train_loss": loss.item(),
    "learning_rate": lr,
    "epoch": epoch
})
```

### Model Versioning
- **Git**: Source code versioning
- **DVC**: Data and model versioning
- **W&B Artifacts**: Model artifact storage and lineage

### Infrastructure as Code
- **Docker**: Containerized deployments
- **AWS CloudFormation**: Infrastructure provisioning
- **Terraform**: Alternative IaC tool

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Model Tests**: Validation of model outputs
- **Performance Tests**: Latency and throughput testing

## Best Practices

### 1. Reproducibility
- Pin all dependency versions
- Use random seeds for deterministic results
- Document all environment configurations
- Version control everything (code, data, configs)

### 2. Monitoring and Alerting
- Set up alerts for model performance degradation
- Monitor resource utilization (CPU, GPU, memory)
- Track business metrics alongside technical metrics
- Implement automated health checks

### 3. Security
- Never commit secrets to version control
- Use environment variables for configuration
- Implement proper authentication and authorization
- Regular security scanning of dependencies

### 4. Documentation
- Document all design decisions
- Maintain up-to-date API documentation
- Create runbooks for common operations
- Keep learning notes for future reference
EOF

    log "✅ Learning documentation created"
}

# Setup aliases and shortcuts
setup_aliases() {
    log "⚡ Setting up helpful aliases..."
    
    cat >> ~/.bashrc << 'EOF'

# AI Recruitment Assistant Aliases
alias ara-activate='source $HOME/miniconda3/bin/activate recruitment-assistant'
alias ara-train='cd /mnt/c/Users/wesle_b0bdufu/engineer-blog/ai-recruitment-assistant && python src/training/train_recruitment_model.py'
alias ara-serve='cd /mnt/c/Users/wesle_b0bdufu/engineer-blog/ai-recruitment-assistant && python src/api/main.py'
alias ara-test='cd /mnt/c/Users/wesle_b0bdufu/engineer-blog/ai-recruitment-assistant && pytest tests/'
alias ara-notebook='cd /mnt/c/Users/wesle_b0bdufu/engineer-blog/ai-recruitment-assistant && jupyter lab'
alias ara-logs='cd /mnt/c/Users/wesle_b0bdufu/engineer-blog/ai-recruitment-assistant/logs && tail -f training.log'
alias ara-gpu='watch -n 1 nvidia-smi'

# General development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias gpu='nvidia-smi'
alias htop='htop'
EOF

    source ~/.bashrc
    
    log "✅ Aliases configured"
}

# Final setup and validation
final_setup() {
    log "🏁 Final setup and validation..."
    
    # Create empty __init__.py files
    cd /mnt/c/Users/$(whoami)/engineer-blog/ai-recruitment-assistant
    
    find src -type d -exec touch {}/__init__.py \;
    
    # Set permissions
    chmod +x scripts/*.sh
    
    # Create initial log file
    mkdir -p logs
    touch logs/setup.log
    echo "$(date): WSL environment setup completed" >> logs/setup.log
    
    log "✅ Final setup completed"
}

# Main execution
main() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           AI Recruitment Assistant Setup                     ║"
    echo "║        Deep Learning Environment for RTX 4060               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_wsl
    show_system_info
    
    log "🚀 Starting comprehensive setup..."
    
    update_system
    install_cuda
    install_miniconda
    setup_python_environment
    install_pytorch
    install_dev_tools
    setup_ssh
    create_config_files
    test_gpu
    create_learning_docs
    setup_aliases
    final_setup
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete! 🎉                       ║"
    echo "║                                                              ║"
    echo "║  Your deep learning environment is ready for fine-tuning    ║"
    echo "║  Llama 3.1 8B on your RTX 4060!                            ║"
    echo "║                                                              ║"
    echo "║  Next Steps:                                                 ║"
    echo "║  1. Restart your terminal: source ~/.bashrc                 ║"
    echo "║  2. Activate environment: ara-activate                      ║"
    echo "║  3. Test GPU: python -c 'import torch; print(torch.cuda.is_available())' ║"
    echo "║  4. Start training: ara-train                               ║"
    echo "║                                                              ║"
    echo "║  Happy learning! 🚀                                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log_info "Setup completed successfully! Check logs/setup.log for details"
}

# Run main function
main "$@"
