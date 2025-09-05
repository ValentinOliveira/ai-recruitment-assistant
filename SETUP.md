# AI Recruitment Assistant - Setup Instructions

## 🐍 Python Installation Issue Detected

I notice that Python is not properly installed on your system. The current Python installation seems to be pointing to the Microsoft Store version which isn't working correctly.

## 📋 Step-by-Step Setup Instructions:

### 1. Install Python 3.10+ (Recommended: Python 3.10.x)
- **Download:** Go to [python.org](https://www.python.org/downloads/windows/)
- **Version:** Install Python 3.10.x (3.10.11 recommended for best compatibility with PyTorch)
- **Important:** During installation, check "Add Python to PATH"
- **Important:** Check "Install pip"

### 2. Verify Installation
After installation, open a new PowerShell/Command Prompt and run:
```cmd
python --version
pip --version
```

### 3. Install Dependencies
Once Python is properly installed, run:
```cmd
pip install -r requirements.txt
```

### 4. GPU Setup (For RTX 4060)
After basic installation, install PyTorch with CUDA support:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Verify GPU Setup
Run this Python script to verify GPU is detected:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🚨 Alternative: Use Conda/Miniconda

If you prefer Conda (recommended for ML projects):

### 1. Install Miniconda
- Download from: https://docs.conda.io/miniconda.html
- Install with default settings

### 2. Create Environment
```cmd
conda create -n ai-recruit python=3.10
conda activate ai-recruit
```

### 3. Install Dependencies
```cmd
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets peft accelerate bitsandbytes wandb
```

## 📦 What Each Package Does:

- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library (for Llama models)
- **datasets**: Data loading and processing
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **accelerate**: Distributed training support
- **bitsandbytes**: 4-bit quantization for memory efficiency
- **wandb**: Weights & Biases for experiment tracking

## 🔧 Next Steps After Installation:

1. **Test the training script:**
   ```cmd
   python src/training/train_recruitment_model.py --help
   ```

2. **Create sample training data:**
   ```cmd
   python src/data/create_sample_data.py
   ```

3. **Run training (when ready):**
   ```cmd
   python src/training/train_recruitment_model.py
   ```

## 💡 Tips:

- **Virtual Environment**: Always use a virtual environment for ML projects
- **CUDA Drivers**: Make sure you have the latest NVIDIA drivers installed
- **Memory**: The training script is optimized for 8GB VRAM (RTX 4060)
- **Monitoring**: Sign up for a free Weights & Biases account for training monitoring

## 🆘 Common Issues:

1. **"CUDA not available"**: Install PyTorch with CUDA support
2. **Out of memory**: Reduce batch size in training config
3. **Import errors**: Make sure all packages are installed in the same environment
4. **Permission errors**: Run PowerShell as Administrator if needed

Once you have Python properly installed, we can proceed with the dependency installation and training setup!
