#!/usr/bin/env python3
"""
GPU Setup Verification Script
=============================

Quick script to verify that your GPU setup is working correctly
for the AI Recruitment Assistant fine-tuning.
"""

import sys

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Python Version Check:")
    print(f"   Version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_pytorch():
    """Check PyTorch installation and GPU support."""
    print("\n🔥 PyTorch Check:")
    
    try:
        import torch
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            
            # Check each GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
                print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Test tensor operations
            print("\n   🧪 GPU Test:")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print(f"   ✅ GPU tensor operations work! Result shape: {z.shape}")
            
            return True
        else:
            print("   ⚠️  CUDA not available - will use CPU (very slow)")
            return False
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_transformers():
    """Check Hugging Face transformers."""
    print("\n🤗 Transformers Check:")
    
    try:
        import transformers
        print(f"   Version: {transformers.__version__}")
        print("   ✅ Transformers installed")
        return True
    except ImportError:
        print("   ❌ Transformers not installed")
        return False

def check_other_packages():
    """Check other required packages."""
    print("\n📦 Other Packages Check:")
    
    packages = {
        'datasets': 'Datasets library',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'accelerate': 'Training acceleration',
        'bitsandbytes': '4-bit quantization',
        'wandb': 'Weights & Biases',
    }
    
    all_good = True
    
    for package, description in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {package} ({description}): {version}")
        except ImportError:
            print(f"   ❌ {package} ({description}): Not installed")
            all_good = False
    
    return all_good

def check_memory():
    """Check system and GPU memory."""
    print("\n💾 Memory Check:")
    
    try:
        import psutil
        
        # System RAM
        ram = psutil.virtual_memory()
        print(f"   System RAM: {ram.total / 1e9:.1f} GB total, {ram.available / 1e9:.1f} GB available")
        
        # GPU Memory
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i} VRAM: {total:.1f} GB total, {allocated:.1f} GB allocated, {cached:.1f} GB cached")
        
        # Recommendations
        if ram.total < 16e9:
            print("   ⚠️  Less than 16GB RAM - consider closing other applications during training")
        else:
            print("   ✅ Sufficient system RAM")
            
        return True
        
    except ImportError:
        print("   ❌ psutil not installed - can't check memory")
        return False

def main():
    """Main verification function."""
    print("🚀 AI Recruitment Assistant - GPU Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_pytorch),
        ("Transformers", check_transformers),
        ("Other Packages", check_other_packages),
        ("Memory", check_memory),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to train the AI Recruitment Assistant!")
        print("\nNext steps:")
        print("1. Create training data: python src/data/create_sample_data.py")
        print("2. Start training: python src/training/train_recruitment_model.py")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()
