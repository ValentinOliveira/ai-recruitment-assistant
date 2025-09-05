#!/usr/bin/env python3
"""Test GPU availability and CUDA setup."""

import torch
import sys

def test_gpu():
    print("🚀 Testing GPU and CUDA setup...")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Test tensor operations on GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"\n✅ GPU tensor operations successful!")
            print(f"   Test tensor shape: {z.shape}")
            print(f"   Test tensor device: {z.device}")
            
            # Test memory usage
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"   GPU memory allocated: {allocated:.2f} GB")
            print(f"   GPU memory cached: {cached:.2f} GB")
            
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
            
    else:
        print("❌ CUDA not available!")
        return False
    
    print("\n🎉 Deep learning environment ready for fine-tuning!")
    return True

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)
