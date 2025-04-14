#!/usr/bin/env python3
"""
Basic setup test for CoT-UQ environment.
Tests imports and system information to verify the container setup.
"""

import os
import sys
import platform
import torch
import transformers
import nltk

def check_environment():
    """Print system and library information to verify the setup"""
    print("\n" + "=" * 50)
    print("CoT-UQ Environment Test")
    print("=" * 50)
    
    # System information
    print("\nSystem Information:")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Current directory: {os.getcwd()}")
    
    # PyTorch and CUDA
    print("\nPyTorch & CUDA Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Transformers and other libraries
    print("\nLibrary Information:")
    print(f"Transformers version: {transformers.__version__}")
    print(f"NLTK version: {nltk.__version__}")
    
    # Check key project files
    print("\nProject Files:")
    key_files = [
        "README.md", 
        "requirements.txt",
        "run_llama_pipeline.sh",
        "config.py", 
        "src/model/llama2_predict.py"
    ]
    
    for file in key_files:
        status = "✓ Found" if os.path.exists(file) else "✗ Not found"
        print(f"{file}: {status}")
    
    print("\n" + "=" * 50)
    print("Environment test completed")
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
    # Return success exit code
    sys.exit(0)