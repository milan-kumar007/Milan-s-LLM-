import sys

def check_environment():
    print("=" * 60)
    print(" CUDA & Environment Verification for Milan's LLM Training")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")

    # Check PyTorch & CUDA
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        print(f"CUDA Available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"Device Count      : {torch.cuda.device_count()}")
            device_name = torch.cuda.get_device_name(0)
            device_capability = torch.cuda.get_device_capability(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Device 0 Name     : {device_name}")
            print(f"Compute Capability: {device_capability[0]}.{device_capability[1]}")
            print(f"Total VRAM        : {total_vram_gb:.2f} GB")

            # Tensor test
            print("\n[Running GPU Tensor Computation Test...]")
            x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
            y = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
            print(f"Matrix multiplication (2048x2048) on GPU successful!")
            print(f"VRAM Allocated: {allocated_mb:.2f} MB | VRAM Reserved: {reserved_mb:.2f} MB")
            del x, y, z
            torch.cuda.empty_cache()
            print("GPU verification passed!")
        else:
            print("WARNING: CUDA is NOT available to PyTorch. Check drivers and PyTorch build.")
    except ImportError as e:
        print(f"PyTorch Import Error: {e}")

    print("\n" + "-" * 60)
    print(" Library Imports Verification")
    print("-" * 60)

    libraries = [
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("tokenizers", "Tokenizers"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "TQDM"),
    ]

    for mod_name, display_name in libraries:
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {display_name:15}: OK (v{version})")
        except ImportError as e:
            print(f"  {display_name:15}: FAILED ({e})")

    print("=" * 60)

if __name__ == "__main__":
    check_environment()
