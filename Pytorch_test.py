import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test tensor operations
x = torch.rand(2, 3)
print(f"\nRandom tensor:\n{x}")
print(f"Tensor shape: {x.shape}")

# Test CUDA if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    print(f"\nTensor moved to GPU: {x_gpu.device}")