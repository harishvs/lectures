import torch
import torch.utils.cpp_extension
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    # Use file-based compilation instead of inline
    module = torch.utils.cpp_extension.load(
        name='square',
        sources=['square_wrapper.cpp', 'square_kernel.cu'],
        verbose=True
    )
else:
    print("CUDA not available, falling back to CPU implementation")
    module = None

def square(input_tensor):
    if device.type == 'cuda' and module is not None:
        return module.square_cuda(input_tensor)
    else:
        # CPU fallback
        return input_tensor * input_tensor

# Example usage
input_tensor = torch.randn(100, device=device)
print(f"Using device: {device}")
print(f"Input tensor shape: {input_tensor.shape}")

output_tensor = square(input_tensor)
print(f"Output tensor shape: {output_tensor.shape}")

# Verify correctness
expected = input_tensor * input_tensor
print(f"Results match: {torch.allclose(output_tensor, expected)}")
print(f"Max difference: {torch.max(torch.abs(output_tensor - expected)).item()}")
