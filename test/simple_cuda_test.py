# test_cuda_simple.py
import os
os.environ['NUMBA_CUDA_DEFAULT_PTX_VERSION'] = '8.4'

from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < out.size:
        out[idx] = x[idx] + y[idx]

# Test
x = np.ones(10, dtype=np.float32)
y = np.ones(10, dtype=np.float32)
out = np.zeros(10, dtype=np.float32)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.to_device(out)

add_kernel[1, 10](d_x, d_y, d_out)

result = d_out.copy_to_host()
print(f"Result: {result}")
print("✓ CUDA test passed" if np.allclose(result, 2.0) else "✗ CUDA test failed")