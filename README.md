# TinyTensor

A small 2D `Float32` tensor library written in [Mojo](https://docs.modular.com/mojo). It explores two things that matter in modern AI frameworks: **API ergonomics** and **CPU performance** (SIMD, parallelism, tiling, op fusion).

## Features

- `Tensor` struct with explicit shape and strides, heap-backed, with deep copy and move semantics
- Operators: `a + b`, `a * b`, `a @ b`, plus `.relu()`
- Two matmul kernels:
  - `matmul_naive`: textbook triple loop, used as the baseline and for correctness checks
  - `matmul`: SIMD vectorized, parallelized over rows, tiled over the inner dimension
- `matmul_relu`: fused matmul + ReLU, applied while each output row is still in cache
- Readable shape errors: `matmul: (2, 3) @ (4, 5) incompatible (inner dims 3 != 4)`
- Built-in benchmark against NumPy through Python interop

## Quick start

```bash
pip install numpy
mojo tinytensor.mojo
```

> Mojo runs on Linux and macOS. On Windows, use WSL2.

Running the file prints an error-handling demo, correctness checks (fast vs naive, fused vs unfused), and a 512x512 benchmark with timings, GFLOP/s, and speedups for naive, fast, NumPy, and fused vs unfused.

## Usage

```mojo
var a = Tensor.arange(128, 256)
var b = Tensor.arange(256, 64)

var c = a @ b                # fast matmul
var d = matmul_relu(a, b)    # fused matmul + relu
var e = c + c                # elementwise add

try:
    _ = Tensor.zeros(2, 3) @ Tensor.zeros(4, 5)
except err:
    print(err)               # shape mismatch with a readable message
```

## How the fast matmul works

1. **Loop order i-k-j.** The inner loop walks rows of B and C contiguously, so it maps cleanly onto SIMD loads and stores.
2. **`vectorize`** processes several `Float32` values per instruction along the column dimension.
3. **`parallelize`** splits output rows across CPU cores.
4. **Tiling over k** keeps a block of B hot in cache while it is reused.
5. **Fusion** (`matmul_relu`) applies ReLU to each output row right after it is computed, instead of making a second pass over the full matrix.

## Design decisions

| Decision | Why |
|---|---|
| `Tensor.zeros(r, c)` static constructors | Intent is clear at the call site, similar to `torch.zeros`. |
| `a @ b` and `a + b` operators | Reads like math and matches NumPy/PyTorch habits. |
| Fallible ops are `raises` with shapes in the message | Shape bugs are the most common tensor error, so the message should name the shapes that clashed. |
| Strides stored explicitly | Leaves room for views (transpose, slicing) without copying, as in PyTorch. |
| `fuse_relu` as a compile-time parameter | Fusion costs nothing at runtime and avoids maintaining two copies of the kernel. |
| `a @ b` defaults to the fast kernel | The obvious call should also be the fast one. |

## Project layout

```
tinytensor.mojo   # Tensor, kernels, correctness checks, benchmark
README.md
```

## Roadmap

- Views and transpose using strides (no copy)
- N-dimensional shapes and broadcasting
- Compile-time shapes as type parameters, so mismatches become compile errors
- Compare the API against PyTorch and document where the tradeoffs differ
