"""TinyTensor: a small 2D Float32 tensor library in Mojo.

Run:  mojo tinytensor.mojo
(Needs NumPy in the same environment for the benchmark: pip install numpy)

NOTE: Mojo syntax moves fast. This targets recent Mojo (comptime, alloc,
`deinit`). If the compiler complains, check the changelog for renames of:
alloc / UnsafePointer, comptime (was alias), `comptime if` (was @parameter if).
"""

from algorithm import parallelize, vectorize
from memory import UnsafePointer, alloc
from python import Python
from sys import simd_width_of
from time import perf_counter_ns

comptime SIMD_W = simd_width_of[DType.float32]()
comptime TILE_K = 64


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------
struct Tensor(Copyable, Movable):
    """Row-major 2D Float32 tensor with explicit shape and strides."""

    var rows: Int
    var cols: Int
    var stride0: Int  # elements to skip to move one row
    var stride1: Int  # elements to skip to move one column
    var data: UnsafePointer[Float32]

    # -- construction -------------------------------------------------------
    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.stride0 = cols
        self.stride1 = 1
        self.data = alloc[Float32](rows * cols)
        for i in range(rows * cols):
            self.data[i] = 0

    @staticmethod
    def zeros(rows: Int, cols: Int) -> Self:
        return Self(rows, cols)

    @staticmethod
    def full(rows: Int, cols: Int, value: Float32) -> Self:
        var t = Self(rows, cols)
        for i in range(rows * cols):
            t.data[i] = value
        return t^

    @staticmethod
    def arange(rows: Int, cols: Int) -> Self:
        """Deterministic test data: small repeating values."""
        var t = Self(rows, cols)
        for i in range(rows):
            for j in range(cols):
                t[i, j] = Float32((i * 7 + j * 3) % 10) / 10.0
        return t^

    # -- lifecycle (deep copy, cheap move, free on destroy) -----------------
    def __copyinit__(out self, existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.stride0 = existing.stride0
        self.stride1 = existing.stride1
        var n = existing.rows * existing.cols
        self.data = alloc[Float32](n)
        for i in range(n):
            self.data[i] = existing.data[i]

    def __moveinit__(out self, deinit existing: Self):
        self.rows = existing.rows
        self.cols = existing.cols
        self.stride0 = existing.stride0
        self.stride1 = existing.stride1
        self.data = existing.data

    def __del__(deinit self):
        self.data.free()

    # -- indexing -----------------------------------------------------------
    def __getitem__(self, i: Int, j: Int) -> Float32:
        return self.data[i * self.stride0 + j * self.stride1]

    def __setitem__(mut self, i: Int, j: Int, value: Float32):
        self.data[i * self.stride0 + j * self.stride1] = value

    # -- helpers ------------------------------------------------------------
    def shape_str(self) -> String:
        return "(" + String(self.rows) + ", " + String(self.cols) + ")"

    def _check_same_shape(self, other: Self, op: String) raises:
        if self.rows != other.rows or self.cols != other.cols:
            raise Error(
                op + ": shape mismatch " + self.shape_str()
                + " vs " + other.shape_str()
            )

    # -- elementwise ops (operators) ----------------------------------------
    def __add__(self, other: Self) raises -> Self:
        self._check_same_shape(other, "add")
        var out = Self(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            out.data[i] = self.data[i] + other.data[i]
        return out^

    def __mul__(self, other: Self) raises -> Self:
        self._check_same_shape(other, "mul")
        var out = Self(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            out.data[i] = self.data[i] * other.data[i]
        return out^

    def relu(self) -> Self:
        var out = Self(self.rows, self.cols)
        for i in range(self.rows * self.cols):
            var v = self.data[i]
            out.data[i] = v if v > 0 else 0
        return out^

    # `a @ b` uses the fast kernel by default.
    def __matmul__(self, other: Self) raises -> Self:
        return matmul(self, other)


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------
def _check_matmul(a: Tensor, b: Tensor) raises:
    if a.cols != b.rows:
        raise Error(
            "matmul: " + a.shape_str() + " @ " + b.shape_str()
            + " incompatible (inner dims " + String(a.cols)
            + " != " + String(b.rows) + ")"
        )


def matmul_naive(a: Tensor, b: Tensor) raises -> Tensor:
    """Textbook triple loop. The baseline."""
    _check_matmul(a, b)
    var c = Tensor(a.rows, b.cols)
    for i in range(a.rows):
        for j in range(b.cols):
            var acc: Float32 = 0
            for k in range(a.cols):
                acc += a[i, k] * b[k, j]
            c[i, j] = acc
    return c^


def _matmul_fast[fuse_relu: Bool](a: Tensor, b: Tensor) raises -> Tensor:
    """i-k-j loop order: B and C are walked contiguously, so the inner loop
    vectorizes. Rows run in parallel; k is tiled for cache reuse. Optionally
    applies ReLU while the C row is still hot in cache (fusion)."""
    _check_matmul(a, b)
    var m = a.rows
    var kdim = a.cols
    var n = b.cols
    var c = Tensor(m, n)

    var ap = a.data
    var bp = b.data
    var cp = c.data

    @parameter
    def do_row(i: Int):
        var crow = cp + i * n
        for k0 in range(0, kdim, TILE_K):
            var k1 = min(k0 + TILE_K, kdim)
            for kk in range(k0, k1):
                var aik = ap[i * kdim + kk]
                var brow = bp + kk * n

                @parameter
                def fma_col[w: Int](j: Int):
                    var av = SIMD[DType.float32, w](aik)
                    crow.store(j, crow.load[width=w](j) + av * brow.load[width=w](j))

                vectorize[fma_col, SIMD_W](n)

        comptime if fuse_relu:

            @parameter
            def relu_col[w: Int](j: Int):
                var zero = SIMD[DType.float32, w](0)
                crow.store(j, max(crow.load[width=w](j), zero))

            vectorize[relu_col, SIMD_W](n)

    parallelize[do_row](m)
    return c^


def matmul(a: Tensor, b: Tensor) raises -> Tensor:
    return _matmul_fast[False](a, b)


def matmul_relu(a: Tensor, b: Tensor) raises -> Tensor:
    """Fused matmul + relu: one pass over C instead of two."""
    return _matmul_fast[True](a, b)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def max_abs_diff(a: Tensor, b: Tensor) raises -> Float32:
    a._check_same_shape(b, "max_abs_diff")
    var worst: Float32 = 0
    for i in range(a.rows * a.cols):
        var d = abs(a.data[i] - b.data[i])
        if d > worst:
            worst = d
    return worst


def ms_since(start: UInt) -> Float64:
    return Float64(perf_counter_ns() - start) / 1e6


def gflops(n: Int, ms: Float64) -> Float64:
    return 2.0 * Float64(n) ** 3 / (ms / 1000.0) / 1e9


def main() raises:
    # 1) Error ergonomics demo
    try:
        var bad = Tensor.zeros(2, 3) @ Tensor.zeros(4, 5)
        _ = bad^
    except e:
        print("expected error ->", e)

    # 2) Correctness: fast vs naive, fused vs separate
    var a = Tensor.arange(128, 128)
    var b = Tensor.arange(128, 128)
    print("fast vs naive max diff:", max_abs_diff(matmul(a, b), matmul_naive(a, b)))
    print(
        "fused vs unfused max diff:",
        max_abs_diff(matmul_relu(a, b), matmul(a, b).relu()),
    )

    # 3) Benchmark
    var n = 512
    var reps = 5
    var ta = Tensor.arange(n, n)
    var tb = Tensor.arange(n, n)

    var t0 = perf_counter_ns()
    for _ in range(reps):
        _ = matmul_naive(ta, tb)
    var naive_ms = ms_since(t0) / Float64(reps)

    t0 = perf_counter_ns()
    for _ in range(reps):
        _ = matmul(ta, tb)
    var fast_ms = ms_since(t0) / Float64(reps)

    t0 = perf_counter_ns()
    for _ in range(reps):
        _ = matmul(ta, tb).relu()
    var unfused_ms = ms_since(t0) / Float64(reps)

    t0 = perf_counter_ns()
    for _ in range(reps):
        _ = matmul_relu(ta, tb)
    var fused_ms = ms_since(t0) / Float64(reps)

    var np = Python.import_module("numpy")
    var na = np.random.rand(n, n).astype(np.float32)
    var nb = np.random.rand(n, n).astype(np.float32)
    t0 = perf_counter_ns()
    for _ in range(reps):
        _ = np.matmul(na, nb)
    var numpy_ms = ms_since(t0) / Float64(reps)

    print("\n--- ", n, "x", n, " matmul, avg of", reps, "runs ---")
    print("naive         :", naive_ms, "ms", gflops(n, naive_ms), "GFLOP/s")
    print("fast          :", fast_ms, "ms", gflops(n, fast_ms), "GFLOP/s")
    print("numpy         :", numpy_ms, "ms", gflops(n, numpy_ms), "GFLOP/s")
    print("matmul + relu :", unfused_ms, "ms (separate)")
    print("fused         :", fused_ms, "ms")
    print("speedup fast/naive :", naive_ms / fast_ms, "x")
    print("speedup fused/unfused :", unfused_ms / fused_ms, "x")
