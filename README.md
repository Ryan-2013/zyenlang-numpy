# numpy for ZyenLang 0.3.0

A small, dependency-free N-dimensional numeric array package for ZyenLang 0.3.
It uses an ABI v3 ARC handle backed by contiguous row-major `f64` memory.

```powershell
zy install numpy
```

```zy
import numpy as np
import std::io as io

fn main() i32 throws Error {
    let tensor = np::generate([2, 3, 4], 1.5)
    tensor.set_at([1, 2, 3], 9.0)

    let result = tensor
        .reshape_shape([4, 6])
        .scale(2.0)
    let shape = result.shape()
    io::print(f"rank={result.rank()}, rows={shape[0]}, columns={shape[1]}")
    io::print((str)result.get(3, 5))
    return 0
}
```

## API

- ND constructors: `zeros_shape`, `full_shape`, `generate`, `from_shape`
- 2D constructors: `zeros`, `full`, `identity`, `from_list`, `from_nested`
- Shape and indexing: `rank`, `shape`, `length`, `get_at`, `set_at`
- 2D indexing: `rows`, `columns`, `get`, `set`
- Layout: `copy`, `reshape_shape`, `reshape`, `transpose`
- Arithmetic: `add`, `subtract`, `multiply`, `scale`, `matmul`
- Reductions: `sum`, `mean`, `minimum`, `maximum`
- Conversion: `to_list`, `to_nested`

`generate(shape, value)` is the strongly typed replacement for a Python-style
`gen_list(shape)`: the runtime shape may have any rank from 0 through 16 while
the returned type remains `Array`. `from_nested` is a convenient rank-2 bridge
that uses `LIST_SHAPE__()` and rejects ragged input. Dimensions, indexes,
reshape element counts, and binary-operation shapes are checked.
Operations that return an `Array` allocate an independent result; native
storage is released when the final ARC reference leaves scope.

`transpose`, `matmul`, and `to_nested` currently require rank 2. Element-wise
arithmetic, scaling, reductions, flat conversion, copying, dynamic indexing,
and reshaping work at every rank. This is not a complete Python NumPy clone;
the smaller surface keeps its ABI predictable for game engines and C interop.
