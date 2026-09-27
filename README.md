# numpy for ZyenLang 0.2.0

A small, dependency-free numeric array package for ZyenLang 0.3. It uses an
ABI v3 ARC handle backed by contiguous `f64` memory.

```powershell
zy install numpy
```

```zy
import numpy as np
import std::io as io

fn main() i32 throws Error {
    let left = np::from_nested([[1.0, 2.0], [3.0, 4.0]])
    let right = np::identity(2)
    let result = left.matmul(right).transpose().scale(2.0)
    let shape = result.shape()
    io::print((str)result.get(1, 1))
    return 0
}
```

## API

- Constructors: `zeros`, `full`, `identity`, `from_list`, `from_nested`
- Shape and indexing: `shape`, `rows`, `columns`, `length`, `get`, `set`
- Layout: `copy`, `reshape`, `transpose`
- Arithmetic: `add`, `subtract`, `multiply`, `scale`, `matmul`
- Reductions: `sum`, `mean`, `minimum`, `maximum`
- Conversion: `to_list`, `to_nested`

`from_nested` uses `LIST_SHAPE__()` and rejects ragged input. Dimensions,
indexes, reshape element counts, and binary-operation shapes are checked.
Operations that return an `Array` allocate an independent result; native
storage is released when the final ARC reference leaves scope.

The package is intentionally a two-dimensional contiguous `f64` array library,
not a complete Python NumPy clone. This keeps its ABI small and predictable for
game engines and C interop.
