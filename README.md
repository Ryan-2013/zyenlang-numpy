# numpy for ZyenLang

A small, dependency-free numeric array package for ZyenLang 0.3. It uses an
ABI v3 ARC handle backed by contiguous `f64` memory.

```powershell
zy install numpy
```

```zy
import numpy as np
import std::io as io

fn main() i32 throws Error {
    let left = np::from_list([1.0, 2.0, 3.0, 4.0], 2, 2)
    let right = np::identity(2)
    let result = left.matmul(right).scale(2.0)
    io::print((str)result.get(1, 1))
    return 0
}
```

## API

- `zeros(rows, columns)`, `full(rows, columns, value)`, `identity(size)`
- `from_list(values, rows, columns)`
- `Array.rows()`, `columns()`, `length()`, `get()`, `set()`
- `Array.sum()`, `mean()`, `add()`, `scale()`, `matmul()`, `to_list()`

Dimensions and indexes are checked. Array storage is released when the final
`Array` reference leaves scope. The package is intentionally not a full NumPy
clone; it is the stable numeric foundation on which larger packages can grow.
