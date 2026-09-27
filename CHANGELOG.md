# Changelog

## 0.3.0 - 2026-09-27

- Reworked native Array storage to retain dynamic rank and shape metadata.
- Added `zeros_shape`, `full_shape`, `generate`, and `from_shape` constructors.
- Added `rank`, dynamic `shape`, `get_at`, `set_at`, and `reshape_shape`.
- Preserved the complete 2D convenience API for matrices.
- Added rank-0 scalar, zero-dimension, 3D indexing, and ND reshape tests.

## 0.2.0 - 2026-09-27

- Added `from_nested()` with rectangular validation through `LIST_SHAPE__()`.
- Added `Array.shape()`, `to_nested()`, `copy()`, `reshape()`, and `transpose()`.
- Added element-wise subtraction and multiplication.
- Added minimum and maximum reductions with checked empty-array errors.
- Expanded success, ownership-isolation, and error-path tests.

## 0.1.0 - 2026-09-27

- Initial contiguous two-dimensional `f64` Array package using ABI v3 handles.
