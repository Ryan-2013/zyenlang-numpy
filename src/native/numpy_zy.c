#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zyenlang_c_abi.h"

#define ZY_NUMPY_TYPE "zyenlang.numpy.Array"
#define ZY_NUMPY_MAX_ELEMENTS ((size_t)100000000)
#define ZY_NUMPY_MAX_RANK ((size_t)16)

typedef struct ZyNumpyArray {
    size_t rank;
    size_t length;
    uintptr_t* shape;
    double* values;
} ZyNumpyArray;

static _Thread_local char zy_numpy_error[256] = "numpy operation failed";

static void zy_numpy_set_error(const char* message) {
    snprintf(zy_numpy_error, sizeof(zy_numpy_error), "%s", message);
}

ZL_String zy_numpy_last_error(void) {
    return zl_string_borrow(zy_numpy_error);
}

void zy_numpy_array_drop(void* raw) {
    ZyNumpyArray* array = (ZyNumpyArray*)raw;
    if (array == NULL) return;
    free(array->shape);
    free(array->values);
    free(array);
}

static int zy_numpy_element_count(const uintptr_t* shape, size_t rank, size_t* result) {
    if (rank > ZY_NUMPY_MAX_RANK) {
        zy_numpy_set_error("numpy Array rank cannot exceed 16");
        return 0;
    }
    size_t count = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        size_t dimension = (size_t)shape[axis];
        if (dimension > ZY_NUMPY_MAX_ELEMENTS) {
            zy_numpy_set_error("numpy Array is too large");
            return 0;
        }
        if (dimension != 0 && count > ZY_NUMPY_MAX_ELEMENTS / dimension) {
            zy_numpy_set_error("numpy Array is too large");
            return 0;
        }
        count *= dimension;
    }
    if (count > ZY_NUMPY_MAX_ELEMENTS || count > SIZE_MAX / sizeof(double)) {
        zy_numpy_set_error("numpy Array is too large");
        return 0;
    }
    *result = count;
    return 1;
}

static int zy_numpy_read_shape(
    ZL_Slice shape,
    const uintptr_t** dimensions,
    size_t* rank,
    size_t* count
) {
    if (shape.element_size != sizeof(uintptr_t) || (shape.len > 0 && shape.data == NULL)) {
        zy_numpy_set_error("numpy shape must be a List<usize>");
        return 0;
    }
    const uintptr_t* items = (const uintptr_t*)shape.data;
    if (!zy_numpy_element_count(items, shape.len, count)) return 0;
    *dimensions = items;
    *rank = shape.len;
    return 1;
}

static ZyNumpyArray* zy_numpy_allocate_shape(const uintptr_t* shape, size_t rank) {
    size_t count = 0;
    if (!zy_numpy_element_count(shape, rank, &count)) return NULL;
    ZyNumpyArray* array = (ZyNumpyArray*)calloc(1, sizeof(ZyNumpyArray));
    if (array == NULL) {
        zy_numpy_set_error("cannot allocate numpy Array metadata");
        return NULL;
    }
    if (rank > 0) {
        array->shape = (uintptr_t*)calloc(rank, sizeof(uintptr_t));
        if (array->shape == NULL) {
            free(array);
            zy_numpy_set_error("cannot allocate numpy Array shape");
            return NULL;
        }
        memcpy(array->shape, shape, rank * sizeof(uintptr_t));
    }
    if (count > 0) {
        array->values = (double*)calloc(count, sizeof(double));
        if (array->values == NULL) {
            free(array->shape);
            free(array);
            zy_numpy_set_error("cannot allocate numpy Array data");
            return NULL;
        }
    }
    array->rank = rank;
    array->length = count;
    return array;
}

static ZyNumpyArray* zy_numpy_allocate_2d(int64_t rows, int64_t columns) {
    if (rows < 0 || columns < 0) {
        zy_numpy_set_error("numpy dimensions cannot be negative");
        return NULL;
    }
    uintptr_t shape[2] = {(uintptr_t)rows, (uintptr_t)columns};
    return zy_numpy_allocate_shape(shape, 2);
}

static ZyNumpyArray* zy_numpy_allocate_slice(ZL_Slice shape) {
    const uintptr_t* dimensions = NULL;
    size_t rank = 0;
    size_t count = 0;
    if (!zy_numpy_read_shape(shape, &dimensions, &rank, &count)) return NULL;
    (void)count;
    return zy_numpy_allocate_shape(dimensions, rank);
}

static ZL_Handle zy_numpy_wrap(ZyNumpyArray* array) {
    if (array == NULL) return (ZL_Handle){0};
    return zl_handle_adopt(array, ZY_NUMPY_TYPE, zy_numpy_array_drop);
}

static ZyNumpyArray* zy_numpy_data(ZL_Handle handle) {
    return (ZyNumpyArray*)zl_handle_data(handle, ZY_NUMPY_TYPE);
}

static int zy_numpy_same_shape(const ZyNumpyArray* left, const ZyNumpyArray* right) {
    if (left->rank != right->rank) return 0;
    for (size_t axis = 0; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) return 0;
    }
    return 1;
}

static int zy_numpy_offset(const ZyNumpyArray* array, ZL_Slice indices, size_t* result) {
    if (indices.element_size != sizeof(uintptr_t) || indices.len != array->rank ||
        (indices.len > 0 && indices.data == NULL)) {
        zy_numpy_set_error("numpy index rank does not match Array rank");
        return 0;
    }
    const uintptr_t* items = (const uintptr_t*)indices.data;
    size_t offset = 0;
    for (size_t axis = 0; axis < array->rank; ++axis) {
        if (items[axis] >= array->shape[axis]) {
            zy_numpy_set_error("numpy Array index out of range");
            return 0;
        }
        offset = offset * (size_t)array->shape[axis] + (size_t)items[axis];
    }
    *result = offset;
    return 1;
}

ZL_Handle zy_numpy_zeros(int64_t rows, int64_t columns) {
    return zy_numpy_wrap(zy_numpy_allocate_2d(rows, columns));
}

ZL_Handle zy_numpy_zeros_shape(ZL_Slice shape) {
    return zy_numpy_wrap(zy_numpy_allocate_slice(shape));
}

ZL_Handle zy_numpy_full(int64_t rows, int64_t columns, double value) {
    ZyNumpyArray* array = zy_numpy_allocate_2d(rows, columns);
    if (array == NULL) return (ZL_Handle){0};
    for (size_t index = 0; index < array->length; ++index) array->values[index] = value;
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_full_shape(ZL_Slice shape, double value) {
    ZyNumpyArray* array = zy_numpy_allocate_slice(shape);
    if (array == NULL) return (ZL_Handle){0};
    for (size_t index = 0; index < array->length; ++index) array->values[index] = value;
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_from_slice(int64_t rows, int64_t columns, ZL_Slice values) {
    ZyNumpyArray* array = zy_numpy_allocate_2d(rows, columns);
    if (array == NULL) return (ZL_Handle){0};
    if (values.element_size != sizeof(double) || values.len != array->length ||
        (array->length > 0 && values.data == NULL)) {
        zy_numpy_array_drop(array);
        zy_numpy_set_error("List length does not match numpy Array shape");
        return (ZL_Handle){0};
    }
    if (array->length > 0) memcpy(array->values, values.data, array->length * sizeof(double));
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_from_shape(ZL_Slice shape, ZL_Slice values) {
    ZyNumpyArray* array = zy_numpy_allocate_slice(shape);
    if (array == NULL) return (ZL_Handle){0};
    if (values.element_size != sizeof(double) || values.len != array->length ||
        (array->length > 0 && values.data == NULL)) {
        zy_numpy_array_drop(array);
        zy_numpy_set_error("List length does not match numpy Array shape");
        return (ZL_Handle){0};
    }
    if (array->length > 0) memcpy(array->values, values.data, array->length * sizeof(double));
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_identity(int64_t size) {
    ZyNumpyArray* array = zy_numpy_allocate_2d(size, size);
    if (array == NULL) return (ZL_Handle){0};
    for (int64_t index = 0; index < size; ++index) {
        array->values[(size_t)(index * size + index)] = 1.0;
    }
    return zy_numpy_wrap(array);
}

uintptr_t zy_numpy_rank(ZL_Handle handle) {
    return (uintptr_t)zy_numpy_data(handle)->rank;
}

uintptr_t zy_numpy_dimension(ZL_Handle handle, uintptr_t axis) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    return axis < array->rank ? array->shape[axis] : 0;
}

int64_t zy_numpy_rows(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    return array->rank > 0 ? (int64_t)array->shape[0] : 0;
}

int64_t zy_numpy_columns(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    return array->rank > 1 ? (int64_t)array->shape[1] : 0;
}

int64_t zy_numpy_length(ZL_Handle handle) {
    return (int64_t)zy_numpy_data(handle)->length;
}

double zy_numpy_get(ZL_Handle handle, int64_t row, int64_t column) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (array->rank != 2 || row < 0 || column < 0 ||
        (uintptr_t)row >= array->shape[0] || (uintptr_t)column >= array->shape[1]) {
        return 0.0;
    }
    return array->values[(size_t)((uintptr_t)row * array->shape[1] + (uintptr_t)column)];
}

void zy_numpy_set(ZL_Handle handle, int64_t row, int64_t column, double value) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (array->rank != 2 || row < 0 || column < 0 ||
        (uintptr_t)row >= array->shape[0] || (uintptr_t)column >= array->shape[1]) {
        return;
    }
    array->values[(size_t)((uintptr_t)row * array->shape[1] + (uintptr_t)column)] = value;
}

double zy_numpy_get_at(ZL_Handle handle, ZL_Slice indices) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    size_t offset = 0;
    return zy_numpy_offset(array, indices, &offset) ? array->values[offset] : 0.0;
}

void zy_numpy_set_at(ZL_Handle handle, ZL_Slice indices, double value) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    size_t offset = 0;
    if (zy_numpy_offset(array, indices, &offset)) array->values[offset] = value;
}

double zy_numpy_sum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    double total = 0.0;
    for (size_t index = 0; index < array->length; ++index) total += array->values[index];
    return total;
}

double zy_numpy_minimum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (array->length == 0) return 0.0;
    double result = array->values[0];
    for (size_t index = 1; index < array->length; ++index) {
        if (array->values[index] < result) result = array->values[index];
    }
    return result;
}

double zy_numpy_maximum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (array->length == 0) return 0.0;
    double result = array->values[0];
    for (size_t index = 1; index < array->length; ++index) {
        if (array->values[index] > result) result = array->values[index];
    }
    return result;
}

ZL_Handle zy_numpy_copy(ZL_Handle handle) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    ZyNumpyArray* result = zy_numpy_allocate_shape(source->shape, source->rank);
    if (result == NULL) return (ZL_Handle){0};
    if (source->length > 0) {
        memcpy(result->values, source->values, source->length * sizeof(double));
    }
    return zy_numpy_wrap(result);
}

static ZL_Handle zy_numpy_reshape_dimensions(
    ZyNumpyArray* source,
    const uintptr_t* shape,
    size_t rank
) {
    size_t requested = 0;
    if (!zy_numpy_element_count(shape, rank, &requested)) return (ZL_Handle){0};
    if (requested != source->length) {
        zy_numpy_set_error("numpy reshape must preserve the element count");
        return (ZL_Handle){0};
    }
    ZyNumpyArray* result = zy_numpy_allocate_shape(shape, rank);
    if (result == NULL) return (ZL_Handle){0};
    if (source->length > 0) {
        memcpy(result->values, source->values, source->length * sizeof(double));
    }
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_reshape(ZL_Handle handle, int64_t rows, int64_t columns) {
    if (rows < 0 || columns < 0) {
        zy_numpy_set_error("numpy dimensions cannot be negative");
        return (ZL_Handle){0};
    }
    uintptr_t shape[2] = {(uintptr_t)rows, (uintptr_t)columns};
    return zy_numpy_reshape_dimensions(zy_numpy_data(handle), shape, 2);
}

ZL_Handle zy_numpy_reshape_shape(ZL_Handle handle, ZL_Slice shape) {
    const uintptr_t* dimensions = NULL;
    size_t rank = 0;
    size_t count = 0;
    if (!zy_numpy_read_shape(shape, &dimensions, &rank, &count)) return (ZL_Handle){0};
    (void)count;
    return zy_numpy_reshape_dimensions(zy_numpy_data(handle), dimensions, rank);
}

ZL_Handle zy_numpy_transpose(ZL_Handle handle) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    if (source->rank != 2) {
        zy_numpy_set_error("numpy transpose currently requires a rank-2 Array");
        return (ZL_Handle){0};
    }
    uintptr_t shape[2] = {source->shape[1], source->shape[0]};
    ZyNumpyArray* result = zy_numpy_allocate_shape(shape, 2);
    if (result == NULL) return (ZL_Handle){0};
    for (uintptr_t row = 0; row < source->shape[0]; ++row) {
        for (uintptr_t column = 0; column < source->shape[1]; ++column) {
            result->values[(size_t)(column * result->shape[1] + row)] =
                source->values[(size_t)(row * source->shape[1] + column)];
        }
    }
    return zy_numpy_wrap(result);
}

typedef double (*ZyNumpyBinaryOperation)(double left, double right);

static double zy_numpy_add_values(double left, double right) { return left + right; }
static double zy_numpy_subtract_values(double left, double right) { return left - right; }
static double zy_numpy_multiply_values(double left, double right) { return left * right; }

static ZL_Handle zy_numpy_binary(
    ZL_Handle left_handle,
    ZL_Handle right_handle,
    const char* operation,
    ZyNumpyBinaryOperation apply
) {
    ZyNumpyArray* left = zy_numpy_data(left_handle);
    ZyNumpyArray* right = zy_numpy_data(right_handle);
    if (!zy_numpy_same_shape(left, right)) {
        snprintf(zy_numpy_error, sizeof(zy_numpy_error), "numpy %s requires equal shapes", operation);
        return (ZL_Handle){0};
    }
    ZyNumpyArray* result = zy_numpy_allocate_shape(left->shape, left->rank);
    if (result == NULL) return (ZL_Handle){0};
    for (size_t index = 0; index < left->length; ++index) {
        result->values[index] = apply(left->values[index], right->values[index]);
    }
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_add(ZL_Handle left_handle, ZL_Handle right_handle) {
    return zy_numpy_binary(left_handle, right_handle, "add", zy_numpy_add_values);
}

ZL_Handle zy_numpy_subtract(ZL_Handle left_handle, ZL_Handle right_handle) {
    return zy_numpy_binary(left_handle, right_handle, "subtract", zy_numpy_subtract_values);
}

ZL_Handle zy_numpy_multiply(ZL_Handle left_handle, ZL_Handle right_handle) {
    return zy_numpy_binary(left_handle, right_handle, "multiply", zy_numpy_multiply_values);
}

ZL_Handle zy_numpy_scale(ZL_Handle handle, double factor) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    ZyNumpyArray* result = zy_numpy_allocate_shape(source->shape, source->rank);
    if (result == NULL) return (ZL_Handle){0};
    for (size_t index = 0; index < source->length; ++index) {
        result->values[index] = source->values[index] * factor;
    }
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_matmul(ZL_Handle left_handle, ZL_Handle right_handle) {
    ZyNumpyArray* left = zy_numpy_data(left_handle);
    ZyNumpyArray* right = zy_numpy_data(right_handle);
    if (left->rank != 2 || right->rank != 2) {
        zy_numpy_set_error("numpy matmul currently requires rank-2 Arrays");
        return (ZL_Handle){0};
    }
    if (left->shape[1] != right->shape[0]) {
        zy_numpy_set_error("numpy matmul requires left.columns == right.rows");
        return (ZL_Handle){0};
    }
    uintptr_t shape[2] = {left->shape[0], right->shape[1]};
    ZyNumpyArray* result = zy_numpy_allocate_shape(shape, 2);
    if (result == NULL) return (ZL_Handle){0};
    for (uintptr_t row = 0; row < left->shape[0]; ++row) {
        for (uintptr_t column = 0; column < right->shape[1]; ++column) {
            double total = 0.0;
            for (uintptr_t inner = 0; inner < left->shape[1]; ++inner) {
                total += left->values[(size_t)(row * left->shape[1] + inner)] *
                    right->values[(size_t)(inner * right->shape[1] + column)];
            }
            result->values[(size_t)(row * result->shape[1] + column)] = total;
        }
    }
    return zy_numpy_wrap(result);
}
