#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zyenlang_c_abi.h"

#define ZY_NUMPY_TYPE "zyenlang.numpy.Array"
#define ZY_NUMPY_MAX_ELEMENTS ((uint64_t)100000000)

typedef struct ZyNumpyArray {
    int64_t rows;
    int64_t columns;
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
    free(array->values);
    free(array);
}

static int zy_numpy_element_count(int64_t rows, int64_t columns, size_t* result) {
    if (rows < 0 || columns < 0) {
        zy_numpy_set_error("numpy dimensions cannot be negative");
        return 0;
    }
    uint64_t left = (uint64_t)rows;
    uint64_t right = (uint64_t)columns;
    if (right != 0 && left > ZY_NUMPY_MAX_ELEMENTS / right) {
        zy_numpy_set_error("numpy Array is too large");
        return 0;
    }
    uint64_t count = left * right;
    if (count > ZY_NUMPY_MAX_ELEMENTS || count > SIZE_MAX / sizeof(double)) {
        zy_numpy_set_error("numpy Array is too large");
        return 0;
    }
    *result = (size_t)count;
    return 1;
}

static ZyNumpyArray* zy_numpy_allocate(int64_t rows, int64_t columns) {
    size_t count = 0;
    if (!zy_numpy_element_count(rows, columns, &count)) return NULL;
    ZyNumpyArray* array = (ZyNumpyArray*)calloc(1, sizeof(ZyNumpyArray));
    if (array == NULL) {
        zy_numpy_set_error("cannot allocate numpy Array metadata");
        return NULL;
    }
    if (count > 0) {
        array->values = (double*)calloc(count, sizeof(double));
        if (array->values == NULL) {
            free(array);
            zy_numpy_set_error("cannot allocate numpy Array data");
            return NULL;
        }
    }
    array->rows = rows;
    array->columns = columns;
    return array;
}

static ZL_Handle zy_numpy_wrap(ZyNumpyArray* array) {
    if (array == NULL) return (ZL_Handle){0};
    return zl_handle_adopt(array, ZY_NUMPY_TYPE, zy_numpy_array_drop);
}

static ZyNumpyArray* zy_numpy_data(ZL_Handle handle) {
    return (ZyNumpyArray*)zl_handle_data(handle, ZY_NUMPY_TYPE);
}

ZL_Handle zy_numpy_zeros(int64_t rows, int64_t columns) {
    return zy_numpy_wrap(zy_numpy_allocate(rows, columns));
}

ZL_Handle zy_numpy_full(int64_t rows, int64_t columns, double value) {
    ZyNumpyArray* array = zy_numpy_allocate(rows, columns);
    if (array == NULL) return (ZL_Handle){0};
    size_t count = (size_t)(rows * columns);
    for (size_t index = 0; index < count; ++index) array->values[index] = value;
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_from_slice(int64_t rows, int64_t columns, ZL_Slice values) {
    size_t count = 0;
    if (!zy_numpy_element_count(rows, columns, &count)) return (ZL_Handle){0};
    if (values.element_size != sizeof(double) || values.len != count || (count > 0 && values.data == NULL)) {
        zy_numpy_set_error("List length does not match numpy Array shape");
        return (ZL_Handle){0};
    }
    ZyNumpyArray* array = zy_numpy_allocate(rows, columns);
    if (array == NULL) return (ZL_Handle){0};
    if (count > 0) memcpy(array->values, values.data, count * sizeof(double));
    return zy_numpy_wrap(array);
}

ZL_Handle zy_numpy_identity(int64_t size) {
    ZyNumpyArray* array = zy_numpy_allocate(size, size);
    if (array == NULL) return (ZL_Handle){0};
    for (int64_t index = 0; index < size; ++index) {
        array->values[(size_t)(index * size + index)] = 1.0;
    }
    return zy_numpy_wrap(array);
}

int64_t zy_numpy_rows(ZL_Handle handle) { return zy_numpy_data(handle)->rows; }
int64_t zy_numpy_columns(ZL_Handle handle) { return zy_numpy_data(handle)->columns; }
int64_t zy_numpy_length(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    return array->rows * array->columns;
}

double zy_numpy_get(ZL_Handle handle, int64_t row, int64_t column) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (row < 0 || column < 0 || row >= array->rows || column >= array->columns) return 0.0;
    return array->values[(size_t)(row * array->columns + column)];
}

void zy_numpy_set(ZL_Handle handle, int64_t row, int64_t column, double value) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    if (row < 0 || column < 0 || row >= array->rows || column >= array->columns) return;
    array->values[(size_t)(row * array->columns + column)] = value;
}

double zy_numpy_sum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    size_t count = (size_t)(array->rows * array->columns);
    double total = 0.0;
    for (size_t index = 0; index < count; ++index) total += array->values[index];
    return total;
}

double zy_numpy_minimum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    size_t count = (size_t)(array->rows * array->columns);
    if (count == 0) return 0.0;
    double result = array->values[0];
    for (size_t index = 1; index < count; ++index) {
        if (array->values[index] < result) result = array->values[index];
    }
    return result;
}

double zy_numpy_maximum(ZL_Handle handle) {
    ZyNumpyArray* array = zy_numpy_data(handle);
    size_t count = (size_t)(array->rows * array->columns);
    if (count == 0) return 0.0;
    double result = array->values[0];
    for (size_t index = 1; index < count; ++index) {
        if (array->values[index] > result) result = array->values[index];
    }
    return result;
}

ZL_Handle zy_numpy_copy(ZL_Handle handle) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    ZyNumpyArray* result = zy_numpy_allocate(source->rows, source->columns);
    if (result == NULL) return (ZL_Handle){0};
    size_t count = (size_t)(source->rows * source->columns);
    if (count > 0) memcpy(result->values, source->values, count * sizeof(double));
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_reshape(ZL_Handle handle, int64_t rows, int64_t columns) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    size_t requested = 0;
    if (!zy_numpy_element_count(rows, columns, &requested)) return (ZL_Handle){0};
    size_t available = (size_t)(source->rows * source->columns);
    if (requested != available) {
        zy_numpy_set_error("numpy reshape must preserve the element count");
        return (ZL_Handle){0};
    }
    ZyNumpyArray* result = zy_numpy_allocate(rows, columns);
    if (result == NULL) return (ZL_Handle){0};
    if (available > 0) memcpy(result->values, source->values, available * sizeof(double));
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_transpose(ZL_Handle handle) {
    ZyNumpyArray* source = zy_numpy_data(handle);
    ZyNumpyArray* result = zy_numpy_allocate(source->columns, source->rows);
    if (result == NULL) return (ZL_Handle){0};
    for (int64_t row = 0; row < source->rows; ++row) {
        for (int64_t column = 0; column < source->columns; ++column) {
            result->values[(size_t)(column * result->columns + row)] =
                source->values[(size_t)(row * source->columns + column)];
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
    if (left->rows != right->rows || left->columns != right->columns) {
        snprintf(zy_numpy_error, sizeof(zy_numpy_error), "numpy %s requires equal shapes", operation);
        return (ZL_Handle){0};
    }
    ZyNumpyArray* result = zy_numpy_allocate(left->rows, left->columns);
    if (result == NULL) return (ZL_Handle){0};
    size_t count = (size_t)(left->rows * left->columns);
    for (size_t index = 0; index < count; ++index) {
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
    ZyNumpyArray* result = zy_numpy_allocate(source->rows, source->columns);
    if (result == NULL) return (ZL_Handle){0};
    size_t count = (size_t)(source->rows * source->columns);
    for (size_t index = 0; index < count; ++index) result->values[index] = source->values[index] * factor;
    return zy_numpy_wrap(result);
}

ZL_Handle zy_numpy_matmul(ZL_Handle left_handle, ZL_Handle right_handle) {
    ZyNumpyArray* left = zy_numpy_data(left_handle);
    ZyNumpyArray* right = zy_numpy_data(right_handle);
    if (left->columns != right->rows) {
        zy_numpy_set_error("numpy matmul requires left.columns == right.rows");
        return (ZL_Handle){0};
    }
    ZyNumpyArray* result = zy_numpy_allocate(left->rows, right->columns);
    if (result == NULL) return (ZL_Handle){0};
    for (int64_t row = 0; row < left->rows; ++row) {
        for (int64_t column = 0; column < right->columns; ++column) {
            double total = 0.0;
            for (int64_t inner = 0; inner < left->columns; ++inner) {
                total += left->values[(size_t)(row * left->columns + inner)] *
                    right->values[(size_t)(inner * right->columns + column)];
            }
            result->values[(size_t)(row * result->columns + column)] = total;
        }
    }
    return zy_numpy_wrap(result);
}
