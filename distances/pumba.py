from numba import jit
from numba.typed import Dict
from numba import types


def sparse_array_set(sa, r, c):
    return 0.0


def sparse_array_get(sa, r, c):
    return 0.0


class SparseArray(object):

    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.array = Dict.empty(key_type=types.int64, value_type=types.float64)

    def __setitem__(self, key, value):
        self.array[key] = value

    def __getitem__(self, item):
        return sparse_array_get()

    def __contains__(self, item):
        return item in self.array
