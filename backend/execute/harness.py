"""ctypes harness: call ``_mlir_ciface_forward`` in a built ``model.so``."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from torch_mlir.runtime import np_to_memref as nm


class CompiledModel:
    def __init__(self, so_path: str | Path, n_inputs: int, out_shape: tuple[int, ...]):
        self.lib = ctypes.CDLL(str(so_path))
        self.fn = self.lib._mlir_ciface_forward
        self.fn.restype = None
        self.fn.argtypes = [ctypes.c_void_p] * (1 + n_inputs)
        self.free = self.lib._mlir_memref_to_llvm_free
        self.free.argtypes = [ctypes.c_void_p]
        self.free.restype = None
        self.lib.optiler_stats_get.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.out_shape = tuple(out_shape)
        rank = len(self.out_shape)
        self._res_t = (nm.make_nd_memref_descriptor(rank, ctypes.c_float) if rank
                       else nm.make_zero_d_memref_descriptor(ctypes.c_float))

    def prepare(self, arrays: list[np.ndarray]):
        """Build input descriptors once, outside any timed region."""
        arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
        descs = [nm.get_ranked_memref_descriptor(a) for a in arrays]
        return arrays, descs, [ctypes.cast(ctypes.pointer(d), ctypes.c_void_p) for d in descs]

    def call_raw(self, ptrs):
        res = self._res_t()
        self.fn(ctypes.cast(ctypes.pointer(res), ctypes.c_void_p), *ptrs)
        return res

    def result_to_numpy(self, res) -> np.ndarray:
        if not self.out_shape:
            return np.array(res.aligned[res.offset], dtype=np.float32)
        return np.array(nm.ranked_memref_to_numpy(ctypes.pointer(res)), copy=True)

    def release(self, res) -> None:
        self.free(res.allocated)

    def __call__(self, arrays: list[np.ndarray]) -> np.ndarray:
        keep, _descs, ptrs = self.prepare(arrays)
        res = self.call_raw(ptrs)
        out = self.result_to_numpy(res)
        self.release(res)
        del keep
        return out

    def alloc_stats(self) -> dict:
        buf = (ctypes.c_uint64 * 5)()
        self.lib.optiler_stats_get(buf)
        return dict(zip(["alloc_count", "free_count", "live_bytes", "peak_bytes", "total_bytes"], list(buf)))

    def reset_stats(self) -> None:
        self.lib.optiler_stats_reset()
