import subprocess
import ctypes
import os.path
import numpy as np
import datetime as dt
from moran_process.core.graph_core import GraphCore

dir_path = os.path.dirname(__file__)
lib_path = dir_path + "/numeric.so"

if not os.path.isfile(lib_path):
    subprocess.run(
        [
            "rustc",
            "--crate-type=cdylib",
            "-O",
            "-C",
            "panic=abort",
            "--edition",
            "2024",
            "-o",
            lib_path,
            dir_path + "/src/lib.rs",
        ],
        check=True,
    )

rustlib = ctypes.cdll.LoadLibrary(lib_path)

ptr_f64 = ctypes.POINTER(ctypes.c_double)
ptr_u32 = ctypes.POINTER(ctypes.c_uint32)

rustlib.compute.restype = None
rustlib.compute.argtypes = [
    ctypes.c_uint64,
    ptr_u32,
    ptr_u32,
    ctypes.c_double,
    ptr_f64,
    ctypes.c_uint8,
    ctypes.c_uint8,
]


def fixation_prob(g: GraphCore, r: float, thrds: int = 0) -> np.ndarray:
    """
    For each possible starting node, provides the fixation probability of the
    mutant, assuming it started there.
    """
    res = np.zeros([g.n_nodes], dtype=np.float64)
    rustlib.compute(
        g.n_nodes,
        ctypes.cast(g.nbrs.ctypes.data, ptr_u32),
        ctypes.cast(g.offsets.ctypes.data, ptr_u32),
        r,
        ctypes.cast(res.ctypes.data, ptr_f64),
        0,
        thrds,
    )
    return res


def absorb_time(g: GraphCore, r: float, thrds: int = 0) -> np.ndarray:
    """
    For each possible starting node, provides the average time until system
    homogeny, assuming the mutant started there.
    """
    res = np.zeros([g.n_nodes], dtype=np.float64)
    rustlib.compute(
        g.n_nodes,
        ctypes.cast(g.nbrs.ctypes.data, ptr_u32),
        ctypes.cast(g.offsets.ctypes.data, ptr_u32),
        r,
        ctypes.cast(res.ctypes.data, ptr_f64),
        1,
        thrds,
    )
    return res
