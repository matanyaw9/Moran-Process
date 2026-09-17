import subprocess
import ctypes
import os.path
import numpy as np
import numpy.typing as npt
import datetime as dt
from dataclasses import dataclass
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

ptr_f32 = ctypes.POINTER(ctypes.c_float)
ptr_u32 = ctypes.POINTER(ctypes.c_uint32)

rustlib.compute.restype = None
rustlib.compute.argtypes = [
    ctypes.c_uint64,
    ptr_u32,
    ptr_u32,
    ctypes.c_float,
    ptr_f32,
    ctypes.c_uint64,
]


@dataclass
class Result:
    prob: npt.NDArray[np.float32]
    time: npt.NDArray[np.float32]
    ctime: npt.NDArray[np.float32]


def compute(
    g: GraphCore,
    selection_cffnt: float,
    thrds: int = 0,
) -> Result:
    """
    TODO doc me

    Computes the mutant's fixation probability for each starting point on the
    graph. Returned array is of the graph's size, with the value at each index
    representing the result of the mutant starting in the corresponding node.

    Computes the system's average time to homogenity (unconditional fixation
    time) for every possible mutant starting point.

    Computes the system's average time to mutant takeover (conditional fixation
    time) for every possible mutant starting point.

    The `thrds` parameter dictates the number of threads to use in the
    computation, a value of `0` uses all available cores.
    """
    res = np.zeros([3 * g.n_nodes], dtype=np.float32)
    rustlib.compute(
        g.n_nodes,
        ctypes.cast(g.nbrs.ctypes.data, ptr_u32),
        ctypes.cast(g.offsets.ctypes.data, ptr_u32),
        selection_cffnt,
        ctypes.cast(res.ctypes.data, ptr_f32),
        thrds,
    )
    return Result(
        prob=res[: g.n_nodes],
        time=res[g.n_nodes : 2 * g.n_nodes],
        ctime=res[2 * g.n_nodes :],
    )
