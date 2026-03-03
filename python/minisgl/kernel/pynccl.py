from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV
from minisgl.utils.arch import is_hip

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


class _TorchDistributedPyNCCL:
    """PyNCCLCommunicator-compatible wrapper around torch.distributed ops (RCCL-backed on ROCm)."""

    def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None:
        import torch.distributed as dist

        dist.all_reduce(input, op=dist.ReduceOp.SUM)

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        import torch.distributed as dist

        dist.all_gather_into_tensor(output, input)

    def get_buffer(self) -> int:
        return 0


if not is_hip():

    @functools.cache
    def _load_nccl_module() -> Module:
        from .utils import load_aot

        return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])

    @functools.cache
    def _get_pynccl_wrapper_cls():
        import tvm_ffi

        @tvm_ffi.register_object("minisgl.NCCLWrapper")
        class PyNCCLImpl(tvm_ffi.Object):
            def __init__(self, *args):
                self.__ffi_init__(*args)

        return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    if is_hip():
        return _TorchDistributedPyNCCL()  # type: ignore

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
