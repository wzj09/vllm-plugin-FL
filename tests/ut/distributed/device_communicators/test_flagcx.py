import ctypes
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import ReduceOp
from vllm.distributed.utils import StatelessProcessGroup


class DummyStatelessProcessGroup(StatelessProcessGroup):
    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size

    def broadcast_obj(self, obj, src=0):
        return obj


class FakeUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


class FakeFLAGCXLibrary:
    def __init__(self, *args, **kwargs):
        self.flagcxGetUniqueId = lambda *a, **kw: ctypes.pointer(FakeUniqueId())
        self.flagcxCommInitRank = lambda *a, **kw: "fake_comm"

        self.flagcxAllReduce = lambda *a, **kw: None
        self.flagcxAllGather = lambda *a, **kw: None
        self.flagcxReduceScatter = lambda *a, **kw: None
        self.flagcxReduce = lambda *a, **kw: None
        self.flagcxBroadcast = lambda *a, **kw: None

        self.flagcxSend = lambda *a, **kw: None
        self.flagcxRecv = lambda *a, **kw: None

        self.flagcxGroupStart = lambda *a, **kw: None
        self.flagcxGroupEnd = lambda *a, **kw: None

        self.adaptor_stream_copy = lambda *a, **kw: "fake_stream"
        self.adaptor_stream_free = lambda *a, **kw: None


fake_wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
fake_wrapper.FLAGCXLibrary = FakeFLAGCXLibrary
fake_wrapper.buffer_type = lambda *args, **kwargs: None
fake_wrapper.cudaStream_t = ctypes.c_void_p
fake_wrapper.flagcxComm_t = ctypes.c_void_p
fake_wrapper.flagcxUniqueId = FakeUniqueId

fake_wrapper.flagcxDataTypeEnum = MagicMock()
fake_wrapper.flagcxDataTypeEnum.from_torch = MagicMock(return_value=0)

fake_wrapper.flagcxRedOpTypeEnum = MagicMock()
fake_wrapper.flagcxRedOpTypeEnum.from_torch = MagicMock(return_value=0)

with patch.dict(
    "sys.modules",
    {
        "plugin": types.ModuleType("plugin"),
        "plugin.interservice": types.ModuleType("plugin.interservice"),
        "plugin.interservice.flagcx_wrapper": fake_wrapper,
    },
):

    from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

    @pytest.fixture
    def device():
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda:0")

    @pytest.fixture
    def group():
        return DummyStatelessProcessGroup(rank=0, world_size=2)

    @pytest.fixture
    def communicator(group, device):
        return PyFlagcxCommunicator(group=group, device=device, library_path="dummy.so")

    def test_init(communicator):
        assert communicator.comm == "fake_comm"
        assert communicator.available

    def test_all_reduce(communicator):
        x = torch.ones(2, device=communicator.device)
        y = communicator.all_reduce(x, op=ReduceOp.SUM)
        assert y.shape == x.shape

    def test_all_gather(communicator):
        inp = torch.ones(2, device=communicator.device)
        out = torch.empty(4, device=communicator.device)
        communicator.all_gather(out, inp)
        assert out.numel() == 4

    def test_all_gatherv(communicator):
        inp = torch.ones(2, device=communicator.device)
        out = torch.empty(4, device=communicator.device)
        communicator.all_gatherv(out, inp, sizes=[2, 2])
        assert out.numel() == 4

    def test_reduce_scatter(communicator):
        inp = torch.ones(4, device=communicator.device)
        out = torch.empty(2, device=communicator.device)
        communicator.reduce_scatter(out, inp)
        assert out.numel() == 2

    def test_reduce_scatterv(communicator):
        inp = torch.ones(4, device=communicator.device)
        out = torch.empty(2, device=communicator.device)
        communicator.reduce_scatterv(out, inp, sizes=[2, 2])
        assert out.numel() == 2

    def test_broadcast(communicator):
        t = torch.ones(4, device=communicator.device)
        communicator.broadcast(t, src=communicator.rank)
        communicator.broadcast(t, src=(communicator.rank + 1) % 2)

    def test_send_recv(communicator):
        t = torch.ones(4, device=communicator.device)
        communicator.send(t, dst=1)
        communicator.recv(t, src=1)

    def test_group_start_end(communicator):
        communicator.group_start()
        communicator.group_end()

    def test_device_mismatch_raises(communicator):
        cpu_tensor = torch.ones(2)
        with pytest.raises(AssertionError):
            communicator.all_reduce(cpu_tensor)
