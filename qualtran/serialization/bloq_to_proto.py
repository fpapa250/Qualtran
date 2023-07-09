import dataclasses
from typing import Any, Callable, Dict, List, Optional

import attrs
import inspect

from qualtran.protos import args_pb2, bloq_pb2
from qualtran.quantum_graph.bloq import Bloq
from qualtran.quantum_graph.bloq_counts import SympySymbolAllocator
from qualtran.quantum_graph.composite_bloq import CompositeBloq
from qualtran.quantum_graph.meta_bloq import ControlledBloq
from qualtran.quantum_graph.util_bloqs import Split, Join, Allocate, Free, ArbitraryClifford
from qualtran.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    Soquet,
    LeftDangle,
    RightDangle,
)
from qualtran.serialization import annotations_to_proto, args_to_proto, registers_to_proto

from qualtran import bloq_algos
from qualtran.bloq_algos import basic_gates, factoring

RESOLVER_DICT = {
    'CNOT': basic_gates.CNOT,
    'Rx': basic_gates.Rx,
    'Ry': basic_gates.Ry,
    'Rz': basic_gates.Rz,
    'CSwap': basic_gates.CSwap,
    'TwoBitCSwap': basic_gates.TwoBitCSwap,
    'TwoBitSwap': basic_gates.TwoBitSwap,
    'TGate': basic_gates.TGate,
    'MinusEffect': basic_gates.MinusEffect,
    'MinusState': basic_gates.MinusState,
    'PlusState': basic_gates.PlusState,
    'PlusEffect': basic_gates.PlusEffect,
    'XGate': basic_gates.XGate,
    'IntEffect': basic_gates.IntEffect,
    'IntState': basic_gates.IntState,
    'OneEffect': basic_gates.OneEffect,
    'OneState': basic_gates.OneState,
    'ZeroEffect': basic_gates.ZeroEffect,
    'ZeroState': basic_gates.ZeroState,
    'ZGate': basic_gates.ZGate,
    'CtrlAddK': factoring.CtrlAddK,
    'CtrlModAddK': factoring.CtrlModAddK,
    'CtrlScaleModAdd': factoring.CtrlScaleModAdd,
    'ModExp': factoring.ModExp,
    'CtrlModMul': factoring.CtrlModMul,
    'And': bloq_algos.and_bloq.And,
    'MultiAnd': bloq_algos.and_bloq.MultiAnd,
    'Add': bloq_algos.arithmetic.Add,
    'Square': bloq_algos.arithmetic.Square,
    'SumOfSquares': bloq_algos.arithmetic.SumOfSquares,
    'Product': bloq_algos.arithmetic.Product,
    'GreaterThan': bloq_algos.arithmetic.GreaterThan,
    'Comparator': bloq_algos.sorting.Comparator,
    'BitonicSort': bloq_algos.sorting.BitonicSort,
    'CSwapApprox': bloq_algos.swap_network.CSwapApprox,
    'SwapWithZero': bloq_algos.swap_network.SwapWithZero,
    'Split': Split,
    'Join': Join,
    'Allocate': Allocate,
    'Free': Free,
    'ArbitraryClifford': ArbitraryClifford,
    'ControlledBloq': ControlledBloq,
}


def bloqs_to_proto(
    *bloqs: Bloq,
    name: str = '',
    pred: Callable[[BloqInstance], bool] = lambda _: True,
    max_depth: int = 1,
) -> bloq_pb2.BloqLibrary:
    """Serializes one or more Bloqs as a `BloqLibrary`."""

    bloq_to_idx: Dict[Bloq, int] = {}
    for bloq in bloqs:
        _add_bloq_to_dict(bloq, bloq_to_idx)
        _populate_bloq_to_idx(bloq, bloq_to_idx, pred, max_depth)

    # `bloq_to_idx` would now contain a list of all bloqs that should be serialized.
    library = bloq_pb2.BloqLibrary(name=name)
    for bloq, bloq_id in bloq_to_idx.items():
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            decomposition = [_connection_to_proto(cxn, bloq_to_idx) for cxn in cbloq.connections]
        except (NotImplementedError, KeyError):
            # NotImplementedError is raised if `bloq` does not have a decomposition.
            # KeyError is raises if `bloq` has a decomposition but we do not wish to serialize it
            # because of conditions checked by `pred` and `max_depth`.
            decomposition = None

        try:
            bloq_counts = {
                bloq_to_idx[b]: args_to_proto.int_or_sympy_to_proto(c)
                for c, b in bloq.bloq_counts(SympySymbolAllocator())
            }
        except (NotImplementedError, KeyError):
            # NotImplementedError is raised if `bloq` does not implement bloq_counts.
            # KeyError is raises if `bloq` has `bloq_counts` but we do not wish to serialize it
            # because of conditions checked by `pred` and `max_depth`.
            bloq_counts = None

        library.table.add(
            bloq_id=bloq_id,
            decomposition=decomposition,
            bloq_counts=bloq_counts,
            bloq=_bloq_to_proto(bloq, bloq_to_idx=bloq_to_idx),
        )
    return library


def bloqs_from_proto(lib: bloq_pb2.BloqLibrary) -> List[Bloq]:
    idx_to_proto = {b.bloq_id: b for b in lib.table}
    idx_to_bloq: Dict[int, Bloq] = {}
    for bloq in lib.table:
        _populate_idx_to_bloq(bloq, idx_to_proto, idx_to_bloq)
    return list(idx_to_bloq.values())


def _populate_idx_to_bloq(
    bloq: bloq_pb2.BloqLibrary.BloqWithDecomposition,
    idx_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition],
    idx_to_bloq: Dict[int, Bloq],
):
    if bloq.bloq_id in idx_to_bloq:
        return
    idx_to_bloq[bloq.bloq_id] = _construct_bloq(bloq, idx_to_proto, idx_to_bloq)


def _construct_bloq(
    bloq: bloq_pb2.BloqLibrary.BloqWithDecomposition,
    idx_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition],
    idx_to_bloq: Dict[int, Bloq],
) -> Bloq:
    if bloq.bloq.name == 'CompositeBloq':
        return CompositeBloq(
            cxns=[
                _connection_from_proto(cxn, idx_to_proto, idx_to_bloq) for cxn in bloq.decomposition
            ],
            registers=registers_to_proto.registers_from_proto(bloq.bloq.registers),
        )
    elif bloq.bloq.name in RESOLVER_DICT:
        print("DEBUG:", bloq.bloq.args)
        kwargs = {}
        for arg in bloq.bloq.args:
            if arg.HasField('subbloq'):
                kwargs[arg.name] = _idx_to_bloq(arg.subbloq, idx_to_proto, idx_to_bloq)
            else:
                kwargs.update(args_to_proto.arg_from_proto(arg))
        return RESOLVER_DICT[bloq.bloq.name](**kwargs)
    raise ValueError(f"Unable to deserialize {bloq=}")


def _connection_from_proto(
    cxn: bloq_pb2.Connection,
    idx_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition],
    idx_to_bloq: Dict[int, Bloq],
) -> Connection:
    return Connection(
        left=_soquet_from_proto(cxn.left, idx_to_proto, idx_to_bloq),
        right=_soquet_from_proto(cxn.right, idx_to_proto, idx_to_bloq),
    )


def _soquet_from_proto(
    soq: bloq_pb2.Soquet,
    idx_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition],
    idx_to_bloq: Dict[int, Bloq],
) -> Soquet:
    binst = (
        eval(soq.dangling_t)
        if soq.HasField('dangling_t')
        else BloqInstance(
            i=soq.bloq_instance.instance_id,
            bloq=_idx_to_bloq(soq.bloq_instance.bloq_id, idx_to_proto, idx_to_bloq),
        )
    )
    return Soquet(
        binst=binst, reg=registers_to_proto.register_from_proto(soq.register), idx=tuple(soq.index)
    )


def _idx_to_bloq(
    bloq_id: int,
    idx_to_proto: Dict[int, bloq_pb2.BloqLibrary.BloqWithDecomposition],
    idx_to_bloq: Dict[int, Bloq],
):
    if bloq_id not in idx_to_bloq:
        _populate_idx_to_bloq(idx_to_proto[bloq_id], idx_to_proto, idx_to_bloq)
    return idx_to_bloq[bloq_id]


def _iter_fields(bloq: Bloq):
    """Yields fields of `bloq` iff `type(bloq)` is implemented using `dataclasses` or `attr`."""
    if dataclasses.is_dataclass(type(bloq)):
        for field in dataclasses.fields(bloq):
            if field.name in inspect.signature(type(bloq).__init__).parameters:
                yield field
    elif attrs.has(type(bloq)):
        for field in attrs.fields(type(bloq)):
            if field.name in inspect.signature(type(bloq).__init__).parameters:
                yield field


def _connection_to_proto(cxn: Connection, bloq_to_idx: Dict[Bloq, int]):
    return bloq_pb2.Connection(
        left=_soquet_to_proto(cxn.left, bloq_to_idx), right=_soquet_to_proto(cxn.right, bloq_to_idx)
    )


def _soquet_to_proto(soq: Soquet, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Soquet:
    if isinstance(soq.binst, DanglingT):
        return bloq_pb2.Soquet(
            dangling_t=repr(soq.binst),
            register=registers_to_proto.register_to_proto(soq.reg),
            index=soq.idx,
        )
    else:
        return bloq_pb2.Soquet(
            bloq_instance=_bloq_instance_to_proto(soq.binst, bloq_to_idx),
            register=registers_to_proto.register_to_proto(soq.reg),
            index=soq.idx,
        )


def _bloq_instance_to_proto(
    binst: BloqInstance, bloq_to_idx: Dict[Bloq, int]
) -> bloq_pb2.BloqInstance:
    return bloq_pb2.BloqInstance(instance_id=binst.i, bloq_id=bloq_to_idx[binst.bloq])


def _add_bloq_to_dict(bloq: Bloq, bloq_to_idx: Dict[Bloq, int]):
    """Adds `{bloq: len(bloq_to_idx)}` to `bloq_to_idx` dictionary if it doesn't exist already."""
    if bloq not in bloq_to_idx:
        next_idx = len(bloq_to_idx)
        bloq_to_idx[bloq] = next_idx


def _cbloq_dot_bloq_instances(cbloq: CompositeBloq) -> List[BloqInstance]:
    """Equivalent to `cbloq.bloq_instances`, but preserves insertion order among Bloq instances."""
    ret = {}
    for cxn in cbloq.connections:
        for soq in [cxn.left, cxn.right]:
            if not isinstance(soq.binst, DanglingT):
                ret[soq.binst] = 0
    return list(ret.keys())


def _populate_bloq_to_idx(
    bloq: Bloq, bloq_to_idx: Dict[Bloq, int], pred: Callable[[BloqInstance], bool], max_depth: int
):
    """Recursively track all primitive Bloqs to be serialized, as part of `bloq_to_idx` dictionary."""

    assert bloq in bloq_to_idx
    if max_depth > 0:
        # Decompose the current Bloq and track it's decomposed Bloqs.
        try:
            cbloq = bloq if isinstance(bloq, CompositeBloq) else bloq.decompose_bloq()
            for binst in _cbloq_dot_bloq_instances(cbloq):
                _add_bloq_to_dict(binst.bloq, bloq_to_idx)
                if pred(binst):
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, max_depth - 1)
                else:
                    _populate_bloq_to_idx(binst.bloq, bloq_to_idx, pred, 0)
        except NotImplementedError:
            # NotImplementedError is raised if `bloq` does not have a decomposition.
            ...

        # Approximately decompose the current Bloq and it's decomposed Bloqs.
        try:
            for _, subbloq in bloq.bloq_counts(SympySymbolAllocator()):
                _add_bloq_to_dict(subbloq, bloq_to_idx)
                _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)

        except NotImplementedError:
            # NotImplementedError is raised if `bloq` does not implement bloq_counts.
            ...

    # If the current Bloq contains other Bloqs as sub-bloqs, add them to the `bloq_to_idx` dict.
    # This is only supported for Bloqs implemented as dataclasses / attrs.
    for field in _iter_fields(bloq):
        subbloq = getattr(bloq, field.name)
        if isinstance(subbloq, Bloq):
            _add_bloq_to_dict(subbloq, bloq_to_idx)
            _populate_bloq_to_idx(subbloq, bloq_to_idx, pred, 0)


def _bloq_to_proto(bloq: Bloq, *, bloq_to_idx: Dict[Bloq, int]) -> bloq_pb2.Bloq:
    try:
        t_complexity = annotations_to_proto.t_complexity_to_proto(bloq.t_complexity())
    except:
        t_complexity = None

    return bloq_pb2.Bloq(
        name=bloq.__class__.__name__,
        registers=registers_to_proto.registers_to_proto(bloq.registers),
        t_complexity=t_complexity,
        args=_bloq_args_to_proto(bloq, bloq_to_idx=bloq_to_idx),
    )


def _bloq_args_to_proto(
    bloq: Bloq, *, bloq_to_idx: Dict[Bloq, int]
) -> Optional[List[args_pb2.BloqArg]]:
    if isinstance(bloq, CompositeBloq):
        return None

    ret = [
        _bloq_arg_to_proto(name=field.name, val=getattr(bloq, field.name), bloq_to_idx=bloq_to_idx)
        for field in _iter_fields(bloq)
    ]
    return ret if ret else None


def _bloq_arg_to_proto(name: str, val: Any, bloq_to_idx: Dict[Bloq, int]) -> args_pb2.BloqArg:
    if isinstance(val, Bloq):
        return args_pb2.BloqArg(name=name, subbloq=bloq_to_idx[val])
    return args_to_proto.arg_to_proto(name=name, val=val)
