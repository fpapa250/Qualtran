import dataclasses
from typing import Union

import attrs
import cirq_ft
import sympy

import qualtran
from qualtran.bloq_algos.factoring.mod_exp import ModExp
from qualtran.protos import registers_pb2
from qualtran.quantum_graph.bloq_test import TestCNOT
from qualtran.quantum_graph.composite_bloq_test import TestTwoCNOT
from qualtran.quantum_graph.fancy_registers import FancyRegisters
from qualtran.quantum_graph.meta_bloq import ControlledBloq
from qualtran.serialization import bloq as bloq_serialization


def test_bloq_to_proto_cnot():
    bloq_serialization.RESOLVER_DICT.update({'TestCNOT': TestCNOT})

    cnot = TestCNOT()
    proto_lib = bloq_serialization.bloqs_to_proto(cnot)
    assert len(proto_lib.table) == 1
    proto = proto_lib.table[0]
    assert len(proto.decomposition) == 0
    proto = proto.bloq
    assert proto.name == "TestCNOT"
    assert len(proto.registers.registers) == 2
    assert proto.registers.registers[0].name == 'control'
    assert proto.registers.registers[0].bitsize.int_val == 1
    assert proto.registers.registers[0].side == registers_pb2.Register.Side.THRU

    assert bloq_serialization.bloqs_from_proto(proto_lib) == [cnot]


def test_cbloq_to_proto_two_cnot():
    bloq_serialization.RESOLVER_DICT.update({'TestCNOT': TestCNOT})
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCNOT': TestTwoCNOT})

    cbloq = TestTwoCNOT().decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2  # TestTwoCNOT and TestCNOT
    # First one is always the CompositeBloq.
    assert len(proto_lib.table[0].decomposition) == 6
    assert proto_lib.table[0].bloq.t_complexity.clifford == 2
    # Test round trip.
    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


@attrs.frozen
class TestCSwap(qualtran.Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def t_complexity(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)


@dataclasses.dataclass(frozen=True)
class TestTwoCSwap(qualtran.Bloq):
    bitsize: Union[int, sympy.Expr]

    @property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def build_composite_bloq(self, bb, ctrl, x, y):
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=x, y=y)
        ctrl, x, y = bb.add(TestCSwap(self.bitsize), ctrl=ctrl, x=y, y=x)
        return {'ctrl': ctrl, 'x': x, 'y': y}


def test_cbloq_to_proto_test_two_cswap():
    bloq_serialization.RESOLVER_DICT.update({'TestCSwap': TestCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCSwap': TestTwoCSwap})

    bitsize = sympy.Symbol("a") * sympy.Symbol("b")
    cswap_proto_lib = bloq_serialization.bloqs_to_proto(TestCSwap(bitsize))
    assert len(cswap_proto_lib.table) == 1
    assert len(cswap_proto_lib.table[0].decomposition) == 0
    cswap_proto = cswap_proto_lib.table[0].bloq
    assert cswap_proto.name == "TestCSwap"
    assert len(cswap_proto.args) == 1
    assert cswap_proto.args[0].name == "bitsize"
    assert sympy.parse_expr(cswap_proto.args[0].sympy_expr) == bitsize
    assert len(cswap_proto.registers.registers) == 3

    assert TestCSwap(bitsize) in bloq_serialization.bloqs_from_proto(cswap_proto_lib)

    cswap_proto = bloq_serialization.bloqs_to_proto(TestCSwap(100)).table[0].bloq
    cbloq = TestTwoCSwap(100).decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq)
    assert len(proto_lib.table) == 2
    assert proto_lib.table[1].bloq == cswap_proto
    assert proto_lib.table[0].bloq.t_complexity.t == 7 * 100 * 2
    assert proto_lib.table[0].bloq.t_complexity.clifford == 10 * 100 * 2
    assert len(proto_lib.table[0].decomposition) == 9

    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


def test_cbloq_to_proto_test_mod_exp():
    mod_exp = ModExp.make_for_shor(17 * 19, g=8)
    cbloq = mod_exp.decompose_bloq()
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq, max_depth=1)
    num_binst = len(set(binst.bloq for binst in cbloq.bloq_instances))
    assert len(proto_lib.table) == 1 + num_binst

    cbloq = ControlledBloq(mod_exp)
    proto_lib = bloq_serialization.bloqs_to_proto(cbloq, max_depth=1)
    # 2x that of ModExp.make_for_shor(17 * 19).decompose_bloq() because each bloq in the
    # decomposition is now controlled and each Controlled(subbloq) requires 2 entries in the
    # table - one for ControlledBloq and second for subbloq.
    assert len(proto_lib.table) == 2 * (1 + num_binst)

    assert cbloq in bloq_serialization.bloqs_from_proto(proto_lib)


@attrs.frozen
class TestMetaBloq(qualtran.Bloq):
    sub_bloq_one: qualtran.Bloq
    sub_bloq_two: qualtran.Bloq

    def __post_attrs_init__(self):
        assert self.sub_bloq_one.registers == self.sub_bloq_two.registers

    @property
    def registers(self) -> 'FancyRegisters':
        return self.sub_bloq_one.registers

    def build_composite_bloq(self, bb, **soqs):
        soqs |= zip(soqs.keys(), bb.add(self.sub_bloq_one, **soqs))
        soqs |= zip(soqs.keys(), bb.add(self.sub_bloq_two, **soqs))
        return soqs


def test_meta_bloq_to_proto():
    bloq_serialization.RESOLVER_DICT.update({'TestCSwap': TestCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestTwoCSwap': TestTwoCSwap})
    bloq_serialization.RESOLVER_DICT.update({'TestMetaBloq': TestMetaBloq})

    sub_bloq_one = TestTwoCSwap(20)
    sub_bloq_two = TestTwoCSwap(20).decompose_bloq()
    bloq = TestMetaBloq(sub_bloq_one, sub_bloq_two)
    proto_lib = bloq_serialization.bloqs_to_proto(bloq, name="Meta Bloq Test")
    assert proto_lib.name == "Meta Bloq Test"
    assert len(proto_lib.table) == 3  # TestMetaBloq, TestTwoCSwap, CompositeBloq

    proto_lib = bloq_serialization.bloqs_to_proto(bloq, max_depth=2)
    assert len(proto_lib.table) == 4  # TestMetaBloq, TestTwoCSwap, CompositeBloq, TestCSwap

    assert proto_lib.table[0].bloq.name == 'TestMetaBloq'
    assert len(proto_lib.table[0].decomposition) == 9

    assert proto_lib.table[1].bloq.name == 'TestTwoCSwap'
    assert len(proto_lib.table[1].decomposition) == 9

    assert proto_lib == bloq_serialization.bloqs_to_proto(bloq, bloq, TestTwoCSwap(20), max_depth=2)
    assert bloq in bloq_serialization.bloqs_from_proto(proto_lib)