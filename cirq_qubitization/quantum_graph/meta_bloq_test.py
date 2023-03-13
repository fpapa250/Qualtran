import pytest

from cirq_qubitization.quantum_graph.composite_bloq_test import (
    Atom,
    TestParallelBloq,
    TestSerialBloq,
)
from cirq_qubitization.quantum_graph.meta_bloq import ControlledBloq


def test_controlled_serial():
    bloq = ControlledBloq(subbloq=TestSerialBloq()).decompose_bloq()
    assert (
        bloq.debug_text()
        == """\
C[Atom()]<0>
  LeftDangle.control -> control
  LeftDangle.stuff -> stuff
  control -> C[Atom()]<1>.control
  stuff -> C[Atom()]<1>.stuff
--------------------
C[Atom()]<1>
  C[Atom()]<0>.control -> control
  C[Atom()]<0>.stuff -> stuff
  control -> C[Atom()]<2>.control
  stuff -> C[Atom()]<2>.stuff
--------------------
C[Atom()]<2>
  C[Atom()]<1>.control -> control
  C[Atom()]<1>.stuff -> stuff
  stuff -> RightDangle.stuff
  control -> RightDangle.control"""
    )


def test_controlled_parallel():
    bloq = ControlledBloq(subbloq=TestParallelBloq()).decompose_bloq()
    assert (
        bloq.debug_text()
        == """\
C[Split(n=3)]<0>
  LeftDangle.control -> control
  LeftDangle.stuff -> split
  control -> C[Atom()]<1>.control
  split[0] -> C[Atom()]<1>.stuff
  split[1] -> C[Atom()]<2>.stuff
  split[2] -> C[Atom()]<3>.stuff
--------------------
C[Atom()]<1>
  C[Split(n=3)]<0>.control -> control
  C[Split(n=3)]<0>.split[0] -> stuff
  control -> C[Atom()]<2>.control
  stuff -> C[Join(n=3)]<4>.join[0]
--------------------
C[Atom()]<2>
  C[Atom()]<1>.control -> control
  C[Split(n=3)]<0>.split[1] -> stuff
  control -> C[Atom()]<3>.control
  stuff -> C[Join(n=3)]<4>.join[1]
--------------------
C[Atom()]<3>
  C[Atom()]<2>.control -> control
  C[Split(n=3)]<0>.split[2] -> stuff
  control -> C[Join(n=3)]<4>.control
  stuff -> C[Join(n=3)]<4>.join[2]
--------------------
C[Join(n=3)]<4>
  C[Atom()]<3>.control -> control
  C[Atom()]<3>.stuff -> join[2]
  C[Atom()]<1>.stuff -> join[0]
  C[Atom()]<2>.stuff -> join[1]
  join -> RightDangle.stuff
  control -> RightDangle.control"""
    )


def test_doubly_controlled():
    with pytest.raises(NotImplementedError):
        # TODO: https://github.com/quantumlib/cirq-qubitization/issues/149
        ControlledBloq(ControlledBloq(Atom()))
