from functools import cached_property
from typing import Any, Dict, Iterable, Optional, Tuple

import attrs
import networkx as nx
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization.bloq_algos.basic_gates import IntState
from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_counts import (
    big_O,
    get_counts_graph,
    SympySymbolAllocator,
)
from cirq_qubitization.quantum_graph.classical_sim import ClassicalValT
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.quantum_graph.util_bloqs import ArbitraryClifford
from cirq_qubitization.t_complexity_protocol import TComplexity


@frozen
class BigBloq(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(x=self.bitsize)

    def rough_decompose(self, mgr):
        return [(sympy.log(self.bitsize), SubBloq(unrelated_param=0.5))]


@frozen
class SubBloq(Bloq):
    unrelated_param: float

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(q=1)

    def rough_decompose(self, mgr):
        return [(big_O(1), TGate())]


def get_big_bloq_counts_graph_1(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
    from cirq_qubitization.bloq_algos.basic_gates import IntState
    from cirq_qubitization.bloq_algos.shors.shors import CtrlScaleModAdd

    ss = SympySymbolAllocator()
    n_c = ss.new_symbol('n_c')

    def canon(bloq: Bloq) -> Optional[Bloq]:
        if isinstance(bloq, ArbitraryClifford):
            return attrs.evolve(bloq, n=n_c)

        return bloq

    return get_counts_graph(bloq, ss, canon)
