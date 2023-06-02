from functools import cached_property
from typing import Dict, Optional

from attrs import frozen

from cirq_qubitization.bloq_algos.qubitization.prepare import BlackBoxPrepare, Prepare
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class CtrlPauli(Bloq):
    pauli: str
    ctrl_bitsize: int
    cv: int = 0

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(ctrl=self.ctrl_bitsize, target=1)

    def short_name(self) -> str:
        return self.pauli
