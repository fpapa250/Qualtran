"""Bloqs that wrap subbloqs."""

from functools import cached_property
from typing import List, Tuple

from attrs import field, frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    CompositeBloqBuilder,
    map_soqs,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.musical_score import Circle, WireSymbol
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


def _no_nesting_ctrls_yet(instance, field, val):
    # https://github.com/quantumlib/cirq-qubitization/issues/149
    assert isinstance(val, Bloq)
    if 'control' in [reg.name for reg in val.registers]:
        raise NotImplementedError("`ControlledBloq` doesn't support nesting yet.") from None


@frozen
class ControlledBloq(Bloq):
    """A controlled version of `subbloq`."""

    subbloq: Bloq = field(validator=_no_nesting_ctrls_yet)

    def pretty_name(self) -> str:
        return f'C[{self.subbloq.pretty_name()}]'

    def short_name(self) -> str:
        return f'C[{self.subbloq.short_name()}]'

    def __str__(self) -> str:
        return f'C[{self.subbloq}]'

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [FancyRegister(name="control", bitsize=1)] + list(self.subbloq.registers)
        )

    def decompose_bloq(self) -> 'CompositeBloq':
        if not isinstance(self.subbloq, CompositeBloq):
            return ControlledBloq(self.subbloq.decompose_bloq()).decompose_bloq()

        bb, initial_soqs = CompositeBloqBuilder.from_registers(self.registers)
        ctrl = initial_soqs['control']

        soq_map: List[Tuple[SoquetT, SoquetT]] = []
        for binst, in_soqs, old_out_soqs in self.subbloq.iter_bloqsoqs():
            in_soqs = map_soqs(in_soqs, soq_map)
            ctrl, *new_out_soqs = bb.add(ControlledBloq(binst.bloq), control=ctrl, **in_soqs)
            soq_map.extend(zip(old_out_soqs, new_out_soqs))

        fsoqs = map_soqs(self.subbloq.final_soqs(), soq_map)
        return bb.finalize(control=ctrl, **fsoqs)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'ctrl':
            return Circle(filled=True)
        return self.subbloq.wire_symbol(soq)
