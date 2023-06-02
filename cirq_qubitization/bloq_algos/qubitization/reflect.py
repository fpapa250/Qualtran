from functools import cached_property
from typing import Dict, Optional

import attrs
from attrs import frozen

from cirq_qubitization.bloq_algos.ctrl_pauli import CtrlPauli
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
class Reflect(Bloq):
    r"""Applies reflection around a state prepared by `prepare`

    Applies $R_{s} = I - 2|s><s|$ using $R_{s} = P^†(I - 2|0><0|)P$ s.t. $P|0> = |s>$.
    Here
        $|s>$: The state along which we want to reflect.
        $P$: Unitary that prepares that state $|s>$ from the zero state $|0>$
        $R_{s}$: Reflection operator that adds a `-1` phase to all states in the subspace
            spanned by $|s>$.

    The composite gate corresponds to implementing the following circuit:

    |control> ------------------ Z -------------------
                                 |
    |L>       ---- PREPARE^† --- o --- PREPARE -------


    Args:
        prepare_gate: An instance of `cq.StatePreparationAliasSampling` gate the corresponds to
            `PREPARE`.
        control_val: If 0/1, a controlled version of the reflection operator is constructed.
            Defaults to None, in which case the resulting reflection operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    prepare: BlackBoxPrepare
    cv: Optional[int] = None

    @cached_property
    def registers(self) -> FancyRegisters:
        registers = [] if self.cv is None else [FancyRegister('ctrl', 1)]
        registers.append(FancyRegister('selection', self.prepare.bitsize))
        return FancyRegisters(registers)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **soqs: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        pauli = '-Z'
        if self.cv is None:
            phase_trg = bb.allocate(n=1)
        else:
            phase_trg = soqs.pop('ctrl')
            if self.cv == 1:
                pauli = '+Z'

        sel = soqs.pop('selection')

        (sel,) = bb.add(self.prepare.dagger(), selection=sel)
        sel, phase_trg = bb.add(
            CtrlPauli(pauli, ctrl_bitsize=sel.reg.bitsize), ctrl=sel, target=phase_trg
        )
        (sel,) = bb.add(self.prepare, selection=sel)

        ret_soqs = {'selection': sel}
        if self.cv is None:
            bb.free(phase_trg)
        else:
            ret_soqs['ctrl'] = phase_trg
        return ret_soqs

    def controlled(self) -> 'Bloq':
        assert self.cv is None
        return attrs.evolve(self, cv=1)

    def short_name(self) -> str:
        return f'R[{self.prepare.short_name()}]'

    def wire_symbols(self, soq: Soquet):
        from cirq_qubitization.quantum_graph.musical_score import Circle, TextBox

        if soq.reg.name == 'ctrl':
            assert self.cv is not None
            return Circle(filled=self.cv == 1)
        assert soq.reg.name == 'selection'
        return TextBox(text=self.short_name())
