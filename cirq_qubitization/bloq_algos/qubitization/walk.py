from functools import cached_property
from typing import Dict, Optional

import attrs
from attrs import frozen

from cirq_qubitization.bloq_algos.qubitization.prepare import BlackBoxPrepare, Prepare
from cirq_qubitization.bloq_algos.qubitization.reflect import Reflect
from cirq_qubitization.bloq_algos.qubitization.selectt import BlackBoxSelect, Select
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class Walk(Bloq):
    r"""Constructs a Szegedy Quantum Walk operator using `select` and `prepare`.

    Constructs the walk operator $W = R_L \cdot \mathrm{SELECT}$, which is a product of
    two reflections $R_L = (2|L><L| - I)$ and $\mathrm{SELECT}=\sum_{l}|l><l|H_{l}$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces. For an arbitrary eigenstate $|k>$ of $H$ with eigenvalue $E_k$, $|\ell>|k>$ and
    an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell>|k>$ is
    in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2\arccos(E_k / \lambda)$ s.t. $W = e^{i \arccos(E_k / \lambda) Y}$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $spectrum(H) = \lambda cos(arg(spectrum(W)))$ where $arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $SELECT=\sum_{l}|l><l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $PREPARE|00...00> = \sum_{l=0}^{L - 1}\sqrt{\frac{w_{l}}{\lambda}} |l> = |\ell>$
        cv: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.
        power: Constructs $W^{power}$ by repeatedly decomposing into `power` copies of $W$.
            Defaults to 1.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """
    select: BlackBoxSelect
    prepare: BlackBoxPrepare
    cv: Optional[int] = None
    power: int = 1

    def __attrs_post_init__(self):
        assert self.select.selection_bitsize == self.prepare.bitsize

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(
            selection=self.select.selection_bitsize, system=self.select.system_bitsize
        )

    @cached_property
    def reflect(self):
        return Reflect(prepare=self.prepare)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', selection: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        for _ in range(self.power):
            selection, system = bb.add(self.select, selection=selection, system=system)
            (selection,) = bb.add(self.reflect, selection=selection)

        return {'selection': selection, 'system': system}

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return attrs.evolve(self, power=self.power * power)

    def short_name(self) -> str:
        if self.power == 1:
            return 'W'
        return f'W^{self.power}'
