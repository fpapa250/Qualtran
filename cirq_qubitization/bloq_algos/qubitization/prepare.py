import abc
from functools import cached_property
from typing import Dict, Tuple

import attrs
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side


class Prepare(Bloq, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def selection_registers(self) -> FancyRegisters:
        ...

    @property
    @abc.abstractmethod
    def junk_registers(self) -> FancyRegisters:
        ...

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([*self.selection_registers, *self.junk_registers])

    @abc.abstractmethod
    def dagger(self) -> 'Bloq':
        ...


@frozen
class DummyPrepare(Prepare):
    adjoint: bool = False

    @property
    def selection_registers(self) -> FancyRegisters:
        return FancyRegisters.build(x=5, y=5)

    @property
    def junk_registers(self) -> FancyRegisters:
        return FancyRegisters([])

    def dagger(self) -> 'Bloq':
        return attrs.evolve(self, adjoint=not self.adjoint)


@frozen
class Reshape(Bloq):
    n: int
    shapes: Tuple[Tuple[int, Tuple[int, ...]], ...]

    reg_prefix: str = 'x'
    ungroup: bool = True

    @cached_property
    def registers(self) -> 'FancyRegisters':
        lumped = Side.LEFT if self.ungroup else Side.RIGHT
        partitioned = Side.RIGHT if self.ungroup else Side.LEFT

        return FancyRegisters(
            [FancyRegister(self.reg_prefix, bitsize=self.n, side=lumped)]
            + [
                FancyRegister(f'{self.reg_prefix}{i}', bitsize=bs, wireshape=sh, side=partitioned)
                for i, (bs, sh) in enumerate(self.shapes)
            ]
        )

    def dagger(self):
        return attrs.evolve(self, ungroup=not self.ungroup)


@frozen
class BlackBoxPrepare(Bloq):
    prepare: Prepare
    adjoint: bool = False

    def short_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'Prep{dag}'

    @cached_property
    def bitsize(self):
        return sum(reg.total_bits() for reg in self.prepare.registers)

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters([FancyRegister('selection', self.bitsize)])

    def dagger(self) -> 'BlackBoxPrepare':
        return attrs.evolve(self, adjoint=not self.adjoint)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', selection: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        shapes = tuple((reg.bitsize, reg.wireshape) for reg in self.prepare.registers)
        reshape = Reshape(n=self.bitsize, shapes=shapes)

        sel_parts = bb.add(reshape, x=selection)
        sel_parts = bb.add(
            self.prepare, **{reg.name: sp for reg, sp in zip(self.prepare.registers, sel_parts)}
        )
        (selection,) = bb.add(reshape.dagger(), **{f'x{i}': sp for i, sp in enumerate(sel_parts)})
        return {'selection': selection}
