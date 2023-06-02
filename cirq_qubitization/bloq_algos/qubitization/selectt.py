import abc
from functools import cached_property

from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


class Select(Bloq, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def control_registers(self) -> FancyRegisters:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> FancyRegisters:
        ...

    @property
    @abc.abstractmethod
    def system_registers(self) -> FancyRegisters:
        ...

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [*self.control_registers, *self.selection_registers, *self.system_registers]
        )


@frozen
class BlackBoxSelect(Bloq):
    select: Select

    def short_name(self) -> str:
        return f'Select'

    @cached_property
    def selection_bitsize(self):
        return sum(reg.total_bits() for reg in self.select.selection_registers)

    @cached_property
    def system_bitsize(self):
        return sum(reg.total_bits() for reg in self.select.system_registers)

    @cached_property
    def registers(self) -> 'FancyRegisters':
        return FancyRegisters.build(selection=self.selection_bitsize, system=self.system_bitsize)


@frozen
class DummySelect(Select):
    @property
    def control_registers(self) -> FancyRegisters:
        return FancyRegisters([])

    @property
    def selection_registers(self) -> FancyRegisters:
        return FancyRegisters.build(x=5, y=5)

    @property
    def system_registers(self) -> FancyRegisters:
        return FancyRegisters.build(x=128)
