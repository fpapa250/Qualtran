from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import TComplexity
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


@frozen
class SelectChem(Bloq):
    """

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:

    References:
    """

    M: int
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('theta', 1),
                FancyRegister('U', 1),
                FancyRegister('V', 1),
                FancyRegister('p', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('alpha', 1),
                FancyRegister('q', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('beta', 1),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'SelectChem{dag}'

    def t_complexity(self) -> TComplexity:
        N = 2 * self.M**3
        t_count = 12 * N + 8 * int(np.ceil(np.log2(N)))
        return TComplexity(t=t_count)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, ctrl: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        pass


@frozen
class UnaryIteration(Bloq):
    """Placeholder for unary iteration bloq
    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    iteration_shape: Tuple[int, ...]

    @cached_property
    def registers(self) -> FancyRegisters:
        bitsize = (max(self.iteration_shape) - 1).bit_length()
        return FancyRegisters([FancyRegister('x', bitsize, wireshape=self.iteration_shape)])

    def pretty_name(self) -> str:
        reg_name = self.registers[0].name
        return f'In[{reg_name}]'

    def t_complexity(self) -> TComplexity:
        t_count = 4 * int(np.prod(self.iteration_shape)) - 4
        return TComplexity(t=t_count)


# @frozen
# class SelectedMajoranaFermion(Bloq):
#     """ """

#     iteration_lengths: Tuple[int, ...]
#     cvs: Tuple[int, ...] = tuple()
#     gate: Bloq

#     @cached_property
#     def registers(self, selection_names: Tuple[str], target_bitsize: int) -> FancyRegisters:
#         regs = [FancyRegister(f'In[{name}]', bitsize) for name in selection_names]
#         regs.append(FancyRegister(f"{self.gate.short_name()}", target_bitsize))
#         return FancyRegisters(regs)

#     def t_complexity(self) -> TComplexity:
#         pass

#     def build_composite_bloq(
#         self,
#         bb: 'CompositeBloqBuilder',
#         *,
#         ctrl: NDArray[Soquet],
#         sel_regs: List[SoquetT],
#         trg_reg: SoquetT,
#     ) -> Dict[str, 'SoquetT']:
#         """"""
#         regs = self.registers()
#         out = {}
#         for it_len, reg in zip(iteration_lengths, sel_regs):
#             out = bb.add(UnaryIteration(it_len))
#         # y, x = bb.add(CNOT(), control=y, target=x)
#         # x, y = bb.add(CNOT(), control=x, target=y)


@frozen
class PrepareChem(Bloq):
    """

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:
     - p: A two-bit control register.
     - (right) target: The output bit.

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
        (Verifying Measurement Based Uncomputation)[https://algassert.com/post/1903].
            Gidney, C. 2019.
    """

    M: int
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('theta', 1),
                FancyRegister('U', 1),
                FancyRegister('V', 1),
                FancyRegister('p', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('alpha', 1),
                FancyRegister('q', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('beta', 1),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'PrepareChem{dag}'

    def t_complexity(self) -> TComplexity:
        N = 2 * self.M**3
        t_count = 6 * N  # + O(mu + log N)
        return TComplexity(t=t_count)
