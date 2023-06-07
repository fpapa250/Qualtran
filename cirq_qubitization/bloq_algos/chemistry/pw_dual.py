from functools import cached_property
from typing import Tuple

import numpy as np
from attrs import frozen

from cirq_qubitization import TComplexity
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters


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
