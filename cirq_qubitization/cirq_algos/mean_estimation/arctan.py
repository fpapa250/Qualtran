from typing import Sequence, Union, Iterable

import numpy as np
from attrs import frozen
import cirq

from cirq_qubitization import t_complexity_protocol, bit_tools


@frozen
class ArcTan(cirq.ArithmeticGate):
    """Applies U|x>|0>|0000...0> = |x>|sign>|abs(-2 arctan(x) / pi)>.

    Args:
        selection_bitsize: The bitsize of input register |x>.
        target_bitsize: The bitsize of output register. The computed quantity,
            $\abs(-2 * \arctan(x) / \pi)$ is stored as a fixed-length binary approximation
            in the output register of size `target_bitsize`.
    """

    selection_bitsize: int
    target_bitsize: int

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return (2,) * self.selection_bitsize, (2,), (2,) * self.target_bitsize

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> "ArcTan":
        raise NotImplementedError()

    def apply(self, input_val: int, target_sign: int, target_val: int) -> Union[int, Iterable[int]]:
        output_val = -2 * np.arctan(input_val, dtype=np.double) / np.pi
        assert -1 <= output_val <= 1
        output_sign = 1 if np.sign(output_val) < 0 else 0
        output_bin = bit_tools.float_as_fixed_width_int(np.abs(output_val), self.target_bitsize)
        if input_val == 0 and target_sign == 0 and target_val == 0:
            print(output_sign, output_bin, target_val ^ output_bin)
        # TODO: Verify float to int conversion.
        return input_val, target_sign ^ output_sign, target_val ^ output_bin

    def _t_complexity_(self) -> t_complexity_protocol.TComplexity:
        # Approximate T-complexity of O(target_bitsize)
        return t_complexity_protocol.TComplexity(t=self.target_bitsize)

    def __pow__(self, power) -> 'ArcTan':
        if power in [+1, -1]:
            return self
        raise NotImplementedError("__pow__ is only implemented for +1/-1.")