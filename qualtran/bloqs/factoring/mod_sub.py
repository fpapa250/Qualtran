#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from functools import cached_property
from typing import Optional, Set, Union, Dict

import sympy
from attrs import frozen

from qualtran import Bloq, Register, Signature, SoquetT
from qualtran.bloqs.basic_gates import Toffoli, XGate
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator
from qualtran.bloqs.arithmetic.addition import Add, SimpleAddConstant
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli


@frozen
class MontogomeryModSub(Bloq):
    r"""An n-bit modular subtraction gate.
    This gate is designed to operate on integers in the Montogomery form.
    Implements $U|x\rangle|y\rangle \rightarrow |x\rangle|y - x \mod p\rangle$ using $6n$ Toffoli
    gates.
    Args:
        bitsize: Number of bits used to represent each integer.
        p: The modulus for the subtraction.
    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).
    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585)
        Fig 6c and 8
    """

    bitsize: int
    p: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('y', bitsize=self.bitsize)])
    
    def on_classical_vals(
        self, x: 'ClassicalValT', y: 'ClassicalValT'
    ) -> Dict[str, 'ClassicalValT']:

        return {'x': x, 'y': y}

    def build_composite_bloq(self, bb: 'BloqBuilder', x: SoquetT, y: SoquetT) -> Dict[str, 'SoquetT']:

        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Add constant p+1 to the x register.
        x = bb.add(SimpleAddConstant(bitsize=self.bitsize, k=self.p + 1, signed=True, cvs=()), x=x)

        # Perform in-place addition on quantum register y.
        x, y = bb.add(Add(bitsize=self.bitsize), a=x, b=y)

        # Add constant -(p+1) to the x register to uncompute the first addition.
        x = bb.add(
            SimpleAddConstant(bitsize=self.bitsize, k=-1 * (self.p + 1), signed=True, cvs=()), x=x
        )

        # Bit flip all qubits in register x.
        x_split = bb.split(x)
        for i in range(self.bitsize):
            x_split[i] = bb.add(XGate(), q=x_split[i])
        x = bb.join(x_split)

        # Return the output registers.
        return {'x': x, 'y': y}

    def short_name(self) -> str:
        return f'y = y - x mod {self.p}'
