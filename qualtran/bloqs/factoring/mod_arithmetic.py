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
from typing import Optional, Set, Union

import sympy
from attrs import frozen

from qualtran import Bloq, Register, Signature
from qualtran.bloqs.basic_gates import Toffoli
from qualtran.resource_counting import BloqCountT, SympySymbolAllocator


@frozen
class ModAdd(Bloq):
    r"""An n-bit modular addition gate.

    Implements $U|x\rangle|y\rangle|p\rangle \rightarrow |x\rangle|y + x mod p\rangle|p\rangle$ using $4n Toffoli$ gates.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).
        p: A bitsize-sized input register (register p above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig. 6., 8.
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', bitsize=self.bitsize),
                Register('y', bitsize=self.bitsize),
                Register('p', bitsize=self.bitsize),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(4 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return 'y = y + x mod p'


@frozen
class ModSub(Bloq):
    r"""An n-bit modular subtraction gate.

    Implements $U|x\rangle|y\rangle|p\rangle \rightarrow |x\rangle|y - x mod p\rangle|p\rangle$ using $6n Toffoli$ gates.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input/output register (register y above).
        p: A bitsize-sized input register (register p above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig. 6., 8.
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', bitsize=self.bitsize),
                Register('y', bitsize=self.bitsize),
                Register('p', bitsize=self.bitsize),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(6 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return 'y = y - x mod p'


@frozen
class ModNeg(Bloq):
    r"""An n-bit modular negation gate.

    Implements $U|x\rangle|p\rangle \rightarrow |-x mod p\rangle|p\rangle$ using $2n Toffoli$ gates.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
        x: A bitsize-sized input register (register x above).
        p: A bitsize-sized input register (register p above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig. 6., 8.
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('p', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(2 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return 'x = -x mod p'


@frozen
class ModDbl(Bloq):
    r"""An n-bit modular doubling gate.

    Implements $U|x\rangle|p\rangle \rightarrow |2 * x mod p\rangle|p\rangle$ using $2n Toffoli$ gates.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
        x: A bitsize-sized input register (register x above).
        p: A bitsize-sized input register (register p above).

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig. 6., 8.
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature([Register('x', bitsize=self.bitsize), Register('p', bitsize=self.bitsize)])

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        return {(2 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return 'x = 2 * x mod p'


@frozen
class ModMult(Bloq):
    r"""An n-bit modular multiplication gate.

    Implements $U|x\rangle|y\rangle|p\rangle|0\rangle \rightarrow |x\rangle|y\rangle|p\rangle|x * y mod p\rangle$ using $2.25n^2 + 9n Toffoli$ gates.

    Args:
        bitsize: Number of bits used to represent each integer.

    Registers:
        x: A bitsize-sized input register (register x above).
        y: A bitsize-sized input register (register y above).
        p: A bitsize-sized input register (register p above).
        out: A bitsize-sized input register holding the output of the modular multiplication.

    References:
        [How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates](https://arxiv.org/abs/2306.08585) Fig. 6., 8.
    """

    bitsize: Union[int, sympy.Expr]

    @cached_property
    def signature(self) -> 'Signature':
        return Signature(
            [
                Register('x', bitsize=self.bitsize),
                Register('y', bitsize=self.bitsize),
                Register('p', bitsize=self.bitsize),
                Register('out', bitsize=self.bitsize),
            ]
        )

    def bloq_counts(self, ssa: Optional['SympySymbolAllocator'] = None) -> Set['BloqCountT']:
        # Upper bound is 2.25 * (n ** 2) + 9n. We truncate to 2 so that the return value is always an integer.
        return {(2 * (self.bitsize**2) + 9 * self.bitsize, Toffoli())}

    def short_name(self) -> str:
        return 'out = x * y mod p'
