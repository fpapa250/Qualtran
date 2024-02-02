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

import pytest

from qualtran.bloqs.factoring.mod_sub import MontogomeryModSub
from qualtran.testing import assert_valid_bloq_decomposition


@pytest.mark.parametrize('bitsize', [5])
@pytest.mark.parametrize('k', [-5, 8])
@pytest.mark.parametrize('cvs', [[], [0, 1], [1, 0], [1, 1]])
def test_simple_add_constant_decomp_signed(bitsize, k, cvs):
    bloq = SimpleAddConstant(bitsize=bitsize, k=k, cvs=cvs, signed=True)
    assert_valid_bloq_decomposition(bloq)


@pytest.mark.parametrize(
    'bitsize,k,x,cvs,ctrls,result',
    [
        (5, 1, 2, (), (), 3),
        (5, 3, 2, (1,), (1,), 5),
        (5, 2, 0, (1, 0), (1, 0), 2),
        (5, 1, 2, (1, 0, 1), (0, 0, 0), 2),
    ],
)
def test_classical_simple_add_constant_unsigned(bitsize, k, x, cvs, ctrls, result):
    bloq = SimpleAddConstant(bitsize=bitsize, k=k, cvs=cvs, signed=False)
    cbloq = bloq.decompose_bloq()
    bloq_classical = bloq.call_classically(ctrls=ctrls, x=x)
    cbloq_classical = cbloq.call_classically(ctrls=ctrls, x=x)

    assert len(bloq_classical) == len(cbloq_classical)
    for i in range(len(bloq_classical)):
        np.testing.assert_array_equal(bloq_classical[i], cbloq_classical[i])

    assert bloq_classical[-1] == result