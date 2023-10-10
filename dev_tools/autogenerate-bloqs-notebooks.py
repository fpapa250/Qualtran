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

"""Autogeneration of stub Jupyter notebooks.

For each module listed in the `NOTEBOOK_SPECS` global variable (in this file)
we write a notebook with a title, module docstring,
standard imports, and information on each bloq listed in the
`gate_specs` field. For each gate, we render a docstring and diagrams.

## Adding a new gate.

 1. Create a new function that takes no arguments
    and returns an instance of the desired gate.
 2. If this is a new module: add a new key/value pair to the NOTEBOOK_SPECS global variable
    in this file. The key should be the name of the module with a `NotebookSpec` value. See
    the docstring for `NotebookSpec` for more information.
 3. Update the `NotebookSpec` `gate_specs` field to include a `BloqNbSpec` for your new gate.
    Provide your factory function from step (1).

## Autogen behavior.

Each autogenerated notebook cell is tagged, so we know it was autogenerated. Each time
this script is re-run, these cells will be re-rendered. *Modifications to generated _cells_
will not be persisted*.

If you add additional cells to the notebook it will *preserve them* even when this script is
re-run

Usage as a script:
    python dev_tools/autogenerate-bloqs-notebooks.py
"""

from typing import List

from qualtran_dev_tools.git_tools import get_git_root
from qualtran_dev_tools.jupyter_autogen import BloqNbSpec, NotebookSpec, render_notebook

import qualtran.bloqs.and_bloq
import qualtran.bloqs.and_bloq_test
import qualtran.bloqs.arithmetic
import qualtran.bloqs.arithmetic_test
import qualtran.bloqs.basic_gates.cnot_test
import qualtran.bloqs.basic_gates.hadamard_test
import qualtran.bloqs.basic_gates.rotation_test
import qualtran.bloqs.basic_gates.swap_test
import qualtran.bloqs.basic_gates.t_gate_test
import qualtran.bloqs.basic_gates.toffoli_test
import qualtran.bloqs.basic_gates.x_basis_test
import qualtran.bloqs.basic_gates.z_basis_test
import qualtran.bloqs.chemistry.single_factorization_test
import qualtran.bloqs.factoring.mod_exp
import qualtran.bloqs.factoring.mod_exp_test
import qualtran.bloqs.factoring.mod_mul_test
import qualtran.bloqs.sorting
import qualtran.bloqs.sorting_test
import qualtran.bloqs.swap_network
import qualtran.bloqs.swap_network_test

SOURCE_DIR = get_git_root() / 'qualtran/'

NOTEBOOK_SPECS: List[NotebookSpec] = [
    NotebookSpec(
        title='Swap Network',
        module=qualtran.bloqs.swap_network,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.basic_gates.swap_test._make_CSwap),
            BloqNbSpec(qualtran.bloqs.swap_network_test._make_CSwapApprox),
            BloqNbSpec(qualtran.bloqs.swap_network_test._make_SwapWithZero),
        ],
        directory=f'{SOURCE_DIR}/bloqs',
    ),
    NotebookSpec(
        title='Basic Gates',
        module=qualtran.bloqs.basic_gates,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.basic_gates.cnot_test._make_CNOT),
            BloqNbSpec(qualtran.bloqs.basic_gates.x_basis_test._make_plus_state),
            BloqNbSpec(qualtran.bloqs.basic_gates.z_basis_test._make_zero_state),
            BloqNbSpec(qualtran.bloqs.basic_gates.t_gate_test._make_t_gate),
            BloqNbSpec(qualtran.bloqs.basic_gates.rotation_test._make_Rz),
            BloqNbSpec(qualtran.bloqs.basic_gates.toffoli_test._make_Toffoli),
            BloqNbSpec(qualtran.bloqs.basic_gates.hadamard_test._make_Hadamard),
        ],
        directory=f'{SOURCE_DIR}/bloqs',
    ),
    NotebookSpec(
        title='And',
        module=qualtran.bloqs.and_bloq,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.and_bloq_test._make_and),
            BloqNbSpec(qualtran.bloqs.and_bloq_test._make_multi_and),
        ],
        directory=f'{SOURCE_DIR}/bloqs',
    ),
    NotebookSpec(
        title='Arithmetic',
        module=qualtran.bloqs.arithmetic,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_add),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_product),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_square),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_sum_of_squares),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_greater_than),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_scale_int_by_real),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_multiply_two_reals),
            BloqNbSpec(qualtran.bloqs.arithmetic_test._make_square_real_number),
        ],
        directory=f'{SOURCE_DIR}/bloqs',
    ),
    NotebookSpec(
        title='Sorting',
        module=qualtran.bloqs.sorting,
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.sorting_test._make_comparator),
            BloqNbSpec(qualtran.bloqs.sorting_test._make_bitonic_sort),
        ],
        directory=f'{SOURCE_DIR}/bloqs',
    ),
    NotebookSpec(
        title='Modular arithmetic',
        module=qualtran.bloqs.factoring,
        path_stem='ref-factoring',
        gate_specs=[
            BloqNbSpec(qualtran.bloqs.factoring.mod_exp_test._make_modexp),
            BloqNbSpec(qualtran.bloqs.factoring.mod_mul_test._make_modmul),
        ],
        directory=f'{SOURCE_DIR}/bloqs/factoring',
    ),
]


def render_notebooks():
    for nbspec in NOTEBOOK_SPECS:
        render_notebook(nbspec)


if __name__ == '__main__':
    render_notebooks()
