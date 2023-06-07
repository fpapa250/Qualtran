from cirq_qubitization.bloq_algos.chemistry.pw_dual import PrepareChem, SelectChem
from cirq_qubitization.jupyter_tools import execute_notebook


def _make_select():
    from cirq_qubitization.bloq_algos.chemistry.pw_dual import SelectChem

    M = 8
    return SelectChem(8)


def _make_prepare():
    from cirq_qubitization.bloq_algos.chemistry.pw_dual import PrepareChem

    M = 8
    return PrepareChem(M)


def _make_unary_iteration():
    from cirq_qubitization.bloq_algos.chemistry.pw_dual import UnaryIteration

    shape = (2, 3, 5)
    return UnaryIteration(shape)


def _make_selected_majorana():
    import numpy as np

    from cirq_qubitization.bloq_algos.basic_gates import XGate
    from cirq_qubitization.bloq_algos.chemistry.pw_dual import SelectedMajoranaFermion

    selection_desc = (("p", (3, 3, 3)), ("alpha", (2,)))
    return SelectedMajoranaFermion(selection_desc, XGate())


def test_notebook():
    execute_notebook('chemistry')


def test_select():
    select = SelectChem(8)


def test_prepare():
    select = PrepareChem(8)
