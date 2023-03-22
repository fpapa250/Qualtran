import cirq
import numpy as np
import pytest

import cirq_qubitization


@pytest.mark.parametrize("n", [*range(3, 20), 25, 41])
@pytest.mark.parametrize("num_controls", [0, 1])
def test_prepare_uniform_superposition(n, num_controls):
    gate = cirq_qubitization.PrepareUniformSuperposition(n, num_controls=num_controls)
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    control, target = (all_qubits[:num_controls], all_qubits[num_controls:])
    turn_on_controls = [cirq.X(c) for c in control]
    prepare_uniform_op = gate.on(*control, *target)
    circuit = cirq.Circuit(turn_on_controls, cirq.decompose_once(prepare_uniform_op))
    qubit_order = cirq.QubitOrder.explicit(all_qubits, fallback=cirq.QubitOrder.DEFAULT)
    result = cirq.Simulator().simulate(circuit, qubit_order=qubit_order)
    final_target_state = cirq.sub_state_vector(
        result.final_state_vector,
        keep_indices=list(range(num_controls, num_controls + len(target))),
    )
    expected_target_state = np.asarray([np.sqrt(1.0 / n)] * n + [0] * (2 ** len(target) - n))
    cirq.testing.assert_allclose_up_to_global_phase(
        expected_target_state, final_target_state, atol=1e-6
    )
