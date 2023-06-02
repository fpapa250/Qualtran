import cirq
import numpy as np

from cirq_qubitization.bloq_algos.qubitization.walk import Walk
from cirq_qubitization.quantum_graph.cirq_conversion import CirqGateAsBloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder


def get_resource_state(m: int):
    """Returns a state vector representing the resource state on m qubits from Eq.17 of Ref-1.

    Returns a numpy array of size 2^{m} representing the state vector corresponding to the state
    $$
        \sqrt{\frac{2}{2^m + 1}} \sum_{n=0}^{2^{m}-1} \sin{\frac{\pi(n + 1)}{2^{m}+1}}\ket{n}
    $$

    Args:
        m: Number of qubits to prepare the resource state on.

    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Eq. 17
    """
    den = 1 + 2**m
    norm = np.sqrt(2 / den)
    return norm * np.sin(np.pi * (1 + np.arange(2**m)) / den)


def phase_estimation(walk: Walk, m: int):
    """Heisenberg limited phase estimation circuit for learning eigenphase of `walk`.

     Heisenberg limited phase estimation circuit
    for learning eigenphases of the `walk` operator with `m` bits of accuracy. The
    circuit is implemented as given in Fig.2 of Ref-1.

    Args:
        walk: Qubitization walk operator.
        m: Number of bits of accuracy for phase estimation.

    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Fig. 2
    """
    bb = CompositeBloqBuilder()
    phase = bb.add_register('phase', m)
    selection = bb.add_register('selection', walk.select.selection_bitsize)
    system = bb.add_register('system', walk.select.system_bitsize)
    reflect = walk.reflect

    creflect = reflect.controlled()  # TODO: cv = 0
    cwalk = walk.controlled()

    phase_bits = bb.split(phase)
    (phase_bits,) = bb.add(
        CirqGateAsBloq(cirq.StatePreparationChannel(get_resource_state(m), name='𝜒_m')),
        qubits=phase_bits,
    )

    phase_bits[0], selection, system = bb.add(
        cwalk, ctrl=phase_bits[0], selection=selection, system=system
    )
    for i in range(1, m):
        phase_bits[i], selection = bb.add(creflect, ctrl=phase_bits[i], selection=selection)
        selection, system = bb.add(walk, selection=selection, system=system)
        walk = walk**2
        phase_bits[i], selection = bb.add(creflect, ctrl=phase_bits[i], selection=selection)

    # TODO: inverse
    (phase_bits,) = bb.add(CirqGateAsBloq(cirq.QuantumFourierTransformGate(m)), qubits=phase_bits)
    return bb.finalize(phase=bb.join(phase_bits), selection=selection, system=system)
