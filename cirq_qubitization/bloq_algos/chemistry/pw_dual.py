from functools import cached_property
from typing import Dict, Tuple

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import TComplexity
from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
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
        trg_name, trg_bitsize = self.target_desc
        target = regs[trg_name]
        it_ranges = tuple(it_range for _, it_range in self.selection_desc)
        sel_regs = {k: v for k, v in regs.items() if k != trg_name}
        add_regs = {f"x{i}": sel_regs[sel] for i, sel in enumerate(sel_regs.keys())}
        add_regs["trg"] = target
        xs = bb.add(UnaryIteration(it_ranges, trg_bitsize), **add_regs)
        out = {k: xs[i] for i, k in enumerate(sel_regs.keys())}
        (out[trg_name],) = bb.add(GateOn(trg_bitsize, gate=self.gate), on=xs[-1])
        return out


@frozen
class UnaryIteration(Bloq):
    """Placeholder for unary iteration bloq
    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    iteration_ranges: Tuple[int, ...]
    target_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        bitsizes = [(isize - 1).bit_length() for isize in self.iteration_ranges]
        regs = [FancyRegister(f'x{i}', bs) for i, bs in enumerate(bitsizes)]
        regs += [FancyRegister('trg', self.target_bitsize)]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        reg_name = self.registers[0].name
        return f'In[{reg_name}]'

    def t_complexity(self) -> TComplexity:
        t_count = 4 * int(np.prod(self.iteration_shape)) - 4
        return TComplexity(t=t_count)

    def rough_decompose(self, mgr):
        t_count = 4 * int(np.prod(self.iteration_ranges)) - 4
        return [(t_count, TGate())]


@frozen
class GateOn(Bloq):
    """Placeholder for N-qubit gate
    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    bitsize: int
    gate: Bloq

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister('on', self.bitsize)])

    def pretty_name(self) -> str:
        return f'[{self.gate}]'


@frozen
class SelectedMajoranaFermion(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    selection_desc: Tuple[Tuple[str, int], ...]
    target_desc: Tuple[str, int]
    gate: Bloq
    cvs: Tuple[int, ...] = tuple()

    @cached_property
    def registers(self) -> FancyRegisters:
        trg_name, bitsize = self.target_desc
        regs = [
            FancyRegister(n, bitsize=(it_len - 1).bit_length()) for n, it_len in self.selection_desc
        ]
        regs += [FancyRegister(trg_name, bitsize)]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        name = f"{self.gate}"[0]
        return rf'In[Z{name}]'

    def t_complexity(self) -> TComplexity:
        iteration_size = sum([np.prod(it_shape) for _, it_shape in self.selection_desc])
        return TComplexity(t=4 * iteration_size - 4)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        trg_name, trg_bitsize = self.target_desc
        target = regs[trg_name]
        it_ranges = tuple(it_range for _, it_range in self.selection_desc)
        sel_regs = {k: v for k, v in regs.items() if k != trg_name}
        add_regs = {f"x{i}": sel_regs[sel] for i, sel in enumerate(sel_regs.keys())}
        add_regs["trg"] = target
        xs = bb.add(UnaryIteration(it_ranges, trg_bitsize), **add_regs)
        out = {k: xs[i] for i, k in enumerate(sel_regs.keys())}
        (out[trg_name],) = bb.add(GateOn(trg_bitsize, gate=self.gate), on=xs[-1])
        return out


@frozen
class PrepareChem(Bloq):
    """PrepareChem Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
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


@frozen
class CSwapApprox(Bloq):
    r"""Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements $\mathrm{CSWAP}_n = |0 \rangle\langle 0| I + |1 \rangle\langle 1| \mathrm{SWAP}_n$
    such that the output state is correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm
    and thus ignored. See the reference for more details.

    Args:
        bitsize: The bitsize of the two registers being swapped.

    Registers:
        ctrl: the control bit
        x: the first register
        y: the second register

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters.build(ctrl=1, x=self.bitsize, y=self.bitsize)

    def cirq_decomposition(
        self, ctrl: Sequence[cirq.Qid], x: Sequence[cirq.Qid], y: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Write the decomposition as a cirq circuit.

        This method is taken from the cirq.GateWithRegisters implementation and is used
        to partially support `build_composite_bloq` by relying on this cirq implementation.
        """
        (ctrl,) = ctrl

        def g(q: cirq.Qid, adjoint=False) -> cirq.OP_TREE:
            yield [cirq.S(q), cirq.H(q)]
            yield cirq.T(q) ** (1 - 2 * adjoint)
            yield [cirq.H(q), cirq.S(q) ** -1]

        cnot_x_to_y = [cirq.CNOT(x, y) for x, y in zip(x, y)]
        cnot_y_to_x = [cirq.CNOT(y, x) for x, y in zip(x, y)]
        g_inv_on_y = [list(g(q, True)) for q in y]  # Uses len(target_y) T-gates
        g_on_y = [list(g(q)) for q in y]  # Uses len(target_y) T-gates

        yield [cnot_y_to_x, g_inv_on_y, cnot_x_to_y, g_inv_on_y]
        yield multi_control_multi_target_pauli.MultiTargetCNOT(len(y)).on(ctrl, *y)
        yield [g_on_y, cnot_x_to_y, g_on_y, cnot_y_to_x]

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', ctrl: 'SoquetT', x: 'SoquetT', y: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        from cirq_qubitization.quantum_graph.cirq_conversion import cirq_circuit_to_cbloq

        cirq_quregs = self.registers.get_cirq_quregs()
        cbloq = cirq_circuit_to_cbloq(cirq.Circuit(self.cirq_decomposition(**cirq_quregs)))

        # Split our registers to "flat" api from cirq circuit; add the circuit; join back up.
        qvars = np.concatenate(([ctrl], bb.split(x), bb.split(y)))
        (qvars,) = bb.add_from(cbloq, qubits=qvars)
        return {
            'ctrl': qvars[0],
            'x': bb.join(qvars[1 : self.bitsize + 1]),
            'y': bb.join(qvars[-self.bitsize :]),
        }

    def t_complexity(self) -> TComplexity:
        """T complexity as explained in Appendix B.2.c of https://arxiv.org/abs/1812.00954"""
        n = self.bitsize
        # 4 * n: G gates, each wth 1 T and 4 cliffords
        # 4 * n: CNOTs
        # 2 * n - 1: CNOTs from 1 MultiTargetCNOT
        return TComplexity(t=4 * n, clifford=22 * n - 1)

    def short_name(self) -> str:
        return '~swap'
