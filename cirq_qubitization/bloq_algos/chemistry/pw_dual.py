from abc import ABCMeta
from functools import cached_property
from typing import Dict, Tuple

import cirq
import numpy as np
import sympy
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import TComplexity
from cirq_qubitization.bloq_algos.basic_gates.cnot import CNOT
from cirq_qubitization.bloq_algos.basic_gates.hadamard import HGate
from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.bloq_algos.basic_gates.toffoli import ToffoliGate
from cirq_qubitization.bloq_algos.basic_gates.x_basis import XGate
from cirq_qubitization.bloq_algos.swap_network import CSwapApprox
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_counts import big_O
from cirq_qubitization.quantum_graph.cirq_conversion import CirqGateAsBloq
from cirq_qubitization.quantum_graph.composite_bloq import (
    CompositeBloq,
    CompositeBloqBuilder,
    SoquetT,
)
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet


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
        if isinstance(self.iteration_ranges[0], sympy.Symbol):
            # TODO tidy this up
            bitsizes = [sympy.log(i) for i in self.iteration_ranges]
        else:
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
        if isinstance(self.iteration_ranges[0], sympy.Symbol):
            # t_count = big_O(4 * np.prod(self.iteration_ranges) - 4
            t_count = big_O(np.prod(self.iteration_ranges))
        else:
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

    selection_bitsizes: Tuple[int, ...]
    target_bitsize: int
    gate: Bloq
    cvs: Tuple[int, ...] = tuple()

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [
            FancyRegister(f's_{i}', bitsize=bs) for (i, bs) in enumerate(self.selection_bitsizes)
        ]
        regs += [FancyRegister(f't', bitsize=self.target_bitsize)]
        regs += [FancyRegister(f'c_{i}', bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        name = f"{self.gate}"[0]
        return rf'In[Z{name}]'

    def t_complexity(self) -> TComplexity:
        iteration_size = sum([np.prod(it_shape) for _, it_shape in self.selection_desc])
        return TComplexity(t=4 * iteration_size - 4)

    # def build_composite_bloq(
    #     self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    # ) -> Dict[str, 'SoquetT']:
    # it_ranges = tuple(it_range for _, it_range in self.selection_desc)
    # sel_regs = {k: v for k, v in regs.items() if k != trg_name}
    # add_regs = {f"x{i}": sel_regs[sel] for i, sel in enumerate(sel_regs.keys())}
    # add_regs["trg"] = target
    # xs = bb.add(UnaryIteration(it_ranges, trg_bitsize), **add_regs)
    # out = {k: xs[i] for i, k in enumerate(sel_regs.keys())}
    # (out[trg_name],) = bb.add(GateOn(trg_bitsize, gate=self.gate), on=xs[-1])
    # return out


class _SelectedMajoranaFermion(Bloq):
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
        if isinstance(self.selection_desc[0][1], sympy.Symbol):
            # TODO tidy this up
            bitsizes = [sympy.log(i) for _, i in self.selection_desc]
        else:
            bitsizes = [(isize - 1).bit_length() for _, isize in self.selection_desc]
        regs = [FancyRegister(n, bitsize=bs) for (n, _), bs in zip(self.selection_desc, bitsizes)]
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
class SelectChem(Bloq):
    """

    Args:
        cv1: Whether the first bit is a positive control.
        cv2: Whether the second bit is a positive control.

    Registers:

    References:
    """

    l_bitsize: int
    k_bitsize: int
    p_bitsize: int
    target_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('succ', 1),
                FancyRegister('l', 1),
                FancyRegister('two_body', 1),
                FancyRegister('Q', self.k_bitsize, wireshape=(3,)),
                FancyRegister('succ_kpq', 1),
                FancyRegister('k', self.k_bitsize, wireshape=(3,)),
                FancyRegister('p', self.p_bitsize),
                FancyRegister('q', self.p_bitsize),
                FancyRegister('Re/Im', 1),
                FancyRegister('A/B', 1),
                FancyRegister('term', 1),
                FancyRegister('anc', 1),
                FancyRegister('alpha', 1),
                FancyRegister('psi_a', self.target_bitsize),
                FancyRegister('psi_b', self.target_bitsize),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'SFBE{dag}'

    # def build_composite_bloq(
    #     self, bb: 'CompositeBloqBuilder', *, ctrl: NDArray[Soquet]
    # ) -> Dict[str, 'SoquetT']:
    #     """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

    #     This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
    #     will throw if `self.adjoint=True`.
    #     """
    # trg_name, trg_bitsize = self.target_desc
    # target = regs[trg_name]
    # it_ranges = tuple(it_range for _, it_range in self.selection_desc)
    # sel_regs = {k: v for k, v in regs.items() if k != trg_name}
    # add_regs = {f"x{i}": sel_regs[sel] for i, sel in enumerate(sel_regs.keys())}
    # add_regs["trg"] = target
    # xs = bb.add(unaryiteration(it_ranges, trg_bitsize), **add_regs)
    # out = {k: xs[i] for i, k in enumerate(sel_regs.keys())}
    # (out[trg_name],) = bb.add(gateon(trg_bitsize, gate=self.gate), on=xs[-1])
    # return out


@frozen
class SelectMajoranaFromCirq(Bloq):
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
        num_spin_orb = int(2 * self.M**3)
        return FancyRegisters(
            [
                FancyRegister('theta', 1),
                FancyRegister('U', 1),
                FancyRegister('V', 1),
                FancyRegister('p', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('alpha', 1),
                FancyRegister('q', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('beta', 1),
                FancyRegister('target', num_spin_orb),
                FancyRegister('control', 1),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'SelectChem{dag}'

    def t_complexity(self) -> TComplexity:
        N = 2 * self.M**3
        t_count = 12 * N + 8 * int(np.ceil(np.log2(N)))
        return TComplexity(t=t_count)

    def cirq_decomposition(self, **regs: cirq.Qid) -> cirq.OP_TREE:
        from cirq_qubitization.cirq_algos.chemistry import SelectChem as SelectChemCirq

        gate = SelectChemCirq(self.M, 1)
        context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        expand_pq = {}
        _xyz = "xyz"
        for k, v in regs.items():
            if k == "p" or k == "q":
                for i, dim in enumerate(v):
                    expand_pq[f"{k}{_xyz[i]}"] = list(dim)
            else:
                expand_pq[k] = v

        return gate.decompose_from_registers(context, **expand_pq)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        from cirq_qubitization.quantum_graph.cirq_conversion import cirq_circuit_to_cbloq

        cirq_quregs = self.registers.get_cirq_quregs()
        cbloq = cirq_circuit_to_cbloq(cirq.Circuit(self.cirq_decomposition(**cirq_quregs)))

        # Split our registers to "flat" api from cirq circuit; add the circuit; join back up.
        qvars = []
        disps = {}
        disp = 0
        for k, v in regs.items():
            if isinstance(v, np.ndarray):
                for _r in v:
                    spl = bb.split(_r)
                    qvars.append(spl)
                disps[k] = [disp, v.size]
                disp += v.size
            else:
                spl = bb.split(v)
                disps[k] = [disp, len(spl)]
                disp += len(spl)
                qvars.append(spl)
        # qvars = np.concatenate([bb.split(r) for _, r in regs.items()])
        (qvars,) = bb.add_from(cbloq, qubits=np.concatenate(qvars))
        out_soqs = {}
        for k, (s, d) in disps.items():
            print(k, s, d)
            if d > 1:
                out_soqs[k] = qvars[s : s + d]
            else:
                out_soqs[k] = qvars[s]
        return out_soqs


@frozen
class SelectChemFromCirq(Bloq):
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
        num_spin_orb = int(2 * self.M**3)
        return FancyRegisters(
            [
                FancyRegister('theta', 1),
                FancyRegister('U', 1),
                FancyRegister('V', 1),
                FancyRegister('p', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('alpha', 1),
                FancyRegister('q', (self.M - 1).bit_length(), wireshape=(3,)),
                FancyRegister('beta', 1),
                FancyRegister('target', num_spin_orb),
                FancyRegister('control', 1),
            ]
        )

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return f'SelectChem{dag}'

    def t_complexity(self) -> TComplexity:
        N = 2 * self.M**3
        t_count = 12 * N + 8 * int(np.ceil(np.log2(N)))
        return TComplexity(t=t_count)

    def cirq_decomposition(self, **regs: cirq.Qid) -> cirq.OP_TREE:
        from cirq_qubitization.cirq_algos.chemistry import SelectChem as SelectChemCirq

        gate = SelectChemCirq(self.M, 1)
        context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        expand_pq = {}
        _xyz = "xyz"
        for k, v in regs.items():
            if k == "p" or k == "q":
                for i, dim in enumerate(v):
                    expand_pq[f"{k}{_xyz[i]}"] = list(dim)
            else:
                expand_pq[k] = v

        return gate.decompose_from_registers(context, **expand_pq)

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        from cirq_qubitization.quantum_graph.cirq_conversion import cirq_circuit_to_cbloq

        cirq_quregs = self.registers.get_cirq_quregs()
        cbloq = cirq_circuit_to_cbloq(cirq.Circuit(self.cirq_decomposition(**cirq_quregs)))

        # Split our registers to "flat" api from cirq circuit; add the circuit; join back up.
        qvars = []
        disps = {}
        disp = 0
        for k, v in regs.items():
            if isinstance(v, np.ndarray):
                for _r in v:
                    spl = bb.split(_r)
                    qvars.append(spl)
                disps[k] = [disp, v.size]
                disp += v.size
            else:
                spl = bb.split(v)
                disps[k] = [disp, len(spl)]
                disp += len(spl)
                qvars.append(spl)
        # qvars = np.concatenate([bb.split(r) for _, r in regs.items()])
        (qvars,) = bb.add_from(cbloq, qubits=np.concatenate(qvars))
        out_soqs = {}
        for k, (s, d) in disps.items():
            print(k, s, d)
            if d > 1:
                out_soqs[k] = qvars[s : s + d]
            else:
                out_soqs[k] = qvars[s]
        return out_soqs


@frozen
class QROM(Bloq):
    data_bitsizes: Tuple[int, ...]
    selection_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...]

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"s_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)]
        regs = [FancyRegister(f"t_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'QROM{dag}'

    def t_complexity(self) -> TComplexity:
        t_count = 4 * np.prod(sympy.exp(s) for s in self.selection_bitsizes)
        return TComplexity(t=big_O(t_count))

    def rough_decompose(self, mgr):
        t_count = big_O(4 * np.prod(sympy.exp(s) for s in self.selection_bitsizes))
        return [(t_count, TGate())]


@frozen
class PrepareChem(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    data_bitsizes: Tuple[int, ...]
    output_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"d_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [
            FancyRegister(f"o_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)
        ]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'prep{dag}'

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        pass


@frozen
class Prepare(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    data_bitsizes: Tuple[int, ...]
    output_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"d_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"o_{i}", bitsize=bs) for i, bs in enumerate(self.output_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'prep{dag}'

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        pass


@frozen
class IndexedPrepare(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
    """

    selection_bitsizes: Tuple[int, ...]
    data_bitsizes: Tuple[int, ...]
    output_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"s_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)]
        regs = [FancyRegister(f"d_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"o_{i}", bitsize=bs) for i, bs in enumerate(self.output_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'IndPrep{dag}'


@frozen
class SingleFactorization(Bloq):
    """

    Args:

    Registers:

    References:
    """

    l_bitsize: int
    k_bitsize: int
    p_bitsize: int
    target_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('succ', 1),
                FancyRegister('l', 1),
                FancyRegister('two_body', 1),
                FancyRegister('Q', self.k_bitsize, wireshape=(3,)),
                FancyRegister('succ_kpq', 1),
                # FancyRegister('k', self.k_bitsize, wireshape=(3,)),
                FancyRegister('p', self.p_bitsize),
                FancyRegister('q', self.p_bitsize),
                FancyRegister('Re/Im', 1),
                FancyRegister('A/B', 1),
                FancyRegister('term', 1),
                FancyRegister('anc', 1),
                FancyRegister('alpha', 1),
                FancyRegister('psi_a', self.target_bitsize),
                FancyRegister('psi_b', self.target_bitsize),
            ]
        )

    @classmethod
    def build(cls, num_kpts: int, num_spat_orb: int, num_aux: int) -> "SingleFactorization":
        l_size = num_kpts * num_aux
        l_bitsize = l_size.bit_length()
        k_bitsize = 3 * num_kpts.bit_length()
        p_bitsize = num_spat_orb.bit_length()
        return SingleFactorization(
            l_bitsize=l_bitsize,
            k_bitsize=k_bitsize,
            p_bitsize=p_bitsize,
            target_bitsize=num_kpts * num_spat_orb,
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, ctrl: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        # 1. Prepare


@frozen
class AddMod(Bloq):
    input_bitsize: int
    output_bitsize: int

    @cached_property
    def register(self) -> FancyRegister:
        return FancyRegisters(
            [
                FancyRegister('input', self.input_bitsize),
                FancyRegister('output', self.output_bitsize),
            ]
        )

    def rough_decompose(self, mgr):
        if isinstance(self.input_bitsize, sympy.Symbol):
            # t_count = big_O(4 * np.prod(self.iteration_ranges) - 4
            t_count = big_O(self.input_bitsize)
        else:
            # TODO: this is worst case use proper decompose
            t_count = 2 * self.input_bitsize
        return [(t_count, TGate())]


@frozen
class BlockEncoding(Bloq):
    l_bitsize: int
    k_bitsize: int
    p_bitsize: int
    target_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('succ', 1),
                FancyRegister('l', self.l_bitsize),
                FancyRegister('two_body', 1),
                FancyRegister('Q', self.k_bitsize),  # TODO fix wireshape
                FancyRegister('succ_kpq', 1),
                FancyRegister('k', self.k_bitsize),  # TODO fix wireshape
                FancyRegister('p', self.p_bitsize),
                FancyRegister('q', self.p_bitsize),
                FancyRegister('ReIm', 1),
                FancyRegister('AB', 1),
                FancyRegister('term', 1),
                FancyRegister('anc', 1),
                FancyRegister('alpha', 1),
                FancyRegister('psia', self.target_bitsize),
                FancyRegister('psib', self.target_bitsize),
            ]
        )

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        # 1. Prepare
        out = {}
        l, two_body, Q, succ = bb.add(
            Prepare((self.l_bitsize,), (1, self.k_bitsize), cvs=(1,)),
            d_0=regs['l'],
            o_0=regs['two_body'],
            o_1=regs['Q'],
            c_0=regs['succ'],
        )
        l, k, p, q, re_im, succ_kpq = bb.add(
            IndexedPrepare(
                (self.l_bitsize,), (self.k_bitsize, self.p_bitsize, self.p_bitsize, 1), cvs=(1,)
            ),
            s_0=l,
            d_0=regs['k'],
            d_1=regs['p'],
            d_2=regs['q'],
            d_3=regs['ReIm'],
            c_0=regs['succ_kpq'],
        )
        (ab,) = bb.add(HGate(), q=regs['AB'])
        (term,) = bb.add(HGate(), q=regs['term'])
        (anc,) = bb.add(HGate(), q=regs['anc'])
        (alpha,) = bb.add(HGate(), q=regs['alpha'])
        (alpha, psia, psib) = bb.add(
            CSwapApprox(self.target_bitsize), ctrl=alpha, x=regs['psia'], y=regs['psib']
        )
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        (ab, two_body, re_im) = bb.add(ToffoliGate(), c0=ab, c1=two_body, t=re_im)
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (Q, q, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=Q, s_1=q, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        # Missing Multi control
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (k, p, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=k, s_1=p, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        (ab, two_body, re_im) = bb.add(ToffoliGate(), c0=ab, c1=two_body, t=re_im)
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (alpha, psia, psib) = bb.add(CSwapApprox(self.target_bitsize), ctrl=alpha, x=psia, y=psib)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        l, k, p, q, re_im, succ_kpq = bb.add(
            IndexedPrepare(
                (self.l_bitsize,),
                (self.k_bitsize, self.p_bitsize, self.p_bitsize, 1),
                cvs=(1,),
                adjoint=True,
            ),
            s_0=l,
            d_0=k,
            d_1=p,
            d_2=q,
            d_3=re_im,
            c_0=succ_kpq,
        )
        (ab,) = bb.add(HGate(), q=ab)
        (term,) = bb.add(HGate(), q=term)
        (anc,) = bb.add(HGate(), q=anc)
        (alpha,) = bb.add(HGate(), q=alpha)

        out = {
            'l': l,
            'two_body': two_body,
            'Q': Q,
            'k': k,
            'succ': succ,
            'succ_kpq': succ_kpq,
            'ReIm': re_im,
            'p': p,
            'q': q,
            'AB': ab,
            'term': term,
            'anc': anc,
            'alpha': alpha,
            'psia': psia,
            'psib': psib,
        }
        return out
