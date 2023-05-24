import itertools
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
)

import attrs
import IPython.display
import networkx as nx
import pydot
import sympy
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.cirq_conversion import CirqGateAsBloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters, Side
from cirq_qubitization.quantum_graph.quantum_graph import (
    BloqInstance,
    Connection,
    DanglingT,
    LeftDangle,
    RightDangle,
    Soquet,
)
from cirq_qubitization.quantum_graph.util_bloqs import ArbitraryClifford, Join, Split


def big_O(expr):
    if isinstance(expr, (int, float)):
        return sympy.Order(expr)
    return sympy.Order(expr, *[(var, sympy.oo) for var in expr.free_symbols])


class SympySymbolAllocator:
    """Issue placeholder sympy symbols."""

    def __init__(self):
        self._idxs: Dict[str, int] = defaultdict(lambda: 0)

    def new_symbol(self, prefix: str) -> sympy.Symbol:
        s = sympy.Symbol(f'_{prefix}{self._idxs[prefix]}')
        self._idxs[prefix] += 1
        return s


def _cbloq_bloq_counts(cbloq: CompositeBloq):
    counts: Dict[Bloq, int] = defaultdict(lambda: 0)
    for binst in cbloq.bloq_instances:
        counts[binst.bloq] += 1

    return {(n, bloq) for bloq, n in counts.items()}


def _descend_counts(
    parent: Bloq,
    g: nx.DiGraph,
    ss: SympySymbolAllocator,
    canon: Callable[[Bloq], Optional[Bloq]],
    keep: Sequence[Bloq],
) -> Dict[Bloq, int]:
    g.add_node(parent)
    if parent in keep:
        return {parent: 1}
    try:
        count_decomp = parent.rough_decompose(ss)
    except NotImplementedError:
        return {parent: 1}

    sigma: Dict[Bloq, int] = defaultdict(lambda: 0)
    for n, child in count_decomp:
        child = canon(child)
        if child is None:
            continue

        if (parent, child) in g.edges:
            g.edges[parent, child]['n'] += n
        else:
            g.add_edge(parent, child, n=n)

        child_counts = _descend_counts(child, g, ss, canon, keep)
        for k in child_counts.keys():
            before = sigma[k]
            sigma[k] += child_counts[k] * n
            after = sigma[k]
            # print(parent, child, k, before, child_counts[k], after)
    return dict(sigma)


def get_counts_graph(
    bloq: Bloq,
    ss: SympySymbolAllocator = None,
    canon: Callable[[Bloq], Optional[Bloq]] = None,
    keep=None,
) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
    if ss is None:
        ss = SympySymbolAllocator()
    if keep is None:
        keep = []
    if canon is None:
        canon = lambda b: b

    g = nx.DiGraph()
    bloq = canon(bloq)
    if bloq is None:
        raise ValueError()
    sigma = _descend_counts(bloq, g, ss, canon, keep)
    return g, sigma


def get_shor_counts_graph_1(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
    from cirq_qubitization.bloq_algos.basic_gates import IntState
    from cirq_qubitization.bloq_algos.shors.shors import CtrlScaleModAdd

    ss = SympySymbolAllocator()
    n_c = ss.new_symbol('n_c')

    def canon(bloq: Bloq) -> Optional[Bloq]:
        if isinstance(bloq, ArbitraryClifford):
            return attrs.evolve(bloq, n=n_c)

        return bloq

    return get_counts_graph(bloq, ss, canon)


def get_shor_counts_graph_2(bloq: Bloq) -> Tuple[nx.DiGraph, Dict[Bloq, int]]:
    from cirq_qubitization.bloq_algos.basic_gates import IntState
    from cirq_qubitization.bloq_algos.shors.shors import CtrlScaleModAdd

    ss = SympySymbolAllocator()
    n_c = ss.new_symbol('n_c')

    def canon(bloq: Bloq) -> Optional[Bloq]:
        if isinstance(bloq, ArbitraryClifford):
            return attrs.evolve(bloq, n=n_c)

        if isinstance(bloq, IntState):
            return None

        return bloq

    return get_counts_graph(bloq, ss, canon)


def print_counts_graph(g: nx.DiGraph):
    for b in nx.topological_sort(g):
        for succ in g.succ[b]:
            print(b, '--', g.edges[b, succ]['n'], '->', succ)


class GraphvizCounts:
    def __init__(self, g):
        self.g = g
        self._ids = {}
        self._i = 0

    def get_id(self, b: Bloq) -> int:
        if b in self._ids:
            return self._ids[b]
        new_id = self._i
        self._i += 1
        self._ids[b] = new_id
        return new_id

    def get_node_props(self, b: Bloq):
        label = [
            '<',
            f'{b.pretty_name().replace("<", "&lt;").replace(">", "&gt;")}<br />',
            f'<font face="monospace" point-size="10">{repr(b)}</font><br/>',
            '>',
        ]
        return {'label': ''.join(label), 'shape': 'rect'}

    def add_nodes(self, graph):
        b: Bloq
        for b in nx.topological_sort(self.g):
            graph.add_node(pydot.Node(self.get_id(b), **self.get_node_props(b)))

    def add_edges(self, graph):
        for b1, b2 in self.g.edges:
            n = self.g.edges[b1, b2]['n']
            label = f'{n}'
            label = sympy.printing.pretty(n)
            graph.add_edge(pydot.Edge(self.get_id(b1), self.get_id(b2), label=label))

    def get_graph(self):
        graph = pydot.Dot('my_graph', graph_type='digraph', rankdir='TB')
        self.add_nodes(graph)
        self.add_edges(graph)
        return graph

    def get_svg_bytes(self) -> bytes:
        """Get the SVG code (as bytes) for drawing the graph."""
        return self.get_graph().create_svg()

    def get_svg(self) -> IPython.display.SVG:
        """Get an IPython SVG object displaying the graph."""
        return IPython.display.SVG(self.get_svg_bytes())


def markdown_bloq_expr(bloq: Bloq, expr: sympy.Expr):
    return f'`{bloq}`: {expr._repr_latex_()}'


def markdown_counts_graph(graph: nx.DiGraph):
    m = ""
    for bloq in nx.topological_sort(graph):
        if not graph.succ[bloq]:
            continue
        m += f' - `{bloq}`\n'
        for succ in graph.succ[bloq]:
            expr = sympy.sympify(graph.edges[bloq, succ]['n'])
            m += f'   - `{succ}`: {expr._repr_latex_()}\n'

    return IPython.display.Markdown(m)
