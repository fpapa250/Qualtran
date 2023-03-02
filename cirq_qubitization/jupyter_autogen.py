"""Autogeneration of stub Jupyter notebooks.

For each module listed in the `NOTEBOOK_SPECS` global variable (in this file)
we write a notebook with a title, module docstring,
standard imports, and information on each `GateWithRegisters` listed in the
`gate_specs` field. For each gate, we render a docstring and diagrams.

## Adding a new gate.

 1. Create a new function in `jupyter_autogen_factories.py` that takes no arguments
    and returns an instance of the desired gate.
 2. If this is a new module: add a new key/value pair to the NOTEBOOK_SPECS global variable
    in this file. The key should be the name of the module with a `NotebookSpec` value. See
    the docstring for `NotebookSpec` for more information.
 3. Update the `NotebookSpec` `gate_specs` field to include a `GateNbSpec` for your new gate.
    Provide your factory function from step (1).

## Autogen behavior.

Each autogenerated notebook cell is tagged so we know it was autogenerated. Each time
this script is re-run, these cells will be re-rendered. *Modifications to generated _cells_
will not be persisted*.

If you add additional cells to the notebook it will *preserve them* even when this script is
re-run

Usage as a script:
    cd cirq_qubitization/ && python jupyter_autogen.py
"""

import dataclasses
import inspect
import re
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Tuple, Type, Union

import nbformat
from sphinx.ext.napoleon import Config, GoogleDocstring

import cirq_qubitization.algos.and_bloq_test
import cirq_qubitization.algos.basic_gates.cnot_test
import cirq_qubitization.algos.basic_gates.x_basis_test
import cirq_qubitization.jupyter_autogen_factories as jaf
import cirq_qubitization.quantum_graph
from cirq_qubitization.gate_with_registers import GateWithRegisters
from cirq_qubitization.quantum_graph.bloq import Bloq


@dataclasses.dataclass
class GateNbSpec:
    """Notebook specification for a particular Gate.

    Attributes:
        factory: A factory function that produces the gate. Its source code will be rendered
            to the notebook. See the `jupyter_autogen_factories` module.
        draw_vertical: Whether to render `vertical=True` in the rendered call to
            `display_gate_and_compilation`
    """

    factory: Callable[[], GateWithRegisters]
    draw_vertical: bool = False

    @property
    def cqid(self):
        """A cirq_qubitization-unique id for this `GateNbSpec` to tag generated cells."""
        return self.factory.__name__

    @property
    def gate_cls(self):
        gate_obj = self.factory()
        return gate_obj.__class__


@dataclasses.dataclass
class BloqNbSpec:
    """Notebook specification for a particular Bloq.

    Attributes:
        factory: A factory function that produces the bloq. Its source code will be rendered
            to the notebook. See the `jupyter_autogen_factories_2` module.
    """

    factory: Callable[[], Bloq]

    @property
    def cqid(self):
        """A cirq_qubitization-unique id for this `BloqNbSpec` to tag generated cells."""
        return self.factory.__name__

    @property
    def gate_cls(self):
        gate_obj = self.factory()
        return gate_obj.__class__


@dataclasses.dataclass
class NotebookSpec:
    """Specification for rendering a jupyter notebook for a given module.

    Attributes:
        title: The title of the notebook
        module: The module it documents. This is used to render the module docstring
            at the top of the notebook.
        gate_specs: A list of gate or bloq specs.
    """

    title: str
    module: ModuleType
    gate_specs: List[Union[GateNbSpec, BloqNbSpec]]
    directory: str = '.'


NOTEBOOK_SPECS: Dict[str, NotebookSpec] = {
    'apply_gate_to_lth_target': NotebookSpec(
        title='Apply to L-th Target',
        module=cirq_qubitization.apply_gate_to_lth_target,
        gate_specs=[GateNbSpec(jaf._make_ApplyGateToLthQubit)],
    ),
    'qrom': NotebookSpec(
        title='QROM', module=cirq_qubitization.qrom, gate_specs=[GateNbSpec(jaf._make_QROM)]
    ),
    'swap_network': NotebookSpec(
        title='Swap Network',
        module=cirq_qubitization.swap_network,
        gate_specs=[
            GateNbSpec(jaf._make_MultiTargetCSwap),
            GateNbSpec(jaf._make_MultiTargetCSwapApprox),
        ],
    ),
    'generic_select': NotebookSpec(
        title='Generic Select',
        module=cirq_qubitization.generic_select,
        gate_specs=[GateNbSpec(jaf._make_GenericSelect, draw_vertical=True)],
    ),
    'generic_subprepare': NotebookSpec(
        title='Subprepare',
        module=cirq_qubitization.generic_subprepare,
        gate_specs=[GateNbSpec(jaf._make_GenericSubPrepare)],
    ),
    'basic_gates': NotebookSpec(
        title='Basic Gates',
        module=cirq_qubitization.algos.basic_gates,
        gate_specs=[
            BloqNbSpec(cirq_qubitization.algos.basic_gates.cnot_test._make_CNOT),
            BloqNbSpec(cirq_qubitization.algos.basic_gates.plus_state_test._make_plus_state),
        ],
        directory='./algos',
    ),
    'and_bloq': NotebookSpec(
        title='And',
        module=cirq_qubitization.algos.and_bloq,
        gate_specs=[BloqNbSpec(cirq_qubitization.algos.and_bloq_test._make_and)],
        directory='./algos',
    ),
}


class _GoogleDocstringToMarkdown(GoogleDocstring):
    """Subclass of sphinx's parser to emit Markdown from Google-style docstrings."""

    def _parse_parameters_section(self, section: str) -> List[str]:
        """Sphinx method to emit a 'Parameters' section."""

        def _template(name, desc_lines):
            desc = ' '.join(desc_lines)
            return f' - `{name}`: {desc}'

        return [
            '#### Parameters',
            *[_template(name, desc) for name, _type, desc in self._consume_fields()],
            '',
        ]

    def _parse_references_section(self, section: str) -> List[str]:
        """Sphinx method to emit a 'References' section."""
        return [
            '#### References',
            ' '.join(line.strip() for line in self._consume_to_next_section()),
            '',
        ]


def get_markdown_docstring_lines(cls: Type) -> List[str]:
    """From a class `cls`, return its docstring as Markdown."""

    # 1. Sphinx incantation
    config = Config()
    docstring = cls.__doc__ if cls.__doc__ else ""
    gds = _GoogleDocstringToMarkdown(inspect.cleandoc(docstring), config=config, what='class')

    # 2. Pre-pend a header.
    lines = [f'## `{cls.__name__}`'] + gds.lines()

    # 3. Substitute restructured text inline-code blocks to markdown-style backticks.
    lines = [re.sub(r':py:func:`(\w+)`', r'`\1`', line) for line in lines]

    return lines


def _get_lines_for_constructing_an_object(func: Callable):
    """Parse out the source code from a factory function, so we can render it into a cell.

    Args:
        func: The factory function. Its definition must be one line; its body must be
            indented with four spaces; and it must end with a top-level return statement that
            is one line.

    Returns:
        trimmed_lines: The un-indented body of the function without the return statement.
        obj_expression: The expression used in the terminal `return` statement.
    """
    def_line, *pre, ret_line = inspect.getsource(func).splitlines()
    assert def_line.startswith('def '), def_line

    trimmed_lines = []
    for line in pre:
        assert line == '' or line.startswith(' ' * 4), line
        trimmed_lines.append(line[4:])
    assert ret_line.startswith('    return '), ret_line
    obj_expression = ret_line[len('    return ') :]
    return trimmed_lines, obj_expression


_IMPORTS = """\
import cirq
import numpy as np
import cirq_qubitization
import cirq_qubitization.testing as cq_testing
from cirq_qubitization.jupyter_tools import display_gate_and_compilation, show_bloq
from typing import *\
"""

_GATE_DISPLAY = """\
{lines}
g = cq_testing.GateHelper(
    {obj_expression}
)

display_gate_and_compilation(g{vert_str})\
"""

_BLOQ_DISPLAY = """\
{lines}
bloq = {obj_expression}
show_bloq(bloq)\
"""


def _get_code_for_demoing_a_gate(gate_func: Callable, vertical: bool) -> str:
    """Render the Python code for constructing and visualizing a Gate.

    This renders into the `_GATE_DISPLAY` template.
    """
    lines, obj_expression = _get_lines_for_constructing_an_object(gate_func)

    vert_str = ''
    if vertical:
        vert_str = ', vertical=True'

    return _GATE_DISPLAY.format(
        lines='\n'.join(lines), obj_expression=obj_expression, vert_str=vert_str
    )


def _get_code_for_demoing_a_bloq(bloq_func: Callable) -> str:
    """Render the Python code for constructing and visualizing a Bloq.

    This renders into the `_BLOQ_DISPLAY` template.
    """
    lines, obj_expression = _get_lines_for_constructing_an_object(bloq_func)

    return _BLOQ_DISPLAY.format(lines='\n'.join(lines), obj_expression=obj_expression)


def _get_code_for_demoing(spec: Union[GateNbSpec, BloqNbSpec]) -> str:
    if isinstance(spec, GateNbSpec):
        return _get_code_for_demoing_a_gate(gate_func=spec.factory, vertical=spec.draw_vertical)
    if isinstance(spec, BloqNbSpec):
        return _get_code_for_demoing_a_bloq(bloq_func=spec.factory)


def _get_title_cell_with_module_docstring(title: str, mod: ModuleType) -> str:
    """Return markdown text for the title cell.

    This consists of the specified title as well as the associated module's docstring.
    """
    lines = [f'# {title}', '']

    if mod.__doc__ is None:
        return lines[0]

    lines += inspect.cleandoc(mod.__doc__).splitlines()
    return '\n'.join(lines)


_K_CQ_AUTOGEN = 'cq.autogen'
"""The jupyter metadata key we use to identify cells we've autogenerated."""


def _md_cell(source: str, cqid: str) -> nbformat.NotebookNode:
    """Helper function to return a markdown cell with correct metadata"""
    return nbformat.v4.new_markdown_cell(source, metadata={_K_CQ_AUTOGEN: cqid})


def _code_cell(source: str, cqid: str) -> nbformat.NotebookNode:
    """Helper function to return a code cell with correct metadata"""
    return nbformat.v4.new_code_cell(source, metadata={_K_CQ_AUTOGEN: cqid})


@dataclasses.dataclass
class _GateCells:
    """Rendered cells for a gate."""

    md: nbformat.NotebookNode
    py: nbformat.NotebookNode


@dataclasses.dataclass
class NbCells:
    """Rendered notebook cells.

    Attributes:
        title_cell: The title cell
        top_imports: The cell for doing imports at the top of the notebook
        gate_cells: A mapping from "cqid" to the `_GateCells` pair of cells for each
            gate.
    """

    title_cell: nbformat.NotebookNode
    top_imports: nbformat.NotebookNode
    gate_cells: Dict[str, _GateCells]

    def get_all_cqids(self) -> List[str]:
        """Get all the cqid's for generated cells.

        These are ordered in the way the cells should appear in a rendered notebook.
        It is a flat list that includes both notebook-level cells and all the cqids
        for the `GateCells` in `self.gate_cells`.
        """
        cqids = ['title_cell', 'top_imports']
        for gate_cqid in self.gate_cells.keys():
            cqids.append(f'{gate_cqid}.md')
            cqids.append(f'{gate_cqid}.py')
        return cqids

    def get_cell_from_cqid(self, cqid: str) -> nbformat.NotebookNode:
        """Look up the cell from its cqid metadata string.

        For `_GateCells` this will look up the constituent `md` or `py` cell.
        """
        if '.' in cqid:
            gate_cqid, ext = cqid.split('.')
            return getattr(self.gate_cells[gate_cqid], ext)

        return getattr(self, cqid)


def render_notebook_cells(nbspec: NotebookSpec) -> NbCells:
    """Generate cells for a given notebook."""

    return NbCells(
        title_cell=_md_cell(
            _get_title_cell_with_module_docstring(title=nbspec.title, mod=nbspec.module),
            cqid='title_cell',
        ),
        top_imports=_code_cell(_IMPORTS, cqid='top_imports'),
        gate_cells={
            gspec.cqid: _GateCells(
                md=_md_cell(
                    '\n'.join(get_markdown_docstring_lines(cls=gspec.gate_cls)),
                    cqid=f'{gspec.cqid}.md',
                ),
                py=_code_cell(_get_code_for_demoing(gspec), cqid=f'{gspec.cqid}.py'),
            )
            for gspec in nbspec.gate_specs
        },
    )


def _init_notebook(
    modname: str, overwrite=False, directory: str = '.'
) -> Tuple[nbformat.NotebookNode, Path]:
    """Initialize a jupyter notebook.

    If one already exists: load it in. Otherwise, create a new one.

    Args:
        modname: The name used to find the notebook if it exists.
        overwrite: If set, remove any existing notebook and start from scratch.
    """

    nb_path = Path(f'{directory}/{modname}.ipynb')

    if overwrite:
        nb_path.unlink(missing_ok=True)

    if nb_path.exists():
        with nb_path.open('r') as f:
            return nbformat.read(f, as_version=4), nb_path

    nb = nbformat.v4.new_notebook()
    nb['metadata'].update(
        {
            'kernelspec': {'language': 'python', 'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'},
        }
    )
    return nb, nb_path


def render_notebooks():
    for modname, nbspec in NOTEBOOK_SPECS.items():
        # 1. get a notebook (existing or empty)
        nb, nb_path = _init_notebook(modname=modname, directory=nbspec.directory)

        # 2. Render all the cells we can render
        cells = render_notebook_cells(nbspec)

        # 3. Merge rendered cells into the existing notebook.
        #     -> we use the cells metadata field to match up cells.
        cqids_to_render: List[str] = cells.get_all_cqids()
        for i in range(len(nb.cells)):
            cell = nb.cells[i]
            if _K_CQ_AUTOGEN in cell.metadata:
                cqid: str = cell.metadata[_K_CQ_AUTOGEN]
                print(f"[{modname}] Replacing {cqid} cell.")
                new_cell = cells.get_cell_from_cqid(cqid)
                new_cell.id = cell.id  # keep id from existing cell
                nb.cells[i] = new_cell
                cqids_to_render.remove(cqid)

        # 4. Any rendered cells that weren't already there, append.
        for cqid in cqids_to_render:
            print(f"[{modname}] Adding {cqid}")
            new_cell = cells.get_cell_from_cqid(cqid)
            nb.cells.append(new_cell)

        # 5. Write the notebook.
        with nb_path.open('w') as f:
            nbformat.write(nb, f)


if __name__ == '__main__':
    render_notebooks()
