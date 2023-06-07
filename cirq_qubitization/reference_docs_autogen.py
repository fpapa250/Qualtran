import os
from typing import Optional, Sequence

import nbformat
from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.pretty_docs import (
    ClassPageInfo,
    FunctionPageInfo,
    ModulePageInfo,
    TypeAliasPageInfo,
)
from tensorflow_docs.api_generator.pretty_docs.base_page import (
    build_bottom_compat,
    build_signature,
    build_top_compat,
    DECORATOR_ALLOWLIST,
    format_docstring,
    small_source_link,
)
from tensorflow_docs.api_generator.pretty_docs.class_page import _build_method_section, MethodInfo
from tensorflow_docs.api_generator.pretty_docs.docs_for_object import (
    _DEFAULT_PAGE_BUILDER_CLASSES,
    docs_for_object,
)
from tensorflow_docs.api_generator.signature import TfSignature

from cirq_qubitization.jupyter_autogen import _code_cell, _init_notebook, _md_cell


def build_signature_mine(
    name: str, signature: TfSignature, decorators: Optional[Sequence[str]], type_alias: bool = False
) -> str:
    """Returns a markdown code block containing the function signature.

    Wraps the signature and limits it to 80 characters.

    Args:
      name: the name to put in the template.
      signature: the signature object.
      decorators: a list of decorators to apply.
      type_alias: If True, uses an `=` instead of `()` for the signature.
        For example: `TensorLike = (Union[str, tf.Tensor, int])`. Defaults to
          `False`.

    Returns:
      The signature of the object.
    """
    full_signature = str(signature)

    parts = []

    if decorators:
        parts.extend([f'@{dec}' for dec in decorators if dec in DECORATOR_ALLOWLIST])

    if type_alias:
        parts.append(f'{name} = {full_signature}')
    else:
        parts.append(f'{name}{full_signature}')

    return '\n'.join(parts)


def _build_method_section_parts(method_info: MethodInfo, heading_level=3):
    """Generates a markdown section for a method.

    Args:
      method_info: A `MethodInfo` object.
      heading_level: An Int, which HTML heading level to use.

    Returns:
      A markdown string.
    """
    parts1 = []
    heading = (
        '<h{heading_level} id="{short_name}">' '<code>{short_name}</code>' '</h{heading_level}>\n\n'
    )
    parts1.append(heading.format(heading_level=heading_level, **method_info._asdict()))

    if method_info.defined_in:
        parts1.append(small_source_link(method_info.defined_in))

    if method_info.signature is not None:
        sig = build_signature(
            name=method_info.short_name,
            signature=method_info.signature,
            decorators=method_info.decorators,
        )

    parts2 = []
    parts2.append(method_info.doc.brief + '\n')

    parts2.append(build_top_compat(method_info, h_level=4))

    for item in method_info.doc.docstring_parts:
        parts2.append(format_docstring(item, table_title_template=None, anchors=False))

    parts2.append(build_bottom_compat(method_info, h_level=4))

    return ''.join(parts1), sig, ''.join(parts2)


class MyClassPageInfo(ClassPageInfo):
    def docs_for_object(self):
        # call the super method to build all the fields of ClassPageInfo
        super().docs_for_object()

        # And turn our methods into markdown ourselves.
        methods: Sequence[MethodInfo] = self.methods
        ordering = {
            method_name: i for i, method_name in enumerate(self.api_node.py_object.__dict__.keys())
        }
        methods = sorted(methods, key=lambda m: ordering.get(m.short_name, len(ordering)))

        self.method_markdowns = []
        for method in methods:
            meth_md = _build_method_section(method)
            self.method_markdowns.append(meth_md)

        return self.build()

    def build(self) -> str:
        # This is required by PageInfo and would be important if we
        # were using the full doc generation flow as this sets the actual
        # output string. We only use `docs_for_object` and handle outputting ourselves.
        return '[no string output]'


class MyModulePageInfo(ModulePageInfo):
    def __init__(self, *, api_node, **kwargs):
        super().__init__(api_node=api_node, **kwargs)
        self.cells = []

    def docs_for_object(self):
        # call the super method to build all the fields
        super().docs_for_object()
        self.page = self.build()

    def write_to_nb(self, nb):
        pass


my_page_builders = {
    ObjType.CLASS: MyClassPageInfo,
    ObjType.CALLABLE: FunctionPageInfo,
    ObjType.MODULE: MyModulePageInfo,
    ObjType.TYPE_ALIAS: TypeAliasPageInfo,
}


def filter_type_checking(path, parent, children):
    return [(name, obj) for name, obj in children if name != 'TYPE_CHECKING']


def render_reference_docs():
    import cirq_qubitization.quantum_graph

    # This is the ordinary main entry point on which we call `build`. Here, we use it
    # just to walk the AST and give us a `parser_config`.
    doc_generator = DocGenerator(
        root_title="Qualtran",
        py_modules=[("cirq_qubitization.quantum_graph", cirq_qubitization.quantum_graph)],
        code_url_prefix='google.com',
        base_dir=os.path.dirname(__file__),
        page_builder_classes=my_page_builders,
        callbacks=[filter_type_checking],
    )
    parser_config = doc_generator.run_extraction()
    parser_config.reference_resolver._link_prefix = '/'  # TODO: what??

    # Reproduce the for loop that calls `docs_for_object` on the AST.
    # This is the first half of the normal `build` method. Then it outputs
    # rendered markdown. We handle that part ourselves.
    page_info = None
    for api_node in parser_config.api_tree.iter_nodes():
        if api_node.output_type() is api_node.OutputType.FRAGMENT:
            continue
        print(api_node)
        page_info = docs_for_object(
            api_node=api_node,
            parser_config=parser_config,
            extra_docs=None,
            search_hints=doc_generator._search_hints,
            page_builder_classes=my_page_builders,
        )
        print(page_info)

        # if isinstance(page_info, MyClassPageInfo):
        #     break

    sys.exit(1)
    # Ok, we've walked the AST. Organize it into a notebook using our tools.
    nb, nb_path = _init_notebook(basename='bloq', directory='./quantum_graph', overwrite=True)

    md = _md_cell("class docstring", cqid=f'bloq.md')
    nb.cells.append(md)

    for i, meth_doc in enumerate(page_info.method_markdowns):
        p1, sig, p2 = meth_doc
        md = _md_cell(p1, cqid=f'bloq.md.method{i}')
        nb.cells.append(md)
        nb.cells += [
            _md_cell(p1, cqid=f'bloq.md1.method{i}'),
            _code_cell(sig, cqid=f'bloq.py.method{i}'),
            _md_cell(p2, cqid=f'bloq.md2.method{i}'),
        ]

    with nb_path.open('w') as f:
        nbformat.write(nb, f)


if __name__ == '__main__':
    render_reference_docs()
