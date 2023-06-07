import contextlib
import functools
import inspect
import os
import posixpath
import subprocess
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nbformat
from attrs import frozen
from tensorflow_docs.api_generator.config import ParserConfig
from tensorflow_docs.api_generator.doc_controls import should_skip_class_attr
from tensorflow_docs.api_generator.doc_generator_visitor import ApiTreeNode
from tensorflow_docs.api_generator.generate_lib import DocGenerator
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.parser import (
    DocstringInfo,
    documentation_path,
    FileLocation,
    get_defined_in,
    get_defining_class,
    parse_md_docstring,
    TitleBlock,
)
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
    MemberInfo,
    small_source_link,
)
from tensorflow_docs.api_generator.pretty_docs.class_page import _build_method_section, MethodInfo
from tensorflow_docs.api_generator.pretty_docs.docs_for_object import docs_for_object
from tensorflow_docs.api_generator.public_api import local_definitions_filter
from tensorflow_docs.api_generator.signature import (
    extract_decorators,
    FormatArguments,
    generate_signature,
    get_method_type,
    TfSignature,
)

from cirq_qubitization.jupyter_autogen import _code_cell, _init_notebook, _md_cell

EMPTY = inspect.Signature.empty


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
    def collect_docs(self):
        # call the super method to build all the fields of ClassPageInfo
        super().collect_docs()

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


def get_git_root() -> Path:
    """Get the root git repository path."""
    cp = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], capture_output=True, universal_newlines=True
    )
    path = Path(cp.stdout.strip()).absolute()
    assert path.exists()
    print('git root', path)
    return path


def render_reference_docs():
    import cirq_qubitization.quantum_graph

    reporoot = get_git_root()

    # This is the ordinary main entry point on which we call `build`. Here, we use it
    # just to walk the AST and give us a `parser_config`.
    doc_generator = DocGenerator(
        root_title="Qualtran",
        py_modules=[("cirq_qubitization.quantum_graph", cirq_qubitization.quantum_graph)],
        code_url_prefix="https://github.com/quantumlib/cirq-qubitization/blob/master/",
        base_dir=str(reporoot / 'cirq_qubitization/quantum_graph'),
        page_builder_classes=my_page_builders,
        callbacks=[local_definitions_filter, filter_type_checking],
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


@contextlib.contextmanager
def with_rel_prefix_path(n: ApiTreeNode, parser_config: ParserConfig):
    relative_path = os.path.relpath(
        path='.', start=os.path.dirname(documentation_path(n.full_name)) or '.'
    )

    # Convert from OS-specific path to URL/POSIX path.
    relative_path = posixpath.join(*relative_path.split(os.path.sep))

    with parser_config.reference_resolver.temp_prefix(relative_path):
        yield


def page_info_logic(
    n: ApiTreeNode, parser_config: ParserConfig
) -> Tuple[DocstringInfo, List[str], Optional[FileLocation]]:
    with with_rel_prefix_path(n, parser_config):
        doc = parse_md_docstring(n.py_object, n.full_name, parser_config)

        # self.collect_docs()

        aliases = ['.'.join(alias) for alias in n.aliases]
        if n.full_name in aliases:
            aliases.remove(n.full_name)

        defined_in = get_defined_in(n.py_object, parser_config)
        x: TitleBlock

        # page_text = self.build()
        return doc, aliases, defined_in


def class_collect_bases(n: ApiTreeNode, parser_config: ParserConfig) -> List[MemberInfo]:
    with with_rel_prefix_path(n, parser_config):
        bases = []
        for base in n.py_object.__mro__[1:]:
            base_api_node = parser_config.api_tree.node_for_object(base)
            if base_api_node is None:
                continue
            base_full_name = base_api_node.full_name
            base_doc = parse_md_docstring(base, base_full_name, parser_config)
            base_url = parser_config.reference_resolver.reference_to_url(base_full_name)

            link_info = MemberInfo(
                short_name=base_full_name.split('.')[-1],
                full_name=base_full_name,
                py_object=base,
                doc=base_doc,
                url=base_url,
            )
            bases.append(link_info)

        return bases


def member_info_to_method_info(member_info: MemberInfo, parser_config: ParserConfig) -> MethodInfo:
    defined_in = get_defined_in(member_info.py_object, parser_config)
    decorators = extract_decorators(member_info.py_object)
    py_obj = member_info.py_object
    func_type = get_method_type(py_obj, member_info.short_name, is_dataclass=False)
    signature = generate_signature(py_obj, parser_config, func_type=func_type)
    return MethodInfo.from_member_info(member_info, signature, decorators, defined_in)


def filter_and_sort_members(py_object, members):
    ordering = {name: i for i, name in enumerate(py_object.__dict__.keys())}
    fmembs = [memb for memb in members if memb.short_name in ordering]
    return sorted(fmembs, key=lambda m: ordering[m.short_name])


def class_collect_members(
    n: ApiTreeNode, parser_config: ParserConfig
) -> Tuple[List[MethodInfo], List[MemberInfo]]:
    members = []
    with with_rel_prefix_path(n, parser_config):
        class_path_node = parser_config.path_tree[n.path]
        for _, path_node in sorted(class_path_node.children.items()):
            # Don't document anything that is defined in object or by protobuf.
            defining_class = get_defining_class(n.py_object, path_node.short_name)
            if defining_class in [object, type, tuple, BaseException, Exception]:
                continue

            # The following condition excludes most protobuf-defined symbols.
            if defining_class and defining_class.__name__ in ['CMessage', 'Message', 'MessageMeta']:
                continue

            if should_skip_class_attr(n.py_object, path_node.short_name):
                continue

            child_doc = parse_md_docstring(path_node.py_object, n.full_name, parser_config)
            child_url = parser_config.reference_resolver.reference_to_url(path_node.full_name)

            member_info = MemberInfo(
                path_node.short_name, path_node.full_name, path_node.py_object, child_doc, child_url
            )

            members.append(member_info)

    sfmembs = filter_and_sort_members(n.py_object, members)

    methods = []
    properties = []
    for memb in sfmembs:
        if ObjType.get(memb.py_object) is ObjType.PROPERTY:
            properties.append(memb)
        else:
            methods.append(member_info_to_method_info(memb, parser_config))

    return methods, properties


def module_collect_members(api_node: ApiTreeNode, parser_config: ParserConfig):
    module_path_node = parser_config.path_tree[api_node.path]
    members = []
    for (_, path_node) in sorted(module_path_node.children.items()):
        member_doc = parse_md_docstring(path_node.py_object, api_node.full_name, parser_config)

        url = parser_config.reference_resolver.reference_to_url(path_node.full_name)

        member_info = MemberInfo(
            path_node.short_name, path_node.full_name, path_node.py_object, member_doc, url
        )
        members.append(member_info)
    return members


def render_part(part):
    if isinstance(part, str):
        return part
    elif isinstance(part, TitleBlock):
        return part.table_view()
    raise ValueError()


def property_doc(mi: MemberInfo) -> List[str]:
    return [f'#### `{mi.short_name}`\n', mi.doc.brief, ''] + [
        render_part(part) for part in mi.doc.docstring_parts
    ]


def method_doc(mi: MethodInfo) -> List[str]:
    return [
        f'#### `{mi.short_name}`\n',
        mi.doc.brief,
        '',
        f'<pre><code>\n{mi.short_name}{mi.signature}\n</pre></code>',
    ] + [render_part(part) for part in mi.doc.docstring_parts]


def class_doc(
    n: ApiTreeNode,
    doc: DocstringInfo,
    bases: Sequence[MemberInfo],
    methods: Sequence[MethodInfo],
    properties: Sequence[MemberInfo],
) -> List[str]:
    lines = [
        f'## `class {n.short_name}`\n',
        f'`{n.full_name}`\n',
        doc.brief,
        '',
        'Inherits from: ' + ', '.join([f'[{base.short_name}]({base.url})' for base in bases]),
    ] + [render_part(part) for part in doc.docstring_parts]

    lines.append('### Properties')
    for prop in properties:
        lines.extend(property_doc(prop))

    lines.append('')
    lines.append('### Methods')
    for meth in methods:
        lines.extend(method_doc(meth))

    return lines


if __name__ == '__main__':
    render_reference_docs()
