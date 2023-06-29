"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import api.annotations_pb2
import api.args_pb2
import api.registers_pb2
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class Bloq(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    ARGS_FIELD_NUMBER: builtins.int
    REGISTERS_FIELD_NUMBER: builtins.int
    T_COMPLEXITY_FIELD_NUMBER: builtins.int
    name: builtins.str
    @property
    def args(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[api.args_pb2.BloqArg]: ...
    @property
    def registers(self) -> api.registers_pb2.Registers: ...
    @property
    def t_complexity(self) -> api.annotations_pb2.TComplexity: ...
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        args: collections.abc.Iterable[api.args_pb2.BloqArg] | None = ...,
        registers: api.registers_pb2.Registers | None = ...,
        t_complexity: api.annotations_pb2.TComplexity | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["registers", b"registers", "t_complexity", b"t_complexity"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["args", b"args", "name", b"name", "registers", b"registers", "t_complexity", b"t_complexity"]) -> None: ...

global___Bloq = Bloq

@typing_extensions.final
class CompositeBloq(google.protobuf.message.Message):
    """A composite bloq is a heirarchical definition in terms of other simpler bloqs."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TABLE_FIELD_NUMBER: builtins.int
    CBLOQ_FIELD_NUMBER: builtins.int
    T_COMPLEXITY_FIELD_NUMBER: builtins.int
    BLOQ_COUNTS_FIELD_NUMBER: builtins.int
    @property
    def table(self) -> global___BloqTable: ...
    @property
    def cbloq(self) -> global___CompositeBloqLite: ...
    @property
    def t_complexity(self) -> api.annotations_pb2.TComplexity: ...
    @property
    def bloq_counts(self) -> api.annotations_pb2.BloqCounts: ...
    def __init__(
        self,
        *,
        table: global___BloqTable | None = ...,
        cbloq: global___CompositeBloqLite | None = ...,
        t_complexity: api.annotations_pb2.TComplexity | None = ...,
        bloq_counts: api.annotations_pb2.BloqCounts | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bloq_counts", b"bloq_counts", "cbloq", b"cbloq", "t_complexity", b"t_complexity", "table", b"table"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bloq_counts", b"bloq_counts", "cbloq", b"cbloq", "t_complexity", b"t_complexity", "table", b"table"]) -> None: ...

global___CompositeBloq = CompositeBloq

@typing_extensions.final
class BloqInstance(google.protobuf.message.Message):
    """Messages to enable efficient description of CompositeBloq in terms of other simpler bloqs."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    BLOQ_ID_FIELD_NUMBER: builtins.int
    id: builtins.int
    bloq_id: builtins.int
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        bloq_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["bloq_id", b"bloq_id", "id", b"id"]) -> None: ...

global___BloqInstance = BloqInstance

@typing_extensions.final
class Soquet(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BLOQ_INSTANCE_FIELD_NUMBER: builtins.int
    DANGLING_T_FIELD_NUMBER: builtins.int
    REGISTER_FIELD_NUMBER: builtins.int
    INDEX_FIELD_NUMBER: builtins.int
    @property
    def bloq_instance(self) -> global___BloqInstance: ...
    dangling_t: builtins.str
    @property
    def register(self) -> api.registers_pb2.Register: ...
    @property
    def index(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        bloq_instance: global___BloqInstance | None = ...,
        dangling_t: builtins.str = ...,
        register: api.registers_pb2.Register | None = ...,
        index: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["binst", b"binst", "bloq_instance", b"bloq_instance", "dangling_t", b"dangling_t", "register", b"register"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["binst", b"binst", "bloq_instance", b"bloq_instance", "dangling_t", b"dangling_t", "index", b"index", "register", b"register"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["binst", b"binst"]) -> typing_extensions.Literal["bloq_instance", "dangling_t"] | None: ...

global___Soquet = Soquet

@typing_extensions.final
class Connection(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LEFT_FIELD_NUMBER: builtins.int
    RIGHT_FIELD_NUMBER: builtins.int
    @property
    def left(self) -> global___Soquet: ...
    @property
    def right(self) -> global___Soquet: ...
    def __init__(
        self,
        *,
        left: global___Soquet | None = ...,
        right: global___Soquet | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["left", b"left", "right", b"right"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["left", b"left", "right", b"right"]) -> None: ...

global___Connection = Connection

@typing_extensions.final
class CompositeBloqLite(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    REGISTERS_FIELD_NUMBER: builtins.int
    CONNECTIONS_FIELD_NUMBER: builtins.int
    name: builtins.str
    @property
    def registers(self) -> api.registers_pb2.Registers: ...
    @property
    def connections(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Connection]: ...
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        registers: api.registers_pb2.Registers | None = ...,
        connections: collections.abc.Iterable[global___Connection] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["registers", b"registers"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["connections", b"connections", "name", b"name", "registers", b"registers"]) -> None: ...

global___CompositeBloqLite = CompositeBloqLite

@typing_extensions.final
class BloqOrCbloq(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BLOQ_FIELD_NUMBER: builtins.int
    CBLOQ_FIELD_NUMBER: builtins.int
    @property
    def bloq(self) -> global___Bloq: ...
    @property
    def cbloq(self) -> global___CompositeBloqLite: ...
    def __init__(
        self,
        *,
        bloq: global___Bloq | None = ...,
        cbloq: global___CompositeBloqLite | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bloq", b"bloq", "bloq_or_cbloq", b"bloq_or_cbloq", "cbloq", b"cbloq"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bloq", b"bloq", "bloq_or_cbloq", b"bloq_or_cbloq", "cbloq", b"cbloq"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["bloq_or_cbloq", b"bloq_or_cbloq"]) -> typing_extensions.Literal["bloq", "cbloq"] | None: ...

global___BloqOrCbloq = BloqOrCbloq

@typing_extensions.final
class BloqTable(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    BLOQS_FIELD_NUMBER: builtins.int
    @property
    def bloqs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___BloqOrCbloq]:
        """A lookup table for all unique Bloqs."""
    def __init__(
        self,
        *,
        bloqs: collections.abc.Iterable[global___BloqOrCbloq] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["bloqs", b"bloqs"]) -> None: ...

global___BloqTable = BloqTable
