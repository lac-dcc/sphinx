import program_graph_pb2 as _program_graph_pb2
import features_pb2 as _features_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Ir(_message.Message):
    __slots__ = ["cmd", "compiler_version", "text", "type"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    CMD_FIELD_NUMBER: _ClassVar[int]
    COMPILER_VERSION_FIELD_NUMBER: _ClassVar[int]
    LLVM: Ir.Type
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: Ir.Type
    XLA_HLO: Ir.Type
    cmd: str
    compiler_version: int
    text: str
    type: Ir.Type
    def __init__(self, type: _Optional[_Union[Ir.Type, str]] = ..., compiler_version: _Optional[int] = ..., cmd: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class IrList(_message.Message):
    __slots__ = ["ir"]
    IR_FIELD_NUMBER: _ClassVar[int]
    ir: _containers.RepeatedCompositeFieldContainer[Ir]
    def __init__(self, ir: _Optional[_Iterable[_Union[Ir, _Mapping]]] = ...) -> None: ...

class NodeIndexList(_message.Message):
    __slots__ = ["node"]
    NODE_FIELD_NUMBER: _ClassVar[int]
    node: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, node: _Optional[_Iterable[int]] = ...) -> None: ...

class ProgramGraphFeatures(_message.Message):
    __slots__ = ["edge_features", "features", "function_features", "module_features", "node_features"]
    EDGE_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    MODULE_FEATURES_FIELD_NUMBER: _ClassVar[int]
    NODE_FEATURES_FIELD_NUMBER: _ClassVar[int]
    edge_features: _features_pb2.FeatureLists
    features: _features_pb2.Features
    function_features: _features_pb2.FeatureLists
    module_features: _features_pb2.FeatureLists
    node_features: _features_pb2.FeatureLists
    def __init__(self, node_features: _Optional[_Union[_features_pb2.FeatureLists, _Mapping]] = ..., edge_features: _Optional[_Union[_features_pb2.FeatureLists, _Mapping]] = ..., function_features: _Optional[_Union[_features_pb2.FeatureLists, _Mapping]] = ..., module_features: _Optional[_Union[_features_pb2.FeatureLists, _Mapping]] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...

class ProgramGraphFeaturesList(_message.Message):
    __slots__ = ["context", "graph"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    context: _features_pb2.Features
    graph: _containers.RepeatedCompositeFieldContainer[ProgramGraphFeatures]
    def __init__(self, context: _Optional[_Union[_features_pb2.Features, _Mapping]] = ..., graph: _Optional[_Iterable[_Union[ProgramGraphFeatures, _Mapping]]] = ...) -> None: ...

class ProgramGraphList(_message.Message):
    __slots__ = ["context", "graph"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    context: _features_pb2.Features
    graph: _containers.RepeatedCompositeFieldContainer[_program_graph_pb2.ProgramGraph]
    def __init__(self, context: _Optional[_Union[_features_pb2.Features, _Mapping]] = ..., graph: _Optional[_Iterable[_Union[_program_graph_pb2.ProgramGraph, _Mapping]]] = ...) -> None: ...

class ProgramGraphOptions(_message.Message):
    __slots__ = ["ignore_call_returns", "instructions_only", "ir_path", "opt_level", "strict"]
    IGNORE_CALL_RETURNS_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_ONLY_FIELD_NUMBER: _ClassVar[int]
    IR_PATH_FIELD_NUMBER: _ClassVar[int]
    OPT_LEVEL_FIELD_NUMBER: _ClassVar[int]
    STRICT_FIELD_NUMBER: _ClassVar[int]
    ignore_call_returns: bool
    instructions_only: bool
    ir_path: str
    opt_level: int
    strict: bool
    def __init__(self, strict: bool = ..., instructions_only: bool = ..., ignore_call_returns: bool = ..., opt_level: _Optional[int] = ..., ir_path: _Optional[str] = ...) -> None: ...

class Repo(_message.Message):
    __slots__ = ["created_ms_timestamp", "sha1", "url"]
    CREATED_MS_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SHA1_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    created_ms_timestamp: int
    sha1: str
    url: str
    def __init__(self, url: _Optional[str] = ..., sha1: _Optional[str] = ..., created_ms_timestamp: _Optional[int] = ...) -> None: ...

class SourceFile(_message.Message):
    __slots__ = ["language", "relpath", "text"]
    class Language(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    C: SourceFile.Language
    CXX: SourceFile.Language
    FORTRAN: SourceFile.Language
    HASKELL: SourceFile.Language
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    OPENCL: SourceFile.Language
    RELPATH_FIELD_NUMBER: _ClassVar[int]
    SWIFT: SourceFile.Language
    TEXT_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: SourceFile.Language
    language: SourceFile.Language
    relpath: str
    text: str
    def __init__(self, language: _Optional[_Union[SourceFile.Language, str]] = ..., relpath: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...
