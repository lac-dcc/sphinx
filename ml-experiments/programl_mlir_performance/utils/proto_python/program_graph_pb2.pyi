import features_pb2 as _features_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Edge(_message.Message):
    __slots__ = ["features", "flow", "position", "source", "target"]
    class Flow(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    CALL: Edge.Flow
    CONTROL: Edge.Flow
    DATA: Edge.Flow
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    FLOW_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TYPE: Edge.Flow
    features: _features_pb2.Features
    flow: Edge.Flow
    position: int
    source: int
    target: int
    def __init__(self, flow: _Optional[_Union[Edge.Flow, str]] = ..., position: _Optional[int] = ..., source: _Optional[int] = ..., target: _Optional[int] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...

class Function(_message.Message):
    __slots__ = ["features", "module", "name"]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    features: _features_pb2.Features
    module: int
    name: str
    def __init__(self, name: _Optional[str] = ..., module: _Optional[int] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...

class Module(_message.Message):
    __slots__ = ["features", "name"]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    features: _features_pb2.Features
    name: str
    def __init__(self, name: _Optional[str] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ["block", "features", "function", "text", "type"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BLOCK_FIELD_NUMBER: _ClassVar[int]
    CONSTANT: Node.Type
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION: Node.Type
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TYPE: Node.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VARIABLE: Node.Type
    block: int
    features: _features_pb2.Features
    function: int
    text: str
    type: Node.Type
    def __init__(self, type: _Optional[_Union[Node.Type, str]] = ..., text: _Optional[str] = ..., function: _Optional[int] = ..., block: _Optional[int] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...

class ProgramGraph(_message.Message):
    __slots__ = ["edge", "features", "function", "module", "node"]
    EDGE_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    edge: _containers.RepeatedCompositeFieldContainer[Edge]
    features: _features_pb2.Features
    function: _containers.RepeatedCompositeFieldContainer[Function]
    module: _containers.RepeatedCompositeFieldContainer[Module]
    node: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, node: _Optional[_Iterable[_Union[Node, _Mapping]]] = ..., edge: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., function: _Optional[_Iterable[_Union[Function, _Mapping]]] = ..., module: _Optional[_Iterable[_Union[Module, _Mapping]]] = ..., features: _Optional[_Union[_features_pb2.Features, _Mapping]] = ...) -> None: ...
