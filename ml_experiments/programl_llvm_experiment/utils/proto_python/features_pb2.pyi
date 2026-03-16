from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BytesList(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, value: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Feature(_message.Message):
    __slots__ = ["bytes_list", "float_list", "int64_list"]
    BYTES_LIST_FIELD_NUMBER: _ClassVar[int]
    FLOAT_LIST_FIELD_NUMBER: _ClassVar[int]
    INT64_LIST_FIELD_NUMBER: _ClassVar[int]
    bytes_list: BytesList
    float_list: FloatList
    int64_list: Int64List
    def __init__(self, bytes_list: _Optional[_Union[BytesList, _Mapping]] = ..., float_list: _Optional[_Union[FloatList, _Mapping]] = ..., int64_list: _Optional[_Union[Int64List, _Mapping]] = ...) -> None: ...

class FeatureList(_message.Message):
    __slots__ = ["feature"]
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    feature: _containers.RepeatedCompositeFieldContainer[Feature]
    def __init__(self, feature: _Optional[_Iterable[_Union[Feature, _Mapping]]] = ...) -> None: ...

class FeatureLists(_message.Message):
    __slots__ = ["feature_list"]
    class FeatureListEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: FeatureList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[FeatureList, _Mapping]] = ...) -> None: ...
    FEATURE_LIST_FIELD_NUMBER: _ClassVar[int]
    feature_list: _containers.MessageMap[str, FeatureList]
    def __init__(self, feature_list: _Optional[_Mapping[str, FeatureList]] = ...) -> None: ...

class Features(_message.Message):
    __slots__ = ["feature"]
    class FeatureEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Feature
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Feature, _Mapping]] = ...) -> None: ...
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    feature: _containers.MessageMap[str, Feature]
    def __init__(self, feature: _Optional[_Mapping[str, Feature]] = ...) -> None: ...

class FloatList(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, value: _Optional[_Iterable[float]] = ...) -> None: ...

class Int64List(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, value: _Optional[_Iterable[int]] = ...) -> None: ...
