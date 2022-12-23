import contextlib
import dataclasses
import inspect
import types
import typing
from collections import ChainMap
from dataclasses import dataclass
from enum import Enum

import typing_extensions
import yaml

T = typing.TypeVar("T")

# TODO: add datetime, timedelta, date, time
VALID_TYPES = (int, float, str, bool, types.NoneType, dict, list, tuple, set, typing.Union, typing.Optional)


def pretty_type_str(type_: typing.Any) -> str:  # noqa: C901
    """Pretty print a type.

    Args:
        type_: The type to pretty print.

    Returns:
        The pretty printed type.

    Example:

        >>> pretty_type_str(int)
        'int'
        >>> pretty_type_str(Union[int, str])
        'int | str'
        >>> pretty_type_str(Annotated[int, "Additional annotations."])
        'Annotated[int, "Additional annotations."]'
        >>> pretty_type_str(Optional[int])
        'int | None'
        >>> pretty_type_str(List[int])
        '[int]'
        >>> pretty_type_str(Dict[str, int])
        '{str: int}'
        >>> pretty_type_str(Tuple[int, str])
        '(int, str)'
        >>> pretty_type_str(Tuple[int, ...])
        '(int, ...)'
        >>> pretty_type_str(Set[int])
        '{int}'
    """
    # TODO: handle types without get_args
    origin_type = typing.get_origin(type_) or type_
    if origin_type == types.NoneType:  # noqa
        return "None"
    if origin_type == typing.Any:
        return "Any"
    if origin_type == typing_extensions.Annotated:
        return f"Annotated[{pretty_type_str(typing.get_args(type_)[0])}, {typing.get_args(type_)[1:]}]"
    if origin_type == typing.Union:
        if not typing.get_args(type_):
            return pretty_type_str(typing.Any)
        return " | ".join(pretty_type_str(t) for t in typing.get_args(type_))
    if origin_type == typing.Optional:
        return f"{pretty_type_str(typing.get_args(type_)[0])} | None"
    if origin_type == list:
        if not typing.get_args(type_) or typing.get_args(type_)[0] == typing.Any:
            return "list"
        return f"[{pretty_type_str(typing.get_args(type_)[0])}]"
    if origin_type == dict:
        if not typing.get_args(type_):
            return "dict"
        return f"{{{pretty_type_str(typing.get_args(type_)[0])}: {pretty_type_str(typing.get_args(type_)[1])}}}"
    if origin_type == tuple:
        if not typing.get_args(type_):
            return "tuple"
        return f"({', '.join(pretty_type_str(t) for t in typing.get_args(type_))})"
    if origin_type == set:
        if not typing.get_args(type_) or typing.get_args(type_)[0] == typing.Any:
            return "set"
        return f"{{{pretty_type_str(typing.get_args(type_)[0])}}}"
    return type_.__name__  # type: ignore


def get_type_and_metadata_entry(field: dataclasses.Field) -> dict[str, str | typing.Sequence]:
    # Support annotations
    if typing.get_origin(field.type) == typing_extensions.Annotated:
        type_ = typing.get_args(field.type)[0]
        entry = {"type": pretty_type_str(type_), "metadata": typing.get_args(field.type)[1:]}
    else:
        type_ = field.type
        entry = {"type": pretty_type_str(type_)}
    return entry


def is_valid_schema_type(type_: type) -> bool:
    return (
        dataclasses.is_dataclass(type_)
        or type_ in VALID_TYPES
        or typing.get_origin(type_) in VALID_TYPES
        or (isinstance(type_, type) and issubclass(type_, Enum))
    )


# Type alias for definitions
Definition = dict[str, typing.Any]


@dataclass
class DataclassesSchema:
    definitions: typing.MutableMapping[str, Definition] = dataclasses.field(default_factory=dict)

    @staticmethod
    def extend_parent(parent: "DataclassesSchema | None") -> "DataclassesSchema":
        if parent is None:
            return DataclassesSchema()

        definitions = ChainMap[str, Definition]({}, parent.definitions)
        return DataclassesSchema(definitions)

    def add_complex_type(self, type_: type, add_subclasses: bool = True) -> None:  # noqa: C901
        # Step through Annotated types
        origin_type = typing.get_origin(type_) or type_
        if origin_type == typing_extensions.Annotated:
            type_ = typing.get_args(type_)[0]
            origin_type = typing.get_origin(type_) or type_

        if type_.__name__ in self.definitions:
            return

        if dataclasses.is_dataclass(origin_type):
            self.add_dataclass_type(type_, add_subclasses=add_subclasses)
        # Check whether type has Enum as a base class
        elif isinstance(origin_type, type) and issubclass(origin_type, Enum):
            self.add_enum_type(origin_type)
        elif origin_type == typing.Union:
            for t in typing.get_args(type_):
                self.add_complex_type(t, add_subclasses=add_subclasses)
        elif origin_type == typing.Optional:
            self.add_complex_type(typing.get_args(type_)[0], add_subclasses=add_subclasses)
        elif origin_type == list:
            if typing.get_args(type_):
                self.add_complex_type(typing.get_args(type_)[0], add_subclasses=add_subclasses)
        elif origin_type == dict:
            if typing.get_args(type_):
                self.add_complex_type(typing.get_args(type_)[0], add_subclasses=add_subclasses)
                self.add_complex_type(typing.get_args(type_)[1], add_subclasses=add_subclasses)
        elif origin_type == tuple:
            for t in typing.get_args(type_):
                self.add_complex_type(t, add_subclasses=add_subclasses)
        elif origin_type == set:
            if typing.get_args(type_):
                self.add_complex_type(typing.get_args(type_)[0], add_subclasses=add_subclasses)
        elif not is_valid_schema_type(type_):
            raise ValueError(f"Unsupported type {type_}")

    def add_complex_value(self, value: object, add_subclasses: bool = False) -> None:
        if type(value).__name__ in self.definitions:
            return

        if isinstance(value, (list, tuple, set)):
            for v in value:
                self.add_complex_value(v, add_subclasses=add_subclasses)
        elif isinstance(value, dict):
            for k, v in value.items():
                self.add_complex_value(k, add_subclasses=add_subclasses)
                self.add_complex_value(v, add_subclasses=add_subclasses)
        elif isinstance(value, Enum):
            self.add_complex_type(type(value), add_subclasses=add_subclasses)
        elif dataclasses.is_dataclass(value):
            self.add_dataclass(value, add_subclasses=add_subclasses)
        elif not is_valid_schema_type(type(value)):
            raise ValueError(f"Unsupported value {value}")

    def add_enum_type(self, enum_type: type) -> None:
        assert issubclass(enum_type, Enum)
        self.definitions[enum_type.__name__] = {"type": "enum", "values": {member.name for member in enum_type}}

    def add_dataclass_type(
        self,
        dataclass_type: type,
        *,
        add_subclasses: bool = True,
    ) -> None:
        """Add a dataclass type to the schema.

        Recursively convert nested dataclasses and other complex types.

        This only uses static type information.

        Args:
            dataclass_type: The dataclass type to convert.
            add_subclasses: Whether to add subclasses to the schema.

        Example:

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> schema = DataclassesSchema(); schema.add_dataclass_type(Foo); schema.definitions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}

            >>> @dataclass
            ... class Bar:
            ...     c: Foo
            >>> schema = DataclassesSchema(); self.add_dataclass_type(Bar); schema.definitions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}, 'Bar': {'c': {'type': 'Foo'}}}

            >>> @dataclass
            ... class Baz:
            ...     d: Annotated[int, "Additional annotations."]
            >>> schema = DataclassesSchema(); self.add_dataclass_type(Baz); schema.definitions
            {'Baz': {'d': {'type': 'int', 'metadata': ['Additional annotations.']}}}
        """
        schema = self.definitions[dataclass_type.__name__] = {}

        field: dataclasses.Field
        for field in dataclasses.fields(dataclass_type):
            self.add_complex_type(field.type)
            entry = get_type_and_metadata_entry(field)
            schema[field.name] = entry

        self._process_class_hierarchy(dataclass_type, add_subclasses)

    def _process_class_hierarchy(self, dataclass_type: type, add_subclasses: bool) -> None:
        schema = self.definitions[dataclass_type.__name__]

        # Add base classes
        if dataclass_type.__bases__ != (object,):
            schema["bases"] = []
            for base_class in dataclass_type.__bases__:
                if base_class != object:
                    self.add_complex_type(base_class, add_subclasses=False)
                    schema["bases"].append(base_class.__name__)
        # Add subclasses
        if add_subclasses:
            for subclass in dataclass_type.__subclasses__():
                self.add_dataclass_type(subclass, add_subclasses=True)

    def add_dataclass(self, dataclass_instance: T, *, add_subclasses: bool = False) -> None:
        """Add a dataclass to the schema.

        Recursively add nested dataclasses and other complex types.

        This uses both static and runtime type information.

        Args:
            dataclass_instance: The dataclass type to convert.
            add_subclasses: Whether to add subclasses to the schema.

        Example:

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> schema = DataclassesSchema(); schema.add_dataclass(Foo(a=1, b="2")); schema.defintions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}


            >>> @dataclass
            ... class Bar:
            ...     c: Foo
            >>> schema = DataclassesSchema(); schema.add_dataclass(Bar); schema.definitions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}, 'Bar': {'c': {'type': 'Foo'}}}

            >>> @dataclass
            ... class Baz:
            ...     d: Annotated[int, "Additional annotations."]
            >>> schema = DataclassesSchema(); schema.add_dataclass(Baz); schema.definitions
            {'Baz': {'d': {'type': 'int', 'metadata': ['Additional annotations.']}}}

        """
        schema = self.definitions[dataclass_instance.__class__.__name__] = {}

        field: dataclasses.Field
        for field in dataclasses.fields(dataclass_instance):
            field_value = getattr(dataclass_instance, field.name)
            self.add_complex_type(field.type)
            if dataclasses.is_dataclass(field_value):
                if field.type.__name__ not in schema:
                    self.add_dataclass(field_value)
            else:
                # Try other complex runtime types.
                self.add_complex_type(type(field_value))
            schema[field.name] = get_type_and_metadata_entry(field)

        self._process_class_hierarchy(dataclass_instance.__class__, add_subclasses)

    def add_return_annotation(self, signature: inspect.Signature, *, add_subclasses: bool = True) -> None:
        """
        Add the return annotation to the schema.

        Args:
            signature: The signature to get the return annotation from.
            add_subclasses: Whether to add subclasses to the schema.

        Returns:
            The return schema.

        Example:
            >>> def foo(a: int, b: str) -> int:
            ...     pass
            >>> schema = DataclassesSchema(); schema.add_return_annotation(inspect.signature(foo)); schema.definitions
            {}
        """
        assert signature.return_annotation != inspect.Signature.empty, "Return annotation is required."
        assert is_valid_schema_type(
            signature.return_annotation
        ), f"Unsupported return type {signature.return_annotation}!"
        self.add_complex_type(signature.return_annotation, add_subclasses=add_subclasses)

    def add_signature(self, signature: inspect.Signature, *, add_subclasses: bool = True) -> None:
        """
        Add the signature to the schema.

        This only adds complex types using static type information.

        Args:
            signature: a function signature.
            add_subclasses: Whether to add subclasses to the schema.

        Returns:
            The schema.

        Example:
            >>> def foo(a: int, b: str) -> int:
            ...     pass

            >>> schema = DataclassesSchema()
            >>> schema.add_signature(inspect.signature(foo))
            >>> schema.definitions
            {}

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> schema = DataclassesSchema()
            >>> schema.add_signature(inspect.signature(foo))
            >>> schema.definitions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}
        """
        for parameter in signature.parameters.values():
            if parameter.annotation != inspect.Parameter.empty:
                self.add_complex_type(parameter.annotation, add_subclasses=add_subclasses)
        self.add_return_annotation(signature, add_subclasses=add_subclasses)

    def add_bound_arguments(
        self,
        bound_arguments: inspect.BoundArguments,
        *,
        add_subclasses: bool = False,
    ) -> None:
        """Add any dataclasses from BoundArguments to the schema.

        Args:
            bound_arguments: The bound arguments to add.
            add_subclasses: Whether to add subclasses to the schema.

        Example:

            >>> def foo(a: int, b: str) -> None:
            ...     pass
            >>> signature = inspect.signature(foo)
            >>> bound_arguments = signature.bind(1, "2")
            >>> schema = DataclassesSchema()
            >>> schema.add_bound_arguments_to_schema(bound_arguments, schema)
            >>> schema.definitions
            {}

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> def bar(c) -> None:
            ...     pass
            >>> signature = inspect.signature(bar)
            >>> bound_arguments = signature.bind(Foo(1, "2"))
            >>> schema = DataclassesSchema()
            >>> schema.add_bound_arguments_to_schema(bound_arguments, schema)
            >>> schema.definitions
            {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}}
        """
        for _, value in bound_arguments.arguments.items():
            self.add_complex_value(value, add_subclasses=add_subclasses)


def deserialize_yaml(decoded_yaml: object, type_: type) -> object:  # noqa: C901
    """
    Deserialize `decoded_yaml` into an instance of `type_`. If `decoded_yaml` is a dict, then
    `type_` can be a dataclass; if `decoded_yaml` is a list, then `type_` can be a
    list of dataclasses; etc.

    Args:
        decoded_yaml: The dict to deserialize.
        type_: The type of the dataclass.

    Returns:
        The deserialized object.

    Example:

            >>> @dataclass
            ... class Foo:
            ...     a: int
            ...     b: str
            >>> deserialize_yaml({'a': 1, 'b': "2"}, Foo, {})
            Foo(a=1, b='2')

            >>> @dataclass
            ... class Bar:
            ...     c: Foo
            >>> deserialize_yaml({'c': {'a': 1, 'b': "2"}}, Bar, {'Foo': {'a': {'type': 'int'}, 'b': {'type': 'str'}}})
            Bar(c=Foo(a=1, b='2'))
    """
    origin_type_ = typing.get_origin(type_) or type_
    if isinstance(origin_type_, types.NoneType):
        if decoded_yaml is None:
            return None
        else:
            raise ValueError(f"Expected None, but got {decoded_yaml} of type {type(decoded_yaml)}")
    if origin_type_ == typing.Union:
        union_args = typing.get_args(type_)
        for union_type in union_args:
            with contextlib.suppress(yaml.YAMLError):
                return deserialize_yaml(decoded_yaml, union_type)
        raise ValueError(f"Could not deserialize {decoded_yaml} into {origin_type_}.")
    if origin_type_ == list and typing.get_args(type_):
        assert isinstance(decoded_yaml, typing.Sequence), f"Expected sequence, but got {decoded_yaml}."
        assert len(typing.get_args(type_)) == 1, "Only one type argument is supported for lists."
        item_type = typing.get_args(type_)[0]
        return [deserialize_yaml(item, item_type) for item in decoded_yaml]
    if origin_type_ == tuple and typing.get_args(type_):
        assert isinstance(decoded_yaml, typing.Sequence), f"Expected sequence, but got {decoded_yaml}."
        assert len(decoded_yaml) == len(
            typing.get_args(type_)
        ), f"Expected {len(typing.get_args(type_))} items in tuple, but got {len(decoded_yaml)}."
        item_types = typing.get_args(type_)
        return tuple(deserialize_yaml(item, item_type) for item, item_type in zip(decoded_yaml, item_types))
    if origin_type_ == set and typing.get_args(type_):
        assert isinstance(decoded_yaml, typing.Sequence), f"Expected sequence, but got {decoded_yaml}."
        assert len(typing.get_args(type_)) == 1, "Only one type argument is supported for sets."
        item_type = typing.get_args(type_)[0]
        return {deserialize_yaml(item, item_type) for item in decoded_yaml}
    if origin_type_ == dict and typing.get_args(type_):
        assert isinstance(decoded_yaml, dict), f"Expected dict, but got {decoded_yaml}."
        assert len(typing.get_args(type_)) == 2, "Only two type arguments are supported for dicts."
        item_type = typing.get_args(type_)[1]
        return {key: deserialize_yaml(item, item_type) for key, item in decoded_yaml.items()}
    if isinstance(origin_type_, type) and issubclass(origin_type_, Enum):
        assert isinstance(decoded_yaml, str), f"Expected str, but got {decoded_yaml}."
        return origin_type_[decoded_yaml]
    if dataclasses.is_dataclass(origin_type_):
        assert isinstance(decoded_yaml, dict), f"Expected dict, but got {decoded_yaml} of type {type(decoded_yaml)}."
        kwargs = {}
        fields = dataclasses.fields(origin_type_)
        for field in fields:
            if field.name in decoded_yaml:
                kwargs[field.name] = deserialize_yaml(decoded_yaml[field.name], field.type)
        return origin_type_(**kwargs)

    # Just return the values.
    assert isinstance(decoded_yaml, origin_type_), f"Expected {origin_type_}, but got {decoded_yaml}."
    return decoded_yaml
