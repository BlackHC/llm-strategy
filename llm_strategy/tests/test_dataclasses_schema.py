import dataclasses
import inspect
import typing
from dataclasses import dataclass
from enum import Enum

import typing_extensions
import yaml

from llm_strategy.dataclasses_schema import (
    DataclassesSchema,
    deserialize_yaml,
    get_type_and_metadata_entry,
    is_valid_schema_type,
    pretty_type_str,
)


def test_is_valid_schema_type():
    assert is_valid_schema_type(int)
    assert is_valid_schema_type(float)
    assert is_valid_schema_type(str)
    assert is_valid_schema_type(bool)
    assert is_valid_schema_type(type(None))
    assert is_valid_schema_type(dict)
    assert is_valid_schema_type(list)
    assert is_valid_schema_type(tuple)
    assert is_valid_schema_type(set)

    assert is_valid_schema_type(typing.List)
    assert is_valid_schema_type(typing.Dict)
    assert is_valid_schema_type(typing.Tuple)
    assert is_valid_schema_type(typing.Set)

    assert is_valid_schema_type(typing.Optional[int])
    assert is_valid_schema_type(typing.Union[int, str])
    assert is_valid_schema_type(typing.Union[int, str, None])
    assert is_valid_schema_type(typing.Union[int, str, None, float])

    assert is_valid_schema_type(typing.List[int])
    assert is_valid_schema_type(typing.Dict[str, int])
    assert is_valid_schema_type(typing.Tuple[int, str])
    assert is_valid_schema_type(typing.Set[int])

    # classes (that are not dataclasses) don't work
    class A:
        pass

    assert not is_valid_schema_type(A)

    # dataclasses work
    @dataclass
    class B:
        pass

    assert is_valid_schema_type(B)

    # enums work
    class C(Enum):
        pass

    assert is_valid_schema_type(C)


def test_pretty_type():
    assert pretty_type_str(int) == "int"
    assert pretty_type_str(float) == "float"
    assert pretty_type_str(str) == "str"
    assert pretty_type_str(bool) == "bool"
    assert pretty_type_str(type(None)) == "None"
    assert pretty_type_str(dict) == "dict"
    assert pretty_type_str(list) == "list"
    assert pretty_type_str(tuple) == "tuple"
    assert pretty_type_str(set) == "set"

    # Check that the typing names without generics are the same as the built-in names
    assert pretty_type_str(typing.List) == "list"
    assert pretty_type_str(typing.Dict) == "dict"
    assert pretty_type_str(typing.Tuple) == "tuple"
    assert pretty_type_str(typing.Set) == "set"

    # check that we use the pretty type for the generics
    assert pretty_type_str(typing.List[int]) == "[int]"
    assert pretty_type_str(typing.Dict[str, int]) == "{str: int}"
    assert pretty_type_str(typing.Tuple[int, str]) == "(int, str)"
    assert pretty_type_str(typing.Set[int]) == "{int}"

    # check that we use the pretty type for the union
    assert pretty_type_str(typing.Optional[int]) == "int | None"
    assert pretty_type_str(typing.Union[int, str]) == "int | str"

    # check that we don't change dataclasses or enums
    @dataclass
    class A:
        pass

    assert pretty_type_str(A) == "A"

    class B(Enum):
        pass

    assert pretty_type_str(B) == "B"


def test_get_type_and_metadata_entry():
    # def get_type_and_metadata_entry(field) -> dict[str, typing.Any]:
    #     # Support annotations
    #     if typing.get_origin(field.type) == typing_extensions.Annotated:
    #         type_ = typing.get_args(field.type)[0]
    #         entry = {'type': pretty_type(type_), 'metadata': typing.get_args(field.type)[1:]}
    #     else:
    #         type_ = field.type
    #         entry = {'type': pretty_type(type_)}
    #     return entry
    @dataclass
    class A:
        a: int
        b: typing_extensions.Annotated[int, "a", "b"]

    assert get_type_and_metadata_entry(dataclasses.fields(A)[0]) == {"type": "int"}
    assert get_type_and_metadata_entry(dataclasses.fields(A)[1]) == {"type": "int", "metadata": ("a", "b")}


def test_schema():
    @dataclass
    class A:
        a: int
        b: str

    @dataclass
    class B:
        a: A
        b: typing.List[A]

    schema = DataclassesSchema()
    schema.add_dataclass_type(A)
    assert schema.definitions == {"A": {"a": {"type": "int"}, "b": {"type": "str"}}}

    schema.add_dataclass_type(B)
    assert schema.definitions == {
        "A": {"a": {"type": "int"}, "b": {"type": "str"}},
        "B": {"a": {"type": "A"}, "b": {"type": "[A]"}},
    }

    # enum
    class C(Enum):
        a = 1
        b = 2

    schema = DataclassesSchema()
    schema.add_enum_type(C)
    assert schema.definitions == {
        "C": {"type": "enum", "values": {"b", "a"}},
    }


def test_schema_with_instances():
    @dataclass
    class A:
        a: int
        b: str

    @dataclass
    class B:
        a: A
        b: typing.List[A]

    schema = DataclassesSchema()
    schema.add_dataclass(A(1, "2"))
    assert schema.definitions == {"A": {"a": {"type": "int"}, "b": {"type": "str"}}}

    schema.add_dataclass(B(A(1, "2"), [A(1, "2")]))
    assert schema.definitions == {
        "A": {"a": {"type": "int"}, "b": {"type": "str"}},
        "B": {"a": {"type": "A"}, "b": {"type": "[A]"}},
    }

    # Enum
    class C(Enum):
        a = 1
        b = 2

    schema = DataclassesSchema()
    schema.add_complex_value(C.a)
    assert schema.definitions == {
        "C": {"type": "enum", "values": {"b", "a"}},
    }


def test_schema_with_annotations():
    @dataclass
    class A:
        a: typing_extensions.Annotated[str, "a", "b"]
        b: typing.List[int]

    schema = DataclassesSchema()
    schema.add_dataclass_type(A)
    assert schema.definitions == {"A": {"a": {"type": "str", "metadata": ("a", "b")}, "b": {"type": "[int]"}}}


def test_schema_with_base_classes():
    @dataclass
    class A:
        a: int
        b: str

    @dataclass
    class B(A):
        c: int

    schema = DataclassesSchema()
    schema.add_dataclass_type(B)
    # B will contain all of A's fields and a 'base' list that points to A
    assert schema.definitions == {
        "A": {"a": {"type": "int"}, "b": {"type": "str"}},
        "B": {"a": {"type": "int"}, "b": {"type": "str"}, "c": {"type": "int"}, "bases": ["A"]},
    }


def test_add_return_annotation():
    def foo_int(a: int, b: str) -> int:
        raise NotImplementedError()

    schema = DataclassesSchema()
    schema.add_return_annotation(inspect.signature(foo_int))
    assert schema.definitions == {}

    def foo_list_int(a: int, b: str) -> typing.List[int]:
        raise NotImplementedError()

    schema = DataclassesSchema()
    schema.add_return_annotation(inspect.signature(foo_list_int))
    assert schema.definitions == {}

    @dataclass
    class A:
        a: int
        b: str

    def foo_A(a: int, b: str) -> A:
        raise NotImplementedError()

    schema = DataclassesSchema()
    schema.add_return_annotation(inspect.signature(foo_A))
    assert schema.definitions == {"A": {"a": {"type": "int"}, "b": {"type": "str"}}}

    def foo_list_A(a: int, b: str) -> typing.List[A]:
        raise NotImplementedError()

    schema = DataclassesSchema()
    schema.add_return_annotation(inspect.signature(foo_list_A))
    assert schema.definitions == {"A": {"a": {"type": "int"}, "b": {"type": "str"}}}


def test_extend_parent():
    @dataclass
    class A:
        a: int
        b: str

    schema = DataclassesSchema()
    schema.add_dataclass_type(A)
    assert schema.definitions == {
        "A": {"a": {"type": "int"}, "b": {"type": "str"}},
    }

    @dataclass
    class B:
        c: int

    schema2 = DataclassesSchema.extend_parent(schema)
    schema2.add_dataclass_type(B)
    assert schema2.definitions == {
        "A": {"a": {"type": "int"}, "b": {"type": "str"}},
        "B": {"c": {"type": "int"}},
    }

    assert "B" not in schema.definitions


def test_deserialize_yaml():
    @dataclass
    class A:
        a: int
        b: str

    schema = DataclassesSchema()
    schema.add_dataclass_type(A)

    decoded_yaml = yaml.safe_load(
        """
    a: 1
    b: "2"
    """
    )

    a = deserialize_yaml(decoded_yaml, A)
    assert a == A(1, "2")


def test_deserialize_yaml_with_base_classes():
    @dataclass
    class A:
        a: int
        b: str

    @dataclass
    class B(A):
        c: int

    decoded_yaml = yaml.safe_load(
        """
    a: 1
    b: "2"
    c: 3
    """
    )

    b = deserialize_yaml(decoded_yaml, B)
    assert b == B(1, "2", 3)


def test_deserialize_yaml_with_enum():
    class C(Enum):
        a = 1
        b = 2

    decoded_yaml = yaml.safe_load(
        """
    "a"
    """
    )

    c = deserialize_yaml(decoded_yaml, C)
    assert c == C.a
