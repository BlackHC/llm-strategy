import typing
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from llm_strategy.pydantic_generic_type_resolution import PydanticGenericTypeMap


def test_get_pydantic_generic_type_map() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    U = typing.TypeVar("U")
    V = typing.TypeVar("V")

    X = typing.TypeVar("X")

    class GenericType(BaseModel, typing.Generic[T, S]):
        value: T
        value2: S

    assert PydanticGenericTypeMap.get_generic_type_map(GenericType) == {T: T, S: S}
    assert PydanticGenericTypeMap.get_generic_type_map(GenericType[S, T]) == {T: S, S: T}
    assert PydanticGenericTypeMap.get_generic_type_map(GenericType[S, T][U, V]) == {T: U, S: V}  # type: ignore
    assert PydanticGenericTypeMap.get_generic_type_map(GenericType[S, T][U, V][X, X]) == {T: X, S: X}  # type: ignore
    assert PydanticGenericTypeMap.get_generic_type_map(GenericType[U, U][X]) == {T: X, S: X}  # type: ignore
    assert PydanticGenericTypeMap.get_generic_type_map(GenericType[int, U][str]) == {T: int, S: str}  # type: ignore


def test_pydantic_resolve_generic_types() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    class GenericType(BaseModel, typing.Generic[T, S]):
        a: T
        b: S

    assert PydanticGenericTypeMap.resolve_generic_types(GenericType, GenericType[int, str](a=1, b="Hello")) == {
        T: int,
        S: str,
    }


def test_pydantic_resolve_generic_types_right_nested() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    class GenericType(BaseModel, typing.Generic[T, S]):
        a: T
        b: S

    assert PydanticGenericTypeMap.resolve_generic_types(
        GenericType, GenericType[GenericType[int, float], str](a=GenericType[int, float](a=1, b=2.0), b="Hello")
    ) == {
        T: GenericType[int, float],
        S: str,
    }


def test_pydantic_resolve_generic_types_left_nested() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    class GenericType(BaseModel, typing.Generic[T, S]):
        a: T
        b: S

    assert PydanticGenericTypeMap.resolve_generic_types(
        GenericType[GenericType[T, float], S],
        GenericType[GenericType[int, float], str](a=GenericType[int, float](a=1, b=2.0), b="Hello"),
    ) == {
        T: int,
        S: str,
    }


def test_simple_resolve_generic_types() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    @dataclass
    class SimpleGenericType(typing.Generic[T, S]):
        a: T
        b: S

    with pytest.raises(TypeError):
        PydanticGenericTypeMap.resolve_generic_types(SimpleGenericType, SimpleGenericType[int, str](a=1, b="Hello"))
    # assert PydanticGenericTypeMap.resolve_generic_types(SimpleGenericType, SimpleGenericType[int, str](a=1, b="Hello")) == {
    #     T: int,
    #     S: str,
    # }


def test_get_base_generic_type() -> None:
    T = typing.TypeVar("T")
    S = typing.TypeVar("S")

    # Generic Pydantic model
    @dataclass
    class SimpleGenericType(typing.Generic[T, S]):
        a: T
        b: S

    assert PydanticGenericTypeMap.get_base_generic_type(list[int]) == list
    assert PydanticGenericTypeMap.get_base_generic_type(SimpleGenericType[int, str]) == SimpleGenericType

    assert PydanticGenericTypeMap.get_base_generic_type(list) == list
    assert PydanticGenericTypeMap.get_base_generic_type(SimpleGenericType) == SimpleGenericType

    assert PydanticGenericTypeMap.get_base_generic_type(int) == int
