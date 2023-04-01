# type: ignore
import dataclasses
import functools  # noqa: F401
import inspect
import typing
from dataclasses import dataclass

import typing_extensions
from langchain.llms.base import BaseLLM

from llm_strategy.llm_function import (
    LLMFunction,
    can_wrap_function_in_llm,
    llm_function,
)

P = typing_extensions.ParamSpec("P")
T = typing.TypeVar("T")


def llm_strategy(llm: BaseLLM) -> typing.Callable[[T], T]:  # noqa: C901
    """
    A strategy that implements what ever it decorates (or is called on) using the LLM.
    """

    @typing.no_type_check
    def decorator(f: T) -> T:
        assert can_wrap_member_in_llm(f)

        # For an instance of dataclass, call llm_strategy_dataclass with the fields.
        if dataclasses.is_dataclass(f):
            if isinstance(f, type):
                return llm_dataclass(f, llm)
            else:
                implemented_dataclass = llm_dataclass(type(f), llm)
                # Create an instance of the implemented dataclass using the fields from f
                params = {field.name: getattr(f, field.name) for field in dataclasses.fields(f)}
                return implemented_dataclass(**params)
        else:
            return llm_function(llm)(f)

    return decorator


def can_wrap_member_in_llm(f: typing.Callable[P, T]) -> bool:
    """
    Return True if f can be wrapped in an LLMCall.
    """
    if isinstance(f, LLMFunction):
        return True
    if dataclasses.is_dataclass(f):
        return True

    return can_wrap_function_in_llm(f)


@typing_extensions.dataclass_transform()
def llm_dataclass(dataclass_type: type, llm: BaseLLM) -> type:
    global long_unlikely_prefix__llm
    global long_unlikely__member_name, long_unlikely__member
    long_unlikely_prefix__llm = llm
    long_unlikely__member_name, long_unlikely__member = None, None

    @dataclass
    class SpecificLLMImplementation(dataclass_type):
        global long_unlikely__member_name, long_unlikely__member
        for long_unlikely__member_name, long_unlikely__member in inspect.getmembers_static(  # noqa: B007
            dataclass_type, can_wrap_member_in_llm
        ):
            exec(
                f"""
@llm_strategy(long_unlikely_prefix__llm)
@functools.wraps(long_unlikely__member)
def {long_unlikely__member_name}(*args, **kwargs):
    raise NotImplementedError()
"""
            )

    SpecificLLMImplementation.__name__ = f"{llm.__class__.__name__}_{dataclass_type.__name__}"
    SpecificLLMImplementation.__qualname__ = f"{llm.__class__.__name__}_{dataclass_type.__qualname__}"

    del long_unlikely__member_name, long_unlikely__member
    del long_unlikely_prefix__llm

    return SpecificLLMImplementation
