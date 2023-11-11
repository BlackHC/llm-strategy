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
    apply_decorator,
    is_not_implemented,
    unwrap_function,
)

P = typing_extensions.ParamSpec("P")
T = typing.TypeVar("T")


def can_wrap_function_in_llm(f: typing.Callable[P, T]) -> bool:
    """
    Return True if f can be wrapped in an LLMCall.
    """
    unwrapped = unwrap_function(f)
    if not callable(unwrapped):
        return False
    if inspect.isgeneratorfunction(unwrapped):
        return False
    if inspect.iscoroutinefunction(unwrapped):
        return False
    if not inspect.isfunction(unwrapped) and not inspect.ismethod(unwrapped):
        return False
    return is_not_implemented(unwrapped)


def llm_strategy(llm: BaseLLM) -> typing.Callable[[T], T]:  # noqa: C901
    """
    A strategy that implements whatever it decorates (or is called on) using the LLM.
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

            def inner_decorator(unwrapped_f):
                llm_f = None

                @functools.wraps(unwrapped_f)
                def strategy_wrapper(*args, **kwargs):
                    nonlocal llm_f
                    if llm_f is None:
                        # Get the signature of f
                        sig = inspect.signature(unwrapped_f, eval_str=True)
                        # Add a llm parameter to the signature as first argument
                        new_params = [inspect.Parameter("__llm", inspect.Parameter.POSITIONAL_ONLY)]
                        new_params.extend(sig.parameters.values())

                        new_sig = sig.replace(parameters=new_params)

                        def dummy_f(*args, **kwargs):
                            raise NotImplementedError()

                        new_f = functools.wraps(unwrapped_f)(dummy_f)
                        new_f.__module__ = unwrapped_f.__module__
                        # Set the signature of the new function
                        new_f.__signature__ = new_sig

                        del new_f.__wrapped__

                        # Wrap the function in an LLMFunction
                        llm_f = functools.wraps(new_f)(LLMFunction())

                    return llm_f(llm, *args, **kwargs)

                return strategy_wrapper

            return apply_decorator(f, inner_decorator)

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
