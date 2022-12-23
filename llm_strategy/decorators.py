# type: ignore
import dataclasses
import dis
import functools
import inspect
import types
import typing
from dataclasses import dataclass

import typing_extensions
from langchain.llms.base import BaseLLM

from llm_strategy.dataclasses_schema import DataclassesSchema
from llm_strategy.llm_implement import LLMCall, unwrap_function

P = typing_extensions.ParamSpec("P")
T = typing.TypeVar("T")


def check_not_implemented(f: typing.Callable) -> bool:
    """Check that a function (property getter, regular method, class method or static method)
    only raises NotImplementedError."""
    unwrapped_f = unwrap_function(f)

    if not hasattr(unwrapped_f, "__code__"):
        raise ValueError(f"Cannot check whether {f} is implemented. Where is __code__?")

    # Inspect the opcodes
    code = unwrapped_f.__code__
    # Get the opcodes
    opcodes = list(dis.get_instructions(code))
    # Check that it only uses the following opcodes:
    # - RESUME
    # - LOAD_GLOBAL
    # - PRECALL
    # - CALL
    # - RAISE_VARARGS
    valid_opcodes = {
        "RESUME",
        "LOAD_GLOBAL",
        "PRECALL",
        "CALL",
        "RAISE_VARARGS",
    }
    # We allow at most a function of length len(valid_opcodes)
    if len(opcodes) > len(valid_opcodes):
        return False
    for opcode in opcodes:
        if opcode.opname not in valid_opcodes:
            return False
        # Check that the function only raises NotImplementedError
        if opcode.opname == "LOAD_GLOBAL" and opcode.argval != "NotImplementedError":
            return False
        if opcode.opname == "RAISE_VARARGS" and opcode.argval != 1:
            return False
        valid_opcodes.remove(opcode.opname)
    # Check that the function raises a NotImplementedError at the end.
    if opcodes[-1].opname != "RAISE_VARARGS":
        return False
    return True


def llm_strategy(  # noqa: C901
    llm: BaseLLM, parent_dataclasses_schema: DataclassesSchema | None = None
) -> typing.Callable[[T], T]:
    """
    A strategy that implements what ever it decorates (or is called on) using the LLM.
    """

    @typing.no_type_check
    def decorator(f: T) -> T:
        # For an instance of dataclass, call llm_strategy_dataclass with the fields.
        if dataclasses.is_dataclass(f):
            if isinstance(f, type):
                return llm_strategy_dataclass(f, llm, parent_dataclasses_schema)
            else:
                implemented_dataclass = llm_strategy_dataclass(type(f), llm, parent_dataclasses_schema)
                # Create an instance of the implemented dataclass using the fields from f
                params = {field.name: getattr(f, field.name) for field in dataclasses.fields(f)}
                return implemented_dataclass(**params)
        elif isinstance(f, classmethod):
            return classmethod(decorator(f.__func__))
        elif isinstance(f, staticmethod):
            return staticmethod(decorator(f.__func__))
        elif isinstance(f, property):
            return property(decorator(f.fget), doc=f.__doc__)
        elif isinstance(f, types.MethodType):
            return types.MethodType(decorator(f.__func__), f.__self__)
        # Does the function have a __wrapped__ attribute?
        elif hasattr(f, "__wrapped__"):
            # If so, it is a wrapped function. Unwrap it.
            return decorator(f.__wrapped__)
        elif isinstance(f, LLMCall):
            return f
        elif callable(f):
            llm_call = LLMCall.wrap_callable(f, llm, parent_dataclasses_schema)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return llm_call(*args, **kwargs)

            return wrapper
        else:
            raise ValueError(f"Cannot decorate {f} with llm_strategy.")

    return decorator


def can_wrap_member_in_llm(f: typing.Callable[P, T]) -> bool:
    """
    Return True if f can be wrapped in an LLMCall.
    """
    # An existing LLMCall was already determined to be wrappable.
    if isinstance(f, LLMCall):
        return True
    if dataclasses.is_dataclass(f):
        return True

    unwrapped = unwrap_function(f)
    if not callable(unwrapped):
        return False
    if inspect.isgeneratorfunction(unwrapped):
        return False
    if inspect.iscoroutinefunction(unwrapped):
        return False
    if not inspect.isfunction(unwrapped) and not inspect.ismethod(unwrapped):
        return False
    return check_not_implemented(unwrapped)


@typing_extensions.dataclass_transform()
def llm_strategy_dataclass(
    dataclass_type: type, llm: BaseLLM, parent_dataclasses_schema: DataclassesSchema | None = None
) -> type:
    global long_unlikely__dataclasses_schema
    global long_unlikely_prefix__llm
    global long_unlikely__member_name, long_unlikely__member
    long_unlikely_prefix__llm = llm
    long_unlikely__dataclasses_schema = parent_dataclasses_schema
    long_unlikely__member_name, long_unlikely__member = None, None

    @dataclass
    class SpecificLLMImplementation(dataclass_type):
        global long_unlikely__member_name, long_unlikely__member
        for long_unlikely__member_name, long_unlikely__member in inspect.getmembers_static(  # noqa: B007
            dataclass_type, can_wrap_member_in_llm
        ):
            exec(
                f"""
@llm_strategy(long_unlikely_prefix__llm, long_unlikely__dataclasses_schema)
@functools.wraps(long_unlikely__member)
def {long_unlikely__member_name}(*args, **kwargs):
    raise NotImplementedError()
"""
            )

    SpecificLLMImplementation.__name__ = f"{llm.__class__.__name__}_{dataclass_type.__name__}"
    SpecificLLMImplementation.__qualname__ = f"{llm.__class__.__name__}_{dataclass_type.__qualname__}"

    del long_unlikely__member_name, long_unlikely__member
    del long_unlikely_prefix__llm, long_unlikely__dataclasses_schema

    return SpecificLLMImplementation
