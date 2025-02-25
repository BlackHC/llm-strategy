import functools
import inspect
import string
import typing
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, Field, RootModel, create_model
from typing_extensions import TypedDict

from llm_hyperparameters.utils.callable_wrapper import CallableWrapper

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


class Hyperparameters(typing.Mapping[str, typing.Any]):
    pass


class HyperparameterScope(dict[Callable | str, typing.Any], ContextDecorator):
    """
    A context manager that allows tracking and overriding hyperparameters for decorated functions.

    This class acts as both a dictionary mapping function names to their hyperparameters,
    and a context manager that establishes a scope where those hyperparameter values are active.

    Functions decorated with @track_hyperparameters can have their hyperparameters (parameters
    prefixed with `hparam_`) modified within the context of a HyperparameterScope instance.
    """

    @staticmethod
    def convert_module_name(module_name: str):
        # replace . with _ and remove prefix "__"
        module_escaped_name = module_name.replace(".", "_")
        if module_escaped_name.startswith("__"):
            module_escaped_name = module_escaped_name[2:]
        return module_escaped_name

    @staticmethod
    def convert_function_name(f_qual_name: str):
        # convert f_qual_name to a valid class name (. and <. are not allowed)
        f_escaped_name = f_qual_name.replace(".", "_").replace("<", "_").replace(">", "_")
        return f_escaped_name

    @staticmethod
    def convert_callable_to_name(f: Callable) -> str:
        module_escaped_name = HyperparameterScope.convert_module_name(f.__module__)
        f_escaped_name = HyperparameterScope.convert_function_name(f.__qualname__)
        return f"{module_escaped_name}.{f_escaped_name}"

    def __getitem__(self, key: Callable | str) -> dict[str, typing.Any]:
        if isinstance(key, Callable):
            return super().__getitem__(HyperparameterScope.convert_callable_to_name(key))
        else:
            return super().__getitem__(key)

    def __setitem__(self, key: Callable | str, value: typing.Any):
        if isinstance(key, Callable):
            return super().__setitem__(HyperparameterScope.convert_callable_to_name(key), value)
        else:
            return super().__setitem__(key, value)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, Callable):
            return super().__contains__(HyperparameterScope.convert_callable_to_name(key))
        else:
            return super().__contains__(key)

    def __delitem__(self, key: str) -> None:
        if isinstance(key, Callable):
            return super().__delitem__(HyperparameterScope.convert_callable_to_name(key))
        else:
            return super().__delitem__(key)

    def __enter__(self):
        _hyperparameters_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _hyperparameters_stack.pop()

    def freeze(self):
        FrozenHyperparametersDict = TypedDict(
            "FrozenHyperparametersDict", {k: type(v) for k, v in self.items()}, total=False  # type: ignore
        )

        class FrozenHyperparameters(RootModel[FrozenHyperparametersDict], typing.Mapping[str, typing.Any]):
            root: FrozenHyperparametersDict

            def __getitem__(self, key: str) -> typing.Any:
                return self.root[key]

            def __iter__(self) -> typing.Iterator[str]:
                return iter(self.root)

            def __len__(self) -> int:
                return len(self.root)

        return FrozenHyperparameters(self)


_hyperparameters_stack: list[Hyperparameters] = []

PARAM_PREFIX = "hparam_"


@dataclass
class TrackedFunction(CallableWrapper, typing.Callable[P, T], typing.Generic[P, T]):  # type: ignore
    """
    A function that is tracked for hyperparameters.
    """

    __wrapped__: typing.Callable[P, T]
    hparams_model_type: typing.Type[BaseModel]

    @staticmethod
    def from_function(f: typing.Callable[P, T]) -> "TrackedFunction[P, T]":
        # Get signature and extract hyperparameter args
        sig = inspect.signature(f)
        field_definitions = {}
        for name, param in sig.parameters.items():
            if name.startswith(PARAM_PREFIX):
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"Hyperparameter {name} has no explicit value!")
                if param.annotation is inspect.Parameter.empty:
                    raise ValueError(f"Hyperparameter {name} has no explicit type!")
                field_definitions[name.removeprefix(PARAM_PREFIX)] = (param.annotation, Field(default=param.default))

        escaped_name = string.capwords(f.__name__, sep="_").replace("_", "")
        class_name = f"{escaped_name}Hyperparameters"

        hparams_model_type = create_model(class_name, __module__=f.__module__, **field_definitions)

        tracked_function = functools.wraps(f)(
            TrackedFunction(
                f,
                hparams_model_type,
            )
        )
        return tracked_function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if not _hyperparameters_stack:
            return self.__wrapped__(*args, **kwargs)
        else:
            if self.__wrapped__ in _hyperparameters_stack[-1]:
                hparams = _hyperparameters_stack[-1][self.__wrapped__]
            else:
                hparams = self.hparams_model_type()
                _hyperparameters_stack[-1][self.__wrapped__] = hparams
            updated_kwargs = {PARAM_PREFIX + k: v for k, v in hparams.model_dump().items()}
            updated_kwargs.update(kwargs)
            return self.__wrapped__(*args, **updated_kwargs)


def track_hyperparameters(f: typing.Callable[P, T]) -> typing.Callable[P, T]:
    """
    Decorator that enables dynamic hyperparameter tracking and modification for functions.

    Any parameter prefixed with 'hparam_' will be treated as a hyperparameter that can be
    modified at runtime. Hyperparameters must have explicit type annotations and default values.

    Example:
        @track_hyperparameters
        def train_model(*, hparam_learning_rate: float = 0.01, hparam_batch_size: int = 32):
            ...

        with HyperparameterScope() as hparams:
            train_model()  # Uses default hyperparameters

        hparams[train_model].learning_rate = 0.001  # Modify learning rate

        with hparams:
            train_model()  # Uses modified learning rate
    """
    return TrackedFunction.from_function(f)
