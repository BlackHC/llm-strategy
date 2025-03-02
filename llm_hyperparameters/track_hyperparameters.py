import functools
import inspect
import logging
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


class HyperparametersMixin(ContextDecorator, typing.Mapping[Callable | str, typing.Any]):
    root: dict[str, typing.Any]

    def __getitem__(self, key: "TrackedFunction | str") -> dict[str, typing.Any]:
        if isinstance(key, TrackedFunction):
            return self.root[key.config_name]
        else:
            return self.root[key]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __setitem__(self, key: "TrackedFunction | str", value: typing.Any):
        if isinstance(key, TrackedFunction):
            self.root[key.config_name] = value
        else:
            self.root[key] = value

    def __contains__(self, key: object) -> bool:
        if isinstance(key, TrackedFunction):
            return key.config_name in self.root
        else:
            return key in self.root

    def __enter__(self):
        _hyperparameter_scope_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _hyperparameter_scope_stack.pop()


class Hyperparameters(HyperparametersMixin, RootModel[dict[str, typing.Any]]):
    """
    A context manager that allows tracking and overriding hyperparameters for decorated functions.

    This class acts as both a dictionary mapping function names to their hyperparameters,
    and a context manager that establishes a scope where those hyperparameter values are active.

    Functions decorated with @track_hyperparameters can have their hyperparameters (parameters
    prefixed with `hparam_`) modified within the context of a HyperparameterScope instance.
    """

    root: dict[str, typing.Any] = Field(default_factory=dict)

    def __new__(cls, *args, **kwargs):
        global _hyperparameters_typed_dict_type, _hyperparameters_type

        # Update Hyperparameters if necessary.
        # Check that the existing typed_dict_type with all tracked functions that are registered.
        # Get all tracked function config names and types
        tracked_configs = {
            tracked_function.config_name: tracked_function.config_model_type for tracked_function in _tracked_functions
        }

        # Get existing typed dict annotations if any
        existing_annotations = getattr(_hyperparameters_typed_dict_type, "__annotations__", {})

        # Check if annotations match tracked functions
        if _hyperparameters_type is None or existing_annotations != tracked_configs:
            # Print a warning and drop the existing typed dict type.
            if existing_annotations:
                logging.warning(
                    "The tracked functions have changed since the last HyperparameterScope was created. "
                    "Dropping the existing typed dict type and creating a new one."
                )

            # Create the typed dict type if it doesn't exist
            _hyperparameters_typed_dict_type = TypedDict(
                "Hyperparameters",
                tracked_configs,
                total=False,
            )

            def create_hyperparameters_type(typed_dict_type: type) -> type:
                class ActualHyperparameters(HyperparametersMixin, RootModel[typed_dict_type]):
                    root: typed_dict_type = Field(default_factory=dict)

                assert issubclass(ActualHyperparameters, ContextDecorator)
                return ActualHyperparameters

            _hyperparameters_type = functools.wraps(Hyperparameters, updated=())(
                create_hyperparameters_type(_hyperparameters_typed_dict_type)
            )

        return _hyperparameters_type(*args, **kwargs)


_hyperparameter_scope_stack: list[Hyperparameters] = []
_hyperparameters_typed_dict_type: type | None = None
_hyperparameters_type: type | None = None

PARAM_PREFIX = "_"


@dataclass
class TrackedFunction(CallableWrapper, typing.Callable[P, T], typing.Generic[P, T]):  # type: ignore
    """
    A function that is tracked for hyperparameters.
    """

    __wrapped__: typing.Callable[P, T]
    config_model_type: typing.Type[BaseModel]
    config_name: str

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
                field_definitions[name.removeprefix(PARAM_PREFIX)] = (
                    param.annotation,
                    Field(default=param.default),
                )

        escaped_name = string.capwords(f.__name__, sep="_").replace("_", "")
        class_name = f"{escaped_name}Hyperparameters"

        config_model_type = create_model(class_name, __module__=f.__module__, **field_definitions)

        tracked_function = functools.wraps(f)(
            TrackedFunction(
                f,
                config_model_type,
                config_name=convert_callable_to_name(f),
            )
        )
        _tracked_functions.append(tracked_function)
        return tracked_function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if not _hyperparameter_scope_stack:
            return self.__wrapped__(*args, **kwargs)
        else:
            if self.config_name in _hyperparameter_scope_stack[-1]:
                hparams = _hyperparameter_scope_stack[-1][self.config_name]
            else:
                hparams = self.config_model_type()
                _hyperparameter_scope_stack[-1][self.config_name] = hparams

            updated_kwargs = {PARAM_PREFIX + k: v for k, v in hparams.model_dump().items()}
            updated_kwargs.update(kwargs)
            return self.__wrapped__(*args, **updated_kwargs)


_tracked_functions: list[TrackedFunction] = []


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


def convert_module_name(module_name: str):
    # replace . with _ and remove prefix "__"
    module_escaped_name = module_name.replace(".", "_")
    if module_escaped_name.startswith("__"):
        module_escaped_name = module_escaped_name[2:]
    return module_escaped_name


def convert_function_name(f_qual_name: str):
    # convert f_qual_name to a valid class name (. and <. are not allowed)
    f_escaped_name = f_qual_name.replace(".", "_").replace("<", "_").replace(">", "_")
    return f_escaped_name


def convert_callable_to_name(f: Callable) -> str:
    module_escaped_name = convert_module_name(f.__module__)
    f_escaped_name = convert_function_name(f.__qualname__)
    return f"{module_escaped_name}.{f_escaped_name}"
