import dataclasses
import functools
import string
import typing
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo, Undefined, UndefinedType
from pydantic.typing import NoArgAnyCallable

from llm_hyperparameters.utils.callable_wrapper import CallableWrapper

if typing.TYPE_CHECKING:
    from pydantic.typing import AbstractSetIntStr, MappingIntStrAny

T = typing.TypeVar("T")
P = typing.ParamSpec("P")


@dataclass
class HyperparameterDefinition:
    name: str
    explicit_type: type | UndefinedType
    field_info: FieldInfo

    def __matmul__(self, default: T) -> T:
        if _hyperparameter_context is not None:
            self.field_info.default = default
            return _hyperparameter_context.hyperparameter_builder.get_hyperparameter(self.name, default)
        return default


class IdentityDefinition:
    def __matmul__(self, default: T) -> T:
        return default


def Hyperparameter(
    name: str | None = None,
    default: T = Undefined,  # type: ignore
    *,
    description: str | None = None,
    explicit_type: type[T] = Undefined,  # type: ignore
    default_factory: NoArgAnyCallable | None = None,
    alias: str | None = None,
    title: str | None = None,
    exclude: typing.Union["AbstractSetIntStr", "MappingIntStrAny", typing.Any, None] = None,
    include: typing.Union["AbstractSetIntStr", "MappingIntStrAny", typing.Any, None] = None,
    const: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    allow_inf_nan: bool | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
    unique_items: bool | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    allow_mutation: bool = True,
    regex: str | None = None,
    discriminator: str | None = None,
    repr: bool = True,
    **extra: typing.Any,
) -> T | HyperparameterDefinition | IdentityDefinition:  # type: ignore
    definition: HyperparameterDefinition | IdentityDefinition
    if _hyperparameter_context is not None:
        field = Field(
            default_factory=default_factory,  # type: ignore
            alias=alias,
            title=title,
            description=description,
            exclude=exclude,
            include=include,
            const=const,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            multiple_of=multiple_of,
            allow_inf_nan=allow_inf_nan,
            max_digits=max_digits,
            decimal_places=decimal_places,
            min_items=min_items,
            max_items=max_items,
            unique_items=unique_items,
            min_length=min_length,
            max_length=max_length,
            allow_mutation=allow_mutation,
            regex=regex,
            discriminator=discriminator,
            repr=repr,
            **extra,
        )

        if name is None:
            name = f"hparam{_hyperparameter_context.unique_id}"
            _hyperparameter_context.unique_id += 1

        definition = HyperparameterDefinition(name, explicit_type, field)

        _hyperparameter_context.hyperparameter_builder.hyperparameter_definitions[name] = definition
    else:
        definition = IdentityDefinition()
        # log warning
        warnings.warn(UserWarning(f"Hyperparameter {name or default} defined outside of a tracked function!"))

    if default is Undefined:
        # return a wrapper that supports matmul
        return definition
    else:
        return definition @ default


@dataclass
class HyperparameterBuilder:
    module: str
    qualname: str
    hyperparameter_definitions: dict[str, HyperparameterDefinition] = dataclasses.field(default_factory=dict)
    hyperparameters: dict[str | int, typing.Any] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key):
        return self.hyperparameters[key]

    def __setitem__(self, key, value):
        self.hyperparameters[key] = value
        if key not in self.hyperparameter_definitions:
            self.hyperparameter_definitions[key] = HyperparameterDefinition(key, type(value), ...)

    @staticmethod
    def from_function(f: typing.Callable) -> "HyperparameterBuilder":
        escaped_name = string.capwords(f.__name__, sep="_").replace("_", "")
        class_name = f"{escaped_name}Hyperparameters"
        return HyperparameterBuilder(f.__module__, class_name)

    @staticmethod
    def from_model(model: BaseModel) -> "HyperparameterBuilder":
        hyperparameter_definitions: dict[str, HyperparameterDefinition] = {
            name: HyperparameterDefinition(name, field.type_, field.field_info)
            for name, field in model.__fields__.items()
        }
        return HyperparameterBuilder(
            model.__module__, model.__class__.__name__, hyperparameter_definitions, dict(model)
        )

    def get_hyperparameter(self, name: str | int, default: T) -> T:
        return self.hyperparameters.get(name, default)

    def get_type(self, name):
        definition = self.hyperparameter_definitions[name]
        if definition.explicit_type is Undefined:
            return type(self.hyperparameters.get(name, definition.field_info.default))
        else:
            return definition.explicit_type

    def build(self):
        field_definitions = {
            name: (self.get_type(name), definition.field_info)
            for name, definition in self.hyperparameter_definitions.items()
        }

        model = create_model(self.qualname, __module__=self.module, **field_definitions)
        return model(**self.hyperparameters)

    def context(self):
        return HyperparameterContext(self)


@dataclass
class HyperparameterContext:
    hyperparameter_builder: HyperparameterBuilder
    unique_id: int = 0

    @contextmanager
    def scope(self):
        global _hyperparameter_context
        old_hyperparameter_context = _hyperparameter_context
        _hyperparameter_context = self
        try:
            yield
        finally:
            _hyperparameter_context = old_hyperparameter_context


# TODO: replace with a context var
_hyperparameter_context: HyperparameterContext | None = None


class Hyperparameters(BaseModel):
    def __getitem__(self, f: typing.Callable):
        module_escaped_name = HyperparametersBuilder.convert_module_name(f.__module__)
        f_escaped_name = HyperparametersBuilder.convert_function_name(f.__qualname__)

        module = getattr(self, module_escaped_name, None)
        if module is None:
            raise KeyError(f"Module {module_escaped_name} for {f} not found in hyperparameters!")
        f_hyperparameters = getattr(module, f_escaped_name, None)
        if f_hyperparameters is None:
            raise KeyError(f"Function {f_escaped_name} for {f} not found in hyperparameters!")
        return f_hyperparameters

    @staticmethod
    def merge(hyperparameters_iterable: typing.Iterable["Hyperparameters"]) -> "Hyperparameters":
        hyperparameters_builder = HyperparametersBuilder()
        for hyperparameters in hyperparameters_iterable:
            hyperparameters_builder.update(hyperparameters)

        hyperparameters = hyperparameters_builder.build()
        return hyperparameters


@dataclass
class HyperparametersBuilder:
    hyperparameter_builders: defaultdict[str, dict[str, HyperparameterBuilder]] = dataclasses.field(
        default_factory=lambda: defaultdict(dict)
    )

    @staticmethod
    def from_optional_model(hyperparameters: BaseModel | None):
        builder = HyperparametersBuilder()
        if hyperparameters is not None:
            builder.update(hyperparameters)
        return builder

    def __getitem__(self, f: typing.Callable) -> HyperparameterBuilder:
        """
        Look up f in a model created with `create_model`.
        """
        module_escaped_name = HyperparametersBuilder.convert_module_name(f.__module__)
        f_escaped_name = HyperparametersBuilder.convert_function_name(f.__qualname__)

        builder = self.hyperparameter_builders[module_escaped_name].setdefault(
            f_escaped_name, HyperparameterBuilder.from_function(f)
        )
        return builder

    def update(self, hyperparameters: BaseModel):
        """
        Update the hyperparameters from a model created with `create_model`.
        """
        for module_name, module_hparams in hyperparameters:
            for f_name, f_hparams in module_hparams:
                self.hyperparameter_builders[module_name][f_name] = HyperparameterBuilder.from_model(f_hparams)

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

    def build(self):
        # turn function name into a class name
        class_name = "Hyperparameters"

        fields = {}

        # create a model for each model (first nesting level)
        for module_name, module_hparams in self.hyperparameter_builders.items():
            module_fields = {}
            for f_qual_name, f_builder in module_hparams.items():
                model = f_builder.build()
                module_fields[f_qual_name] = model

            module_class_name = f"{module_name}Module"

            module_model = create_model(module_class_name, __module__=__name__, **module_fields)

            fields[module_name] = module_model()

        hyperparameter_model = create_model(class_name, __module__=__name__, __base__=Hyperparameters, **fields)
        return hyperparameter_model()

    @contextmanager
    def scope(self):
        """
        Context manager that allows to track hyperparameters.
        """
        global _hyperparameters_builder
        old_hyperparameters_builder = _hyperparameters_builder
        _hyperparameters_builder = self

        try:
            yield
        finally:
            _hyperparameters_builder = old_hyperparameters_builder


_hyperparameters_builder: HyperparametersBuilder | None = None


@dataclass
class TrackedFunction(CallableWrapper, typing.Callable[P, T], typing.Generic[P, T]):  # type: ignore
    """
    A callable that can be called with a chat model.
    """

    __wrapped__: typing.Callable[P, T]

    @staticmethod
    def from_function(f: typing.Callable[P, T]):
        tracked_function: TrackedFunction = functools.wraps(f)(
            TrackedFunction(
                f,
            )
        )

        return tracked_function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if _hyperparameters_builder is None:
            return self.__wrapped__(*args, **kwargs)

        with _hyperparameters_builder[self].context().scope():
            return self.__wrapped__(*args, **kwargs)


def track_hyperparameters(f: typing.Callable[P, T]) -> typing.Callable[P, T]:
    return TrackedFunction.from_function(f)


@dataclass
class HyperparametersScope:
    hyperparameters: BaseModel | None = None

    @contextmanager
    def __call__(self):
        builder = HyperparametersBuilder.from_optional_model(self.hyperparameters)
        with builder.scope():
            yield self
        self.hyperparameters = builder.build()


@contextmanager
def hyperparameters_scope(hyperparameters: BaseModel | None = None) -> typing.Iterator[HyperparametersScope]:
    scope = HyperparametersScope(hyperparameters)
    with scope():
        yield scope
