# type: ignore
import dis
import functools
import inspect
import json
import re
import string
import types
import typing
from copy import deepcopy
from dataclasses import dataclass

import pydantic
import typing_extensions
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, BaseLanguageModel, BaseLLM
from llmtracer import TraceNodeKind, trace_calls, update_event_properties, update_name
from llmtracer.trace_builder import slicer
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic._internal._generics import get_origin
from pydantic.fields import FieldInfo, PydanticUndefined
from pydantic.type_adapter import TypeAdapter

from llm_hyperparameters.track_hyperparameters import (
    TrackedFunction,
    track_hyperparameters,
)
from llm_strategy.chat_chain import ChatChain
from llm_strategy.pydantic_generic_type_resolution import PydanticGenericTypeMap

T = typing.TypeVar("T")
S = typing.TypeVar("S")
P = typing_extensions.ParamSpec("P")
B = typing.TypeVar("B", bound=BaseModel)
C = typing.TypeVar("C", bound=BaseModel)
F = typing.TypeVar("F", bound=typing.Callable)


_model_cache: dict[str, dict[tuple[tuple[type, ...], tuple], type[BaseModel]]] = {}


def cached_create_model(
    model_name: str,
    *,
    __base__: tuple[type, ...] | type = None,
    __module__: str | None = None,
    __validators__: dict[str, classmethod] | None = None,
    __cls_kwargs__: dict[str, typing.Any] | None = None,
    **field_definitions: typing.Any,
) -> type[BaseModel]:
    """
    A wrapper around pydantic's create_model that caches models by their specification.

    This helps avoid creating duplicate model classes with the same structure.
    """
    # Convert __base__ to a tuple if it's a single type
    if __base__ is not None and not isinstance(__base__, tuple):
        __base__ = (__base__,)

    # Create a hashable representation of field definitions
    field_defs_hashable = tuple(field_definitions.items())

    if model_name not in _model_cache:
        _model_cache[model_name] = {}

    specific_model_cache = _model_cache[model_name]

    # Create a cache key
    cache_key = (__base__ or (), field_defs_hashable)

    # Check if we already have this model in the cache
    if cache_key in specific_model_cache:
        return specific_model_cache[cache_key]

    if specific_model_cache:
        unique_model_name = f"{model_name}_{len(specific_model_cache)}"
    else:
        unique_model_name = model_name

    # Create the model
    model = create_model(
        unique_model_name,
        __base__=__base__,
        __module__=__module__,
        __validators__=__validators__,
        __cls_kwargs__=__cls_kwargs__,
        **field_definitions,
    )

    # Cache the model
    specific_model_cache[cache_key] = model

    return model


def reset_model_cache():
    """
    Reset the model cache.
    """
    global _model_cache
    _model_cache = {}


def get_json_schema_hyperparameters(schema: dict):
    """
    Get the hyperparameters from a JSON schema recursively.

    The hyperparameters are all fields for keys with "title" or "description".
    """
    hyperparameters = {}
    for key, value in schema.items():
        if key == "description":
            hyperparameters[key] = value
        elif isinstance(value, dict):
            sub_hyperparameters = get_json_schema_hyperparameters(value)
            if sub_hyperparameters:
                hyperparameters[key] = sub_hyperparameters
    return hyperparameters


def update_json_schema_hyperparameters(schema: dict, hyperparameters: dict):
    """
    Nested merge of the schema dict with the hyperparameters dict.
    """
    for key, value in hyperparameters.items():
        if key in schema:
            if isinstance(value, dict):
                update_json_schema_hyperparameters(schema[key], value)
            else:
                schema[key] = value
        else:
            schema[key] = value


def unwrap_function(f: typing.Callable[P, T]) -> typing.Callable[P, T]:
    # is f a property?
    if isinstance(f, property):
        f = f.fget
    # is f a wrapped function?
    elif hasattr(f, "__wrapped__"):
        f = inspect.unwrap(f)
    elif inspect.ismethod(f):
        f = f.__func__
    else:
        return f

    return unwrap_function(f)


def is_not_implemented(f: typing.Callable) -> bool:
    """Check that a function only raises NotImplementedError."""
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


class TyperWrapper(str):
    """
    A wrapper we use as an annotation instead of `type` to make types serializable.

    This is used to support scenarios where we need to serialize type objects in Pydantic models.
    """

    @classmethod
    def __get_validators__(cls) -> typing.Iterator[typing.Callable]:
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v: type) -> str:
        if not isinstance(v, type):
            raise TypeError("type required")
        return v.__qualname__


class Output(BaseModel, typing.Generic[T]):
    return_value: T


DEFAULT_JSON_SCHEMA = pydantic.json_schema.GenerateJsonSchema()


@dataclass
class LLMStructuredPrompt(typing.Generic[B, T]):
    """
    A structured prompt for a language model.
    """

    docstring: str
    input_type: type[B]
    output_type: type[Output[T]]
    return_annotation: T

    input: B

    @staticmethod
    def extract_from_definitions(definitions: dict, type_: type) -> dict:
        type_name = type_.__name__
        normalized_name = DEFAULT_JSON_SCHEMA.normalize_name(type_name)

        # TODO: somehow we have different numbers of _ in the definitions
        # We should fix this properly but right now manually matching is the best alternative.

        # so let's normalize both by replacing _{1,} with _
        # This is not using regex correctly - replace with proper regex pattern
        normalized_name = re.sub(r"_+", "_", normalized_name)
        for name, sub_schema in definitions.items():
            normalized_def_name = re.sub(r"_+", "_", name)
            if normalized_name == normalized_def_name:
                del definitions[name]
                return sub_schema

        raise ValueError(f"Could not find type {type_name} in definitions {definitions}")

    def get_json_schema(self, exclude_default: bool = True) -> dict:
        type_adapter = TypeAdapter(tuple[self.input_type, self.output_type])
        schema = type_adapter.json_schema(ref_template="{model}")
        definitions: dict = deepcopy(schema["$defs"])
        # remove title and type from each sub dict in the definitions
        for value in definitions.values():
            value.pop("title", None)
            value.pop("type", None)

            for property in value.get("properties", {}).values():
                property.pop("title", None)
                if exclude_default:
                    property.pop("default", None)

        input_schema = self.extract_from_definitions(definitions, self.input_type)
        output_schema = self.extract_from_definitions(definitions, self.output_type)

        schema = dict(
            input_schema=input_schema,
            output_schema=output_schema,
            additional_definitions=definitions,
        )
        return schema

    @staticmethod
    def create(docstring: str, input_type: type[B], return_annotation: T, input: B) -> "LLMStructuredPrompt[B, T]":
        """Create an LLMExplicitFunction."""

        # determine the return type
        # the return type can be a type annotation or an Annotated type with annotation being a FieldInfo
        if get_origin(return_annotation) is typing.Annotated:
            return_info = typing.get_args(return_annotation)
        else:
            return_info = (return_annotation, ...)

        # resolve generic types
        generic_type_map = PydanticGenericTypeMap.resolve_generic_types(input_type, input)
        return_type: type = generic_type_map.resolve_type(return_info[0])

        if return_type is types.NoneType:  # noqa: E721
            raise ValueError(f"Resolve return type {return_info[0]} is None! This would be a NOP.")

        return_info = (return_type, return_info[1])

        if get_origin(return_annotation) is typing.Annotated:
            assert hasattr(return_annotation, "copy_with")
            resolved_return_annotation = return_annotation.copy_with([return_info[0]])
        else:
            resolved_return_annotation = return_info[0]

        # create the output model
        resolved_output_model_type = Output[return_type]  # noqa

        # resolve input_type
        resolved_input_type = generic_type_map.resolve_type(input_type)

        return LLMStructuredPrompt(
            docstring=docstring,
            input_type=resolved_input_type,
            output_type=resolved_output_model_type,
            return_annotation=resolved_return_annotation,
            input=input,
        )

    @trace_calls(
        name="LLMStructuredPrompt",
        kind=TraceNodeKind.CHAIN,
        capture_args=False,
        capture_return=False,
    )
    def __call__(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain,
    ) -> T:
        """Call the function."""

        # check that the first argument is an instance of BaseLanguageModel
        # or a TrackedChatChain or UntrackedChatChain
        if not isinstance(language_model_or_chat_chain, BaseLanguageModel | ChatChain):
            raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # get the input and output schema as JSON dict
        schema = self.get_json_schema()
        # print(json.dumps(schema, indent=1))

        update_json_schema_hyperparameters(
            schema,
            get_json_schema_hyperparameters(schema),
        )

        update_event_properties(
            dict(
                arguments=dict(self.input),
            )
        )

        parsed_output = self.query(language_model_or_chat_chain, schema)

        # print(f"Input: {self.input.json(indent=1)}")
        # print(f"Output: {json.dumps(json.loads(parsed_output.json())['return_value'], indent=1)}")

        update_event_properties(dict(result=parsed_output.return_value))

        return parsed_output.return_value

    @track_hyperparameters
    def query(  # noqa: C901
        self,
        language_model_or_chat_chain,
        schema,
        _json_dumps_kwargs: dict = dict(indent=None),  # noqa: B006
        _additional_definitions_prompt_template: str = "Here is the schema for additional data types:\n```\n{additional_definitions}\n```\n\n",
        _llm_structured_prompt_template: str = Field(  # noqa: B008
            "{docstring}\n\nThe input and output are formatted as a JSON interface that conforms to the JSON schemas"
            ' below.\n\nAs an example, for the schema {{"properties": {{"foo": {{"description": "a list of strings",'
            ' "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}} the object {{"foo": ["bar",'
            ' "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar",'
            ' "baz"]}}}} is not well-formatted.\n\n{optional_additional_definitions_prompt}Here is the input'
            " schema:\n```\n{input_schema}\n```\n\nHere is the output schema:\n```\n{output_schema}\n```\nNow output"
            " the results for the following inputs:\n```\n{inputs}\n```",
            description=(
                "The general-purpose prompt for the structured prompt execution. It tells the LLM what to "
                "do and how to read function arguments and structure return values. "
            ),
        ),
        _num_retries_on_parser_failure: int = 3,
        _output_prompt: str = "Received the output\n\n",
        _error_prompt: str = "Tried to parse your output but failed:\n\n",
        _retry_prompt: str = "Please try again and avoid this issue.",
    ):
        # create the prompt
        optional_additional_definitions_prompt = ""
        if schema["additional_definitions"]:
            optional_additional_definitions_prompt = _additional_definitions_prompt_template.format(
                additional_definitions=json.dumps(schema["additional_definitions"], **_json_dumps_kwargs)
            )

        prompt = _llm_structured_prompt_template.format(
            docstring=self.docstring,
            optional_additional_definitions_prompt=optional_additional_definitions_prompt,
            input_schema=json.dumps(schema["input_schema"], **_json_dumps_kwargs),
            output_schema=json.dumps(schema["output_schema"], **_json_dumps_kwargs),
            inputs=self.input.model_dump_json(**_json_dumps_kwargs),
        )

        # get the response
        if language_model_or_chat_chain is None:
            raise ValueError("The language model or chat chain must be provided.")

        if isinstance(language_model_or_chat_chain, BaseChatModel):
            language_model_or_chat_chain = ChatChain(language_model_or_chat_chain, [])

        if isinstance(language_model_or_chat_chain, ChatChain):
            chain = language_model_or_chat_chain
            for _ in range(_num_retries_on_parser_failure):
                output, chain = chain.query(prompt, model_args=chain.enforce_json_response())

                try:
                    parsed_output = parse(output, self.output_type)
                    break
                except OutputParserException as e:
                    prompt = prompt + _output_prompt + output + _error_prompt + str(e) + _retry_prompt
            else:
                exception = OutputParserException(
                    f"Failed to parse the output after {_num_retries_on_parser_failure} retries."
                )
                exception.add_note(repr(chain))
                raise exception
        elif isinstance(language_model_or_chat_chain, BaseLLM):
            model: BaseChatModel = language_model_or_chat_chain
            # Check if the language model is of type "openai" and extend model args with a response format in that case
            model_dict = model.dict()
            if "openai" in model_dict["_type"] and model_dict.get("model_name") in (
                "gpt-4-1106-preview",
                "gpt-3.5-turbo-1106",
            ):
                model_args = dict(response_format=dict(type="json_object"))
            else:
                model_args = {}

            for _ in range(_num_retries_on_parser_failure):
                output = model(prompt, **model_args)

                try:
                    parsed_output = parse(output, self.output_type)
                    break
                except OutputParserException as e:
                    prompt = prompt + _output_prompt + output + _error_prompt + str(e) + _retry_prompt
            else:
                exception = OutputParserException(
                    f"Failed to parse the output after {_num_retries_on_parser_failure} retries."
                )
                exception.add_note(prompt)
                raise exception
        else:
            raise ValueError("The language model or chat chain must be provided.")
        return parsed_output


@dataclass
class LLMBoundSignature:
    """
    A function call that can be used to generate a prompt.
    """

    structured_prompt: LLMStructuredPrompt
    signature: inspect.Signature

    @property
    def input_type(self) -> type[BaseModel]:
        """Return the input type."""
        return self.structured_prompt.input_type

    @property
    def output_type(self) -> type[BaseModel]:
        """Return the output type."""
        return self.structured_prompt.output_type

    @property
    def docstring(self) -> str:
        """Return the docstring."""
        return self.structured_prompt.docstring

    @property
    def return_annotation(self) -> str:
        """Return the name."""
        return self.structured_prompt.return_annotation

    def get_input_object(self, *args: P.args, **kwargs: P.kwargs) -> BaseModel:
        """Call the function and return the inputs."""
        # bind the inputs to the signature
        bound_arguments = LLMBoundSignature.bind(self.signature, args, kwargs)
        # get the arguments
        arguments = bound_arguments.arguments
        inputs = self.structured_prompt.input_type(**arguments)
        return inputs

    @staticmethod
    def from_call(  # noqa: C901
        f: typing.Callable[P, T],
        args: P.args,
        kwargs: P.kwargs,
        _docstring: str | None = None,
    ) -> "LLMBoundSignature":  # noqa: C901
        """Create an LLMBoundSignature from a function.

        Args:
            f: The function to create the LLMBoundSignature from.
            args: The positional arguments to the function (but excluding the language model/first param).
            kwargs: The keyword arguments to the function.

        """

        # get clean docstring
        docstring = _docstring or inspect.getdoc(f)
        if docstring is None:
            raise ValueError("The function must have a docstring.")

        # get the type of the first argument
        globals_from_f = f.__globals__ if hasattr(f, "__globals__") else None
        locals_from_f = f.__dict__ if hasattr(f, "__dict__") else None
        signature = inspect.signature(f, eval_str=True, globals=globals_from_f, locals=locals_from_f)

        # get all parameters
        parameters_items: list[tuple[str, inspect.Parameter]] = list(signature.parameters.items())
        # check that there is at least one parameter
        if not parameters_items:
            raise ValueError("The function must have at least one parameter.")
        # check that the first parameter has a type annotation that is an instance of BaseLanguageModel
        # or a TrackedChatChain
        first_parameter: inspect.Parameter = parameters_items[0][1]
        if first_parameter.annotation is not inspect.Parameter.empty:
            if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain):
                raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        return_type = signature.return_annotation
        if return_type is inspect.Parameter.empty:
            raise ValueError("The function must have a return type.")

        # create a pydantic model from the parameters
        parameter_dict = LLMBoundSignature.parameter_items_to_field_tuple(parameters_items[1:])

        # turn function name into a class name
        class_name = string.capwords(f.__name__, sep="_").replace("_", "")

        # create the input model
        # collect all __parameters__ from the type annotations
        # this is necessary because we need to know the types of the parameters
        # to create the pydantic model
        generic_parameters: set[typing.TypeVar] = set()
        for parameter in parameter_dict.values():
            annotation = parameter[0]
            # unwrap any Annotated types
            while hasattr(annotation, "__metadata__"):
                annotation = annotation.__origin__
            # if the annotation is already a type variable, add it to the set
            if isinstance(annotation, typing.TypeVar):
                generic_parameters.add(annotation)
            # if the annotation is a generic type, add the parameters to the set
            if hasattr(annotation, "__parameters__"):
                generic_parameters.update(annotation.__parameters__)

        model_spec = LLMBoundSignature.field_tuples_to_model_spec(parameter_dict)
        if generic_parameters:
            bases = (BaseModel, typing.Generic[*generic_parameters])
            input_type = cached_create_model(
                f"{class_name}Inputs",
                __base__=bases,
                __module__=f.__module__,
                **model_spec,
            )
        else:
            input_type = cached_create_model(f"{class_name}Inputs", __module__=f.__module__, **model_spec)
        input_type.model_rebuild()

        # update parameter_dict types with bound_arguments
        # this ensures that we serialize the actual types
        # might not be optimal because the language model won't be aware of original types, however

        bound_arguments = LLMBoundSignature.bind(signature, args, kwargs)
        for parameter_name in parameter_dict:
            if parameter_name in bound_arguments.arguments:
                parameter_dict[parameter_name] = (
                    type(bound_arguments.arguments[parameter_name]),
                    parameter_dict[parameter_name][1],
                )

        specific_model_spec = LLMBoundSignature.field_tuples_to_model_spec(parameter_dict)
        specific_input_type = cached_create_model(
            f"Specific{class_name}Inputs",
            __module__=f.__module__,
            **specific_model_spec,
        )
        specific_input_type.model_rebuild()

        input = specific_input_type(**bound_arguments.arguments)

        llm_structured_prompt: LLMStructuredPrompt = LLMStructuredPrompt.create(
            docstring=docstring,
            input_type=input_type,
            return_annotation=return_type,
            input=input,
        )

        return LLMBoundSignature(llm_structured_prompt, signature)

    @staticmethod
    def parameter_items_to_field_tuple(
        parameters_items: list[tuple[str, inspect.Parameter]],
    ):
        """
        Get the parameter definitions for a function call from the parameters and arguments.
        """
        parameter_dict: dict = {}
        for parameter_name, parameter in parameters_items:
            # every parameter must be annotated or have a default value
            annotation = parameter.annotation
            if annotation is type:
                annotation = TyperWrapper

            if parameter.default is inspect.Parameter.empty:
                parameter_dict[parameter_name] = (annotation, ...)
            else:
                parameter_dict[parameter_name] = (annotation, parameter.default)

        return parameter_dict

    @staticmethod
    def field_tuples_to_model_spec(
        field_tuples_dict: dict[str, tuple[str, tuple[type, ...]]],
    ) -> dict[str, tuple[type, object] | object]:
        """
        Get the parameter definitions for a function call from the parameters and arguments.
        """
        parameter_dict: dict = {}
        for parameter_name, (annotation, default) in field_tuples_dict.items():
            # every parameter must be annotated or have a default value
            if default is ...:
                parameter_dict[parameter_name] = (annotation, ...)
            else:
                if annotation is not inspect.Parameter.empty:
                    parameter_dict[parameter_name] = (annotation, default)
                else:
                    parameter_dict[parameter_name] = default

        return parameter_dict

    @staticmethod
    def get_or_create_pydantic_default(field: FieldInfo):
        if field.default is not PydanticUndefined:
            if field.default is Ellipsis:
                return inspect.Parameter.empty
            return field.default
        if field.default_factory is not None:
            return field.default_factory()
        return inspect.Parameter.empty

    @staticmethod
    def bind(signature: inspect.Signature, args: tuple, kwargs: dict) -> inspect.BoundArguments:
        """
        Bind function taking into account Field definitions and defaults.

        The first parameter from the original signature is dropped (as it is the language model or chat chain).
        args and kwargs are bound to the remaining parameters.
        """
        # resolve parameter defaults to FieldInfo.default if the parameter is a field
        signature_fixed_defaults = signature.replace(
            parameters=[
                parameter.replace(default=LLMBoundSignature.get_or_create_pydantic_default(parameter.default))
                if isinstance(parameter.default, FieldInfo)
                else parameter
                for parameter in list(signature.parameters.values())[1:]
            ]
        )
        bound_arguments = signature_fixed_defaults.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return bound_arguments


class LLMFunctionInterface(typing.Generic[P, T], typing.Callable[P, T]):
    def get_structured_prompt(self, *args: P.args, **kwargs: P.kwargs) -> LLMStructuredPrompt:
        raise NotImplementedError

    def llm_bound_signature(self, *args, **kwargs) -> LLMBoundSignature:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LLMFunction(LLMFunctionInterface[P, T], typing.Generic[P, T]):
    """
    A callable that can be called with a chat model.
    """

    def __init__(self):
        self._tracked_function: TrackedFunction[P, T] | None = None

    def llm_bound_signature(self, *args, **kwargs) -> LLMBoundSignature:
        return LLMBoundSignature.from_call(self, args, kwargs)

    def get_input_object(self, *args, **kwargs) -> BaseModel:
        return self.llm_bound_signature(*args, **kwargs).get_input_object(*args, **kwargs)

    def __get__(self, instance: object, owner: type | None = None) -> typing.Callable:
        """Support instance methods."""
        if instance is None:
            return self

        # Bind self to instance as MethodType
        return types.MethodType(self, instance)

    def __getattr__(self, item):
        return getattr(self.__wrapped__, item)

    def explicit(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain,
        input_object: BaseModel,
    ):
        """Call the function with explicit inputs."""

        return self._tracked_function(language_model_or_chat_chain, **dict(input_object))

    @trace_calls(kind=TraceNodeKind.CHAIN, capture_return=slicer[1:], capture_args=True)
    def __call__(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Call the function."""
        update_name(self.__name__)

        # check that the first argument is an instance of BaseLanguageModel
        # or a TrackedChatChain or UntrackedChatChain
        if not isinstance(language_model_or_chat_chain, BaseLanguageModel | ChatChain):
            raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # We expect that we wrap a function that raises NotImplementedError
        # We call it, so we can set breakpoints in the function
        try:
            self.__wrapped__(language_model_or_chat_chain, *args, **kwargs)
            raise ValueError("The function must raise NotImplementedError.")
        except NotImplementedError:
            pass

        llm_bound_signature = LLMBoundSignature.from_call(self, args, kwargs)
        return_value = llm_bound_signature.structured_prompt(language_model_or_chat_chain)
        return return_value


class LLMExplicitFunction(LLMFunctionInterface[P, T], typing.Generic[P, T]):
    """
    A callable that can be called with a chat model.
    """

    def llm_bound_signature(self, input: BaseModel) -> LLMBoundSignature:
        """Create an LLMFunctionSpec from a function."""
        # get clean docstring of
        docstring = inspect.getdoc(self)
        if docstring is None:
            raise ValueError("The function must have a docstring.")

        # get the type of the first argument
        signature = inspect.signature(self, eval_str=True)

        # get all parameters
        parameters_items: list[tuple[str, inspect.Parameter]] = list(signature.parameters.items())
        # check that there is at least one parameter
        if not parameters_items:
            raise ValueError("The function must have at least one parameter.")
        # check that the first parameter has a type annotation that is an instance of BaseLanguageModel
        # or a TrackedChatChain
        first_parameter: inspect.Parameter = parameters_items[0][1]
        if first_parameter.annotation is not inspect.Parameter.empty:
            if not issubclass(first_parameter.annotation, BaseLanguageModel | ChatChain):
                raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        second_parameter: inspect.Parameter = parameters_items[1][1]

        llm_structured_prompt = LLMStructuredPrompt.create(
            docstring=docstring,
            input_type=second_parameter.annotation,
            return_annotation=signature.return_annotation,
            input=input,
        )
        return LLMBoundSignature(llm_structured_prompt, signature)

    def __get__(self, instance: object, owner: type | None = None) -> typing.Callable:
        """Support instance methods."""
        if instance is None:
            return self

        # Bind self to instance as MethodType
        return types.MethodType(self, instance)

    def __getattr__(self, item):
        return getattr(self.__wrapped__, item)

    @trace_calls(kind=TraceNodeKind.CHAIN, capture_return=True, capture_args=slicer[1:])
    def __call__(
        self,
        language_model_or_chat_chain: BaseLanguageModel | ChatChain,
        input: BaseModel,
    ) -> T:
        """Call the function."""
        update_name(self.__name__)

        # check that the first argument is an instance of BaseLanguageModel
        # or a TrackedChatChain or UntrackedChatChain
        if not isinstance(language_model_or_chat_chain, BaseLanguageModel | ChatChain):
            raise ValueError("The first parameter must be an instance of BaseLanguageModel or ChatChain.")

        # We expect that we wrap a function that raises NotImplementedError
        # We call it, so we can set breakpoints in the function
        try:
            self.__wrapped__(language_model_or_chat_chain, input)
            raise ValueError("The function must raise NotImplementedError.")
        except NotImplementedError:
            pass

        llm_bound_signature = self.llm_bound_signature(input)
        return_value = llm_bound_signature.structured_prompt(language_model_or_chat_chain)
        return return_value


F_types: typing.TypeAlias = (
    F
    | LLMFunction[P, T]
    | LLMExplicitFunction[P, T]
    | types.MethodType
    | types.FunctionType
    | types.ClassMethodDescriptorType
    | types.MethodDescriptorType
    | types.MemberDescriptorType
    | types.MethodWrapperType
    | LLMFunctionInterface
)


def apply_decorator(f: F_types, decorator) -> F_types:
    """
    Apply a decorator to a function.

    This function is used to apply a decorator to a function, while preserving the function type.
    This is useful when we want to apply a decorator to a function that is a classmethod, staticmethod, property,
    or a method of a class.

    Parameters
    ----------

    f: F_types
        The function to decorate.
    decorator: Callable
        The decorator to apply.

    Returns
    -------

    F_types
        The decorated function.

    Raises
    ------

    ValueError
        If the function is a classmethod, staticmethod, property, or a method of a class.
    """
    specific_llm_function: object
    if isinstance(f, classmethod):
        raise ValueError("Cannot decorate classmethod with llm_strategy (no translation of cls: type atm).")
    elif isinstance(f, staticmethod):
        specific_llm_function = staticmethod(apply_decorator(f.__func__, decorator))
    elif isinstance(f, property):
        specific_llm_function = property(apply_decorator(f.fget, decorator), doc=f.__doc__)
    elif isinstance(f, types.MethodType):
        specific_llm_function = types.MethodType(apply_decorator(f.__func__, decorator), f.__self__)
    elif hasattr(f, "__wrapped__"):
        return apply_decorator(f.__wrapped__, decorator)
    elif isinstance(f, LLMFunctionInterface):
        specific_llm_function = f
    elif not callable(f):
        raise ValueError(f"Cannot decorate {f} with llm_strategy.")
    else:
        if not is_not_implemented(f):
            raise ValueError("The function must not be implemented.")

        decorated = decorator(f)
        specific_llm_function = track_hyperparameters(functools.wraps(f)(decorated))
        decorated._tracked_function = specific_llm_function

    return typing.cast(F_types, specific_llm_function)


def llm_explicit_function(f: F_types) -> F_types:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.
    """
    return apply_decorator(f, lambda f: LLMExplicitFunction())


def llm_function(f: F_types) -> F_types:
    """
    Decorator to wrap a function with a chat model.

    f is a function to a dataclass or Pydantic model.

    The docstring of the function provides instructions for the model.
    """
    return apply_decorator(f, lambda f: LLMFunction())


def parse(text: str, output_model: type[B]) -> B:
    try:
        # Greedy search for 1st json candidate.
        match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
        json_str = ""
        if match:
            json_str = match.group()
        json_object = json.loads(json_str)
        return output_model.parse_obj(json_object)

    except (json.JSONDecodeError, ValidationError) as e:
        msg = f'Failed to parse the last reply. Expected: `{{"return_value": ...}}` Got: {e}'
        raise OutputParserException(msg)


_typing_GenericAlias = type(typing.List[int])


# TODO: move this to the hyperparameter optimizer
def get_concise_type_repr(return_type: type):
    """Return a shorter (string) representation of the return type.

    Examples:

        <class 'str'> -> str
        <class 'int'> -> int
        <class 'CustomType'> -> CustomType
        <class 'typing.List[typing.Dict[str, int]]'> -> List[Dict[str, int]]

    For generic types, we want to keep the type arguments as well.

        <class 'typing.List[typing.Dict[str, int]]'> -> List[Dict[str, int]]
        <class 'BaseModel[typing.Dict[str, int]]'> -> BaseModel[Dict[str, int]]

    For unspecialized generic types, we want to keep the type arguments as well.

        so for class BaseModel(Generic[T]): pass:
            -> BaseModel[T]
    """
    assert isinstance(return_type, type | types.GenericAlias | _typing_GenericAlias | typing.TypeVar), return_type
    name = return_type.__name__
    # is it a specialized generic type?
    if hasattr(return_type, "__origin__"):
        origin = return_type.__origin__
        if origin is not None:
            # is it a generic type with type arguments?
            if hasattr(return_type, "__args__"):
                args = return_type.__args__
                if args:
                    # is it a generic type with type arguments?
                    args_str = ", ".join([get_concise_type_repr(arg) for arg in args])
                    return f"{origin.__name__}[{args_str}]"
    # is it a unspecialized generic type?
    if hasattr(return_type, "__parameters__"):
        parameters = return_type.__parameters__
        if parameters:
            # is it a generic type without type arguments?
            parameters_str = ", ".join([get_concise_type_repr(parameter) for parameter in parameters])
            return f"{name}[{parameters_str}]"

    return name
