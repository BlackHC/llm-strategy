"""
Defines a protocol for invoking a function as a black-box interface with a dataclass as input and output.
"""
import dataclasses
import inspect
import typing
from dataclasses import dataclass

import typing_extensions
import yaml
from langchain.llms.base import BaseLLM

from llm_strategy.dataclasses_schema import (
    DataclassesSchema,
    deserialize_yaml,
    pretty_type_str,
)
from llm_strategy.prompt_doc_string import extract_prompt_template
from llm_strategy.prompt_template import PromptTemplateMixin

# A type variable that must be a user-defined class?
T = typing.TypeVar("T")
P = typing_extensions.ParamSpec("P")


# TODO(blackhc): include the function name, and include dataclass docstrings in the schema


@dataclass
class TaskPrompt(PromptTemplateMixin):
    """
    A prompt that asks the user to execute a function with inputs and return the output.

    The function is described via a doc string. The prompt is generated from the doc string and the input
    and output types and a schema of all dataclasses that are used.
    """

    description: str
    inputs: str
    dataclasses_schema: str
    input_types: str
    output_type: str

    prompt_template = """Execute the following function that is described via a doc string:

{description}

# Task

Execute the function with the inputs that follow in the next section and finally return the output using the output type
as YAML document in an # Output section. (If the value is a literal, then just write the value. We parse the text in the
# Output section using `yaml.safe_load` in Python.)
{dataclasses_schema}
# Input Types

{input_types}

# Inputs

{inputs}

# Output Type

{output_type}

# Execution Scratch-Pad (Think Step by Step)

"""

    def get_raw_response(self, llm: BaseLLM) -> str:
        """Parse the outputs of a function call.

        Args:
            llm: The LLM to use to parse the outputs.
        """

        raw_response = llm(self.render())
        return raw_response


def unwrap_function(f: typing.Callable[P, T]) -> typing.Callable[P, T]:
    # is f a property?
    if isinstance(f, property):
        f = f.fget
    # is f a wrapped function?
    elif hasattr(f, "__wrapped__"):
        f = unwrap_function(f.__wrapped__)
    elif inspect.ismethod(f):
        f = f.__func__
    else:
        return f

    return unwrap_function(f)


@dataclass
class LLMCall:
    """
    A call to a LLM. This wraps a function with an LLM execution context.
    """

    dataclasses_schema: DataclassesSchema
    prompt_template: str
    signature: inspect.Signature
    __wrapped__: typing.Callable
    llm: BaseLLM

    @staticmethod
    def wrap_callable(
        f: typing.Callable, llm: BaseLLM, parent_dataclasses_schema: DataclassesSchema | None = None
    ) -> "LLMCall":
        """Create an LLMCall from a function."""
        unwrapped_f = unwrap_function(f)

        docstring = f.__doc__
        if docstring is None:
            raise ValueError(f"{f} does not have a docstring.")

        prompt_template: str = extract_prompt_template(docstring) or docstring
        signature = inspect.Signature.from_callable(unwrapped_f, eval_str=True)

        dataclasses_schema = DataclassesSchema.extend_parent(parent_dataclasses_schema)
        dataclasses_schema.add_return_annotation(signature)

        llm_call = LLMCall(
            dataclasses_schema=dataclasses_schema,
            prompt_template=prompt_template,
            signature=signature,
            __wrapped__=f,
            llm=llm,
        )

        return llm_call

    @staticmethod
    def _seralize_input(input_value: object) -> object:
        """Serialize an input value to a string."""
        if isinstance(input_value, type):
            return str(input_value)
        elif dataclasses.is_dataclass(input_value):
            return dataclasses.asdict(input_value)
        else:
            return input_value

    def __call__(self, *args, **kwargs):  # type: ignore
        bound_inputs = self.signature.bind(*args, **kwargs)
        bound_inputs.apply_defaults()
        inputs = {name: self._seralize_input(value) for name, value in bound_inputs.arguments.items()}
        inputs_yaml_block = yaml.safe_dump(inputs)

        input_types = {name: pretty_type_str(type(value)) for name, value in bound_inputs.arguments.items()}
        input_types_yaml_block = yaml.safe_dump(input_types)

        runtime_schema = DataclassesSchema.extend_parent(self.dataclasses_schema)
        runtime_schema.add_bound_arguments(bound_inputs)

        if runtime_schema.definitions:
            runtime_schema_yaml = yaml.safe_dump(dict(types=dict(runtime_schema.definitions)))  # # noqa: C408
            runtime_schema_text = f"\n# Dataclasses Schema\n\n{runtime_schema_yaml}\n"
        else:
            runtime_schema_text = ""

        output_type_text = pretty_type_str(self.signature.return_annotation)

        task_prompt = TaskPrompt(
            description=self.prompt_template,
            inputs=inputs_yaml_block,
            input_types=input_types_yaml_block,
            dataclasses_schema=runtime_schema_text,
            output_type=output_type_text,
        )

        rendered_prompt = task_prompt.render()

        # Get the output
        cot_answer = self.llm(rendered_prompt, stop=["# Output"])
        # Get the YAML
        yaml_block_prefix = "---\nresult:"
        yaml_prompt = task_prompt.render() + cot_answer + "# Output\n\n" + yaml_block_prefix
        yaml_part = self.llm(yaml_prompt)
        yaml_dict = yaml.safe_load(yaml_block_prefix + yaml_part + "\n")["result"]

        result = deserialize_yaml(yaml_dict, self.signature.return_annotation)
        return result


def llm_implement(
    f: typing.Callable[P, T], llm: BaseLLM, parent_dataclasses_schema: DataclassesSchema | None = None
) -> typing.Callable[P, T]:
    """Create an LLMCall from a function."""
    return LLMCall.wrap_callable(f, llm, parent_dataclasses_schema)
