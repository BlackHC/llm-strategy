import typing
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import parse
import typing_extensions
from langchain.llms.base import BaseLLM

T = TypeVar("T")


@dataclass
class PromptResult(typing.Generic[T]):
    prompt: T
    response: str


TT = TypeVar("TT", bound="PromptTemplateMixin")


@dataclass
class PromptTemplateMixin:
    """
    A template for the prompt.
    """

    prompt_template: ClassVar[str]

    def render(self) -> str:
        """
        Returns a prompt string with the given arguments.
        """
        return self.__class__.prompt_template.format(**vars(self))

    @classmethod
    def parse(cls: type[TT], text: str) -> TT:
        result = parse.parse(cls.prompt_template, text)
        if result is None:
            raise ValueError(f"Could not parse prompt {text}")
        assert len(result.fixed) == 0
        assert set(result.named.keys()) == set(cls.__dataclass_fields__.keys()) - {"prompt_template"}
        prompt_instance = cls(**result.named)
        return prompt_instance

    def __call__(self, lmm: BaseLLM) -> PromptResult[typing_extensions.Self]:  # type: ignore
        response_text = lmm(self.render())
        return PromptResult(prompt=self, response=response_text)
