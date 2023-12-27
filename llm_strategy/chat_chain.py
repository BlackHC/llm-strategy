import dataclasses
import typing
from dataclasses import dataclass
from typing import Tuple, cast

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseMessage, HumanMessage
from pydantic import BaseModel, create_model

T = typing.TypeVar("T")
B = typing.TypeVar("B", bound=BaseModel)


@dataclass
class ChatChain:
    chat_model: BaseChatModel
    messages: list[BaseMessage]

    @property
    def response(self) -> str:
        assert len(self.messages) >= 1
        return cast(str, self.messages[-1].content)

    def append(self, messages: list[BaseMessage]) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages + messages)

    def __add__(self, other: list[BaseMessage]) -> "ChatChain":
        return self.append(other)

    def query(self, question: str, model_args: dict | None = None) -> Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        messages = self.messages + [HumanMessage(content=question)]
        model_args = model_args or {}
        reply = self.chat_model.invoke(messages, **model_args)
        messages.append(reply)
        return cast(str, reply.content), dataclasses.replace(self, messages=messages)

    def enforce_json_response(self, model_args: dict | None = None) -> dict:
        model_args = model_args or {}
        # Check if the language model is of type "openai" and extend model args with a response format in that case
        model_dict = self.chat_model.dict()
        if "openai" in model_dict["_type"] and model_dict.get("model_name") in (
            "gpt-4-1106-preview",
            "gpt-3.5-turbo-1106",
        ):
            model_args = {**model_args, "response_format": dict(type="json_object")}
        return model_args

    def structured_query(
        self, question: str, return_type: type[B], model_args: dict | None = None
    ) -> Tuple[B, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:

        if typing.get_origin(return_type) is typing.Annotated:
            return_info = typing.get_args(return_type)
        else:
            return_info = (return_type, ...)

        output_model = create_model("StructuredOutput", result=return_info)
        parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=output_model)
        question_and_formatting = question + "\n\n" + parser.get_format_instructions()

        reply_content, chain = self.query(question_and_formatting, **self.enforce_json_response(model_args))
        parsed_reply: B = typing.cast(B, parser.parse(reply_content))

        return parsed_reply, chain

    def branch(self) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages.copy())
