import dataclasses
import typing
from dataclasses import dataclass
from typing import Tuple

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseMessage, HumanMessage
from pydantic import create_model

T = typing.TypeVar("T")


@dataclass
class ChatChain:
    chat_model: BaseChatModel
    messages: list[BaseMessage]

    @property
    def response(self):
        assert len(self.messages) >= 1
        return self.messages[-1].content

    def append(self, messages: list[BaseMessage]) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages + messages)

    # overload operator +
    def __add__(self, other: list[BaseMessage]) -> "ChatChain":
        return self.append(other)

    def query(self, question: str) -> Tuple[str, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:
        messages = self.messages + [HumanMessage(content=question)]
        reply = self.chat_model(messages)
        messages.append(reply)
        return reply.content, dataclasses.replace(self, messages=messages)

    def structured_query(self, question: str, return_type: type[T]) -> Tuple[T, "ChatChain"]:
        """Asks a question and returns the result in a single block."""
        # Build messages:

        if typing.get_origin(return_type) is typing.Annotated:
            return_info = typing.get_args(return_type)
        else:
            return_info = (return_type, ...)

        output_model = create_model("StructuredOutput", result=return_info)  # type: ignore
        parser = PydanticOutputParser(pydantic_object=output_model)
        question_and_formatting = question + "\n\n" + parser.get_format_instructions()
        reply_content, chain = self.query(question_and_formatting)
        parsed_reply = parser.parse(reply_content)

        return parsed_reply, chain

    def branch(self) -> "ChatChain":
        return dataclasses.replace(self, messages=self.messages.copy())
