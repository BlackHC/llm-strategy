#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, Collection, Iterable, List, Mapping, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    messages_from_dict,
    messages_to_dict,
)
from pydantic import BaseModel, Field


def dict_to_tuple(d: Mapping[Any, Any]) -> tuple[tuple[Any, Any], ...]:
    """Convert a dict to a tuple of tuples, sorted by key."""
    # Convert values that are dicts to tuples as well.
    return tuple(sorted((k, dict_to_tuple(v) if isinstance(v, dict) else v) for k, v in d.items()))


def tuple_to_dict(t: Iterable[tuple[Any, Any]]) -> dict[Any, Any]:
    """Convert a tuple of tuples to a dict."""
    return {k: tuple_to_dict(v) if isinstance(v, tuple) else v for k, v in t}


def is_prefix_list(prefix_candidate: Collection, messages: Collection) -> bool:
    """Return whether `prefix_candidate` is a prefix of `messages`."""
    if len(prefix_candidate) > len(messages):
        return False
    for prefix_message, message in zip(prefix_candidate, messages):
        if prefix_message != message:
            return False
    return True


class FakeChatModel(BaseChatModel, BaseModel):
    """Fake ChatModel wrapper for testing purposes.

    We can use this to test the behavior of the LLM wrapper without having to actually call the LLM.

    We support an `external_llm` argument, which is an LLM that will be called if the query is not found in the `texts`
    dict. We store the responses. On exit, we deduplicate them and print them to stdout so that they can be copied into
    constructor call for the next run by hand if needed.

    We support stop words, which are words that are removed from the response if they are found. To do so, we store
    the full response (as it is build over time) and return the part before the query and the stop word.

    This also means that there is no non-determinism in the output, which is good for testing, but bad for variance.
    Especially if we want to test the behavior of the LLM wrapper when the LLM is not deterministic. (Create different
    outputs for different calls, for example.)
    """

    messages_tuples_bag: set[tuple] = Field(default_factory=set)
    """The texts to return on call."""
    external_chat_model: BaseChatModel | None = None
    """An external LLM to use if the query is not found."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    @staticmethod
    def from_messages(messages_bag: Collection[list[BaseMessage]]) -> "FakeChatModel":
        messages_tuples_bag = {tuple(dict_to_tuple(m) for m in messages_to_dict(messages)) for messages in messages_bag}
        return FakeChatModel(messages_tuples_bag=messages_tuples_bag)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None) -> ChatResult:
        raise NotImplementedError

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        raise NotImplementedError

    def __del__(self) -> None:
        # If we have an external LLM, we write out all our responses to stdout so that they can be copied into the
        # constructor call for the next run by hand if needed.
        if self.external_chat_model is not None:
            # Deduplicate the messages (any shared prefixes can be removed)
            self.messages_tuples_bag = {
                messages
                for messages in self.messages_tuples_bag
                if not any(is_prefix_list(messages, other) for other in self.messages_tuples_bag if other != messages)
            }
            print(f"messages_bag = {self.messages_tuples_bag!r}")

    def invoke(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> BaseMessage:
        """Return the query if it exists, else print the code to update the query."""
        assert stop is None, "Stop words are not supported for FakeChatModel."

        messages_tuple = tuple(dict_to_tuple(m) for m in messages_to_dict(messages))

        for cached_messages in self.messages_tuples_bag:
            if is_prefix_list(messages_tuple, cached_messages):
                # check that the next message is an AIMessage
                if len(cached_messages) == len(messages_tuple):
                    raise ValueError("No response found in messages_bag.")
                next_message = messages_from_dict([tuple_to_dict(cached_messages[len(messages)])])[0]
                if not isinstance(next_message, AIMessage):
                    raise ValueError("No response found in messages_bag.")
                return next_message

        if self.external_chat_model is not None:
            message = self.external_chat_model.invoke(messages, stop=stop, **kwargs)
            message_tuple = dict_to_tuple(messages_to_dict([message])[0])
            self.messages_tuples_bag.add(tuple(list(messages_tuple) + [message_tuple]))
            return message

        # If no queries are provided, print the code to update the query
        code_snippet = f"""# Add the following to the queries dict:
{messages!r}, # TODO: Append the correct response here
"""
        print(code_snippet)
        raise NotImplementedError("No query provided. Add the following to the queries dict:\n\n" + code_snippet)

    __call__ = invoke
