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

from typing import Collection, List, Mapping, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    message_to_dict,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field


def to_hashable(d: object) -> object:
    """Convert a dict to a tuple of tuples, sorted by key."""
    # Convert values that are dicts to tuples as well.
    if isinstance(d, Mapping):
        return tuple(sorted((k, to_hashable(v)) for k, v in d.items()))
    elif isinstance(d, (list, set)):
        return tuple(to_hashable(v) for v in d)
    else:
        return d


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

    messages_tuples_bag: dict[tuple, list[dict]] = Field(default_factory=dict)
    """The texts to return on call."""
    external_chat_model: BaseChatModel | None = None
    """An external LLM to use if the query is not found."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    @staticmethod
    def from_messages(
        messages_bag: Collection[list[BaseMessage | dict]],
    ) -> "FakeChatModel":
        """Create a FakeChatModel from a list of messages."""
        messages_dict_bag = [
            [message_to_dict(m) if isinstance(m, BaseMessage) else m for m in messages] for messages in messages_bag
        ]
        messages_tuples_bag = {tuple(to_hashable(m) for m in messages): messages for messages in messages_dict_bag}
        return FakeChatModel(messages_tuples_bag=messages_tuples_bag)

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> ChatResult:
        raise NotImplementedError

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        raise NotImplementedError

    def __del__(self) -> None:
        # If we have an external LLM, we write out all our responses to stdout so that they can be copied into the
        # constructor call for the next run by hand if needed.
        if self.external_chat_model is not None:
            # Deduplicate the messages (any shared prefixes can be removed)
            deduplicated_keys = {
                messages
                for messages in self.messages_tuples_bag
                if not any(is_prefix_list(messages, other) for other in self.messages_tuples_bag if other != messages)
            }
            deduplicated_messages = [self.messages_tuples_bag[key] for key in deduplicated_keys]
            print(f"messages_bag = {deduplicated_messages!r}")

    def invoke(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> BaseMessage:
        """Return the query if it exists, else print the code to update the query."""
        assert stop is None, "Stop words are not supported for FakeChatModel."

        messages_tuple = tuple(to_hashable(m) for m in messages_to_dict(messages))

        for cached_tuple, cached_messages in self.messages_tuples_bag.items():
            if is_prefix_list(messages_tuple, cached_tuple):
                # check that the next message is an AIMessage
                if len(cached_tuple) == len(messages_tuple):
                    raise ValueError("No response found in messages_bag.")
                next_message = messages_from_dict([cached_messages[len(messages_tuple)]])[0]
                if not isinstance(next_message, AIMessage):
                    raise ValueError("No response found in messages_bag.")
                return next_message

        if self.external_chat_model is not None:
            message = self.external_chat_model.invoke(messages, stop=stop, **kwargs)
            message_tuple = to_hashable(messages_to_dict([message])[0])
            self.messages_tuples_bag[tuple(list(messages_tuple) + [message_tuple])] = messages_to_dict(
                list(messages) + [message]
            )

            return message

        # If no queries are provided, print the code to update the query
        code_snippet = f"""# Add the following to the queries dict:
{messages!r}, # TODO: Append the correct response here
"""
        print(code_snippet)
        raise NotImplementedError("No query provided. Add the following to the queries dict:\n\n" + code_snippet)

    __call__ = invoke
