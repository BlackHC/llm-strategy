import typing
from typing import List, Optional

from langchain.schema import BaseMessage, ChatMessage, ChatResult, LLMResult
from langchain_core.language_models import BaseChatModel, BaseLanguageModel, BaseLLM
from llmtracer import TraceNodeKind, trace_calls, trace_object_converter
from llmtracer.object_converter import ObjectConverter
from pydantic import BaseModel, Field

from llm_strategy.chat_chain import ChatChain

T = typing.TypeVar("T")
P = typing.ParamSpec("P")

LangchainInterface: typing.TypeAlias = BaseLanguageModel | ChatChain
LI = typing.TypeVar("LI", BaseLanguageModel, ChatChain)


class ChatTree(BaseModel):
    message: BaseMessage | None
    children: list["ChatTree"]

    @staticmethod
    def create_root():
        return ChatTree(message=None, children=[])

    def insert(self, messages: list[BaseMessage]) -> "ChatTree":
        node = self
        for message in messages:
            for child in node.children:
                if child.message == message:
                    node = child
                    break
            else:
                new_node = ChatTree(message=message, children=[])
                node.children.append(new_node)
                node = new_node
        return node

    def build_compact_dict(self) -> dict:
        """
        Returns a compact JSON representation of the chat tree.
        If we only have one child, we concatenate the messages until we hit a child with more than one child.

        We include the role of the message in the JSON.
        """
        node = self
        messages_json = []
        while True:
            message = node.message
            if message is not None:
                if message.type == "human":
                    role = "user"
                elif message.type == "ai":
                    role = "assistant"
                elif message.type == "system":
                    role = "system"
                elif message.type == "chat":
                    assert isinstance(message, ChatMessage)
                    role = message.role
                else:
                    raise ValueError(f"Unknown message type {message.type}")
                messages_json.append({"role": role, "content": message.content})
            if len(node.children) == 1:
                node = node.children[0]
            else:
                break

        return {
            "messages": messages_json,
            "children": [child.build_compact_dict() for child in node.children],
        }


class PromptTree(BaseModel):
    fragment: str = ""
    children: list["PromptTree"]

    @staticmethod
    def create_root():
        return PromptTree(message=None, children=[])

    def insert(self, text: str) -> "PromptTree":
        node = self
        while len(text):
            for child in node.children:
                if text.startswith(child.fragment):
                    node = child
                    text = text.removeprefix(child.fragment)
                    break
            else:
                new_node = PromptTree(fragment=text, children=[])
                node.children.append(new_node)
                return new_node
        return node

    def build_compact_dict(self) -> dict:
        """
        Returns a compact JSON representation of the chat chain.
        If we only have one child, we concatenate the messages until we hit a child with more than one child.

        We include the role of the message in the JSON.
        """
        node = self
        fragments_json = []
        while True:
            if len(node.fragment):
                fragments_json.append(node.fragment)
            if len(node.children) == 1:
                node = node.children[0]
            else:
                break

        return {
            "fragments": fragments_json,
            "children": [child.build_compact_dict() for child in node.children],
        }


class TrackedLLM(BaseLLM):
    llm: BaseLLM
    tracked_prompts: PromptTree = Field(default_factory=PromptTree.create_root)

    @trace_calls(name="TrackedLLM", kind=TraceNodeKind.LLM, capture_args=True, capture_return=True)
    def invoke(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        node = self.tracked_prompts.insert(prompt)
        response = self.llm.invoke(prompt, stop, **kwargs)
        node.insert(response)
        return response

    __call__ = invoke

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        raise NotImplementedError()

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type


class TrackedChatModel(BaseChatModel):
    chat_model: BaseChatModel
    tracked_chats: ChatTree = Field(default_factory=ChatTree.create_root)

    @property
    def _llm_type(self) -> str:
        return self.chat_model._llm_type

    @trace_calls(name="TrackedChatModel", kind=TraceNodeKind.LLM, capture_args=True, capture_return=True)
    def invoke(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        response_message = self.chat_model.invoke(messages, stop, **kwargs)
        self.tracked_chats.insert(messages + [response_message])
        return response_message

    __call__ = invoke

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        chat_result = self.chat_model._generate(messages, stop, **kwargs)
        node = self.tracked_chats.insert(messages)
        for generation in chat_result.generations:
            node.insert([generation.message])
        return chat_result

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        chat_result = await self.chat_model._agenerate(messages, stop, **kwargs)
        node = self.tracked_chats.insert(messages)
        for generation in chat_result.generations:
            node.insert([generation.message])
        return chat_result


def track_langchain(language_model_or_chat_chain: LI) -> LI:
    if isinstance(language_model_or_chat_chain, ChatChain):
        return ChatChain(
            chat_model=TrackedChatModel(chat_model=language_model_or_chat_chain.chat_model),
            messages=language_model_or_chat_chain.messages,
        )
    elif isinstance(language_model_or_chat_chain, BaseLLM):
        return TrackedLLM(llm=language_model_or_chat_chain)
    elif isinstance(language_model_or_chat_chain, BaseChatModel):
        return TrackedChatModel(language_model=language_model_or_chat_chain)
    else:
        raise ValueError(f"Unknown language model type {type(language_model_or_chat_chain)}")


def get_tracked_chats(chat_model_or_chat_chain: ChatChain | TrackedChatModel) -> dict:
    if isinstance(chat_model_or_chat_chain, ChatChain):
        model = chat_model_or_chat_chain.chat_model
    elif isinstance(chat_model_or_chat_chain, TrackedChatModel):
        model = chat_model_or_chat_chain
    else:
        raise ValueError(f"Unknown language model type {type(chat_model_or_chat_chain)}")
    return model.tracked_chats.build_compact_dict()["children"]


@trace_object_converter.register_converter()
def _convert_llm(llm: BaseLLM, converter: ObjectConverter) -> dict:
    return converter(llm.dict(), converter)  # type: ignore


@trace_object_converter.register_converter()
def _convert_chat_model(chat_model: BaseChatModel, converter: ObjectConverter) -> dict:
    return dict(_type=chat_model.__class__.__name__)


@trace_object_converter.register_converter()
def _convert_tracked_chat_model(chat_model: TrackedChatModel, converter: ObjectConverter) -> dict:
    return dict(_type=chat_model.chat_model.__class__.__name__)
