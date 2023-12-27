from typing import Any, Dict, List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema import AIMessage, BaseMessage, ChatMessage, ChatResult, LLMResult
from typing_extensions import override


class ChatModelAsLLM(BaseLLM):
    chat_model: BaseChatModel

    @override
    def dict(self, **kwargs: Any) -> Dict:
        return self.chat_model.dict()

    @override
    def invoke(self, prompt: str, *, stop: Optional[List[str]] = None, **kwargs) -> str:
        response = self.chat_model.call_as_llm(prompt, stop=stop, **kwargs)
        return response

    __call__ = invoke

    @override
    def _generate(self, prompts: List[str], *, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        raise NotImplementedError()

    @override
    async def _agenerate(self, prompts: List[str], *, stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        raise NotImplementedError()

    @override
    @property
    def _llm_type(self) -> str:
        return self.chat_model._llm_type


class LLMAsChatModel(BaseChatModel):
    llm: BaseLLM

    @override
    def dict(self, **kwargs: Any) -> Dict:
        return self.llm.dict()

    @staticmethod
    def convert_messages_to_prompt(messages: list[BaseMessage]) -> str:
        prompt = ""
        for message in messages:
            if message.type == "human":
                role = "user"
            elif message.type == "ai":
                role = "assistant"
            elif message.type == "system":
                role = "system"
            elif message.type == "chat":
                assert isinstance(message, ChatMessage)
                role = message.role.capitalize()
            else:
                raise ValueError(f"Unknown message type {message.type}")
            prompt += f"<|im_start|>{role}\n{message.content}<|im_end|>"
        prompt += "<|im_start|>assistant\n"
        return prompt

    @override
    @property
    def _llm_type(self) -> str:
        return self.llm._llm_type

    @override
    def invoke(self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs) -> BaseMessage:
        prompt = self.convert_messages_to_prompt(messages)
        stop = [] if stop is None else list(stop)
        response = self.llm.invoke(prompt, stop=["<|im_end|>"] + stop, **kwargs)
        return AIMessage(content=response)

    __call__ = invoke

    @override
    def _generate(self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        raise NotImplementedError()

    @override
    async def _agenerate(
        self, messages: List[BaseMessage], *, stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        raise NotImplementedError()
