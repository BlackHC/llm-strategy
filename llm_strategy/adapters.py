from typing import List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.schema import AIMessage, BaseMessage, ChatMessage, ChatResult, LLMResult


class ChatModelAsLLM(BaseLLM):
    chat_model: BaseChatModel

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, track: bool = False) -> str:
        response = self.chat_model.call_as_llm(prompt, stop)
        return response

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        raise NotImplementedError()

    @property
    def _llm_type(self) -> str:
        return self.chat_model.__repr_name__().lower()


class LLMAsChatModel(BaseChatModel):
    llm: BaseLLM

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

    def __call__(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> BaseMessage:
        prompt = self.convert_messages_to_prompt(messages)
        stop = [] if stop is None else list(stop)
        response = self.llm(prompt, ["<|im_end|>"] + stop)
        return AIMessage(content=response)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        raise NotImplementedError()

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        raise NotImplementedError()
