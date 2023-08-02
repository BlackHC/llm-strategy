from langchain.schema import AIMessage, HumanMessage

from llm_strategy import adapters
from llm_strategy.testing import fake_chat_model, fake_llm


def test_chat_model_as_llm():
    chat_model = fake_chat_model.FakeChatModel.from_messages(
        [[HumanMessage(content="Hello", additional_kwargs={}), AIMessage(content="World", additional_kwargs={})]]
    )

    chat_model_as_llm = adapters.ChatModelAsLLM(chat_model=chat_model)

    assert chat_model_as_llm("Hello") == "World"


def test_llm_as_chat_model():
    llm = fake_llm.FakeLLM(texts={"<|im_start|>user\nHello<|im_end|><|im_start|>assistant\nWorld<|im_end|>"})

    chat_model_as_llm = adapters.LLMAsChatModel(llm=llm)

    assert chat_model_as_llm([HumanMessage(content="Hello")]) == AIMessage(content="World")
