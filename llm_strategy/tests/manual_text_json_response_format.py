# Add a manual test that makes sure that we can use the json response format for OpenAI
import openai

# %%
from langchain_community.chat_models import ChatOpenAI

from llm_strategy.chat_chain import ChatChain

base_llm = ChatOpenAI(max_tokens=64, model_name="gpt-3.5-turbo-1106")
# %%

chain = ChatChain(base_llm, [])
try:
    response = chain.query("What is 1+1? Give your reasoning, too.", model_args=chain.enforce_json_response())
except openai.BadRequestError as e:
    assert e.code == 400
    assert "must contain the word 'json'" in e.message
else:
    raise AssertionError("Expected an exception")
