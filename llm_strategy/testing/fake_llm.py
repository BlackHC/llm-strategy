from typing import Any, Mapping

from langchain.llms.base import LLM, BaseLLM
from pydantic import BaseModel


class FakeLLM(LLM, BaseModel):
    """Fake LLM wrapper for testing purposes.

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

    texts: set[str] = set()
    """The texts to return on call."""
    external_llm: BaseLLM | None = None
    """An external LLM to use if the query is not found."""

    def __del__(self) -> None:
        # If we have an external LLM, we write out all our responses to stdout so that they can be copied into the
        # constructor call for the next run by hand if needed.
        if self.external_llm is not None:
            # Deduplicate the texts (any shared prefixes can be removed)
            self.texts = {
                text for text in self.texts if not any(other.startswith(text) for other in self.texts if other != text)
            }
            print(f"texts = {self.texts!r}")

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        """Return the query if it exists, else print the code to update the query."""
        for text in self.texts:
            if text.startswith(prompt):
                # Remainder:
                response = text[len(prompt) :]

                # Emulate stop behavior
                if stop is not None:
                    for stop_word in stop:
                        if stop_word in response:
                            # Only return the answer up to the stop word
                            response = response[: response.index(stop_word)]
                return response

        if self.external_llm is not None:
            response = self.external_llm(prompt, stop=stop)
            text = prompt + response
            self.texts.add(text)
            return response

        # If no queries are provided, print the code to update the query
        code_snippet = f"""# Add the following to the queries dict:
{prompt!r}, # TODO: Append the correct response here
"""
        raise NotImplementedError("No query provided. Add the following to the queries dict:\n\n" + code_snippet)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
