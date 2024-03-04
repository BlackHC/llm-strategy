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

from typing import List, Optional

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    Generation,
)


class CachedChatOpenAI(ChatOpenAI):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                assert len(results) == 1
                result: Generation = results[0]
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text))],
                    llm_output=result.generation_info,
                )
                return chat_result
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            assert len(chat_result.generations) == 1
            result = Generation(text=chat_result.generations[0].message.content, generation_info=chat_result.llm_output)
            langchain.llm_cache.update(messages_prompt, self.model_name, [result])
        return chat_result
