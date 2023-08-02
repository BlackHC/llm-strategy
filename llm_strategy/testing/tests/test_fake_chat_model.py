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

import pytest
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from llm_strategy.testing import fake_chat_model


def test_fake_chat_model_query():
    """Test that the fake LLM returns the correct query."""
    chat_model = fake_chat_model.FakeChatModel.from_messages(
        [[SystemMessage(content="foo"), HumanMessage(content="bar"), AIMessage(content="doo")]]
    )
    assert chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")]) == AIMessage(content="doo")


def test_fake_chat_model_query_with_stop_raise():
    """Test that the fake LLM returns the correct query."""
    chat_model = fake_chat_model.FakeChatModel.from_messages(
        [[SystemMessage(content="foo"), HumanMessage(content="bar"), AIMessage(content="doo")]]
    )

    with pytest.raises(AssertionError):
        chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")], stop=["a"])


def test_chat_model_llm_missing_query():
    """Test that the fake LLM raises an error if the query is missing."""
    chat_model = fake_chat_model.FakeChatModel(messages_tuples_bag=set())
    with pytest.raises(NotImplementedError):
        chat_model([SystemMessage(content="foo"), HumanMessage(content="bar")])
