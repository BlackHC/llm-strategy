from dataclasses import dataclass

from llm_strategy import llm_strategy
from llm_strategy.llm_function import is_not_implemented
from llm_strategy.testing.fake_chat_model import FakeChatModel
from llm_strategy.testing.fake_llm import FakeLLM


@dataclass
class DummyInterface:
    """A dataclass with some members that are not implemented."""

    a: int = -1

    @staticmethod
    def static_method(a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    # @classmethod
    # def class_method(cls: type, a: int, b: int) -> int:
    #     """Add two numbers."""
    #     raise NotImplementedError()

    @property
    def property_getter(self: "DummyInterface") -> int:
        """Return 0."""
        raise NotImplementedError()

    def bound_method(self: "DummyInterface", a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    def bound_method_implemented(self: "DummyInterface", b: int = 1) -> int:
        """Add two numbers."""
        return self.a + b

    @staticmethod
    def static_method_implemented(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @classmethod
    def class_method_implemented(cls, a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @property
    def property_getter_implemented(self) -> int:
        """Return 1."""
        return 1


def not_implemented_function():
    raise NotImplementedError


def test_is_not_implemented_functon():
    assert is_not_implemented(not_implemented_function)
    assert not is_not_implemented(lambda: 1)


def test_llm_strategy_on_functions():
    def add_two_ints(a: int, b: int) -> int:
        """Add two integers."""
        raise NotImplementedError()

    def add_two_ints_with_default(a: int, b: int = 1) -> int:
        """Add two integers with a default value."""
        raise NotImplementedError()

    def add_two_ints_with_default_and_kwarg(*, a: int, c: int = 2) -> int:
        """Add two integers with a default value."""
        raise NotImplementedError()

    llm = FakeLLM(
        texts={
            (
                "Add two integers.\n\nThe input and output are formatted as a JSON interface that conforms to the JSON"
                ' schemas below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of'
                ' strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the object {"foo":'
                ' ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar",'
                ' "baz"]}} is not well-formatted.\n\nHere is the input schema:\n```\n{"properties": {"a": {"type":'
                ' "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}\n```\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"a":1,"b":2}\n```{"return_value": 3}\n'
            ),
            (
                "Add two integers with a default value.\n\nThe input and output are formatted as a JSON interface that"
                ' conforms to the JSON schemas below.\n\nAs an example, for the schema {"properties": {"foo":'
                ' {"description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required":'
                ' ["foo"]}} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object'
                ' {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the input'
                ' schema:\n```\n{"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required":'
                ' ["a"]}\n```\n\nHere is the output schema:\n```\n{"properties": {"return_value": {"type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"a":1,"b":1}\n```{"return_value": 2}\n'
            ),
            (
                "Add two integers with a default value.\n\nThe input and output are formatted as a JSON interface that"
                ' conforms to the JSON schemas below.\n\nAs an example, for the schema {"properties": {"foo":'
                ' {"description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required":'
                ' ["foo"]}} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object'
                ' {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the input'
                ' schema:\n```\n{"properties": {"a": {"type": "integer"}, "c": {"type": "integer"}}, "required":'
                ' ["a"]}\n```\n\nHere is the output schema:\n```\n{"properties": {"return_value": {"type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"a":1,"c":2}\n```{"return_value": 3}\n'
            ),
        },
        # external_llm=OpenAI(),
    )

    assert llm_strategy(llm)(add_two_ints)(1, 2) == 3
    assert llm_strategy(llm)(add_two_ints_with_default)(1) == 2
    assert llm_strategy(llm)(add_two_ints_with_default_and_kwarg)(a=1) == 3


def test_llm_strategy_dummy_interface():
    llm = FakeLLM(
        texts=[
            (
                "Return 0.\n\nThe input and output are formatted as a JSON interface that conforms to the JSON schemas"
                ' below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of strings",'
                ' "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the object {"foo": ["bar",'
                ' "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar",'
                ' "baz"]}} is not well-formatted.\n\nHere is the schema for additional data'
                ' types:\n```\n{"DummyInterface": {"properties": {"a": {"type": "integer"}}}}\n```\n\nHere is the input'
                ' schema:\n```\n{"properties": {"self": {"$ref": "DummyInterface"}}, "required": ["self"]}\n```\n\nHere'
                ' is the output schema:\n```\n{"properties": {"return_value": {"type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"self":{"a":-1}}\n```{"return_value": 0}\n'
            ),
            (
                "Add two numbers.\n\nThe input and output are formatted as a JSON interface that conforms to the JSON"
                ' schemas below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of'
                ' strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the object {"foo":'
                ' ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar",'
                ' "baz"]}} is not well-formatted.\n\nHere is the input schema:\n```\n{"properties": {"a": {"type":'
                ' "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}\n```\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"a":1,"b":2}\n```{"return_value": 3}\n'
            ),
            (
                "Add two numbers.\n\nThe input is formatted as a JSON interface of Inputs that conforms to the JSON "
                "schema below, and the output should be formatted as a JSON instance of Outputs that conforms to the "
                'JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", '
                '"description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ['
                '"foo"]}} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {'
                '"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the schema:\n```\n{'
                '"DummyInterface": {"properties": {"a": {"title": "A", "default": -1, "type": "integer"}}}, '
                '"Inputs": {"properties": {"self": {"$ref": "#/definitions/DummyInterface"}, "a": {"title": "A", '
                '"type": "integer"}, "b": {"title": "B", "type": "integer"}}, "required": ["self", "a", "b"]}, '
                '"Outputs": {"properties": {"return_value": {"title": "Return Value", "type": "integer"}}, '
                '"required": ["return_value"]}}\n```\n\nNow output the results for the following inputs:\n```\n{'
                '"self": {"a": -1}, "a": 1, "b": 2}\n```\n '
                '{"return_value": 3}\n'
            ),
            (
                "Add two numbers.\n\nThe input and output are formatted as a JSON interface that conforms to the JSON"
                ' schemas below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of'
                ' strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the object {"foo":'
                ' ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar",'
                ' "baz"]}} is not well-formatted.\n\nHere is the schema for additional data'
                ' types:\n```\n{"DummyInterface": {"properties": {"a": {"type": "integer"}}}}\n```\n\nHere is the input'
                ' schema:\n```\n{"properties": {"self": {"$ref": "DummyInterface"}, "a": {"type": "integer"}, "b":'
                ' {"type": "integer"}}, "required": ["self", "a", "b"]}\n```\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"self":{"a":-1},"a":1,"b":2}\n```{"return_value": 3}\n'
            ),
        ]
        # external_llm=OpenAI(),
    )

    llm_DummyInterface = llm_strategy(llm)(DummyInterface)
    llm_DummyInstance = llm_DummyInterface()
    assert llm_DummyInstance.property_getter == 0
    assert llm_DummyInterface.static_method(1, 2) == 3
    assert llm_DummyInstance.static_method(1, 2) == 3
    assert llm_DummyInstance.bound_method(1, 2) == 3
    # assert llm_DummyInterface.class_method(1, 2) == 3
    # assert llm_DummyInstance.class_method(1, 2) == 3

    assert llm_DummyInterface.static_method_implemented(1, 2) == 3
    assert llm_DummyInstance.bound_method_implemented(1) == 0
    assert llm_DummyInterface.class_method_implemented(1, 2) == 3
    assert llm_DummyInstance.property_getter_implemented == 1


@dataclass
class Customer:
    first_name: str
    last_name: str
    birthday: str
    city: str


@dataclass
class CustomerDatabase:
    customers: list[Customer]

    def find_customer_index(self: "CustomerDatabase", query: str) -> int:
        """Find the index of the customer that matches the natural language query best.

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".

        Args:
            query: Natural language query

        Returns:
            The index of the best matching customer in the database.
        """
        raise NotImplementedError()

    @staticmethod
    def create_mock_customers(num_customers: int = 1) -> list[Customer]:
        """
        Create mock customers with believable data (our customers are world citizens).
        """
        raise NotImplementedError()

    @staticmethod
    def create_mock_queries(customer_database: "CustomerDatabase", num_queries: int = 1) -> list[str]:
        """
        Create mock queries that match one of the mock customers better than the others.

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".
        """
        raise NotImplementedError()


def test_llm_strategy():
    # llm = FakeChatModel(
    #     external_chat_model=ChatOpenAI(model="gpt-4o-mini"),  # noqa: F821
    # )
    llm = FakeChatModel.from_messages(
        messages_bag=[
            [
                {
                    "type": "human",
                    "data": {
                        "content": (
                            "Find the index of the customer that matches the natural language query best.\n\nWe support"
                            ' semantic queries instead of SQL, so we can search for things like\n"the customer that was'
                            ' born in 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of'
                            " the best matching customer in the database.\n\nThe input and output are formatted as a"
                            " JSON interface that conforms to the JSON schemas below.\n\nAs an example, for the schema"
                            ' {"properties": {"foo": {"description": "a list of strings", "type": "array", "items":'
                            ' {"type": "string"}}}, "required": ["foo"]}} the object {"foo": ["bar", "baz"]} is a'
                            ' well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}}'
                            " is not well-formatted.\n\nHere is the schema for additional data"
                            ' types:\n```\n{"Customer": {"properties": {"first_name": {"type": "string"}, "last_name":'
                            ' {"type": "string"}, "birthday": {"type": "string"}, "city": {"type": "string"}},'
                            ' "required": ["first_name", "last_name", "birthday", "city"]}, "CustomerDatabase":'
                            ' {"properties": {"customers": {"items": {"$ref": "Customer"}, "type": "array"}},'
                            ' "required": ["customers"]}}\n```\n\nHere is the input schema:\n```\n{"properties":'
                            ' {"self": {"$ref": "CustomerDatabase"}, "query": {"type": "string"}}, "required": ["self",'
                            ' "query"]}\n```\n\nHere is the output schema:\n```\n{"properties": {"return_value":'
                            ' {"type": "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the'
                            " following"
                            ' inputs:\n```\n{"self":{"customers":[{"first_name":"Amina","last_name":"Khan","birthday":"1990-07-12","city":"Lahore"},{"first_name":"Elias","last_name":"Andersen","birthday":"1985-03-25","city":"Copenhagen"},{"first_name":"Sofia","last_name":"García","birthday":"1992-11-15","city":"Madrid"}]},"query":"the'
                            ' customer that was born in 1990"}\n```'
                        ),
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": '```json\n{"return_value":0}\n```',
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "token_usage": {
                                "completion_tokens": 11,
                                "prompt_tokens": 472,
                                "total_tokens": 483,
                                "completion_tokens_details": {
                                    "accepted_prediction_tokens": 0,
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "rejected_prediction_tokens": 0,
                                },
                                "prompt_tokens_details": {
                                    "audio_tokens": 0,
                                    "cached_tokens": 0,
                                },
                            },
                            "model_name": "gpt-4o-mini-2024-07-18",
                            "system_fingerprint": "fp_709714d124",
                            "finish_reason": "stop",
                            "logprobs": None,
                        },
                        "type": "ai",
                        "name": None,
                        "id": "run-dd4e88f1-6a31-4540-a218-1630e9a61771-0",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 472,
                            "output_tokens": 11,
                            "total_tokens": 483,
                            "input_token_details": {"audio": 0, "cache_read": 0},
                            "output_token_details": {"audio": 0, "reasoning": 0},
                        },
                    },
                },
            ],
            [
                {
                    "type": "human",
                    "data": {
                        "content": (
                            "Create mock queries that match one of the mock customers better than the others.\n\nWe"
                            ' support semantic queries instead of SQL, so we can search for things like\n"the customer'
                            ' that was born in 1990".\n\nThe input and output are formatted as a JSON interface that'
                            ' conforms to the JSON schemas below.\n\nAs an example, for the schema {"properties":'
                            ' {"foo": {"description": "a list of strings", "type": "array", "items": {"type":'
                            ' "string"}}}, "required": ["foo"]}} the object {"foo": ["bar", "baz"]} is a well-formatted'
                            ' instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not'
                            ' well-formatted.\n\nHere is the schema for additional data types:\n```\n{"Customer":'
                            ' {"properties": {"first_name": {"type": "string"}, "last_name": {"type": "string"},'
                            ' "birthday": {"type": "string"}, "city": {"type": "string"}}, "required": ["first_name",'
                            ' "last_name", "birthday", "city"]}, "CustomerDatabase": {"properties": {"customers":'
                            ' {"items": {"$ref": "Customer"}, "type": "array"}}, "required":'
                            ' ["customers"]}}\n```\n\nHere is the input schema:\n```\n{"properties":'
                            ' {"customer_database": {"$ref": "CustomerDatabase"}, "num_queries": {"type": "integer"}},'
                            ' "required": ["customer_database"]}\n```\n\nHere is the output'
                            ' schema:\n```\n{"properties": {"return_value": {"items": {"type": "string"}, "type":'
                            ' "array"}}, "required": ["return_value"]}\n```\nNow output the results for the following'
                            ' inputs:\n```\n{"customer_database":{"customers":[{"first_name":"Amina","last_name":"Khan","birthday":"1990-07-12","city":"Lahore"},{"first_name":"Elias","last_name":"Andersen","birthday":"1985-03-25","city":"Copenhagen"},{"first_name":"Sofia","last_name":"García","birthday":"1992-11-15","city":"Madrid"}]},"num_queries":3}\n```'
                        ),
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": (
                            '```json\n{"return_value":["the customer that was born in 1990","the customer from'
                            ' Lahore","the customer whose name starts with S"]}\n```'
                        ),
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "token_usage": {
                                "completion_tokens": 32,
                                "prompt_tokens": 453,
                                "total_tokens": 485,
                                "completion_tokens_details": {
                                    "accepted_prediction_tokens": 0,
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "rejected_prediction_tokens": 0,
                                },
                                "prompt_tokens_details": {
                                    "audio_tokens": 0,
                                    "cached_tokens": 0,
                                },
                            },
                            "model_name": "gpt-4o-mini-2024-07-18",
                            "system_fingerprint": "fp_709714d124",
                            "finish_reason": "stop",
                            "logprobs": None,
                        },
                        "type": "ai",
                        "name": None,
                        "id": "run-6afcc9a5-2104-47b5-903e-df6d805b6e3f-0",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 453,
                            "output_tokens": 32,
                            "total_tokens": 485,
                            "input_token_details": {"audio": 0, "cache_read": 0},
                            "output_token_details": {"audio": 0, "reasoning": 0},
                        },
                    },
                },
            ],
            [
                {
                    "type": "human",
                    "data": {
                        "content": (
                            "Find the index of the customer that matches the natural language query best.\n\nWe support"
                            ' semantic queries instead of SQL, so we can search for things like\n"the customer that was'
                            ' born in 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of'
                            " the best matching customer in the database.\n\nThe input and output are formatted as a"
                            " JSON interface that conforms to the JSON schemas below.\n\nAs an example, for the schema"
                            ' {"properties": {"foo": {"description": "a list of strings", "type": "array", "items":'
                            ' {"type": "string"}}}, "required": ["foo"]}} the object {"foo": ["bar", "baz"]} is a'
                            ' well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}}'
                            " is not well-formatted.\n\nHere is the schema for additional data"
                            ' types:\n```\n{"Customer": {"properties": {"first_name": {"type": "string"}, "last_name":'
                            ' {"type": "string"}, "birthday": {"type": "string"}, "city": {"type": "string"}},'
                            ' "required": ["first_name", "last_name", "birthday", "city"]}, "CustomerDatabase":'
                            ' {"properties": {"customers": {"items": {"$ref": "Customer"}, "type": "array"}},'
                            ' "required": ["customers"]}}\n```\n\nHere is the input schema:\n```\n{"properties":'
                            ' {"self": {"$ref": "CustomerDatabase"}, "query": {"type": "string"}}, "required": ["self",'
                            ' "query"]}\n```\n\nHere is the output schema:\n```\n{"properties": {"return_value":'
                            ' {"type": "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the'
                            " following"
                            ' inputs:\n```\n{"self":{"customers":[{"first_name":"Amina","last_name":"Khan","birthday":"1990-07-12","city":"Lahore"},{"first_name":"Elias","last_name":"Andersen","birthday":"1985-03-25","city":"Copenhagen"},{"first_name":"Sofia","last_name":"García","birthday":"1992-11-15","city":"Madrid"}]},"query":"the'
                            ' customer whose name starts with S"}\n```'
                        ),
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": '```json\n{"return_value": 2}\n```',
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "token_usage": {
                                "completion_tokens": 12,
                                "prompt_tokens": 470,
                                "total_tokens": 482,
                                "completion_tokens_details": {
                                    "accepted_prediction_tokens": 0,
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "rejected_prediction_tokens": 0,
                                },
                                "prompt_tokens_details": {
                                    "audio_tokens": 0,
                                    "cached_tokens": 0,
                                },
                            },
                            "model_name": "gpt-4o-mini-2024-07-18",
                            "system_fingerprint": "fp_709714d124",
                            "finish_reason": "stop",
                            "logprobs": None,
                        },
                        "type": "ai",
                        "name": None,
                        "id": "run-c9d61925-8d88-47db-89dc-a8c9f93b8f70-0",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 470,
                            "output_tokens": 12,
                            "total_tokens": 482,
                            "input_token_details": {"audio": 0, "cache_read": 0},
                            "output_token_details": {"audio": 0, "reasoning": 0},
                        },
                    },
                },
            ],
            [
                {
                    "type": "human",
                    "data": {
                        "content": (
                            "Find the index of the customer that matches the natural language query best.\n\nWe support"
                            ' semantic queries instead of SQL, so we can search for things like\n"the customer that was'
                            ' born in 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of'
                            " the best matching customer in the database.\n\nThe input and output are formatted as a"
                            " JSON interface that conforms to the JSON schemas below.\n\nAs an example, for the schema"
                            ' {"properties": {"foo": {"description": "a list of strings", "type": "array", "items":'
                            ' {"type": "string"}}}, "required": ["foo"]}} the object {"foo": ["bar", "baz"]} is a'
                            ' well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}}'
                            " is not well-formatted.\n\nHere is the schema for additional data"
                            ' types:\n```\n{"Customer": {"properties": {"first_name": {"type": "string"}, "last_name":'
                            ' {"type": "string"}, "birthday": {"type": "string"}, "city": {"type": "string"}},'
                            ' "required": ["first_name", "last_name", "birthday", "city"]}, "CustomerDatabase":'
                            ' {"properties": {"customers": {"items": {"$ref": "Customer"}, "type": "array"}},'
                            ' "required": ["customers"]}}\n```\n\nHere is the input schema:\n```\n{"properties":'
                            ' {"self": {"$ref": "CustomerDatabase"}, "query": {"type": "string"}}, "required": ["self",'
                            ' "query"]}\n```\n\nHere is the output schema:\n```\n{"properties": {"return_value":'
                            ' {"type": "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the'
                            " following"
                            ' inputs:\n```\n{"self":{"customers":[{"first_name":"Amina","last_name":"Khan","birthday":"1990-07-12","city":"Lahore"},{"first_name":"Elias","last_name":"Andersen","birthday":"1985-03-25","city":"Copenhagen"},{"first_name":"Sofia","last_name":"García","birthday":"1992-11-15","city":"Madrid"}]},"query":"the'
                            ' customer from Lahore"}\n```'
                        ),
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": '```json\n{"return_value":0}\n```',
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "token_usage": {
                                "completion_tokens": 11,
                                "prompt_tokens": 467,
                                "total_tokens": 478,
                                "completion_tokens_details": {
                                    "accepted_prediction_tokens": 0,
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "rejected_prediction_tokens": 0,
                                },
                                "prompt_tokens_details": {
                                    "audio_tokens": 0,
                                    "cached_tokens": 0,
                                },
                            },
                            "model_name": "gpt-4o-mini-2024-07-18",
                            "system_fingerprint": "fp_709714d124",
                            "finish_reason": "stop",
                            "logprobs": None,
                        },
                        "type": "ai",
                        "name": None,
                        "id": "run-68189c48-8774-42aa-b965-cf0add467855-0",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 467,
                            "output_tokens": 11,
                            "total_tokens": 478,
                            "input_token_details": {"audio": 0, "cache_read": 0},
                            "output_token_details": {"audio": 0, "reasoning": 0},
                        },
                    },
                },
            ],
            [
                {
                    "type": "human",
                    "data": {
                        "content": (
                            "Create mock customers with believable data (our customers are world citizens).\n\nThe"
                            " input and output are formatted as a JSON interface that conforms to the JSON schemas"
                            ' below.\n\nAs an example, for the schema {"properties": {"foo": {"description": "a list of'
                            ' strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}} the'
                            ' object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object'
                            ' {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the schema for'
                            ' additional data types:\n```\n{"Customer": {"properties": {"first_name": {"type":'
                            ' "string"}, "last_name": {"type": "string"}, "birthday": {"type": "string"}, "city":'
                            ' {"type": "string"}}, "required": ["first_name", "last_name", "birthday",'
                            ' "city"]}}\n```\n\nHere is the input schema:\n```\n{"properties": {"num_customers":'
                            ' {"type": "integer"}}}\n```\n\nHere is the output schema:\n```\n{"properties":'
                            ' {"return_value": {"items": {"$ref": "Customer"}, "type": "array"}}, "required":'
                            ' ["return_value"]}\n```\nNow output the results for the following'
                            ' inputs:\n```\n{"num_customers":3}\n```'
                        ),
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False,
                    },
                },
                {
                    "type": "ai",
                    "data": {
                        "content": (
                            '```json\n{\n  "return_value": [\n    {\n      "first_name": "Amina",\n      "last_name":'
                            ' "Khan",\n      "birthday": "1990-07-12",\n      "city": "Lahore"\n    },\n    {\n     '
                            ' "first_name": "Elias",\n      "last_name": "Andersen",\n      "birthday": "1985-03-25",\n'
                            '      "city": "Copenhagen"\n    },\n    {\n      "first_name": "Sofia",\n     '
                            ' "last_name": "García",\n      "birthday": "1992-11-15",\n      "city": "Madrid"\n    }\n '
                            " ]\n}\n```"
                        ),
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "token_usage": {
                                "completion_tokens": 142,
                                "prompt_tokens": 287,
                                "total_tokens": 429,
                                "completion_tokens_details": {
                                    "accepted_prediction_tokens": 0,
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "rejected_prediction_tokens": 0,
                                },
                                "prompt_tokens_details": {
                                    "audio_tokens": 0,
                                    "cached_tokens": 0,
                                },
                            },
                            "model_name": "gpt-4o-mini-2024-07-18",
                            "system_fingerprint": "fp_7fcd609668",
                            "finish_reason": "stop",
                            "logprobs": None,
                        },
                        "type": "ai",
                        "name": None,
                        "id": "run-ca897a2a-b9b9-4a34-885d-210893c175b8-0",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 287,
                            "output_tokens": 142,
                            "total_tokens": 429,
                            "input_token_details": {"audio": 0, "cache_read": 0},
                            "output_token_details": {"audio": 0, "reasoning": 0},
                        },
                    },
                },
            ],
        ]
    )

    LLMCustomerDatabase = llm_strategy(llm)(CustomerDatabase)

    customers = LLMCustomerDatabase.create_mock_customers(3)

    customer_database = LLMCustomerDatabase(customers)

    mock_queries = LLMCustomerDatabase.create_mock_queries(customer_database, 3)

    print(customer_database)

    for query in mock_queries:
        index = customer_database.find_customer_index(query)
        print(query)
        print(customer_database.customers[index])
        assert index >= 0
        assert index < len(customers)
