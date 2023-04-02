from dataclasses import dataclass

from llm_strategy import llm_strategy
from llm_strategy.llm_function import is_not_implemented
from llm_strategy.testing.fake_llm import FakeLLM


@dataclass
class DummyInterface:
    """A dataclass with some members that are not implemented."""

    a: int = -1

    @staticmethod
    def static_method(a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    @classmethod
    def class_method(cls: type, a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

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
                'Add two integers.\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties":'
                ' {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}, "required": ["a",'
                ' "b"]}\n```\n\nThe output should be formatted as a JSON instance that conforms to the JSON schema'
                ' below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a'
                ' list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}\nthe object'
                ' {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo":'
                ' ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties":'
                ' {"return_value": {"title": "Return Value", "type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"a": 1, "b":'
                ' 2}\n```\n{"return_value": 3}\n'
            ),
            (
                "Add two integers with a default value.\nThe inputs are formatted as JSON using the following"
                ' schema:\n```\n{"properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "default":'
                ' 1, "type": "integer"}}, "required": ["a"]}\n```\n\nThe output should be formatted as a JSON instance'
                ' that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo":'
                ' {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}},'
                ' "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.'
                ' The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"a": 1,'
                ' "b": 1}\n```\n{"return_value": 2}\n'
            ),
            (
                "Add two integers with a default value.\nThe inputs are formatted as JSON using the following"
                ' schema:\n```\n{"properties": {"a": {"title": "A", "type": "integer"}, "c": {"title": "C", "default":'
                ' 2, "type": "integer"}}, "required": ["a"]}\n```\n\nThe output should be formatted as a JSON instance'
                ' that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo":'
                ' {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}},'
                ' "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema.'
                ' The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"a": 1,'
                ' "c": 2}\n```\n{"return_value": 3}\n'
            ),
        },
        # external_llm=OpenAI(),
    )

    assert llm_strategy(llm)(add_two_ints)(1, 2) == 3
    assert llm_strategy(llm)(add_two_ints_with_default)(1) == 2
    assert llm_strategy(llm)(add_two_ints_with_default_and_kwarg)(a=1) == 3


def test_llm_strategy_dummy_interface():
    llm = FakeLLM(
        texts={
            (
                'Return 0.\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties": {"self":'
                ' {"$ref": "#/definitions/DummyInterface"}}, "required": ["self"], "definitions": {"DummyInterface":'
                ' {"title": "DummyInterface", "type": "object", "properties": {"a": {"title": "A", "default": -1,'
                ' "type": "integer"}}}}}\n```\n\nThe output should be formatted as a JSON instance that conforms to the'
                ' JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo",'
                ' "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required":'
                ' ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object'
                ' {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"self":'
                ' {"a": -1}}\n```\n{"return_value": 0}\n'
            ),
            (
                'Add two numbers.\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties":'
                ' {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}, "required": ["a",'
                ' "b"]}\n```\n\nThe output should be formatted as a JSON instance that conforms to the JSON schema'
                ' below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a'
                ' list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}}\nthe object'
                ' {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo":'
                ' ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties":'
                ' {"return_value": {"title": "Return Value", "type": "integer"}}, "required":'
                ' ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"a": 1, "b":'
                ' 2}\n```\n{"return_value": 3}\n'
            ),
            (
                'Add two numbers.\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties":'
                ' {"cls": {"title": "Cls", "type": "string"}, "a": {"title": "A", "type": "integer"}, "b": {"title":'
                ' "B", "type": "integer"}}, "required": ["cls", "a", "b"]}\n```\n\nThe output should be formatted as a'
                ' JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties":'
                ' {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type":'
                ' "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance'
                ' of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is'
                ' the output schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type":'
                ' "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"cls": "FakeLLM_DummyInterface", "a": 1, "b": 2}\n```\n{"return_value": 3}\n'
            ),
            (
                'Add two numbers.\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties":'
                ' {"self": {"$ref": "#/definitions/DummyInterface"}, "a": {"title": "A", "type": "integer"}, "b":'
                ' {"title": "B", "type": "integer"}}, "required": ["self", "a", "b"], "definitions": {"DummyInterface":'
                ' {"title": "DummyInterface", "type": "object", "properties": {"a": {"title": "A", "default": -1,'
                ' "type": "integer"}}}}}\n```\n\nThe output should be formatted as a JSON instance that conforms to the'
                ' JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo",'
                ' "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required":'
                ' ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object'
                ' {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output'
                ' schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type": "integer"}},'
                ' "required": ["return_value"]}\n```\nNow output the results for the following inputs:\n```\n{"self":'
                ' {"a": -1}, "a": 1, "b": 2}\n```\n{"return_value": 3}\n'
            ),
        }
        # external_llm=OpenAI(),
    )

    llm_DummyInterface = llm_strategy(llm)(DummyInterface)
    llm_DummyInstance = llm_DummyInterface()
    assert llm_DummyInstance.property_getter == 0
    assert llm_DummyInterface.static_method(1, 2) == 3
    assert llm_DummyInstance.static_method(1, 2) == 3
    assert llm_DummyInstance.bound_method(1, 2) == 3
    assert llm_DummyInterface.class_method(1, 2) == 3
    assert llm_DummyInstance.class_method(1, 2) == 3

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
    llm = FakeLLM(
        texts={
            (
                "Create mock queries that match one of the mock customers better than the others.\n\nWe support"
                ' semantic queries instead of SQL, so we can search for things like\n"the customer that was born in'
                ' 1990".\nThe inputs are formatted as JSON using the following schema:\n```\n{"properties":'
                ' {"customer_database": {"$ref": "#/definitions/CustomerDatabase"}, "num_queries": {"title": "Num'
                ' Queries", "default": 1, "type": "integer"}}, "required": ["customer_database"], "definitions":'
                ' {"Customer": {"title": "Customer", "type": "object", "properties": {"first_name": {"title": "First'
                ' Name", "type": "string"}, "last_name": {"title": "Last Name", "type": "string"}, "birthday":'
                ' {"title": "Birthday", "type": "string"}, "city": {"title": "City", "type": "string"}}, "required":'
                ' ["first_name", "last_name", "birthday", "city"]}, "CustomerDatabase": {"title": "CustomerDatabase",'
                ' "type": "object", "properties": {"customers": {"title": "Customers", "type": "array", "items":'
                ' {"$ref": "#/definitions/Customer"}}}, "required": ["customers"]}}}\n```\n\nThe output should be'
                " formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema"
                ' {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items":'
                ' {"type": "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted'
                ' instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not'
                ' well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"return_value": {"title": "Return'
                ' Value", "type": "array", "items": {"type": "string"}}}, "required": ["return_value"]}\n```\nNow'
                ' output the results for the following inputs:\n```\n{"customer_database": {"customers":'
                ' [{"first_name": "John", "last_name": "Smith", "birthday": "05/12/1991", "city": "New York"},'
                ' {"first_name": "Jane", "last_name": "Doe", "birthday": "03/03/1988", "city": "London"},'
                ' {"first_name": "Carol", "last_name": "Lee", "birthday": "07/01/1995", "city": "Berlin"}]},'
                ' "num_queries": 3}\n```\n\nThe output should be:\n\n{"return_value": ["John Smith", "Jane Doe", "Carol'
                ' Lee"]}'
            ),
            (
                "Find the index of the customer that matches the natural language query best.\n\nWe support semantic"
                ' queries instead of SQL, so we can search for things like\n"the customer that was born in'
                ' 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of the best matching'
                " customer in the database.\nThe inputs are formatted as JSON using the following"
                ' schema:\n```\n{"properties": {"self": {"$ref": "#/definitions/CustomerDatabase"}, "query": {"title":'
                ' "Query", "type": "string"}}, "required": ["self", "query"], "definitions": {"Customer": {"title":'
                ' "Customer", "type": "object", "properties": {"first_name": {"title": "First Name", "type": "string"},'
                ' "last_name": {"title": "Last Name", "type": "string"}, "birthday": {"title": "Birthday", "type":'
                ' "string"}, "city": {"title": "City", "type": "string"}}, "required": ["first_name", "last_name",'
                ' "birthday", "city"]}, "CustomerDatabase": {"title": "CustomerDatabase", "type": "object",'
                ' "properties": {"customers": {"title": "Customers", "type": "array", "items": {"$ref":'
                ' "#/definitions/Customer"}}}, "required": ["customers"]}}}\n```\n\nThe output should be formatted as a'
                ' JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties":'
                ' {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type":'
                ' "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance'
                ' of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is'
                ' the output schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type":'
                ' "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"self": {"customers": [{"first_name": "John", "last_name": "Smith", "birthday":'
                ' "05/12/1991", "city": "New York"}, {"first_name": "Jane", "last_name": "Doe", "birthday":'
                ' "03/03/1988", "city": "London"}, {"first_name": "Carol", "last_name": "Lee", "birthday":'
                ' "07/01/1995", "city": "Berlin"}]}, "query": "Carol Lee"}\n```\n\nThe output should'
                ' be:\n```\n{"return_value": 2}\n```'
            ),
            (
                "Find the index of the customer that matches the natural language query best.\n\nWe support semantic"
                ' queries instead of SQL, so we can search for things like\n"the customer that was born in'
                ' 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of the best matching'
                " customer in the database.\nThe inputs are formatted as JSON using the following"
                ' schema:\n```\n{"properties": {"self": {"$ref": "#/definitions/CustomerDatabase"}, "query": {"title":'
                ' "Query", "type": "string"}}, "required": ["self", "query"], "definitions": {"Customer": {"title":'
                ' "Customer", "type": "object", "properties": {"first_name": {"title": "First Name", "type": "string"},'
                ' "last_name": {"title": "Last Name", "type": "string"}, "birthday": {"title": "Birthday", "type":'
                ' "string"}, "city": {"title": "City", "type": "string"}}, "required": ["first_name", "last_name",'
                ' "birthday", "city"]}, "CustomerDatabase": {"title": "CustomerDatabase", "type": "object",'
                ' "properties": {"customers": {"title": "Customers", "type": "array", "items": {"$ref":'
                ' "#/definitions/Customer"}}}, "required": ["customers"]}}}\n```\n\nThe output should be formatted as a'
                ' JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties":'
                ' {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type":'
                ' "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance'
                ' of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is'
                ' the output schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type":'
                ' "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"self": {"customers": [{"first_name": "John", "last_name": "Smith", "birthday":'
                ' "05/12/1991", "city": "New York"}, {"first_name": "Jane", "last_name": "Doe", "birthday":'
                ' "03/03/1988", "city": "London"}, {"first_name": "Carol", "last_name": "Lee", "birthday":'
                ' "07/01/1995", "city": "Berlin"}]}, "query": "Jane Doe"}\n```\n\nThe output should'
                ' be:\n```\n{"return_value": 1}\n```'
            ),
            (
                "Create mock customers with believable data (our customers are world citizens).\nThe inputs are"
                ' formatted as JSON using the following schema:\n```\n{"properties": {"num_customers": {"title": "Num'
                ' Customers", "default": 1, "type": "integer"}}}\n```\n\nThe output should be formatted as a JSON'
                ' instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties":'
                ' {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type":'
                ' "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance'
                ' of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is'
                ' the output schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type": "array",'
                ' "items": {"$ref": "#/definitions/Customer"}}}, "required": ["return_value"], "definitions":'
                ' {"Customer": {"title": "Customer", "type": "object", "properties": {"first_name": {"title": "First'
                ' Name", "type": "string"}, "last_name": {"title": "Last Name", "type": "string"}, "birthday":'
                ' {"title": "Birthday", "type": "string"}, "city": {"title": "City", "type": "string"}}, "required":'
                ' ["first_name", "last_name", "birthday", "city"]}}}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"num_customers": 3}\n```\n\nOutput:\n{"return_value": [{"first_name": "John",'
                ' "last_name": "Smith", "birthday": "05/12/1991", "city": "New York"}, {"first_name": "Jane",'
                ' "last_name": "Doe", "birthday": "03/03/1988", "city": "London"}, {"first_name": "Carol", "last_name":'
                ' "Lee", "birthday": "07/01/1995", "city": "Berlin"}]}'
            ),
            (
                "Find the index of the customer that matches the natural language query best.\n\nWe support semantic"
                ' queries instead of SQL, so we can search for things like\n"the customer that was born in'
                ' 1990".\n\nArgs:\n    query: Natural language query\n\nReturns:\n    The index of the best matching'
                " customer in the database.\nThe inputs are formatted as JSON using the following"
                ' schema:\n```\n{"properties": {"self": {"$ref": "#/definitions/CustomerDatabase"}, "query": {"title":'
                ' "Query", "type": "string"}}, "required": ["self", "query"], "definitions": {"Customer": {"title":'
                ' "Customer", "type": "object", "properties": {"first_name": {"title": "First Name", "type": "string"},'
                ' "last_name": {"title": "Last Name", "type": "string"}, "birthday": {"title": "Birthday", "type":'
                ' "string"}, "city": {"title": "City", "type": "string"}}, "required": ["first_name", "last_name",'
                ' "birthday", "city"]}, "CustomerDatabase": {"title": "CustomerDatabase", "type": "object",'
                ' "properties": {"customers": {"title": "Customers", "type": "array", "items": {"$ref":'
                ' "#/definitions/Customer"}}}, "required": ["customers"]}}}\n```\n\nThe output should be formatted as a'
                ' JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties":'
                ' {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type":'
                ' "string"}}}, "required": ["foo"]}}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance'
                ' of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is'
                ' the output schema:\n```\n{"properties": {"return_value": {"title": "Return Value", "type":'
                ' "integer"}}, "required": ["return_value"]}\n```\nNow output the results for the following'
                ' inputs:\n```\n{"self": {"customers": [{"first_name": "John", "last_name": "Smith", "birthday":'
                ' "05/12/1991", "city": "New York"}, {"first_name": "Jane", "last_name": "Doe", "birthday":'
                ' "03/03/1988", "city": "London"}, {"first_name": "Carol", "last_name": "Lee", "birthday":'
                ' "07/01/1995", "city": "Berlin"}]}, "query": "John Smith"}\n```\n\nThe output should be:'
                ' \n```\n{"return_value": 0}\n```'
            ),
        }
        # external_llm=OpenAI(),
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
