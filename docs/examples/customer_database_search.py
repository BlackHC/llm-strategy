"""
A simple CUI application to visualize and query a customer database using the `textual` package.
"""
from dataclasses import dataclass

import langchain
from langchain.cache import SQLiteCache
from langchain.llms import OpenAI
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Footer, Header, Input

from llm_strategy import llm_strategy

langchain.llm_cache = SQLiteCache()


@llm_strategy(OpenAI(max_tokens=256))
@dataclass
class Customer:
    key: str
    first_name: str
    last_name: str
    birthdate: str
    address: str

    @property
    def age(self) -> int:
        """Return the current age of the customer.

        This is a computed property based on `birthdate` and the current year (2022).
        """

        raise NotImplementedError()


@dataclass
class CustomerDatabase:
    customers: list[Customer]

    def find_customer_key(self, query: str) -> list[str]:
        """Find the keys of the customers that match a natural language query best (sorted by closeness to the match).

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".

        Args:
            query: Natural language query

        Returns:
            The index of the best matching customer in the database.
        """
        raise NotImplementedError()

    def load(self):
        """Load the customer database from a file."""
        raise NotImplementedError()

    def store(self):
        """Store the customer database to a file."""
        raise NotImplementedError()


@llm_strategy(OpenAI(max_tokens=1024))
@dataclass
class MockCustomerDatabase(CustomerDatabase):
    def load(self):
        self.customers = self.create_mock_customers(10)

    def store(self):
        pass

    @staticmethod
    def create_mock_customers(num_customers: int = 1) -> list[Customer]:
        """
        Create mock customers with believable data (our customers are world citizens).
        """
        raise NotImplementedError()


class CustomerDatabaseApp(App):
    """A simple textual application to visualize and query a customer database.

    We show all the customers in a table and allow the user to query the database using natural language
    in a search box at the bottom of the screen.
    """

    PRIORITY_BINDINGS = False
    BINDINGS = [("q", "quit", "Quit the application"), ("s", "screenshot", "Take a screenshot")]

    database: CustomerDatabase = MockCustomerDatabase([])

    data_table = DataTable(id="customer_table")
    search_box = Input(id="search_box", placeholder="Search for a customer (use any kind of query")
    footer_bar = Horizontal(search_box)

    def on_mount(self) -> None:
        self.database.load()

        self.data_table.add_columns("First Name", "Last Name", "Birthdate", "Address", "Age")
        self.search("")

    def compose(self) -> ComposeResult:
        self.footer_bar.styles.dock = "bottom"
        self.footer_bar.styles.width = "100%"
        self.footer_bar.styles.height = 4

        self.data_table.styles.height = "auto"
        self.data_table.styles.width = "100%"
        self.screen.styles.height = "100%"
        self.search_box.styles.width = "100%"

        yield Header()
        yield self.footer_bar
        yield Footer()

        yield self.data_table

    def search(self, query: str):
        """Search the customer database using a natural language query."""
        self.data_table.clear()
        if not query:
            for customer in self.database.customers:
                self.data_table.add_row(
                    # customer.key,
                    customer.first_name,
                    customer.last_name,
                    customer.birthdate,
                    customer.address,
                    str(customer.age),
                )
        else:
            keys = self.database.find_customer_key(query)
            for key in keys:
                customers_for_key = [customer for customer in self.database.customers if customer.key == key]
                assert len(customers_for_key) == 1
                customer = customers_for_key[0]
                self.data_table.add_row(
                    # customer.key,
                    customer.first_name,
                    customer.last_name,
                    customer.birthdate,
                    customer.address,
                    str(customer.age),
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is self.exit_button:
            self.exit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is self.search_box:
            self.search(event.value)


if __name__ == "__main__":
    app = CustomerDatabaseApp()
    app.run()
