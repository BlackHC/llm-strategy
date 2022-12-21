from dataclasses import dataclass

from llm_strategy.llm_implement import llm_implement
from llm_strategy.testing.fake_llm import FakeLLM


def test_llm_implement_add_two_ints():
    def add_two_ints(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def add_two_ints_with_default(a: int, b: int = 1) -> int:
        """Add two integers with a default value."""
        return a + b

    def add_two_ints_with_default_and_kwarg(*, a: int, c: int = 2) -> int:
        """Add two integers with a default value."""
        return a + c

    llm = FakeLLM(
        texts={
            (
                "Execute the following function that is described via a doc string:\n\nAdd two integers.\n\n#"
                " Task\n\nExecute the function with the inputs that follow in the next section and finally return the"
                " output using the output type\nas YAML document in an # Output section. (If the value is a literal,"
                " then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in"
                " Python.)\n\n# Input Types\n\na: int\nb: int\n\n\n# Inputs\n\na: 1\nb: 2\n\n\n# Output"
                " Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult: 3"
            ),
            (
                "Execute the following function that is described via a doc string:\n\nAdd two integers with a default"
                " value.\n\n# Task\n\nExecute the function with the inputs that follow in the next section and finally"
                " return the output using the output type\nas YAML document in an # Output section. (If the value is a"
                " literal, then just write the value. We parse the text in the\n# Output section using `yaml.safe_load`"
                " in Python.)\n\n# Input Types\n\na: int\nc: int\n\n\n# Inputs\n\na: 1\nc: 2\n\n\n# Output"
                " Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\ndef add_two_integers(a, c=0):\n   "
                ' """Add two integers with a default value."""\n    return a + c \n\n\n# Output\n\n---\nresult: 3'
            ),
            (
                "Execute the following function that is described via a doc string:\n\nAdd two integers with a default"
                " value.\n\n# Task\n\nExecute the function with the inputs that follow in the next section and finally"
                " return the output using the output type\nas YAML document in an # Output section. (If the value is a"
                " literal, then just write the value. We parse the text in the\n# Output section using `yaml.safe_load`"
                " in Python.)\n\n# Input Types\n\na: int\nb: int\n\n\n# Inputs\n\na: 1\nb: 1\n\n\n# Output"
                " Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult: 2"
            ),
        },
        # external_llm=OpenAI(),
    )

    assert llm_implement(add_two_ints, llm)(1, 2) == 3
    assert llm_implement(add_two_ints_with_default, llm)(1) == 2
    assert llm_implement(add_two_ints_with_default_and_kwarg, llm)(a=1) == 3


@dataclass
class Foo:
    a: int
    b: int

    @property
    def c(self: "Foo") -> int:
        """Add the two integer fields."""
        return self.a + self.b


def test_llm_implement_property():
    llm = FakeLLM(
        texts={
            "Execute the following function that is described via a doc string:\n\nAdd the two integer fields.\n\n# "
            "Task\n\nExecute the function with the inputs that follow in the next section and finally return the "
            "output using the output type\nas YAML document in an # Output section. (If the value is a literal, "
            "then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in "
            "Python.)\n\n# Dataclasses Schema\n\ntypes:\n  Foo:\n    a:\n      type: int\n    b:\n      type: "
            "int\n\n\n# Input Types\n\nself: Foo\n\n\n# Inputs\n\nself:\n  a: 1\n  b: 2\n\n\n# Output "
            "Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult: 3 "
        },
        # external_llm=OpenAI()
    )

    assert llm_implement(Foo.c, llm)(Foo(1, 2)) == 3
