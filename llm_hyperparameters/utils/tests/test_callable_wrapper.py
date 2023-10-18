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

from dataclasses import dataclass

from llm_hyperparameters.utils.callable_wrapper import CallableWrapper


def test_wrapper_functor():
    @dataclass
    class MyFunctor(CallableWrapper):
        __wrapped__: object

        def __call__(self, *args, **kwargs):
            return self.__wrapped__(*args, **kwargs)

    def my_functor(x: int) -> str:
        """
        my_functor(x: int) -> str
        """
        return str(x)

    my_functor = MyFunctor(my_functor)
    assert my_functor(3) == "3"

    # test that we can access the wrapped function's attributes
    @dataclass
    class InnerFunctor:
        a: int
        b: str

        def __call__(self, x: int) -> str:
            return str(x + self.a) + self.b

    my_functor = MyFunctor(InnerFunctor(3, "foo"))

    assert my_functor(3) == "6foo"
    assert my_functor.a == 3
    assert my_functor.b == "foo"

    # assert that the dir of the wrapper contains all attributes of the wrapped object (superset)
    assert set(dir(my_functor)) >= set(dir(my_functor.__wrapped__))
