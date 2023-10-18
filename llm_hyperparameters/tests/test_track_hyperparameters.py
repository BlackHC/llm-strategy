import pytest

from llm_hyperparameters.track_hyperparameters import (
    Hyperparameter,
    hyperparameters_scope,
    track_hyperparameters,
)


def test_all():
    @track_hyperparameters
    def f():
        return Hyperparameter() @ 1

    @track_hyperparameters
    def g():
        return Hyperparameter() @ 2

    with hyperparameters_scope() as scope:
        assert f() == 1
        assert g() == 2

    scope.hyperparameters[f].hparam0 = 3

    with scope():
        assert f() == 3
        assert g() == 2


def test_no_scope():
    @track_hyperparameters
    def f():
        return Hyperparameter() @ 1

    with pytest.warns(UserWarning):
        assert f() == 1


def test_no_name():
    @track_hyperparameters
    def f():
        return Hyperparameter() @ 1

    with hyperparameters_scope() as scope:
        assert f() == 1

    scope.hyperparameters[f].hparam0 = 2

    with scope():
        assert f() == 2

    @track_hyperparameters
    def g():
        return Hyperparameter() @ "Hello" + Hyperparameter() @ "Hello"

    with hyperparameters_scope() as scope:
        assert g() == "HelloHello"

    scope.hyperparameters[g].hparam1 = "World"

    with scope():
        assert g() == "HelloWorld"


def test_with_name():
    @track_hyperparameters
    def f():
        return Hyperparameter("hello") @ 1

    with hyperparameters_scope() as scope:
        assert f() == 1

    scope.hyperparameters[f].hello = 2

    with scope():
        assert f() == 2

    @track_hyperparameters
    def g():
        return Hyperparameter("hello") @ "Hello" + Hyperparameter("hello") @ "Hello"

    with hyperparameters_scope() as scope:
        assert g() == "HelloHello"

    scope.hyperparameters[g].hello = "World"

    with scope():
        assert g() == "WorldWorld"


def test_nested():
    @track_hyperparameters
    def f():
        return Hyperparameter("hello") @ 1 + Hyperparameter("world") @ 2

    assert f() == 3

    @track_hyperparameters
    def g():
        return Hyperparameter("hello") @ 3 + f()

    with hyperparameters_scope() as scope:
        assert g() == 6

    scope.hyperparameters[g].hello = 4

    with scope():
        assert g() == 7

    scope.hyperparameters[f].hello = 5

    with scope():
        assert g() == 11
