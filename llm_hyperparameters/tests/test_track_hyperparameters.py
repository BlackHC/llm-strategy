from llm_hyperparameters.track_hyperparameters import (
    Hyperparameters,
    track_hyperparameters,
)


def test_all():
    @track_hyperparameters
    def f(*, hparam_a: int = 1):
        return hparam_a

    @track_hyperparameters
    def g(*, hparam_b: int = 2):
        return hparam_b

    with Hyperparameters() as hparams:
        assert f() == 1
        assert g() == 2

    hparams[f].a = 3

    with hparams:
        assert f() == 3
        assert g() == 2


def test_no_scope():
    @track_hyperparameters
    def f(hparam_a: int = 1):
        return hparam_a

    assert f() == 1


def test_manual():
    @track_hyperparameters
    def f(hparam_a: int = 1):
        return hparam_a

    with Hyperparameters() as hparams:
        assert f() == 1

    hparams[f].a = 2

    with hparams:
        assert f() == 2

    @track_hyperparameters
    def g(hparam_a: str = "Hello", hparam_b: str = "Hello"):
        return hparam_a + hparam_b

    with Hyperparameters() as hparams:
        assert g() == "HelloHello"
        assert g(hparam_b="World") == "HelloWorld"

    hparams[g].b = "World"

    with hparams:
        assert g() == "HelloWorld"


def test_nested():
    @track_hyperparameters
    def f(hparam_a: int = 1, hparam_b: int = 2):
        return hparam_a + hparam_b

    assert f() == 3

    @track_hyperparameters
    def g(hparam_c: int = 3):
        return hparam_c + f()

    with Hyperparameters() as hparams:
        assert g() == 6

    hparams[g].c = 4

    with hparams:
        assert g() == 7

    hparams[f].a = 5

    with hparams:
        assert g() == 11
