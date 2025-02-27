import llm_hyperparameters.track_hyperparameters
from llm_hyperparameters.track_hyperparameters import (
    HyperparameterScope,
    track_hyperparameters,
)


def test_all():
    @track_hyperparameters
    def f(*, hparam_a: int = 1):
        return hparam_a

    @track_hyperparameters
    def g(*, hparam_b: int = 2):
        return hparam_b

    with HyperparameterScope() as hparams:
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

    with HyperparameterScope() as hparams:
        assert f() == 1

    hparams[f].a = 2

    with hparams:
        assert f() == 2

    @track_hyperparameters
    def g(hparam_a: str = "Hello", hparam_b: str = "Hello"):
        return hparam_a + hparam_b

    with HyperparameterScope() as hparams:
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

    with HyperparameterScope() as hparams:
        assert g() == 6

    hparams[g].c = 4

    with hparams:
        assert g() == 7

    hparams[f].a = 5

    with hparams:
        assert g() == 11


def test_serialization():
    @track_hyperparameters
    def f(hparam_a: int = 1, hparam_b: int = 2):
        return hparam_a + hparam_b

    @track_hyperparameters
    def g(hparam_c: int = 3):
        return hparam_c + f()

    with HyperparameterScope() as hparams:
        assert g() == 6

    hparams[g].c = 4
    hparams[f].a = 5

    specific_hparams = llm_hyperparameters.track_hyperparameters.Hyperparameters(hparams)
    serialized = specific_hparams.model_dump_json()
    new_specific_hparams = specific_hparams.model_validate_json(serialized)

    # Verify the deserialized hyperparameters work the same
    with HyperparameterScope(new_specific_hparams):
        assert g() == 11
