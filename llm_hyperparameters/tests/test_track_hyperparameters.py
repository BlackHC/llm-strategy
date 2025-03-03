from pydantic import Field

from llm_hyperparameters.track_hyperparameters import (
    Hyperparameters,
    track_hyperparameters,
)


def test_all():
    @track_hyperparameters
    def f(*, _a: int = 1):
        return _a

    @track_hyperparameters
    def g(*, _b: int = 2):
        return _b

    with Hyperparameters() as hparams:
        assert f() == 1
        assert g() == 2

    hparams[f].a = 3

    with hparams:
        assert f() == 3
        assert g() == 2


def test_field():
    @track_hyperparameters
    def f(*, _a: int = Field(default=1, description="A test field")):  # noqa: B008
        return _a

    assert f() == 1

    with Hyperparameters() as hparams:
        assert f() == 1

    hparams[f].a = 2

    with hparams:
        assert f() == 2


def test_no_scope():
    @track_hyperparameters
    def f(_a: int = 1):
        return _a

    assert f() == 1


def test_manual():
    @track_hyperparameters
    def f(_a: int = 1):
        return _a

    with Hyperparameters() as hparams:
        assert f() == 1

    hparams[f].a = 2

    with hparams:
        assert f() == 2

    @track_hyperparameters
    def g(_a: str = "Hello", _b: str = "Hello"):
        return _a + _b

    with Hyperparameters() as hparams:
        assert g() == "HelloHello"
        assert g(_b="World") == "HelloWorld"

    hparams[g].b = "World"

    with hparams:
        assert g() == "HelloWorld"


def test_nested():
    @track_hyperparameters
    def f(_a: int = 1, _b: int = 2):
        return _a + _b

    assert f() == 3

    @track_hyperparameters
    def g(_c: int = 3):
        return _c + f()

    with Hyperparameters() as hparams:
        assert g() == 6

    hparams[g].c = 4

    with hparams:
        assert g() == 7

    hparams[f].a = 5

    with hparams:
        assert g() == 11


def test_serialization():
    @track_hyperparameters
    def f(_a: int = 1, _b: int = 2):
        return _a + _b

    @track_hyperparameters
    def g(_c: int = 3):
        return _c + f()

    with Hyperparameters() as hparams:
        assert g() == 6

    hparams[g].c = 4
    hparams[f].a = 5

    serialized = hparams.model_dump_json()
    new_hparams = hparams.model_validate_json(serialized)

    # Verify the deserialized hyperparameters work the same
    with Hyperparameters(new_hparams):
        assert g() == 11


def test_type_compatibility():
    @track_hyperparameters
    def f(_a: int = 1):
        return _a

    @track_hyperparameters
    def g():
        return 0

    hps = Hyperparameters()
    hps[f] = f.config_model_type()
    hps[g] = g.config_model_type()

    Hyperparameters(hps)
