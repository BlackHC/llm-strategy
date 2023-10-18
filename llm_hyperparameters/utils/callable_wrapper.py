import types
import typing
from dataclasses import dataclass

P = typing.ParamSpec("P")
T = typing.TypeVar("T")


@dataclass
class CallableWrapper:
    """
    A functor that wraps a callable and forwards all attributes to the wrapped callable.
    """

    __wrapped__: object

    def __get__(self, instance: object, owner: type | None = None) -> typing.Callable:
        """Support instance methods."""
        if instance is None:
            return self

        # Bind self to instance as MethodType
        return types.MethodType(self, instance)

    def __getattr__(self, item):
        return getattr(self.__wrapped__, item)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __dir__(self) -> typing.Iterable[str]:
        # merge the attributes of the wrapped object with the attributes of the wrapper
        self_dir = list(super().__dir__())
        wrapped_dir = list(dir(self.__wrapped__))
        return list(set(self_dir + wrapped_dir))
