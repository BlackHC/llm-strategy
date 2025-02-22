# %%
import inspect
import types
import typing

from pydantic import BaseModel
from pydantic._internal._generics import (
    get_model_typevars_map,
    get_origin,
    iter_contained_typevars,
    replace_types,
)


class PydanticGenericTypeMap(dict[typing.TypeVar, typing.Type]):
    @staticmethod
    def get_generic_type_map(generic_type, base_generic_type=None):
        """Build a generic type map for a generic type.

        It maps the generic type variables to the actual types.
        """
        if base_generic_type is None:
            base_generic_type = PydanticGenericTypeMap.get_base_generic_type(generic_type)

        base_classes = inspect.getmro(generic_type)
        # we have to iterate through the base classes
        generic_parameter_type_map = {
            generic_type: generic_type for generic_type in iter_contained_typevars(generic_type)
        }
        for base_class in base_classes:
            if not issubclass(base_class, base_generic_type) or base_class is base_generic_type:
                break
            assignment = get_model_typevars_map(base_class)
            generic_parameter_type_map = {
                old_generic_type: generic_parameter_type_map.get(new_generic_type, new_generic_type)
                for old_generic_type, new_generic_type in assignment.items()
            }

        return PydanticGenericTypeMap(generic_parameter_type_map)

    @staticmethod
    def resolve_generic_types(model: type[BaseModel], instance: BaseModel) -> "PydanticGenericTypeMap":
        """
        Resolves the generic types of a given model instance and returns a generic type map.

        Args:
            model (type[BaseModel]): The model type.
            instance (BaseModel): The instance of the model.

        Returns:
            GenericTypeMap: The generic type map.
        """
        generic_type_map = PydanticGenericTypeMap()

        for field_name, attr_value in list(instance):
            if field_name not in model.model_fields:
                continue

            annotation = model.model_fields[field_name].annotation

            # if the annotation is an Annotated type, get the type annotation
            if typing.get_origin(annotation) is typing.Annotated:
                annotation = typing.get_args(annotation)[0]

            # if the annotation is a type var, resolve it into the generic type map
            if isinstance(annotation, typing.TypeVar):
                PydanticGenericTypeMap.add_resolved_type(generic_type_map, annotation, type(attr_value))
            # if the annotation is a generic type alias ignore
            elif isinstance(annotation, types.GenericAlias):
                # The generic type alias is not supported yet
                # The problem is that GenericAlias types are elided: e.g. type(list[str](["hello"])) -> list and not list[str].
                # But either way, we would need to resolve the types based on the actual elements and their mros.
                continue
            # if the annotation is a type, check if it is a Pydantic type
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                generic_definition_type_map = PydanticGenericTypeMap.get_generic_type_map(annotation)

                argument_type = type(attr_value)
                generic_instance_type_map = PydanticGenericTypeMap.get_generic_type_map(argument_type)

                assert list(generic_definition_type_map.keys()) == list(generic_instance_type_map.keys())

                parameters = set(iter_contained_typevars(annotation))
                # update the generic type map
                for generic_parameter, generic_parameter_target in generic_definition_type_map.items():
                    if generic_parameter_target not in parameters:
                        continue
                    resolved_type = generic_instance_type_map[generic_parameter]
                    generic_type_map.add_resolved_type(generic_parameter_target, resolved_type)
            else:
                # Let Pydantic handle the rest
                continue

        return generic_type_map

    def resolve_type(self, source_type: type | typing.TypeVar) -> type:
        """
        Resolve a type using the generic type map.

        Supports Pydantic.BaseModel and typing.Generic.
        """
        if source_type in self:
            assert isinstance(source_type, typing.TypeVar)
            source_type = self[source_type]

        # we let Pydantic handle the rest
        source_type = replace_types(source_type, self)

        return source_type

    def add_resolved_type(self, source_type, resolved_type):
        """
        Add a resolved type to the generic type map.
        """
        if source_type in self:
            # TODO: support finding the common base class?
            if (previous_resolution := self[source_type]) is not resolved_type:
                raise ValueError(
                    f"Cannot resolve generic type {source_type}, conflicting "
                    f"resolution: {previous_resolution} and {resolved_type}."
                )
        else:
            self[source_type] = resolved_type

    @staticmethod
    def get_base_generic_type(field_type: type) -> type:
        """Determine the base generic type of a generic type. E.g. List[str] -> List.

        Args:
            field_type (type): The generic type.

        Raises:
            ValueError: If the base generic type cannot be found.

        Returns:
            type: The base generic type.
        """
        origin = get_origin(field_type)
        if origin is not None:
            return origin
        else:
            # If origin is None, field_type might be a non-parameterized generic type or a concrete type
            return field_type


# %%
