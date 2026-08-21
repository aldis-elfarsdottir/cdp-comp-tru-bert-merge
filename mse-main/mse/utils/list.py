from typing import Any


def to_batch(data: str | list[str]) -> list[str]:
    # Add a dimension to the given string field
    return [data] if isinstance(data, str) else data


def flatten(nested: list[Any]) -> list[Any]:

    def _flatten(nested: Any):
        for el in nested:
            if isinstance(el, list | tuple):
                yield from _flatten(el)
            else:
                yield el

    return list(_flatten(nested))


def unflatten(flattened: list[Any], structure: list[Any]) -> list[Any]:
    iterator = iter(flattened)

    def _unflatten(structure):
        if isinstance(structure, list | tuple):
            return [_unflatten(s) for s in structure]
        else:
            return next(iterator)
    
    return _unflatten(structure)