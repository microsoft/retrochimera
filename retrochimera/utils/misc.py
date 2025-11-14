import pydoc
import re
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_type_hints

import h5py
import numpy as np
import scipy
from rdkit import RDLogger


def save_into_h5py(group: h5py.Group, key: str, data: Any) -> None:
    """Save a given object into a `h5py.Group`.

    This function only works for the processed data sample dataclasses and their subcomponents.
    """
    cls = type(data)
    cls_name = f"{cls.__module__}.{cls.__name__}"

    if is_dataclass(data):
        data_dict = {field.name: getattr(data, field.name) for field in fields(data)}
    elif isinstance(data, scipy.sparse.dok_matrix):
        assert data.dtype == np.int32

        data_dict = {
            "shape": np.asarray(data.shape),
            "keys": np.asarray(list(data.keys())),
            "values": np.asarray(list(data.values())),
        }
    else:
        dataset = group.create_dataset(key, data=data)
        dataset.attrs["type"] = cls_name
        return

    subgroup = group.create_group(key)
    subgroup.attrs["type"] = cls_name

    for subkey, value in data_dict.items():
        if value is not None:
            save_into_h5py(subgroup, key=subkey, data=value)


def load_from_h5py(entry: Union[h5py.Dataset, h5py.Group]) -> Any:
    """Parse a given class from a `h5py` entry.

    This function only works for the processed data sample dataclasses and their subcomponents.
    """
    cls_name = entry.attrs["type"]
    cls = pydoc.locate(cls_name)

    if is_dataclass(cls):
        assert isinstance(entry, h5py.Group)

        type_hints = get_type_hints(cls)
        return cls(
            **{
                key: (load_from_h5py(entry[key]) if key in entry else None)
                for key in type_hints.keys()
            }
        )  # type: ignore
    elif cls is scipy.sparse.dok_matrix:
        assert isinstance(entry, h5py.Group)

        matrix = scipy.sparse.dok_matrix(tuple(entry["shape"]), dtype="i")
        for (key_row, key_col), value in zip(entry["keys"], entry["values"]):
            matrix[key_row, key_col] = value

        return matrix
    elif cls in (int, bool):
        return np.asarray(entry).item()
    elif cls is list:
        return list(entry)
    else:
        assert isinstance(entry, h5py.Dataset)
        return np.asarray(entry)


def lookup_by_name(module, name):
    return module.__dict__[name]


def silence_rdkit_warnings() -> None:
    RDLogger.DisableLog("rdApp.*")


def convert_camel_to_snake(name: str) -> str:
    """Convert a CamelCase name to snake_case.

    Consecutive capital letters (e.g. "GNN") are treated as a single word.

    >>> convert_camel_to_snake('TemplateLocalization')
    'template_localization'

    >>> convert_camel_to_snake('TemplateClassificationGNN')
    'template_classification_gnn'

    >>> convert_camel_to_snake('MEGAN')
    'megan'
    """
    return re.sub("(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()
