import hashlib
from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any


def make_hash(obj: Any) -> str:
    """
    Given a Python object, returns a 32 character hash string to uniquely identify
    the content of the object. The object can be arbitrary nested (i.e. dictionary
    of dictionary of list etc), and hashing is applied recursively to uniquely
    identify the content.

    For dictionaries (at any level), the key order is ignored when hashing
    so that {"a":5, "b": 3, "c": 4} and {"b": 3, "a": 5, "c": 4} will both
    give rise to the same hash. Exception to this rule is when an OrderedDict
    is passed, in which case difference in key order is respected. To keep
    compatible with previous versions of Python and the assumed general
    intentions, key order will be ignored even in Python 3.7+ where the
    default dictionary is officially an ordered dictionary.

    Args:
        obj - A (potentially nested) Python object

    Returns:
        A 32 character long hash string to uniquely identify the object.
    """
    hashed = hashlib.md5()

    if isinstance(obj, str):
        hashed.update(obj.encode())
    elif isinstance(obj, OrderedDict):
        for k, v in obj.items():
            hashed.update(str(k).encode())
            hashed.update(make_hash(v).encode())
    elif isinstance(obj, Mapping):
        for k in sorted(obj, key=str):
            hashed.update(str(k).encode())
            hashed.update(make_hash(obj[k]).encode())
    elif isinstance(obj, Iterable):
        for v in obj:
            hashed.update(make_hash(v).encode())
    else:
        hashed.update(str(obj).encode())

    return hashed.hexdigest()
