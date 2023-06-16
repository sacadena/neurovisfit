from collections import OrderedDict

from cookie_test.common.hash import make_hash


def test_make_hash_string():
    result = make_hash("hello world")
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_ordered_dict():
    ordered_dict1 = OrderedDict([("a", 5), ("b", 3), ("c", 4)])
    ordered_dict2 = OrderedDict([("b", 3), ("a", 5), ("c", 4)])
    result1 = make_hash(ordered_dict1)
    result2 = make_hash(ordered_dict2)
    assert result1 != result2


def test_make_hash_nested_dict():
    nested_dict = {"a": {"b": {"c": [1, 2, 3]}}}
    result = make_hash(nested_dict)
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_iterable():
    iterable = [1, 2, 3, ["a", "b", "c"]]
    result = make_hash(iterable)
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_integer():
    result = make_hash(42)
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_float():
    result = make_hash(3.14159)
    assert isinstance(result, str)
    assert len(result) == 32


def test_make_hash_boolean():
    result = make_hash(True)
    assert isinstance(result, str)
    assert len(result) == 32
