from cookie_test.foo2 import add_numbers


def test_foo2() -> None:
    assert add_numbers(3, 6) == 9
