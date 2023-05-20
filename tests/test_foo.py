from cookie_test.foo import delayed_hello
from cookie_test.foo import foo


def test_foo() -> None:
    assert foo() == "foo"


def test_delayed_hello() -> None:
    assert delayed_hello() == "Hello world"
