from cookie_test.foo import foo
from cookie_test.foo import hello


def test_foo() -> None:
    assert foo() == "foo"


def test_hello() -> None:
    assert hello() == "Hello world"
