from cookie_test.foo import foo


def test_foo() -> None:
    assert foo() == "foo"
