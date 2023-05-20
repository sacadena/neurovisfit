from time import sleep


def foo() -> str:
    """Summary line.

    Extended description of function.

    Args:
        foo (str): Description of arg1

    Returns:
        str: Description of return value
    """

    return "foo"


def delayed_hello() -> str:
    """Delayed hello"""
    sleep(10)
    return "Hello world"


if __name__ == "__main__":  # pragma: no cover
    pass
