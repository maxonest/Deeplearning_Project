from start_windows import describe_return_code


def test_describe_return_code_translates_windows_stack_overflow():
    assert describe_return_code(3221225725) == (
        "3221225725 (0xC00000FD: Windows stack overflow)"
    )


def test_describe_return_code_keeps_unknown_code():
    assert describe_return_code(1) == "1"
