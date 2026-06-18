from start_windows import describe_return_code, render_knowledge_base_progress


def test_describe_return_code_translates_windows_stack_overflow():
    assert describe_return_code(3221225725) == (
        "3221225725 (0xC00000FD: Windows stack overflow)"
    )


def test_describe_return_code_keeps_unknown_code():
    assert describe_return_code(1) == "1"


def test_render_knowledge_base_progress(capsys):
    render_knowledge_base_progress(5, 10)

    output = capsys.readouterr().out
    assert "50.00%" in output
    assert "5/10" in output
