from backend.memory import ConversationMemory


def test_memory_keeps_recent_turns_and_compacts_old_messages():
    memory = ConversationMemory(max_recent_turns=1, max_context_chars=1000)
    session_id = memory.create_session()

    memory.add_message(session_id, "user", "问题1")
    memory.add_message(session_id, "assistant", "答案1")
    memory.add_message(session_id, "user", "问题2")
    memory.add_message(session_id, "assistant", "答案2")

    messages = memory.get_messages(session_id)
    context = memory.build_context(session_id)

    assert [message.content for message in messages] == ["问题2", "答案2"]
    assert "历史摘要" in context
    assert "问题1" in context
    assert "答案2" in context


def test_memory_clear_resets_session():
    memory = ConversationMemory()
    session_id = memory.create_session()
    memory.add_message(session_id, "user", "hello")

    memory.clear(session_id)

    assert memory.get_messages(session_id) == []
