"""Multi-turn conversation memory for local QA."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationMemory:
    """Keep recent turns and a lightweight summary placeholder."""

    def __init__(self, max_recent_turns: int = 6, max_context_chars: int = 6000) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_context_chars = max_context_chars
        self.sessions: dict[str, list[Message]] = {}
        self.summaries: dict[str, str] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = []
        self.summaries[session_id] = ""
        return session_id

    def add_message(self, session_id: str, role: Role, content: str) -> None:
        self.sessions.setdefault(session_id, [])
        self.sessions[session_id].append(Message(role=role, content=content.strip()))
        self._maybe_summarize(session_id)

    def get_recent_messages(self, session_id: str) -> list[Message]:
        messages = self.sessions.get(session_id, [])
        return messages[-self.max_recent_turns * 2 :]

    def build_context(self, session_id: str) -> str:
        summary = self.summaries.get(session_id, "")
        recent = self.get_recent_messages(session_id)
        parts: list[str] = []
        if summary:
            parts.append(f"历史摘要:\n{summary}")
        if recent:
            turns = "\n".join(f"{msg.role}: {msg.content}" for msg in recent)
            parts.append(f"最近对话:\n{turns}")
        return "\n\n".join(parts)[-self.max_context_chars :]

    def clear(self, session_id: str) -> None:
        self.sessions[session_id] = []
        self.summaries[session_id] = ""

    def _maybe_summarize(self, session_id: str) -> None:
        """Summarize old turns.

        In production, replace this heuristic with a small local model or an
        LLM summarization call. Old turns can also be embedded and stored for
        vector recall.
        """

        messages = self.sessions.get(session_id, [])
        max_messages = self.max_recent_turns * 2
        if len(messages) <= max_messages:
            return

        old_messages = messages[:-max_messages]
        old_text = "\n".join(f"{msg.role}: {msg.content}" for msg in old_messages)
        previous_summary = self.summaries.get(session_id, "")
        combined = f"{previous_summary}\n{old_text}".strip()
        self.summaries[session_id] = combined[-self.max_context_chars :]
        self.sessions[session_id] = messages[-max_messages:]
