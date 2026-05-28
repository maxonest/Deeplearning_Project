"""Thread-safe multi-turn conversation memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Literal
from uuid import uuid4


Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ConversationMemory:
    """Keep recent turns and a compact summary of older turns."""

    def __init__(self, max_recent_turns: int = 6, max_context_chars: int = 6000) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_context_chars = max_context_chars
        self._sessions: dict[str, list[Message]] = {}
        self._summaries: dict[str, str] = {}
        self._lock = RLock()

    def create_session(self) -> str:
        with self._lock:
            session_id = str(uuid4())
            self._sessions[session_id] = []
            self._summaries[session_id] = ""
            return session_id

    def has_session(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def add_message(self, session_id: str, role: Role, content: str) -> None:
        content = content.strip()
        if not content:
            return
        with self._lock:
            self._sessions.setdefault(session_id, [])
            self._summaries.setdefault(session_id, "")
            self._sessions[session_id].append(Message(role=role, content=content))
            self._compact_old_messages(session_id)

    def get_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_recent_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            messages = self._sessions.get(session_id, [])
            return list(messages[-self.max_recent_turns * 2 :])

    def build_context(self, session_id: str) -> str:
        with self._lock:
            summary = self._summaries.get(session_id, "")
            recent = list(self._sessions.get(session_id, []))

        parts: list[str] = []
        if summary:
            parts.append(f"历史摘要:\n{summary}")
        if recent:
            turns = "\n".join(f"{message.role}: {message.content}" for message in recent)
            parts.append(f"最近对话:\n{turns}")
        return "\n\n".join(parts)[-self.max_context_chars :]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = []
            self._summaries[session_id] = ""

    def _compact_old_messages(self, session_id: str) -> None:
        messages = self._sessions.get(session_id, [])
        max_messages = self.max_recent_turns * 2
        if len(messages) <= max_messages:
            return

        old_messages = messages[:-max_messages]
        old_text = "\n".join(f"{message.role}: {message.content}" for message in old_messages)
        previous = self._summaries.get(session_id, "")
        self._summaries[session_id] = f"{previous}\n{old_text}".strip()[-self.max_context_chars :]
        self._sessions[session_id] = messages[-max_messages:]
