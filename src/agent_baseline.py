from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-thread memory only. New thread_id => facts are gone."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        del user_id  # Baseline intentionally ignores long-term user identity.
        if self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return 0 if session is None else session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return 0 if session is None else session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def memory_file_size(self, user_id: str) -> int:
        del user_id
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        answer = self._offline_response(session, message)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)
        return {"answer": answer, "thread_id": thread_id}

    def _offline_response(self, session: SessionState, message: str) -> str:
        prior_user = [
            item["content"] for item in session.messages[:-1] if item["role"] == "user"
        ]
        if not prior_user:
            return (
                "Mình chưa có bộ nhớ dài hạn giữa các phiên. "
                "Trong thread mới này mình chưa nhớ tên, nghề nghiệp hay sở thích của bạn."
            )
        blob = "\n".join(prior_user)
        return (
            "Trong cùng thread này mình còn nhớ vài ý gần đây: "
            + blob[-400:]
        )

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += estimate_tokens(
            "\n".join(item["content"] for item in session.messages)
        )
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = _content_from_langchain(result)
        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)
        return {"answer": answer, "thread_id": thread_id}

    def _maybe_build_langchain_agent(self):
        """Optional live path: create_agent + InMemorySaver for the chosen provider."""

        provider = self.config.model.provider
        if provider not in {"ollama", "custom"} and not self.config.model.api_key:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver

            model = build_chat_model(self.config.model)
            return create_agent(
                model,
                tools=[],
                checkpointer=InMemorySaver(),
                system_prompt=(
                    "Bạn là trợ lý chỉ nhớ ngữ cảnh trong cùng một thread. "
                    "Không bịa thông tin dài hạn từ các phiên khác."
                ),
            )
        except Exception:
            return None


def _content_from_langchain(result: Any) -> str:
    if isinstance(result, dict) and "messages" in result:
        last = result["messages"][-1]
        content = getattr(last, "content", last)
        return str(content)
    return str(result)
