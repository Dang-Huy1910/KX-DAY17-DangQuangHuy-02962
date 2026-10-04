from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: short-term thread memory + persistent User.md + compact memory."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return int(self.thread_tokens.get(thread_id, 0))

    def prompt_token_usage(self, thread_id: str) -> int:
        return int(self.thread_prompt_tokens.get(thread_id, 0))

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _persist_facts(self, user_id: str, message: str) -> dict[str, str]:
        updates = extract_profile_updates(
            message,
            min_confidence=self.config.profile_confidence_threshold,
        )
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        return updates

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_facts(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(
            answer
        )
        return {
            "answer": answer,
            "thread_id": thread_id,
            "user_id": user_id,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        recent = "\n".join(
            str(item.get("content", "")) for item in (ctx.get("messages") or [])
        )
        return estimate_tokens(profile + "\n" + str(ctx.get("summary") or "") + "\n" + recent)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.ranked_facts(
            user_id,
            half_life_days=self.config.memory_decay_half_life_days,
        )
        ctx = self.compact_memory.context(thread_id)
        looking_up = _looks_like_question(message)

        if looking_up and not facts:
            return "Mình chưa lưu được hồ sơ bền vững cho bạn."

        lines: list[str] = []
        if facts.get("name"):
            lines.append(f"Tên bạn là {facts['name']}.")
        if facts.get("location"):
            lines.append(f"Nơi ở hiện tại là {facts['location']}.")
        if facts.get("profession"):
            lines.append(f"Nghề nghiệp hiện tại là {facts['profession']}.")
        if facts.get("style"):
            lines.append(f"Style trả lời mình thích: {facts['style']}.")
        if facts.get("favorite_drink"):
            lines.append(f"Đồ uống yêu thích là {facts['favorite_drink']}.")
        if facts.get("favorite_food"):
            lines.append(f"Món ăn yêu thích là {facts['favorite_food']}.")
        if facts.get("pet"):
            lines.append(f"Bạn nuôi {facts['pet']}.")
        if facts.get("interests"):
            lines.append(f"Mối quan tâm chính: {facts['interests']}.")

        if looking_up:
            if not lines:
                return "Mình chưa nhớ đủ fact ổn định để trả lời câu hỏi này."
            style = facts.get("style", "")
            if "3 bullet" in style:
                return "\n".join(f"- {line}" for line in lines[:6])
            return " ".join(lines)

        ack = "Mình đã lưu các fact ổn định vào User.md."
        if lines:
            ack += " " + " ".join(lines[:4])
        summary = str(ctx.get("summary") or "")
        if summary:
            ack += " Compact memory đang giữ tóm tắt hội thoại dài."
        return ack

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_facts(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        prompt = (
            "Hồ sơ bền vững (User.md):\n"
            f"{profile}\n\n"
            f"Tóm tắt cũ: {ctx.get('summary') or '(trống)'}\n\n"
            f"Người dùng: {message}"
        )
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = _content_from_langchain(result)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + estimate_tokens(
            answer
        )
        return {
            "answer": answer,
            "thread_id": thread_id,
            "user_id": user_id,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _maybe_build_langchain_agent(self):
        """Optional live path: provider model + InMemorySaver + profile-aware prompt."""

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
                    "Bạn là trợ lý có bộ nhớ dài hạn. Ưu tiên fact mới nhất trong User.md, "
                    "bỏ nhiễu (câu đùa, nơi họp tạm), trả lời ngắn gọn."
                ),
            )
        except Exception:
            return None


def _looks_like_question(message: str) -> bool:
    text = message.strip().lower()
    if text.endswith("?"):
        return True
    return bool(
        any(
            cue in text
            for cue in (
                "nhắc lại",
                "tóm tắt",
                "tên gì",
                "ở đâu",
                "nghề gì",
                "làm nghề",
                "nghề nghiệp",
                "nơi ở",
                "đâu mới",
                "bạn biết",
                "bạn có biết",
                "đồ uống",
                "món ăn",
                "nuôi con",
                "style trả lời",
            )
        )
    )


def _content_from_langchain(result: Any) -> str:
    if isinstance(result, dict) and "messages" in result:
        last = result["messages"][-1]
        content = getattr(last, "content", last)
        return str(content)
    return str(result)
