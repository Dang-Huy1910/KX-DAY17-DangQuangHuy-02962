from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state lives under tmp_path, compact fires quickly."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    dummy = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,
        compact_keep_messages=4,
        model=dummy,
        judge_model=dummy,
        profile_confidence_threshold=0.65,
        memory_decay_half_life_days=30.0,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    path = store.write_text("alice", "# User Profile\n\n- name: Alice\n")
    assert path.exists()
    assert "Alice" in store.read_text("alice")
    assert store.file_size("alice") > 0

    changed = store.edit_text("alice", "Alice", "Alicia")
    assert changed is True
    assert "Alicia" in store.read_text("alice")
    assert store.edit_text("alice", "missing-text", "x") is False

    store.upsert_fact("alice", "location", "Huế")
    facts = store.facts("alice")
    assert facts["name"] == "Alicia"
    assert facts["location"] == "Huế"

    store.upsert_fact("alice", "location", "Đà Nẵng")
    assert store.facts("alice")["location"] == "Đà Nẵng"
    assert store.facts("alice")["location"] != "Huế"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )
    long_turn = "Đây là một lượt hội thoại khá dài để ép compact memory kích hoạt. " * 8
    for index in range(12):
        manager.append("thread-compact", "user", f"{long_turn} #{index}")

    assert manager.compaction_count("thread-compact") >= 1
    ctx = manager.context("thread-compact")
    assert str(ctx["summary"]).strip()
    assert len(ctx["messages"]) <= config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    intro = (
        "Chào bạn, mình tên là DũngCT. Mình ở Huế và đang làm MLOps engineer. "
        "Mình muốn bạn trả lời ngắn gọn."
    )
    baseline.reply("dungct", "session-1", intro)
    advanced.reply("dungct", "session-1", intro)

    question = "Mình tên gì và hiện tại mình làm nghề gì? Mình đang ở đâu?"
    baseline_answer = baseline.reply("dungct", "session-2", question)["answer"]
    advanced_answer = advanced.reply("dungct", "session-2", question)["answer"]

    assert "DũngCT" not in baseline_answer
    assert "MLOps engineer" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "MLOps engineer" in advanced_answer
    assert "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.compact_threshold_tokens = 60
    config.compact_keep_messages = 4
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    long_turn = (
        "Đây là stress test compact. Mình tên là DũngCT Stress, đang ở Đà Nẵng "
        "và làm MLOps engineer. " + ("Tin tức dài để đội ngữ cảnh. " * 25)
    )
    thread_id = "long-thread"
    for index in range(16):
        baseline.reply("stress", thread_id, f"{long_turn} lượt {index}")
        advanced.reply("stress", thread_id, f"{long_turn} lượt {index}")

    assert advanced.compaction_count(thread_id) >= 1
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
