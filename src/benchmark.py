from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 0.0
    haystack = (answer or "").lower()
    hits = sum(1 for item in expected if item.lower() in haystack)
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for the deterministic offline path."""

    text = (answer or "").strip()
    if not text:
        return 0.0
    recall = recall_points(text, expected)
    length_score = 1.0 if 40 <= len(text) <= 900 else 0.7
    structure = 0.1 if any(mark in text for mark in ("-", "•", "\n")) else 0.0
    return round(min(1.0, 0.7 * recall + 0.2 * length_score + structure), 3)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    users = {str(conv.get("user_id", "")) for conv in conversations}
    memory_start = 0
    if hasattr(agent, "memory_file_size"):
        memory_start = sum(agent.memory_file_size(user) for user in users if user)

    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = str(conv["user_id"])
        thread_id = str(conv["id"])
        for turn in conv.get("turns") or []:
            agent.reply(user_id, thread_id, str(turn))

        agent_tokens += int(agent.token_usage(thread_id))
        prompt_tokens += int(agent.prompt_token_usage(thread_id))
        compactions += int(agent.compaction_count(thread_id))

        for index, item in enumerate(conv.get("recall_questions") or []):
            expected = list(item.get("expected_contains") or [])
            recall_thread = f"{thread_id}-recall-{index}"
            result = agent.reply(user_id, recall_thread, str(item["question"]))
            answer = str(result.get("answer") or "")
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    memory_end = 0
    if hasattr(agent, "memory_file_size"):
        memory_end = sum(agent.memory_file_size(user) for user in users if user)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    del config
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=max(0, memory_end - memory_start),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        widths = [len(h) for h in headers]
        for rec in table:
            for i, cell in enumerate(rec):
                widths[i] = max(widths[i], len(str(cell)))
        def fmt(values: list[Any]) -> str:
            return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"
        line = "| " + " | ".join("-" * w for w in widths) + " |"
        return "\n".join([fmt(headers), line, *[fmt(rec) for rec in table]])


def _print_analysis(standard: list[BenchmarkRow], stress: list[BenchmarkRow]) -> None:
    std = {row.agent_name: row for row in standard}
    lng = {row.agent_name: row for row in stress}
    print("\n## Phân tích nhanh")
    print(
        "- Advanced recall cao hơn Baseline vì `User.md` sống qua thread mới, "
        "còn Baseline chỉ nhớ trong cùng `thread_id`."
    )
    if "Advanced" in std and "Baseline" in std:
        print(
            "- Ở hội thoại ngắn, Advanced thường tốn hơn Baseline về "
            "`Prompt tokens processed` vì mỗi lượt còn kéo theo User.md + summary, "
            "trong khi compact chưa kịp kích hoạt."
        )
        print(
            f"- Standard prompt tokens: Baseline {std['Baseline'].prompt_tokens_processed} vs "
            f"Advanced {std['Advanced'].prompt_tokens_processed}."
        )
    if "Advanced" in lng and "Baseline" in lng:
        print(
            "- Compact tối ưu chủ yếu `Prompt tokens processed`: hội thoại dài bị nén "
            "thành summary + cửa sổ message gần nhất, không gửi lại toàn bộ lịch sử."
        )
        print(
            f"- Stress prompt tokens: Baseline {lng['Baseline'].prompt_tokens_processed} vs "
            f"Advanced {lng['Advanced'].prompt_tokens_processed} "
            f"(compactions={lng['Advanced'].compactions})."
        )
        print(
            f"- Memory growth của Advanced trên stress: {lng['Advanced'].memory_growth_bytes} bytes. "
            "File `User.md` phình theo số fact; ghi sai correction sẽ làm bẩn hồ sơ."
        )


def _fresh_config(config: LabConfig, suffix: str) -> LabConfig:
    state = config.state_dir / "runs" / suffix
    if state.exists():
        shutil.rmtree(state)
    state.mkdir(parents=True, exist_ok=True)
    return replace(config, state_dir=state)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"
    standard = load_conversations(standard_path)
    stress = load_conversations(stress_path)

    print("# Standard Benchmark")
    standard_rows = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(_fresh_config(config, "standard-baseline"), force_offline=True),
            standard,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(_fresh_config(config, "standard-advanced"), force_offline=True),
            standard,
            config,
        ),
    ]
    print(format_rows(standard_rows))

    print("\n# Long-Context Stress Benchmark")
    stress_rows = [
        run_agent_benchmark(
            "Baseline",
            BaselineAgent(_fresh_config(config, "stress-baseline"), force_offline=True),
            stress,
            config,
        ),
        run_agent_benchmark(
            "Advanced",
            AdvancedAgent(_fresh_config(config, "stress-advanced"), force_offline=True),
            stress,
            config,
        ),
    ]
    print(format_rows(stress_rows))
    _print_analysis(standard_rows, stress_rows)


if __name__ == "__main__":
    main()
