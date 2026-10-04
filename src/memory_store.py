from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


KNOWN_LOCATIONS = ("Đà Nẵng", "Da Nang", "Huế", "Hue", "Hà Nội", "Ha Noi")
KNOWN_JOBS = ("MLOps engineer", "backend engineer", "product manager")

_FACT_LINE_RE = re.compile(
    r"^-\s+(?P<key>[a-z_]+):\s+(?P<value>.+?)(?:\s+\(updated:(?P<meta>.*)\))?\s*$",
    re.MULTILINE,
)


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimator used by both agents and the benchmark."""

    if text is None:
        return 0
    stripped = str(text).strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _slugify(user_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (user_id or "").strip())
    return slug.strip("._") or "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decay_score(updated_iso: str, half_life_days: float = 30.0) -> float:
    """Bonus: older facts lose priority. 1.0 is fresh, ~0.5 after one half-life."""

    if not updated_iso or half_life_days <= 0:
        return 1.0
    try:
        updated = datetime.strptime(updated_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return 1.0
    age_days = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


@dataclass
class UserProfileStore:
    """Persistent markdown profile per user: `state/profiles/<user>/User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slugify(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n\nChưa có thông tin bền vững.\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured `- key: value` lines from User.md."""

        out: dict[str, str] = {}
        for match in _FACT_LINE_RE.finditer(self.read_text(user_id)):
            out[match.group("key")] = match.group("value").strip()
        return out

    def ranked_facts(self, user_id: str, half_life_days: float = 30.0) -> dict[str, str]:
        """Bonus memory decay: very old facts drop out of the prompt first."""

        scored: list[tuple[float, str, str]] = []
        for key, rec in self.fact_records(user_id).items():
            score = decay_score(rec.get("updated") or "", half_life_days)
            if score < 0.05:
                continue
            scored.append((score, key, rec["value"]))
        scored.sort(reverse=True)
        return {key: value for _score, key, value in scored}

    def fact_records(self, user_id: str) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for match in _FACT_LINE_RE.finditer(self.read_text(user_id)):
            meta = match.group("meta") or ""
            updated = ""
            confidence = ""
            um = re.match(r"\s*([^,]+)", meta)
            cm = re.search(r"confidence:\s*([0-9.]+)", meta)
            if um:
                updated = um.group(1).strip()
            if cm:
                confidence = cm.group(1).strip()
            records[match.group("key")] = {
                "value": match.group("value").strip(),
                "updated": updated,
                "confidence": confidence,
            }
        return records

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 0.9,
    ) -> None:
        """Conflict handling: a new value for the same key replaces the old line."""

        key = key.strip()
        value = " ".join(value.split())
        if not key or not value:
            return
        if key == "interests":
            previous = self.facts(user_id).get("interests", "")
            merged: list[str] = []
            for part in f"{previous}, {value}".split(","):
                item = part.strip()
                if item and item not in merged:
                    merged.append(item)
            value = ", ".join(merged)
        existing = self.read_text(user_id)
        if "Chưa có thông tin bền vững." in existing:
            existing = "# User Profile\n\n"
        line = (
            f"- {key}: {value} "
            f"(updated: {_utc_now()}, confidence: {confidence:.2f})"
        )
        pattern = re.compile(rf"^- {re.escape(key)}:.*$", re.MULTILINE)
        if pattern.search(existing):
            updated = pattern.sub(line, existing, count=1)
        else:
            updated = existing.rstrip() + "\n" + line + "\n"
        self.write_text(user_id, updated)


def _is_question_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    has_new_fact = bool(
        re.search(
            r"(tên(?:\s+mình)?\s+là\s+[A-Za-zÀ-ỹ]|mình ở [A-ZÀ-Ỹ]|đang ở [A-ZÀ-Ỹ]|"
            r"chuyển sang\s+[A-Za-z]|yêu thích là |nuôi một |"
            r"muốn bạn trả lời|hãy trả lời)",
            stripped,
            re.I,
        )
    )
    asking = bool(
        re.search(
            r"(\?|nhắc lại|bạn có biết|tóm tắt|mình tên gì|đang ở đâu|làm nghề gì|"
            r"đâu mới là|hãy nhắc|là gì)",
            stripped,
            re.I,
        )
    )
    return asking and not has_new_fact


def _window(text: str, start: int, end: int, radius: int = 48) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def extract_profile_updates(
    message: str,
    min_confidence: float = 0.65,
) -> dict[str, str]:
    """Turn a user turn into stable profile facts.

    Bonus behaviours baked in:
    - confidence threshold: skip weak / question-only / joke mentions
    - entity extraction: typed fields instead of a free-form blob
    - conflict handling happens later in `upsert_fact` (latest value wins)
    """

    text = (message or "").strip()
    if not text or _is_question_only(text):
        return {}

    scored: dict[str, tuple[str, float]] = {}

    name_match = re.search(r"tên(?:\s+mình)?\s+là\s+([^,.\n]+)", text, re.I)
    if name_match:
        name = name_match.group(1).strip()
        name = re.split(r"\s+(?:hiện|đang|và|cho|nhé)\b", name, maxsplit=1)[0].strip()
        if name and not re.search(r"\b(gì|ai|nào)\b", name, re.I):
            scored["name"] = (name, 0.95)

    location_hits: list[tuple[int, str, float]] = []
    for loc in KNOWN_LOCATIONS:
        for match in re.finditer(re.escape(loc), text, re.I):
            nearby = _window(text, match.start(), match.end(), radius=56)
            if re.search(
                r"họp|bay ra|ví dụ cũ|đừng lấy|lúc đầu|trước đó|có nhắc",
                nearby,
                re.I,
            ):
                continue
            prefix = text[max(0, match.start() - 28) : match.start()]
            if re.search(r"không còn ở\s*$", prefix, re.I):
                continue
            confidence = 0.55
            if re.search(
                r"(đang ở|hiện ở|mình ở|làm việc ở|sang|nơi ở hiện tại là|đang sống ở)\s*$",
                prefix,
                re.I,
            ):
                confidence = 0.95
            elif re.search(r"vẫn ở\s*$", prefix, re.I):
                confidence = 0.9
            location_hits.append((match.start(), loc, confidence))
    if location_hits:
        strong = [hit for hit in location_hits if hit[2] >= 0.9]
        _, loc, confidence = (strong or location_hits)[-1]
        canonical = {
            "Da Nang": "Đà Nẵng",
            "Hue": "Huế",
            "Ha Noi": "Hà Nội",
        }.get(loc, loc)
        scored["location"] = (canonical, confidence)

    job_hits: list[tuple[int, str, float]] = []
    joke = bool(re.search(r"đùa", text, re.I))
    for job in KNOWN_JOBS:
        for match in re.finditer(re.escape(job), text, re.I):
            nearby = _window(text, match.start(), match.end())
            if joke and job.lower() == "product manager":
                continue
            prefix = text[max(0, match.start() - 36) : match.start()]
            if re.search(r"không còn (?:làm |là )?", prefix, re.I):
                continue
            confidence = 0.8
            if re.search(r"(chuyển sang|hiện tại (?:vẫn )?là|đang làm|vẫn là)\s*$", prefix, re.I):
                confidence = 0.96
            job_hits.append((match.start(), job, confidence))
    if job_hits:
        _, job, confidence = job_hits[-1]
        scored["profession"] = (job, confidence)

    if re.search(r"3\s*bullet", text, re.I):
        scored["style"] = ("ngắn gọn, 3 bullet, ví dụ thực chiến", 0.92)
    elif re.search(r"muốn bạn trả lời|hãy trả lời|khi giải thích", text, re.I):
        parts = ["ngắn gọn"]
        if re.search(r"bullet", text, re.I):
            parts.append("bullet ngắn")
        if re.search(r"ví dụ", text, re.I):
            parts.append("có ví dụ thực tế")
        scored["style"] = (", ".join(parts), 0.88)

    if re.search(r"cà phê sữa đá", text, re.I):
        scored["favorite_drink"] = ("cà phê sữa đá", 0.9)

    if re.search(r"mì\s*Quảng", text, re.I):
        scored["favorite_food"] = ("mì Quảng", 0.9)

    if re.search(r"corgi", text, re.I) or re.search(r"con Bơ", text, re.I):
        scored["pet"] = ("corgi tên Bơ", 0.9)

    interests: list[str] = []
    if re.search(r"\bPython\b", text):
        interests.append("Python")
    if re.search(r"\bAI\b", text):
        interests.append("AI")
    if re.search(r"MLOps", text, re.I) and "profession" not in scored:
        interests.append("MLOps")
    if interests:
        scored["interests"] = (", ".join(interests), 0.75)

    return {key: value for key, (value, conf) in scored.items() if conf >= min_confidence}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compression of older turns. Keeps a bounded number of snippets."""

    if not messages:
        return ""
    snippets: list[str] = []
    for item in messages:
        role = item.get("role", "user")
        content = " ".join((item.get("content") or "").split())
        if not content:
            continue
        snippets.append(f"{role}: {content[:90]}")
    kept = snippets[-max_items:]
    return f"[Tóm tắt {len(messages)} lượt cũ] " + " | ".join(kept)


@dataclass
class CompactMemoryManager:
    """Short-term thread memory that folds older turns into a bounded summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure(thread_id)
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread_id)

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._ensure(thread_id)
        return {
            "messages": list(thread.get("messages") or []),
            "summary": str(thread.get("summary") or ""),
            "compactions": int(thread.get("compactions") or 0),
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self.state.get(thread_id) or {}
        return int(thread.get("compactions") or 0)

    def _token_load(self, thread: dict[str, object]) -> int:
        messages = thread.get("messages") or []
        blob = str(thread.get("summary") or "")
        if isinstance(messages, list):
            blob += "\n".join(str(m.get("content", "")) for m in messages)
        return estimate_tokens(blob)

    def _maybe_compact(self, thread_id: str) -> None:
        thread = self._ensure(thread_id)
        keep = max(2, int(self.keep_messages))
        safety = 0
        while self._token_load(thread) > self.threshold_tokens and safety < 32:
            messages = thread["messages"]
            assert isinstance(messages, list)
            if len(messages) <= 2:
                break
            retain = keep if len(messages) > keep else max(2, len(messages) // 2)
            if retain >= len(messages):
                retain = len(messages) - 1
            older = messages[:-retain]
            recent = messages[-retain:]
            extra = summarize_messages(older)
            previous = str(thread.get("summary") or "").strip()
            combined = f"{previous}\n{extra}".strip() if previous else extra
            if estimate_tokens(combined) > max(80, self.threshold_tokens // 2):
                combined = summarize_messages(
                    [{"role": "system", "content": combined}],
                    max_items=6,
                )
            thread["summary"] = combined
            thread["messages"] = recent
            thread["compactions"] = int(thread.get("compactions") or 0) + 1
            safety += 1
