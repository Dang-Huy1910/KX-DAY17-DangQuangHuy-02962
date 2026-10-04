# Student implementation

Offline-first memory lab: Baseline (thread-only) vs Advanced (`User.md` + compact memory).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/benchmark.py
pytest src/test_agents.py -v
```

Supported live providers (optional, via `.env`): `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`. Benchmark and tests force the deterministic offline path.

See `ANALYSIS.md` at the repo root for the Standard vs Long-Context comparison.
