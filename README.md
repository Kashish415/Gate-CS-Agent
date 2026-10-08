# GATE-CS Agent 🚀

An autonomous dual-LLM pipeline that generates, independently verifies, and publishes daily GATE Computer Science practice questions to a Telegram channel.

Built with Python 3.13, LangChain, Groq, DeepEval, SQLite, Cloudflare R2, and GitHub Actions.

---

## 💡 How It Works

Instead of relying on a single LLM to generate practice questions (which often produces incorrect answers or flawed logic), **GATE-CS Agent** uses a **Dual-LLM Consensus Protocol**:

1. **Slot Picking:** Daily cron picks 3 random subtopics from the GATE CS syllabus (`data/syllabus.json`).
2. **Generator LLM (`llama-3.3-70b-versatile`):** Generates a candidate question (MCQ, MSQ, or NAT) with few-shot context (`data/examples.json`).
3. **Verifier LLM (`llama-3.3-70b-versatile`):** Independently solves the generated question from scratch **without seeing the generator's answer**.
4. **Answer Consensus Resolution:**
   - **Match & High Confidence:** Question is published to Telegram (native Quiz poll for MCQ, Poll + Spoiler answer for MSQ/NAT).
   - **Mismatch or Low Confidence:** Question is discarded and generation is retried (up to 3 retries per slot).
   - **Retry Exhausted:** Falls back to serving an earlier verified question from `verified_cache.json`.
5. **Observability & Storage:** Every attempt, latency, retries, and agreement score are logged to SQLite (`observability/daily_log.db`) and pushed to Cloudflare R2 for state persistence.

```
   ┌──────────────────┐
   │ Daily Cron Trigger│
   └────────┬─────────┘
            │
   ┌────────▼─────────┐
   │  Generator LLM   ├─────────────┐
   └────────┬─────────┘             │
            │ Candidate Question    │ Independent
   ┌────────▼─────────┐             │ Verification
   │   Verifier LLM   │◄────────────┘
   └────────┬─────────┘
            │ Independent Solution
   ┌────────▼─────────┐
   │ Consensus Check  ├── (Mismatch) ──► Retry / Cache Fallback
   └────────┬─────────┘
     (Agree)│
   ┌────────▼─────────┐
   │ Telegram Channel │ + SQLite Log & Cloudflare R2 Backup
   └──────────────────┘
```

---

## 📁 Codebase Structure

Clean, simple structure with no unnecessary abstractions or bloated code:

```
GATE-CS-AGENT/
├── src/
│   ├── domain.py          # Pydantic schemas (QuestionType, SlotSpec, SlotResult, etc.)
│   ├── config.py          # Environment settings, daily slot picker
│   ├── pipeline.py        # Core async pipeline (generate, verify, resolve, publish)
│   ├── observability.py   # SQLite database logger & Cloudflare R2 synced cache
│   └── main.py            # CLI entry point
│
├── evals/                 # DeepEval Evaluation Suite
│   ├── judge.py           # DeepEval custom LLM Judge (Groq llama-3.3-70b)
│   ├── eval_generator.py  # Component Eval: Topic Adherence, Answer Correctness, Spec Compliance
│   ├── eval_verifier.py   # Component Eval: Verifier Accuracy, Confidence Calibration
│   ├── eval_pipeline.py   # Pipeline Eval: Faithfulness, Agreement Quality, Question Quality
│   └── eval_application.py# Application Eval: Toxicity, Bias, SQLite Operational Health
│
├── data/
│   ├── syllabus.json      # GATE CS Syllabus breakdown
│   ├── examples.json      # Few-shot prompts for question generation
│   └── golden.json        # Golden dataset (10 human-verified benchmark questions)
│
├── tests/                 # Unit & Integration Tests (41 tests)
│   ├── test_domain.py
│   ├── test_pipeline.py
│   └── test_observability.py
│
└── .github/workflows/
    └── daily_post.yml     # Daily GitHub Actions cron workflow with R2 persistence & pytest step
```

---

## ⚡ Quick Start

### 1. Requirements & Setup

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

```bash
# Clone the repository
git clone https://github.com/Kashish415/Gate-CS-Agent.git
cd Gate-CS-Agent

# Install dependencies
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
TELEGRAM_BOT_TOKEN=1234567890:AAA...
TELEGRAM_CHANNEL_ID=@your_channel_or_chat_id

# Optional: Observability tracing via LangSmith
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=gate-cs-agent
```

### 3. Run the Daily Pipeline

```bash
uv run gate-cs-agent
```

### 4. Run Unit Tests

```bash
uv run pytest tests/ -v
```

---

## 📊 Evals Framework (DeepEval)

The project includes an **Evals Suite** using [DeepEval](https://github.com/confident-ai/deepeval) to evaluate component quality, end-to-end performance, and safety metrics.

### Running Evals Step-by-Step

| Eval Script | Type | What it Evaluates | Command |
|-------------|------|-------------------|---------|
| `evals/eval_generator.py` | Component | Topic Adherence, Answer Correctness, Spec Compliance | `uv run python -m evals.eval_generator` |
| `evals/eval_verifier.py` | Component | Verifier Accuracy & Confidence Calibration on Golden Dataset | `uv run python -m evals.eval_verifier` |
| `evals/eval_pipeline.py` | Pipeline | End-to-end question quality & agreement on cached pipeline outputs | `uv run python -m evals.eval_pipeline` |
| `evals/eval_application.py`| App / Safety | Safety metrics (Toxicity, Bias) + SQLite operational health stats | `uv run python -m evals.eval_application` |

> **Note:** Component evals (`eval_generator` and `eval_verifier`) run 10 test cases against Groq API and take ~30–40 seconds.

---

## 🛠️ CI/CD Workflow

Automated via **GitHub Actions** (`.github/workflows/daily_post.yml`):
- Runs automatically every day at **02:30 UTC** (08:00 AM IST).
- Restores `daily_log.db` and `verified_cache.json` from **Cloudflare R2** (S3 API).
- Runs `uv run pytest tests/ -v` to ensure code integrity.
- Executes `uv run gate-cs-agent` to post daily questions to Telegram.
- Syncs updated SQLite logs back to Cloudflare R2.

---

## 📜 License

MIT License.
