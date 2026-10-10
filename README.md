# GATE-CS Agent

An agentic pipeline that generates, independently verifies, and publishes daily GATE Computer Science practice questions to a Telegram channel.

Built with LangGraph, LangChain, Groq, Pydantic, DeepEval, SQLite, and GitHub Actions.

## How It Works

A single LLM generating exam questions often produces wrong answers or flawed logic. This pipeline uses two separate LLMs in a generate-then-verify loop:

1. **Slot picking**: A daily cron picks 3 random subtopics from the GATE CS syllabus, seeded by date for reproducibility.
2. **Generator** (`gpt-oss-120b` via Groq): Produces a candidate question (MCQ, MSQ, or NAT) using few-shot examples from past GATE papers.
3. **Validation**: Deterministic checks — option count, duplicate options, answer format, answer leak in question text.
4. **Verifier** (`gpt-oss-20b` via Groq): Independently solves the question from scratch without seeing the generator's answer. Reports its own answer, confidence (1-5), and whether the question is ambiguous.
5. **Resolution**: If answers match and confidence is high enough, the question is accepted. Otherwise it retries (up to 2 times) or falls back to a previously verified cached question.
6. **Publishing**: Accepted questions go to Telegram — native quiz poll for MCQ, regular poll + spoiler answer for MSQ, text message + spoiler for NAT.

All attempts are logged to SQLite for observability.

```
START --> generate --> validate --+--> verify --> resolve --+--> publish --> END
                                 |                         |
                                 +--> resolve (skip) ------+
                                                           |
                                        retry: generate <--+
                                        exhausted: fallback --> publish --> END
```

## Project Structure

```
src/
    domain.py       Pydantic models (QuestionPayload, VerifierPayload, VerifiedQuestion)
    config.py       Constants, paths, daily slot picker
    pipeline.py     LangGraph pipeline (generate, validate, verify, resolve, fallback, publish)
    storage.py      SQLite database + JSON cache for verified questions
    telegram.py     Telegram publishing (quiz polls, text messages with spoilers)
    main.py         Entry point

evals/
    judge.py            DeepEval custom LLM judge (Groq)
    eval_generator.py   Component eval: topic adherence, answer correctness
    eval_verifier.py    Component eval: verifier accuracy, confidence calibration
    eval_pipeline.py    Pipeline eval: spec faithfulness, agreement quality, question quality
    eval_application.py Application eval: toxicity, bias, operational metrics from SQLite

tests/
    test_validators.py   Validation logic (option count, answer leak, format checks)
    test_comparators.py  Answer comparison (MCQ exact match, MSQ set match, NAT tolerance)
    test_pipeline.py     SQLite logging, Telegram markdown escaping

data/
    syllabus.json         GATE CS syllabus breakdown by subject and subtopic
    examples.json         Few-shot examples from past GATE papers
    golden.json           10 human-verified benchmark questions for eval
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Kashish415/Gate-CS-Agent.git
cd Gate-CS-Agent
uv sync
```

Create a `.env` file:

```
GROQ_API_KEY=gsk_...
TELEGRAM_BOT_TOKEN=1234567890:AAA...
TELEGRAM_CHANNEL_ID=@your_channel
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=gate-cs-agent
```

## Usage

```bash
# Run the daily pipeline
uv run gate-cs-agent

# Run unit tests (36 tests)
uv run pytest tests/ -v

# Run evals (requires GROQ_API_KEY)
uv run python -m evals.eval_generator
uv run python -m evals.eval_verifier
uv run python -m evals.eval_pipeline
uv run python -m evals.eval_application
```

## Eval Results

Evals run at three levels:

| Level | Script | Metrics | Latest |
|-------|--------|---------|--------|
| Component | eval_generator | Topic Adherence, Answer Correctness | 100%, 90% |
| Component | eval_verifier | Verifier Accuracy, Confidence Calibration | 100%, 100% |
| Pipeline | eval_pipeline | Spec Faithfulness, Agreement Quality, Question Quality | 100%, 55%, 90% |
| Application | eval_application | Toxicity, Bias | Pass, Pass |

Agreement Quality scores low because the GEval judge sometimes outputs miscalibrated scores even when its own reasoning confirms correctness. This is a known GEval issue with binary evaluation rubrics.

## Known Limitations

- **Same model family**: Both generator and verifier are from the Groq model family. Agreement is not proof of correctness — it filters obvious errors but can miss shared blind spots.
- **Eval judge circularity**: The GEval judge uses the same model as the generator. Eval scores measure self-consistency, not ground truth accuracy.
- **Small sample sizes**: Evals run on 10 golden questions (component) and ~20 cached questions (pipeline). Larger datasets would give more reliable metrics.
- **No human-in-the-loop**: Published questions are not reviewed by a human before posting. The pipeline relies entirely on automated verification.

## CI/CD

GitHub Actions runs daily at 03:00 UTC (`.github/workflows/daily_post.yml`):
1. Restores SQLite DB and verified cache from Cloudflare R2
2. Runs pytest
3. Runs the pipeline to generate and publish questions
4. Persists state back to R2

R2 (S3-compatible storage) is used because GitHub Actions runners are ephemeral — without it, the database and cache would be lost after each run.

## License

MIT
