# GATE-CS Agent

An agentic pipeline that generates, independently verifies, and publishes daily GATE Computer Science practice questions to a Telegram channel.

Built with LangGraph, LangChain, Groq, Pydantic, DeepEval, SQLite, and GitHub Actions.

## How It Works

A single LLM generating exam questions often produces wrong answers or flawed logic. This pipeline uses two separate LLMs in a generate-then-verify loop:

1. **Slot picking**: A daily cron job picks 3 random subtopics from the GATE CS syllabus.
2. **Generator** (`gpt-oss-120b` model): Produces a candidate question (MCQ, MSQ and NAT) using few-shot examples from past GATE papers.
3. **Validation**: Deterministic checks — option count, duplicate options, answer format, answer leak in question text.
4. **Verifier** (`gpt-oss-20b` model): Independently solves the question from scratch without seeing the generator's answer. Reports its own answer, confidence score, and whether the question is ambiguous.
5. **Resolution**: If answers match and confidence is high enough, the question is accepted. Otherwise it retries (up to 2 times) or falls back to a previously verified cached question.
6. **Publishing**: Accepted questions gets published to a dedicated telegram channel.

All attempts are logged to SQLite for observability and traced via LangSmith.

### Pipeline Graph (LangGraph StateGraph)

<img width="166" height="729" alt="image" src="https://github.com/user-attachments/assets/0756b4cb-472d-4ce0-94bc-517be62c8c1b" />


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

## Eval Results

Evals run at three levels using DeepEval:

| Level | Script | Metrics | Results (n=10-22) |
|-------|--------|---------|-------------------|
| Component | eval_generator | Topic Adherence, Answer Correctness | 100%, 90% |
| Component | eval_verifier | Verifier Accuracy, Confidence Calibration | 100%, 100% |
| Pipeline | eval_pipeline | Spec Faithfulness, Agreement Quality, Question Quality | 100%, 100%, 90% |
| Application | eval_application | Toxicity, Bias | Pass, Pass |

## Observability

Pipeline runs are traced via LangSmith with alerts configured for:
- Pipeline errors (run failures)
- High latency (runs exceeding 60s)

LangSmith dashboard tracks trace count, latency percentiles, error rate, and token usage over time.

## CI/CD

GitHub Actions runs daily at 03:00 IST (`.github/workflows/daily_post.yml`):
1. Restores SQLite DB and verified cache from GitHub Actions cache
2. Runs pytest
3. Runs the pipeline to generate and publish questions
4. Saves state back to cache

## License

MIT
