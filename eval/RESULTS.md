# Company classifier results (8-class taxonomy, gpt-5-mini)

| Version | Dev strict (65) | Dev lenient | Held-out strict (32) | Held-out lenient | Median latency |
|---|---|---|---|---|---|
| One LLM call, no guide | 0.80 | 0.83 | - | - | ~5s |
| One LLM call + evidence guide | 0.91 | 0.91 | 0.91 | 0.91 | ~5s |
| LangGraph agent (rule-routed web + validator + retry) | 0.95 | 0.97 | 0.94 | 0.97 | ~14s |

Held-out set committed to git before any model run.
Web search used: 47/65 dev, 22/32 held-out (skipped when a decisive FDA signal exists).
Validator retries: 2 total (HealthFirst, Twin Med); both fixed the answer, none broke one.
Human review triggered: 0/97. Silent errors: American Health Packaging, Major (dev); Clarion Brands (held-out).
Key finding: LLM self-reported confidence was uncalibrated (wrong answers at 0.90-0.95), so escalation uses an independent validator, not confidence.