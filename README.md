# Dyslexia-detector

## Optional LLM guidance

The analysis view can request plain-language learning guidance from OpenRouter
after a screening completes. Copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY`. `OPENROUTER_MODEL` may be set to any model available
through OpenRouter.

Only the status, score, threshold, line/crop counts, and the three numerical
class counts are sent to the guidance service. The uploaded handwriting image
is never sent. If OpenRouter is unavailable or not configured, the primary
analysis results remain available and the guidance panel shows a retry state.
