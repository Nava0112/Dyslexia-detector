import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from .llm_config import LLM_CONFIG

load_dotenv()


class GuidanceError(RuntimeError):
    pass


def create_guidance(summary: dict) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise GuidanceError("OpenRouter is not configured. Add OPENROUTER_API_KEY to .env.")

    model = os.getenv("OPENROUTER_MODEL", LLM_CONFIG["model"])
    prompt_values = dict(summary)
    prompt_values["Normal"] = summary["class_counts"]["Normal"]
    prompt_values["Corrected"] = summary["class_counts"]["Corrected"]
    prompt_values["Reversal"] = summary["class_counts"]["Reversal"]
    prompt = LLM_CONFIG["prompt_template"].format(**prompt_values)
    payload = json.dumps({
        "model": model,
        "temperature": LLM_CONFIG["temperature"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "messages": [
            {"role": "system", "content": "You provide safe educational handwriting guidance."},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    request = Request(
        LLM_CONFIG["endpoint"],
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": "ScriptSense Handwriting Screening",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise GuidanceError("The guidance service is temporarily unavailable.") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GuidanceError("The guidance service returned an unexpected response.") from exc
    if not isinstance(content, str) or not content.strip():
        raise GuidanceError("The guidance service returned no guidance.")
    return content.strip()
