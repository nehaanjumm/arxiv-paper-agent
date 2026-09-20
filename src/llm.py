import os
import time
import json
from groq import Groq, RateLimitError

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def chat(messages: list[dict], temperature: float = 0.2,
         json_mode: bool = False, retries: int = 3) -> str:
    """Call Groq with simple backoff for free-tier rate limits."""
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL, messages=messages,
                temperature=temperature, **kwargs)
            return resp.choices[0].message.content
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("LLM rate limit: please wait a minute and retry.")


def chat_json(messages: list[dict], **kw) -> dict:
    return json.loads(chat(messages, json_mode=True, **kw))