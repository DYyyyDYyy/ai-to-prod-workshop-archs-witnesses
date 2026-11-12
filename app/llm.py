"""Тонка обгортка над LLM API (Gemini через OpenAI-сумісний endpoint).

Обмеження:
  * викликайте модель лише через офіційний `openai` SDK (без streaming);
  * модель беріть з LLM_MODEL — у фінальному прогоні організатори задають її самі;
  * ключ — тільки зі змінної середовища GEMINI_API_KEY, ніколи не хардкодьте.

Ретраї робимо самі (SDK max_retries=0), щоб контролювати загальний дедлайн:
дефолтні 600 с таймауту і непередбачувані паузи SDK нам не підходять.
"""
import os
import random
import threading
import time

from dotenv import load_dotenv
import openai
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

PER_CALL_TIMEOUT = 7.0   # секунд на одну спробу
MAX_ATTEMPTS = 6

_client = None
_client_lock = threading.Lock()


class LLMError(Exception):
    pass


def get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise LLMError("GEMINI_API_KEY не задано")
            _client = OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=PER_CALL_TIMEOUT)
        return _client


_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def chat(messages, *, budget: float = 12.0, **kwargs) -> str:
    """Надсилає повідомлення моделі й повертає текст відповіді ("" якщо порожньо).

    Ретраїть 429/5xx/таймаути з backoff, поки не вичерпано `budget` секунд.
    Кидає LLMError, якщо так і не вдалося.
    """
    deadline = time.monotonic() + budget
    last = None
    for attempt in range(MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining < 0.5:
            break
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=min(PER_CALL_TIMEOUT, remaining),
                **kwargs,
            )
            if not resp.choices:
                return ""
            return resp.choices[0].message.content or ""
        except _RETRYABLE as e:
            last = e
        except openai.APIStatusError as e:  # 4xx окрім 429 — ретрай не допоможе
            if e.status_code >= 500 or e.status_code == 408:
                last = e
            else:
                raise LLMError(f"{type(e).__name__}: {e}") from e
        sleep = min(0.3 * (2 ** attempt), 3.0) + random.uniform(0, 0.2)
        if time.monotonic() + sleep > deadline - 0.5:
            break
        time.sleep(sleep)
    raise LLMError(f"LLM недоступна: {type(last).__name__ if last else 'budget'}: {last}")
