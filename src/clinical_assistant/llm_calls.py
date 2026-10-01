"""On-disk cache + Langfuse generation tracing for the offline document-processing step only
(applicability.py's guideline-metadata extraction, the one thing kept cached — see
notebooks/README.md). Every per-patient LLM call (note extraction, sub-queries, screening,
generation) bypasses this entirely: it uses Langfuse's native `langfuse.openai` client wrapper
directly instead, which auto-traces the full prompt, output and token usage with zero extra code
— see patient.py, retrieval.py and generation.py for those call sites.
"""

import json
import threading
import time
from typing import NamedTuple

from . import config

_cache: dict | None = None
_lock = threading.Lock()


class LLMResult(NamedTuple):
    """Return this from `cached_llm_call`'s `fn` instead of a bare value to also report token
    usage (so Langfuse can compute cost). `value` is what's cached/returned; `usage` is trace-only."""

    value: object
    usage: dict | None


def extract_token_usage(resp) -> dict | None:
    """Pull Langfuse's usage_details shape out of an OpenAI Responses API result's `.usage`
    (client.responses.parse / .create). Keys per Langfuse's OpenAI cost model: 'input',
    'output', plus optional 'input_cached_tokens' / 'output_reasoning_tokens'."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    details = {"input": usage.input_tokens, "output": usage.output_tokens}
    if usage.input_tokens_details and usage.input_tokens_details.cached_tokens:
        details["input_cached_tokens"] = usage.input_tokens_details.cached_tokens
    if usage.output_tokens_details and usage.output_tokens_details.reasoning_tokens:
        details["output_reasoning_tokens"] = usage.output_tokens_details.reasoning_tokens
    return details


def _call_with_backoff(fn, attempts: int = 8):
    # the org has a 30k tokens/minute limit; wait it out instead of failing the call
    import openai

    for attempt in range(attempts):
        try:
            return fn()
        except openai.RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(15 * (attempt + 1))


def _read_cache_from_disk() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(config.LLM_CACHE_PATH.read_text()) if config.LLM_CACHE_PATH.exists() else {}
    return _cache


def _write_cache_to_disk(cache: dict, key: str, value) -> None:
    with _lock:
        cache[key] = value
        config.LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.LLM_CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False))


def cached_llm_call(key: str, fn, trace_name: str, trace_input=None, trace_model: str | None = None):
    """Document-processing only: read `key` from the on-disk cache, or call `fn` and persist the
    result. Wraps the call (cache hit or real call) in a Langfuse generation — a cache hit still
    needs one of these manually, since there's no real OpenAI call for the native wrapper to
    auto-trace in that case."""
    cache = _read_cache_from_disk()
    cache_hit = key in cache
    langfuse = config.get_langfuse()

    if langfuse is None:
        if cache_hit:
            return cache[key]
        result = _call_with_backoff(fn)
        value = result.value if isinstance(result, LLMResult) else result
        _write_cache_to_disk(cache, key, value)
        return value

    with langfuse.start_as_current_observation(
        as_type="generation",
        name=trace_name,
        model=trace_model,
        input=trace_input,
        version=config.PIPELINE_VERSION,
    ) as gen:
        try:
            if cache_hit:
                value, usage = cache[key], None
            else:
                result = _call_with_backoff(fn)
                value, usage = (result.value, result.usage) if isinstance(result, LLMResult) else (result, None)
                _write_cache_to_disk(cache, key, value)
        except Exception as e:
            gen.update(level="ERROR", status_message=str(e))
            raise
        gen.update(output=value, metadata={"cache_hit": cache_hit}, usage_details=usage)
        return value
