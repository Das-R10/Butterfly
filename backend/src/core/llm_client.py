"""
src/core/llm_client.py

Dual-provider LLM abstraction supporting:
  - Local Ollama (HTTP localhost:11434)
  - Fast Cloud APIs (Groq, Google Gemini)
  - Deterministic Mock Provider (for offline testing & CI verification only)

Enforces strict JSON schema generation, safe bounded retries, and detailed
structured error diagnostics without ever falling back silently to Mock.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Type
from loguru import logger
from pydantic import BaseModel, ValidationError
import requests

from src.core.schemas import Archetype, InterpretationOutput, ModelExecutionStatus

# Model defaults
DEFAULT_PROVIDER_A = os.getenv("BUTTERFLY_PROVIDER_A", "mock")
DEFAULT_MODEL_A = os.getenv("BUTTERFLY_MODEL_A", "qwen3.5:4b")
DEFAULT_PROVIDER_B = os.getenv("BUTTERFLY_PROVIDER_B", "mock")
DEFAULT_MODEL_B = os.getenv("BUTTERFLY_MODEL_B", "gemma3:4b")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10.0"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))


def _get_int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


OLLAMA_MAX_RETRIES = _get_int_env("OLLAMA_MAX_RETRIES", 2, minimum=0)
OLLAMA_NUM_PREDICT = _get_int_env("OLLAMA_NUM_PREDICT", 1792, minimum=256)
OLLAMA_THINK = _get_bool_env("OLLAMA_THINK", False)


def sanitize_error_text(text: str) -> str:
    """Scrubs potential API key / bearer token substrings from error telemetry."""
    if not text:
        return ""
    scrubbed = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", str(text))
    scrubbed = re.sub(r"\b(sk|gsk|AIza)[A-Za-z0-9_\-]{10,}\b", "[REDACTED_KEY]", scrubbed)
    return scrubbed


class LLMInferenceError(RuntimeError):
    """
    Raised when real LLM inference fails. Prevents silent fallback to mock
    and carries structured diagnostics (category, status, retries, finish reason).
    """

    def __init__(
        self,
        message: str,
        *,
        execution_status: str = ModelExecutionStatus.PROVIDER_ERROR.value,
        error_category: str = "provider_error",
        retry_count: int = 0,
        finish_reason: Optional[str] = None,
        response_length: Optional[int] = None,
        thinking_length: Optional[int] = None,
        duration_sec: float = 0.0,
        original_error: Optional[str] = None,
    ):
        clean_msg = sanitize_error_text(message)
        super().__init__(clean_msg)
        self.execution_status = execution_status
        self.error_category = error_category
        self.retry_count = retry_count
        self.finish_reason = finish_reason
        self.response_length = response_length
        self.thinking_length = thinking_length
        self.duration_sec = duration_sec
        self.original_error = sanitize_error_text(original_error or clean_msg)


def clean_json_response(raw_text: str) -> str:
    """Controlled recovery helper extracting JSON object from fences or outer braces."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text


def parse_and_validate_json(
    raw_text: Optional[str],
    *,
    model_name: str,
    finish_reason: Optional[str] = None,
    thinking_length: int = 0,
    response_model: Optional[Type[BaseModel]] = None,
) -> Tuple[dict, bool]:
    """
    Strict JSON parsing & optional Pydantic schema validation.
    Distinguishes: empty_output, truncated_output, malformed_json, schema_validation_failure.
    Returns (parsed_dict, used_controlled_recovery).
    """
    raw = "" if raw_text is None else str(raw_text)
    stripped = raw.strip()
    resp_len = len(raw)

    if not stripped:
        reason_note = f"finish_reason={finish_reason or 'unknown'}, thinking_length={thinking_length}"
        if finish_reason == "length" and thinking_length > 0:
            raise LLMInferenceError(
                f"Model '{model_name}' returned an empty response after internal thinking consumed token budget ({reason_note}).",
                execution_status=ModelExecutionStatus.INVALID_JSON.value,
                error_category="empty_output",
                finish_reason=finish_reason,
                response_length=resp_len,
                thinking_length=thinking_length,
            )
        raise LLMInferenceError(
            f"Model '{model_name}' returned an empty response ({reason_note}).",
            execution_status=ModelExecutionStatus.INVALID_JSON.value,
            error_category="empty_output",
            finish_reason=finish_reason,
            response_length=resp_len,
            thinking_length=thinking_length,
        )

    used_recovery = False
    parsed: Any = None

    # Attempt strict direct parse first
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as direct_err:
        # Controlled recovery only if response wraps a valid JSON object (e.g. ```json ... ```)
        recovered = clean_json_response(stripped)
        if recovered != stripped and recovered.startswith("{") and recovered.endswith("}"):
            try:
                parsed = json.loads(recovered)
                used_recovery = True
            except json.JSONDecodeError as rec_err:
                category = "truncated_output" if finish_reason == "length" else "malformed_json"
                raise LLMInferenceError(
                    f"Model '{model_name}' returned {category} ({rec_err}). Preview: {stripped[:160]}",
                    execution_status=ModelExecutionStatus.INVALID_JSON.value,
                    error_category=category,
                    finish_reason=finish_reason,
                    response_length=resp_len,
                    thinking_length=thinking_length,
                    original_error=str(rec_err),
                )
        else:
            category = "truncated_output" if finish_reason == "length" else "malformed_json"
            raise LLMInferenceError(
                f"Model '{model_name}' returned {category} ({direct_err}). Preview: {stripped[:160]}",
                execution_status=ModelExecutionStatus.INVALID_JSON.value,
                error_category=category,
                finish_reason=finish_reason,
                response_length=resp_len,
                thinking_length=thinking_length,
                original_error=str(direct_err),
            )

    if not isinstance(parsed, dict):
        raise LLMInferenceError(
            f"Model '{model_name}' returned JSON {type(parsed).__name__} instead of a JSON object.",
            execution_status=ModelExecutionStatus.VALIDATION_FAILURE.value,
            error_category="schema_validation_failure",
            finish_reason=finish_reason,
            response_length=resp_len,
            thinking_length=thinking_length,
        )

    if response_model is not None:
        try:
            validated = response_model(**parsed)
            return validated.model_dump(), used_recovery
        except ValidationError as ve:
            raise LLMInferenceError(
                f"Model '{model_name}' JSON failed Pydantic {response_model.__name__} validation: {ve.errors()[:2]}",
                execution_status=ModelExecutionStatus.VALIDATION_FAILURE.value,
                error_category="schema_validation_failure",
                finish_reason=finish_reason,
                response_length=resp_len,
                thinking_length=thinking_length,
                original_error=str(ve),
            )

    return parsed, used_recovery


def _build_corrective_prompt(base_prompt: str, err: LLMInferenceError) -> str:
    """Appends a targeted correction instruction on retry without mutating signal facts."""
    cat = err.error_category
    if cat == "empty_output":
        hint = (
            "CORRECTION REQUIRED: Your previous output was completely empty. "
            "Emit ONLY a single compact JSON object matching all required keys immediately."
        )
    elif cat == "truncated_output":
        hint = (
            "CORRECTION REQUIRED: Your previous output was truncated before completing the JSON object. "
            "Keep narrative strings concise (1-2 sentences) and close all JSON brackets properly."
        )
    elif cat == "schema_validation_failure":
        hint = (
            f"CORRECTION REQUIRED: Your previous JSON failed schema validation ({err.original_error[:180]}). "
            "Ensure all required fields exist with exact types and valid numeric ranges."
        )
    else:
        hint = (
            f"CORRECTION REQUIRED: Your previous output was malformed JSON ({err.original_error[:180]}). "
            "Return ONLY valid JSON with no prose outside braces."
        )
    return f"{base_prompt}\n\n{hint}"


class LLMClient:
    """Wrapper that routes calls to Ollama, Groq, Gemini, or Mock with bounded retries & telemetry."""

    def __init__(self, provider: str, model_name: str):
        self.provider = (provider or "mock").lower().strip()
        self.model_name = (model_name or "default").strip()
        self._groq_client = None
        self._gemini_client = None
        self._ollama_disable_think_param = False
        self.last_duration_sec: float = 0.0
        self.last_error: Optional[str] = None
        self.last_call_metadata: Dict[str, Any] = {}

        if self.provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self._groq_client = Groq(api_key=api_key)
                else:
                    logger.warning("GROQ_API_KEY is not set.")
            except ImportError:
                logger.warning("groq Python package not installed.")

        elif self.provider in ("gemini", "google"):
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._gemini_client = genai.GenerativeModel(self.model_name)
                else:
                    logger.warning("GEMINI_API_KEY is not set.")
            except ImportError:
                logger.warning("google-generativeai package not installed.")

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        json_schema: Optional[dict] = None,
        response_model: Optional[Type[BaseModel]] = InterpretationOutput,
        max_retries: Optional[int] = None,
    ) -> dict:
        """
        Executes structured JSON generation with Pydantic schema enforcement,
        bounded retries on empty/malformed/schema failures, and full telemetry.
        """
        schema = json_schema
        if schema is None and response_model is not None and hasattr(response_model, "strict_json_schema"):
            schema = response_model.strict_json_schema()

        return self.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            json_schema=schema,
            response_model=response_model,
            max_retries=max_retries,
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        json_schema: Optional[dict] = None,
        response_model: Optional[Type[BaseModel]] = None,
        max_retries: Optional[int] = None,
    ) -> dict:
        """
        Calls the configured provider and guarantees a parsed dictionary.
        Does NOT silently fall back to mock if a real provider fails.
        """
        if self.provider not in ("ollama", "groq", "gemini", "google", "mock"):
            err = LLMInferenceError(
                f"Unsupported LLM provider: '{self.provider}'",
                execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="provider_error",
            )
            self.last_error = str(err)
            raise err

        retries_allowed = (
            _get_int_env("OLLAMA_MAX_RETRIES", OLLAMA_MAX_RETRIES, 0)
            if max_retries is None
            else max(0, max_retries)
        )
        # Mock doesn't need multi-attempt retries
        if self.provider == "mock":
            retries_allowed = 0

        start_t = time.perf_counter()
        active_prompt = prompt
        first_err: Optional[LLMInferenceError] = None
        last_err: Optional[LLMInferenceError] = None
        num_predict_override: Optional[int] = None

        for attempt in range(retries_allowed + 1):
            attempt_start = time.perf_counter()
            meta: Dict[str, Any] = {
                "model": self.model_name,
                "provider": self.provider,
                "attempt": attempt + 1,
                "retry_count": attempt,
                "finish_reason": None,
                "response_length": None,
                "thinking_length": 0,
                "recovered": False,
                "error_category": None,
                "execution_status": ModelExecutionStatus.SUCCESS.value,
            }

            try:
                if self.provider == "ollama":
                    raw_text, ollama_meta = self._call_ollama_raw(
                        active_prompt,
                        system_prompt,
                        json_schema=json_schema,
                        num_predict_override=num_predict_override,
                    )
                    meta.update(ollama_meta)
                    parsed, used_recovery = parse_and_validate_json(
                        raw_text,
                        model_name=self.model_name,
                        finish_reason=ollama_meta.get("finish_reason"),
                        thinking_length=ollama_meta.get("thinking_length", 0),
                        response_model=response_model,
                    )
                    meta["recovered"] = used_recovery
                elif self.provider == "groq":
                    raw_text, groq_meta = self._call_groq_raw(active_prompt, system_prompt)
                    meta.update(groq_meta)
                    parsed, used_recovery = parse_and_validate_json(
                        raw_text,
                        model_name=self.model_name,
                        finish_reason=groq_meta.get("finish_reason"),
                        response_model=response_model,
                    )
                    meta["recovered"] = used_recovery
                elif self.provider in ("gemini", "google"):
                    raw_text, gemini_meta = self._call_gemini_raw(active_prompt, system_prompt)
                    meta.update(gemini_meta)
                    parsed, used_recovery = parse_and_validate_json(
                        raw_text,
                        model_name=self.model_name,
                        finish_reason=gemini_meta.get("finish_reason"),
                        response_model=response_model,
                    )
                    meta["recovered"] = used_recovery
                else:  # mock
                    parsed = self._generate_mock(active_prompt, system_prompt)
                    if response_model is not None:
                        parsed = response_model(**parsed).model_dump()
                    meta["finish_reason"] = "stop"
                    meta["response_length"] = len(json.dumps(parsed))

                total_dt = round(time.perf_counter() - start_t, 3)
                meta["duration_sec"] = total_dt
                meta["retry_count"] = attempt
                self.last_duration_sec = total_dt
                self.last_error = None
                self.last_call_metadata = meta

                if self.provider != "mock":
                    logger.info(
                        f"[LLM] provider={self.provider} model={self.model_name} "
                        f"duration={total_dt}s finish_reason={meta.get('finish_reason')} "
                        f"resp_len={meta.get('response_length')} retries={attempt}"
                    )
                return parsed

            except LLMInferenceError as e:
                attempt_dt = round(time.perf_counter() - attempt_start, 3)
                total_dt = round(time.perf_counter() - start_t, 3)
                if first_err is None:
                    first_err = e
                last_err = e

                retryable = e.error_category in (
                    "empty_output",
                    "truncated_output",
                    "malformed_json",
                    "schema_validation_failure",
                )

                logger.warning(
                    f"[LLM] Failure attempt={attempt + 1}/{retries_allowed + 1} "
                    f"provider={self.provider} model={self.model_name} "
                    f"category={e.error_category} status={e.execution_status} "
                    f"finish_reason={e.finish_reason} resp_len={e.response_length} "
                    f"think_len={e.thinking_length} dt={attempt_dt}s | {e}"
                )

                if retryable and attempt < retries_allowed:
                    active_prompt = _build_corrective_prompt(prompt, e)
                    if e.error_category in ("truncated_output", "empty_output"):
                        current_budget = num_predict_override or _get_int_env("OLLAMA_NUM_PREDICT", OLLAMA_NUM_PREDICT, 256)
                        num_predict_override = min(2304, current_budget + 512)
                    continue

                # Exhausted retries or non-retryable failure -> preserve original error context
                final_err = LLMInferenceError(
                    str(e) if attempt == 0 else f"{e} (after {attempt} retries; initial error: {first_err})",
                    execution_status=e.execution_status,
                    error_category=e.error_category,
                    retry_count=attempt,
                    finish_reason=e.finish_reason,
                    response_length=e.response_length,
                    thinking_length=e.thinking_length,
                    duration_sec=total_dt,
                    original_error=first_err.original_error if first_err else e.original_error,
                )
                self.last_duration_sec = total_dt
                self.last_error = str(final_err)
                self.last_call_metadata = {
                    **meta,
                    "duration_sec": total_dt,
                    "retry_count": attempt,
                    "execution_status": final_err.execution_status,
                    "error_category": final_err.error_category,
                    "finish_reason": final_err.finish_reason,
                    "response_length": final_err.response_length,
                    "thinking_length": final_err.thinking_length,
                    "error_message": str(final_err),
                    "original_error": final_err.original_error,
                }
                raise final_err

            except Exception as e:
                total_dt = round(time.perf_counter() - start_t, 3)
                wrapped = LLMInferenceError(
                    f"Unexpected inference failure for '{self.model_name}': {e}",
                    execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                    error_category="provider_error",
                    retry_count=attempt,
                    duration_sec=total_dt,
                    original_error=str(e),
                )
                self.last_duration_sec = total_dt
                self.last_error = str(wrapped)
                self.last_call_metadata = {
                    **meta,
                    "duration_sec": total_dt,
                    "retry_count": attempt,
                    "execution_status": wrapped.execution_status,
                    "error_category": wrapped.error_category,
                    "error_message": str(wrapped),
                }
                logger.error(f"Inference failure for provider '{self.provider}' ({self.model_name}): {wrapped}")
                raise wrapped

        # Defensive fallback (unreachable)
        assert last_err is not None
        raise last_err

    def _call_groq_raw(self, prompt: str, system_prompt: str) -> Tuple[str, Dict[str, Any]]:
        if not self._groq_client:
            raise LLMInferenceError(
                "Groq provider selected but GROQ_API_KEY is not configured in backend environment.",
                execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="provider_error",
            )
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = self._groq_client.chat.completions.create(
                model=self.model_name if "llama" in self.model_name else "llama-3.3-70b-versatile",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = resp.choices[0].message.content or ""
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            return raw, {
                "finish_reason": finish_reason,
                "response_length": len(raw),
                "thinking_length": 0,
            }
        except LLMInferenceError:
            raise
        except Exception as e:
            err_lower = str(e).lower()
            is_timeout = "timeout" in err_lower or "timed out" in err_lower
            raise LLMInferenceError(
                f"Groq API call failed: {e}",
                execution_status=ModelExecutionStatus.TIMEOUT.value if is_timeout else ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="timeout" if is_timeout else "provider_error",
                original_error=str(e),
            )

    def _call_gemini_raw(self, prompt: str, system_prompt: str) -> Tuple[str, Dict[str, Any]]:
        if not self._gemini_client:
            raise LLMInferenceError(
                "Gemini provider selected but GEMINI_API_KEY is not configured in backend environment.",
                execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="provider_error",
            )
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            resp = self._gemini_client.generate_content(
                full_prompt,
                generation_config={"temperature": 0.3, "response_mime_type": "application/json"},
            )
            raw = getattr(resp, "text", None) or ""
            return raw, {
                "finish_reason": "stop",
                "response_length": len(raw),
                "thinking_length": 0,
            }
        except LLMInferenceError:
            raise
        except Exception as e:
            err_lower = str(e).lower()
            is_timeout = "timeout" in err_lower or "timed out" in err_lower
            raise LLMInferenceError(
                f"Gemini API call failed: {e}",
                execution_status=ModelExecutionStatus.TIMEOUT.value if is_timeout else ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="timeout" if is_timeout else "provider_error",
                original_error=str(e),
            )

    def _call_ollama_raw(
        self,
        prompt: str,
        system_prompt: str,
        *,
        json_schema: Optional[dict] = None,
        num_predict_override: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
        connect_timeout = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", str(OLLAMA_CONNECT_TIMEOUT)))
        read_timeout = float(os.getenv("OLLAMA_TIMEOUT", str(OLLAMA_TIMEOUT)))
        num_predict = num_predict_override or _get_int_env("OLLAMA_NUM_PREDICT", OLLAMA_NUM_PREDICT, 256)
        think_enabled = _get_bool_env("OLLAMA_THINK", OLLAMA_THINK)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "format": json_schema if json_schema is not None else "json",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": num_predict,
            },
            "keep_alive": "5m",  # Retain model in VRAM during sequential archetype passes
        }
        if not self._ollama_disable_think_param and not think_enabled:
            # Prevents thinking models (e.g. Qwen 3.5) from exhausting tokens inside `thinking` with empty `response`.
            payload["think"] = False

        timeout_tuple = (connect_timeout, read_timeout)

        def _post_ollama(body: Dict[str, Any]):
            try:
                return requests.post(f"{url}/api/generate", json=body, timeout=timeout_tuple)
            except requests.exceptions.ConnectTimeout:
                raise LLMInferenceError(
                    f"Ollama connection timed out after {connect_timeout}s at {url}.",
                    execution_status=ModelExecutionStatus.TIMEOUT.value,
                    error_category="timeout",
                )
            except requests.exceptions.ReadTimeout:
                raise LLMInferenceError(
                    f"Ollama inference timed out after {read_timeout}s at {url} for model '{self.model_name}'.",
                    execution_status=ModelExecutionStatus.TIMEOUT.value,
                    error_category="timeout",
                )
            except requests.exceptions.ConnectionError:
                raise LLMInferenceError(
                    "Ollama is not available. Install Ollama and start its local service before running local inference.",
                    execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                    error_category="provider_error",
                )
            except requests.exceptions.Timeout:
                raise LLMInferenceError(
                    f"Ollama inference timed out after {read_timeout}s at {url} for model '{self.model_name}'.",
                    execution_status=ModelExecutionStatus.TIMEOUT.value,
                    error_category="timeout",
                )
            except Exception as e:
                raise LLMInferenceError(
                    f"Ollama connection error: {str(e)}",
                    execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                    error_category="provider_error",
                    original_error=str(e),
                )

        resp = _post_ollama(payload)

        # Compatibility fallback: if an Ollama version or model rejects `"think": False`, retry cleanly without it
        if resp.status_code == 400 and "think" in payload and "think" in (resp.text or "").lower():
            logger.info(f"[Ollama] Model '{self.model_name}' does not accept 'think' parameter; retrying without it.")
            self._ollama_disable_think_param = True
            payload.pop("think", None)
            resp = _post_ollama(payload)

        if resp.status_code == 200:
            try:
                body = resp.json()
            except Exception as je:
                raise LLMInferenceError(
                    f"Ollama returned non-JSON HTTP envelope from '{self.model_name}': {je}",
                    execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                    error_category="provider_error",
                    original_error=str(je),
                )

            raw_response = body.get("response", "")
            raw_thinking = body.get("thinking", "") or ""
            finish_reason = body.get("done_reason") or ("stop" if body.get("done") else None)
            if not isinstance(raw_response, str):
                raw_response = json.dumps(raw_response)

            return raw_response, {
                "finish_reason": finish_reason,
                "response_length": len(raw_response),
                "thinking_length": len(raw_thinking),
                "eval_count": body.get("eval_count"),
                "prompt_eval_count": body.get("prompt_eval_count"),
            }

        if resp.status_code == 404:
            raise LLMInferenceError(
                f"Ollama model '{self.model_name}' is not installed. "
                f"Run 'ollama pull {self.model_name}' before running local inference.",
                execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
                error_category="provider_error",
            )

        raise LLMInferenceError(
            f"Ollama returned HTTP error {resp.status_code}: {resp.text[:300]}",
            execution_status=ModelExecutionStatus.PROVIDER_ERROR.value,
            error_category="provider_error",
        )

    def _generate_mock(self, prompt: str, system_prompt: str) -> dict:
        """
        Deterministic, signal-aware, archetype-aligned mock generator for development,
        testing, and zero-dependency execution ONLY.
        """
        sys_lower = system_prompt.lower()
        if "skeptic" in sys_lower or "debunker" in sys_lower:
            archetype = Archetype.SKEPTIC_DEBUNKER
        elif "pattern" in sys_lower or "seeker" in sys_lower:
            archetype = Archetype.PATTERN_SEEKER
        elif "doomer" in sys_lower or "fragil" in sys_lower or "collapse" in sys_lower:
            archetype = Archetype.DOOMER
        elif "amplifier" in sys_lower or "influencer" in sys_lower:
            archetype = Archetype.AMPLIFIER_INFLUENCER
        elif "institutional" in sys_lower or "diploma" in sys_lower:
            archetype = Archetype.INSTITUTIONAL_ANALYST
        else:
            prompt_lower = prompt.lower()
            if "skeptic" in prompt_lower or "debunker" in prompt_lower:
                archetype = Archetype.SKEPTIC_DEBUNKER
            elif "pattern" in prompt_lower or "seeker" in prompt_lower:
                archetype = Archetype.PATTERN_SEEKER
            elif "doomer" in prompt_lower:
                archetype = Archetype.DOOMER
            elif "amplifier" in prompt_lower:
                archetype = Archetype.AMPLIFIER_INFLUENCER
            else:
                archetype = Archetype.INSTITUTIONAL_ANALYST

        # Extract the actual initial signal from prompt if present
        signal_text = prompt.strip()
        src_match = re.search(
            r"SOURCE MATERIAL:\s*([\s\S]*?)(?:\n\s*SOURCE REFERENCE URL:|\n\s*PRIOR ITERATION|\n\s*SCIENTIFIC|$)",
            prompt,
        )
        if src_match and src_match.group(1).strip():
            signal_text = re.sub(r"\s+", " ", src_match.group(1)).strip()

        first_sentence = re.split(r"(?<=[.!?])\s+", signal_text, maxsplit=1)[0].strip()
        if len(first_sentence) > 170:
            first_sentence = first_sentence[:167].rstrip() + "..."
        event_focus = first_sentence or "the reported initial event"

        from src.core.event_grounding import classify_event_domain, extract_event_entities

        dom_ctx = classify_event_domain(signal_text)
        dom_key = dom_ctx.get("domain", "general")
        ent_info = extract_event_entities(signal_text)
        subj_lbl = ent_info.get("subject_label") or "focal organization"

        if dom_key == "trade_supply":
            domain = "trade_supply_chain"
            actor_primary = "regulators, logistics operators, and importers"
            downstream_sector = "manufacturing inventories, freight costs, and downstream retail pricing"
            policy_channel = "trade exemptions, customs guidance, and emergency routing"
            audit_action = "inventory audits"
            buffer_action = "Deploy existing buffer stock or contingency agreements"
            hesitation_action = "Informed buyers pause panic hedging after verifying baseline data"
            shift_behavior = "Shift from just-in-time efficiency toward cautious buffer hoarding"
            dependency_target = "indirect tier-2 vendor dependencies"
            invalidation_hard = f"Confirmed physical halt or hard binding bottleneck across {downstream_sector}"
            backstop_label = f"Credible supply backstop plus de-escalatory execution via {policy_channel}"
            tail_mechanism = f"Fear-driven defensive withdrawals trigger supply hoarding, overwhelming buffers and stressing {downstream_sector}"
        elif dom_key == "corporate_governance":
            domain = "corporate_governance"
            actor_primary = f"the board oversight committee, executive leadership ({subj_lbl}), and institutional shareholders"
            downstream_sector = "governance oversight, leadership continuity, enterprise customer confidence, and national-security compliance"
            policy_channel = "independent board disclosures, SEC governance filings, and external counsel oversight"
            audit_action = "forensic governance and internal security verification audits"
            buffer_action = "Issue formal governance timeline and interim leadership continuity safeguards"
            hesitation_action = "Enterprise customers and institutional investors await verified board findings before reallocating partnerships"
            shift_behavior = "Heightened due diligence across key-person reliance and customer concentration agreements"
            dependency_target = "executive continuity, cloud/enterprise compute contracts, and national-security compliance controls"
            invalidation_hard = f"Verified evidence of compromised intellectual property or forced executive transition across {downstream_sector}"
            backstop_label = f"Independent inquiry findings confirming no criminal wrongdoing or operational breach via {policy_channel}"
            tail_mechanism = f"Substantiated security breach or leaked contradictory findings erode partner trust and disrupt {downstream_sector}"
        elif dom_key == "geopolitics_security":
            domain = "geopolitics_security"
            actor_primary = f"counterintelligence oversight authorities, diplomatic ministries, and affected leadership ({subj_lbl})"
            downstream_sector = "national-security compliance, cross-border commercial partnerships, and technology oversight"
            policy_channel = "classified security briefings, CFIUS/regulatory compliance reviews, and official disclosures"
            audit_action = "counterintelligence provenance and access-control audits"
            buffer_action = "Enforce compartmentalized security controls and formal disclosure protocols"
            hesitation_action = "Institutional partners verify primary intelligence attribution before altering bilateral commitments"
            shift_behavior = "Expanded vetting of cross-border executive travel and sensitive technology partnerships"
            dependency_target = "sensitive technology supply agreements, government clearances, and allied regulatory posture"
            invalidation_hard = f"Confirmed foreign-intelligence compromise or binding regulatory sanction impacting {downstream_sector}"
            backstop_label = f"Official clearance confirming intact security perimeter via {policy_channel}"
            tail_mechanism = f"Confirmed intelligence exploitation triggers reciprocal security restrictions across {downstream_sector}"
        elif dom_key == "ai_software":
            domain = "ai_software_ecosystem"
            actor_primary = "model developers, enterprise IT buyers, cloud/GPU providers, and governance teams"
            downstream_sector = "AI infrastructure capex, software engineering workflows, and inference deployment"
            policy_channel = "enterprise security audits, model evaluation standards, and acceptable-use governance"
            audit_action = "reproducible cost, capability, and safety benchmarks"
            buffer_action = "Enforce human-in-the-loop verification gates and sandboxed staging"
            hesitation_action = "Engineering and finance teams validate independent benchmarks before shifting workloads"
            shift_behavior = "Rapid developer experimentation alongside cost-driven architectural re-evaluation"
            dependency_target = "cloud compute commitments, specialized hardware capacity, and model reliability validation"
            invalidation_hard = f"Verified critical failure, unresolved security exploit, or sustained audit failure across {downstream_sector}"
            backstop_label = f"Transparent reproducible audits paired with staged enterprise verification via {policy_channel}"
            tail_mechanism = f"Hasty unvetted rollout or mispriced capex commitments outpace security auditing across {downstream_sector}"
        elif dom_key == "public_health":
            domain = "public_health"
            actor_primary = "health ministries, clinicians, hospital networks, and research agencies"
            downstream_sector = "clinical surge capacity, public vaccination uptake, and community guidance"
            policy_channel = "epidemiological advisories, clinical trials, and transparent public health briefings"
            audit_action = "clinical surveillance audits"
            buffer_action = "Activate regional hospital triage capacity and evidence briefings"
            hesitation_action = "Clinicians and citizens rely on verified epidemiological data rather than rumor"
            shift_behavior = "Adoption of preventive health precautions alongside scrutiny of treatment guidance"
            dependency_target = "regional diagnostic capacity and primary care bottlenecks"
            invalidation_hard = f"Verified surge in severe clinical caseload exceeding {downstream_sector}"
            backstop_label = f"Transparent clinical data plus coordinated public response via {policy_channel}"
            tail_mechanism = f"Erosion of public health trust coupled with clinical strain triggers compounding overload across {downstream_sector}"
        elif dom_key == "finance_macro":
            domain = "macro_finance"
            actor_primary = "central banks, institutional desks, and corporate treasurers"
            downstream_sector = "borrowing costs, credit spreads, and household consumption"
            policy_channel = "liquidity facilities, forward guidance, and prudential stress testing"
            audit_action = "balance-sheet stress audits"
            buffer_action = "Deploy standing central bank liquidity lines and capital buffers"
            hesitation_action = "Institutional desks verify primary reserve data before repricing"
            shift_behavior = "Defensive portfolio rebalancing toward higher-quality collateral"
            dependency_target = "interbank funding and counterparty duration mismatches"
            invalidation_hard = f"Confirmed solvency impairment or persistent clearing freeze across {downstream_sector}"
            backstop_label = f"Credible liquidity backstop plus de-escalatory execution via {policy_channel}"
            tail_mechanism = f"Fear-driven counterparty withdrawals tighten credit conditions, overwhelming buffers across {downstream_sector}"
        elif dom_key == "cyber_infra":
            domain = "cyber_infrastructure"
            actor_primary = "critical-infrastructure operators, CERT teams, and enterprise CIOs"
            downstream_sector = "service continuity, failover resilience, vendor reviews, and settlement recovery"
            policy_channel = "mandatory incident disclosures, failover audits, and emergency patching directives"
            audit_action = "telemetry and failover incident audits"
            buffer_action = "Activate multi-region failover routing and redundant clearing nodes"
            hesitation_action = "Engineering teams isolate root-cause misconfigurations without widespread panic"
            shift_behavior = "Migration toward multi-cloud redundancy and zero-trust vendor segmentation"
            dependency_target = "shared single-provider routing and identity infrastructure"
            invalidation_hard = f"Confirmed multi-day systemic outage or irreversible data corruption across {downstream_sector}"
            backstop_label = f"Rapid patch verification plus redundant failover execution via {policy_channel}"
            tail_mechanism = f"Cascading authentication or routing failures overwhelm backup nodes and disrupt {downstream_sector}"
        elif dom_key == "geopolitics_energy":
            domain = "geopolitics_energy"
            actor_primary = "diplomatic ministries, commodity traders, and energy distributors"
            downstream_sector = "benchmark energy prices, reserve positioning, and coalition alignment"
            policy_channel = "bilateral de-escalation hotlines and strategic reserve coordination"
            audit_action = "reserve and diplomatic posture reviews"
            buffer_action = "Coordinate strategic reserve releases and diplomatic backchannels"
            hesitation_action = "Market desks discount headline noise pending verified treaty or pipeline data"
            shift_behavior = "Strategic diversification of energy corridors and diplomatic hedging"
            dependency_target = "cross-border transit chokepoints and allied security commitments"
            invalidation_hard = f"Confirmed physical supply interruption or breakdown of talks affecting {downstream_sector}"
            backstop_label = f"Bilateral de-escalation commitments plus coordinated stabilization via {policy_channel}"
            tail_mechanism = f"Tit-for-tat escalation fractures stabilization channels and triggers acute dislocation in {downstream_sector}"
        else:
            domain = "public_policy_market"
            actor_primary = "institutional authorities, industry operators, and engaged communities"
            downstream_sector = "public trust, operational compliance, and resource allocation"
            policy_channel = "official clarifications, supervisory reviews, and coordinated standards"
            audit_action = "primary fact and operational impact reviews"
            buffer_action = "Publish verified baseline data and standard operating guidance"
            hesitation_action = "Observers wait for primary evidence before committing major changes"
            shift_behavior = "Cautious adaptation toward resilient procedures and clearer accountability"
            dependency_target = "downstream institutional and community coordination links"
            invalidation_hard = f"Confirmed sustained breakdown across {downstream_sector}"
            backstop_label = f"Transparent oversight plus timely coordination via {policy_channel}"
            tail_mechanism = f"Unchecked feedback between public pressure and coordination lags stresses {downstream_sector}"

        # Subtle feedback round sensitivity (without inventing new fake facts)
        round_match = re.search(r"MACRO FEEDBACK FROM ROUND (\d+)", prompt)
        round_num = int(round_match.group(1)) + 1 if round_match else 1
        round_stance_dampen = -0.04 * max(0, round_num - 1) if round_num > 1 else 0.0

        archetype_profiles = {
            Archetype.INSTITUTIONAL_ANALYST: {
                "stance": 0.15,
                "virality_estimate": 0.25,
                "perceived_credibility": 0.85,
                "emotional_charge": 0.20,
                "risk_severity": 0.25,
                "evidence_status": "supported",
                "supporting_evidence": [
                    f"Direct event premise documented in signal: \"{first_sentence}\""
                ],
                "uncertainty": 0.20,
                "claims": [
                    f"Primary trigger involves {actor_primary}",
                    f"Official response likely centers on {policy_channel}",
                ],
                "analytical_question": (
                    f"How will governance boards, regulators, and institutional buyers adapt {policy_channel} following this event?"
                ),
                "stakeholder_focus": f"Governance bodies, compliance officers, and institutional procurement teams in {downstream_sector}",
                "primary_narrative": (
                    f"Verified details in '{event_focus}' point to measured institutional adjustment via {policy_channel} "
                    f"while monitoring spillover into {downstream_sector}."
                ),
                "key_arguments": [
                    f"Institutional actors typically rely on {policy_channel} before structural shifts lock in.",
                    f"Secondary effects depend on whether {actor_primary} coordinate cleanly.",
                ],
                "narrative_tags": [domain, "institutional_response", "measured_containment"],
                "causal_mechanism": (
                    f"Initial event triggers compliance reviews by {actor_primary}, channeling adjustment through {policy_channel} "
                    f"and moderating abrupt disruption in {downstream_sector}."
                ),
                "strongest_supporting_consideration": (
                    f"Established institutional procurement cycles favor formalized review via {policy_channel} prior to structural pivots."
                ),
                "strongest_opposing_consideration": (
                    f"Organizational inertia, legacy integration contracts, and vendor lock-in can delay {policy_channel} enforcement."
                ),
                "expected_stakeholder_actions": [
                    f"Issue formal clarifications and document {policy_channel}",
                    f"Audit near-term exposure across {downstream_sector}",
                ],
                "behavioural_changes": [
                    "Institutions adopt cautious wait-and-see procurement",
                    "Analysts separate confirmed announcements from rumor",
                ],
                "emergent_consequences": [
                    f"1st-order: Structured compliance review across {downstream_sector} (Hypothesis)",
                    f"2nd-order: Temporary policy uncertainty followed by codified guidance (Hypothesis)",
                ],
                "time_horizon": "medium_term_months",
                "trigger_conditions": [
                    f"Binding regulatory or ministerial follow-up altering {policy_channel}"
                ],
                "invalidation_conditions": [
                    "Formal institutional waiver, bilateral postponement, or non-adoption notice"
                ],
            },
            Archetype.SKEPTIC_DEBUNKER: {
                "stance": -0.60,
                "virality_estimate": 0.35,
                "perceived_credibility": 0.78,
                "emotional_charge": 0.15,
                "risk_severity": 0.18,
                "evidence_status": "inferred",
                "supporting_evidence": [
                    f"Source anchors only what is explicitly stated: \"{first_sentence}\""
                ],
                "uncertainty": 0.28,
                "claims": [
                    "Immediate systemic crisis claims go beyond documented primary evidence",
                    f"Existing operational slack in {downstream_sector} moderates near-term extremes",
                ],
                "analytical_question": (
                    f"Which empirical base-rate constraints, verification gaps, or operational buffers limit extreme claims around {downstream_sector}?"
                ),
                "stakeholder_focus": f"Independent auditors, cost-benchmark engineers, and risk-averse operators across {downstream_sector}",
                "primary_narrative": (
                    f"While '{event_focus}' warrants monitoring, viral escalation claims exceed verified primary data and "
                    f"overlook stabilizing friction moderating near-term change in {downstream_sector}."
                ),
                "key_arguments": [
                    "Base rates show most initial announcements undergo exemptions, revisions, or gradual rollout.",
                    f"No primary evidence confirms uncontrolled breakdown across {downstream_sector}.",
                ],
                "narrative_tags": [domain, "base_rate_check", "dampening_buffers"],
                "causal_mechanism": (
                    f"Early alarm triggers {audit_action} and clarifying statements by {actor_primary}, "
                    "causing speculative projections to mean-revert as practical constraints moderate adoption or disruption."
                ),
                "strongest_supporting_consideration": (
                    f"Historical rollouts show full enterprise switching costs and {audit_action} delay headline disruptions."
                ),
                "strongest_opposing_consideration": (
                    f"Genuine performance or cost breakthroughs verified by {audit_action} could overcome incumbent switching inertia."
                ),
                "expected_stakeholder_actions": [
                    "Publish primary benchmarks or empirical metrics testing early claims",
                    buffer_action,
                ],
                "behavioural_changes": [
                    hesitation_action,
                    "Commentary shifts toward corrective updates and nuance",
                ],
                "emergent_consequences": [
                    "1st-order: Short-lived attention spike followed by measured normalization (Hypothesis)",
                    f"2nd-order: Targeted incremental adjustments across {downstream_sector} rather than abrupt displacement (Hypothesis)",
                ],
                "time_horizon": "short_term_weeks",
                "trigger_conditions": [
                    "Audited independent benchmarks confirming sustained multi-week shift"
                ],
                "invalidation_conditions": [
                    invalidation_hard
                ],
            },
            Archetype.PATTERN_SEEKER: {
                "stance": 0.65,
                "virality_estimate": 0.75,
                "perceived_credibility": 0.65,
                "emotional_charge": 0.68,
                "risk_severity": 0.35,
                "evidence_status": "inferred",
                "supporting_evidence": [
                    f"Observable focal point in source signal: \"{first_sentence}\""
                ],
                "uncertainty": 0.42,
                "claims": [
                    f"Indirect linkages connect {actor_primary} decisions to {downstream_sector}",
                    "Second-order repositioning across adjacent cohorts amplifies modest initial shifts",
                ],
                "analytical_question": (
                    f"What indirect 2nd- and 3rd-order incentive realignments and substitution patterns emerge across {downstream_sector}?"
                ),
                "stakeholder_focus": f"Ecosystem challengers, application builders, and capital allocators re-pricing {dependency_target}",
                "primary_narrative": (
                    f"Connecting '{event_focus}' across adjacent systems suggests strategic adaptation by {actor_primary} "
                    f"will transmit second-order pressure into {downstream_sector}."
                ),
                "key_arguments": [
                    f"Actors pre-emptively adjust strategies when {policy_channel} outlook remains fluid.",
                    f"Cross-network dependencies turn localized changes in {downstream_sector} into broader structural realignment.",
                ],
                "narrative_tags": [domain, "second_order_spillover", "structural_realignment"],
                "causal_mechanism": (
                    f"Shifting incentives for {actor_primary} drive synchronized repositioning across {dependency_target}, "
                    f"propagating second-order adaptation pressures into {downstream_sector}."
                ),
                "strongest_supporting_consideration": (
                    f"Comparable technological shifts triggered multi-quarter cost curves and downstream restructuring across {downstream_sector}."
                ),
                "strongest_opposing_consideration": (
                    f"High migration complexity across {dependency_target} may slow ecosystem substitution to multi-quarter horizons."
                ),
                "expected_stakeholder_actions": [
                    f"Accelerate competitive adaptation and workflow redesign across {downstream_sector}",
                    f"Audit and diversify {dependency_target}",
                ],
                "behavioural_changes": [
                    shift_behavior,
                    f"Elevated scrutiny of {dependency_target}",
                ],
                "emergent_consequences": [
                    f"2nd-order: Cost, workflow, and competitive realignment across {downstream_sector} (Hypothesis)",
                    "3rd-order: Structural advantage shifts toward adaptable multi-vendor ecosystems (Hypothesis)",
                ],
                "time_horizon": "medium_term_months",
                "trigger_conditions": [
                    f"Peer actors simultaneously adjust standards across {downstream_sector}"
                ],
                "invalidation_conditions": [
                    "Clear institutional consensus or stable equilibrium removing pressure for structural redesign"
                ],
            },
            Archetype.DOOMER: {
                "stance": 0.85,
                "virality_estimate": 0.72,
                "perceived_credibility": 0.55,
                "emotional_charge": 0.90,
                "risk_severity": 0.78,
                "evidence_status": "unsupported",
                "supporting_evidence": [
                    f"Downside tail scenario anchored (as hypothesis) to: \"{first_sentence}\""
                ],
                "uncertainty": 0.65,
                "claims": [
                    f"Hypothesis: Tightly coupled dependencies in {downstream_sector} risk compounding safety or operational failures",
                    "Hypothesis: Slow verification controls could amplify systemic vulnerability or financial strain",
                ],
                "analytical_question": (
                    f"Through which concrete harm mechanisms could lagging safeguards or vulnerability exploitation damage {downstream_sector}?"
                ),
                "stakeholder_focus": f"Safety/security teams, exposed operational systems, and risk officers managing failure modes in {downstream_sector}",
                "primary_narrative": (
                    f"Tail-risk hypothesis: If '{event_focus}' induces rushed deployment without security or governance controls in {downstream_sector}, "
                    f"lagging verification could produce operational failures or acute institutional strain."
                ),
                "key_arguments": [
                    f"Feedback between rushed adoption and {downstream_sector} security gaps creates non-linear failure exposure.",
                    f"Weak or fragmented {policy_channel} increases vulnerability to misuse or mispriced commitments.",
                ],
                "narrative_tags": [domain, "tail_risk_cascade", "systemic_fragility"],
                "causal_mechanism": (
                    f"{tail_mechanism} (downside harm hypothesis)."
                ),
                "strongest_supporting_consideration": (
                    f"Unvetted rapid transition consistently strains verification controls and exposes {dependency_target}."
                ),
                "strongest_opposing_consideration": (
                    f"Enterprise security sandboxing and staged rollouts in {policy_channel} normally arrest failure cascades before systemic damage."
                ),
                "expected_stakeholder_actions": [
                    "Issue security vulnerability advisories and emergency compliance quarantines",
                    "Audit critical workflow dependencies before scaling automated exposure",
                ],
                "behavioural_changes": [
                    f"Risk-averse lockouts following verified safety/reliability failures in {downstream_sector}",
                    "Escalating pressure for mandatory liability and safety testing prior to rollout",
                ],
                "emergent_consequences": [
                    f"2nd-order: Localized operational failures or security incidents from unvetted rollout in {downstream_sector} (Worst-Case Hypothesis)",
                    f"3rd-order: Acute confidence shock and protracted trust impairment across {downstream_sector} (Worst-Case Hypothesis)",
                ],
                "time_horizon": "long_term_quarters",
                "trigger_conditions": [
                    f"Verified safety/security failure coinciding with absent safeguards in {policy_channel}"
                ],
                "invalidation_conditions": [
                    backstop_label
                ],
            },
            Archetype.AMPLIFIER_INFLUENCER: {
                "stance": 0.88,
                "virality_estimate": 0.92,
                "perceived_credibility": 0.48,
                "emotional_charge": 0.92,
                "risk_severity": 0.45,
                "evidence_status": "inferred",
                "supporting_evidence": [
                    f"Public conversation centers around the dramatic framing of: \"{first_sentence}\""
                ],
                "uncertainty": 0.45,
                "claims": [
                    "High-arousal social sharing compresses institutional response windows",
                    f"Retail and media sentiment swings amplify visible volatility across {downstream_sector}",
                ],
                "analytical_question": (
                    f"How do viral narrative salience and media framing widen the gap between simulated public buzz and verified impact in {downstream_sector}?"
                ),
                "stakeholder_focus": "Media commentators, retail/developer social feeds, and momentum-driven public observers",
                "primary_narrative": (
                    f"Viral reaction to '{event_focus}' is accelerating fast — high headline salience and social media amplification "
                    f"pressure {actor_primary} ahead of methodical empirical audits."
                ),
                "key_arguments": [
                    "High emotional urgency drives immediate public engagement, viral experimentation, or vocal debate.",
                    f"Viral narrative pressure compels {actor_primary} to respond publicly ahead of full data.",
                ],
                "narrative_tags": [domain, "viral_amplification", "crowd_psychology"],
                "causal_mechanism": (
                    "Emotionally salient headlines spread rapidly across broadcast and peer networks, inducing synchronized "
                    f"public attention surges that pressure {downstream_sector}."
                ),
                "strongest_supporting_consideration": (
                    "Early social-network momentum frequently triggers short-term headline reactivity and rapid public trial surges."
                ),
                "strongest_opposing_consideration": (
                    "Attention decay and competing news cycles commonly deflate short-term viral hype within days to weeks."
                ),
                "expected_stakeholder_actions": [
                    "Public briefings, rapid executive statements, and high-visibility media commentary",
                    "Media coverage prioritizing dramatic disruption angles over technical base rates",
                ],
                "behavioural_changes": [
                    "Accelerated social reposting, front-loaded public reaction, and polarized debate",
                    "Reduced patience for multi-week technical verification cycles",
                ],
                "emergent_consequences": [
                    "1st-order: Acute short-term simulated attention surge decoupling headlines from fundamentals (Hypothesis)",
                    "2nd-order: Reputational pressure forcing visible public communications or reactive pivots (Hypothesis)",
                ],
                "time_horizon": "immediate_days",
                "trigger_conditions": [
                    "Viral social trend amplification, sharp sentiment swings, or prominent public figure commentary"
                ],
                "invalidation_conditions": [
                    "Competing news cycle displacement paired with uneventful near-term empirical metrics"
                ],
            },
        }

        profile = archetype_profiles[archetype]
        is_model_b = (
            "b" in self.model_name.lower()
            or "gemini" in self.model_name.lower()
            or "gemma" in self.model_name.lower()
        )
        model_mod = 0.05 if is_model_b else -0.04
        stance_val = round(max(-1.0, min(1.0, profile["stance"] + model_mod + round_stance_dampen)), 2)
        virality_val = round(max(0.0, min(1.0, profile["virality_estimate"] + (model_mod / 2.0))), 2)
        cred_val = round(max(0.0, min(1.0, profile["perceived_credibility"])), 2)
        emo_val = round(max(0.0, min(1.0, profile["emotional_charge"])), 2)

        return {
            "evidence_status": profile["evidence_status"],
            "supporting_evidence": profile["supporting_evidence"],
            "uncertainty": profile["uncertainty"],
            "claims": profile["claims"],
            "stance": stance_val,
            "virality_estimate": virality_val,
            "perceived_credibility": cred_val,
            "emotional_charge": emo_val,
            "risk_severity": profile["risk_severity"],
            "virality": virality_val,
            "credibility": cred_val,
            "emotion": emo_val,
            "analytical_question": profile["analytical_question"],
            "stakeholder_focus": profile["stakeholder_focus"],
            "primary_narrative": profile["primary_narrative"],
            "key_arguments": profile["key_arguments"],
            "narrative_tags": profile["narrative_tags"],
            "causal_mechanism": profile["causal_mechanism"],
            "strongest_supporting_consideration": profile["strongest_supporting_consideration"],
            "strongest_opposing_consideration": profile["strongest_opposing_consideration"],
            "expected_stakeholder_actions": profile["expected_stakeholder_actions"],
            "behavioural_changes": profile["behavioural_changes"],
            "emergent_consequences": profile["emergent_consequences"],
            "time_horizon": profile["time_horizon"],
            "trigger_conditions": profile["trigger_conditions"],
            "invalidation_conditions": profile["invalidation_conditions"],
        }


def create_dual_providers(
    provider_a: Optional[str] = None,
    model_a: Optional[str] = None,
    provider_b: Optional[str] = None,
    model_b: Optional[str] = None,
) -> tuple[LLMClient, LLMClient]:
    """Factory creating the two configured LLM clients for Model A and Model B."""
    pa = provider_a or DEFAULT_PROVIDER_A
    ma = model_a or DEFAULT_MODEL_A
    pb = provider_b or DEFAULT_PROVIDER_B
    mb = model_b or DEFAULT_MODEL_B

    client_a = LLMClient(provider=pa, model_name=ma)
    client_b = LLMClient(provider=pb, model_name=mb)
    return client_a, client_b
