"""LLM access: OpenAI-compatible client with structured output, retries, throttling,
caching and cost logging, plus an offline heuristic stand-in for tests and demos."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from .cache import JsonCache, content_key
from .config import Settings

# Strict JSON schemas for native structured output. Every property is required;
# optional values are expressed as nullable types (what strict mode expects).
SCHEMAS: dict[str, dict] = {
    "requirement_list": {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirements"],
        "properties": {
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text", "must_have", "category", "years"],
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "must_have": {"type": "boolean"},
                        "category": {"type": "string",
                                     "enum": ["experience", "skill", "domain", "education", "other"]},
                        "years": {"type": ["number", "null"]},
                    },
                },
            }
        },
    },
    "verdict": {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirement_id", "verdict", "evidence_ids", "evidence_quote", "reason", "gap_type"],
        "properties": {
            "requirement_id": {"type": "string"},
            "verdict": {"type": "string", "enum": ["met", "partial", "missing"]},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_quote": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "gap_type": {"type": ["string", "null"], "enum": ["reword", "real_gap", None]},
        },
    },
}


@dataclass
class UsageTotals:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 0
    cached_calls: int = 0

    def snapshot(self) -> "UsageTotals":
        return UsageTotals(**self.__dict__)

    def since(self, before: "UsageTotals") -> "UsageTotals":
        return UsageTotals(**{k: getattr(self, k) - getattr(before, k) for k in self.__dict__})


@dataclass
class BaseLLM:
    settings: Settings
    model_name: str = "base"
    totals: UsageTotals = field(default_factory=UsageTotals)

    def __post_init__(self) -> None:
        self.cache = JsonCache(self.settings.cache_dir, "llm")
        self.log_path: Path = self.settings.runs_dir / "llm_calls.jsonl"

    def complete_json(self, system: str, user: str, model: Type[BaseModel], schema_name: str,
                      context: dict[str, Any] | None = None) -> BaseModel:
        key = content_key(self.model_name, schema_name, system, user)
        hit = self.cache.get(key)
        if hit is not None:
            try:
                obj = model.model_validate(hit)
                self.totals.cached_calls += 1
                return obj
            except ValidationError:
                pass
        start = time.perf_counter()
        obj, tin, tout = self._complete(system, user, model, schema_name, context or {})
        latency = time.perf_counter() - start
        cost = (tin * self.settings.price_in_per_m + tout * self.settings.price_out_per_m) / 1_000_000
        self.totals.tokens_in += tin
        self.totals.tokens_out += tout
        self.totals.cost_usd += cost
        self.totals.latency_s += latency
        self.totals.calls += 1
        self.cache.set(key, obj.model_dump())
        self._log(schema_name, tin, tout, cost, latency)
        return obj

    def _complete(self, system: str, user: str, model: Type[BaseModel], schema_name: str,
                  context: dict[str, Any]) -> tuple[BaseModel, int, int]:
        raise NotImplementedError

    def _log(self, schema_name: str, tin: int, tout: int, cost: float, latency: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model_name, "task": schema_name,
            "tokens_in": tin, "tokens_out": tout,
            "cost_usd": round(cost, 6), "latency_s": round(latency, 3),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


class OpenAICompatLLM(BaseLLM):
    """Any OpenAI-compatible chat endpoint (Gemini, Groq, OpenAI, OpenRouter, local servers)."""

    RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, settings: Settings):
        if not (settings.llm_base_url and settings.llm_api_key and settings.llm_model):
            raise ValueError("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (see .env.example), "
                             "or run with --offline.")
        super().__init__(settings=settings, model_name=settings.llm_model)
        from openai import OpenAI

        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                             max_retries=0, timeout=60)
        # Downgrades if the provider rejects native schemas: json_schema -> json_object -> prompt.
        self.schema_mode = "json_schema"
        self._last_call = 0.0

    # --- transport ---

    def _throttle(self) -> None:
        rpm = self.settings.llm_max_rpm
        if rpm > 0:
            wait = 60.0 / rpm - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def _response_format(self, schema_name: str) -> dict | None:
        if self.schema_mode == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": SCHEMAS[schema_name]}}
        if self.schema_mode == "json_object":
            return {"type": "json_object"}
        return None

    def _create(self, messages: list[dict], schema_name: str):
        import openai

        attempt = 0
        while True:
            self._throttle()
            kwargs: dict[str, Any] = {"model": self.settings.llm_model, "messages": messages,
                                      "temperature": 0}
            fmt = self._response_format(schema_name)
            if fmt:
                kwargs["response_format"] = fmt
            try:
                return self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError as e:
                # Provider does not support this structured-output mode: fall back one level.
                msg = f"{getattr(e, 'message', '')} {e} {getattr(e, 'body', '')}".lower()
                if self.schema_mode != "prompt" and ("response_format" in msg or "schema" in msg
                                                     or "json" in msg):
                    self.schema_mode = "json_object" if self.schema_mode == "json_schema" else "prompt"
                    continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError, openai.APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if isinstance(e, openai.APIStatusError) and status not in self.RETRYABLE_STATUS:
                    raise
                attempt += 1
                if attempt > self.settings.llm_max_retries:
                    raise
                time.sleep(self._backoff(attempt, e))

    @staticmethod
    def _backoff(attempt: int, error: Exception, base: float = 1.0, cap: float = 30.0) -> float:
        """Exponential backoff with full jitter; honours Retry-After when the server sends it."""
        response = getattr(error, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after") if hasattr(response, "headers") else None
            if retry_after:
                try:
                    return min(float(retry_after), cap) + random.uniform(0, 1)
                except ValueError:
                    pass
        return random.uniform(0, min(cap, base * 2 ** attempt))

    # --- structured completion ---

    def _complete(self, system, user, model, schema_name, context):
        sys_prompt = system
        if self.schema_mode != "json_schema":
            sys_prompt += "\n\nJSON schema:\n" + json.dumps(SCHEMAS[schema_name])
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        tin = tout = 0
        for repair in range(2):
            resp = self._create(messages, schema_name)
            usage = getattr(resp, "usage", None)
            tin += getattr(usage, "prompt_tokens", 0) or 0
            tout += getattr(usage, "completion_tokens", 0) or 0
            content = resp.choices[0].message.content or ""
            try:
                return model.model_validate(json.loads(_strip_fences(content))), tin, tout
            except (json.JSONDecodeError, ValidationError) as err:
                if repair == 1:
                    raise RuntimeError(f"Model returned invalid JSON twice for {schema_name}: {err}")
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That output was invalid: {err}. "
                                                "Return only the corrected JSON object."},
                ]
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Offline stand-in. NOT a model: simple rules so the pipeline, tests and demo
# run without an API key. Never report its numbers as the tool's accuracy.
# ---------------------------------------------------------------------------

STOPWORDS = set("""a an and or the of to in on for with at by from as is are be been being this that
these those it its you your we our they their will would should can could may might must
have has had do does did not no yes using use used experience experienced years year plus
strong solid good great excellent ability able knowledge understanding familiarity proven
working work hands-on hands on including include such like e.g etc least minimum preferred
required requirement requirements nice bonus""".split())

MUST_HEADINGS = ("require", "qualification", "must", "what you'll need", "what you need",
                 "you have", "you bring", "skills", "responsibilit")
NICE_HEADINGS = ("prefer", "nice", "bonus", "plus", "desirable", "extra")
SKIP_HEADINGS = ("about", "benefit", "offer", "perks", "salary", "compensation", "culture",
                 "equal", "who we are", "why join", "location")


def content_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9+#./-]*", text.lower())
    return [t.strip(".-/") for t in tokens if t.strip(".-/") and t not in STOPWORDS and len(t) > 1]


class HeuristicLLM(BaseLLM):
    def __init__(self, settings: Settings):
        super().__init__(settings=settings, model_name="offline-heuristic")

    def _complete(self, system, user, model, schema_name, context):
        if schema_name == "requirement_list":
            return self._extract(context["jd_text"]), 0, 0
        if schema_name == "verdict":
            return self._judge(context["requirement"], context["chunks"],
                               context.get("experience_years")), 0, 0
        raise ValueError(schema_name)

    @staticmethod
    def _extract(jd_text: str):
        from .models import Requirement, RequirementList

        reqs, mode, n = [], None, 0
        for raw in jd_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", line)
            if not bullet:
                low = line.lower().lstrip("#").strip()
                if any(h in low for h in SKIP_HEADINGS):
                    mode = "skip"
                elif any(h in low for h in NICE_HEADINGS):
                    mode = "nice"
                elif any(h in low for h in MUST_HEADINGS):
                    mode = "must"
                continue
            if mode in (None, "skip"):
                continue
            text = bullet.group(1).strip()
            years = re.search(r"(\d+)\s*\+?\s*years?", text, re.I)
            low = text.lower()
            if years:
                category = "experience"
            elif re.search(r"\b(degree|bachelor|master|bs|bsc|ms|phd|certif)", low):
                category = "education"
            else:
                category = "skill"
            n += 1
            reqs.append(Requirement(id=f"r{n}", text=text, must_have=(mode == "must"),
                                    category=category,
                                    years=float(years.group(1)) if years else None))
        return RequirementList(requirements=reqs)

    @staticmethod
    def _judge(requirement, chunks, experience_years=None):
        from .models import Verdict

        if requirement.years is not None and experience_years is not None:
            headers = [c for c in chunks if c.kind == "role_header"]
            if headers and experience_years >= requirement.years:
                return Verdict(requirement_id=requirement.id, verdict="met",
                               evidence_ids=[headers[0].id], evidence_quote=headers[0].text[:40],
                               reason=f"Dated roles total {experience_years:g} years.")
        req_tokens = set(content_tokens(requirement.text))
        best, best_overlap = None, 0.0
        for c in chunks:
            overlap = len(req_tokens & set(content_tokens(c.text))) / max(1, len(req_tokens))
            if overlap > best_overlap:
                best, best_overlap = c, overlap
        if best is None or best_overlap < 0.34:
            return Verdict(requirement_id=requirement.id, verdict="missing", evidence_ids=[],
                           reason="No chunk shares the requirement's key terms.", gap_type="real_gap")
        quote = " ".join(best.text.split()[:10])
        if best_overlap >= 0.67 and best.kind != "skills":
            return Verdict(requirement_id=requirement.id, verdict="met", evidence_ids=[best.id],
                           evidence_quote=quote, reason="Chunk uses most of the requirement's terms.")
        return Verdict(requirement_id=requirement.id, verdict="partial", evidence_ids=[best.id],
                       evidence_quote=quote, reason="Chunk covers some of the requirement's terms.",
                       gap_type="reword")


def make_llm(settings: Settings, offline: bool = False) -> BaseLLM:
    return HeuristicLLM(settings) if offline else OpenAICompatLLM(settings)
