import json
from types import SimpleNamespace

import openai

try:  # the HTTP library the installed openai SDK is built on
    import httpx
except ImportError:  # newer SDK releases
    import httpx2 as httpx
import pytest

from matcher import llm_client
from matcher.llm_client import OpenAICompatLLM
from matcher.models import RequirementList

REQ_JSON = json.dumps({"requirements": [
    {"id": "r1", "text": "Kotlin", "must_have": True, "category": "skill", "years": None}]})


def _response(content, tin=100, tout=20):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


def _status_error(cls, status, headers=None):
    req = httpx.Request("POST", "http://example.invalid/v1/chat/completions")
    return cls("error", response=httpx.Response(status, request=req, headers=headers or {}), body=None)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make(settings, outcomes):
    llm = OpenAICompatLLM(settings)
    fake = FakeCompletions(outcomes)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return llm, fake


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: slept.append(s))
    return slept


def test_structured_call_logs_cost_and_caches(settings):
    llm, fake = make(settings, [_response(REQ_JSON)])
    out = llm.complete_json("sys", "user", RequirementList, "requirement_list")
    assert out.requirements[0].text == "Kotlin"
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[0]["temperature"] == 0
    # 100 in * $1/M + 20 out * $2/M
    assert llm.totals.cost_usd == pytest.approx(0.00014)
    log = (settings.runs_dir / "llm_calls.jsonl").read_text().strip().splitlines()
    assert json.loads(log[0])["tokens_in"] == 100
    # second identical call is served from cache, no API call
    llm.complete_json("sys", "user", RequirementList, "requirement_list")
    assert len(fake.calls) == 1 and llm.totals.cached_calls == 1


def test_rate_limit_retries_with_retry_after(settings, no_sleep):
    err = _status_error(openai.RateLimitError, 429, {"retry-after": "2"})
    llm, fake = make(settings, [err, _response(REQ_JSON)])
    llm.complete_json("sys", "u2", RequirementList, "requirement_list")
    assert len(fake.calls) == 2
    assert 2.0 <= no_sleep[-1] <= 3.0


def test_retries_give_up_after_max(settings):
    settings.llm_max_retries = 2
    errs = [_status_error(openai.InternalServerError, 503) for _ in range(3)]
    llm, fake = make(settings, errs)
    with pytest.raises(openai.InternalServerError):
        llm.complete_json("sys", "u3", RequirementList, "requirement_list")
    assert len(fake.calls) == 3


def test_non_retryable_status_raises_immediately(settings):
    llm, fake = make(settings, [_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(openai.AuthenticationError):
        llm.complete_json("sys", "u4", RequirementList, "requirement_list")
    assert len(fake.calls) == 1


def test_falls_back_when_schema_mode_unsupported(settings):
    bad = _status_error(openai.BadRequestError, 400)
    bad.message = "response_format json_schema is not supported"
    llm, fake = make(settings, [bad, _response(REQ_JSON)])
    llm.complete_json("sys", "u5", RequirementList, "requirement_list")
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    assert llm.schema_mode == "json_object"


def test_repairs_invalid_json_once(settings):
    llm, fake = make(settings, [_response("```json\n{\"requirements\": [{\"id\": 1}]}\n```"),
                                _response(REQ_JSON)])
    out = llm.complete_json("sys", "u6", RequirementList, "requirement_list")
    assert out.requirements[0].id == "r1"
    assert "invalid" in fake.calls[1]["messages"][-1]["content"]
    assert llm.totals.tokens_in == 200  # both attempts counted


def test_backoff_is_bounded():
    for attempt in range(1, 10):
        assert 0 <= OpenAICompatLLM._backoff(attempt, Exception()) <= 30


def test_missing_config_raises(settings):
    settings.llm_api_key = ""
    with pytest.raises(ValueError):
        OpenAICompatLLM(settings)
