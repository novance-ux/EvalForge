"""Offline, deterministic *simulated* LLM provider.

This provider exists so that unit tests, the classroom demo and the
experiment scripts run without API keys or a GPU. It is **not** a language
model and its numbers must never be reported as real model results.

How it behaves
--------------
* **Classification prompts** (text inside triple quotes, e.g. ``\"\"\"...\"\"\"``):
  a keyword classifier picks a support-ticket category. The *prompt* changes
  how well it does, mimicking real prompt-engineering effects:

  - explicit "valid JSON only" instruction  -> fewer malformed outputs
  - category definitions                    -> ambiguous tickets resolved better
  - an explicit list of allowed categories  -> no invented labels

* **Extraction prompts** (contain "extract"): regex-based field extraction.
* **Judge prompts** (produced by ``app.scorers.judge``): scores the output
  against the reference answer, with optional noise when ``temperature > 0``
  and a first-position bias in pairwise mode (so order randomisation matters).

Model names change the simulated quality/latency/cost profile:
``mock-small`` (cheap, fast, more mistakes) and ``mock-large`` (the opposite).
Any other name behaves like ``mock-small``.

Latency: a realistic latency (hundreds of ms) is *reported*, but the
provider only sleeps for ``time_scale`` x that value so demos stay fast.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from dataclasses import dataclass

from app.models.results import LLMResponse
from app.providers.base import LLMProvider
from app.utils.misc import estimate_tokens

CATEGORIES = ["billing", "account", "technical", "refund", "shipping", "cancellation"]

# keyword -> (category, weight)
_LEXICON: dict[str, tuple[str, float]] = {
    # billing
    "charged": ("billing", 2), "charge": ("billing", 2), "invoice": ("billing", 2),
    "payment": ("billing", 1.5), "card": ("billing", 1), "bill": ("billing", 2),
    "billing": ("billing", 2), "price": ("billing", 1), "tax": ("billing", 1.5),
    "receipt": ("billing", 1.5), "overcharged": ("billing", 2.5), "plan": ("billing", 0.5),
    "upgrade": ("billing", 1), "coupon": ("billing", 1.5), "discount": ("billing", 1.5),
    # account
    "password": ("account", 2.5), "login": ("account", 2), "log in": ("account", 2),
    "sign in": ("account", 2), "username": ("account", 2), "email address": ("account", 1.5),
    "account": ("account", 1), "locked": ("account", 2), "2fa": ("account", 2),
    "two-factor": ("account", 2), "profile": ("account", 1.5), "verification": ("account", 1),
    "delete my account": ("account", 2),
    # technical
    "crash": ("technical", 2.5), "crashes": ("technical", 2.5), "error": ("technical", 2),
    "bug": ("technical", 2.5), "slow": ("technical", 1.5), "upload": ("technical", 1.5),
    "not loading": ("technical", 2), "freezes": ("technical", 2.5), "app": ("technical", 0.5),
    "api": ("technical", 2), "500": ("technical", 1.5), "sync": ("technical", 1.5),
    "notifications": ("technical", 1.5), "blank screen": ("technical", 2.5),
    "export": ("technical", 1),
    # refund
    "refund": ("refund", 3), "money back": ("refund", 3), "reimburse": ("refund", 2.5),
    "return": ("refund", 1.5), "defective": ("refund", 1.5), "broken": ("refund", 1),
    "damaged": ("refund", 1.5),
    # shipping
    "package": ("shipping", 2.5), "delivery": ("shipping", 2.5), "shipped": ("shipping", 2),
    "tracking": ("shipping", 2.5), "courier": ("shipping", 2), "arrived": ("shipping", 1.5),
    "address": ("shipping", 1), "shipping": ("shipping", 2.5), "order": ("shipping", 0.5),
    "parcel": ("shipping", 2.5), "late": ("shipping", 1),
    # cancellation
    "cancel": ("cancellation", 2.5), "cancellation": ("cancellation", 3),
    "unsubscribe": ("cancellation", 2.5), "stop my subscription": ("cancellation", 3),
    "end my subscription": ("cancellation", 3), "terminate": ("cancellation", 2),
    "auto-renew": ("cancellation", 1.5), "close my": ("cancellation", 1),
    "subscription": ("cancellation", 1),
}

# What an unconstrained model tends to call each category ("invented" labels).
_SYNONYMS = {
    "billing": "payments", "account": "login_issue", "technical": "bug_report",
    "refund": "refund", "shipping": "delivery", "cancellation": "cancel_subscription",
}

# Rules a model can apply only when the prompt defines the categories.
_DEFINITION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"refund|money back|reimburse", re.I), "refund"),
    (re.compile(r"\bcancel|unsubscribe|end my subscription|stop my subscription", re.I), "cancellation"),
    (re.compile(r"password|log ?in|sign in|locked out|2fa|two-factor", re.I), "account"),
    (re.compile(r"package|parcel|tracking|delivery|courier|shipped", re.I), "shipping"),
    (re.compile(r"charged|invoice|overcharged|billing|receipt", re.I), "billing"),
]


@dataclass(frozen=True)
class _Profile:
    base_error: float
    format_slip: float
    latency_ms: float
    latency_per_token_ms: float


_PROFILES = {
    "mock-small": _Profile(base_error=0.10, format_slip=0.10, latency_ms=350, latency_per_token_ms=4),
    "mock-large": _Profile(base_error=0.03, format_slip=0.03, latency_ms=900, latency_per_token_ms=9),
    "mock-judge": _Profile(base_error=0.05, format_slip=0.0, latency_ms=600, latency_per_token_ms=6),
}

_TICKET = re.compile(r'"""(.*?)"""', re.DOTALL)


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def keyword_scores(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores = {c: 0.0 for c in CATEGORIES}
    for kw, (cat, w) in _LEXICON.items():
        if re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", lowered):
            scores[cat] += w
    return scores


class MockProvider(LLMProvider):
    name = "mock"
    supports_structured_output = True

    def __init__(self, default_model: str | None = "mock-small", time_scale: float = 0.01, **kw):
        super().__init__(default_model=default_model, **kw)
        self.time_scale = time_scale
        self.calls = 0

    async def _request(self, prompt, model, temperature, max_tokens, response_schema, seed):
        self.calls += 1
        profile = _PROFILES.get(model, _PROFILES["mock-small"])
        if temperature > 0:
            rng = random.Random()  # genuinely non-deterministic, like sampling
        else:
            rng = random.Random(_seed(model, prompt, seed))

        if "IMPARTIAL EVALUATOR" in prompt:
            text = self._judge(prompt, rng, temperature, pairwise="PAIRWISE" in prompt)
        elif "extract" in prompt.lower() and _TICKET.search(prompt):
            text = self._extract(prompt, profile, rng)
        elif _TICKET.search(prompt):
            text = self._classify(prompt, profile, rng, temperature)
        else:
            text = "This is a simulated response from the mock provider."

        out_tokens = estimate_tokens(text)
        latency = profile.latency_ms + profile.latency_per_token_ms * out_tokens
        latency *= rng.uniform(0.75, 1.35)
        if rng.random() < 0.04:  # occasional slow tail request
            latency *= rng.uniform(1.8, 3.0)
        await asyncio.sleep(latency / 1000 * self.time_scale)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            input_tokens=estimate_tokens(prompt),
            output_tokens=out_tokens,
            latency_ms=round(latency, 1),
        )

    # ------------------------------------------------------------------ #
    def _classify(self, prompt: str, p: _Profile, rng: random.Random, temperature: float) -> str:
        ticket = _TICKET.search(prompt).group(1)
        instructions = _TICKET.sub("", prompt).lower()
        wants_json = "json" in instructions
        strict_json = "only valid json" in instructions or "only json" in instructions
        has_definitions = "definition" in instructions
        allowed = [c for c in CATEGORIES if c in instructions]
        forbids_invented = allowed and ("do not invent" in instructions or "exactly one of" in instructions)

        scores = keyword_scores(ticket)
        ranked = sorted(CATEGORIES, key=lambda c: (-scores[c], rng.random()))
        best, second = ranked[0], ranked[1]
        ambiguous = scores[best] - scores[second] < 1.0

        if has_definitions:
            for pattern, cat in _DEFINITION_RULES:
                if pattern.search(ticket):
                    best = cat
                    break
            ambiguous = False

        error_p = p.base_error * (0.5 if has_definitions else 1.0) + (0.30 if ambiguous else 0.0)
        error_p += 0.10 * temperature
        label = second if rng.random() < error_p else best

        if allowed and label not in allowed:  # the prompt never offered this category
            label = next((c for c in ranked if c in allowed), allowed[0])
        if not allowed or (not forbids_invented and rng.random() < 0.08):
            label = _SYNONYMS[label]

        payload = {"category": label}
        slip = p.format_slip * (0.1 if strict_json else 1.0)
        if not wants_json:
            if rng.random() < 0.7:
                return f"Category: {label.title()}\nReason: the customer mentions {ranked[0]}-related issues."
            return json.dumps(payload)
        if rng.random() < slip:
            return f"Sure! Here is the classification:\n{json.dumps(payload)}"
        if rng.random() < 0.15 and not strict_json:
            return f"```json\n{json.dumps(payload, indent=2)}\n```"
        return json.dumps(payload)

    def _extract(self, prompt: str, p: _Profile, rng: random.Random) -> str:
        text = _TICKET.search(prompt).group(1)
        amount = re.search(r"\$\s?([\d,]+(?:\.\d+)?)", text)
        invoice = re.search(r"\b(INV-\d+)\b", text, re.I)
        email = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text)
        date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        result = {
            "invoice_number": invoice.group(1).upper() if invoice else None,
            "total": float(amount.group(1).replace(",", "")) if amount else None,
            "email": email.group(0) if email else None,
            "date": date.group(1) if date else None,
        }
        if result["total"] is not None and rng.random() < p.base_error:
            result["total"] = round(result["total"] * rng.choice([0.9, 1.1, 10]), 2)
        return json.dumps(result)

    def _judge(self, prompt: str, rng: random.Random, temperature: float, pairwise: bool) -> str:
        reference = _section(prompt, "REFERENCE").lower().strip()
        if pairwise:
            a, b = _section(prompt, "RESPONSE A").lower(), _section(prompt, "RESPONSE B").lower()
            a_ok = bool(reference) and reference in a
            b_ok = bool(reference) and reference in b
            if a_ok == b_ok:  # tie -> position bias towards the first response
                winner = "A" if rng.random() < 0.75 else "B"
            else:
                winner = "A" if a_ok else "B"
            return json.dumps({"winner": winner, "reason": "Simulated pairwise judgement."})

        output = _section(prompt, "MODEL OUTPUT")
        try:
            parsed = json.loads(output.strip())
            valid_json = True
        except (json.JSONDecodeError, ValueError):
            parsed, valid_json = None, False
        correct = bool(reference) and reference in output.lower()
        if correct and valid_json:
            score, reason = 5, "Correct category in valid JSON."
        elif correct:
            score, reason = 3, "Correct category but the output is not clean JSON."
        elif valid_json and isinstance(parsed, dict):
            score, reason = 2, "Well-formed but the category appears wrong."
        else:
            score, reason = 1, "Output is malformed and does not match the reference."
        # Simulated judge imperfection: occasional leniency / strictness.
        noise_p = 0.12 + 0.35 * temperature
        if rng.random() < noise_p:
            score = min(5, max(1, score + rng.choice([-1, 1])))
            reason += " (borderline)"
        return json.dumps({"score": score, "reason": reason})


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"<{name}>\n?(.*?)\n?</{name}>", prompt, re.DOTALL)
    return m.group(1) if m else ""
