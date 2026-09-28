"""Unified LLM client for OpenAI and Anthropic using plain HTTP.

* Every call is cached on disk (keyed by provider, model, messages and params), so
  re-running any stage costs nothing and interrupted runs resume where they stopped.
* Token usage and list-price cost are recorded for every uncached call; a hard cost
  cap (config: max_total_cost_usd) stops the run before the budget is exceeded.
"""
import json
import os
import random
import threading
import time

import requests

from .common import cfg, out_dir, sha, read_json, write_json, append_jsonl

_ledger_lock = threading.Lock()
_ledger = None


class BudgetExceeded(RuntimeError):
    pass


def _ledger_path():
    return out_dir() / "cost_ledger.json"


def ledger():
    global _ledger
    if _ledger is None:
        _ledger = read_json(_ledger_path(), {"total_usd": 0.0, "by_model": {}, "calls": 0})
    return _ledger


def _price(model):
    prices = cfg()["prices_per_million"]
    if model in prices:
        return prices[model]
    for k, v in prices.items():
        if model.startswith(k):
            return v
    raise KeyError(f"No price configured for model {model}; add it to config.yaml prices_per_million")


def cost_of(model, usage):
    p = _price(model)
    inp = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    cache_write = usage.get("cache_write_tokens", 0)
    out = usage.get("output_tokens", 0)
    return ((inp - cached - cache_write) * p["input"] + cached * p.get("cached_input", p["input"])
            + cache_write * p.get("cache_write", p["input"]) + out * p.get("output", 0)) / 1e6


def _record(model, usage, stage):
    c = cost_of(model, usage)
    with _ledger_lock:
        L = ledger()
        L["total_usd"] += c
        L["calls"] += 1
        m = L["by_model"].setdefault(model, {"usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                                              "cached_input_tokens": 0})
        m["usd"] += c
        m["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cached_input_tokens"):
            m[k] += usage.get(k, 0)
        s = L.setdefault("by_stage", {}).setdefault(stage, 0.0)
        L["by_stage"][stage] = s + c
        if L["calls"] % 25 == 0:
            write_json(_ledger_path(), L)
    return c


def flush_ledger():
    with _ledger_lock:
        write_json(_ledger_path(), ledger())


def spend_by_provider():
    out = {}
    for m, v in ledger()["by_model"].items():
        p = provider_of(m)
        out[p] = out.get(p, 0.0) + v["usd"]
    return out


def _check_budget(model=None):
    cap = cfg().get("max_total_cost_usd", 20)
    if ledger()["total_usd"] >= cap:
        flush_ledger()
        raise BudgetExceeded(f"Cost cap reached (${ledger()['total_usd']:.2f} >= ${cap}). "
                             "Raise max_total_cost_usd in config.yaml to continue.")
    if model:
        prov = provider_of(model)
        pcap = (cfg().get("max_cost_by_provider") or {}).get(prov)
        if pcap is not None and spend_by_provider().get(prov, 0.0) >= pcap:
            flush_ledger()
            raise BudgetExceeded(f"Cost cap reached for {prov} (${spend_by_provider()[prov]:.2f} >= ${pcap}).")


def provider_of(model):
    for m in cfg()["models"]:
        if m["name"] == model:
            return m["provider"]
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("dummy"):
        return "dummy"
    if model.startswith("ollama:"):
        return "ollama"
    return "openai"


def _cache_file(key):
    d = out_dir("cache", "llm", key[:2])
    return d / f"{key}.json"


def _post_with_retry(url, headers, body, timeout=180):
    delay = 2.0
    last = None
    for attempt in range(8):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code in (400, 401, 403, 404):
                raise RuntimeError(last)
        except requests.RequestException as e:
            last = str(e)
        time.sleep(delay + random.random())
        delay = min(delay * 2, 60)
    raise RuntimeError(f"LLM call failed after retries: {last}")


def _is_reasoning_openai(model):
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def _call_openai(model, system, messages, temperature, max_tokens, stage="misc"):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = {"model": model, "messages": msgs}
    if _is_reasoning_openai(model):
        body["max_completion_tokens"] = max(max_tokens, 2048)
        key = "agent_reasoning_effort" if stage == "agent" else "verifier_reasoning_effort"
        body["reasoning_effort"] = cfg().get(key, "minimal")
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
    r = _post_with_retry("https://api.openai.com/v1/chat/completions", headers, body)
    u = r.get("usage", {})
    usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
             "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)}
    return r["choices"][0]["message"]["content"] or "", usage


def _call_anthropic(model, system, messages, temperature, max_tokens):
    # prompt caching: mark the system prompt and the most recent user turn
    msgs = []
    for i, m in enumerate(messages):
        content = [{"type": "text", "text": m["content"]}]
        msgs.append({"role": m["role"], "content": content})
    user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    for i in user_idx[-2:]:
        msgs[i]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    body = {"model": model, "max_tokens": max_tokens, "messages": msgs}
    if model not in (cfg().get("no_temperature_models") or []):
        body["temperature"] = temperature
    if system:
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    r = _post_with_retry("https://api.anthropic.com/v1/messages", headers, body)
    u = r.get("usage", {})
    cr, cw = u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
    usage = {"input_tokens": (u.get("input_tokens", 0) or 0) + cr + cw, "output_tokens": u.get("output_tokens", 0),
             "cached_input_tokens": cr, "cache_write_tokens": cw}
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    return text, usage


# ---- dummy provider for offline pipeline tests ------------------------------------------
_DUMMY_HOOK = None


def set_dummy_hook(fn):
    global _DUMMY_HOOK
    _DUMMY_HOOK = fn


def _call_ollama(model, system, messages, temperature, max_tokens):
    base = cfg().get("ollama_base", "http://localhost:11434")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = {"model": model.split("ollama:", 1)[-1], "messages": msgs, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": cfg().get("ollama_ctx", 32768)}}
    last = None
    for attempt in range(4):
        try:
            r = requests.post(f"{base}/api/chat", json=body, timeout=900)
            if r.status_code == 200:
                d = r.json()
                return d["message"]["content"], {"input_tokens": d.get("prompt_eval_count", 0),
                                                  "output_tokens": d.get("eval_count", 0)}
            last = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(3 + 3 * attempt)
    raise RuntimeError(f"Ollama call failed: {last}")


def _call_dummy(model, system, messages, temperature, max_tokens):
    text = _DUMMY_HOOK(model, system, messages) if _DUMMY_HOOK else "THOUGHT: done\nFINISH([-1])"
    n = sum(len(m["content"]) for m in messages) // 4
    return text, {"input_tokens": n, "output_tokens": len(text) // 4, "cached_input_tokens": 0}


def chat(model, messages, system=None, temperature=0.0, max_tokens=1024, stage="misc", salt=None):
    """Returns dict(text, usage, cost, cached)."""
    params = {"temperature": temperature, "max_tokens": max_tokens}
    key = sha({"model": model, "system": system, "messages": messages, "params": params, "salt": salt})
    cf = _cache_file(key)
    hit = read_json(cf)
    if hit is not None:
        hit["cached"] = True
        return hit
    prov = provider_of(model)
    if prov not in ("dummy", "ollama"):
        _check_budget(model)
    t0 = time.time()
    if prov == "openai":
        text, usage = _call_openai(model, system, messages, temperature, max_tokens, stage)
    elif prov == "anthropic":
        text, usage = _call_anthropic(model, system, messages, temperature, max_tokens)
    elif prov == "ollama":
        text, usage = _call_ollama(model, system, messages, temperature, max_tokens)
    else:
        text, usage = _call_dummy(model, system, messages, temperature, max_tokens)
    c = cost_of(model, usage) if prov not in ("dummy", "ollama") else 0.0
    if prov not in ("dummy", "ollama"):
        _record(model, usage, stage)
    res = {"text": text, "usage": usage, "cost": c, "latency_s": time.time() - t0, "model": model}
    write_json(cf, res)
    res["cached"] = False
    return res


def embed(texts, model=None, stage="embed"):
    """OpenAI embeddings with per-text disk cache. Returns list of vectors."""
    model = model or cfg().get("embedding_model", "text-embedding-3-small")
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        t = (t or " ")[:8000]
        cf = _cache_file(sha({"embed": model, "t": t}))
        v = read_json(cf)
        if v is not None:
            out[i] = v
        else:
            todo.append((i, t, cf))
    if todo and provider_of(model) == "dummy":
        import hashlib
        for i, t, cf in todo:
            h = hashlib.sha256(t.encode()).digest()
            v = [(b - 128) / 128 for b in h] * 8
            out[i] = v
        return out
    for s in range(0, len(todo), 256):
        chunk = todo[s:s + 256]
        _check_budget(model)
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}
        r = _post_with_retry("https://api.openai.com/v1/embeddings", headers,
                             {"model": model, "input": [c[1] for c in chunk]})
        _record(model, {"input_tokens": r.get("usage", {}).get("prompt_tokens", 0), "output_tokens": 0}, stage)
        for (i, t, cf), d in zip(chunk, r["data"]):
            out[i] = d["embedding"]
            write_json(cf, d["embedding"])
    return out
