#!/usr/bin/env python3
"""After a tool result the model must still reason before its next call (patch 0109, TF_GLM_TOOL_REMINDER).

Usage: tools/tool_reasoning_check.py [pad_tokens]      (API_URL / PORT as in client.py). Exit code 1 on failure.
       tools/tool_reasoning_check.py --loop [steps]  (a real loop, the reminder off: where the reasoning goes)

The reported agent loop: four tool calls made, the last message a tool result, tools declared, temperature 0. The same
request goes out with chat_template_kwargs.tool_reasoning_reminder false (the stock prompt: the model tends to close its
think block at once, reasoning_tokens 1) and true, then a next step (the reminded reply and a new result appended) shows
what the reminder costs the kept prompt: cached_tokens against the prompt. pad_tokens (default 0) puts that much prose
in the user's message, for a long-context run. Fails when the reminded reply has no reasoning or no tool call.

--loop: the model runs the log review itself for up to `steps` calls (default 8), reminder off, each step's history
resent three ways: "returned" sends every step's reasoning back; "dropped" sends reasoning_content "" (as many agent
harnesses do), so the server's kept reasoning (patch 0036, TF_GLM_KEEP_REASONING) has to put it back; "edited" also
sends "" and changes the system prompt every step (a timestamp), so the kept reasoning, keyed by the whole conversation
before the calls, is never found and every earlier step renders as an empty <think></think>. Prints each step's
reasoning_tokens; informational, exit code 0.
"""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import URL, open_url, prose  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": f"tool_{i}", "description": f"Analysis step {i} of the log review.",
    "parameters": {"type": "object", "required": ["step"], "properties": {
        "step": {"type": "integer", "description": "The step number."}}}}} for i in range(8)]
FINDINGS = ["3 timeouts on db-2 between 02:10 and 02:14", "retry storm from worker-7 after the timeouts",
            "queue depth peaked at 18,400 at 02:16", "p99 latency back under 200 ms by 02:31"]


def history(pad: int) -> list:
    task = "Analyze these logs step by step, one tool call per step, continue the sequence."
    if pad:
        task += "\n\nArchived log context (background only):\n" + prose(pad, 7)
    out = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": task}]
    for i, finding in enumerate(FINDINGS):
        out.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function",
                    "function": {"name": f"tool_{i}", "arguments": json.dumps({"step": i})}}]})
        out.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps({"step": i, "finding": finding})})
    return out


def ask(messages: list, reminder: bool, label: str = "") -> tuple:
    body = {"model": os.environ.get("MODEL", "GLM-5.3-Flash-EXL3"), "max_tokens": 4096, "temperature": 0,
            "tools": TOOLS, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": True, "tool_reasoning_reminder": reminder}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.time()
    reply = json.load(open_url(req, 3600))
    choice, usage = reply["choices"][0], reply.get("usage") or {}
    msg = choice["message"]
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    calls = msg.get("tool_calls") or []
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    shown = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in calls) or "none"
    print(f"{label or ('reminder on ' if reminder else 'reminder off')}: reasoning_tokens {rt}, {len(reasoning)} reasoning chars, "
          f"finish {choice.get('finish_reason')}, calls {shown}, prompt {usage.get('prompt_tokens')} "
          f"(cached {cached}), {time.time() - start:.1f} s")
    if reasoning and not label:
        print(f"    reasoning: {reasoning[:160]!r}{'...' if len(reasoning) > 160 else ''}")
    if (msg.get("content") or "").strip() and not label:
        print(f"    content: {msg['content'].strip()[:160]!r}")
    return msg, reasoning, calls, choice.get("finish_reason"), usage


def loop(kind: str, steps: int) -> list:
    """The log review run by the model for up to ``steps`` calls; each step's reasoning_tokens."""

    task = ("Analyze these logs step by step: call tool_0 through tool_7 in order, one tool call per step, and read "
            "each result before the next call. After the last step, summarize the incident in two sentences.")
    chat, counts = [{"role": "user", "content": task}], []
    for i in range(steps):
        system = "You are an agent." + (f" Current time: 03:{10 + i:02d}:00 UTC." if kind == "edited" else "")
        msg, reasoning, calls, _, usage = ask([{"role": "system", "content": system}] + chat, False, f"{kind:8} step {i}")
        counts.append((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        if not calls:
            break
        chat.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls,
                     "reasoning_content": reasoning if kind == "returned" else ""})
        for call in calls:
            chat.append({"role": "tool", "tool_call_id": call["id"],
                         "content": json.dumps({"step": i, "finding": FINDINGS[i % len(FINDINGS)]})})
    return counts


def main() -> None:
    if sys.argv[1:2] == ["--loop"]:
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        runs = {kind: loop(kind, steps) for kind in ("returned", "dropped", "edited")}
        for kind, counts in runs.items():
            later = [c for c in counts[1:] if c is not None]
            print(f"{kind:8}: reasoning_tokens by step {counts}; after the first step, {sum(c <= 1 for c in later)} "
                  f"of {len(later)} closed the think block at once")
        return
    pad = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    messages = history(pad)
    ask(messages, False)                                    # the stock prompt, for comparison
    msg, reasoning, calls, finish, _ = ask(messages, True)
    ok = bool(reasoning) and bool(calls) and finish == "tool_calls"
    if calls:
        # the next step, as an agent sends it: the reply (with its reasoning) and its result appended
        step = {"role": "assistant", "content": msg.get("content") or "", "reasoning_content": reasoning,
                "tool_calls": calls}
        result = {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"step": 4, "finding": "no "
                  "errors after 02:31; the incident is closed"})}
        _, more, _, _, usage = ask(messages + [step, result], True)
        prompt, cached = usage.get("prompt_tokens") or 0, (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is not None and prompt:
            print(f"next step: {cached} of {prompt} prompt tokens cached, {prompt - cached} prefilled again")
        ok &= bool(more)
    print("tool_reasoning_check:", "OK: the reminded replies reason before their calls" if ok else
          "FAILED: a reminded reply has no reasoning or no tool call")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
