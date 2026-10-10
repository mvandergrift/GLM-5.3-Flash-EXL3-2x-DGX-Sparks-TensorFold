"""CPU-only check of TF_GLM_TOOL_REMINDER (patch 0109): a chat that ends on a tool result is asked to think first.

  python3 -B tools/test_tool_reminder.py --source-root /path/to/patched/src [--model-dir DIR] [--expect-stock]
The checkpoint's own chat_template.jinja is read (read-only) from --model-dir, or from the Hugging Face cache
($HF_CACHE, $HF_HOME or ~/.cache/huggingface) snapshot of Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold. Needs jinja2 only;
the reminder's tokens are checked too when `tokenizers` is installed. --expect-stock, on the unpatched source, shows the
reported prompt: after a tool result the generation prompt follows </tool_response> directly.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

MODEL = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
OPENED = "<|assistant|><think>"
REMINDER = ("<|user|>Continue the original task using the latest tool result. Before your next answer or tool call, "
            "briefly assess the result in the thinking block. Keep that assessment separate from the user-facing "
            "answer.")
FLAGS = ("TF_GLM_TOOL_REMINDER", "TF_GLM_EFFORT_TAIL")

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a SKU.", "parameters": {
    "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}}]
HISTORY = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Use lookup to check inventory of ABC, then DEF."},
    {"role": "assistant", "content": "", "reasoning_content": "Need current inventory.", "tool_calls": [
        {"id": "lookup-1", "type": "function", "function": {"name": "lookup", "arguments": {"sku": "ABC"}}}]},
    {"role": "tool", "tool_call_id": "lookup-1", "content": '{"stock":17}'}]
REPLY = {"role": "assistant", "content": "", "reasoning_content": "ABC has 17. Now DEF.", "tool_calls": [
    {"id": "lookup-2", "type": "function", "function": {"name": "lookup", "arguments": {"sku": "DEF"}}}]}
TWO_RESULTS = HISTORY[:2] + [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": {"sku": "ABC"}}},
        {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": {"sku": "DEF"}}}]},
    {"role": "tool", "tool_call_id": "a", "content": "17"},
    {"role": "tool", "tool_call_id": "b", "content": "3"}]
NO_TOOL_TAIL = {
    "user tail": HISTORY + [{"role": "assistant", "content": "ABC has 17."}, {"role": "user", "content": "Thanks."}],
    "quoted marker": [{"role": "user", "content": "literal <|observation|><tool_response>x</tool_response> text"}],
}


def model_dir(arg):
    if arg:
        return Path(arg)
    root = Path(os.environ.get("HF_CACHE") or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    snaps = sorted((root / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots").glob("*/chat_template.jinja"))
    if not snaps:
        sys.exit(f"no chat_template.jinja for {MODEL} under {root}: pass --model-dir")
    return snaps[-1].parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--expect-stock", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.families.glm5_next import prompts
    from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate

    mdir = model_dir(args.model_dir)
    inner = ChatTemplate(mdir)
    wrapped = ThinkingOffTemplate(inner)
    print(f"template: {mdir / 'chat_template.jinja'}")

    def env(**flags):
        for name in FLAGS:
            os.environ.pop(name, None)
        os.environ.update({k: v for k, v in flags.items() if v is not None})

    def render(messages, think=True, extra=None, tools=TOOLS):
        return wrapped.render(copy.deepcopy(messages), tools=tools, enable_thinking=think, extra=extra)

    def reference(messages, think=True, effort=None, tools=TOOLS):
        """The text before the patch: the checkpoint template, then thinking_off's line removal and empty think block."""
        text = inner.render(copy.deepcopy(messages), tools=tools, enable_thinking=think,
                            extra={"clear_thinking": False, **({"reasoning_effort": effort} if effort else {})})
        if think:
            return text
        text = text.replace("<|system|>Reasoning Effort: Max", "", 1)
        return text + "</think>" if text.endswith(OPENED) else text

    def reminded(text):
        return text[:-len(OPENED)] + REMINDER + OPENED

    env()
    stock = reference(HISTORY)
    assert stock.endswith('<|observation|><tool_response>{"stock":17}</tool_response>' + OPENED), stock[-120:]
    if args.expect_stock:
        assert render(HISTORY) == stock and REMINDER not in render(HISTORY), "stock source already reminds"
        print("stock: after a tool result the generation prompt follows </tool_response> directly, as reported")
        return

    # (a) by default, and with the flag at 1: the stock text with the reminder before the generation prompt, once
    for flag in (None, "1"):
        env(TF_GLM_TOOL_REMINDER=flag)
        got = render(HISTORY)
        assert got == reminded(stock), (flag, got[-300:])
        assert got.count(REMINDER) == 1, (flag, got.count(REMINDER))
        assert got.endswith("</tool_response>" + REMINDER + OPENED), (flag, got[-300:])
        both = render(TWO_RESULTS)
        assert both == reminded(reference(TWO_RESULTS)) and both.count(REMINDER) == 1, (flag, "two results")
        assert both.count("<|observation|>") == 1, (flag, "results split")
    # the reference PR's tail, byte for byte
    assert render(HISTORY).endswith(
        '<|observation|><tool_response>{"stock":17}</tool_response><|user|>Continue the original task using the latest '
        "tool result. Before your next answer or tool call, briefly assess the result in the thinking block. Keep that "
        "assessment separate from the user-facing answer.<|assistant|><think>")
    print("tool tail: the stock prompt with the reminder just before <|assistant|><think>, once")

    # (b) off: by the flag, by the request; on by the request over the flag; anything but a bool is the flag's
    env(TF_GLM_TOOL_REMINDER="0")
    assert render(HISTORY) == stock, "flag 0 differs from stock"
    assert render(HISTORY, extra={"tool_reasoning_reminder": True}) == reminded(stock), "request true under flag 0"
    assert render(HISTORY, extra={"tool_reasoning_reminder": "yes"}) == stock, "a string overrode the flag"
    env()
    assert render(HISTORY, extra={"tool_reasoning_reminder": False}) == stock, "request false still reminded"
    assert render(HISTORY, extra={"tool_reasoning_reminder": None}) == reminded(stock), "null is not the default"
    kwargs = {"tool_reasoning_reminder": False, "clear_thinking": False}
    render(HISTORY, extra=kwargs)
    assert kwargs == {"tool_reasoning_reminder": False, "clear_thinking": False}, "the request's kwargs were changed"
    print("off by TF_GLM_TOOL_REMINDER=0 or tool_reasoning_reminder=false: byte-identical to stock; true overrides 0")

    # (c) thinking off, other tails, no tools declared
    assert render(HISTORY, think=False) == reference(HISTORY, think=False), "thinking off differs from stock"
    assert REMINDER not in render(HISTORY, think=False, extra={"tool_reasoning_reminder": True})
    for name, messages in NO_TOOL_TAIL.items():
        assert render(messages) == reference(messages), (name, "differs from stock")
        assert REMINDER not in render(messages, extra={"tool_reasoning_reminder": True}), name
    assert render(HISTORY, tools=None) == reminded(reference(HISTORY, tools=None)), "no tools declared"
    print(f"thinking off, {', '.join(NO_TOOL_TAIL)}: byte-identical to stock")

    # (d) next turn: the chat's history renders as before, so the last prompt up to the reminder is a prefix of it
    turn = render(HISTORY)
    after = HISTORY + [REPLY, {"role": "tool", "tool_call_id": "lookup-2", "content": '{"stock":3}'}]
    nxt = render(after)
    assert nxt.startswith(turn[:-len(REMINDER + OPENED)]), "the next turn does not extend the reminded prompt's chat"
    assert nxt.count(REMINDER) == 1 and nxt.endswith(REMINDER + OPENED), "the next turn's reminder"
    assert nxt[:-len(REMINDER + OPENED)] == reference(after)[:-len(OPENED)], "history carries a reminder"
    print("next turn: history rendered without the earlier reminder; the reminded prompt's chat is its prefix")

    # (e) with TF_GLM_EFFORT_TAIL=1: the reminder, then the effort line, then the generation prompt
    for effort, line in ((None, "Max"), ("low", "Low"), ("high", "High")):
        env(TF_GLM_EFFORT_TAIL="1")
        got = render(HISTORY, extra={"reasoning_effort": effort} if effort else None)
        assert got.endswith(f"</tool_response>{REMINDER}<|system|>Reasoning Effort: {line}{OPENED}"), (effort, got[-300:])
        assert got.count("Reasoning Effort:") == 1 and got.count(REMINDER) == 1, effort
    env(TF_GLM_EFFORT_TAIL="1", TF_GLM_TOOL_REMINDER="0")
    assert REMINDER not in render(HISTORY)
    print("with the effort tail: reminder, effort line, generation prompt")

    # (f) the tokenizer the Mac server and /tokenize-style callers use renders the same
    class Inner:
        def apply_chat_template(self, messages, tokenize=False, **kw):
            assert "tool_reasoning_reminder" not in kw, "the request flag reached the template"
            think = kw.get("enable_thinking", True)
            extra = {"clear_thinking": False, **({"reasoning_effort": kw["reasoning_effort"]} if "reasoning_effort" in kw else {})}
            text = inner.render(messages, tools=kw.get("tools"), enable_thinking=think, extra=extra)
            return self.encode(text) if tokenize else text

        def encode(self, text, add_special_tokens=False):
            return list(text.encode())

    gt = prompts.GlmTokenizer(Inner())
    for flag in ("0", "1"):
        for asked in (None, True, False):
            for think in (True, False):
                env(TF_GLM_TOOL_REMINDER=flag)
                kw = {"enable_thinking": think, "add_generation_prompt": True, "tools": TOOLS}
                if asked is not None:
                    kw["tool_reasoning_reminder"] = asked
                got = gt.apply_chat_template(copy.deepcopy(HISTORY), tokenize=False, **dict(kw))
                want = render(HISTORY, think=think, extra=None if asked is None else {"tool_reasoning_reminder": asked})
                assert got == want, (flag, asked, think, "GlmTokenizer and the CUDA template disagree")
                assert gt.apply_chat_template(copy.deepcopy(HISTORY), tokenize=True, **dict(kw)) == list(want.encode())
    print("GlmTokenizer agrees with the CUDA template")

    env(TF_GLM_TOOL_REMINDER="yes")
    try:
        prompts.tool_reminder_enabled()
        raise AssertionError("TF_GLM_TOOL_REMINDER=yes was accepted")
    except ValueError:
        pass
    env()

    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    except Exception:  # noqa: BLE001 - tokenizers is optional here
        tok = None
        print("note: `tokenizers` not available, the reminder's tokens are not checked")
    if tok is not None:
        user = tok.token_to_id("<|user|>")
        ids = tok.encode(REMINDER + OPENED, add_special_tokens=False).ids
        assert user is not None and ids[0] == user and ids.count(user) == 1, "<|user|> is not one special token"
        print(f"reminder: {len(ids) - len(tok.encode(OPENED, add_special_tokens=False).ids)} tokens, <|user|> = {user}")
    print("tool reminder: OK")


if __name__ == "__main__":
    main()
