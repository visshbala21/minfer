"""Prompt sets shared by the model-equivalence tests."""

from __future__ import annotations

ENGLISH = (
    "The history of the printing press is usually told as a story about Johannes Gutenberg, "
    "but movable type had been used in China and Korea for centuries before the 1450s. What "
    "changed in Mainz was the combination: an alloy that cast cleanly, an oil-based ink that "
    "clung to metal, and a press adapted from wine-making. Within fifty years there were "
    "printing shops in more than two hundred European towns, and the price of a book had "
    "fallen far enough that a merchant, and not only a monastery, could own a shelf of them."
)

CODE = '''import heapq
from collections import defaultdict


def dijkstra(graph: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """Return the shortest distance from source to every reachable node."""
    dist: dict[str, float] = defaultdict(lambda: float("inf"))
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = d + weight
            if candidate < dist[neighbor]:
                dist[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return dict(dist)
'''

CHINESE = (
    "长江是中国最长的河流，全长约六千三百公里，发源于青藏高原的唐古拉山脉，自西向东流经十一个"
    "省级行政区，最后在上海注入东海。它不仅是重要的航运通道，也养育了沿岸数亿人口。"
    "三峡大坝建成之后，长江中下游的防洪能力显著提高，但也带来了生态和移民方面的争议。"
)

CHAT = [
    {"role": "system", "content": "You are a terse assistant for a Kubernetes on-call rotation."},
    {"role": "user", "content": "Pod payments-7f9c is in CrashLoopBackOff. Where do I start?"},
    {
        "role": "assistant",
        "content": "Run `kubectl logs payments-7f9c --previous` and `kubectl describe pod "
        "payments-7f9c`. Look at the last state's exit code and the Events section.",
    },
    {"role": "user", "content": "Exit code 137, and Events show 'OOMKilled'. Limit is 256Mi."},
]

STRUCTURED = (
    '{"tool_calls": [{"id": "call_8f2a", "type": "function", "function": {"name": '
    '"search_orders", "arguments": {"customer_id": 48213, "status": ["shipped", "returned"], '
    '"since": "2026-01-15T00:00:00Z", "limit": 25}}}], "usage": {"prompt_tokens": 1843, '
    '"completion_tokens": 57}, "latency_ms": 412.75, "ok": true, "note": "naïve café ☕ → 3.14159"}'
)


def model_test_prompts(tokenizer) -> dict[str, list[int]]:
    """Token ids for the five diverse model-equivalence prompts."""
    chat = tokenizer.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
    texts = {
        "english": ENGLISH,
        "code": CODE,
        "chinese": CHINESE,
        "chat": chat,
        "structured": STRUCTURED,
    }
    return {name: tokenizer(text).input_ids for name, text in texts.items()}
