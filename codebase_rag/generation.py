"""Answer generation: prompt building, conversation memory, streaming and citation extraction."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .retrieval import Hit

SYSTEM_PROMPT = """You are a senior software engineer answering questions about a codebase.
Use ONLY the numbered code excerpts provided. Cite every claim with the excerpt number in
square brackets, like [1] or [2][3]. If the excerpts do not contain the answer, say so
instead of guessing. Be concise and show short code snippets when they help."""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class ChatMemory:
    """Keeps the last `max_turns` question/answer pairs for follow-up questions."""

    max_turns: int = 6
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append((question, answer))
        del self.turns[: -self.max_turns]

    def as_messages(self) -> list[dict]:
        messages: list[dict] = []
        for question, answer in self.turns:
            messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return messages

    def clear(self) -> None:
        self.turns.clear()


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f"[{number}] {c.repo}/{c.citation} ({c.kind} {c.symbol})\n```{c.language}\n{c.text}\n```")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit], memory: ChatMemory) -> list[dict]:
    user = f"<CONTEXT>\n{format_context(hits)}\n</CONTEXT>\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *memory.as_messages(), {"role": "user", "content": user}]


def cited_hits(answer: str, hits: list[Hit]) -> list[Hit]:
    """The hits the answer actually cites (in citation order), falling back to all hits."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(hits) and n not in seen:
            seen.append(n)
    return [hits[n - 1] for n in seen] or hits


class Generator:
    """Thin wrapper over any OpenAI-compatible chat API (Groq by default)."""

    def __init__(self, api_key: str, model: str, base_url: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def condense(self, question: str, memory: ChatMemory) -> str:
        """Rewrite a follow-up into a standalone search query using the conversation so far."""
        if not memory.turns:
            return question
        history = "\n".join(f"Q: {q}\nA: {a[:300]}" for q, a in memory.turns[-3:])
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": "Rewrite the last question as a standalone search query for a codebase. "
                    f"Return only the query.\n\n{history}\n\nLast question: {question}",
                }
            ],
        )
        return (response.choices[0].message.content or question).strip()

    def stream(self, question: str, hits: list[Hit], memory: ChatMemory) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            stream=True,
            messages=build_messages(question, hits, memory),
        )
        for event in response:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
