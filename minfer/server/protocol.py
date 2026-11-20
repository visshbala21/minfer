"""Request schemas of the OpenAI-compatible API (the subset minfer implements).

Unknown fields are accepted and ignored, as OpenAI clients routinely send fields a given
server does not use. Fields that would change the output if honoured but are not
implemented are rejected explicitly rather than silently dropped.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from minfer.cache.retention import SessionHint
from minfer.sampling import SamplingParams


class StreamOptions(BaseModel):
    """``stream_options`` of a streaming request."""

    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class _GenerationRequest(BaseModel):
    """Fields shared by completions and chat completions."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=-1)
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    # Extensions also accepted by vLLM and SGLang.
    ignore_eos: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)
    skip_special_tokens: bool = True
    # minfer extensions: the agent session a request belongs to (KV retention policy).
    session_id: str | None = None
    session_tools: list[str] | None = None
    session_final: bool | None = None
    # Accepted only at their neutral values.
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    @model_validator(mode="after")
    def _reject_unsupported(self) -> _GenerationRequest:
        if self.n != 1:
            raise ValueError("only n=1 is supported")
        if self.repetition_penalty != 1.0 or self.presence_penalty or self.frequency_penalty:
            raise ValueError("repetition/presence/frequency penalties are not supported")
        return self

    @property
    def stop_strings(self) -> list[str]:
        """``stop`` normalized to a list of non-empty strings."""
        stop = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        return [s for s in stop if s]

    @property
    def session(self) -> SessionHint | None:
        """The request's session fields, if it names a session.

        ``session_tools`` are the tools the client runs after this response and
        ``session_final`` says the session ends with it; a client that sends neither
        leaves it to the server to read the generated text.
        """
        if self.session_id is None:
            return None
        tools = None if self.session_tools is None else tuple(self.session_tools)
        return SessionHint(self.session_id, tools, self.session_final)

    def sampling_params(self, max_tokens: int) -> SamplingParams:
        """The engine-level parameters for this request."""
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=max(self.top_k, 0),
            stop_token_ids=tuple(self.stop_token_ids),
            ignore_eos=self.ignore_eos,
            seed=self.seed,
        )


class CompletionRequest(_GenerationRequest):
    """Body of ``POST /v1/completions``."""

    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int | None = Field(default=16, ge=1)
    echo: bool = False

    @model_validator(mode="after")
    def _reject_echo(self) -> CompletionRequest:
        if self.echo:
            raise ValueError("echo is not supported")
        return self


class ChatCompletionRequest(_GenerationRequest):
    """Body of ``POST /v1/chat/completions``."""

    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict[str, Any]] | None = None

    @property
    def output_limit(self) -> int | None:
        """The requested output-token limit, if any."""
        return self.max_completion_tokens or self.max_tokens


class TokenizeRequest(BaseModel):
    """Body of ``POST /tokenize``."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str
    add_special_tokens: bool = False


class DetokenizeRequest(BaseModel):
    """Body of ``POST /detokenize``."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    tokens: list[int]
