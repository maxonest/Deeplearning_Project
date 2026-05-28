"""LLM clients used by the backend.

The Transformers model is loaded lazily on the first request. This keeps API
startup fast and lets unit tests use the lightweight placeholder client.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Iterator
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        """Generate text from a prompt."""

    def stream_generate(self, prompt: str, enable_thinking: bool | None = None) -> Iterator[str]:
        """Generate text token by token."""


class PlaceholderLLMClient:
    """Safe fallback when local model loading is disabled."""

    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        return (
            "本地模型尚未启用。请在 .env 中设置 USE_LOCAL_MODEL=true，"
            "并确认 LOCAL_MODEL_PATH 指向本地模型目录。\n\n"
            f"当前已完成 RAG 提示词组装，提示词长度为 {len(prompt)} 字符。"
        )

    def stream_generate(self, prompt: str, enable_thinking: bool | None = None) -> Iterator[str]:
        text = self.generate(prompt, enable_thinking=enable_thinking)
        for index in range(0, len(text), 12):
            yield text[index : index + 12]


class TransformersLLMClient:
    """Lazy local inference client for Qwen/DeepSeek/GLM style causal LMs."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.local_files_only = local_files_only
        self._tokenizer = None
        self._model = None
        self._torch = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def _load(self) -> None:
        if self.is_loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {self.model_path}. "
                "Set LOCAL_MODEL_PATH in .env, for example models/qwen/Qwen3.5-9B."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if not torch.cuda.is_available():
            self._model.to("cpu")
        self._model.eval()

    def _format_chat_prompt(self, prompt: str, enable_thinking: bool | None = None) -> str:
        assert self._tokenizer is not None
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        messages = [
            {"role": "system", "content": "你是一个专业、严谨的本地领域知识问答助手。"},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt

    def _build_generation_inputs(self, prompt: str, enable_thinking: bool | None = None):
        self._load()
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None

        text = self._format_chat_prompt(prompt, enable_thinking=enable_thinking)
        inputs = self._tokenizer(text, return_tensors="pt")
        input_device = next(self._model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.05,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        return inputs, generation_kwargs

    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        inputs, generation_kwargs = self._build_generation_inputs(prompt, enable_thinking=enable_thinking)
        assert self._torch is not None
        assert self._model is not None
        assert self._tokenizer is not None

        with self._torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream_generate(self, prompt: str, enable_thinking: bool | None = None) -> Iterator[str]:
        inputs, generation_kwargs = self._build_generation_inputs(prompt, enable_thinking=enable_thinking)
        assert self._model is not None
        assert self._tokenizer is not None

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60,
        )
        generation_kwargs["streamer"] = streamer
        thread = Thread(target=self._model.generate, kwargs={**inputs, **generation_kwargs}, daemon=True)
        thread.start()

        for text in streamer:
            if text:
                yield text
        thread.join()


def build_llm_client(
    use_local_model: bool,
    model_path: str | Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool = False,
    local_files_only: bool = True,
) -> LLMClient:
    if not use_local_model:
        return PlaceholderLLMClient()
    return TransformersLLMClient(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        local_files_only=local_files_only,
    )
