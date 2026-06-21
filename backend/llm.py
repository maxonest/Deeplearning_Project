"""LLM clients used by the backend."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock, Thread
from typing import Iterator
from typing import Protocol

from utils.prompts import SPORTS_HEALTH_SYSTEM_PROMPT


logger = logging.getLogger("uvicorn.error")


class LLMClient(Protocol):
    def load(self) -> None:
        """Load model resources if needed."""

    def generate(self, prompt: str, enable_thinking: bool | None = None) -> str:
        """Generate text from a prompt."""

    def stream_generate(self, prompt: str, enable_thinking: bool | None = None) -> Iterator[str]:
        """Generate text token by token."""


class PlaceholderLLMClient:
    """Safe fallback when local model loading is disabled."""

    def load(self) -> None:
        return

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
    """Local inference client for Qwen/DeepSeek/GLM style causal LMs."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        local_files_only: bool = True,
        lora_adapter_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.lora_adapter_path = Path(lora_adapter_path) if lora_adapter_path else None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        self.local_files_only = local_files_only
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_lock = RLock()
        self._generation_lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        with self._load_lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self.is_loaded:
            return
        logger.info(
            "Model load stage 1/6: validating paths. base_model=%s, lora_adapter=%s",
            self.model_path,
            self.lora_adapter_path or "disabled",
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {self.model_path}. "
                "Set LOCAL_MODEL_PATH in .env, for example models/qwen/Qwen3.5-9B."
            )
        if self.lora_adapter_path is not None:
            if not self.lora_adapter_path.exists():
                raise FileNotFoundError(
                    f"LoRA adapter path does not exist: {self.lora_adapter_path}. "
                    "Set LOCAL_LORA_ADAPTER_PATH to the trained adapter directory."
                )
            adapter_config = self.lora_adapter_path / "adapter_config.json"
            adapter_weights = (
                self.lora_adapter_path / "adapter_model.safetensors",
                self.lora_adapter_path / "adapter_model.bin",
            )
            if not adapter_config.is_file():
                raise FileNotFoundError(f"LoRA adapter config does not exist: {adapter_config}")
            if not any(path.is_file() for path in adapter_weights):
                expected = " or ".join(str(path) for path in adapter_weights)
                raise FileNotFoundError(f"LoRA adapter weights do not exist. Expected {expected}")

        logger.info("Model load stage 2/6: importing torch and transformers.")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Runtime versions: python dependencies torch=%s, transformers=%s, cuda_runtime=%s.",
            torch.__version__,
            transformers.__version__,
            torch.version.cuda,
        )
        self._torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None

        logger.info("Model load stage 3/6: loading tokenizer.")
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        logger.info(
            "Model load stage 4/6: loading base model. cuda=%s, dtype=%s, device_map=%s",
            torch.cuda.is_available(),
            dtype,
            device_map,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if self.lora_adapter_path is not None:
            logger.info("Model load stage 5/6: importing PEFT and mounting LoRA adapter.")
            import peft
            from peft import PeftModel

            logger.info("Runtime version: peft=%s.", peft.__version__)
            self._model = PeftModel.from_pretrained(
                self._model,
                str(self.lora_adapter_path),
                local_files_only=self.local_files_only,
            )
        else:
            logger.info("Model load stage 5/6: LoRA adapter disabled.")
        if not torch.cuda.is_available():
            self._model.to("cpu")
        self._model.eval()
        logger.info("Model load stage 6/6: model is ready for inference.")

    def _format_chat_prompt(self, prompt: str, enable_thinking: bool | None = None) -> str:
        assert self._tokenizer is not None
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        messages = [
            {"role": "system", "content": SPORTS_HEALTH_SYSTEM_PROMPT},
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
        self.load()
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
        with self._generation_lock:
            inputs, generation_kwargs = self._build_generation_inputs(prompt, enable_thinking=enable_thinking)
            assert self._torch is not None
            assert self._model is not None
            assert self._tokenizer is not None

            with self._torch.inference_mode():
                output_ids = self._model.generate(**inputs, **generation_kwargs)

            new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream_generate(self, prompt: str, enable_thinking: bool | None = None) -> Iterator[str]:
        with self._generation_lock:
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
    lora_adapter_path: str | Path | None = None,
) -> LLMClient:
    if not use_local_model:
        return PlaceholderLLMClient()
    return TransformersLLMClient(
        model_path=model_path,
        lora_adapter_path=lora_adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_thinking=enable_thinking,
        local_files_only=local_files_only,
    )
