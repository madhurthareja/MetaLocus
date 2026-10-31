"""Text-to-speech for spoken directives, via Meta's SeamlessM4T v2.

Not Voicebox/Audiobox: Meta never publicly released those generation
models (withheld over misuse concerns; the only open "Audiobox" artifact
on Hugging Face, `facebook/audiobox-aesthetics`, is an audio *quality
scorer*, not a speech generator). SeamlessM4T v2 is the real, openly
downloadable Meta model that actually does text-to-speech, with an
official `transformers` integration -- lower integration risk than a
bespoke pip package or an Ollama GGUF repackaging (both were tried and
didn't pan out: Ollama's completion API has no vocoder step for a TTS
model's audio-codec-token output).

Lazy-loaded like Sam31NativeSegmenter: constructing this class doesn't
touch the GPU or the ~9GB checkpoint until the first `synthesize()` call.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL_ID = "facebook/seamless-m4t-v2-large"


class SeamlessTTSBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, lang: str = "eng"):
        self._model_id = model_id
        self._lang = lang
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, SeamlessM4Tv2Model

            self._processor = AutoProcessor.from_pretrained(self._model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SeamlessM4Tv2Model.from_pretrained(
                self._model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
            ).to(device)
        return self._processor, self._model

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Returns (waveform, sample_rate). waveform is float32, mono."""
        import torch

        processor, model = self._load()
        inputs = processor(text=text, src_lang=self._lang, return_tensors="pt").to(model.device)
        with torch.no_grad():
            audio = model.generate(**inputs, tgt_lang=self._lang)[0].float().cpu().numpy().squeeze()
        return audio, int(model.config.sampling_rate)

    def synthesize_to_file(self, text: str, path: str) -> None:
        import soundfile as sf

        audio, sample_rate = self.synthesize(text)
        sf.write(path, audio, sample_rate)


def build_tts_backend(config: dict) -> SeamlessTTSBackend | None:
    """None if config.tts.backend is unset -- callers must handle that
    (no TTS configured is a valid, common state, not an error)."""
    backend = config.get("tts", {}).get("backend")
    if not backend:
        return None
    if backend == "seamless_m4t_v2":
        return SeamlessTTSBackend(
            model_id=config["tts"].get("model_id", DEFAULT_MODEL_ID),
            lang=config["tts"].get("lang", "eng"),
        )
    raise ValueError(f"unknown tts.backend: {backend!r}")
