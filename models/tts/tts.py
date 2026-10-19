import base64
from collections.abc import Generator

import requests
from dify_plugin import TTSModel
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
)

BASE_URL = "https://api.modelverse.cn/v1"
REQUEST_TIMEOUT = 120


class RateLimitError(Exception):
    pass


class Qwen3TTSModel(TTSModel):
    """
    Model class for UCloud ModelVerse Qwen3 TTS model.
    """

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: str | None = None,
    ) -> bytes | Generator[bytes, None, None]:
        if not content_text or not content_text.strip():
            raise InvokeBadRequestError("Text-to-speech input must not be empty")

        api_key = credentials.get("openai_api_key")
        if not api_key:
            raise InvokeBadRequestError("API Key is required")

        voices = self.get_tts_model_voices(model=model, credentials=credentials)
        if not voices:
            raise InvokeBadRequestError("No voices found for the model")
        valid_voices = {item["value"] for item in voices}
        if voice not in valid_voices:
            voice = (
                self._get_model_default_voice(model, credentials)
                or voices[0]["value"]
            )

        limit = self._get_model_word_limit(model, credentials) or 600
        audio_chunks = []
        for sentence in self._split_text_into_sentences(content_text, max_length=limit):
            for start in range(0, len(sentence), limit):
                chunk = sentence[start : start + limit].strip()
                if chunk:
                    audio_chunks.append(
                        self._synthesize(model, api_key, chunk, voice)
                    )

        if not audio_chunks:
            raise InvokeBadRequestError("No audio bytes found")

        return self._merge_audio(audio_chunks)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        api_key = credentials.get("openai_api_key")
        if not api_key:
            raise CredentialsValidateFailedError("API Key is required")

        try:
            voice = self._get_model_default_voice(model, credentials) or "Cherry"
            audio = self._synthesize(model, api_key, "你好", voice)
            if not audio:
                raise CredentialsValidateFailedError("Invalid response from API")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def _synthesize(self, model: str, api_key: str, text: str, voice: str) -> bytes:
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": self._get_model_audio_type(model, {}) or "wav",
        }
        try:
            response = requests.post(
                f"{BASE_URL}/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout as ex:
            raise ConnectionError(f"Request timed out: {ex!s}") from ex
        except requests.exceptions.RequestException as ex:
            raise ConnectionError(f"Request failed: {ex!s}") from ex

        if response.status_code in (401, 403):
            raise PermissionError(response.text)
        if response.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {response.text}")
        if response.status_code >= 400:
            raise ValueError(
                f"API error: status={response.status_code}, body={response.text}"
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type or response.content.lstrip().startswith(
            b"{"
        ):
            return self._extract_audio_from_json(response)

        if not response.content:
            raise ValueError("Empty audio response")
        return response.content

    def _extract_audio_from_json(self, response: requests.Response) -> bytes:
        try:
            data = response.json()
        except ValueError as ex:
            raise ValueError(f"Invalid JSON response: {response.text}") from ex

        if data.get("code") or data.get("error"):
            message = data.get("message") or data.get("error") or data.get("code")
            raise ValueError(f"API error: {message}")

        audio = (data.get("output") or {}).get("audio") or {}
        if audio.get("data"):
            return base64.b64decode(audio["data"])
        if audio.get("url"):
            try:
                audio_response = requests.get(
                    audio["url"],
                    timeout=REQUEST_TIMEOUT,
                )
                audio_response.raise_for_status()
            except requests.exceptions.RequestException as ex:
                raise ConnectionError(
                    f"Failed to download audio: {ex!s}"
                ) from ex
            return audio_response.content

        raise ValueError(f"No audio found in response: {response.text}")

    @staticmethod
    def _merge_audio(audio_chunks: list[bytes]) -> bytes:
        if len(audio_chunks) == 1 or not audio_chunks[0].startswith(b"RIFF"):
            return b"".join(audio_chunks)

        first = audio_chunks[0]
        data_index = first.find(b"data")
        if data_index < 0:
            return b"".join(audio_chunks)

        header = bytearray(first[: data_index + 8])
        pcm = bytearray()
        for chunk in audio_chunks:
            index = chunk.find(b"data")
            if index >= 0:
                pcm.extend(chunk[index + 8 :])

        header[4:8] = (len(pcm) + len(header) - 8).to_bytes(4, "little")
        header[data_index + 4 : data_index + 8] = len(pcm).to_bytes(4, "little")
        return bytes(header + pcm)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeAuthorizationError: [PermissionError],
            InvokeConnectionError: [ConnectionError],
            InvokeBadRequestError: [ValueError],
            InvokeRateLimitError: [RateLimitError],
        }
