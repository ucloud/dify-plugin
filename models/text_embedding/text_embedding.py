import time
from typing import Optional

import openai
import tiktoken
from dify_plugin import TextEmbeddingModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    PriceType,
)
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError,
)


class UcloudMaasTextEmbeddingModel(TextEmbeddingModel):
    """
    Model class for UCloud ModelVerse text embedding model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_cache = {}

    def _get_client(self, api_key: str) -> openai.OpenAI:
        if api_key not in self._client_cache:
            self._client_cache[api_key] = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.modelverse.cn/v1",
            )
        return self._client_cache[api_key]

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        api_key = credentials.get("openai_api_key")
        if not api_key:
            raise CredentialsValidateFailedError("API Key is required")

        client = self._get_client(api_key)
        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)

        inputs = []
        for text in texts:
            num_tokens = self._get_num_tokens_for_text(model, text)
            if num_tokens >= context_size:
                cutoff = int(len(text) * (context_size / num_tokens))
                inputs.append(text[:cutoff])
            else:
                inputs.append(text)

        batched_embeddings: list[list[float]] = []
        used_tokens = 0

        extra_kwargs = {}
        if user:
            extra_kwargs["user"] = user

        try:
            for i in range(0, len(inputs), max_chunks):
                batch = inputs[i : i + max_chunks]
                response = client.embeddings.create(
                    model=model,
                    input=batch,
                    **extra_kwargs,
                )
                batched_embeddings.extend(
                    [item.embedding for item in response.data]
                )
                if response.usage:
                    used_tokens += response.usage.total_tokens or 0
        except openai.AuthenticationError:
            raise CredentialsValidateFailedError("Invalid API Key")
        except openai.PermissionDeniedError:
            raise CredentialsValidateFailedError("API Key access denied")
        except openai.RateLimitError:
            raise InvokeError("Rate limit exceeded")
        except openai.APIError as e:
            raise InvokeError(f"API error: {str(e)}")

        usage = self._calc_response_usage(model, credentials, used_tokens)
        return TextEmbeddingResult(
            model=model,
            embeddings=batched_embeddings,
            usage=usage,
        )

    def get_num_tokens(
        self, model: str, credentials: dict, texts: list[str]
    ) -> list[int]:
        if not texts:
            return []
        return [self._get_num_tokens_for_text(model, text) for text in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            api_key = credentials.get("openai_api_key")
            if not api_key:
                raise CredentialsValidateFailedError("API Key is required")

            client = self._get_client(api_key)
            response = client.embeddings.create(
                model=model,
                input=["ping"],
            )
            if not response or not response.data:
                raise CredentialsValidateFailedError("Invalid response from API")
        except openai.AuthenticationError:
            raise CredentialsValidateFailedError("Invalid API Key")
        except openai.PermissionDeniedError:
            raise CredentialsValidateFailedError("API Key access denied")
        except openai.NotFoundError:
            raise CredentialsValidateFailedError(f"Model {model} not found")
        except openai.APIError as e:
            raise CredentialsValidateFailedError(f"API error: {str(e)}")
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        models = self.predefined_models()
        model_map = {item.model: item for item in models}
        if model in model_map:
            base_schema = model_map[model]
            return AIModelEntity(
                model=model,
                label=I18nObject(zh_Hans=model, en_US=model),
                model_type=ModelType.TEXT_EMBEDDING,
                features=list(base_schema.features or []),
                fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
                model_properties=dict(base_schema.model_properties or {}),
                parameter_rules=list(base_schema.parameter_rules or []),
                pricing=base_schema.pricing,
            )

        return AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.TEXT_EMBEDDING,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: 8192,
                ModelPropertyKey.MAX_CHUNKS: 32,
            },
            parameter_rules=[],
        )

    def _get_num_tokens_for_text(self, model: str, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _calc_response_usage(
        self, model: str, credentials: dict, tokens: int
    ) -> EmbeddingUsage:
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )
        return EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )

    def _invoke_error_mapping(self) -> dict:
        return {
            openai.AuthenticationError: CredentialsValidateFailedError,
            openai.PermissionDeniedError: CredentialsValidateFailedError,
            openai.NotFoundError: InvokeError,
            openai.UnprocessableEntityError: InvokeError,
            openai.RateLimitError: InvokeError,
            openai.InternalServerError: InvokeError,
            openai.BadGatewayError: InvokeError,
            openai.ServiceUnavailableError: InvokeError,
            openai.APITimeoutError: InvokeError,
        }
