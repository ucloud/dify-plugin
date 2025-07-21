import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class ModelverseModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            import openai
            
            api_key = credentials.get("openai_api_key")
            if not api_key:
                raise CredentialsValidateFailedError("API Key is required")

            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://deepseek.modelverse.cn/v1"
            )

            test_response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                stream=False
            )

            if not test_response:
                raise CredentialsValidateFailedError("Invalid response from API")

        except openai.AuthenticationError:
            raise CredentialsValidateFailedError("Invalid API Key")
        except openai.PermissionDeniedError:
            raise CredentialsValidateFailedError("API Key access denied")
        except openai.NotFoundError:
            raise CredentialsValidateFailedError("Model not found or access denied")
        except openai.BadRequestError as e:
            raise CredentialsValidateFailedError(f"Bad request: {str(e)}")
        except openai.APIError as e:
            raise CredentialsValidateFailedError(f"API error: {str(e)}")
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise CredentialsValidateFailedError(str(ex))
