import logging
import uuid
from collections.abc import Generator
from typing import Optional, Union, cast

import openai
import tiktoken
from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
    PriceConfig,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)


class UcloudMaasLargeLanguageModel(LargeLanguageModel):
    """
    Model class for ucloud-maas large language model.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client_cache = {}

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model using OpenAI SDK

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        key = credentials.get("openai_api_key")
        if not key:
            raise CredentialsValidateFailedError("API Key is required")

        client = self._get_client(key)

        messages = [self._convert_prompt_message_to_dict(m) for m in prompt_messages]

        request_params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **model_parameters
        }

        if tools:
            # Dify uses the current tool-calling message format (assistant
            # `tool_calls` followed by `role=tool` messages).  Sending those
            # messages together with the deprecated `functions` parameter is
            # rejected by OpenAI-compatible gateways.
            request_params.pop("functions", None)
            request_params.pop("function_call", None)
            request_params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        
        if stop:
            request_params["stop"] = stop

        try:
            if stream:
                return self._invoke_stream(client, request_params, prompt_messages, credentials)
            else:
                return self._invoke_sync(client, request_params, prompt_messages, credentials)
        except openai.AuthenticationError:
            raise CredentialsValidateFailedError("Invalid API Key")
        except openai.PermissionDeniedError:
            raise CredentialsValidateFailedError("API Key access denied")
        except openai.RateLimitError:
            raise InvokeError("Rate limit exceeded")
        except openai.APIError as e:
            raise InvokeError(f"API error: {str(e)}")
        except Exception as e:
            raise InvokeError(f"Unknown error: {str(e)}")

    def _get_client(self, api_key: str) -> openai.OpenAI:
        """Get cached OpenAI client or create new one"""
        if api_key not in self._client_cache:
            self._client_cache[api_key] = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.modelverse.cn/v1"
            )
        return self._client_cache[api_key]

    def _update_usage_with_details(self, usage: LLMUsage, chunk_usage) -> LLMUsage:
        """
        Update usage object with detailed token information from OpenAI API response
        
        :param usage: Current LLMUsage object
        :param chunk_usage: Usage information from OpenAI API response
        :return: Updated LLMUsage object
        """
        usage.prompt_tokens = (
            chunk_usage.prompt_tokens or 0
        )
        usage.completion_tokens = (
            chunk_usage.completion_tokens or 0
        )
        usage.total_tokens = (
            chunk_usage.total_tokens or 0
        )
        
     
        return usage

    def _invoke_stream(self, client: openai.OpenAI, request_params: dict, prompt_messages: list[PromptMessage], credentials: dict) -> Generator:
        """Handle streaming invocation using OpenAI SDK"""
        try:
            stream = client.chat.completions.create(**request_params)
            is_reasoning_started = False
            full_assistant_content = ""
            tool_call_storage: dict[int, AssistantPromptMessage.ToolCall] = {}
            prompt_tokens = 0
            completion_tokens = 0
            
            usage = LLMUsage(
                prompt_tokens=0, prompt_unit_price=0.0, prompt_price_unit="0", prompt_price=0.0,
                completion_tokens=0, completion_unit_price=0.0, completion_price_unit="0", completion_price=0.0,
                total_tokens=0, total_price=0.0, currency="USD", latency=0.0
            )
            
            for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        chunk_usage = chunk.usage
                        prompt_tokens = (
                            chunk_usage.prompt_tokens or 0
                        )
                        completion_tokens = (
                            chunk_usage.completion_tokens or 0
                        )
                    continue
                    
                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = getattr(choice, 'finish_reason', None)
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    chunk_usage = chunk.usage
                    prompt_tokens = (
                        chunk_usage.prompt_tokens or 0
                    )
                    completion_tokens = (
                        chunk_usage.completion_tokens or 0
                    )

                response_tool_calls = getattr(delta, "tool_calls", None)
                if response_tool_calls:
                    self._merge_response_tool_call_deltas(
                        response_tool_calls, tool_call_storage
                    )
                else:
                    # Keep response compatibility with gateways that still
                    # return the deprecated `function_call` shape.
                    response_function_call = getattr(delta, "function_call", None)
                    if response_function_call:
                        self._merge_response_tool_call_deltas(
                            [response_function_call], tool_call_storage
                        )

                tool_calls = []
                if finish_reason and tool_call_storage:
                    tool_calls = [
                        tool_call_storage[index]
                        for index in sorted(tool_call_storage)
                    ]

                delta_message = {}
                
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    delta_message["reasoning_content"] = delta.reasoning_content
                
                if hasattr(delta, 'content') and delta.content:
                    delta_message["content"] = delta.content
                    full_assistant_content += delta.content
                
                if delta_message.get("reasoning_content") or delta_message.get("content") or tool_calls:
                    processed_content, is_reasoning_started = self._wrap_thinking_by_reasoning_content(
                        delta_message, is_reasoning_started
                    )
                    
                    message = AssistantPromptMessage(content=processed_content or "", tool_calls=tool_calls)
                    
                    if finish_reason:
                        if not prompt_tokens:
                            prompt_tokens = (
                                self._num_tokens_from_messages(request_params.get("model"), prompt_messages)
                            )
                        if not completion_tokens:
                            full_assistant_prompt_message = AssistantPromptMessage(
                                content=full_assistant_content, tool_calls=tool_calls
                            )
                            completion_tokens = (
                                self._num_tokens_from_messages(
                                    request_params.get("model"), [full_assistant_prompt_message]
                                )
                            )
                        usage = self._calc_response_usage(
                            request_params.get("model"), credentials, prompt_tokens, completion_tokens
                        )

                    yield LLMResultChunk(
                        model=chunk.model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=choice.index,
                            message=message,
                            usage=usage if finish_reason else None,
                            finish_reason=finish_reason
                        )
                    )
                
                elif finish_reason:
                    tool_calls = [
                        tool_call_storage[index]
                        for index in sorted(tool_call_storage)
                    ]
                    if not prompt_tokens:
                        prompt_tokens = (
                            self._num_tokens_from_messages(request_params.get("model"), prompt_messages)
                        )
                    if not completion_tokens:
                        full_assistant_prompt_message = AssistantPromptMessage(
                            content=full_assistant_content, tool_calls=tool_calls
                        )
                        completion_tokens = (
                            self._num_tokens_from_messages(
                                request_params.get("model"), [full_assistant_prompt_message]
                            )
                        )
                    usage = self._calc_response_usage(
                        request_params.get("model"), credentials, prompt_tokens, completion_tokens
                    )
                    yield LLMResultChunk(
                        model=chunk.model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=choice.index,
                            message=AssistantPromptMessage(
                                content="", tool_calls=tool_calls
                            ),
                            usage=usage,
                            finish_reason=finish_reason
                        )
                    )
                    
        except Exception as e:
            raise InvokeError(f"Streaming error: {str(e)}")

    def _invoke_sync(self, client: openai.OpenAI, request_params: dict, prompt_messages: list[PromptMessage], credentials: dict) -> LLMResult:
        """Handle synchronous invocation using OpenAI SDK"""
        try:
            response = client.chat.completions.create(**request_params)
            
            if not response.choices:
                raise InvokeError("No choices in response")
            
            choice = response.choices[0]
            message = choice.message
            
            message_dict = {}
            
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = f"<think>\n{message.reasoning_content}\n</think>"
                if hasattr(message, 'content') and message.content:
                    content += message.content
                message_dict["content"] = content
            elif hasattr(message, 'content') and message.content:
                message_dict["content"] = message.content
            
            response_tool_calls = getattr(message, "tool_calls", None)
            if response_tool_calls:
                tool_calls = self._extract_response_tool_calls(response_tool_calls)
            else:
                # Keep response compatibility with legacy gateways.
                function_call = self._extract_response_function_call(
                    getattr(message, "function_call", None)
                )
                tool_calls = [function_call] if function_call else []
            
            assistant_message = AssistantPromptMessage(
                content=message_dict.get("content"),
                tool_calls=tool_calls
            )

            prompt_tokens = 0
            completion_tokens = 0
            
            if response.usage:
                prompt_tokens = (
                    response.usage.prompt_tokens
                )
                completion_tokens = (
                    response.usage.completion_tokens
                )
            else:
                prompt_tokens = (
                    self._num_tokens_from_messages(request_params.get("model"), prompt_messages)
                )
                completion_tokens = (
                    self._num_tokens_from_messages(request_params.get("model"), [assistant_message])
                )

            usage = self._calc_response_usage(
                request_params.get("model"), credentials, prompt_tokens, completion_tokens
            )

            return LLMResult(
                model=response.model,
                prompt_messages=[],
                message=assistant_message,
                usage=usage
            )
            
        except Exception as e:
            raise InvokeError(f"Sync error: {str(e)}")

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages using tiktoken
        
        :param model: model name
        :param credentials: model credentials  
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return: token count
        """
        return self._num_tokens_from_messages(model, prompt_messages, tools)



    def _num_tokens_from_messages(
        self,
        model: str,
        messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        messages_dict = [self._convert_prompt_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, list):
                    text = ""
                    for item in value:
                        if isinstance(item, dict) and item["type"] == "text":
                            text += item["text"]
                    value = text

                if key == "tool_calls":
                    for tool_call in value:
                        for t_key, t_value in tool_call.items():
                            num_tokens += len(encoding.encode(t_key))
                            if t_key == "function":
                                for f_key, f_value in t_value.items():
                                    num_tokens += len(encoding.encode(f_key))
                                    num_tokens += len(encoding.encode(f_value))
                            else:
                                num_tokens += len(encoding.encode(t_key))
                                num_tokens += len(encoding.encode(t_value))
                else:
                    num_tokens += len(encoding.encode(str(value)))

                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3

        if tools:
            num_tokens += self._num_tokens_for_tools(encoding, tools)

        return num_tokens

    def _num_tokens_for_tools(
        self, encoding: tiktoken.Encoding, tools: list[PromptMessageTool]
    ) -> int:
        """
        Calculate num tokens for tool calling with tiktoken package.
        """
        num_tokens = 0
        for tool in tools:
            num_tokens += len(encoding.encode("type"))
            num_tokens += len(encoding.encode("function"))
            num_tokens += len(encoding.encode("name"))
            num_tokens += len(encoding.encode(tool.name))
            num_tokens += len(encoding.encode("description"))
            num_tokens += len(encoding.encode(tool.description))
            parameters = tool.parameters
            num_tokens += len(encoding.encode("parameters"))
            if "title" in parameters:
                num_tokens += len(encoding.encode("title"))
                num_tokens += len(encoding.encode(parameters.get("title")))
            num_tokens += len(encoding.encode("type"))
            num_tokens += len(encoding.encode(parameters.get("type")))
            if "properties" in parameters:
                num_tokens += len(encoding.encode("properties"))
                for key, value in parameters.get("properties").items():
                    num_tokens += len(encoding.encode(key))
                    for field_key, field_value in value.items():
                        num_tokens += len(encoding.encode(field_key))
                        if field_key == "enum":
                            for enum_field in field_value:
                                num_tokens += 3
                                num_tokens += len(encoding.encode(enum_field))
                        else:
                            num_tokens += len(encoding.encode(str(field_value)))
            if "required" in parameters:
                num_tokens += len(encoding.encode("required"))
                for required_field in parameters["required"]:
                    num_tokens += 3
                    num_tokens += len(encoding.encode(required_field))
        return num_tokens

    def _convert_prompt_message_to_dict(self, message: PromptMessage) -> dict:
        """
        Convert PromptMessage to dict for OpenAI API
        """
        if isinstance(message, UserPromptMessage):
            if isinstance(message.content, str):
                message_dict = {"role": "user", "content": message.content}
            else:
                sub_messages = []
                for message_content in message.content:
                    if message_content.type == PromptMessageContentType.TEXT:
                        message_content = cast(
                            TextPromptMessageContent, message_content
                        )
                        sub_message_dict = {
                            "type": "text",
                            "text": message_content.data,
                        }
                        sub_messages.append(sub_message_dict)
                    elif message_content.type == PromptMessageContentType.IMAGE:
                        message_content = cast(
                            ImagePromptMessageContent, message_content
                        )
                        sub_message_dict = {
                            "type": "image_url",
                            "image_url": {
                                "url": message_content.data,
                                "detail": message_content.detail.value,
                            },
                        }
                        sub_messages.append(sub_message_dict)
                message_dict = {"role": "user", "content": sub_messages}
        elif isinstance(message, AssistantPromptMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type or "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]
        elif isinstance(message, ToolPromptMessage):
            message_dict = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }
        else:
            message_dict = {
                "role": message.role.value,
                "content": message.content
            }

        if message.name and message_dict.get("role") != "tool":
            message_dict["name"] = message.name

        return message_dict

    def _extract_response_function_call(
        self, response_function_call
    ) -> Optional[AssistantPromptMessage.ToolCall]:
        """
        Extract function call from response
        """
        tool_call = None
        if response_function_call:
            function = AssistantPromptMessage.ToolCall.ToolCallFunction(
                name=response_function_call.name or "",
                arguments=response_function_call.arguments or "",
            )
            tool_call = AssistantPromptMessage.ToolCall(
                id=response_function_call.name or "", type="function", function=function
            )
        return tool_call

    def _extract_response_tool_calls(
        self, response_tool_calls
    ) -> list[AssistantPromptMessage.ToolCall]:
        """Convert OpenAI SDK tool calls into Dify tool-call entities."""
        tool_calls = []
        for response_tool_call in response_tool_calls or []:
            function = getattr(response_tool_call, "function", None)
            if not function:
                continue
            function_name = getattr(function, "name", "") or ""
            tool_calls.append(
                AssistantPromptMessage.ToolCall(
                    id=(
                        getattr(response_tool_call, "id", "")
                        or f"call_{uuid.uuid4().hex}"
                    ),
                    type=getattr(response_tool_call, "type", "") or "function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=function_name,
                        arguments=getattr(function, "arguments", "") or "",
                    ),
                )
            )
        return tool_calls

    def _merge_response_tool_call_deltas(
        self,
        response_tool_calls,
        storage: dict[int, AssistantPromptMessage.ToolCall],
    ) -> None:
        """Merge streamed tool-call fragments, including multiple tool calls."""
        for fallback_index, response_tool_call in enumerate(response_tool_calls):
            index = getattr(response_tool_call, "index", None)
            if index is None:
                index = fallback_index

            function = getattr(response_tool_call, "function", None)
            # Legacy `function_call` deltas have no outer function object.
            if function is None:
                function = response_tool_call

            if index not in storage:
                storage[index] = AssistantPromptMessage.ToolCall(
                    id=getattr(response_tool_call, "id", "") or "",
                    type=getattr(response_tool_call, "type", "") or "function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name="", arguments=""
                    ),
                )

            tool_call = storage[index]
            tool_call_id = getattr(response_tool_call, "id", None)
            if tool_call_id:
                tool_call.id = tool_call_id
            tool_call_type = getattr(response_tool_call, "type", None)
            if tool_call_type:
                tool_call.type = tool_call_type

            function_name = getattr(function, "name", None)
            if function_name:
                tool_call.function.name = function_name
                if not tool_call.id:
                    tool_call.id = f"call_{uuid.uuid4().hex}"
            function_arguments = getattr(function, "arguments", None)
            if function_arguments:
                tool_call.function.arguments += function_arguments

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials using OpenAI SDK

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            key = credentials.get("openai_api_key")
            if not key:
                raise CredentialsValidateFailedError("API Key is required")

            client = self._get_client(key)
            test_response = client.chat.completions.create(
                model=model,
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
            raise CredentialsValidateFailedError(f"Model {model} not found")
        except openai.APIError as e:
            raise CredentialsValidateFailedError(f"API error: {str(e)}")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _invoke_error_mapping(self) -> dict:
        """
        Map the invoke error for the model using OpenAI SDK exceptions

        :return: error mapping
        """
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

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema of the base model
        but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        if not model.startswith("ft:"):
            base_model = model
        else:
            base_model = model.split(":")[1]

        models = self.predefined_models()
        model_map = {model.model: model for model in models}
        if base_model not in model_map:
            base_model_schema = AIModelEntity(
                model=model,
                label=I18nObject(zh_Hans=model, en_US=model),
                model_type=ModelType.LLM,
                features=[],
                fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
                model_properties={},
                parameter_rules=[],
            )
        else:
            base_model_schema = model_map[base_model]

        base_model_schema_features = base_model_schema.features or []
        base_model_schema_model_properties = base_model_schema.model_properties or {}
        base_model_schema_parameters_rules = base_model_schema.parameter_rules or []

        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=list(base_model_schema_features),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties=dict(base_model_schema_model_properties.items()),
            parameter_rules=list(base_model_schema_parameters_rules),
            pricing=base_model_schema.pricing,
        )

        return entity
