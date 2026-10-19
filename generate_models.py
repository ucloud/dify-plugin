#!/usr/bin/env python3
"""
Fetch models from modelverse API and generate YAML files for dify plugin.
Usage: python generate_models.py
"""

import json
import os
import re
import urllib.request

API_URL = "https://api.modelverse.cn/v1/models"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "llm")

# Keywords to filter out (case-insensitive)
FILTER_KEYWORDS = [
    "suno",
    "embedding",
    "rerank",
    "speech",
    "tts",
    "indextts",
    "codex",
    "publishers",
    "wan",
    "image",
    "video",
    "flux",
    "black-forest-labs",
    "kontext",
    "ocr",
    "vidu",
    "veo",
    "text-to-sound",
    "sora",
    "step1x",
    "pixverse",
    "midjourney",
    "happyhorse",
    "kling",
    "seedance",
    "seedream",
    "hailuo"
]
FILTER_EXACT = ["gpt-5.4-pro","easydoc-emr-mask","easydoc-extract","easydoc-fin-chat","easydoc-parse-premium"]

YAML_TEMPLATE = """model: {model_id}
label:
  zh_Hans: {label}
  en_US: {label}
model_type: llm
features:
  - multi-tool-call
  - agent-thought
  - stream-tool-call
  - vision
  - tool-call
model_properties:
  mode: chat
  context_size: {context_size}
parameter_rules:
  - name: temperature
    use_template: temperature
  - name: top_p
    use_template: top_p
  - name: presence_penalty
    use_template: presence_penalty
  - name: frequency_penalty
    use_template: frequency_penalty
  - name: max_tokens
    use_template: max_tokens
    default: {default_max_tokens}
    min: 1
    max: 65535
  - name: response_format
    use_template: response_format
{thinking_rule}pricing:
  input: '0.004'
  output: '0.016'
  unit: '0.001'
  currency: CNY
""".lstrip()

# Claude Sonnet/Opus 4-6 and above: 1M context
CLAUDE_1M_PATTERNS = [
    r"claude-.*(?:sonnet|opus)-4-[6-9]",
    r"claude-(?:sonnet|opus)-4-[6-9]",
]


def is_claude_1m(model_id: str) -> bool:
    lower = model_id.lower()
    return bool(re.search(r"claude.*(?:sonnet|opus).*4[\.\-]?[6-9]", lower))


def is_gpt_5_2_plus(model_id: str) -> bool:
    """Match gpt-5.2 and 5.3"""
    lower = model_id.lower()
    m = re.search(r"gpt-?5[\.\-](\d+)", lower)
    if m:
        return int(m.group(1)) in (2, 3)
    return False


def is_gpt_5_4_plus(model_id: str) -> bool:
    """Match gpt-5.4 and above"""
    lower = model_id.lower()
    m = re.search(r"gpt-?5[\.\-](\d+)", lower)
    if m:
        return int(m.group(1)) >= 4
    return False


def fetch_models():
    req = urllib.request.Request(API_URL)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("data", [])


def should_filter(model_id: str) -> bool:
    lower = model_id.lower()
    if any(kw in lower for kw in FILTER_KEYWORDS):
        return True
    if model_id in FILTER_EXACT:
        return True
    return False


def model_id_to_filename(model_id: str) -> str:
    """Convert model ID to a safe filename, e.g. 'deepseek-ai/DeepSeek-R1' -> 'deepseek-ai-deepseek-r1.yaml'"""
    name = model_id.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = name.strip("-")
    return f"{name}.yaml"


def model_id_to_label(model_id: str) -> str:
    """Use model ID as label directly."""
    return model_id


THINKING_OBJECT = """  - name: thinking
    label:
      zh_Hans: 思考模式
      en_US: Thinking Mode
    type: string
    help:
      zh_Hans: 控制是否开启思考模式（thinking.type）。enabled 强制开启，disabled 强制关闭。
      en_US: Controls thinking mode (thinking.type). enabled forces it on, disabled forces it off.
    required: false
    default: enabled
    options:
      - enabled
      - disabled
"""

THINKING_DOUBAO = """  - name: thinking
    label:
      zh_Hans: 思考模式
      en_US: Thinking Mode
    type: string
    help:
      zh_Hans: 控制思考模式（thinking.type）。enabled 强制开启，disabled 强制关闭，auto 由模型自行判断。
      en_US: Controls thinking mode (thinking.type). enabled forces it on, disabled forces it off, auto lets the model decide.
    required: false
    default: enabled
    options:
      - enabled
      - disabled
      - auto
"""

THINKING_MINIMAX_M3 = """  - name: thinking
    label:
      zh_Hans: 思考模式
      en_US: Thinking Mode
    type: string
    help:
      zh_Hans: 控制思考模式（thinking.type）。adaptive 自适应开启（默认），disabled 关闭思考。
      en_US: Controls thinking mode (thinking.type). adaptive enables it (default), disabled turns it off.
    required: false
    default: adaptive
    options:
      - adaptive
      - disabled
"""

ENABLE_THINKING_ON = """  - name: enable_thinking
    label:
      zh_Hans: 思考模式
      en_US: Thinking Mode
    type: boolean
    help:
      zh_Hans: 是否开启思考模式（enable_thinking）。关闭后模型直接回答，可降低延迟与 Token 消耗。
      en_US: Whether to enable thinking (enable_thinking). Disable it to reduce latency and token usage.
    required: false
    default: true
"""

ENABLE_THINKING_OFF = """  - name: enable_thinking
    label:
      zh_Hans: 思考模式
      en_US: Thinking Mode
    type: boolean
    help:
      zh_Hans: 是否开启思考模式（enable_thinking）。关闭后模型直接回答，可降低延迟与 Token 消耗。
      en_US: Whether to enable thinking (enable_thinking). Disable it to reduce latency and token usage.
    required: false
    default: false
"""

# Models exposing a `thinking.type` switch (enabled / disabled).
THINKING_SWITCH_MODELS = {
    "deepseek-v4-1-flash",
    "deepseek-v4-1-flash-1017",
    "deepseek-v4-flash-0731",
    "deepseek-v4-flash-0731-sg",
    "deepseek-v4-pro-0813",
    "deepseek-v4-pro-0813-sg",
    "deepseek-v4-pro-sg",
    "glm-5-2",
    "glm-5-2-sg",
    "kimi-k2-6",
    "lingdt-3-0-flash",
    "mimo-v2-5",
    "mimo-v2-5-pro",
    "mimo-v2-6-flash",
    "mimo-v2-6-pro",
    "zai-org-glm-4-6",
    "zai-org-glm-4-6v",
    "zai-org-glm-4-7",
}

# Doubao Seed models additionally support `thinking.type=auto`.
THINKING_DOUBAO_MODELS = {
    "doubao-seed-2-0-lite-260215",
    "doubao-seed-2-0-mini-260215",
    "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-1-lite-260915",
    "doubao-seed-2-1-pro-260628",
    "doubao-seed-2-1-turbo-260628",
    "doubao-seed-evolving",
}

# Qwen hybrid-thinking models: thinking is on by default.
QWEN_THINKING_ON_MODELS = {
    "qwen3-5-plus",
    "qwen3-6-35b-a3b",
    "qwen3-6-plus",
    "qwen3-7-max",
    "qwen3-7-plus",
    "qwen3-8-max",
}

# Qwen / ERNIE hybrid models: thinking is off by default.
QWEN_THINKING_OFF_MODELS = {
    "qwen-qwen3-max",
    "qwen3-max-preview",
    "qwen3-vl-flash",
    "baidu-ernie-4-5-turbo-128k",
    "baidu-ernie-4-5-turbo-vl-32k",
}


def thinking_rule_for(model_id: str) -> str:
    """Return the thinking-mode parameter rule block for a model."""
    key = model_id_to_filename(model_id)[:-len(".yaml")]
    if key in THINKING_SWITCH_MODELS:
        return THINKING_OBJECT
    if key in THINKING_DOUBAO_MODELS:
        return THINKING_DOUBAO
    if key == "minimax-m3":
        return THINKING_MINIMAX_M3
    if key in QWEN_THINKING_ON_MODELS:
        return ENABLE_THINKING_ON
    if key in QWEN_THINKING_OFF_MODELS:
        return ENABLE_THINKING_OFF
    return ""


def main():
    models = fetch_models()
    print(f"Fetched {len(models)} models from API")

    # Filter
    filtered = [m for m in models if not should_filter(m["id"])]
    print(f"After filtering: {len(filtered)} models")

    # Sort by model ID
    filtered.sort(key=lambda m: m["id"].lower())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Remove old generated YAMLs (keep llm.py)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".yaml"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    print("Cleaned old YAML files")

    # Generate YAML files
    generated = []
    for m in filtered:
        model_id = m["id"]
        filename = model_id_to_filename(model_id)
        label = model_id_to_label(model_id)

        if is_claude_1m(model_id):
            context_size = 1000000
            default_max_tokens = 32000
        elif is_gpt_5_4_plus(model_id):
            context_size = 1000000
            default_max_tokens = 16000
        elif is_gpt_5_2_plus(model_id):
            context_size = 400000
            default_max_tokens = 16000
        else:
            context_size = 204800
            default_max_tokens = 16000

        content = YAML_TEMPLATE.format(
            model_id=model_id, label=label,
            context_size=context_size, default_max_tokens=default_max_tokens,
            thinking_rule=thinking_rule_for(model_id),
        )
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content)
        generated.append(model_id)

    # Generate _position.yaml
    position_content = "\n".join(f"- {mid}" for mid in generated) + "\n"
    with open(os.path.join(OUTPUT_DIR, "_position.yaml"), "w") as f:
        f.write(position_content)

    print(f"Generated {len(generated)} model YAML files + _position.yaml")
    for mid in generated:
        print(f"  {mid}")


if __name__ == "__main__":
    main()
