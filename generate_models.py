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
pricing:
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
