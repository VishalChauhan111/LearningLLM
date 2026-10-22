"""Minimal client for any OpenAI-compatible Nemotron endpoint.

Works with NVIDIA's hosted API (build.nvidia.com), a local vLLM/SGLang server,
or a NIM container. Configure with environment variables:

    export NEMOTRON_BASE_URL="https://integrate.api.nvidia.com/v1"   # hosted
    # export NEMOTRON_BASE_URL="http://localhost:8000/v1"            # local vLLM
    export NEMOTRON_API_KEY="nvapi-..."      # any string for a local server
    export NEMOTRON_MODEL="<model id>"       # copy from the model's page

Usage:
    python chat.py "Explain mixture-of-experts in two sentences."
"""
import os
import sys

from openai import OpenAI

base_url = os.environ.get("NEMOTRON_BASE_URL", "https://integrate.api.nvidia.com/v1")
api_key = os.environ.get("NEMOTRON_API_KEY", "")
model = os.environ.get("NEMOTRON_MODEL", "")

if not api_key or not model:
    sys.exit("Set NEMOTRON_API_KEY and NEMOTRON_MODEL (see the docstring at the top).")

prompt = " ".join(sys.argv[1:]) or "Say hello and tell me what you can do."
client = OpenAI(base_url=base_url, api_key=api_key)

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.6,
    max_tokens=1024,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
