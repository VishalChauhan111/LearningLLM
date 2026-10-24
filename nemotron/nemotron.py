
import os
from openai import OpenAI

MODEL = os.getenv(
    "NEMOTRON_MODEL",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
)

API_KEY = os.getenv("NEMOTRON_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "NEMOTRON_API_KEY is missing. Set it in CMD first."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)


def ask_nemotron(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048
    )

    return response.choices[0].message.content
