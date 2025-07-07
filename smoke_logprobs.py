"""Quick smoke-test to check whether the current backend really returns
 token-level log-probabilities when requested.

Run with the same OPENAI_API_BASE / OPENAI_API_KEY environment variables
 that you use for the main codebase, for example:

    OPENAI_API_BASE=https://api.deepseek.com \
    OPENAI_API_KEY=... \
    python smoke_logprobs.py

If the backend supports the feature you should see a non-empty
`choice.logprobs` dict containing `tokens`, `token_logprobs`,
`text_offset`, etc. Otherwise it will print `None` / missing.
"""

import json
import os
import sys

import openai

# Honour user envs (DeepSeek uses different base URL)
openai.api_base = os.getenv("OPENAI_API_BASE", openai.api_base)
openai.api_key = os.getenv("OPENAI_API_KEY", openai.api_key)

model = os.getenv("SMOKE_MODEL", "deepseek-chat")

try:
    res = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": "Hello, world!"}],
        logprobs=True,
        top_logprobs=5,
        max_tokens=5,
        temperature=0.0,
    )
except Exception as e:
    print("API call failed:", e)
    sys.exit(1)

# Pretty-print full raw response
print("=== Raw response ===")
print(json.dumps(res.to_dict_recursive(), indent=2))
print()

# Inspect first choice
choice = res.choices[0]
print("=== First choice summary ===")
print("content:", choice.message.content)
print("has logprobs:", hasattr(choice, "logprobs"))
print("logprobs value:")
print(json.dumps(choice.logprobs, indent=2))
