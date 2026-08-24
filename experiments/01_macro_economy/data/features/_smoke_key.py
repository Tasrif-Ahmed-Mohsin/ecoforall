"""One-shot DeepSeek key check: exits 0 on success, prints server message either way."""
import os
import sys

from openai import OpenAI

key = os.environ.get("DEEPSEEK_API_KEY")
if not key:
    print("FAIL: $env:DEEPSEEK_API_KEY is empty in this shell", file=sys.stderr)
    sys.exit(2)

c = OpenAI(api_key=key, base_url="https://api.deepseek.com")
try:
    r = c.chat.completions.create(
        model="deepseek-v4",
        messages=[{"role": "user", "content": 'Respond with JSON {"forecast": 1.5}'}],
        temperature=0,
        max_tokens=50,
        response_format={"type": "json_object"},
        timeout=20,
    )
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

print("OK:", r.choices[0].message.content)
