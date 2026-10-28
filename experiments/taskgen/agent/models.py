"""Model clients for mode A (docs/TASKGEN_DESIGN.html §4b): plain classes, no agent framework.
  FakeModel     replays fixed answers (tests without a GPU)
  LlamaServer   llama.cpp's llama-server, OpenAI-compatible /v1/chat/completions over httpx (serve scripts in
                ~/Documents/Code/Tau-harness/scripts/). Whether it constrains output to the JSON schema is unchecked:
                the schema is sent as response_format and the answer is validated on our side either way.
  Nim           NVIDIA's hosted NIM (integrate.api.nvidia.com, OpenAI-compatible); key NVIDIA_NIM_API_KEY read from
                ~/Documents/Code/robotics_agentic/.env (never logged). No forced schema: the answer is validated on our
                side (the forced schema on llama-server let Qwen3.5-9B skip 'colour' three times, 10-28 smoke test).
Exploratory, UNVERIFIED.
"""
import json


class FakeModel:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def complete(self, messages, schema=None):
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else ""


class LlamaServer:
    def __init__(self, url="http://127.0.0.1:8099", model="qwen3.5-9b", temperature=0.2, timeout=300):
        self.url, self.model, self.temperature, self.timeout = url.rstrip("/"), model, temperature, timeout

    def complete(self, messages, schema=None):
        import httpx
        body = dict(model=self.model, messages=messages, temperature=self.temperature,
                    chat_template_kwargs={"enable_thinking": False})  # reasoning off: the Tau project's leak fix
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "scene_request", "schema": json.loads(schema)}}
        r = httpx.post(f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _nim_key(env="/home/imerit/Documents/Code/robotics_agentic/.env"):
    import os
    if os.environ.get("NVIDIA_NIM_API_KEY"):
        return os.environ["NVIDIA_NIM_API_KEY"]
    for line in open(env):
        if line.startswith("NVIDIA_NIM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("NVIDIA_NIM_API_KEY not found")


class Nim:
    def __init__(self, model, url="https://integrate.api.nvidia.com/v1", temperature=0.2, max_tokens=4096, timeout=180, retries=3):
        self.model, self.url, self.temperature, self.max_tokens, self.timeout, self.retries = model, url, temperature, max_tokens, timeout, retries
        self._key = _nim_key()

    def complete(self, messages, schema=None):
        import httpx, time
        body = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        for i in range(self.retries):
            r = httpx.post(f"{self.url}/chat/completions", json=body, timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:  # rate limit / server error: back off and retry
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or ""
        r.raise_for_status()
        return ""
