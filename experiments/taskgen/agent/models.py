"""Model clients for mode A (docs/TASKGEN_DESIGN.html §4b): plain classes, no agent framework.
  FakeModel     replays fixed answers (tests without a GPU)
  LlamaServer   llama.cpp's llama-server, OpenAI-compatible /v1/chat/completions over httpx (serve scripts in
                ~/Documents/Code/Tau-harness/scripts/). Whether it constrains output to the JSON schema is unchecked:
                the schema is sent as response_format and the answer is validated on our side either way.
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
