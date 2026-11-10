"""Direct OpenAI-compatible chat and embedding requests."""
import httpx


class Client:
    def __init__(self, base, key, model="", embedding_model="bge-m3",
                 json_mode=True, template=True, proxy=None):
        self.base = base.rstrip("/")
        self.key = key.strip().strip(chr(34)).strip(chr(39))
        self.generation_model = model
        self.embedding_model = embedding_model
        self.json_mode = json_mode
        self.template = template
        self.http = httpx.Client(timeout=180, trust_env=False, proxy=proxy)

    def post(self, route, payload):
        response = self.http.post(self.base + "/" + route, json=payload,
                                  headers={"Authorization": "Bearer " + self.key})
        response.raise_for_status()
        return response.json()

    def chat(self, prompt, stage="", identity="", model=None, max_tokens=1200, json_mode=True):
        payload = {"model": model or self.generation_model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": max_tokens}
        if json_mode and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.template and "qwen" in payload["model"].lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        result = self.post("chat/completions", payload)
        return result["choices"][0]["message"]["content"]

    def embed(self, texts, stage="embed"):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = [str(text)[:4000] or " " for text in texts[start:start + 32]]
            result = self.post("embeddings", {"model": self.embedding_model, "input": batch})
            ordered = sorted(result["data"], key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in ordered)
        return vectors
