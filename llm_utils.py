"""
llm_utils.py
Central place for OpenAI client creation, embeddings, and generic
chat-completion helpers used across the app.
"""

from openai import OpenAI


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def chat(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.4) -> str:
    """Simple single-turn chat helper. Returns text or an error string."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"


def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")[:8000]
        resp = client.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Embedding error: {e}")


def get_embeddings_batch(client: OpenAI, texts: list, model: str = "text-embedding-3-small"):
    try:
        cleaned = [t.replace("\n", " ")[:8000] for t in texts]
        resp = client.embeddings.create(model=model, input=cleaned)
        return [d.embedding for d in resp.data]
    except Exception as e:
        raise RuntimeError(f"Embedding batch error: {e}")
