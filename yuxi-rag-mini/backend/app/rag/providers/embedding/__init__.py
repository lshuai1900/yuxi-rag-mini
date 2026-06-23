from app.rag.providers.embedding.base import BaseEmbeddingProvider
from app.rag.providers.embedding.openai_compatible import OpenAICompatibleEmbedding
from app.rag.providers.embedding.ollama import OllamaEmbedding
from app.rag.providers.embedding.huggingface import HuggingFaceEmbedding


def create_embedding_provider(
    provider: str = "openai_compatible",
    model: str = "bge-m3",
    base_url: str = "",
    api_key: str = "",
    dimension: int = 1024,
    batch_size: int = 40,
    **kwargs,
) -> BaseEmbeddingProvider:
    if provider == "openai_compatible":
        return OpenAICompatibleEmbedding(model=model, base_url=base_url, api_key=api_key,
                                          dimension=dimension, batch_size=batch_size, **kwargs)
    elif provider == "ollama":
        return OllamaEmbedding(model=model, base_url=base_url, dimension=dimension,
                               batch_size=batch_size, **kwargs)
    elif provider == "huggingface":
        return HuggingFaceEmbedding(model=model, dimension=dimension, batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
