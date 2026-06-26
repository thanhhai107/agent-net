"""Optional Qdrant semantic index for procedural memories."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Protocol

from openai import OpenAI

from agent.memory.models import StoredMemory

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, model: str) -> None:
        kwargs = {}
        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL")
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class OllamaEmbeddingProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434").rstrip(
            "/"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["embeddings"]


def load_embedding_provider() -> EmbeddingProvider | None:
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        return None
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return OpenAIEmbeddingProvider(model)
    if provider == "ollama":
        return OllamaEmbeddingProvider(model)
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider!r}")


class QdrantMemoryIndex:
    """Rebuildable vector index. Failure never invalidates canonical memory."""

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        try:
            self.provider = provider or load_embedding_provider()
        except Exception as exc:
            logger.warning("Embedding provider disabled: %s", exc)
            self.provider = None
        self.url = os.getenv("QDRANT_URL", "").strip()
        self.api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        self.collection = os.getenv("QDRANT_COLLECTION", "").strip()
        try:
            self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "0") or 0)
        except ValueError:
            logger.warning("Embedding provider disabled: invalid EMBEDDING_DIMENSION")
            self.dimension = 0
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(
            self.provider and self.url and self.collection and self.dimension > 0
        )

    def _client_or_none(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return None
        self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def _ensure_collection(self, vector_size: int) -> bool:
        client = self._client_or_none()
        if client is None:
            return False
        from qdrant_client.models import Distance, VectorParams

        if vector_size != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"configured {self.dimension}, provider returned {vector_size}"
            )
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
        return True

    def upsert(self, memory: StoredMemory) -> None:
        if not self.enabled:
            return
        vectors = self.provider.embed([memory.embedding_text()])
        if not vectors or not self._ensure_collection(len(vectors[0])):
            return
        from qdrant_client.models import PointStruct

        self._client_or_none().upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=memory.memory_id,
                    vector=vectors[0],
                    payload={
                        "bank_id": memory.bank_id,
                        "memory_type": memory.memory_type.value,
                        "status": memory.status.value,
                        "scenario": memory.attributes.scenarios,
                        "topology_class": memory.attributes.topology_classes,
                        "protocols": memory.attributes.protocols,
                        "task_stages": memory.attributes.task_stages,
                        "tools": memory.attributes.tools,
                        "confidence": memory.confidence,
                    },
                )
            ],
        )

    def search(
        self,
        *,
        bank_id: str,
        query: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        vectors = self.provider.embed([query])
        if not vectors or not self._ensure_collection(len(vectors[0])):
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = Filter(
            must=[
                FieldCondition(key="bank_id", match=MatchValue(value=bank_id)),
                FieldCondition(
                    key="status",
                    match=MatchValue(value="validated"),
                ),
            ]
        )
        client = self._client_or_none()
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.collection,
                query=vectors[0],
                query_filter=query_filter,
                limit=limit,
                with_payload=False,
            )
            points = response.points
        else:
            points = client.search(
                collection_name=self.collection,
                query_vector=vectors[0],
                query_filter=query_filter,
                limit=limit,
                with_payload=False,
            )
        return [(str(point.id), float(point.score)) for point in points]

    def delete_bank(self, bank_id: str) -> None:
        client = self._client_or_none()
        if client is None or not client.collection_exists(self.collection):
            return
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="bank_id",
                            match=MatchValue(value=bank_id),
                        )
                    ]
                )
            ),
        )
