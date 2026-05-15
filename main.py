import json
import os
from contextlib import asynccontextmanager
from typing import List, Union
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(
    title="Embedding API",
    version="1.0.0",
    description="API para generar embeddings de posts y queries usando SentenceTransformers",
)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        kwargs = {}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        _model = SentenceTransformer(MODEL_NAME, **kwargs)
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app.router.lifespan_context = lifespan


class PostEmbeddingRequest(BaseModel):
    id: int | None = Field(default=None, example=2)
    title: str = Field(..., example="Introducción a Clean Architecture")
    content: str = Field(
        ..., example="Explicación de capas, puertos y adaptadores en backend."
    )
    tags: Union[List[str], str, None] = Field(
        default_factory=list, example=["golang", "backend", "arquitectura"]
    )
    category: str = Field(default="", example="Software Architecture")


class QueryEmbeddingRequest(BaseModel):
    query: str = Field(..., example="arquitectura limpia en golang")


class EmbeddingResponse(BaseModel):
    dimensions: int
    embedding: List[float]


class PostEmbeddingResponse(EmbeddingResponse):
    id: int | None = None
    text_used: str


def normalize_tags(tags: Union[List[str], str, None]) -> List[str]:
    if tags is None:
        return []

    if isinstance(tags, list):
        return [str(t) for t in tags]

    if isinstance(tags, str):
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except Exception:
            pass
        return [tags]

    return [str(tags)]


def build_post_text(post: PostEmbeddingRequest) -> str:
    tags_text = ", ".join(normalize_tags(post.tags))
    return (
        f"title: {post.title} | "
        f"text: category: {post.category}. "
        f"tags: {tags_text}. "
        f"content: {post.content}"
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embeddings/post", response_model=PostEmbeddingResponse)
def create_post_embedding(payload: PostEmbeddingRequest):
    try:
        model = get_model()
        text = build_post_text(payload)
        vector = model.encode_document(text).tolist()

        return PostEmbeddingResponse(
            id=payload.id,
            text_used=text,
            dimensions=len(vector),
            embedding=vector,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"No se pudo generar el embedding del post: {e}"
        )


@app.post("/embeddings/query", response_model=EmbeddingResponse)
def create_query_embedding(payload: QueryEmbeddingRequest):
    try:
        model = get_model()
        vector = model.encode_query(payload.query).tolist()

        return EmbeddingResponse(dimensions=len(vector), embedding=vector)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"No se pudo generar el embedding del query: {e}"
        )
