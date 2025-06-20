from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Carrega modelo local
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Carrega base com embeddings
with open("base_embed_semantica.json", encoding="utf-8") as f:
    base = json.load(f)

# Inicializa API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input esperado da API
class Pergunta(BaseModel):
    pergunta: str

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/perguntar")
def perguntar(p: Pergunta):
    embedding_pergunta = modelo.encode(p.pergunta)

    melhor_score = -1
    melhor_resposta = "Desculpe, não encontrei uma resposta relevante."

    for item in base:
        score = cosine_similarity(embedding_pergunta, item["embedding"])
        print(f"Score: {score:.4f} - Conteúdo: {item['conteudo'][:60]}...")
        if score > melhor_score:
            melhor_score = score
            melhor_resposta = item["conteudo"]

    print(f">>> Melhor score: {melhor_score:.4f}")
    return {"resposta": melhor_resposta}