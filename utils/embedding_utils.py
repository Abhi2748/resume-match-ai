import numpy as np
import torch
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_mean_embedding(text, embedder=embedder):
    if not text.strip():
        return torch.zeros(1, 384)
    embeddings = embedder.encode(text, convert_to_tensor=True).reshape(1, -1)
    return embeddings

def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item()